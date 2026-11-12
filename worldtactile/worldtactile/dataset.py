"""UniVTAC contact-rich training samples (multi-task, head + wrist cameras).

Sample at frame t, for every t inside the contact-rich range [start, end] of its episode (scripts/preprocess_univtac.py):
  images          {camera: PIL RGB image at t, full resolution} (the collator letterboxes to 224)
  instruction     the task's first "seen" instruction
  state           joint[t], quantile-normalized                                          (9,)
  actions         joint[t+1 .. t+H], repeated last frame past the episode end; delta arm joints (normalization.py)
  action_mask     True for real (not padded) steps                                       (H,)
  tactile         AnyTouch2 CLS of the clip (t-12, t-8, t-4, t) per finger, standardized  (2, 512)
  tactile_target  the same for the future clip ending at t + target_offset                (2, 512)
                  (default 24: clip t+12, t+16, t+20, t+24)
  target_mask     False when t + target_offset is past the episode end
  tactile_corrupt (with tactile_corrupt=True) tactile tokens for the adherence pass: the resting sensor ("mask_out")
                  or a contact-rich frame of another task ("other_task"), chosen at random
CLS standardization uses per-dimension mean/std over all frames of the training tasks.
"""
import json

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import paths
from .normalization import normalize_actions, normalize_state

TARGET_OFFSET = 24
TACTILE_CORRUPT_MODES = ("mask_out", "other_task")
QUANTILE_RANGE_FLOOR = 4e-4  # rad, minimum q99 - q01 of a delta dimension (keeps near-constant joints finite)


def task_instruction(task, root=None):
    return json.loads(((root or paths.UNIVTAC_INSTRUCTIONS) / f"{task}.json").read_text())["seen"][0]


def global_cls_stats(tasks, root):
    """Per-dimension mean/std of the AnyTouch2 CLS over all frames of the given tasks (cached in `root`)."""
    path = root / f"cls_stats_{'+'.join(sorted(tasks))}.npz"
    if not path.exists():
        feats = []
        for task in tasks:
            meta = json.loads((root / task / "index.json").read_text())
            feats += [np.load(root / task / f"{e['episode']}.npz")["cls"].astype(np.float32).reshape(-1, 512)
                      for e in meta["episodes"]]
        feats = np.concatenate(feats)
        np.savez(path, mean=feats.mean(0), std=feats.std(0) + 1e-6)
    s = np.load(path)
    return s["mean"], s["std"]


class UniVTACContactDataset(Dataset):
    def __init__(self, tasks, horizon=50, cameras=("head", "wrist"), stats=None, tactile_corrupt=False,
                 target_offset=TARGET_OFFSET, root=None):
        tasks = [tasks] if isinstance(tasks, str) else list(tasks)
        self.tasks, self.cameras, self.horizon = tasks, list(cameras), horizon
        self.tactile_corrupt, self.target_offset = tactile_corrupt, target_offset
        self.root = root or paths.PROCESSED
        self.instructions = {t: task_instruction(t) for t in tasks}
        self.cls_mean, self.cls_std = global_cls_stats(tasks, self.root)
        self.episodes = []  # dicts: task, episode, raw, T, onset, last, start, end
        for task in tasks:
            meta = json.loads((self.root / task / "index.json").read_text())
            self.episodes += [dict(e, task=task) for e in meta["episodes"]]
        self.samples = [(i, t) for i, e in enumerate(self.episodes) for t in range(e["start"], e["end"] + 1)]
        self.by_task = {t: [i for i, e in enumerate(self.episodes) if e["task"] == t] for t in tasks}
        self._cls, self._joints, self._h5 = {}, {}, {}
        self.stats = stats or self.compute_stats()
        no_contact = np.load(self.root / "no_contact_cls.npy").astype(np.float32)
        self.no_contact = np.stack([(no_contact - self.cls_mean) / self.cls_std] * 2)  # both fingers

    def compute_stats(self):
        """q01/q99 of the joints over all frames, and per task the q01/q99 of the arm deltas per chunk index."""
        js = np.concatenate([self._load_joints(i) for i in range(len(self.episodes))])
        q01, q99 = np.quantile(js, 0.01, axis=0), np.quantile(js, 0.99, axis=0)
        stats = dict(q01=q01.astype(np.float32), q99=np.maximum(q99, q01 + 1e-6).astype(np.float32))
        k = np.arange(1, self.horizon + 1)
        for task in self.tasks:
            deltas = []
            for i in self.by_task[task]:
                j, e = self._load_joints(i), self.episodes[i]
                t = np.arange(e["start"], e["end"] + 1)
                deltas.append(j[np.minimum(t[:, None] + k[None], e["T"] - 1)] - j[t][:, None])
            deltas = np.concatenate(deltas)  # (N, H, 9)
            lo, hi = np.quantile(deltas, 0.01, axis=0), np.quantile(deltas, 0.99, axis=0)
            mid, half = (lo + hi) / 2, np.maximum(hi - lo, QUANTILE_RANGE_FLOOR) / 2
            stats[f"delta_q01__{task}"] = (mid - half).astype(np.float32)
            stats[f"delta_q99__{task}"] = (mid + half).astype(np.float32)
        return stats

    def _load_joints(self, i):
        if i not in self._joints:
            with h5py.File(self.episodes[i]["raw"], "r") as f:
                self._joints[i] = f["embodiment/joint"][:].astype(np.float32)
        return self._joints[i]

    def _load_cls(self, i):
        if i not in self._cls:
            e = self.episodes[i]
            c = np.load(self.root / e["task"] / f"{e['episode']}.npz")["cls"].astype(np.float32)
            self._cls[i] = (c - self.cls_mean) / self.cls_std
        return self._cls[i]

    def _image(self, i, cam, t):
        if i not in self._h5:  # one open handle per episode and worker process
            self._h5[i] = h5py.File(self.episodes[i]["raw"], "r")
        buf = self._h5[i][f"observation/{cam}/rgb"][t]
        return Image.fromarray(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR))

    def _tactile_corruption(self, i, rng):
        others = [x for x in self.tasks if x != self.episodes[i]["task"]]
        modes = [m for m in TACTILE_CORRUPT_MODES if m != "other_task" or others]
        mode = modes[int(rng.integers(len(modes)))]
        if mode == "mask_out":
            return self.no_contact, TACTILE_CORRUPT_MODES.index(mode)
        j = int(rng.choice(self.by_task[others[int(rng.integers(len(others)))]]))
        e = self.episodes[j]
        return self._load_cls(j)[int(rng.integers(e["start"], e["end"] + 1))], TACTILE_CORRUPT_MODES.index(mode)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        i, t = self.samples[idx]
        e = self.episodes[i]
        T = e["T"]
        joints, cls = self._load_joints(i), self._load_cls(i)
        steps = np.arange(t + 1, t + 1 + self.horizon)
        actions = joints[np.minimum(steps, T - 1)]
        tgt_t = t + self.target_offset
        item = dict(
            images={c: self._image(i, c, t) for c in self.cameras},
            instruction=self.instructions[e["task"]],
            state=torch.from_numpy(normalize_state(joints[t], self.stats)),
            actions=torch.from_numpy(normalize_actions(actions, joints[t], e["task"], self.stats)),
            action_mask=torch.from_numpy(steps <= T - 1),
            tactile=torch.from_numpy(cls[t]),
            tactile_target=torch.from_numpy(cls[min(tgt_t, T - 1)]),
            target_mask=torch.tensor(tgt_t <= T - 1),
        )
        if self.tactile_corrupt:
            tac, mode = self._tactile_corruption(i, np.random.default_rng())
            item.update(tactile_corrupt=torch.from_numpy(np.ascontiguousarray(tac, dtype=np.float32)),
                        tactile_corrupt_mode=torch.tensor(mode))
        return item
