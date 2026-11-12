"""WorldTactile pi0.5 policy for closed-loop evaluation.

Per request: images JPEG round-tripped like the training data and letterboxed to 224 (head -> base slot, wrist -> left
wrist slot); AnyTouch2 CLS of the two tactile clips, standardized with the run's cls_stats; joint state normalized into
the prompt. The static K,V (images, prompt, tactile) are computed once and 10 flow steps rerun only the queries and the
actions. Optional real-time chunking (RTC) toward the previous chunk, overlap blending and Savitzky-Golay smoothing.
Delta actions are converted to absolute joint targets.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import paths
from .. import univtac as uv
from ..anytouch2 import AnyTouch2Encoder
from ..normalization import ARM, denormalize_actions, load_norm_stats, normalize_actions, normalize_state
from ..pi05.collate import Pi05Collator, to_device
from ..pi05.model import ATTN_GROUPS, WorldTactilePi05
from .protocol import jpeg_roundtrip
from .query_log import QueryLogger


def load_policy(run_dir, ckpt_tag="stage2_final", device="cuda", base=None):
    """WorldTactilePi05 with a run's trainable weights (trainable_<tag>.pt) and the run's training arguments."""
    run = Path(run_dir)
    args = json.loads((run / "args.json").read_text())
    model = WorldTactilePi05(str(base or paths.PI05_BASE), horizon=args["horizon"], gradient_checkpointing=False)
    model.add_lora(args["lora_rank"])
    missing, unexpected = model.load_state_dict(torch.load(run / f"trainable_{ckpt_tag}.pt", map_location="cpu"),
                                                strict=False)
    assert not unexpected, unexpected[:5]
    return model.to(device).eval(), args


class Policy:
    def __init__(self, run_dir, ckpt_tag="stage2_final", device="cuda", rtc=True, rtc_beta=20.0, rtc_blend=True,
                 execute_steps=25, smooth_window=0, query_log=None):
        run = Path(run_dir)
        self.device = torch.device(device)
        self.model, self.args = load_policy(run, ckpt_tag, device)
        self.cameras = self.args["cameras"].split(",")
        self.target_offset = int(self.args.get("target_offset", 24))
        self.stats = load_norm_stats(run / "norm_stats.npz")
        cs = np.load(run / "cls_stats.npz")
        self.cls_mean, self.cls_std = cs["mean"], cs["std"]
        self.encoder = AnyTouch2Encoder(uv.load_background(), device=device)
        self.col = Pi05Collator(self.cameras)
        self.rtc, self.rtc_beta, self.rtc_blend = rtc, rtc_beta, rtc_blend
        self.execute_steps, self.smooth_window = execute_steps, smooth_window  # must match the adapter
        self.logger = QueryLogger(query_log, self.encode_tactile, self.target_offset) if query_log else None
        self.prev_abs = None  # previous chunk of this episode, absolute joints (H, 9)

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def encode_tactile(self, clips):
        """uint8 clips (2, 4, H, W, 3), simulator RGB -> standardized CLS (2, 512)."""
        clips = np.stack([np.stack([jpeg_roundtrip(f) for f in c]) for c in clips])
        return (self.encoder.cls(clips) - self.cls_mean) / self.cls_std

    @torch.no_grad()
    def act(self, req):
        task, joint = str(req["task"]), req["joint"].astype(np.float32)
        H = self.model.horizon
        tactile = self.encode_tactile(np.stack([req["tac_left"], req["tac_right"]]))
        item = dict(images={c: Image.fromarray(jpeg_roundtrip(req[c])) for c in self.cameras},
                    instruction=str(req["instruction"]),
                    state=torch.from_numpy(normalize_state(joint, self.stats)),
                    actions=torch.zeros(H, 9), action_mask=torch.ones(H, dtype=torch.bool),
                    tactile=torch.from_numpy(tactile.astype(np.float32)), tactile_target=torch.zeros(2, 512),
                    target_mask=torch.tensor(False))
        batch = to_device(self.col([item]), self.device)
        if int(req.get("episode_start", 0)):
            self.prev_abs = None
        s = self.execute_steps
        prev = weights = None
        if self.rtc and self.prev_abs is not None:
            # the previous chunk's steps s..H-1 are this chunk's 0..H-s-1, re-expressed relative to the current joints
            shifted = np.concatenate([self.prev_abs[s:], np.repeat(self.prev_abs[-1:], s, 0)])
            prev = torch.zeros(1, H, 9, device=self.device)
            prev[0, : H - s] = torch.from_numpy(normalize_actions(shifted, joint, task, self.stats)[: H - s])
            weights = self.model.rtc_weights(H, H - s, delay=0)
        if self.logger is not None:
            self.model.attn_probe = {}  # attention mass by key group, logged with the prediction
        with self._autocast():
            cache = self.model.encode_static(batch)
            x, pred = self.model.sample_actions(cache, prev=prev, weights=weights, beta=self.rtc_beta)
        extra_rec = extra_arrays = None
        if self.model.attn_probe is not None:
            steps = torch.stack(self.model.attn_probe["steps"], 0)[:, 0].numpy()  # (flow steps, layers, 2 rows, G)
            share = self.model.attn_probe["share"][0].numpy()
            mean = steps.mean((0, 1))
            extra_rec = {f"attn_{who}": {g: round(float(mean[r, i]), 5) for i, g in enumerate(ATTN_GROUPS)}
                         for r, who in enumerate(("query", "action"))}
            extra_rec["attn_key_share"] = {g: round(float(share[i]), 5) for i, g in enumerate(ATTN_GROUPS)}
            extra_arrays = dict(attn_mass=steps.astype(np.float32), attn_share=share.astype(np.float32))
            self.model.attn_probe = None
        if prev is not None and self.rtc_blend:
            w = weights.to(x.device)[None, :, None]
            x = w * prev + (1 - w) * x
        a = denormalize_actions(x[0].float().cpu().numpy(), joint, task, self.stats)
        if self.smooth_window > 0:
            from scipy.signal import savgol_filter
            a[:, :ARM] = savgol_filter(a[:, :ARM], self.smooth_window, 3, axis=0, mode="interp")
        self.prev_abs = a.copy()
        if self.logger is not None:
            self.logger.log(req, tactile, pred[0].float().cpu().numpy(), extra_rec, extra_arrays)
        return a
