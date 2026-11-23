"""Convert DAgger rollouts (OpenViTac `policy/openpi_dagger`, one hdf5 per episode) into an openpi training dataset with
cached AnyTouch2 features and per-frame tactile event (e.g. "jam") labels.

Which frames become training data (each kept segment is written as its own LeRobot episode, so action chunks never
cross a gap):
  * expert segments (source == 1): the scripted recovery from the failure state the policy reached -> corrective data;
  * whole successful episodes without intervention (the policy's own successes), unless --no-policy-success.
Policy frames of episodes that needed an intervention (and of failed episodes) are dropped: their actions are the
behaviour being corrected. Expert segments that do not end at the plugin's goal (`goal`, e.g. plug inserted; failed
recoveries) are dropped too, unless --no-expert-success-only.

AnyTouch2 features are computed over the full recorded episode (history = 4 frames, stride 2 recorded frames,
background = the episode's first tactile frame, exactly what the OpenPI client sends online) and then subset to the
kept frames, so a segment's first frames still see the tactile history that preceded them.

Event label (tactile/event_<--event-name>.npy, float32 [N]) from the plugin's per-frame failure-mode rule
(`event_rule`; e.g. insert_USB "plug at the opening, misaligned or tilted"):
   1  the failure state the policy reached: the first frames of an expert segment while the rule holds (until the
      expert's first move clears it)
   0  frames where the rule does not hold
  -1  other frames where the rule holds (e.g. transient misalignment while a recovery is still sliding in); unknown,
      masked out of the tactile event loss
Rollouts of the first insert_USB collection have no `event_rule`/`goal`; for them the rule is computed from the USB
diagnostics with the --jam-* options.

Usage:
  uv run examples/openvitac/convert_dagger.py --src /path/to/eval_result/.../dagger --repo-id openvitac/insert_USB_dagger_gelsight
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil

import h5py
import numpy as np
import tqdm
import tyro

from openpi.tactile import anytouch2 as at2

IMAGE_SIZE = 224
KEYS = {
    "image": "head",
    "wrist_image": "wrist",
    "left_tactile_image": "left_tactile",
    "right_tactile_image": "right_tactile",
}


@dataclasses.dataclass
class Args:
    src: list[pathlib.Path]  # one or more directories with <seed>_<result>.hdf5 files
    repo_id: str = "openvitac/insert_USB_dagger_gelsight"
    event_name: str = "jam"  # -> tactile/event_<name>.npy (training config `event_name`)
    overwrite: bool = False
    policy_success: bool = True  # also keep successful episodes that needed no intervention
    min_segment: int = 10
    jam_z: float = 0.0124 + 0.0015  # m above the inserted pose (plug length + 1.5 mm)
    jam_xy: float = 0.0012  # m
    jam_tilt: float = 8.0  # deg
    jam_zone_xy: float = 0.008  # m: only frames near the target slot axis can be jams (not the start slot)
    jam_min_z: float = 0.003  # m: below this the plug is (nearly) inserted
    expert_success_only: bool = True  # keep only expert segments that end with the plug inserted
    inserted_xy: float = 0.002  # m, "inserted" = rel z < jam_min_z and xy error < inserted_xy (task success uses 2 mm)
    fps: int = 60
    anytouch2_checkpoint: str = "checkpoint-4frames.pth"
    sensor: str = "gelsight_mini"
    num_frames: int = 4
    stride: int = 2
    encode_batch: int = 64
    device: str = "cuda"
    dtype: str = "float16"


def event_rule_and_goal(f: h5py.File, args: Args) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame failure-mode rule and recovery sub-goal. Recorded by the task plugin (`event_rule`, `goal`); rollouts
    from the first insert_USB collection (before plugins) only have the USB diagnostics, so compute them here."""
    if "event_rule" in f and "goal" in f:
        return f["event_rule"][()].astype(bool), f["goal"][()].astype(bool)
    rel, xy, tilt = f["rel_xyz"][()], f["xy_err"][()], f["tilt_deg"][()]
    rule = (
        (xy < args.jam_zone_xy)
        & (rel[:, 2] > args.jam_min_z)
        & (rel[:, 2] < args.jam_z)
        & ((xy > args.jam_xy) | (tilt > args.jam_tilt))
    )
    goal = (rel[:, 2] < args.jam_min_z) & (xy < args.inserted_xy)
    return rule, goal


def segments(
    source: np.ndarray, goal: np.ndarray, *, success: bool, n_interventions: int, args: Args
) -> tuple[list[tuple[int, int, str]], int]:
    """Kept (start, end, kind) segments and the number of expert segments dropped as failed recoveries."""
    segs, dropped = [], 0
    if n_interventions == 0:
        if success and args.policy_success:
            segs.append((0, len(source), "policy_success"))
        return segs, dropped
    i = 0
    while i < len(source):
        if source[i] == 1:
            j = i
            while j < len(source) and source[j] == 1:
                j += 1
            if j - i >= args.min_segment:
                if goal[j - 1] or not args.expert_success_only:
                    segs.append((i, j, "expert"))
                else:
                    dropped += 1
            i = j
        else:
            i += 1
    return segs, dropped


def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    files = sorted(p for d in args.src for p in d.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no hdf5 files in {args.src}")
    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)

    plan, n_dropped = [], 0
    for p in files:
        with h5py.File(p, "r") as f:
            segs, dropped = segments(
                f["source"][()],
                event_rule_and_goal(f, args)[1],
                success=bool(f.attrs["success"]),
                n_interventions=int(f.attrs["n_interventions"]),
                args=args,
            )
            plan.append((p, segs))
            n_dropped += dropped
    total = sum(e - s for _, segs in plan for s, e, _ in segs)
    print(
        f"{len(files)} rollouts -> {sum(len(s) for _, s in plan)} segments, {total} frames "
        f"({n_dropped} failed expert recoveries dropped)"
    )

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="Franka_GelSight_mini",
        fps=args.fps,
        features={
            **{
                k: {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
                for k in KEYS
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_threads=8,
        image_writer_processes=4,
    )
    encoder = at2.AnyTouch2Encoder(
        at2.AnyTouch2Config(
            checkpoint=args.anytouch2_checkpoint,
            num_frames=args.num_frames,
            stride=args.stride,
            sensor=args.sensor,
            device=args.device,
            dtype=args.dtype,
        )
    )
    (out / "tactile").mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(
        out / "tactile" / f"anytouch2_{pathlib.Path(args.anytouch2_checkpoint).stem}.npy",
        mode="w+",
        dtype=np.float16,
        shape=(total, 2, at2.N_COMPACT, at2.FEATURE_DIM),
    )
    jam = np.zeros(total, np.float32)
    info = {k: np.zeros(total, np.int32) for k in ("seed", "frame", "source")}
    segment_log = []

    gidx = 0
    for p, segs in tqdm.tqdm(plan, desc="rollouts"):
        if not segs:
            continue
        with h5py.File(p, "r") as f:
            data = {k: f[v][()] for k, v in KEYS.items()}
            state, action, source = f["state"][()], f["action"][()], f["source"][()]
            rule, _ = event_rule_and_goal(f, args)
            seed, instruction = int(f.attrs["seed"]), str(f.attrs["instruction"])
        n = len(state)
        hist = np.array([at2.history_indices(t, args.num_frames, args.stride) for t in range(n)])
        ep_feats = np.zeros((n, 2, at2.N_COMPACT, at2.FEATURE_DIM), np.float16)
        for s, key in enumerate(("left_tactile_image", "right_tactile_image")):
            tac = data[key]
            for b0 in range(0, n, args.encode_batch):
                idx = hist[b0 : b0 + args.encode_batch]
                ep_feats[b0 : b0 + len(idx), s] = encoder.encode(tac[idx], np.repeat(tac[:1], len(idx), axis=0))
        ep_jam = np.where(rule, -1.0, 0.0).astype(np.float32)
        for s0, _, kind in segs:
            t = s0
            while kind == "expert" and t < n and rule[t]:  # the jammed state handed to the expert
                ep_jam[t] = 1.0
                t += 1

        for s0, s1, kind in segs:
            for t in range(s0, s1):
                dataset.add_frame(
                    {**{k: data[k][t] for k in KEYS}, "state": state[t], "actions": action[t], "task": instruction}
                )
            dataset.save_episode()
            m = s1 - s0
            feats[gidx : gidx + m] = ep_feats[s0:s1]
            jam[gidx : gidx + m] = ep_jam[s0:s1]
            info["seed"][gidx : gidx + m] = seed
            info["frame"][gidx : gidx + m] = np.arange(s0, s1)
            info["source"][gidx : gidx + m] = source[s0:s1]
            segment_log.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "start": s0,
                    "end": s1,
                    "jam_frames": int((ep_jam[s0:s1] == 1).sum()),
                    "unknown_frames": int((ep_jam[s0:s1] < 0).sum()),
                }
            )
            gidx += m

    assert gidx == total, (gidx, total)
    feats.flush()
    np.save(out / "tactile" / f"event_{args.event_name}.npy", jam)
    np.savez(out / "tactile" / "frame_info.npz", **info)
    meta = {
        "encoder": "AnyTouch2",
        "checkpoint": args.anytouch2_checkpoint,
        "sensor": args.sensor,
        "num_frames": args.num_frames,
        "stride": args.stride,
        "background": "first tactile frame of the rollout",
        "device": args.device,
        "dtype": args.dtype,
        "frames": int(total),
        "jam_rule": {
            "zone_xy_below": args.jam_zone_xy,
            "z_above": args.jam_min_z,
            "z_below": args.jam_z,
            "xy_above": args.jam_xy,
            "tilt_above_deg": args.jam_tilt,
        },
        "jam_frames": int((jam == 1).sum()),
        "unknown_frames": int((jam < 0).sum()),
        "dropped_failed_expert_segments": n_dropped,
        "segments": segment_log,
    }
    (out / "tactile" / "anytouch2_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"wrote {total} frames ({int((jam == 1).sum())} jam, {int((jam < 0).sum())} unknown) in {len(segment_log)} segments to {out}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
