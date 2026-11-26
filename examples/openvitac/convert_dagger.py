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
With --event-mode dense the label is the rule itself wherever the plugin marks it defined (`event_valid`), -1 elsewhere
(e.g. empty_can_selection: is the lifted can the empty one).
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
    # "onset": 1 = failure state handed to the expert, -1 = other frames where the rule holds, 0 = elsewhere (jam-like
    # events). "dense": label = the plugin's rule wherever `event_valid` (e.g. empty_can_selection: 1 = the lifted can
    # is the empty one, 0 = the heavy one, -1 = no can lifted).
    event_mode: str = "onset"
    overwrite: bool = False
    policy_success: bool = True  # also keep successful episodes that needed no intervention
    min_segment: int = 10
    jam_z: float = 0.0124 + 0.0015  # m above the inserted pose (plug length + 1.5 mm)
    jam_xy: float = 0.0012  # m
    jam_tilt: float = 8.0  # deg
    jam_zone_xy: float = 0.008  # m: only frames near the target slot axis can be jams (not the start slot)
    jam_min_z: float = 0.003  # m: below this the plug is (nearly) inserted
    expert_success_only: bool = True  # keep only expert segments that end with the plug inserted
    # also accept the last expert segment of a successful episode (the goal may only hold after the segment ends, e.g.
    # empty_can_selection: the released can is still falling into the basket at the last expert frame)
    goal_from_episode_success: bool = False
    inserted_xy: float = 0.002  # m, "inserted" = rel z < jam_min_z and xy error < inserted_xy (task success uses 2 mm)
    # Stationary runs in expert segments (the scripted recovery's settle delays: the arm and gripper targets do not
    # change) are cut down to their first `still_keep` frames when longer than that. 0 = keep everything.
    # gear_assembly: insert -> settle 30 steps / wiggle settles made 23% of the frames idle at the inserted pose and the
    # fine-tuned policy learned to wait there instead of turning the gear.
    still_keep: int = 0
    still_joint_eps: float = 2e-4  # rad, max |action[t+1] - action[t]| over the 7 arm joints
    still_gripper_eps: float = 1e-5  # m, gripper target change
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
    last_expert = int(np.flatnonzero(source == 1)[-1]) if np.any(source == 1) else -1
    i = 0
    while i < len(source):
        if source[i] == 1:
            j = i
            while j < len(source) and source[j] == 1:
                j += 1
            if j - i >= args.min_segment:
                ok = goal[j - 1] or (args.goal_from_episode_success and success and j - 1 == last_expert)
                if ok or not args.expert_success_only:
                    segs.append((i, j, "expert"))
                else:
                    dropped += 1
            i = j
        else:
            i += 1
    return segs, dropped


def still_frames_kept(action: np.ndarray, s0: int, s1: int, args: Args) -> np.ndarray:
    """Frame indices of [s0, s1) with stationary runs longer than args.still_keep cut to their first still_keep frames.
    Frame t is still when its action equals the previous frame's (within the eps), i.e. the run starts at the first
    repeated target."""
    idx = np.arange(s0, s1)
    if args.still_keep <= 0 or s1 - s0 < 2:
        return idx
    d = np.abs(np.diff(action[s0:s1], axis=0))
    still = np.concatenate([[False], (d[:, :7].max(1) < args.still_joint_eps) & (d[:, 7] < args.still_gripper_eps)])
    keep = np.ones(s1 - s0, bool)
    run = 0
    for t in range(s1 - s0):
        run = run + 1 if still[t] else 0
        if run > args.still_keep:
            keep[t] = False
    keep[-1] = True  # the segment's last frame (goal check) always stays
    return idx[keep]


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

    plan, n_dropped, n_still_cut = [], 0, 0
    for p in files:
        with h5py.File(p, "r") as f:
            segs, dropped = segments(
                f["source"][()],
                event_rule_and_goal(f, args)[1],
                success=bool(f.attrs["success"]),
                n_interventions=int(f.attrs["n_interventions"]),
                args=args,
            )
            action = f["action"][()]
        # (start, end, kind, kept frame indices): settle-delay runs are cut from expert segments only
        segs = [
            (s, e, k, still_frames_kept(action, s, e, args) if k == "expert" else np.arange(s, e)) for s, e, k in segs
        ]
        n_still_cut += sum(e - s - len(fr) for s, e, _, fr in segs)
        plan.append((p, segs))
        n_dropped += dropped
    total = sum(len(fr) for _, segs in plan for *_, fr in segs)
    print(
        f"{len(files)} rollouts -> {sum(len(s) for _, s in plan)} segments, {total} frames "
        f"({n_dropped} failed expert recoveries dropped, {n_still_cut} stationary expert frames cut)"
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
            valid = f["event_valid"][()].astype(bool) if "event_valid" in f else np.ones(len(rule), bool)
            seed, instruction = int(f.attrs["seed"]), str(f.attrs["instruction"])
        n = len(state)
        hist = np.array([at2.history_indices(t, args.num_frames, args.stride) for t in range(n)])
        ep_feats = np.zeros((n, 2, at2.N_COMPACT, at2.FEATURE_DIM), np.float16)
        for s, key in enumerate(("left_tactile_image", "right_tactile_image")):
            tac = data[key]
            for b0 in range(0, n, args.encode_batch):
                idx = hist[b0 : b0 + args.encode_batch]
                ep_feats[b0 : b0 + len(idx), s] = encoder.encode(tac[idx], np.repeat(tac[:1], len(idx), axis=0))
        if args.event_mode == "dense":  # the rule itself is the label wherever it is defined
            ep_jam = np.where(valid, rule.astype(np.float32), -1.0).astype(np.float32)
        else:  # "onset": 1 only for the failure state handed to the expert
            ep_jam = np.where(rule, -1.0, 0.0).astype(np.float32)
            for s0, _, kind, _ in segs:
                t = s0
                while kind == "expert" and t < n and rule[t]:  # the failure state handed to the expert
                    ep_jam[t] = 1.0
                    t += 1

        for s0, s1, kind, fr in segs:
            for t in fr:
                dataset.add_frame(
                    {**{k: data[k][t] for k in KEYS}, "state": state[t], "actions": action[t], "task": instruction}
                )
            dataset.save_episode()
            m = len(fr)
            feats[gidx : gidx + m] = ep_feats[fr]  # tactile history stays that of the full recording
            jam[gidx : gidx + m] = ep_jam[fr]
            info["seed"][gidx : gidx + m] = seed
            info["frame"][gidx : gidx + m] = fr
            info["source"][gidx : gidx + m] = source[fr]
            segment_log.append(
                {
                    "seed": seed,
                    "kind": kind,
                    "start": s0,
                    "end": s1,
                    "kept_frames": int(m),
                    "jam_frames": int((ep_jam[fr] == 1).sum()),
                    "unknown_frames": int((ep_jam[fr] < 0).sum()),
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
        "still_cut": {
            "keep": args.still_keep,
            "joint_eps": args.still_joint_eps,
            "gripper_eps": args.still_gripper_eps,
            "frames_cut": int(n_still_cut),
        },
        "segments": segment_log,
    }
    (out / "tactile" / "anytouch2_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        f"wrote {total} frames ({int((jam == 1).sum())} jam, {int((jam < 0).sum())} unknown) in {len(segment_log)} segments to {out}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
