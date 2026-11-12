"""Convert an OpenViTac (FVL-Lab/OpenViTac, LeRobot v2.1) simulation task into an openpi training dataset and cache
frozen AnyTouch2 tactile features.

Output (under $HF_LEROBOT_HOME/<repo_id>):
  LeRobot dataset with
    image, wrist_image, left_tactile_image, right_tactile_image : uint8 224x224x3 (cv2 INTER_AREA stretch resize,
        identical to the OpenViTac openpi client `resize_rgb_image`)
    state   : float32[8] = 7 arm joints (rad) + left finger position (m)   == client `embodiment/joint[:8]`
    actions : float32[8] = state of the next frame (absolute joint target; last frame repeats)
    task    : language instruction
  tactile/anytouch2_<ckpt>.npy : float16 memmap [N_frames, 2(left,right), 38, 768] indexed by the dataset `index`
    (see openpi.tactile.anytouch2 for the token layout). History = 4 frames, stride 2, clamped at episode start;
    background = first tactile frame of the episode (same as the online client).

Usage:
  uv run examples/openvitac/convert_openvitac.py --src /workspace/UniVTAC/datasets/openvitac/Simulation/GelSight_mini/hardness_classify \
      --repo-id openvitac/hardness_classify_gelsight
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shutil

import av
import cv2
import numpy as np
import pandas as pd
import tqdm
import tyro

from openpi.tactile import anytouch2 as at2

IMAGE_SIZE = 224
VIDEO_KEYS = {
    "image": "observation.images.head",
    "wrist_image": "observation.images.wrist",
    "left_tactile_image": "observation.images.tactile_left",
    "right_tactile_image": "observation.images.tactile_right",
}


@dataclasses.dataclass
class Args:
    src: pathlib.Path
    repo_id: str
    max_episodes: int | None = None
    overwrite: bool = False
    anytouch2_checkpoint: str = "checkpoint-4frames.pth"
    sensor: str = "gelsight_mini"
    num_frames: int = 4
    stride: int = 2
    encode_batch: int = 64
    skip_features: bool = False
    # AnyTouch2 device / dtype (cpu + float32 while the GPU is busy with training)
    device: str = "cuda"
    dtype: str = "float16"


def resize(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def decode_video(path: pathlib.Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(resize(frame.to_ndarray(format="rgb24")))
    return np.stack(frames)


def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((args.src / "meta" / "info.json").read_text())
    episodes = [json.loads(line) for line in (args.src / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    episodes.sort(key=lambda e: e["episode_index"])
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        shutil.rmtree(out)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=info.get("robot_type", "Franka_GelSight_mini"),
        fps=info["fps"],
        features={
            **{
                k: {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
                for k in VIDEO_KEYS
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total = sum(e["length"] for e in episodes)
    encoder = None
    feats = None
    if not args.skip_features:
        encoder = at2.AnyTouch2Encoder(
            at2.AnyTouch2Config(
                checkpoint=args.anytouch2_checkpoint, num_frames=args.num_frames, stride=args.stride, sensor=args.sensor,
                device=args.device, dtype=args.dtype,
            )
        )
        feat_dir = out / "tactile"
        feat_dir.mkdir(parents=True, exist_ok=True)
        feat_path = feat_dir / f"anytouch2_{pathlib.Path(args.anytouch2_checkpoint).stem}.npy"
        feats = np.lib.format.open_memmap(
            feat_path, mode="w+", dtype=np.float16, shape=(total, 2, at2.N_COMPACT, at2.FEATURE_DIM)
        )

    chunk = info.get("chunks_size", 1000)
    gidx = 0
    for ep in tqdm.tqdm(episodes, desc=args.repo_id):
        e = ep["episode_index"]
        c = e // chunk
        df = pd.read_parquet(args.src / info["data_path"].format(episode_chunk=c, episode_index=e))
        df = df.sort_values("frame_index")
        frames = {
            k: decode_video(args.src / info["video_path"].format(episode_chunk=c, video_key=v, episode_index=e))
            for k, v in VIDEO_KEYS.items()
        }
        n = len(df)
        for k, v in frames.items():
            if len(v) != n:
                raise ValueError(f"episode {e}: {k} has {len(v)} frames, parquet has {n}")
        joints = np.stack(df["observation.joint_position"].to_numpy()).astype(np.float32)
        fingers = np.stack(df["observation.finger_position"].to_numpy()).astype(np.float32)
        state = np.concatenate([joints, fingers[:, :1]], axis=1)
        actions = np.concatenate([state[1:], state[-1:]], axis=0)
        task = ep["tasks"][0]

        for t in range(n):
            dataset.add_frame(
                {
                    **{k: frames[k][t] for k in VIDEO_KEYS},
                    "state": state[t],
                    "actions": actions[t],
                    "task": task,
                }
            )
        dataset.save_episode()

        if encoder is not None:
            hist = np.array([at2.history_indices(t, args.num_frames, args.stride) for t in range(n)])
            for s, key in enumerate(("left_tactile_image", "right_tactile_image")):
                tac = frames[key]
                for b0 in range(0, n, args.encode_batch):
                    idx = hist[b0 : b0 + args.encode_batch]
                    bg = np.repeat(tac[:1], len(idx), axis=0)
                    feats[gidx + b0 : gidx + b0 + len(idx), s] = encoder.encode(tac[idx], bg)
        gidx += n

    if feats is not None:
        feats.flush()
        meta = {
            "encoder": "AnyTouch2",
            "checkpoint": args.anytouch2_checkpoint,
            "sensor": args.sensor,
            "num_frames": args.num_frames,
            "stride": args.stride,
            "background": "first tactile frame of episode",
            "layout": "[N, 2(left,right), 38(cls,5 sensor,2x4x4 pooled patches), 768] post_layernorm",
            "frames": int(total),
        }
        (out / "tactile" / "anytouch2_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {gidx} frames to {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
