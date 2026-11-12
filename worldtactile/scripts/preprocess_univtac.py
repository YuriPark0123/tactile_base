"""Contact-rich trimming and AnyTouch2 CLS precomputation for UniVTAC isaac51 episodes.

  python scripts/preprocess_univtac.py --task insert_HDMI [--device cuda]

For each episode of <WT_UNIVTAC_RAW>/<task>/hdf5/*.hdf5:
  contact[t]   press_depth(left) > thr or press_depth(right) > thr
  onset, last  first / last contact frame
  train range  [max(0, onset - margin), min(T - 1, last + margin)]   (frames used as training samples)
  cls[t]       AnyTouch2 CLS (512) of the 4-frame clip ending at t (stride 4), per finger, for every frame, so input
               clips (t-12..t) and future targets (t+12..t+24) are always available
Writes <WT_PROCESSED>/<task>/<episode>.npz (cls float16, press, contact, marker_disp), <task>/index.json, and once
<WT_PROCESSED>/no_contact_cls.npy (CLS of the resting sensor, used by the tactile corruption of the adherence loss).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldtactile import paths  # noqa: E402
from worldtactile import univtac as uv  # noqa: E402
from worldtactile.anytouch2 import AnyTouch2Encoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--contact_thr", type=float, default=1e-3, help="press depth (mm) counted as contact")
    ap.add_argument("--margin", type=int, default=30, help="frames kept before first / after last contact (0.5 s)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--raw", default=None, help=f"raw root (default {paths.UNIVTAC_RAW})")
    ap.add_argument("--out", default=None, help=f"output root (default {paths.PROCESSED})")
    args = ap.parse_args()
    raw_root = Path(args.raw) if args.raw else paths.UNIVTAC_RAW
    out_root = Path(args.out) if args.out else paths.PROCESSED
    out_dir = out_root / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = AnyTouch2Encoder(uv.load_background(), device=args.device)
    if not (out_root / "no_contact_cls.npy").exists():
        np.save(out_root / "no_contact_cls.npy", enc.no_contact_cls().astype(np.float32))

    index = []
    for p in sorted((raw_root / args.task / "hdf5").glob("*.hdf5"), key=lambda p: int(p.stem)):
        f = uv.open_episode(p)
        T = len(f["step"])
        press = np.stack([uv.press_depth_max(f, k) for k in uv.FINGERS], 1)  # (T, 2)
        contact = (press > args.contact_thr).any(1)
        if not contact.any():
            print(f"skip {p.name}: no contact")
            continue
        onset, last = int(np.argmax(contact)), int(T - 1 - np.argmax(contact[::-1]))
        start, end = max(0, onset - args.margin), min(T - 1, last + args.margin)
        cls = np.zeros((T, 2, 512), np.float32)
        for j, finger in enumerate(uv.FINGERS):
            frames = uv.read_tactile_rgb(f, finger)
            clips = np.stack([frames[uv.clip_indices(t, 4, args.stride, T)] for t in range(T)])
            cls[:, j] = np.concatenate([enc.cls(clips[i:i + args.batch]) for i in range(0, T, args.batch)])
        np.savez_compressed(out_dir / f"{p.stem}.npz", cls=cls.astype(np.float16), press=press.astype(np.float32),
                            contact=contact, marker_disp=np.stack([uv.marker_displacement(f, k) for k in uv.FINGERS], 1))
        index.append(dict(episode=p.stem, raw=str(p), T=T, onset=onset, last=last, start=start, end=end))
        print(f"{p.stem}: T={T} onset={onset} last={last} train=[{start},{end}]", flush=True)

    meta = dict(task=args.task, contact_thr=args.contact_thr, margin=args.margin, stride=args.stride,
                input_clip_offsets=[-12, -8, -4, 0], episodes=index,
                train_samples=int(sum(e["end"] - e["start"] + 1 for e in index)))
    (out_dir / "index.json").write_text(json.dumps(meta, indent=1))
    print(f"episodes={len(index)} train_samples={meta['train_samples']}")


if __name__ == "__main__":
    main()
