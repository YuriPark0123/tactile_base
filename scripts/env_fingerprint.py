"""Print everything that can change training results between servers, to diff two machines.

  uv run scripts/env_fingerprint.py [--tasks insert_USB grasp_chip gear_assembly empty_can_selection] > fp_<server>.txt
  diff fp_a.txt fp_b.txt

Covers: code (git commits, local changes), Python packages that touch the numbers (JAX/XLA, data decoding and
resizing, AnyTouch2), GPU / driver / CUDA, XLA environment variables, pretrained files (pi05_base, PaliGemma
tokenizer, AnyTouch2 checkpoint), and per task the converted data, cached AnyTouch2 features and normalization stats.
Hashes of large files are taken over fixed samples (rows / episodes), so they are cheap but still detect differences.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import subprocess

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGES = (
    "jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "flax", "optax", "orbax-checkpoint", "numpy",
    "torch", "torchvision", "transformers", "lerobot", "datasets", "pyarrow", "av", "opencv-python",
    "opencv-python-headless", "pillow", "sentencepiece", "huggingface-hub", "nvidia-cudnn-cu12", "nvidia-cublas-cu12",
)  # fmt: skip
ENV_VARS = ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS", "JAX_PLATFORMS",
            "JAX_DEFAULT_MATMUL_PRECISION", "TF_CPP_MIN_LOG_LEVEL", "CUDA_VISIBLE_DEVICES")  # fmt: skip


def sh(*cmd, cwd=REPO):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        return "n/a"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def file_sha(path: pathlib.Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()[:16]


def dir_sha(path: pathlib.Path) -> str:
    """Hash of all file names and contents under a directory (small directories only)."""
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def section(title, items):
    print(f"[{title}]")
    for k, v in items:
        print(f"  {k}: {v}")


def code():
    sub = REPO / "third_party" / "AnyTouch2"
    section("code", [
        ("tactile_base commit", sh("git", "rev-parse", "HEAD")),
        ("tactile_base local changes", sh("git", "status", "--porcelain", "--untracked-files=no") or "none"),
        ("uv.lock sha", file_sha(REPO / "uv.lock")),
        ("AnyTouch2 submodule commit", sh("git", "rev-parse", "HEAD", cwd=sub)),
    ])  # fmt: skip


def packages():
    items = [("python", platform.python_version())]
    for p in PACKAGES:
        with contextlib.suppress(md.PackageNotFoundError):
            items.append((p, md.version(p)))
    import cv2

    items.append(("cv2 build", cv2.__version__))
    import av

    items.append(("ffmpeg (libavcodec)", ".".join(map(str, av.library_versions.get("libavcodec", ())))))
    tf_dir = pathlib.Path(__import__("transformers").__file__).parent / "models"
    items.append(("transformers gemma/siglip patched (PyTorch only)", (tf_dir / "siglip" / "check.py").exists()))
    section("packages", items)


def hardware():
    env = [(v, os.environ.get(v, "<unset>")) for v in ENV_VARS]  # as set by the user, before the override below
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # do not grab GPU memory (a training run may be active)
    import jax

    items = [
        ("gpu / driver", sh("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")),
        ("jax devices", [str(d) for d in jax.devices()]),
        ("jax default backend", jax.default_backend()),
    ]
    try:
        import torch

        items += [("torch cuda / cudnn", f"{torch.version.cuda} / {torch.backends.cudnn.version()}")]
    except ImportError:
        pass
    section("hardware / runtime", items + env)


def pretrained():
    data_home = pathlib.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    params = data_home / "openpi-assets" / "checkpoints" / "pi05_base" / "params"
    tok = data_home / "big_vision" / "paligemma_tokenizer.model"
    items = [
        ("pi05_base params metadata", dir_sha(params / "_METADATA") if (params / "_METADATA").is_dir()
         else file_sha(params / "_METADATA") if (params / "_METADATA").exists() else "missing"),
        ("pi05_base params first 64MB of ocdbt", next((file_sha(p, 1 << 26) for p in sorted((params / "d").glob("*"))[:1]),
                                                      "missing") if (params / "d").exists() else "missing"),
        ("paligemma tokenizer", file_sha(tok) if tok.exists() else "missing"),
    ]  # fmt: skip
    try:
        from huggingface_hub import hf_hub_download

        ckpt = pathlib.Path(hf_hub_download("xxuan01/AnyTouch2-Model", "checkpoint-4frames.pth"))
        items.append(("AnyTouch2 checkpoint-4frames.pth", f"{file_sha(ckpt)} ({ckpt.stat().st_size} bytes)"))
    except Exception as e:
        items.append(("AnyTouch2 checkpoint-4frames.pth", f"unavailable: {type(e).__name__}"))
    section("pretrained files", items)


def task(name: str, tactile_config: str, vision_config: str):
    import pyarrow.parquet as pq

    lerobot = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()
    root = lerobot / "openvitac" / f"{name}_gelsight"
    items = []
    if not root.exists():
        section(f"task {name}", [("converted dataset", f"missing ({root})")])
        return
    info = json.loads((root / "meta" / "info.json").read_text())
    items.append(("episodes / frames", f"{info['total_episodes']} / {info['total_frames']}"))
    items.append(("prompt", (root / "meta" / "tasks.jsonl").read_text().splitlines()[0]))
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    for f in (files[0], files[len(files) // 2], files[-1]):
        t = pq.read_table(f)
        h = hashlib.sha256()
        for col in ("image", "wrist_image", "left_tactile_image", "right_tactile_image"):
            for x in t.column(col).to_pylist()[::20]:  # PNG bytes are deterministic given decoded pixels
                h.update(x["bytes"])
        st = np.stack(t.column("state").to_numpy(zero_copy_only=False)).astype(np.float32)
        items.append((f"{f.stem} images (every 20th frame)", h.hexdigest()[:16]))
        items.append((f"{f.stem} state", sha(st.tobytes())))
    cache = root / "tactile" / "anytouch2_checkpoint-4frames.npy"
    if cache.exists():
        mm = np.load(cache, mmap_mode="r")
        rows = np.linspace(0, len(mm) - 1, 64).astype(int)
        sample = np.asarray(mm[rows], dtype=np.float32)
        items.append(("AnyTouch2 cache shape", tuple(mm.shape)))
        items.append(("AnyTouch2 cache 64 rows sha", sha(np.asarray(mm[rows]).tobytes())))
        items.append(("AnyTouch2 cache 64 rows mean / std", f"{sample.mean():.6f} / {sample.std():.6f}"))
        meta = root / "tactile" / "anytouch2_meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            items.append(("AnyTouch2 cache meta", {k: m[k] for k in m if k in ("stride", "num_frames", "device", "dtype")}))
    for cfg in (tactile_config, vision_config):
        ns = REPO / "assets" / cfg / "openvitac" / f"{name}_gelsight" / "norm_stats.json"
        if not ns.exists():
            ns = REPO / "assets" / cfg / "openvitac" / name / "norm_stats.json"
        items.append((f"norm_stats {cfg}", file_sha(ns) if ns.exists() else "missing"))
    section(f"task {name}", items)


def configs(names):
    from openpi.training import config as _config

    items = []
    for n in names:
        try:
            c = _config.get_config(n)
        except ValueError:
            continue
        m = c.model
        items.append((n, dict(  # noqa: C408
            batch=c.batch_size, grad_accum=getattr(c, "grad_accum_steps", 1), steps=c.num_train_steps, seed=c.seed,
            lr=repr(c.lr_schedule), horizon=m.action_horizon, tactile_tokens=getattr(m, "tactile_tokens", 0),
            projector=getattr(m, "tactile_projector", None), paligemma=m.paligemma_variant,
            expert=m.action_expert_variant, dtype=m.dtype,
        )))  # fmt: skip
    section("training configs", items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["insert_USB", "grasp_chip", "gear_assembly", "empty_can_selection"])
    args = ap.parse_args()
    code()
    packages()
    hardware()
    pretrained()
    for t in args.tasks:
        task(t, f"openvtla_{t}", f"pi05_vision_{t}")
    section("task multitask4", [
        (f"norm_stats {c}", file_sha(p) if (p := REPO / "assets" / c / "openvitac" / "multitask4" / "norm_stats.json").exists()
         else "missing") for c in ("openvtla_multitask", "pi05_vision_multitask")
    ])  # fmt: skip
    configs([f"openvtla_{t}" for t in args.tasks] + ["openvtla_multitask"])


if __name__ == "__main__":
    main()
