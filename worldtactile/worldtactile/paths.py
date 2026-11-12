"""External files and directories. Each can be overridden with the environment variable named in the comment."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(var, default):
    return Path(os.environ.get(var, default))


# UniVTAC isaac51 episodes (Hugging Face dataset byml/UniVTAC, folder isaac51/<task>/hdf5/<n>.hdf5)
UNIVTAC_RAW = _path("WT_UNIVTAC_RAW", "/data/datasets/univtac/isaac51")
# UniVTAC task instructions (<UniVTAC repo>/instructions/<task>.json)
UNIVTAC_INSTRUCTIONS = _path("WT_UNIVTAC_INSTRUCTIONS", "/data/UniVTAC/instructions")
# Output of scripts/preprocess_univtac.py (per-episode AnyTouch2 CLS, contact ranges, index.json)
PROCESSED = _path("WT_PROCESSED", "/data/datasets/univtac/processed_rgb")
# AnyTouch2 repository with checkpoints/checkpoint-4frames.pth and CLIP-B-16/config.json
ANYTOUCH2_ROOT = _path("WT_ANYTOUCH2_ROOT", "/data/AnyTouch2")
# pi0.5 base in openpi's PyTorch format (scripts/convert_openpi_checkpoint.py) and the PaliGemma tokenizer
PI05_BASE = _path("WT_PI05_BASE", "/data/models/openpi/pi05_base_pytorch")
PALIGEMMA_TOKENIZER = _path("WT_PALIGEMMA_TOKENIZER", "/data/models/openpi/paligemma_tokenizer.model")
# Training runs (<root>/<run_name>/)
EXP_ROOT = _path("WT_EXP_ROOT", "/data/experiments")
# Simulated GelSight Mini image without contact (true RGB), shipped with the repository
GEL_BACKGROUND = _path("WT_GEL_BACKGROUND", REPO_ROOT / "resources" / "gsmini_sim_background_rgb.png")
