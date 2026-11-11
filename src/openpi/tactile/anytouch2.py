"""Frozen AnyTouch2 tactile encoder (PyTorch) used to produce compact tactile tokens for pi0.5.

The same `AnyTouch2Encoder.encode` is used both offline (feature caching for training) and online (policy server),
so the preprocessing is guaranteed to be identical in both paths.

Preprocessing follows AnyTouch2's quick_start.py:
    x = clamp(frame/255 - background/255 + 130/255, 0, 1) -> resize 224 -> normalize(CLIP mean/std)
History: `num_frames` frames spaced by `stride` env steps, ending at the current step (oldest first). Indices before
the episode start are clamped to the first frame.

Cached / returned features per sensor ("compact set", 38 tokens x 768):
    [0]      CLS token
    [1:6]    5 sensor tokens
    [6:38]   patch tokens average-pooled from (T'=2, 14, 14) to (2, 4, 4), time-major
All are taken after AnyTouch2's post_layernorm (before its 512-d CLIP projection).
"""

from __future__ import annotations

import copy
import dataclasses
import os
import pathlib
import sys
import types

import numpy as np

ANYTOUCH2_ROOT = pathlib.Path(
    os.environ.get("ANYTOUCH2_ROOT", pathlib.Path(__file__).resolve().parents[3] / "third_party" / "AnyTouch2")
)

OFFSET = 130.0 / 255.0
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
SENSOR_IDS = {"gelsight": 0, "digit": 1, "gelslim": 2, "gelsight_mini": 3, "duragel": 4, "dm": 5}

N_CLS = 1
N_SENSOR = 5
POOL_GRID = 4  # cached patch grid per time slice
N_TIME = 2  # 4 frames / tubelet 2
N_COMPACT = N_CLS + N_SENSOR + N_TIME * POOL_GRID * POOL_GRID  # 38
FEATURE_DIM = 768


def history_indices(t: int, num_frames: int = 4, stride: int = 2) -> list[int]:
    """Frame indices (oldest first) of the tactile history ending at step t, clamped at 0."""
    return [max(0, t - stride * (num_frames - 1 - k)) for k in range(num_frames)]


@dataclasses.dataclass
class AnyTouch2Config:
    checkpoint: str = "checkpoint-4frames.pth"
    hf_repo: str = "xxuan01/AnyTouch2-Model"
    num_frames: int = 4
    stride: int = 2
    sensor: str = "gelsight_mini"
    device: str = "cuda"
    dtype: str = "float16"


class AnyTouch2Encoder:
    def __init__(self, cfg: AnyTouch2Config | None = None):
        from huggingface_hub import hf_hub_download
        import torch
        from transformers import AutoConfig

        self.cfg = cfg = cfg or AnyTouch2Config()
        if str(ANYTOUCH2_ROOT) not in sys.path:
            sys.path.insert(0, str(ANYTOUCH2_ROOT))
        # AnyTouch2 targets an older transformers whose modeling_clip exported this docstring constant (used only
        # by decorators). Newer versions removed it; provide a placeholder so the module imports.
        import transformers.models.clip.modeling_clip as _mclip

        if not hasattr(_mclip, "CLIP_VISION_INPUTS_DOCSTRING"):
            _mclip.CLIP_VISION_INPUTS_DOCSTRING = ""
        # AnyTouch2 does `from modeling_clip import *` and relies on non-__all__ names (e.g. CLIPVisionTransformer).
        _mclip.__all__ = [n for n in dir(_mclip) if not n.startswith("_")]
        from model.tactile_mae import TactileVideoMAE

        clip_cfg = AutoConfig.from_pretrained(str(ANYTOUCH2_ROOT / "CLIP-B-16" / "config.json"))
        model = TactileVideoMAE(types.SimpleNamespace(mask_ratio=0.0), clip_cfg, cfg.num_frames, 1)
        ckpt_path = cfg.checkpoint
        if not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download(cfg.hf_repo, cfg.checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt and not any("touch_mae_model" in k for k in ckpt):
            ckpt = ckpt["model"]
        state = {}
        for k, v in ckpt.items():
            if "touch_mae_model" in k and "decoder" not in k and "mask_token" not in k:
                state[k.replace("touch_mae_model.", "")] = copy.deepcopy(v)
        missing = [k for k, _ in model.named_parameters() if k not in state]
        for k, v in model.named_parameters():
            state.setdefault(k, v)
        model.load_state_dict(state, strict=True)
        if missing:
            print(f"[AnyTouch2] params not in checkpoint (kept init): {missing}")
        self.torch = torch
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)
        self.model = model.to(self.device, self.dtype).eval().requires_grad_(False)
        self.sensor_id = SENSOR_IDS[cfg.sensor]
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 1, 3, 1, 1)

    def encode(self, frames: np.ndarray, background: np.ndarray) -> np.ndarray:
        """frames: uint8 [B, T, H, W, 3] (oldest first); background: uint8 [B, H, W, 3].

        Returns float16 [B, N_COMPACT, 768].
        """
        torch = self.torch
        with torch.inference_mode():
            x = torch.as_tensor(np.ascontiguousarray(frames), device=self.device).float() / 255.0
            bg = torch.as_tensor(np.ascontiguousarray(background), device=self.device).float() / 255.0
            x = (x - bg[:, None] + OFFSET).clamp_(0.0, 1.0)  # B T H W C
            x = x.permute(0, 1, 4, 2, 3)  # B T C H W
            b, t, c, h, w = x.shape
            if (h, w) != (224, 224):
                x = torch.nn.functional.interpolate(
                    x.reshape(b * t, c, h, w), size=(224, 224), mode="bilinear", antialias=True, align_corners=False
                ).reshape(b, t, c, 224, 224)
            x = (x - self.mean) / self.std
            sensor = torch.full((b,), self.sensor_id, dtype=torch.long, device=self.device)
            hidden = self.model.forward(x.to(self.dtype), sensor, probe=True)  # B, 1+5+2*196, 768
            hidden = self.model.touch_model.post_layernorm(hidden)
            head = hidden[:, : N_CLS + N_SENSOR]
            patches = hidden[:, N_CLS + N_SENSOR :].reshape(b, N_TIME, 14, 14, FEATURE_DIM)
            patches = patches.permute(0, 1, 4, 2, 3).reshape(b * N_TIME, FEATURE_DIM, 14, 14).float()
            pooled = torch.nn.functional.adaptive_avg_pool2d(patches, POOL_GRID)  # (B*T, D, 4, 4)
            pooled = pooled.reshape(b, N_TIME, FEATURE_DIM, POOL_GRID * POOL_GRID).permute(0, 1, 3, 2)
            pooled = pooled.reshape(b, N_TIME * POOL_GRID * POOL_GRID, FEATURE_DIM)
            out = torch.cat([head.float(), pooled], dim=1)
        return out.to(torch.float16).cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
# Token selection from the cached compact set. Kept in numpy so it is shared by the data loader and the server.
# ---------------------------------------------------------------------------------------------------------------
TOKEN_MODES = {
    # name: tokens per sensor
    "cls": 1,
    "cls_sensor": 6,
    "cls_pool1": 3,  # CLS + per-time global average patch (2)
    "cls_pool2": 9,  # CLS + 2 x (2x2) pooled patches
    "cls_pool4": 33,  # CLS + 2 x (4x4) pooled patches
    "all": N_COMPACT,
}


def select_tokens(compact: np.ndarray, mode: str) -> np.ndarray:
    """compact: [..., N_COMPACT, D] -> [..., TOKEN_MODES[mode], D] (float32)."""
    compact = np.asarray(compact, dtype=np.float32)
    cls = compact[..., :1, :]
    if mode == "cls":
        return cls
    if mode == "cls_sensor":
        return compact[..., : N_CLS + N_SENSOR, :]
    if mode == "all":
        return compact
    grid = compact[..., N_CLS + N_SENSOR :, :]
    lead = grid.shape[:-2]
    grid = grid.reshape(*lead, N_TIME, POOL_GRID, POOL_GRID, FEATURE_DIM)
    if mode == "cls_pool4":
        return np.concatenate([cls, grid.reshape(*lead, -1, FEATURE_DIM)], axis=-2)
    if mode == "cls_pool2":
        g = grid.reshape(*lead, N_TIME, 2, 2, 2, 2, FEATURE_DIM).mean(axis=(-4, -2))  # T,2,2,D
        return np.concatenate([cls, g.reshape(*lead, -1, FEATURE_DIM)], axis=-2)
    if mode == "cls_pool1":
        g = grid.mean(axis=(-3, -2))  # T, D
        return np.concatenate([cls, g], axis=-2)
    raise ValueError(f"Unknown tactile token mode: {mode}")
