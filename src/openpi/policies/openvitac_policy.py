"""OpenViTac (Franka + 2 GelSight Mini) policy transforms for pi0.5 and pi0.5 + AnyTouch2 (OpenVTLA-style).

Inference observation keys (sent by the OpenViTac openvtla client):
    observation/state                 float32[8]  7 joints + left finger (m)
    observation/image                 uint8[224,224,3]  head camera
    observation/wrist_image           uint8[224,224,3]  wrist camera
    observation/left_tactile_history  uint8[T,224,224,3]  tactile frames (oldest first, stride already applied)
    observation/right_tactile_history uint8[T,224,224,3]
    observation/left_tactile_background, observation/right_tactile_background uint8[224,224,3]
    prompt                            str
Training additionally provides `observation/tactile_index` (dataset global index) used to fetch cached features.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.tactile import anytouch2 as at2


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_openvitac_example(*, tactile: bool = True) -> dict:
    ex = {
        "observation/state": np.zeros(8, np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
    if tactile:
        for side in ("left", "right"):
            ex[f"observation/{side}_tactile_history"] = np.random.randint(256, size=(4, 224, 224, 3), dtype=np.uint8)
            ex[f"observation/{side}_tactile_background"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return ex


@dataclasses.dataclass(frozen=True)
class OpenViTacInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    # None -> vision-only pi0.5. Otherwise an openpi.tactile.anytouch2.TOKEN_MODES key.
    tactile_mode: str | None = None
    # Path of the cached AnyTouch2 feature memmap (training only).
    tactile_cache: str | None = None
    # AnyTouch2 checkpoint used for online encoding (inference only); must match the cache.
    anytouch2_checkpoint: str = "checkpoint-4frames.pth"
    # Replace the tactile history with copies of the current frame (temporal ablation, paper Tab. S6).
    repeat_current_frame: bool = False

    _memmaps: ClassVar[dict] = {}
    _encoders: ClassVar[dict] = {}

    def _features(self, data: dict) -> np.ndarray:
        """Returns compact AnyTouch2 features [2, 38, 768]."""
        if "observation/tactile_index" in data:
            if self.tactile_cache is None:
                raise ValueError("tactile_cache is required when training from cached features")
            mm = self._memmaps.get(self.tactile_cache)
            if mm is None:
                mm = np.load(self.tactile_cache, mmap_mode="r")
                self._memmaps[self.tactile_cache] = mm
            return np.asarray(mm[int(np.asarray(data["observation/tactile_index"]))])

        enc = self._encoders.get(self.anytouch2_checkpoint)
        if enc is None:
            enc = at2.AnyTouch2Encoder(at2.AnyTouch2Config(checkpoint=self.anytouch2_checkpoint))
            self._encoders[self.anytouch2_checkpoint] = enc
        frames, bgs = [], []
        for side in ("left", "right"):
            hist = np.asarray(data[f"observation/{side}_tactile_history"])
            if self.repeat_current_frame:
                hist = np.repeat(hist[-1:], hist.shape[0], axis=0)
            frames.append(hist)
            bgs.append(np.asarray(data[f"observation/{side}_tactile_background"]))
        return enc.encode(np.stack(frames), np.stack(bgs))

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if self.tactile_mode is not None:
            tokens = at2.select_tokens(self._features(data), self.tactile_mode)  # [2, k, 768]
            inputs["tactile"] = tokens.reshape(-1, tokens.shape[-1]).astype(np.float32)
            inputs["tactile_mask"] = np.True_

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class OpenViTacOutputs(transforms.DataTransformFn):
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}


def default_tactile_cache(repo_id: str, checkpoint: str = "checkpoint-4frames.pth") -> str:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    return str(HF_LEROBOT_HOME / repo_id / "tactile" / f"anytouch2_{pathlib.Path(checkpoint).stem}.npy")
