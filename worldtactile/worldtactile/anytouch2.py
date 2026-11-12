"""Frozen AnyTouch2 tactile encoder (https://github.com/GeWu-Lab/AnyTouch2).

Preprocessing follows AnyTouch2's quick start: (img - background + 130/255), clamp to [0, 1], resize to 224x224,
CLIP mean/std. Clips are 4 frames; sensor id 3 (GelSight Mini). The output is the projected CLS (512).
"""
import copy
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from . import paths

SENSOR_ID_GELSIGHT_MINI = 3
OFFSET = 130.0 / 255.0
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)


def _load_state(ckpt, model):
    state = {}
    for key, item in ckpt.items():
        if "touch_mae_model" in key and "decoder" not in key and "mask_token" not in key:
            state[key.replace("touch_mae_model.", "")] = copy.deepcopy(item)
    for k, v in model.named_parameters():
        state.setdefault(k, v)
    model.load_state_dict(state, strict=True)
    return model


class AnyTouch2Encoder:
    def __init__(self, background, device="cuda", num_frames=4, root=None):
        """background: (H, W, 3) uint8 image of the sensor without contact."""
        root = str(root or paths.ANYTOUCH2_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from model.tactile_mae import TactileVideoMAE
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(f"{root}/CLIP-B-16/config.json")
        model = TactileVideoMAE(SimpleNamespace(mask_ratio=0.0), config, num_frames, 1)
        model = _load_state(torch.load(f"{root}/checkpoints/checkpoint-4frames.pth", map_location="cpu"), model)
        self.model = model.eval().to(device)
        self.device = device
        self.background = background
        self.bg = torch.from_numpy(background).permute(2, 0, 1).float().div(255.0).to(device)

    def preprocess(self, clips):
        """clips: uint8 (B, T, H, W, 3) -> float (B, T, 3, 224, 224)."""
        x = torch.from_numpy(clips).to(self.device).permute(0, 1, 4, 2, 3).float().div(255.0)
        x = (x - self.bg + OFFSET).clamp(0.0, 1.0)
        b, t = x.shape[:2]
        x = F.interpolate(x.flatten(0, 1), size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
        x = x.view(b, t, 3, 224, 224)
        return (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)

    @torch.no_grad()
    def cls(self, clips):
        """uint8 clips (B, T, H, W, 3) -> projected CLS (B, 512) float32."""
        x = self.preprocess(clips)
        sensor = torch.full((x.shape[0],), SENSOR_ID_GELSIGHT_MINI, dtype=torch.long, device=self.device)
        return self.model(x, sensor, probe=True, get_cls=True).float().cpu().numpy()

    def no_contact_cls(self):
        """CLS of the resting sensor: a clip of four background frames, (512,)."""
        return self.cls(np.stack([np.stack([self.background] * 4)]))[0]
