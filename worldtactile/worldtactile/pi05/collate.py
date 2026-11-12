"""Batching for the pi0.5 model with openpi's image and prompt conventions.

Images  letterboxed (aspect kept, black padding) to 224x224, float in [-1, 1]; camera slots base_0_rgb = head,
        left_wrist_0_rgb = wrist, right_wrist_0_rgb absent (zeros, masked).
Prompt  pi0.5 format "Task: <instruction>, State: <s_1 ... s_32>;\nAction: " with the normalized joint state
        zero-padded to 32 dims and discretized into 256 bins; PaliGemma sentencepiece, BOS, padded/truncated to 200.
"""
import numpy as np
import sentencepiece
import torch
from PIL import Image

from .. import paths

MAX_TOKENS = 200
STATE_DIM = 32
SIZE = 224


def letterbox(img, size=SIZE):
    """PIL image -> (3, size, size) float32 in [-1, 1], aspect kept, black padding (openpi resize_with_pad)."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


class Pi05Collator:
    def __init__(self, cameras=("head", "wrist"), tokenizer_path=None, max_tokens=MAX_TOKENS):
        self.cameras = list(cameras)
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path or paths.PALIGEMMA_TOKENIZER))
        self.max_tokens = max_tokens

    def tokenize(self, instruction, state):
        """Returns token ids, validity mask and a mask of the state-digit tokens (for the attention probe)."""
        st = np.zeros(STATE_DIM, np.float32)
        st[: len(state)] = np.clip(state, -1, 1)
        disc = np.digitize(st, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        clean = instruction.strip().replace("_", " ").replace("\n", " ")
        head = f"Task: {clean}, State: "
        prompt = f"{head}{' '.join(map(str, disc))};\nAction: "
        ids = [self.sp.bos_id()] + self.sp.encode(prompt)
        ids = ids[: self.max_tokens]
        n_head = 1 + len(self.sp.encode(head.rstrip()))  # BOS + "Task: ..., State:" (the state digits follow)
        n_tail = len(self.sp.encode(";\nAction:"))
        is_state = [n_head <= i < len(ids) - n_tail for i in range(len(ids))]
        pad = self.max_tokens - len(ids)
        return ids + [0] * pad, [True] * len(ids) + [False] * pad, is_state + [False] * pad

    def __call__(self, items):
        B = len(items)
        images, image_masks = [], []
        for slot in range(3):
            if slot < len(self.cameras):
                images.append(torch.stack([letterbox(it["images"][self.cameras[slot]]) for it in items]))
                image_masks.append(torch.ones(B, dtype=torch.bool))
            else:
                images.append(torch.zeros(B, 3, SIZE, SIZE))
                image_masks.append(torch.zeros(B, dtype=torch.bool))
        toks = [self.tokenize(it["instruction"], it["state"].numpy()) for it in items]
        batch = dict(
            images=images, image_masks=image_masks,
            tokens=torch.tensor([t for t, _, _ in toks], dtype=torch.long),
            token_masks=torch.tensor([m for _, m, _ in toks], dtype=torch.bool),
            state_token_mask=torch.tensor([s for _, _, s in toks], dtype=torch.bool),
            state=torch.stack([it["state"] for it in items]),
            actions=torch.stack([it["actions"] for it in items]).float(),
            action_mask=torch.stack([it["action_mask"] for it in items]),
            tactile=torch.stack([it["tactile"] for it in items]).float(),
            tactile_target=torch.stack([it["tactile_target"] for it in items]).float(),
            target_mask=torch.stack([it["target_mask"] for it in items]),
        )
        if "tactile_corrupt" in items[0]:
            batch["tactile_corrupt"] = torch.stack([it["tactile_corrupt"] for it in items]).float()
        return batch


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, list):
            out[k] = [x.to(device, non_blocking=True) for x in v]
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
