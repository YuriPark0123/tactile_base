"""Closed-loop log of the tactile world-query predictions.

Each request at step s stores the prediction (standardized future CLS of the clip ending at s + target_offset) and the
current tactile. When the adapter later sends that realized clip (tac_*_target with target_pred_step = s), its CLS is
compared with the prediction and with the copy baseline (the tactile at s). One jsonl line per request; raw vectors
in <vector_dir>/<task>/seed<seed>/step<s>.npz: pred, current (written at s) and actual, target_step (added later),
plus any extra arrays of the request (e.g. attention mass).
"""
import json
from pathlib import Path

import numpy as np


def _cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _mse(a, b):
    return float(((a - b) ** 2).mean())


class QueryLogger:
    def __init__(self, path, encode, target_offset, vector_dir=None):
        """encode: simulator uint8 clips (2, 4, H, W, 3) -> standardized CLS (2, 512) (Policy.encode_tactile)."""
        self.path, self.encode, self.target_offset = path, encode, target_offset
        self.vector_dir = vector_dir if vector_dir is not None else str(Path(path).with_suffix("")) + "_vectors"
        self.pending, self.episode = None, 0

    def log(self, req, tactile, pred, extra_rec=None, extra_arrays=None):
        try:
            task, seed, step = str(req["task"]), int(req.get("seed", -1)), int(req.get("step", -1))
            if int(req.get("episode_start", 0)):
                self.pending, self.episode = {}, self.episode + 1
            rec = dict(task=task, seed=seed, episode=self.episode, step=step, target_offset=self.target_offset)
            if self.pending is None:
                self.pending = {}
            p_key = int(req["target_pred_step"]) if "target_pred_step" in req else (max(self.pending) if self.pending else None)
            if p_key in self.pending and "tac_left_target" in req:
                actual = self.encode(np.stack([req["tac_left_target"], req["tac_right_target"]]))
                p_step, p_pred, p_cur = self.pending.pop(p_key)
                self.pending = {k: v for k, v in self.pending.items() if k > p_key}
                rec.update(pred_step=p_step, target_step=int(req.get("target_step", -1)),
                           mse_pred=_mse(p_pred, actual), mse_copy=_mse(p_cur, actual),
                           cos_pred=_cos(p_pred, actual), cos_copy=_cos(p_cur, actual), change=_mse(actual, p_cur))
                for j, side in enumerate(("left", "right")):
                    rec.update({f"mse_pred_{side}": _mse(p_pred[j], actual[j]), f"mse_copy_{side}": _mse(p_cur[j], actual[j])})
                prev = Path(self.vector_dir) / task / f"seed{seed}" / f"step{p_step:04d}.npz"
                if prev.exists():
                    z = dict(np.load(prev))
                    np.savez(prev, **z, actual=actual.astype(np.float32), target_step=rec["target_step"])
            rec.update(extra_rec or {})
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            d = Path(self.vector_dir) / task / f"seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"step{step:04d}.npz", pred=pred.astype(np.float32), current=tactile.astype(np.float32),
                     step=step, **(extra_arrays or {}))
            self.pending[step] = (step, pred, tactile)
        except Exception as e:  # logging must never break acting
            print(f"query log error: {e}", flush=True)
