"""State / action normalization (openpi-style quantile normalization to [-1, 1]).

state    joint[t] (9)                       -> quantile-normalized with q01/q99 over all frames, clipped to [-1, 1]
actions  joint[t+1 .. t+H] (H, 9)           -> arm joints 0..6 as deltas joint[t+k] - joint[t], quantile-normalized
                                               with per-task, per-chunk-index q01/q99 (delta_q01__<task>, (H, 9));
                                               finger joints 7, 8 absolute, quantile-normalized with q01/q99
A run's statistics are saved as norm_stats.npz next to its checkpoints.
"""
import numpy as np

ARM = 7  # arm joints; joints 7 and 8 are the fingers


def load_norm_stats(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _q(x, lo, hi):
    return 2 * (x - lo) / (hi - lo) - 1


def _q_inv(x, lo, hi):
    return (x + 1) / 2 * (hi - lo) + lo


def normalize_state(state, stats):
    return np.clip(_q(state, stats["q01"], stats["q99"]), -1, 1).astype(np.float32)


def normalize_actions(actions, state, task, stats):
    """Absolute joint chunk (H, 9) and the current joints (9,) -> model space (H, 9)."""
    out = _q(actions, stats["q01"], stats["q99"])
    d = actions[:, :ARM] - state[None, :ARM]
    out[:, :ARM] = _q(d, stats[f"delta_q01__{task}"][:, :ARM], stats[f"delta_q99__{task}"][:, :ARM])
    return out.astype(np.float32)


def denormalize_actions(x, state, task, stats):
    """Model-space chunk (H, 9) -> absolute joint targets (H, 9); inverse of normalize_actions."""
    out = _q_inv(x, stats["q01"], stats["q99"])
    out[:, :ARM] = state[None, :ARM] + _q_inv(x[:, :ARM], stats[f"delta_q01__{task}"][:, :ARM],
                                               stats[f"delta_q99__{task}"][:, :ARM])
    return out
