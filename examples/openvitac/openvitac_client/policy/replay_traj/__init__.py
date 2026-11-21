"""Replays trajectories saved by the OpenPI client (<eval dir>/trajectories/<seed>_<result>.npz): the recorded
absolute qpos8 actions are re-executed one per env step, so the episode (and its video) is reproduced without the
policy server. Run with --start_seed/--max_seed covering the saved seeds; seeds without a file are skipped as errors."""
from pathlib import Path

import numpy as np
import torch

from .._base_policy import BasePolicy


class Policy(BasePolicy):
    def __init__(self, deploy_config):
        super().__init__(deploy_config)
        self.root = Path(deploy_config["replay"]["trajectory_dir"])
        self.only_success = bool(deploy_config["replay"].get("only_success", True))
        self.actions = None
        self.t = 0

    def _load(self, seed: int) -> np.ndarray:
        pattern = f"{seed}_success.npz" if self.only_success else f"{seed}_*.npz"
        files = sorted(self.root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no saved trajectory for seed {seed} in {self.root}")
        return np.load(files[0])["action"]

    def eval(self, task, observation):
        if self.actions is None:
            self.actions = self._load(int(task.cfg.seed))
        action = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        return task.take_action(torch.as_tensor(action, dtype=torch.float32, device=task.device), action_type="qpos")

    def reset(self):
        self.actions = None
        self.t = 0
