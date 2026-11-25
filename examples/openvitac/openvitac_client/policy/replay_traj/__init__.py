"""Replays trajectories saved by the OpenPI client (<eval dir>/trajectories/<seed>_<result>.npz): the recorded
absolute qpos8 actions are re-executed one per env step, so the episode (and its video) is reproduced without the
policy server. Run with --start_seed/--max_seed covering the saved seeds; seeds without a file are skipped as errors.

Force tasks (grasp_chip) are executed with the same `action_repeat` as the OpenPI client. Each replayed episode writes
<eval dir>/replay_check/<seed>.json: the original result (from the file name), the max / final joint-state difference
between the recording and the replay (rad / m), and how many actions were replayed, so reproduction can be checked."""
import json
from pathlib import Path

import numpy as np
import torch

from .._base_policy import BasePolicy
from ..openpi.transforms import state8_from_openvitac_observation


class Policy(BasePolicy):
    def __init__(self, deploy_config):
        super().__init__(deploy_config)
        self.root = Path(deploy_config["replay"]["trajectory_dir"])
        self.only_success = bool(deploy_config["replay"].get("only_success", True))
        self.action_repeat = None
        if deploy_config.get("task_name") in ("grasp_chip",):
            from envs._force_task_utils import force_task_openpi_repeat

            self.action_repeat = force_task_openpi_repeat(
                {**deploy_config, "openpi": {"control_mode": "abs_joint", **deploy_config.get("openpi", {})}})
        self._reset_state()

    def _reset_state(self):
        self.actions = self.states = None
        self.file = None
        self.t = 0
        self.diffs = []
        self.task = None

    def _load(self, seed: int):
        pattern = f"{seed}_success.npz" if self.only_success else f"{seed}_*.npz"
        files = sorted(self.root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no saved trajectory for seed {seed} in {self.root}")
        data = np.load(files[0])
        return files[0], data["action"], (data["state"] if "state" in data else None)

    def eval(self, task, observation):
        if self.actions is None:
            self.seed = int(task.cfg.seed)
            self.file, self.actions, self.states = self._load(self.seed)
            self.task, self.save_root = task, Path(task.save_root)
        if self.states is not None and self.t < len(self.states):
            state = state8_from_openvitac_observation(observation)  # what the OpenPI client recorded
            self.diffs.append(np.abs(state - self.states[self.t]))
        action = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        kwargs = {} if self.action_repeat is None else {"action_repeat": self.action_repeat}
        return task.take_action(torch.as_tensor(action, dtype=torch.float32, device=task.device), action_type="qpos",
                                **kwargs)

    def _write_check(self):
        if self.task is None or self.file is None:
            return
        d = np.stack(self.diffs) if self.diffs else np.zeros((1, 8), np.float32)
        out = self.save_root / "replay_check"
        out.mkdir(parents=True, exist_ok=True)
        row = {
            "seed": self.seed, "file": str(self.file), "original": self.file.stem.split("_", 1)[1],
            "replayed_actions": int(self.t), "recorded_actions": int(len(self.actions)),
            "max_joint_diff": float(d[:, :7].max()), "final_joint_diff": float(d[-1, :7].max()),
            "max_gripper_diff": float(d[:, 7].max()),
            "first_step_joint_diff_over_1e-3": int(np.argmax(d[:, :7].max(1) > 1e-3)) if (d[:, :7] > 1e-3).any() else None,
        }
        (out / f"{row['seed']}.json").write_text(json.dumps(row, indent=1))
        print(f"[replay] {row}", flush=True)

    def reset(self):
        self._write_check()
        self._reset_state()

    def close(self):
        self._write_check()
        self._reset_state()
        super().close()
