"""DAgger (expert-intervention) data collection on top of the OpenPI client.

The learned policy (served by an OpenPI server, e.g. tactile_base `openvtla_<task>`) runs as in evaluation. A task
plugin (`policy/openpi_dagger/tasks/<task>.py`) watches privileged simulator state; when it reports a failure mode the
policy cannot get out of (e.g. a jammed insertion), its scripted expert takes over using the task's own scripted
primitives. Every frame is recorded with its source (policy / expert), so the expert segments can be used as
corrective training data that start from the state the policy itself reached.

Output per episode: <eval dir>/dagger/<seed>_<success|failed>.hdf5
    head, wrist, left_tactile, right_tactile : uint8 [T,224,224,3] (exactly what the OpenPI client sends)
    state  : float32 [T,8]   7 joints + left finger (m)
    action : float32 [T,8]   policy frames: commanded abs qpos target; expert frames: state of the next expert frame
                             (same definition as the scripted demonstrations)
    source : uint8 [T]       0 = policy, 1 = expert
    event_rule : uint8 [T]   1 where the plugin's failure-mode rule holds (used for tactile event labels)
    event_valid: uint8 [T]   0 where the event label is undefined (plugins with dense labels, e.g. "no can lifted")
    goal   : uint8 [T]       1 where the plugin's sub-goal of the recovery is reached (e.g. plug inserted)
    sim_step : int32 [T]
    <plugin diagnostics>     float32 [T,...] task-specific privileged quantities (see the plugin)
    attrs: seed, success, n_interventions, instruction, task, reasons (json)
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from ..openpi.deploy_policy import Policy as OpenPiPolicy
from ..openpi.transforms import openpi_obs_from_openvitac

IMAGE_KEYS = ("head", "wrist", "left_tactile", "right_tactile")


class Policy(OpenPiPolicy):
    def __init__(self, deploy_config: dict):
        super().__init__(deploy_config)
        cfg = dict(deploy_config.get("dagger", {}))
        self.task_name = str(cfg.get("task", deploy_config.get("task_name")))
        module = importlib.import_module(f"{__name__}.tasks.{self.task_name}")
        self.plugin = module.Expert(cfg)
        self.max_interventions = int(cfg.get("max_interventions", 2))
        self.frame_every = int(cfg.get("expert_frame_every", 2))  # sim steps per recorded expert frame (= demos)
        self.save_trajectory = False  # replaced by the richer DAgger record
        self._reset_record()

    # ------------------------------------------------------------------ recording
    def _reset_record(self):
        self._rec: dict[str, list] = {k: [] for k in (*IMAGE_KEYS, "state", "action", "source", "event_rule", "event_valid",
                                                      "goal", "sim_step")}
        self._diag_rec: dict[str, list] = {}
        self._meta = {}
        self._n_interventions = 0
        self._reasons: list[str] = []
        self.plugin.reset()

    def _record(self, task, observation, source: int, diag: dict | None = None):
        obs = openpi_obs_from_openvitac(
            observation=observation, prompt="", image_size=self.image_size, send_tactile=True,
            tactile_image_keys=self.tactile_image_keys, image_color_order=self.image_color_order,
            control_mode=self.control_mode, eef_state_mode=self.eef_state_mode,
        )
        diag = self.plugin.diag(task) if diag is None else diag
        r = self._rec
        r["head"].append(obs["observation/image"])
        r["wrist"].append(obs["observation/wrist_image"])
        r["left_tactile"].append(obs["observation/left_tactile_image"])
        r["right_tactile"].append(obs["observation/right_tactile_image"])
        r["state"].append(np.asarray(obs["observation/state"], np.float32))
        r["action"].append(None)
        r["source"].append(source)
        r["event_rule"].append(int(bool(self.plugin.event_rule(diag))))
        r["event_valid"].append(int(bool(self.plugin.event_valid(diag))) if hasattr(self.plugin, "event_valid") else 1)
        r["goal"].append(int(bool(self.plugin.goal_reached(diag))))
        r["sim_step"].append(int(task.step_count))
        for k, v in diag.items():
            self._diag_rec.setdefault(k, []).append(np.asarray(v, np.float32))
        self._meta.update(seed=int(task.cfg.seed), save_root=str(task.save_root),
                          instruction=self._get_prompt(task))  # the prompt the policy was given
        return obs

    def _flush(self):
        r, meta = self._rec, self._meta
        n = len(r["state"])
        if n == 0 or "seed" not in meta:
            self._reset_record()
            return
        import h5py

        # frames without an executed action (expert frames) take the next recorded state (demo convention)
        for i in range(n):
            if r["action"][i] is None:
                r["action"][i] = r["state"][min(i + 1, n - 1)].copy()
        success = bool(meta.get("success", False))
        out = Path(meta["save_root"]) / "dagger"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{meta['seed']}_{'success' if success else 'failed'}.hdf5"
        with h5py.File(path, "w") as f:
            for k in IMAGE_KEYS:
                f.create_dataset(k, data=np.stack(r[k]), compression="gzip", compression_opts=4, chunks=(1, 224, 224, 3))
            f["state"] = np.stack(r["state"])
            f["action"] = np.stack(r["action"])
            f["source"] = np.asarray(r["source"], np.uint8)
            f["event_rule"] = np.asarray(r["event_rule"], np.uint8)
            f["event_valid"] = np.asarray(r["event_valid"], np.uint8)
            f["goal"] = np.asarray(r["goal"], np.uint8)
            f["sim_step"] = np.asarray(r["sim_step"], np.int32)
            for k, v in self._diag_rec.items():
                f[k] = np.stack(v)
            f.attrs.update(seed=meta["seed"], success=success, n_interventions=self._n_interventions,
                           instruction=meta["instruction"], task=self.task_name, reasons=json.dumps(self._reasons))
        print(f"[dagger] wrote {path} ({n} frames, expert {int(np.sum(r['source']))}, "
              f"interventions {self._n_interventions})", flush=True)
        self._reset_record()

    # ------------------------------------------------------------------ expert takeover
    def _recover(self, task, reason: str):
        self._n_interventions += 1
        self._reasons.append(reason)
        print(f"[dagger] intervention {self._n_interventions} at step {task.take_action_cnt}: {reason}", flush=True)
        original_step = task._step
        counter = {"n": 0}

        def recording_step(*args, **kwargs):
            out = original_step(*args, **kwargs)
            counter["n"] += 1
            if counter["n"] % self.frame_every == 0 and task.plan_success is not False:
                task._update_render()
                self._push_tactile_history(self._record(task, task._get_observations(), source=1))
            return out

        task._step = recording_step
        # In eval mode BaseTask.move() treats gripper targets as raw qpos unless the task is in its pre-move phase,
        # while the scripted primitives (close_gripper(percent), adaptive grasp) expect the collection semantics.
        # Use the scripted-demo gripper path for the takeover.
        prev_in_pre_move = task.in_pre_move
        task.in_pre_move = True
        try:
            # the state the policy reached is the first expert frame
            task._update_render()
            self._push_tactile_history(self._record(task, task._get_observations(), source=1))
            self.plugin.recover(task, reason)
        finally:
            task._step = original_step
            task.in_pre_move = prev_in_pre_move
        if task.plan_success is not False and task.check_success():
            task.eval_success = True
        # robot state jumped: drop chunk predictions made before the intervention
        self._action_chunk, self._chunk_step = None, 0
        self._action_predictions.clear()
        self.plugin.after_recover(task)
        print(f"[dagger] intervention done, success={task.eval_success}, plan_success={task.plan_success}", flush=True)

    def _push_tactile_history(self, obs):
        if self.tactile_history_frames > 0:
            for side in ("left", "right"):
                self._tactile_frames[side].append(obs[f"observation/{side}_tactile_image"])

    # ------------------------------------------------------------------ policy step
    def eval(self, task, observation):
        diag = self.plugin.diag(task)
        reason = self.plugin.trigger(task, diag)
        if reason is not None and self._n_interventions < self.max_interventions and not task.eval_success:
            self._recover(task, reason)
            self._meta["success"] = bool(task.eval_success)
            return True, task.eval_success
        self._record(task, observation, source=0, diag=diag)
        result = super().eval(task, observation)
        self._rec["action"][-1] = self._last_action
        self._meta["success"] = bool(task.eval_success)
        return result

    def _on_executed_action(self, task, obs, torch_action):
        self._last_action = torch_action.detach().cpu().numpy().astype(np.float32)

    def reset(self):
        self._flush()
        super().reset()

    def close(self):
        self._flush()
        super().close()
