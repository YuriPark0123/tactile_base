"""DAgger (expert-intervention) data collection on top of the OpenPI client.

The learned policy (served by an OpenPI server, e.g. tactile_base `openvtla_insert_USB`) runs as in evaluation. When
privileged simulator state shows the insertion is jammed, a scripted expert takes over:
lift -> re-align the USB over the slot opening -> insert (the same primitives as the scripted demonstrations).
Every frame is recorded, with its source (policy / expert), so the expert segments can be used as corrective training
data: they start from the jammed state the policy reached, i.e. with the jam visible in the tactile images.

Output per episode: <eval dir>/dagger/<seed>_<success|failed>.hdf5
    head, wrist, left_tactile, right_tactile : uint8 [T,224,224,3] (exactly what the OpenPI client sends)
    state  : float32 [T,8]   7 joints + left finger (m)
    action : float32 [T,8]   policy frames: commanded abs qpos target; expert frames: state of the next expert frame
                             (same definition as the scripted demonstrations)
    source : uint8 [T]       0 = policy, 1 = expert
    rel_xyz: float32 [T,3]   USB pose in the inserted-target frame (m); z = 0 when inserted, ~0.0124 at the opening
    xy_err, tilt_deg : float32 [T]
    sim_step : int32 [T]
    attrs: seed, success, n_interventions, instruction
Only insert_USB is supported (it uses the task's slot/USB actors and insertion helpers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..openpi.deploy_policy import Policy as OpenPiPolicy
from ..openpi.transforms import openpi_obs_from_openvitac

PLUG_LEN = 0.0124  # USB plug length: rel z at the slot opening


class Policy(OpenPiPolicy):
    def __init__(self, deploy_config: dict):
        super().__init__(deploy_config)
        cfg = dict(deploy_config.get("dagger", {}))
        self.max_interventions = int(cfg.get("max_interventions", 2))
        self.stuck_window = int(cfg.get("stuck_window", 15))  # policy steps without downward progress
        self.stuck_progress = float(cfg.get("stuck_progress", 0.0005))
        self.zone_xy = float(cfg.get("zone_xy", 0.008))  # insertion zone around the slot axis
        self.zone_z = float(cfg.get("zone_z", 0.025))
        self.contact_xy = float(cfg.get("contact_xy", 0.0012))  # misaligned contact at the opening
        self.contact_z = float(cfg.get("contact_z", PLUG_LEN + 0.0015))
        self.lift = float(cfg.get("lift", 0.010))
        self.frame_every = int(cfg.get("expert_frame_every", 2))  # sim steps per recorded expert frame (= demos)
        self.save_trajectory = False  # replaced by the richer DAgger record
        self._reset_record()

    # ------------------------------------------------------------------ recording
    def _reset_record(self):
        self._rec = {k: [] for k in ("head", "wrist", "left_tactile", "right_tactile", "state", "action", "source",
                                     "rel_xyz", "xy_err", "tilt_deg", "sim_step")}
        self._meta = {}
        self._z_hist: list[float] = []
        self._n_interventions = 0

    def _diag(self, task):
        """USB pose in the inserted-target frame (same quantities as insert_USB's success check)."""
        rel = task.prism.get_pose().rebase(task._usb_pose_in_slot(task.slot.get_pose()))
        p = np.asarray(rel.p, np.float32)
        z_axis = rel.to_transformation_matrix()[:3, 2]
        tilt = float(np.degrees(np.arccos(np.clip(float(z_axis[2]), -1.0, 1.0))))
        return p, float(np.linalg.norm(p[:2])), tilt

    def _record(self, task, observation, source: int, action=None):
        obs = openpi_obs_from_openvitac(
            observation=observation, prompt="", image_size=self.image_size, send_tactile=True,
            tactile_image_keys=self.tactile_image_keys, image_color_order=self.image_color_order,
            control_mode=self.control_mode, eef_state_mode=self.eef_state_mode,
        )
        rel, xy, tilt = self._diag(task)
        r = self._rec
        r["head"].append(obs["observation/image"])
        r["wrist"].append(obs["observation/wrist_image"])
        r["left_tactile"].append(obs["observation/left_tactile_image"])
        r["right_tactile"].append(obs["observation/right_tactile_image"])
        r["state"].append(np.asarray(obs["observation/state"], np.float32))
        r["action"].append(None if action is None else np.asarray(action, np.float32))
        r["source"].append(source)
        r["rel_xyz"].append(rel)
        r["xy_err"].append(xy)
        r["tilt_deg"].append(tilt)
        r["sim_step"].append(int(task.step_count))
        self._meta.update(seed=int(task.cfg.seed), save_root=str(task.save_root),
                          instruction=str(getattr(task, "instruction", "")))
        return obs

    def _flush(self):
        r, meta = self._rec, self._meta
        n = len(r["state"])
        if n == 0 or "seed" not in meta:
            self._reset_record()
            return
        import h5py

        # expert frames without an explicit action take the next recorded state (demo convention)
        for i in range(n):
            if r["action"][i] is None:
                r["action"][i] = r["state"][min(i + 1, n - 1)].copy()
        success = bool(meta.get("success", False))
        out = Path(meta["save_root"]) / "dagger"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{meta['seed']}_{'success' if success else 'failed'}.hdf5"
        with h5py.File(path, "w") as f:
            for k in ("head", "wrist", "left_tactile", "right_tactile"):
                f.create_dataset(k, data=np.stack(r[k]), compression="gzip", compression_opts=4, chunks=(1, 224, 224, 3))
            f["state"] = np.stack(r["state"])
            f["action"] = np.stack(r["action"])
            f["source"] = np.asarray(r["source"], np.uint8)
            f["rel_xyz"] = np.stack(r["rel_xyz"])
            f["xy_err"] = np.asarray(r["xy_err"], np.float32)
            f["tilt_deg"] = np.asarray(r["tilt_deg"], np.float32)
            f["sim_step"] = np.asarray(r["sim_step"], np.int32)
            f.attrs.update(seed=meta["seed"], success=success, n_interventions=self._n_interventions,
                           instruction=meta["instruction"])
        print(f"[dagger] wrote {path} ({n} frames, expert {int(np.sum(r['source']))}, "
              f"interventions {self._n_interventions})", flush=True)
        self._reset_record()

    # ------------------------------------------------------------------ jam detection (privileged)
    def _jammed(self, rel, xy) -> str | None:
        z = float(rel[2])
        self._z_hist.append(z)
        in_zone = xy < self.zone_xy and z < self.zone_z
        if not in_zone:
            return None
        if z < self.contact_z and xy > self.contact_xy:
            return "misaligned_contact"
        w = self._z_hist[-self.stuck_window:]
        if len(w) == self.stuck_window and (w[0] - min(w)) < self.stuck_progress and z > 0.003:
            return "stuck"
        return None

    # ------------------------------------------------------------------ expert recovery
    def _recover(self, task, reason: str):
        self._n_interventions += 1
        print(f"[dagger] intervention {self._n_interventions} at step {task.take_action_cnt}: {reason}", flush=True)
        original_step = task._step
        counter = {"n": 0}

        def recording_step(*args, **kwargs):
            out = original_step(*args, **kwargs)
            counter["n"] += 1
            if counter["n"] % self.frame_every == 0 and task.plan_success is not False:
                task._update_render()
                obs = self._record(task, task._get_observations(), source=1)
                self._push_tactile_history(obs)
            return out

        task._step = recording_step
        try:
            # the current (jammed) state is the first expert frame
            task._update_render()
            obs = self._record(task, task._get_observations(), source=1)
            self._push_tactile_history(obs)
            task.move(task.atom.move_by_displacement(z=self.lift, xyz_coord="world"), tag="dagger_lift",
                      time_dilation_factor=0.5)
            task._update_insert_reference_poses()
            task.move(task.atom.place_actor(task.prism, target_pose=task.opening_pose, pre_dis=0.01, dis=0.0,
                                            is_open=False), tag="dagger_realign", time_dilation_factor=0.5)
            depth = max(0.0, float(task.prism.get_pose().p[2] - task.target_pose.p[2]))
            task.move(task.atom.move_by_displacement(z=-depth, xyz_coord="world"), tag="dagger_insert",
                      time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
            task.delay(20)
        finally:
            task._step = original_step
        if task.plan_success is not False and task.check_success():
            task.eval_success = True
        # robot state jumped: drop chunk predictions made before the intervention
        self._action_chunk, self._chunk_step = None, 0
        self._action_predictions.clear()
        self._z_hist.clear()
        print(f"[dagger] intervention done, success={task.eval_success}, plan_success={task.plan_success}", flush=True)

    def _push_tactile_history(self, obs):
        if self.tactile_history_frames > 0:
            for side in ("left", "right"):
                self._tactile_frames[side].append(obs[f"observation/{side}_tactile_image"])

    # ------------------------------------------------------------------ policy step
    def eval(self, task, observation):
        rel, xy, _ = self._diag(task)
        reason = self._jammed(rel, xy)
        if reason is not None and self._n_interventions < self.max_interventions and not task.eval_success:
            self._recover(task, reason)
            self._meta["success"] = bool(task.eval_success)
            return True, task.eval_success
        self._record(task, observation, source=0)
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
