"""grasp_chip DAgger plugin: careless placement on the plate -> scripted tactile placement.

Base-policy failures (openvtla_grasp_chip, 1/20 on seeds 10000-10019: `impact`) happen when the chip reaches the plate:
the physical scorer breaks the chip if it touches a support while moving down at >= 0.09 m/s (SupportImpactTracker),
e.g. lowered too fast or let go above the plate. The scripted demonstration instead stops at a small clearance above
the plate, watches the tactile markers while it seats the chip at 6 mm/s, holds the touch, releases and withdraws.

Diagnostics (privileged): held (bilateral pad contact), gap (lowest chip vertex above the plate floor, m), over_plate
(chip centre within `plate_xy` of the plate centre), vz (chip centre vertical speed, m/s, + = up), grip_open
(finger qpos above the narrowest grip seen, m), success, fractured.
Triggers (policy phase only):
  * grasp_stalled: stall_steps policy steps without having lifted the chip (r1/r2 seeds 20025, 20053, 20055 ended in
    policy_timeout with the chip never grasped; normal episodes lift it at step 131-244) -> the script's grasp;
  * over_squeeze: held with the chip strain >= squeeze_ratio of its fracture strain (eval seed 10020 broke squeezing);
  * slipping: carrying the lifted chip away from the plate while the fingers open by > slip_open over the narrowest
    grip (r1 seed 20006: the chip slipped out at 9 cm and broke on the table);
  * fast_descent: held over the plate, gap < near_gap and moving down so fast that even braking like the script
    (0.9 m/s^2) would leave > touch_speed at contact;
  * release_high: over the plate with gap > release_gap while the fingers start to open (the chip would drop).
Event (onset label): the state handed to the expert (held near the plate and descending fast / opening high).
Goal: the task's physical scorer reports success (chip released, supported in the plate, intact).
Recovery for over_squeeze: ease the grip by 0.2 mm (lifted) / 0.4 mm, close again with the script's staged tactile
close (stops at the contact depth), then lift / carry / place as below.
Perturbation (perturb_prob > 0, r2): unrecorded loosening (fingers +0.2-0.3 mm) or tightening (until the strain reaches
75-90 % of the fracture strain) at a random carry step; the takeover reason gets the suffix `_perturbed`.
Recovery for slipping: re-grip to the narrowest grip - 0.05 mm (direct finger target), lift and carry over the plate like the script, then place.
Recovery (the task's scripted placement, `_play_once` from "lower_intact_chip_into_tray" on): re-grip if the fingers
had opened, move to the approach clearance above the plate slowly, baseline the plate-touch markers, seat the chip
until the support touch, hold, release, withdraw.
"""
from __future__ import annotations

import numpy as np


class Expert:
    def __init__(self, cfg: dict):
        self.near_gap = float(cfg.get("near_gap", 0.025))
        self.decel = float(cfg.get("decel", 0.9))  # m/s^2, the scripted transport's deceleration
        self.touch_speed = float(cfg.get("touch_speed", 0.05))  # m/s predicted at contact -> takeover (breaks at 0.09)
        self.release_gap = float(cfg.get("release_gap", 0.004))
        self.open_margin = float(cfg.get("open_margin", 0.0003))
        self.plate_xy = float(cfg.get("plate_xy", 0.08))
        self.lift_gap = float(cfg.get("lift_gap", 0.01))  # m: chip lifted (above the plate floor level)
        self.slip_open = float(cfg.get("slip_open", 0.00015))  # m: the scorer's own "opening" margin
        # perturbation (more corrective data; the base rarely fails on its own): with probability perturb_prob per
        # episode, open the fingers by a random perturb_open [m] at a random carry step in perturb_steps, unrecorded,
        # so that the slipping takeover starts from the slip onset the base itself produces (r1 seed 20006)
        self.squeeze_ratio = float(cfg.get("squeeze_ratio", 0.7))  # strain / fracture strain -> takeover
        # policy steps without having lifted the chip -> takeover (normal episodes lift it at step 131-244)
        self.stall_steps = int(cfg.get("stall_steps", 400))
        self.perturb_prob = float(cfg.get("perturb_prob", 0.0))
        self.perturb_tighten_frac = float(cfg.get("perturb_tighten_frac", 0.5))  # share of perturbations that squeeze
        self.perturb_squeeze = tuple(float(x) for x in cfg.get("perturb_squeeze", (0.75, 0.9)))
        self.perturb_open = tuple(float(x) for x in cfg.get("perturb_open", (0.0002, 0.0003)))
        self.perturb_steps = tuple(int(x) for x in cfg.get("perturb_steps", (5, 30)))
        self.reset()

    def reset(self):
        self._prev = None  # (step, chip centre z)
        self._grip_q = None
        self._carry_steps = 0
        self._perturb_at = None  # carry step at which to perturb (None: not this episode)
        self._perturb_seed = None
        self.perturbed = False
        self._no_slip = False
        self._gap0 = None
        self._ever_lifted = False

    # ------------------------------------------------------------------ state
    def diag(self, task) -> dict:
        nan = {"held": 0.0, "gap": np.nan, "over_plate": 0.0, "vz": 0.0, "grip_open": 0.0, "success": 0.0,
               "fractured": 0.0, "strain_ratio": 0.0}
        if not hasattr(task, "active_chip") or not hasattr(task, "tray"):
            return nan
        vertices = task.active_chip.vertices
        tray = task.tray.get_pose()
        local = (vertices - tray.p) @ tray.R
        gap = float(local[:, 2].min() - 0.006)
        centre = vertices.mean(axis=0)
        over = float(np.linalg.norm(centre[:2] - tray.p[:2]) < self.plate_xy)
        step = int(task.step_count)
        vz = 0.0
        if self._prev is not None and step > self._prev[0]:
            dt = (step - self._prev[0]) * float(task.cfg.sim.dt * task.cfg.decimation)
            vz = float((centre[2] - self._prev[1]) / dt)
        self._prev = (step, float(centre[2]))
        depths = task._read_tactile_depth()
        held = bool(depths is not None and np.all(depths < 31.0))
        q = float(task._robot_manager.get_gripper_qpos())
        if held:
            self._grip_q = q if self._grip_q is None else min(self._grip_q, q)
        grip_open = 0.0 if self._grip_q is None else max(0.0, q - self._grip_q)
        monitor = getattr(task, "_action_monitor", None)
        return {"held": float(held), "gap": gap, "over_plate": over, "vz": vz, "grip_open": grip_open,
                "success": float(bool(monitor is not None and monitor.scorer.success)),
                "fractured": float(bool(task.fractured)), "strain_ratio": self._strain_ratio(task)}

    @staticmethod
    def _strain_ratio(task) -> float:
        """Chip strain / its fracture strain (privileged; the scorer breaks the chip at 1)."""
        try:
            strain = task._chip_strain()
            return float(strain / task.episode.fracture_strain) if strain is not None else 0.0
        except (AttributeError, TypeError, ZeroDivisionError):
            return 0.0

    def _fast_descent(self, d) -> bool:
        if not (d["held"] > 0.5 and d["over_plate"] > 0.5 and np.isfinite(d["gap"]) and d["gap"] < self.near_gap
                and d["vz"] < 0):
            return False
        # speed left at contact even when braking like the script from now on (the script itself comes down at
        # 0.16 m/s and brakes to a stop 1.5 mm above the plate, so plain speed is no criterion)
        touch = np.sqrt(max(0.0, d["vz"] ** 2 - 2 * self.decel * max(0.0, d["gap"])))
        return bool(touch > self.touch_speed)

    def _release_high(self, d) -> bool:
        return bool(d["over_plate"] > 0.5 and np.isfinite(d["gap"]) and d["gap"] > self.release_gap
                    and d["grip_open"] > self.open_margin and self._grip_q is not None)

    def _slipping(self, d) -> bool:
        # carrying the lifted chip (away from the plate) while the fingers open: it slips out and falls (r1 seed 20006:
        # fingers +0.2 mm, contact lost 15 frames later at 9 cm, impact on the table)
        return bool(d["held"] > 0.5 and np.isfinite(d["gap"]) and d["gap"] > self.lift_gap and d["over_plate"] < 0.5
                    and d["grip_open"] > self.slip_open)

    def _over_squeeze(self, d) -> bool:
        # gripping so hard that the chip is close to cracking (r1 eval seed 10020 broke at 100.2 %; successes reach 74 %)
        return bool(d["held"] > 0.5 and d["strain_ratio"] >= self.squeeze_ratio)

    def event_rule(self, d) -> bool:
        return self._slipping(d) or self._over_squeeze(d) or self._fast_descent(d) or self._release_high(d)

    def goal_reached(self, d) -> bool:
        return bool(d["success"] > 0.5 and d["fractured"] < 0.5)

    def _maybe_perturb(self, task, d) -> bool:
        """Open the fingers a little while the policy carries the chip (unrecorded). Returns True if it did."""
        if self.perturb_prob <= 0 or self.perturbed:
            return False
        seed = int(task.cfg.seed)
        if self._perturb_seed != seed:
            rng = np.random.default_rng(seed + 7919)
            self._perturb_seed = seed
            self._perturb_at = int(rng.integers(*self.perturb_steps)) if rng.random() < self.perturb_prob else None
            self._perturb_delta = float(rng.uniform(*self.perturb_open))
            self._perturb_mode = "tighten" if rng.random() < self.perturb_tighten_frac else "loosen"
            self._perturb_squeeze_target = float(rng.uniform(*self.perturb_squeeze))
        carrying = (d["held"] > 0.5 and np.isfinite(d["gap"]) and d["gap"] > self.lift_gap and d["over_plate"] < 0.5)
        self._carry_steps = self._carry_steps + 1 if carrying else 0
        if self._perturb_at is None or self._carry_steps < self._perturb_at or self._grip_q is None:
            return False
        import torch

        rm = task._robot_manager
        dims = len(rm._gripper_ids)
        start = float(rm.get_gripper_qpos())
        if self._perturb_mode == "tighten":
            # squeeze in 0.05 mm steps until the strain reaches the drawn fraction of the fracture strain (max 3 mm; the
            # gel pads take up most of it: +1.5 mm reached only 43 % in r2 seed 20036)
            q = start
            for _ in range(60):
                if self._strain_ratio(task) >= self._perturb_squeeze_target:
                    break
                q -= 0.00005
                pos = torch.full((dims,), q, device=rm.device)
                rm.set_gripper(pos, torch.zeros_like(pos), force=False)
                for _ in range(2):
                    task._step(is_save=True)
            for _ in range(10):  # let the fingers settle on the squeezed chip before taking it as the grip reference
                task._step(is_save=True)
            info = {"mode": "tighten", "closed_m": start - q, "strain_ratio": self._strain_ratio(task)}
        else:
            for k in range(1, 7):  # open over 6 sim steps (a slow loosening, as in the base's own slips)
                pos = torch.full((dims,), start + self._perturb_delta * k / 6, device=rm.device)
                rm.set_gripper(pos, torch.zeros_like(pos), force=False)
                task._step(is_save=True)
            info = {"mode": "loosen", "open_m": self._perturb_delta}
        info["carry_step"] = self._carry_steps
        # the narrowest grip is the reference for "opening": restart it from the perturbed grip (a transient dip
        # while squeezing must not read as the fingers opening afterwards)
        if self._perturb_mode == "tighten":
            self._grip_q = float(rm.get_gripper_qpos())
            # squeezing that did not reach the target is no takeover state; do not let its after-effects read as a slip
            self._no_slip = self._strain_ratio(task) < self.squeeze_ratio
            info["reached"] = not self._no_slip
        task.metadata["dagger_perturbation"] = info
        print(f"[dagger] perturbation: {info}", flush=True)
        self.perturbed = True
        return True

    def trigger(self, task, d) -> str | None:
        if task.phase_id != task.PHASE_POLICY or d["fractured"] > 0.5 or d["success"] > 0.5:
            return None
        if self._gap0 is None and np.isfinite(d["gap"]):
            self._gap0 = d["gap"]  # chip resting where it was spilled
        if d["held"] > 0.5 and self._gap0 is not None and d["gap"] - self._gap0 > self.lift_gap:
            self._ever_lifted = True
        if not self._ever_lifted and task.take_action_cnt >= self.stall_steps:
            return "grasp_stalled"
        if self._maybe_perturb(task, d):
            d.update(self.diag(task))
        tag = "_perturbed" if self.perturbed else ""
        if self._over_squeeze(d):
            return "over_squeeze" + tag
        if self._slipping(d) and not self._no_slip:
            return "slipping" + tag
        if self._release_high(d):
            return "release_high"
        if self._fast_descent(d):
            return "fast_descent"
        return None

    # ------------------------------------------------------------------ recovery
    def recover(self, task, reason: str):
        from envs import grasp_chip as gc

        rm = task._robot_manager
        if reason == "grasp_stalled":
            # start over with the script's grasp: open, back off upwards, approach and close with the tactile criterion
            task.move(task.atom.open_gripper(1.0), tag="dagger_open_for_regrasp", delay=False)
            task.move(task.atom.move_by_displacement(z=0.05, xyz_coord="world"), tag="dagger_back_off",
                      delay=False, time_dilation_factor=0.5)
            if not task._grasp_chip():
                return
        if reason.startswith("over_squeeze"):
            # ease off until the pads barely hold (a little wider than now), then close again with the script's tactile
            # criterion, which stops at the contact depth instead of squeezing on
            q = float(rm.get_gripper_qpos())
            d0 = self.diag(task)
            ease = 0.0002 if (np.isfinite(d0["gap"]) and d0["gap"] > self.lift_gap) else 0.0004  # lifted: don't drop it
            pct = (q + ease - rm.gripper_close_qpos) / (rm.gripper_open_qpos - rm.gripper_close_qpos)
            task.move(task.atom.open_gripper(float(np.clip(pct, 0.0, 1.0))), tag="dagger_ease_grip", delay=False)
            task._staged_tactile_close(max_steps=120)
        elif reason.split("_perturbed")[0] in ("release_high", "slipping") and self._grip_q is not None:
            # close back to the narrowest grip the policy had (it carried the chip with it) plus 0.05 mm, by setting the
            # finger target directly: the scripted gripper action and the staged tactile close both stop as soon as
            # the tactile depth criterion is met, which a loosened grip still meets (r2 seed 20056: no change, the
            # chip kept slipping and fell)
            import torch

            dims = len(rm._gripper_ids)
            start = float(rm.get_gripper_qpos())
            target = self._grip_q - 0.00005
            for k in range(1, 9):
                pos = torch.full((dims,), start + (target - start) * k / 8, device=rm.device)
                rm.set_gripper(pos, torch.zeros_like(pos), force=False)
                task._step(is_save=True)
            for _ in range(6):
                task._step(is_save=True)
        # the scripted primitives move the chip through its in-hand pose: take it from the current state
        chip = task.active_chip.get_pose()
        task._chip_inhand_pose = chip.rebase(to_coord=task._robot_manager.get_gripper_center_pose())
        task._grasped_chip_pose = chip
        if (reason.startswith("slipping") or reason == "grasp_stalled"
                or (reason.startswith("over_squeeze") and self.diag(task)["over_plate"] < 0.5)):
            # continue the scripted transport: lift, carry over the plate
            chip = task.active_chip.get_pose()
            task._chip_inhand_pose = chip.rebase(to_coord=rm.get_gripper_center_pose())
            task._grasped_chip_pose = chip
            if not task._move_held_chip_linear([chip.p[0], chip.p[1], max(gc.CHIP_LIFT_Z, chip.p[2])],
                                               "dagger_lift_chip", max_speed=gc.CHIP_LIFT_SPEED_M_S,
                                               allow_grip_maintenance=True):
                return
            if not task._move_held_chip_linear([task.tray_target[0], task.tray_target[1],
                                                max(gc.CHIP_LIFT_Z, task.active_chip.get_pose().p[2])],
                                               "dagger_carry_chip_over_tray", max_speed=gc.CHIP_CARRY_SPEED_M_S,
                                               allow_grip_maintenance=True):
                return
            chip = task.active_chip.get_pose()
        # same height as the script's approach: `chip_placement_approach_clearance_m` above the plate floor (up or
        # down from where the policy left it)
        gap = float(task.active_chip.vertices[:, 2].min() - (task.tray.get_pose().p[2] + 0.006))
        approach_z = float(chip.p[2] + task.chip_placement_approach_clearance_m - gap)
        if not task._move_held_chip_linear([task.tray_target[0], task.tray_target[1], approach_z],
                                           "dagger_approach_plate", max_speed=gc.CHIP_LOWER_SPEED_M_S * 0.25):
            return
        task.monitor_chip_contact = True
        task._placement_marker_reference = task._read_placement_markers()
        task._chip_tactile_controller.begin_placement(task._placement_marker_reference)
        for _ in range(task._chip_tactile_controller.limits.baseline_ticks):
            task._step(is_save=True)
            task._chip_tactile_controller.baseline(task._read_placement_markers())
        task._placement_marker_detection_thresholds = task._chip_tactile_controller.thresholds
        task._save_plate_touch_observation()
        chip = task.active_chip.get_pose()
        seated = task._move_held_chip_linear(
            [task.tray_target[0], task.tray_target[1],
             chip.p[2] + task.tray.get_pose().p[2] + 0.006 - float(task.active_chip.vertices[:, 2].min()) - 0.0005],
            "dagger_seat_chip_on_plate",
            max_speed=gc.CHIP_PLACEMENT_TOUCH_SPEED_M_S,
            max_acceleration=gc.CHIP_PLACEMENT_TOUCH_ACCELERATION_M_S2,
            stop_on_support_contact=True,
        )
        held_ok = seated and task._hold_plate_touch()
        if not held_ok:
            # the script aborts here; the expert still lets go if the chip is resting on the plate (r2 seed 20034: the
            # hold check failed on a chip that had been squeezed, and the chip stayed in the hand until the timeout)
            d = self.diag(task)
            if task.fractured or not (d["over_plate"] > 0.5 and np.isfinite(d["gap"]) and d["gap"] < 0.002):
                return
            task.metadata["dagger_release_after_failed_hold"] = True
            task.plan_success = True
        task.monitor_chip_contact = False
        if not task._release_chip():
            return
        task.delay(5, is_save=True)
        task.active_chip.remove_animate(force=True)
        task._actor_manager.update(dt=0.0)
        task.delay(36, is_save=True)
        current = task._robot_manager.get_gripper_center_pose()
        withdraw = type(current)(current.p + [0.0, 0.0, 0.070], [0, 1, 0, 0])
        task._move_chip_approach_linear(withdraw, tag="dagger_withdraw_after_placement")
        task.delay(24, is_save=True)

    def after_recover(self, task):
        self._prev = None
