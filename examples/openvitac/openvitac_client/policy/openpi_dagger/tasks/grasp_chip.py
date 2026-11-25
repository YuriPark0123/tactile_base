"""grasp_chip DAgger plugin: careless placement on the plate -> scripted tactile placement.

Base-policy failures (openvtla_grasp_chip, 1/20 on seeds 10000-10019: `impact`) happen when the chip reaches the plate:
the physical scorer breaks the chip if it touches a support while moving down at >= 0.09 m/s (SupportImpactTracker),
e.g. lowered too fast or let go above the plate. The scripted demonstration instead stops at a small clearance above
the plate, watches the tactile markers while it seats the chip at 6 mm/s, holds the touch, releases and withdraws.

Diagnostics (privileged): held (bilateral pad contact), gap (lowest chip vertex above the plate floor, m), over_plate
(chip centre within `plate_xy` of the plate centre), vz (chip centre vertical speed, m/s, + = up), grip_open
(finger qpos above the narrowest grip seen, m), success, fractured.
Triggers (policy phase only):
  * slipping: carrying the lifted chip away from the plate while the fingers open by > slip_open over the narrowest
    grip (r1 seed 20006: the chip slipped out at 9 cm and broke on the table);
  * fast_descent: held over the plate, gap < near_gap and moving down so fast that even braking like the script
    (0.9 m/s^2) would leave > touch_speed at contact;
  * release_high: over the plate with gap > release_gap while the fingers start to open (the chip would drop).
Event (onset label): the state handed to the expert (held near the plate and descending fast / opening high).
Goal: the task's physical scorer reports success (chip released, supported in the plate, intact).
Recovery for slipping: re-grip to the narrowest grip and tighten with the script's staged tactile close, lift and carry over the plate like the script, then place.
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
        self.reset()

    def reset(self):
        self._prev = None  # (step, chip centre z)
        self._grip_q = None

    # ------------------------------------------------------------------ state
    def diag(self, task) -> dict:
        nan = {"held": 0.0, "gap": np.nan, "over_plate": 0.0, "vz": 0.0, "grip_open": 0.0, "success": 0.0,
               "fractured": 0.0}
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
                "fractured": float(bool(task.fractured))}

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

    def event_rule(self, d) -> bool:
        return self._slipping(d) or self._fast_descent(d) or self._release_high(d)

    def goal_reached(self, d) -> bool:
        return bool(d["success"] > 0.5 and d["fractured"] < 0.5)

    def trigger(self, task, d) -> str | None:
        if task.phase_id != task.PHASE_POLICY or d["fractured"] > 0.5 or d["success"] > 0.5:
            return None
        if self._slipping(d):
            return "slipping"
        if self._release_high(d):
            return "release_high"
        if self._fast_descent(d):
            return "fast_descent"
        return None

    # ------------------------------------------------------------------ recovery
    def recover(self, task, reason: str):
        from envs import grasp_chip as gc

        rm = task._robot_manager
        if reason in ("release_high", "slipping") and self._grip_q is not None:
            # close back to the narrowest grip the policy had (no tighter: the chip breaks under strain)
            pct = (self._grip_q - rm.gripper_close_qpos) / (rm.gripper_open_qpos - rm.gripper_close_qpos)
            task.move(task.atom.close_gripper(float(np.clip(pct, 0.0, 1.0))), tag="dagger_regrip_chip", delay=False)
            # then close in small steps until the script's tactile grip criterion holds (the policy's own grip can be
            # too loose: r1 seed 20023 slipped again on the way down after a re-grip to the policy's width)
            task._staged_tactile_close(max_steps=120)
        # the scripted primitives move the chip through its in-hand pose: take it from the current state
        chip = task.active_chip.get_pose()
        task._chip_inhand_pose = chip.rebase(to_coord=task._robot_manager.get_gripper_center_pose())
        task._grasped_chip_pose = chip
        if reason == "slipping":
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
        if not task._move_held_chip_linear(
            [task.tray_target[0], task.tray_target[1],
             chip.p[2] + task.tray.get_pose().p[2] + 0.006 - float(task.active_chip.vertices[:, 2].min()) - 0.0005],
            "dagger_seat_chip_on_plate",
            max_speed=gc.CHIP_PLACEMENT_TOUCH_SPEED_M_S,
            max_acceleration=gc.CHIP_PLACEMENT_TOUCH_ACCELERATION_M_S2,
            stop_on_support_contact=True,
        ):
            return
        if not task._hold_plate_touch():
            return
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
