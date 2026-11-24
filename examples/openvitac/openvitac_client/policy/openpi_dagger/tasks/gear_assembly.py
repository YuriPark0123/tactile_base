"""gear_assembly DAgger plugin: red gear blocked above the shaft (teeth on teeth) -> lift, re-phase, re-insert, turn.

Base-policy failures (openvtla_gear_assembly, 3/20 on seeds 10000-10019): the red gear is lowered onto the shaft but
stops ~4.6 mm above its assembly height (its teeth rest on the blue gear's teeth: the in-hand yaw is ~2 deg off the
mesh phase). The policy starts the 45 deg turn anyway; with the gears half engaged the red gear walks 3-5 mm off-centre
and the blue gear turns < 30 deg.

Diagnostics (privileged): red_rel = red gear position minus its assembly target (m), red_xy = |red_rel[:2]|,
red_yaw = red gear yaw vs the target (deg), blue_yaw = blue gear yaw vs its reset pose (deg).
Trigger: near the shaft (red_xy < zone_xy) and either
  * blocked: block_dz_min < dz < block_dz_max with < stuck_progress of descent over the last stuck_window policy steps,
  * off-centre: down on the shaft (|dz| < seat_dz) but red_xy > off_xy.
Event rule (tactile label): blocked height (block_dz_min < dz < block_dz_max) or off-centre, near the shaft.
Goal: gear seated (dz < seated_dz) and the pair turned (|blue_yaw| >= 30 deg, opposite to red).
Recovery: re-target the red gear's mesh phase to the blue gear's current rotation (the pair turns 1:1 in opposite
directions), then task._insert_red_gear_onto_base() (lift, carry above the base, align phase, lower). If the gear is
still not seated (it is often tilted in the hand after the policy pressed it onto the teeth): release, retreat,
re-grasp with the scripted approach and insert again. Then the scripted 45 deg wrist turn with the task's gear drive,
as in _play_once (the demonstrations).
"""
from __future__ import annotations

import numpy as np


class Expert:
    def __init__(self, cfg: dict):
        self.zone_xy = float(cfg.get("zone_xy", 0.006))
        self.block_dz_min = float(cfg.get("block_dz_min", 0.0025))
        self.block_dz_max = float(cfg.get("block_dz_max", 0.010))
        self.stuck_window = int(cfg.get("stuck_window", 10))
        self.stuck_progress = float(cfg.get("stuck_progress", 0.0003))
        self.seat_dz = float(cfg.get("seat_dz", 0.005))
        self.off_xy = float(cfg.get("off_xy", 0.0025))
        self.seated_dz = float(cfg.get("seated_dz", 0.002))
        self.regrasp_dz = float(cfg.get("regrasp_dz", 0.002))  # settled re-insert above this -> release + re-grasp
        self.settle_steps = int(cfg.get("settle_steps", 30))  # steps to let the gear settle before judging the insert
        self.wiggle_deg = [float(x) for x in cfg.get("wiggle_deg", [2, -2, 4, -4, 6, -6])]
        self.wiggle_press = float(cfg.get("wiggle_press", 0.0005))  # m pressed below the target height
        self.regrasp_lift = float(cfg.get("regrasp_lift", 0.03))
        self._dz_hist: list[float] = []

    def reset(self):
        self._dz_hist = []

    def diag(self, task) -> dict:
        target = getattr(task, "_red_gear_assembly_target_pose", None)
        if target is None:
            return {"red_rel": np.full(3, np.nan, np.float32), "red_xy": np.nan, "red_yaw": np.nan, "blue_yaw": np.nan}
        rel = np.asarray(task.red_gear.get_pose().p - target.p, np.float32)
        _, _, _, red_yaw = task._get_actor_turn_result(task.red_gear, target)
        blue_ref = getattr(task, "_reset_blue_pose", None)
        if blue_ref is None:
            blue_ref = getattr(task, "initial_blue_pose", None)
        _, _, _, blue_yaw = task._get_actor_turn_result(task.blue_gear, blue_ref)
        return {"red_rel": rel, "red_xy": float(np.linalg.norm(rel[:2])), "red_yaw": float(red_yaw),
                "blue_yaw": float(blue_yaw)}

    def _blocked_height(self, d) -> bool:
        dz = float(d["red_rel"][2])
        return bool(np.isfinite(dz) and d["red_xy"] < self.zone_xy and self.block_dz_min < dz < self.block_dz_max)

    def _off_centre(self, d) -> bool:
        dz, xy = float(d["red_rel"][2]), d["red_xy"]
        return bool(np.isfinite(dz) and abs(dz) < self.seat_dz and self.off_xy < xy < 4 * self.zone_xy)

    def event_rule(self, d) -> bool:
        return self._blocked_height(d) or self._off_centre(d)

    def goal_reached(self, d) -> bool:
        red, blue, dz = d["red_yaw"], d["blue_yaw"], float(d["red_rel"][2])
        return bool(np.isfinite(dz) and dz < self.seated_dz and abs(blue) >= 30.0 and red * blue < 0.0)

    def trigger(self, task, d) -> str | None:
        dz = float(d["red_rel"][2])
        if not np.isfinite(dz):
            return None
        self._dz_hist.append(dz)
        if self._off_centre(d):
            return "seated_off_centre"
        w = self._dz_hist[-self.stuck_window:]
        if self._blocked_height(d) and len(w) == self.stuck_window and (w[0] - min(w)) < self.stuck_progress:
            return "blocked_above_shaft"
        return None

    def _dz(self, task) -> float:
        return float(self.diag(task)["red_rel"][2])

    def _wiggle(self, task):
        """Teeth on teeth: holding the gear, turn it to +-2, +-4, +-6 deg (half the 12 deg tooth pitch) from where it
        sits and press it down at each phase until it drops onto the shaft. The blue gear can turn by a few degrees
        while the red one is pushed onto it (r2 seed 20040: 0.6 deg at the takeover, 2.7 deg after the re-insert), so
        the pre-computed phase can be off by more than the mesh tolerance."""
        cur = 0.0
        for target in self.wiggle_deg:
            if not task.plan_success:
                return
            dz = self._dz(task)
            task.move(task.atom.move_by_displacement(rpy=[0.0, 0.0, np.deg2rad(target - cur)], xyz_coord="local"),
                      tag="dagger_wiggle_red_gear", delay=False, time_dilation_factor=0.5)
            cur = target
            press = min(dz, 0.003) + self.wiggle_press
            task.move(task.atom.move_by_displacement(z=-press, xyz_coord="world"),
                      tag="dagger_press_red_gear", delay=False, time_dilation_factor=0.5)
            task.delay(self.settle_steps // 3)
            if self._dz(task) <= self.regrasp_dz:
                return
            # back to the same hand height (the gear is still on the teeth) before turning to the next phase
            task.move(task.atom.move_by_displacement(z=press, xyz_coord="world"),
                      tag="dagger_unpress_red_gear", delay=False, time_dilation_factor=0.5)

    def recover(self, task, reason: str):
        # The pair meshes 1:1 in opposite directions: if the blue gear has turned by b, the red gear's mesh phase is -b.
        d = self.diag(task)
        target = task._red_gear_assembly_target_pose
        if np.isfinite(d["blue_yaw"]) and abs(d["blue_yaw"]) > 0.05:
            task._red_gear_assembly_target_pose = task._rotate_pose_about_world_z(target, np.deg2rad(-d["blue_yaw"]))
        task._insert_red_gear_onto_base()
        # judge the insertion after the gear has settled (it slides down slowly over ~30 recorded frames)
        task.delay(self.settle_steps)
        if self._dz(task) > self.regrasp_dz and task.plan_success:
            self._wiggle(task)
        if self._dz(task) > self.regrasp_dz and task.plan_success:
            # still up: let go -- released, the gear usually slides down onto the teeth phase by itself (r2 seeds
            # 20032/20035: 3.9 mm while held -> -1.4 mm once released) -- then grasp it again where it is. Only if it is
            # still up after that, insert again (the scripted insert first lifts the gear by 3 cm).
            task.move(task.atom.open_gripper(0.5), tag="dagger_release_red_gear", delay=False)
            task.delay(self.settle_steps)
            task.move(task.atom.move_by_displacement(z=self.regrasp_lift, xyz_coord="world"), tag="dagger_retreat",
                      delay=False, time_dilation_factor=0.5)
            task._grasp_red_gear(enable_drive=False)
            if self._dz(task) > self.regrasp_dz and task.plan_success:
                task._insert_red_gear_onto_base()
                task.delay(self.settle_steps)
        task._record_turn_initial_gear_poses()
        task._start_gear_drive()
        task.move(
            task.atom.move_by_displacement(rpy=[0.0, 0.0, np.pi / 4], xyz_coord="local"),
            tag="dagger_turn_red_gear",
            delay=False,
            time_dilation_factor=0.5,
            constraint_pose=[0, 0, 0, 1, 1, 1],
        )
        task._sync_driven_gears_to_gripper()
        task._step(is_save=True)
        task._stop_gear_drive()
        task.delay(10)

    def after_recover(self, task):
        self._dz_hist = []
