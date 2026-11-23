"""insert_USB DAgger plugin: a jammed insertion -> lift, re-align over the slot opening, insert.

Diagnostics (privileged): rel_xyz = USB pose in the inserted-target frame (z = 0 inserted, ~0.0124 at the opening),
xy_err, tilt_deg.
Trigger: inside the insertion zone, (a) the plug reached the opening misaligned, or (b) no downward progress for
`stuck_window` policy steps.
Event rule (tactile label): at the opening (min_z < z < contact_z, within zone_xy) and misaligned or tilted.
Goal: plug inserted (z < min_z and xy < inserted_xy).
"""
from __future__ import annotations

import numpy as np

PLUG_LEN = 0.0124  # USB plug length: rel z at the slot opening


class Expert:
    def __init__(self, cfg: dict):
        self.stuck_window = int(cfg.get("stuck_window", 15))
        self.stuck_progress = float(cfg.get("stuck_progress", 0.0005))
        self.zone_xy = float(cfg.get("zone_xy", 0.008))
        self.zone_z = float(cfg.get("zone_z", 0.025))
        self.contact_xy = float(cfg.get("contact_xy", 0.0012))
        self.contact_z = float(cfg.get("contact_z", PLUG_LEN + 0.0015))
        self.lift = float(cfg.get("lift", 0.010))
        self.min_z = float(cfg.get("min_z", 0.003))
        self.tilt_deg = float(cfg.get("event_tilt_deg", 8.0))
        self.inserted_xy = float(cfg.get("inserted_xy", 0.002))
        self._z_hist: list[float] = []

    def reset(self):
        self._z_hist = []

    def diag(self, task) -> dict:
        rel = task.prism.get_pose().rebase(task._usb_pose_in_slot(task.slot.get_pose()))
        p = np.asarray(rel.p, np.float32)
        z_axis = rel.to_transformation_matrix()[:3, 2]
        tilt = float(np.degrees(np.arccos(np.clip(float(z_axis[2]), -1.0, 1.0))))
        return {"rel_xyz": p, "xy_err": float(np.linalg.norm(p[:2])), "tilt_deg": tilt}

    def event_rule(self, d) -> bool:
        z, xy = float(d["rel_xyz"][2]), d["xy_err"]
        return xy < self.zone_xy and self.min_z < z < self.contact_z and (xy > self.contact_xy or d["tilt_deg"] > self.tilt_deg)

    def goal_reached(self, d) -> bool:
        return float(d["rel_xyz"][2]) < self.min_z and d["xy_err"] < self.inserted_xy

    def trigger(self, task, d) -> str | None:
        z, xy = float(d["rel_xyz"][2]), d["xy_err"]
        self._z_hist.append(z)
        if not (xy < self.zone_xy and z < self.zone_z):
            return None
        if z < self.contact_z and xy > self.contact_xy:
            return "misaligned_contact"
        w = self._z_hist[-self.stuck_window:]
        if len(w) == self.stuck_window and (w[0] - min(w)) < self.stuck_progress and z > self.min_z:
            return "stuck"
        return None

    def recover(self, task, reason: str):
        task.move(task.atom.move_by_displacement(z=self.lift, xyz_coord="world"), tag="dagger_lift",
                  time_dilation_factor=0.5)
        task._update_insert_reference_poses()
        task.move(task.atom.place_actor(task.prism, target_pose=task.opening_pose, pre_dis=0.01, dis=0.0,
                                        is_open=False), tag="dagger_realign", time_dilation_factor=0.5)
        depth = max(0.0, float(task.prism.get_pose().p[2] - task.target_pose.p[2]))
        task.move(task.atom.move_by_displacement(z=-depth, xyz_coord="world"), tag="dagger_insert",
                  time_dilation_factor=0.5, constraint_pose=[1, 1, 1, 1, 1, 0])
        task.delay(20)

    def after_recover(self, task):
        self._z_hist = []
