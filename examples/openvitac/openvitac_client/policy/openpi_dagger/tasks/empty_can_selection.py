"""empty_can_selection DAgger plugin: wrong weight judgement -> scripted correction.

Base-policy failures (openvtla_empty_can_selection, 4/14 on seeds 10000-10013) are all wrong decisions about which can
is empty (light): the policy lifts the empty can, judges it heavy, puts it back and carries the heavy can to the basket
(3x), or carries the heavy can straight to the basket (1x). The cans look identical; only touch (load on the gel when
lifted) tells them apart.

Diagnostics (privileged): held = index of the lifted can (0 fanta, 1 pepsi, -1 none), held_empty, held_dz (height
above its work pose, m), held_dxy (horizontal distance from its work pose, m), empty_in_basket.
Triggers:
  * carrying_heavy: the heavy can is lifted (> lift_dz) and moved > carry_dxy away from its work pose;
  * returning_empty: the empty can, after being lifted >= test_dz, is lowered back (< return_dz) near its work pose.
Event (tactile label, dense): while a can is lifted, 1 = it is the empty can, 0 = the heavy can; unknown (-1) when no
can is lifted (`event_valid` = 0). Convert with `--event-mode dense --event-name empty`.
Goal: the empty can is in the basket.
Recovery (the task's scripted primitives, as in _play_once): carrying_heavy -> return the heavy can to its work pose,
grasp the empty can, lift, drop it in the basket; returning_empty -> lift it, drop it in the basket.
"""
from __future__ import annotations

import numpy as np

BRANDS = ("fanta", "pepsi")


class Expert:
    def __init__(self, cfg: dict):
        self.lift_dz = float(cfg.get("lift_dz", 0.01))
        self.carry_dxy = float(cfg.get("carry_dxy", 0.03))
        self.test_dz = float(cfg.get("test_dz", 0.025))
        self.return_dz = float(cfg.get("return_dz", 0.012))
        self.near_dxy = float(cfg.get("near_dxy", 0.02))
        self._max_dz = {b: 0.0 for b in BRANDS}

    def reset(self):
        self._max_dz = {b: 0.0 for b in BRANDS}

    def diag(self, task) -> dict:
        if not hasattr(task, "active_cans"):
            return {"held": -1, "held_empty": 0.0, "held_dz": 0.0, "held_dxy": 0.0, "empty_in_basket": 0.0,
                    "can_dz": np.zeros(2, np.float32)}
        dz, dxy = [], []
        for b in BRANDS:
            p = task.active_cans[b].get_pose().p
            w = task.active_work_poses[b].p
            dz.append(float(p[2] - w[2]))
            dxy.append(float(np.linalg.norm(p[:2] - w[:2])))
        held = int(np.argmax(dz)) if max(dz) > self.lift_dz else -1
        local = task.empty_can_actor.get_pose().rebase(task.basket.get_pose()).p
        in_basket = abs(local[0]) <= 0.06108 and abs(local[1]) <= 0.06108 and 0.0 <= local[2] <= 0.05
        return {
            "held": held,
            "held_empty": float(held >= 0 and BRANDS[held] == task.empty_can_brand),
            "held_dz": dz[held] if held >= 0 else 0.0,
            "held_dxy": dxy[held] if held >= 0 else 0.0,
            "empty_in_basket": float(in_basket),
            "can_dz": np.asarray(dz, np.float32),
        }

    def event_rule(self, d) -> bool:  # dense label value (valid only where event_valid)
        return bool(d["held"] >= 0 and d["held_empty"] > 0.5)

    def event_valid(self, d) -> bool:
        return bool(d["held"] >= 0)

    def goal_reached(self, d) -> bool:
        return bool(d["empty_in_basket"] > 0.5)

    def trigger(self, task, d) -> str | None:
        for i, b in enumerate(BRANDS):
            self._max_dz[b] = max(self._max_dz[b], float(d["can_dz"][i]))
        h = d["held"]
        if h < 0:
            # the empty can may already be (almost) back on the table: check it explicitly
            e = BRANDS.index(task.empty_can_brand)
            dz_e = float(d["can_dz"][e])
            if self._max_dz[task.empty_can_brand] >= self.test_dz and dz_e < self.return_dz and dz_e > 0.003:
                return "returning_empty"
            return None
        if d["held_empty"] < 0.5 and d["held_dz"] > self.lift_dz and d["held_dxy"] > self.carry_dxy:
            return "carrying_heavy"
        if (d["held_empty"] > 0.5 and self._max_dz[BRANDS[h]] >= self.test_dz and d["held_dz"] < self.return_dz
                and d["held_dxy"] < self.near_dxy):
            return "returning_empty"
        return None

    def recover(self, task, reason: str):
        from envs.empty_can_selection import EMPTY_CAN_LIFT_HEIGHT

        empty = task.empty_can_brand
        heavy = [b for b in BRANDS if b != empty][0]
        if reason == "carrying_heavy":
            task._return_heavy_can(heavy)
            task._grasp_can(empty)
            task._lift_current_can(empty, EMPTY_CAN_LIFT_HEIGHT)
        else:  # returning_empty: still in hand, lift it back up
            dz = float(task.active_cans[empty].get_pose().p[2] - task.active_work_poses[empty].p[2])
            task._lift_current_can(empty, max(0.0, EMPTY_CAN_LIFT_HEIGHT - dz))
        task._drop_empty_can_in_basket(empty)

    def after_recover(self, task):
        pass
