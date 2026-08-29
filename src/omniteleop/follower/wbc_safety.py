"""Collision-pair filtering and the reactive whole-body safety gate.

The IK uses a proactive all-sphere collision barrier and a CoM-over-base inequality.
After every candidate step, :class:`SafetyGate` applies a second line of defense: a
step below a safety floor is rejected only when it makes the metric worse. Recovery
motion remains possible instead of permanently trapping the robot at a boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

CROSS_GROUPS = (
    ("body", "left"),
    ("body", "right"),
    ("left", "right"),
)


def collision_group(name: str) -> str | None:
    """Map a collision geometry name to body, left arm, or right arm."""
    if name.startswith(("L_arm", "L_ee", "L_robotiq")):
        return "left"
    if name.startswith(("R_arm", "R_ee", "R_robotiq")):
        return "right"
    if name.startswith(("base", "torso", "head")):
        return "body"
    return None


def cross_group_index_pairs(group_of: list[str | None]) -> list[tuple[int, int]]:
    """Return all inter-group pairs while excluding adjacent same-chain geometry."""
    members: dict[str, list[int]] = {"body": [], "left": [], "right": []}
    for index, group in enumerate(group_of):
        if group in members:
            members[group].append(index)

    pairs: list[tuple[int, int]] = []
    for first_group, second_group in CROSS_GROUPS:
        pairs.extend(
            (first, second) for first in members[first_group] for second in members[second_group]
        )
    return pairs


def filter_nominal_overlaps(
    pairs: list[tuple[int, int]],
    nominal_distance: Callable[[int, int], float],
    keep_dist: float,
) -> list[tuple[int, int]]:
    """Drop pairs that overlap by design at the neutral posture.

    This is the equivalent of disabling always-colliding pairs in an SRDF. Without this
    build-time filter, shoulder spheres inside torso spheres would make the proactive
    constraint infeasible and immediately trip the reactive monitor.
    """
    return [pair for pair in pairs if nominal_distance(*pair) >= keep_dist]


@dataclass
class SafetyStatus:
    """Decision returned by :meth:`SafetyGate.check`."""

    held: bool
    message: str
    hold_base: bool = False


class SafetyGate:
    """Reject worsening candidates below the stability or collision floor.

    A collision hold freezes joints but may keep the base moving because base pose is
    invariant to self-collision distance and can help recovery. A tip-over hold freezes
    the entire robot.
    """

    def __init__(
        self,
        *,
        com_safety_margin: float,
        self_collision_floor: float,
        self_collision_warn: float,
        enable_com: bool = True,
        enable_collision: bool = True,
    ) -> None:
        values = (com_safety_margin, self_collision_floor, self_collision_warn)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"safety thresholds must be finite and non-negative, got {values}")
        self.com_safety_margin = float(com_safety_margin)
        self.self_collision_floor = float(self_collision_floor)
        self.self_collision_warn = float(self_collision_warn)
        self.enable_com = bool(enable_com)
        self.enable_collision = bool(enable_collision)
        self._prev_margin = np.inf
        self._prev_dist = np.inf

    def reset(self, margin: float, self_dist: float) -> None:
        """Anchor the accepted-step baselines after a solver reset."""
        self._prev_margin = float(margin)
        self._prev_dist = float(self_dist)

    def check(self, margin: float, self_dist: float) -> SafetyStatus:
        """Evaluate a candidate stability margin and minimum collision distance."""
        warnings: list[str] = []
        held = False
        hold_base = False

        if self.enable_com and margin < self.com_safety_margin:
            warnings.append(
                f"CoM margin {margin * 100:+.1f}cm < {self.com_safety_margin * 100:.0f}cm"
            )
            if margin < self._prev_margin:
                held = True
                hold_base = True

        if self.enable_collision and self_dist < self.self_collision_warn:
            warnings.append(f"self-collision {self_dist * 100:+.1f}cm")
            if self_dist < self.self_collision_floor and self_dist < self._prev_dist:
                held = True

        if not held:
            self._prev_margin = float(margin)
            self._prev_dist = float(self_dist)

        if not warnings:
            return SafetyStatus(False, "ok")
        prefix = "HELD" if held else "WARN"
        return SafetyStatus(held, f"{prefix}: " + "; ".join(warnings), hold_base)
