"""Integrate joystick torso height along a vertical crouch path.

The torso remains pinned in arm IK while the operator sets height; hand targets
shift by the same amount. TorsoCrouchPath keeps torso_l3 at its nominal pitch and
base-frame x, stopping at joint limits or excessive joint-rate demand.

Each height step is gated against arm/body/mount clearance: unsafe approaches
are refused while recovery remains possible. Holds freeze the height command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pinocchio as pin

from omniteleop.follower.whole_body_ik import TORSO_JOINTS

TORSO_TOP_LINK = "torso_l3"
# Height spacing of the precomputed path (m); linear interpolation between knots is
# ~1e-6 rad off the exact solution.
PATH_STEP = 1e-3
# Largest joint rate per metre of height the path may demand (rad/m). Reached at ~+0.10 m,
# where the default 5 cm/s needs 0.5 rad/s -- under the torso's 0.81 rad/s IK velocity limit.
MAX_JOINT_RATE = 10.0
# Clearance kept from the torso joint limits at the ends of the path (rad).
LIMIT_MARGIN = 0.01
# A pair counts as worsened only past this (m): pairs the torso cannot affect (arm vs arm)
# must not block it through FK round-off.
_WORSEN_TOL = 1e-6


class TorsoCrouchPath:
    """Torso joints for a vertical ``torso_l3`` translation of ``height`` metres."""

    def __init__(self, model: pin.Model, nominal_q: np.ndarray) -> None:
        self._model = model
        self._data = model.createData()
        self._fid = model.getFrameId(TORSO_TOP_LINK)
        joint_ids = [model.getJointId(name) for name in TORSO_JOINTS]
        self.idx_q = [model.idx_qs[j] for j in joint_ids]
        self._idx_v = [model.idx_vs[j] for j in joint_ids]
        q0 = np.asarray(nominal_q, dtype=float).copy()
        top = self._top_pose(q0)
        self._p0, self._r0 = top.translation.copy(), top.rotation.copy()
        lo = model.lowerPositionLimit[self.idx_q] + LIMIT_MARGIN
        hi = model.upperPositionLimit[self.idx_q] - LIMIT_MARGIN
        knots = {0.0: q0[self.idx_q].copy()}
        for sign in (-1.0, 1.0):
            prev, k = q0, 1
            while True:
                height = sign * k * PATH_STEP
                q = self._solve(height, prev)
                if q is None:
                    break
                joints = q[self.idx_q]
                rate = np.max(np.abs(joints - prev[self.idx_q])) / PATH_STEP
                if np.any(joints < lo) or np.any(joints > hi) or rate > MAX_JOINT_RATE:
                    break
                knots[height] = joints.copy()
                prev, k = q, k + 1
        self.heights = np.array(sorted(knots))
        self.joints_table = np.array([knots[h] for h in self.heights])
        self.min_height = float(self.heights[0])
        self.max_height = float(self.heights[-1])
        self.max_joint_rate = float(np.max(np.abs(np.diff(self.joints_table, axis=0))) / PATH_STEP)

    def _top_pose(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacement(self._model, self._data, self._fid)
        return self._data.oMf[self._fid]

    def _solve(self, height: float, seed: np.ndarray) -> Optional[np.ndarray]:
        """Newton on torso_l3's (x, z, pitch) from ``seed``; None if it does not converge."""
        q = seed.copy()
        for _ in range(50):
            top = self._top_pose(q)
            err = np.array(
                [
                    top.translation[0] - self._p0[0],
                    top.translation[2] - (self._p0[2] + height),
                    pin.log3(top.rotation @ self._r0.T)[1],
                ]
            )
            if np.max(np.abs(err)) < 1e-10:
                return q
            jac = pin.computeFrameJacobian(
                self._model, self._data, q, self._fid, pin.LOCAL_WORLD_ALIGNED
            )[[0, 2, 4]][:, self._idx_v]
            q[self.idx_q] -= np.linalg.solve(jac, err)
        return None

    def joints(self, height: float) -> np.ndarray:
        """Torso joints ``(torso_j1, torso_j2, torso_j3)`` at ``height``, clamped to the path."""
        h = float(np.clip(height, self.min_height, self.max_height))
        return np.array(
            [np.interp(h, self.heights, self.joints_table[:, i]) for i in range(len(self.idx_q))]
        )


@dataclass
class JoystickTorsoCommand:
    """The torso height in force, advanced once per tick (see the module docstring)."""

    path: TorsoCrouchPath
    speed: float  # fixed height speed (m/s) for any active stick direction
    gate_distance: float  # self_collision_safe_dist: no pair may be pushed below it (m)

    height: float = 0.0  # torso_l3 offset from nominal (m) -- the command in force
    intent: float = 0.0  # effective intent of the last step (m/s, up +)
    blocked: bool = False  # the last step was refused by the collision gate

    def __post_init__(self) -> None:
        if not np.isfinite(self.speed) or self.speed <= 0.0:
            raise ValueError(f"torso speed must be finite and > 0, got {self.speed}")
        if not np.isfinite(self.gate_distance) or self.gate_distance < 0.0:
            raise ValueError(f"gate_distance must be finite and >= 0, got {self.gate_distance}")

    @classmethod
    def from_ik(cls, ik, speed: float) -> "JoystickTorsoCommand":
        """Build for a solver with the torso pinned (``lock_torso_in_ik``) and collision on."""
        if not ik.config.lock_torso_in_ik:
            raise ValueError("the torso command needs lock_torso_in_ik (the QP must not move it)")
        if not ik.collision_enabled:
            raise ValueError("the torso command needs self-collision avoidance for its gate")
        path = TorsoCrouchPath(ik.model, ik.nominal_q())
        limit = (
            float(
                np.min(
                    ik.model.velocityLimit[
                        [ik.model.idx_vs[ik.model.getJointId(n)] for n in TORSO_JOINTS]
                    ]
                )
            )
            * ik.config.velocity_limit_scale
        )
        if speed * path.max_joint_rate > limit:
            raise ValueError(
                f"torso speed {speed:g} m/s needs up to {speed * path.max_joint_rate:.2f} "
                f"rad/s on the crouch path; the torso limit is {limit:.2f} rad/s"
            )
        return cls(path=path, speed=float(speed), gate_distance=ik.config.self_collision_safe_dist)

    def reset(self) -> None:
        """Engage: back to the nominal torso."""
        self.height, self.intent, self.blocked = 0.0, 0.0, False

    def joints(self) -> np.ndarray:
        """Torso joints for the height in force."""
        return self.path.joints(self.height)

    def hold(self) -> None:
        """Any hold (e-stop, stale stream, failed or held solve): the height stays put."""
        self.intent, self.blocked = 0.0, False

    def step(
        self,
        intent: float,
        dt: float,
        q: np.ndarray,
        pair_distances: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Advance by the direction of ``intent`` at the fixed speed, if the gate allows.

        ``q`` is the solver configuration the next solve starts from (its torso is
        ``joints()``); ``pair_distances`` is ``ik.self_collision_pair_distances``.
        """
        self.intent = float(np.sign(intent)) * self.speed if abs(intent) > 1e-9 else 0.0
        self._move_to(self.height + self.intent * dt, q, pair_distances)

    def track(
        self,
        target: float,
        dt: float,
        q: np.ndarray,
        pair_distances: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Replay: move toward a recorded height at no more than ``speed``, gated."""
        delta = float(np.clip(target - self.height, -self.speed * dt, self.speed * dt))
        self.intent = delta / dt
        self._move_to(self.height + delta, q, pair_distances)

    def _move_to(
        self,
        candidate: float,
        q: np.ndarray,
        pair_distances: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        candidate = float(np.clip(candidate, self.path.min_height, self.path.max_height))
        self.blocked = False
        if candidate == self.height:
            return
        q_new = np.array(q, dtype=float)
        q_new[self.path.idx_q] = self.path.joints(candidate)
        now, new = pair_distances(q), pair_distances(q_new)
        if np.any((new < self.gate_distance) & (new < now - _WORSEN_TOL)):
            self.blocked = True
            return
        self.height = candidate
