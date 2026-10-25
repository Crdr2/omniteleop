"""A leader home request must re-anchor the SOLVER, not just the hardware.

``home_to_nominal`` drives the physical robot back to the nominal posture, but the IK
solver keeps its own configuration: without an explicit resync it went on solving the
pre-home end-effector targets, so the reactive safety gate kept reporting that folded
posture's self-collision distance and the headset HUD kept bannering "SELF-COLLISION
NEAR" at a robot standing safely at nominal.

This drives the REAL solver and the REAL gate through ``run_loop`` -- fake driver, no
hardware -- so the regression is checked end to end rather than by call bookkeeping.

Run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_FOLLOWERS = ("wbc_vr_robot.py",)

# Iterations to spend driving the hands together before giving up on provoking the
# warning: the QP barrier settles them ~2cm apart (inside the 3cm warn distance) well
# within this budget, so exhausting it means the fixture -- not the fix -- is broken.
MAX_CROSS_TICKS = 1500
STATIC_TICKS = 5


@pytest.fixture(params=_FOLLOWERS, ids=lambda n: n.removesuffix(".py"))
def follower(request):
    """Load one of the two real-robot followers as a module."""
    path = _SCRIPTS / request.param
    spec = importlib.util.spec_from_file_location(f"{path.stem}_home_resync_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeClock:
    """Virtual clock: ``run_loop`` advances it itself when ``realtime=False``."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _to_mat(pose) -> np.ndarray:
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _shift(mat: np.ndarray, d) -> np.ndarray:
    out = mat.copy()
    out[:3, 3] = out[:3, 3] + np.asarray(d, dtype=float)
    return out


def _frame(**kwargs):
    """One published VRJointData-shaped leader frame."""
    base = dict(
        exit_requested=False,
        estop=True,
        calib_stage="static",
        home_requested=False,
        left_ee_pose=[],
        right_ee_pose=[],
        head_ee_pose=[],
        left_gripper=0.0,
        right_gripper=0.0,
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


class _ScriptedLeader:
    """Leader stream that crosses the hands, then goes static, then requests homing.

    The cross phase runs until the gate actually warns (watching ``statuses``, which the
    fake driver appends to), so the test provokes the real condition instead of assuming
    a fixed number of ticks reaches it. Phases only ever advance.
    """

    segment_duration = None

    def __init__(self, left_cross: np.ndarray, right_cross: np.ndarray, statuses: list):
        self._left = left_cross.reshape(-1).tolist()
        self._right = right_cross.reshape(-1).tolist()
        self._statuses = statuses
        self._ns = 0
        self._phase = "cross"  # cross -> static -> home -> after (monotonic)
        self._cross_ticks = 0
        self._tail = 0
        self.done = False
        self.homed_at = -1  # index into `statuses` of the first post-home tick
        self.warned_before_home = False

    def start(self):
        pass

    def _warning(self) -> bool:
        return bool(self._statuses) and "self-collision" in self._statuses[-1]

    @property
    def latest(self):
        self._ns += 1
        if self._phase == "cross":
            if self._warning() or self._cross_ticks >= MAX_CROSS_TICKS:
                self._phase = "static"  # provoked (or gave up); e-stop from here
            else:
                self._cross_ticks += 1
                return _frame(
                    timestamp_ns=self._ns,
                    estop=False,
                    calib_stage="teleop",
                    left_ee_pose=self._left,
                    right_ee_pose=self._right,
                )

        self._tail += 1
        if self._phase == "static":  # e-stopped: the warning must persist
            self.warned_before_home = self._warning()
            if self._tail >= STATIC_TICKS:
                self._phase = "home"
            return _frame(timestamp_ns=self._ns)

        if self._phase == "home":  # the left-trigger home pulse
            self._phase = "after"
            self._tail = 0
            self.homed_at = len(self._statuses)
            return _frame(timestamp_ns=self._ns, home_requested=True)

        if self._tail >= STATIC_TICKS:
            self.done = True
        return _frame(timestamp_ns=self._ns)


def test_home_gate_refuses_engage_and_publishes_hud_status(follower, monkeypatch):
    cfg = follower.WBCConfig()
    ik = follower.VegaWholeBodyIK(cfg)
    frames = iter(
        [
            _frame(timestamp_ns=1, estop=False, calib_stage="teleop"),
            _frame(timestamp_ns=2, home_requested=True),
            _frame(timestamp_ns=3, estop=False, calib_stage="teleop"),
            _frame(timestamp_ns=4, exit_requested=True),
        ]
    )

    class Source:
        def start(self):
            pass

        @property
        def latest(self):
            return next(frames)

    held, resets, published = [], [], []
    driver = types.SimpleNamespace(name="fake", required=True)
    driver.home_required = lambda: driver.required
    driver.home_to_nominal = lambda: setattr(driver, "required", False)
    driver.extra_hold = lambda *a: "home-required" if driver.required else None
    driver.engage_reset = lambda *a: resets.append(True)
    driver.actuate = lambda result, left, right, enable, hold, dt: held.append(hold)
    driver.stop_all_motion = lambda: None
    monkeypatch.setattr(
        follower,
        "_create_status_publisher",
        lambda source: types.SimpleNamespace(publish=published.append),
    )
    args = types.SimpleNamespace(
        ik_rate=10.0,
        cmd_rate=0.0,
        head_lpf_tau=0.0,
        head_planar_pos_deadband=0.0,
        head_planar_yaw_deadband=0.0,
    )
    follower.run_loop(
        Source(),
        ik,
        cfg,
        driver,
        args,
        replay=False,
        clock=_FakeClock(),
        realtime=False,
        enable=dict(arms=True, torso=True, head=True, base=True),
    )
    assert held[0]  # unhomed active command is held
    assert len(resets) == 2  # home + accepted engage; refused engage never resets IK
    assert published[0]["home_required"] is True
    assert published[0]["hold_reason"] == "home-required"
    assert published[-1]["home_required"] is False


def test_home_request_clears_the_stale_self_collision_status(follower):
    cfg = follower.WBCConfig()
    ik = follower.VegaWholeBodyIK(cfg)
    ik.reset()
    assert ik.collision_enabled
    left0 = _to_mat(ik.frame_pose(follower.LEFT_EE_FRAME))
    right0 = _to_mat(ik.frame_pose(follower.RIGHT_EE_FRAME))

    statuses: list[str] = []
    source = _ScriptedLeader(
        _shift(left0, [0.0, -0.40, 0.0]), _shift(right0, [0.0, 0.40, 0.0]), statuses
    )
    driver = types.SimpleNamespace(
        name="fake",
        home_calls=0,
        extra_hold=lambda *_a, **_k: None,
        engage_reset=lambda *_a, **_k: None,
        actuate=lambda result, *_a, **_k: statuses.append(str(result.safety_status)),
    )
    driver.home_to_nominal = lambda: setattr(driver, "home_calls", driver.home_calls + 1)
    args = types.SimpleNamespace(
        ik_rate=100.0,
        speed=1.0,
        cmd_rate=0.0,  # 0 => interpolator passes commands through
        head_lpf_tau=0.0,
        head_planar_pos_deadband=0.0,
        head_planar_yaw_deadband=0.0,
    )

    follower.run_loop(
        source,
        ik,
        cfg,
        driver,
        args,
        replay=True,
        clock=_FakeClock(),
        realtime=False,
        enable={"arms": True, "torso": True, "head": True, "base": True},
    )

    assert driver.home_calls == 1
    # The fixture actually reproduced the reported situation: the gate warned while the
    # hands were crossed, and the warning was STILL live on the last e-stopped tick
    # before the operator pressed the trigger (e-stop alone never clears it).
    assert any("self-collision" in s for s in statuses[: source.homed_at])
    assert source.warned_before_home
    # ...and the home request cleared it. Every tick after the pulse is clean, because
    # the solver -- not just the robot -- is back at nominal.
    after = statuses[source.homed_at :]
    assert after, "no ticks recorded after the home request"
    assert set(after) == {"ok"}, f"stale safety status after homing: {sorted(set(after))}"
    assert ik.self_collision_distance(ik.configuration.q) > cfg.self_collision_warn_dist
