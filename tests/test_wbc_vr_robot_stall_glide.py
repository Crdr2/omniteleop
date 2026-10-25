"""A blocking call inside the control loop must not fast-forward the engage glide.

``TargetInterpolator.at()`` used to be paced on the raw wall clock, so any long
synchronous call inside ``run_loop`` advanced ``alpha`` without the hardware moving and
the next tick delivered the skipped span in one step. ``start_recording_episode()`` is
exactly such a call -- it spends ~0.84 s on four sequential NTP round-trips -- so pressing
record inside the 2 s engage glide dumped the remaining anchor-to-leader distance into one
tick: measured at 2.1 rad/s (left) and 3.0 rad/s (right) on the arms in episode_0 on
2026-09-20, against 0.11-1.05 rad/s in every clean take that evening.

This drives the REAL ``run_loop`` and the REAL solver with a fake driver (no hardware) and
compares a stalled engage against an unstalled one, so the bound is the loop's own
behaviour rather than a hand-picked constant.

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

IK_RATE = 100.0
CMD_RATE = 10.0
ESTOP_TICKS = 5  # e-stopped ticks before the engage edge
STALL_AT_TICK = 30  # ticks after engage at which the loop blocks
STALL_S = 0.84  # the measured start_recording_episode() block
TOTAL_TICKS = 220  # virtual time runs 2 ticks/iteration: clears the 2 s glide
REACH = 0.15  # m the leader target sits from the nominal anchor


def _max_advance_ticks(follower) -> int:
    return int(follower.MAX_TARGET_CLOCK_ADVANCE_TICKS)


@pytest.fixture(params=_FOLLOWERS, ids=lambda n: n.removesuffix(".py"))
def follower(request):
    path = _SCRIPTS / request.param
    spec = importlib.util.spec_from_file_location(f"{path.stem}_stall_glide_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _StallClock:
    """Virtual wall clock advancing one tick per call, plus one injected stall.

    ``run_loop`` reads it twice per iteration (the tick stamp and the realtime sleep
    budget), so a per-call tick of ``dt`` leaves the sleep at exactly zero -- the loop
    free-runs without patching ``time.sleep`` -- and consecutive tick stamps are ``2*dt``
    apart, comfortably inside the clamp.
    """

    def __init__(self, tick: float) -> None:
        self.t = 0.0
        self._tick = tick
        self._pending = 0.0

    def __call__(self) -> float:
        self.t += self._tick + self._pending
        self._pending = 0.0
        return self.t

    def stall(self, seconds: float) -> None:
        self._pending = seconds


def _to_mat(pose) -> np.ndarray:
    return np.asarray(pose.homogeneous if hasattr(pose, "homogeneous") else pose, dtype=float)


def _shift(mat: np.ndarray, d) -> np.ndarray:
    out = mat.copy()
    out[:3, 3] = out[:3, 3] + np.asarray(d, dtype=float)
    return out


def _frame(**kwargs):
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
        chassis_vx=0.0,
        chassis_vy=0.0,
        chassis_wz=0.0,
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


class _EngageThenReach:
    """E-stop, then engage onto a target REACH metres from the nominal anchor.

    Publishes at ``CMD_RATE`` (the timestamp only advances every 10th tick) so the
    interpolator blends real 10 Hz segments exactly as it does against a live leader.
    Optionally stalls the clock ``STALL_AT_TICK`` ticks after the engage edge, standing in
    for ``start_recording_episode()`` blocking the loop.
    """

    segment_duration = None

    def __init__(self, left, right, clock, *, stall: bool) -> None:
        self._left = left.reshape(-1).tolist()
        self._right = right.reshape(-1).tolist()
        self._clock = clock
        self._stall = stall
        self._tick = 0
        self._ns = 0
        self.done = False

    def start(self):
        pass

    @property
    def latest(self):
        self._tick += 1
        if self._tick >= TOTAL_TICKS:
            self.done = True
            # run_loop only consults ``source.done`` for replay; a live source leaves
            # through the leader's own exit request.
            return _frame(timestamp_ns=self._ns, exit_requested=True)
        if self._tick <= ESTOP_TICKS:
            self._ns += 1
            return _frame(timestamp_ns=self._ns)
        engaged = self._tick - ESTOP_TICKS
        if self._stall and engaged == STALL_AT_TICK:
            self._clock.stall(STALL_S)
        if engaged % int(IK_RATE / CMD_RATE) == 1:
            self._ns += 1  # a fresh 10 Hz leader command
        return _frame(
            timestamp_ns=self._ns,
            estop=False,
            calib_stage="teleop",
            left_ee_pose=self._left,
            right_ee_pose=self._right,
        )


def _run(follower, *, stall: bool):
    """Drive run_loop once; return the per-tick left-hand TARGET path."""
    cfg = follower.WBCConfig()
    ik = follower.VegaWholeBodyIK(cfg)
    ik.reset()
    left0 = _to_mat(ik.frame_pose(follower.LEFT_EE_FRAME))
    right0 = _to_mat(ik.frame_pose(follower.RIGHT_EE_FRAME))

    clock = _StallClock(1.0 / IK_RATE)
    source = _EngageThenReach(
        _shift(left0, [REACH, 0.0, 0.0]),
        _shift(right0, [REACH, 0.0, 0.0]),
        clock,
        stall=stall,
    )
    targets: list[np.ndarray] = []
    driver = types.SimpleNamespace(
        name="fake",
        extra_hold=lambda *_a, **_k: None,
        engage_reset=lambda *_a, **_k: None,
        actuate=lambda *_a, **_k: None,
        stop_all_motion=lambda *_a, **_k: None,
        record_tick=lambda result, hold, now, *, left_target, **_k: targets.append(
            np.asarray(left_target, dtype=float)[:3, 3].copy()
        ),
    )
    args = types.SimpleNamespace(
        ik_rate=IK_RATE,
        speed=1.0,
        cmd_rate=CMD_RATE,
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
        replay=False,
        clock=clock,
        realtime=True,
        enable={"arms": True, "torso": True, "head": True, "base": True},
    )
    return np.asarray(targets)


def _max_step(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).max())


def test_a_loop_stall_cannot_fast_forward_the_engage_glide(follower):
    clean = _run(follower, stall=False)
    stalled = _run(follower, stall=True)

    assert len(clean) > STALL_AT_TICK + 50, "fixture never got past the engage glide"
    assert len(stalled) > STALL_AT_TICK + 50

    clean_step = _max_step(clean)
    stalled_step = _max_step(stalled)
    MAX_ADVANCE_TICKS = _max_advance_ticks(follower)

    # The clamp lets the target clock buy at most MAX_TARGET_CLOCK_ADVANCE_TICKS ticks in
    # one iteration, so that is the most a stall can inflate the per-tick target step. The
    # bound is generous and still decisive: against the pre-fix loop this same fixture
    # steps 64.50 mm (43% of REACH, the lunge) where the fixed loop steps 3.00 mm.
    assert stalled_step <= clean_step * MAX_ADVANCE_TICKS + 1e-9, (
        f"loop stall inflated the per-tick hand-target step: "
        f"{stalled_step * 1000:.2f} mm vs {clean_step * 1000:.2f} mm unstalled"
    )
    # Absolute guard, independent of the unstalled baseline: a lunge covers a large
    # fraction of the anchor-to-leader distance in one tick; a glided approach cannot.
    assert stalled_step < 0.05 * REACH, (
        f"stalled run covered {stalled_step / REACH:.0%} of the reach in one tick"
    )
    # ...and the glide still delivers: both runs finish at the same commanded target.
    assert np.allclose(clean[-1], stalled[-1], atol=1e-6), (
        f"stalled run converged elsewhere: {stalled[-1]} vs {clean[-1]}"
    )


def test_the_joystick_follower_shares_the_fixed_loop():
    """wbc_joystick_robot re-binds this exact run_loop, so it inherits the clamp.

    Asserted rather than re-run: the joystick entry point swaps the base DRIVER, not the
    target pipeline, and the fake driver above already stands in for both.
    """
    spec = importlib.util.spec_from_file_location(
        "wbc_joystick_robot_shared_loop_test", _SCRIPTS / "wbc_joystick_robot.py"
    )
    joystick = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(joystick)
    base = joystick._base  # noqa: SLF001
    assert joystick.run_loop is base.run_loop
    assert base.MAX_TARGET_CLOCK_ADVANCE_TICKS > 0
