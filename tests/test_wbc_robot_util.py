"""Unit tests for omniteleop.wbc_robot_util (enable mask, joint clamp, base encoding).

numpy-only; run via the dexmate env with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1.
"""

from __future__ import annotations

import numpy as np
import pytest

from omniteleop.wbc_robot_util import (
    ENABLE_GROUPS,
    base_quiet_dispatch,
    clamp_joint_step,
    compute_hold_reason,
    parse_enable_mask,
    set_base_in_q,
)


def test_parse_enable_mask_variants():
    assert parse_enable_mask("arms,torso,head,base") == dict.fromkeys(ENABLE_GROUPS, True)
    assert parse_enable_mask("all") == dict.fromkeys(ENABLE_GROUPS, True)
    assert parse_enable_mask("") == dict.fromkeys(ENABLE_GROUPS, False)
    assert parse_enable_mask("none") == dict.fromkeys(ENABLE_GROUPS, False)
    assert parse_enable_mask("arms, head") == {
        "arms": True,
        "torso": False,
        "head": True,
        "base": False,
    }
    # First-run default per the user's choice: arms+torso+head+base, grippers separate.
    assert parse_enable_mask("arms,torso,head,base")["base"] is True


def test_parse_enable_mask_rejects_unknown():
    with pytest.raises(ValueError, match="unknown --enable"):
        parse_enable_mask("arms,legs")


def test_clamp_joint_step_limits_per_joint_delta():
    prev = np.zeros(3)
    target = np.array([0.05, -0.30, 0.011])
    clamped, max_abs = clamp_joint_step(prev, target, max_step=0.02)
    np.testing.assert_allclose(clamped, [0.02, -0.02, 0.011])
    assert max_abs == pytest.approx(0.30)  # the largest requested step, pre-clamp


def test_clamp_joint_step_disabled_passes_through():
    prev = np.zeros(2)
    target = np.array([1.0, -2.0])
    for disabled in (0.0, -1.0, float("inf")):
        clamped, max_abs = clamp_joint_step(prev, target, max_step=disabled)
        np.testing.assert_allclose(clamped, target)
        assert max_abs == pytest.approx(2.0)


def test_clamp_joint_step_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        clamp_joint_step(np.zeros(3), np.zeros(2), max_step=0.1)


def test_set_base_in_q_uses_cos_sin_planar_encoding():
    q = np.arange(10, dtype=float)
    yaw = 0.6
    out = set_base_in_q(q, np.array([1.5, -2.0, yaw]))
    # x, y at 0,1; cos/sin(yaw) at 2,3; everything else untouched.
    assert out[0] == pytest.approx(1.5)
    assert out[1] == pytest.approx(-2.0)
    assert out[2] == pytest.approx(np.cos(yaw))
    assert out[3] == pytest.approx(np.sin(yaw))
    np.testing.assert_allclose(out[4:], q[4:])
    # Unit-norm rotation block (the solver asserts this).
    assert out[2] ** 2 + out[3] ** 2 == pytest.approx(1.0)
    # Input not mutated.
    np.testing.assert_allclose(q, np.arange(10))


def test_set_base_in_q_validates_shapes():
    with pytest.raises(ValueError, match=">= 4"):
        set_base_in_q(np.zeros(3), np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="base_pose"):
        set_base_in_q(np.zeros(6), np.array([0.0, 0.0]))


def test_compute_hold_reason_estop_never_holds_here():
    # e-stop is handled by the caller; this watchdog stands down so it doesn't double-report.
    assert (
        compute_hold_reason(estop=True, last_cmd_wall=None, now=100.0, source_timeout=0.5) is None
    )


def test_compute_hold_reason_awaits_first_command():
    # Engaged but no command yet -> freeze, don't actuate toward the nominal seed.
    assert (
        compute_hold_reason(estop=False, last_cmd_wall=None, now=100.0, source_timeout=0.5)
        == "awaiting-command"
    )


def test_compute_hold_reason_stale_source():
    assert (
        compute_hold_reason(estop=False, last_cmd_wall=100.0, now=100.3, source_timeout=0.5) is None
    )
    assert (
        compute_hold_reason(estop=False, last_cmd_wall=100.0, now=100.6, source_timeout=0.5)
        == "stale-source"
    )


def test_compute_hold_reason_stale_odom_only_when_provided():
    # Fresh command, but odometry is stale -> hold; None odom_age (base off) -> no check.
    assert (
        compute_hold_reason(
            estop=False, last_cmd_wall=100.0, now=100.1, source_timeout=0.5, odom_age=1.0
        )
        == "stale-odom"
    )
    assert (
        compute_hold_reason(
            estop=False, last_cmd_wall=100.0, now=100.1, source_timeout=0.5, odom_age=None
        )
        is None
    )


def test_base_quiet_dispatch_active_drives_and_resets_timer():
    # Any axis above quiet_eps -> drive, regardless of how long we'd been quiet.
    for cmd in ([0.2, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.1]):
        action, elapsed = base_quiet_dispatch(
            np.array(cmd), prev_quiet_elapsed_s=5.0, dt=0.01, quiet_hold_s=0.5
        )
        assert action == "drive"
        assert elapsed == 0.0


def test_base_quiet_dispatch_holds_within_window_then_recenters():
    dt, hold = 0.02, 0.05  # margins (0.04 < 0.05 < 0.06) avoid the exact-boundary float edge
    zero = np.zeros(3)
    elapsed = 0.0
    actions = []
    for _ in range(5):
        action, elapsed = base_quiet_dispatch(zero, elapsed, dt, hold)
        actions.append(action)
    # elapsed = 0.02, 0.04 (< 0.05 -> hold), then 0.06, 0.08, 0.10 (>= 0.05 -> recenter)
    assert actions == ["hold", "hold", "recenter", "recenter", "recenter"]
    assert elapsed == pytest.approx(0.10)


def test_base_quiet_dispatch_first_quiet_tick_holds():
    action, elapsed = base_quiet_dispatch(
        np.zeros(3), prev_quiet_elapsed_s=0.0, dt=0.01, quiet_hold_s=0.5
    )
    assert action == "hold"
    assert elapsed == pytest.approx(0.01)


def test_base_quiet_dispatch_window_zero_always_recenters():
    # quiet_hold_s == 0 reproduces the original always-re-center behavior.
    action, elapsed = base_quiet_dispatch(
        np.zeros(3), prev_quiet_elapsed_s=0.0, dt=0.01, quiet_hold_s=0.0
    )
    assert action == "recenter"
    assert elapsed == pytest.approx(0.01)


def test_base_quiet_dispatch_negative_window_never_recenters():
    zero = np.zeros(3)
    elapsed = 0.0
    for _ in range(1000):
        action, elapsed = base_quiet_dispatch(zero, elapsed, dt=0.01, quiet_hold_s=-1.0)
        assert action == "hold"
    assert elapsed == pytest.approx(10.0)


def test_base_quiet_dispatch_drive_after_quiet_resets():
    # Quiet for a while, then an active command resets the timer to 0.
    _, elapsed = base_quiet_dispatch(np.zeros(3), 0.0, 0.01, 0.5)
    _, elapsed = base_quiet_dispatch(np.zeros(3), elapsed, 0.01, 0.5)
    action, elapsed = base_quiet_dispatch(np.array([0.0, 0.3, 0.0]), elapsed, 0.01, 0.5)
    assert action == "drive"
    assert elapsed == 0.0


def test_base_quiet_dispatch_validates_inputs():
    with pytest.raises(ValueError, match="shape"):
        base_quiet_dispatch(np.zeros(2), 0.0, 0.01, 0.5)
    with pytest.raises(ValueError, match="dt"):
        base_quiet_dispatch(np.zeros(3), 0.0, -0.01, 0.5)
    with pytest.raises(ValueError, match="quiet_hold_s"):
        base_quiet_dispatch(np.zeros(3), 0.0, 0.01, float("nan"))


def test_base_quiet_dispatch_rejects_non_finite_cmd():
    # NaN/inf must NOT slip through as "quiet" (NaN > eps is False) and reach the chassis.
    for bad in ([np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, -np.inf]):
        with pytest.raises(ValueError, match="cmd must be finite"):
            base_quiet_dispatch(np.array(bad), 0.0, 0.01, 0.5)


def test_base_quiet_dispatch_rejects_bad_prev_elapsed():
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="prev_quiet_elapsed_s"):
            base_quiet_dispatch(np.zeros(3), bad, 0.01, 0.5)
