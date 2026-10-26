"""Measured-width collision geometry, feedback bounds and live cache consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from omniteleop.follower.gripper_collision import (
    enclose_interval,
    feedback_interval,
    pose_open_points,
)
from omniteleop.follower.robotiq import build_hande_command
from omniteleop.follower.whole_body_ik import (
    DEFAULT_CONFIG_PATH,
    ROBOTIQ_FIXED_PROXY_SPHERES,
    ROBOTIQ_PROXY_SPHERES,
    VegaWholeBodyIK,
    WBCConfig,
)


def _status(**changes):
    status = dict(
        gPO=204,
        gPR=228,
        gACT=1,
        gSTA_bits=3,
        gFLT=0,
        kFLT=0,
        request_send_monotonic_ns=9_990_000_000,
        read_monotonic_ns=9_995_000_000,
    )
    status.update(changes)
    return status


def test_feedback_uses_achieved_position_and_request_age():
    lo, hi = feedback_interval(_status(), 10, 0.01)
    assert (lo + hi) / 2 == pytest.approx(204 / 255)
    assert hi - 204 / 255 == pytest.approx(0.04 + 2 / 255)
    assert feedback_interval(_status(gPR=0), 10, 0.01) == (lo, hi)
    assert feedback_interval(_status(), 10.1, 0.01)[0] < lo


@pytest.mark.parametrize(
    "changes",
    [
        {"gPO": -1},
        {"gPO": 256},
        {"gPO": float("nan")},
        {"gACT": 0},
        {"gSTA_bits": 1},
        {"gFLT": 1},
        {"kFLT": 1},
        {"read_monotonic_ns": 9_700_000_000},
        {"read_monotonic_ns": 10_001_000_000},
        {"request_send_monotonic_ns": 0},
        {"request_send_monotonic_ns": 9_800_000_000},
    ],
)
def test_bad_feedback_uses_full_sweep(changes):
    assert feedback_interval(_status(**changes), 10, 0.01) == (0, 1)


def test_missing_feedback_uses_full_sweep():
    assert feedback_interval(None, 10, 0.01) == (0, 1)
    assert feedback_interval({}, 10, 0.01) == (0, 1)


def test_echoed_stop_stays_narrow_but_pending_command_keeps_motion_bound():
    status = _status(gGTO=0, gPR=204)
    command = build_hande_command(204 / 255, go_to=False)
    narrow = feedback_interval(status, 10.1, 0.01, command=command, command_change=9.9)
    assert np.mean(narrow) == pytest.approx(204 / 255)
    assert narrow[1] - 204 / 255 == pytest.approx(0.02 + 2 / 255)
    # Requests preceding the new command must not infer a stopped jaw from an old echo.
    assert feedback_interval(status, 10.1, 0.01, command=command, command_change=10)[0] < narrow[0]
    opening = build_hande_command(0)
    assert feedback_interval(status, 10.1, 0.01, command=opening, command_change=9.9)[0] < narrow[0]


def test_closing_contact_keeps_remaining_path_to_echoed_target():
    command = build_hande_command(228 / 255)
    status = _status(gGTO=1, gOBJ=2)
    lo, hi = feedback_interval(status, 10.1, 0.01, command=command, command_change=9.9)
    assert lo == pytest.approx(204 / 255 - 0.02 - 2 / 255)
    assert hi == pytest.approx(228 / 255 + 0.02 + 2 / 255)


def test_stroke_is_rebased_to_open_and_uses_raw_encoder():
    points = np.array([[0, 0.07, 0.25], [0, -0.07, 0.25]])
    signs = np.array([1, -1])
    for side in ("left", "right"):
        np.testing.assert_allclose(pose_open_points(side, points, signs, 3 / 255), points)
        closed = pose_open_points(side, points, signs, 0.89)
        # Expected travel includes fitted rotation, not just a horizontal jaw shift.
        assert np.all(np.abs(closed[:, 1]) < 0.031)
        assert np.all((closed - points)[:, 2] > 0.009)
    assert not np.allclose(
        pose_open_points("left", points, signs, 0.89),
        pose_open_points("right", points, signs, 0.89),
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_interval_encloses_continuous_sphere_motion(side):
    points = np.array([[0.01, 0.055, 0.25], [-0.01, -0.052, 0.16]])
    signs = np.array([1, -1])
    radius = np.array([0.02, 0.025])
    for lo, hi in [(0, 1), (0.2, 0.6), (0.79, 0.89), (0.81, 0.81)]:
        centers, radii = enclose_interval(side, points, signs, radius, (lo, hi))
        for w in np.linspace(lo, hi, 501):
            posed = pose_open_points(side, points, signs, w)
            assert np.all(np.linalg.norm(posed - centers, axis=1) + radius <= radii + 1e-12)


def test_geometry_updates_same_q_barrier_coal_and_probe_caches():
    ik = VegaWholeBodyIK(WBCConfig(dynamic_gripper_collision=True, phone_mount_collision=True))
    q = ik.configuration.q.copy()
    sm = ik.collision_sphere_model
    pairs = [(p.first, p.second) for p in sm.collisionPairs]
    barrier = ik.collision_barrier
    original = barrier.compute_qp_inequalities(ik.configuration, 0.01)[1].copy()
    ik.update_gripper_collision_geometry(left=(0.8, 0.8), right=(0.7, 0.7))
    np.testing.assert_array_equal(ik.configuration.q, q)
    assert [(p.first, p.second) for p in sm.collisionPairs] == pairs
    np.testing.assert_allclose(ik._sc_radius, barrier._radius)
    # Same q must still recompute the QP cache; compare to freshly selected barriers.
    _, rhs = barrier.compute_qp_inequalities(ik.configuration, 0.01)
    np.testing.assert_allclose(np.sort(rhs), np.sort(barrier.compute_barrier(ik.configuration)))
    assert not np.array_equal(rhs, original)
    exact = np.array([r.min_distance for r in ik.configuration.collision_data.distanceResults])
    np.testing.assert_allclose(ik.self_collision_pair_distances(q), exact, atol=1e-12)
    assert ik._min_self_distance() == pytest.approx(ik.self_collision_distance(q))
    # One missing side must widen just that side; restoring both recovers the old envelope.
    right_before = ik._sc_radius[ik._gripper_collision["right"]["indices"]].copy()
    ik.update_gripper_collision_geometry(right=(0.7, 0.7))
    np.testing.assert_array_equal(
        ik._sc_radius[ik._gripper_collision["right"]["indices"]], right_before
    )
    ik.update_gripper_collision_geometry()
    np.testing.assert_allclose(
        np.sort(barrier.compute_qp_inequalities(ik.configuration, 0.01)[1]), np.sort(original)
    )


def test_optional_collision_geometry_defaults_to_fixed_grippers_without_mount():
    cfg = WBCConfig.from_yaml()
    assert not cfg.dynamic_gripper_collision
    assert not cfg.phone_mount_collision
    assert cfg.enable_collision_avoidance


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("mount", [False, True])
def test_yaml_options_select_collision_geometry_independently(tmp_path, dynamic, mount):
    data = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    data.update(dynamic_gripper_collision=dynamic, phone_mount_collision=mount)
    config = tmp_path / "wbik.yaml"
    config.write_text(yaml.safe_dump(data))
    ik = VegaWholeBodyIK(config_path=config)
    geoms = {g.name: g for g in ik.collision_sphere_model.geometryObjects}
    expected = ROBOTIQ_PROXY_SPHERES if dynamic else ROBOTIQ_FIXED_PROXY_SPHERES
    for prefix in ("L_robotiq", "R_robotiq"):
        assert {n for n in geoms if n.startswith(prefix)} == {
            f"{prefix}_{label}" for label, _, _ in expected
        }
    assert sum("phone_mount" in n for n in geoms) == (33 if mount else 0)
    assert bool(ik._gripper_collision) is dynamic
    assert ik.collision_enabled and ik.collision_barrier is not None
    if not dynamic:
        # Fixed spheres keep the earlier centers/radii, not just the earlier count.
        for prefix in ("L_robotiq", "R_robotiq"):
            frame = ik.model.frames[ik.model.getFrameId(prefix)]
            for label, xyz, radius in expected:
                geom = geoms[f"{prefix}_{label}"]
                np.testing.assert_allclose(
                    geom.placement.translation,
                    frame.placement.act(np.asarray(xyz)),
                )
                assert geom.geometry.radius == radius


def test_fixed_gripper_geometry_ignores_feedback_without_recomputing_distances(monkeypatch):
    ik = VegaWholeBodyIK()
    original = ik.collision_barrier.compute_qp_inequalities(ik.configuration, 0.01)[1].copy()

    def unexpected_update(*args):
        pytest.fail("fixed gripper geometry must not trigger a configuration update")

    monkeypatch.setattr(ik.configuration, "update", unexpected_update)
    ik.update_gripper_collision_geometry(left=(0.8, 0.8), right=(0.7, 0.7))
    ik.update_gripper_collision_geometry()
    np.testing.assert_array_equal(
        ik.collision_barrier.compute_qp_inequalities(ik.configuration, 0.01)[1],
        original,
    )
