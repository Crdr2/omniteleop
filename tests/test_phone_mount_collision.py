"""Coverage and recorded-pose regressions for the base-mounted phone geometry."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    PHONE_MOUNT_PROXY_SPHERES,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)

ENVELOPES = np.array(
    [
        [[0.109, 0.190, 0.2817], [0.189, 0.290, 0.334]],
        [[0.134, 0.190, 0.334], [0.153, 0.290, 0.5328]],
    ]
)


def test_mount_and_top_stop_mesh_faces_fit_inside_covered_envelopes():
    """Every triangle, including its interior, lies in one convex envelope."""
    import trimesh

    root = Path(__file__).resolve().parents[1] / "hardware/vega_iphone_mount"
    mount = trimesh.load(root / "07_estop_tower_cradle.stl", force="mesh")
    stop = trimesh.load(root / "06_top_stop.stl", force="mesh")
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    surfaces = [
        mount.triangles * 0.001,
        stop.triangles * 0.001 @ rotation.T + [0.149, 0.240, 0.5248],
    ]
    # Fill the entire case pocket as well as the printed geometry.
    case_corners = np.array(list(product([0.134, 0.149], [0.19855, 0.28145], [0.358, 0.5228])))
    surfaces.append(case_corners[None])
    for points in surfaces:
        contained = (
            (
                (points[:, None] >= ENVELOPES[None, :, None, 0] - 1e-7)
                & (points[:, None] <= ENVELOPES[None, :, None, 1] + 1e-7)
            )
            .all(axis=(2, 3))
            .any(axis=1)
        )
        assert contained.all()


def test_spheres_cover_solid_envelopes_without_surface_or_interior_holes():
    """Tile each envelope; require every solid tile to fit inside a single sphere.

    Containing all eight corners in a convex sphere also contains the entire tile,
    so this checks continuous coverage rather than only sampled mesh vertices.
    """
    centers = np.array([s[1] for s in PHONE_MOUNT_PROXY_SPHERES])
    radii = np.array([s[2] for s in PHONE_MOUNT_PROXY_SPHERES])
    corner_bits = np.array(list(product([0, 1], repeat=3)))
    for lower, upper in ENVELOPES:
        count = np.ceil((upper - lower) / 0.003).astype(int)
        step = (upper - lower) / count
        cells = np.array(list(np.ndindex(*count)))
        for start in range(0, len(cells), 256):
            corners = lower + (cells[start : start + 256, None] + corner_bits) * step
            worst_corner = (
                np.linalg.norm(
                    corners[:, :, None] - centers[None, None],
                    axis=-1,
                ).max(axis=1)
                - radii
            )
            assert np.all(worst_corner.min(axis=1) < 0)


def episode3_pose(ik):
    """rubbish/episode_3, frame 199 sent joints, t=20.003 s (no dataset dependency)."""
    q = ik.nominal_q()
    for names, values in (
        (TORSO_JOINTS, [0.258908272, 0.582906187, -0.466820359]),
        (
            LEFT_ARM_JOINTS,
            [
                0.932987273,
                0.221811637,
                0.100978047,
                -1.782521963,
                1.275421381,
                -0.048460159,
                0.166886076,
            ],
        ),
        (
            RIGHT_ARM_JOINTS,
            [
                -0.231585816,
                -0.331221402,
                -1.071740508,
                -0.348334789,
                -0.660026610,
                0.302543730,
                -0.387447119,
            ],
        ),
        (HEAD_JOINTS, [-0.25, 0, 0]),
    ):
        for name, value in zip(names, values):
            q[ik._idx_q[name]] = value
    return q


def test_recorded_elbow_clearance_removes_false_warning():
    ik = VegaWholeBodyIK(
        WBCConfig(
            lock_base_in_ik=True,
            head_mode="fixed",
            phone_mount_collision=True,
            dynamic_gripper_collision=True,
        )
    )
    q = episode3_pose(ik)
    assert ik.config.self_collision_warn_dist == 0.03
    assert ik.config.self_collision_safe_dist == 0.02
    assert ik.config.self_collision_floor == 0.01
    clearance = ik.self_collision_pair_distances(q).min()
    assert 0.05 < clearance < 0.065
    assert ik.self_collision_distance(q) == pytest.approx(clearance, abs=1e-9)


def test_deliberate_mount_approach_retains_collision_protection():
    ik = VegaWholeBodyIK(
        WBCConfig(
            lock_base_in_ik=True,
            lock_torso_in_ik=True,
            head_mode="fixed",
            phone_mount_collision=True,
            dynamic_gripper_collision=True,
        )
    )
    ik.reset(episode3_pose(ik))
    left = ik.frame_pose("L_ee").homogeneous
    right = ik.frame_pose("R_ee").homogeneous
    model = ik.collision_sphere_model
    mount_pairs = [
        i
        for i, pair in enumerate(model.collisionPairs)
        if any("phone_mount" in model.geometryObjects[j].name for j in (pair.first, pair.second))
    ]
    closest = np.inf
    for i in range(600):
        target = left.copy()
        target[:3, 3] += min(i / 100, 1) * np.array([-0.1, 0, -0.35])
        result = ik.solve(target, right, 0.01)
        assert result.success and not result.held
        closest = min(closest, ik.self_collision_pair_distances(ik.q)[mount_pairs].min())
    # The reach enters the warning region but stays outside the hard floor.
    assert ik.config.self_collision_floor < closest < ik.config.self_collision_warn_dist
    assert result.left_ee_error > 0.05
