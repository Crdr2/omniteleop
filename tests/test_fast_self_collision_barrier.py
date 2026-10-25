"""Equivalence tests for the Tier 2 vectorized self-collision barrier.

:class:`~omniteleop.follower.fast_self_collision_barrier.FastSelfCollisionBarrier` is a
drop-in for pink's :class:`~pink.barriers.SelfCollisionBarrier` that replaces coal's
all-pair narrowphase read with closed-form sphere math. These tests pin the equivalence
that makes it safe to swap in: the barrier value ``h``, the Jacobian ``J``, and the QP
inequality rows ``(G, h)`` it feeds the solver must match the stock barrier (the QP
objective is identically zero here because ``collision_safe_displacement_gain == 0``).

Equivalence is checked across random configurations (including selected pairs in
penetration, where the barrier bites hardest) and on real teleop joint trajectories
recorded from the robot, if present on this workstation.

Needs the dexmate env (pinocchio + coal). Run with pytest in that environment::

    <dexmate-python> -m pytest tests/test_fast_self_collision_barrier.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pinocchio as pin
from pink.barriers import SelfCollisionBarrier

from omniteleop.follower.fast_self_collision_barrier import (
    FastSelfCollisionBarrier,
    build_centers_from_placements,
)
from omniteleop.follower.whole_body_ik import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    TORSO_JOINTS,
    VegaWholeBodyIK,
    WBCConfig,
)

try:
    import coal as _coal
except ImportError:  # pragma: no cover
    import hppfcl as _coal

DT = 0.01  # 100 Hz control rate
RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Equivalence bars: ~4 orders above the observed machine-epsilon residuals (h~5e-17,
# J~3e-16, G=J/dt~3e-14), yet far below the QP solver tolerance (~1e-8) and any
# control-relevant scale -- so passing means the solver cannot tell the barriers apart.
_ATOL_H = 1e-12
_ATOL_J = 1e-10
_ATOL_G = 1e-9


# --- helpers -------------------------------------------------------------------


def _make():
    """Build the solver plus matching fast + stock barriers (same params).

    The solver now installs the fast barrier itself (``ik.collision_barrier`` is a
    ``FastSelfCollisionBarrier`` after integration), so the equivalence reference is a
    freshly-built stock ``SelfCollisionBarrier`` -- never ``ik.collision_barrier``, which
    would make the comparison vacuous.
    """
    ik = VegaWholeBodyIK(WBCConfig())
    ik.reset()
    cfg = ik.config
    kw = dict(
        n_collision_pairs=ik.collision_barrier.dim,
        gain=cfg.collision_barrier_gain,
        safe_displacement_gain=cfg.collision_safe_displacement_gain,
        d_min=cfg.self_collision_safe_dist,
    )
    fast = FastSelfCollisionBarrier(ik.collision_sphere_model, **kw)
    stock = SelfCollisionBarrier(**kw)
    return ik, fast, stock


def _joint_q(ik, *, torso, left_arm, right_arm, head, base_pose=None):
    """Assemble a model ``q`` from per-group joint vectors (dexcontrol order, 1:1).

    ``base_pose`` is an optional ``(x, y, yaw)``; the planar root is ``[x, y, cos, sin]``.
    Joints follow the model names torso_j{1..3}, L/R_arm_j{1..7}, head_j{1..3}.
    """
    q = ik.nominal_q().copy()
    if base_pose is not None:
        x, y, yaw = (float(v) for v in base_pose)
        q[0], q[1], q[2], q[3] = x, y, np.cos(yaw), np.sin(yaw)
    for names, vals in (
        (TORSO_JOINTS, torso),
        (LEFT_ARM_JOINTS, left_arm),
        (RIGHT_ARM_JOINTS, right_arm),
        (HEAD_JOINTS, head),
    ):
        for name, val in zip(names, vals, strict=True):
            q[ik._idx_q[name]] = float(val)  # noqa: SLF001
    return q


def _random_configs(ik, n, seed=0):
    """Random configs over the full actuated joint ranges (hits the penetration regime)."""
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit
    idx = [
        ik._idx_q[n_]
        for n_ in TORSO_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS  # noqa: SLF001
    ]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = ik.nominal_q().copy()
        for i in idx:
            q[i] = float(rng.uniform(lo[i] + 1e-3, hi[i] - 1e-3))
        yield q


def _selected_set(barrier, configuration):
    """The set of closest-pair geometry-index pairs the barrier selects at this config."""
    centers = barrier._centers(configuration)  # noqa: SLF001
    _, idx = barrier._select(centers)  # noqa: SLF001
    return {frozenset((int(barrier._gi[k]), int(barrier._gj[k]))) for k in idx}  # noqa: SLF001


def _stock_selected_set(ik):
    """Same selection read from coal's per-pair distances (the stock barrier's source)."""
    cd = ik.configuration.collision_data
    pairs = ik.configuration.collision_model.collisionPairs
    dists = np.array([cd.distanceResults[k].min_distance for k in range(len(pairs))])
    idx = np.argpartition(-dists, -ik.collision_barrier.dim)[-ik.collision_barrier.dim :]
    return {frozenset((int(pairs[int(k)].first), int(pairs[int(k)].second))) for k in idx}


# --- equivalence: synthetic configs --------------------------------------------


def test_barrier_value_matches_stock():
    """``compute_barrier`` (h) matches pink's stock barrier across random configs."""
    ik, fast, stock = _make()
    for q in _random_configs(ik, 60):
        ik.configuration.update(q)
        np.testing.assert_allclose(
            fast.compute_barrier(ik.configuration),
            stock.compute_barrier(ik.configuration),
            atol=_ATOL_H,
        )


def test_barrier_jacobian_matches_stock():
    """``compute_jacobian`` (J, the dim x nv gradient matrix) matches stock."""
    ik, fast, stock = _make()
    for q in _random_configs(ik, 60):
        ik.configuration.update(q)
        np.testing.assert_allclose(
            fast.compute_jacobian(ik.configuration),
            stock.compute_jacobian(ik.configuration),
            atol=_ATOL_J,
        )


def test_qp_inequality_rows_match_stock():
    """The assembled QP rows ``(G, h)`` -- what the solver actually consumes -- match."""
    ik, fast, stock = _make()
    for q in _random_configs(ik, 60):
        ik.configuration.update(q)
        Gs, hs = stock.compute_qp_inequalities(ik.configuration, dt=DT)
        Gf, hf = fast.compute_qp_inequalities(ik.configuration, dt=DT)
        np.testing.assert_allclose(Gf, Gs, atol=_ATOL_G)
        np.testing.assert_allclose(hf, hs, atol=_ATOL_H)


def test_qp_objective_is_zero_both():
    """With safe_displacement_gain=0 the barrier objective is identically zero (both)."""
    ik, fast, stock = _make()
    ik.configuration.update(ik.nominal_q())
    for barrier in (stock, fast):
        H, c = barrier.compute_qp_objective(ik.configuration)
        assert not np.any(H) and not np.any(c)


def test_selection_set_matches_stock():
    """The chosen closest-pair *set* matches coal's ranking (QP feasible set identity).

    The QP constraint is a conjunction, so its feasible set depends on the *set* of
    (J row, h) pairs, not their order -- this asserts the set the vectorized broadphase
    selects equals the set coal's per-pair distances would.
    """
    ik, fast, _ = _make()  # reference is coal-direct (_stock_selected_set), not `stock`
    for q in _random_configs(ik, 60):
        ik.configuration.update(q)
        assert _selected_set(fast, ik.configuration) == _stock_selected_set(ik)


def test_penetration_regime_covered_and_equivalent():
    """The sweep exercises penetrating selected pairs (h<0) and stays equivalent there.

    The barrier matters most as pairs cross d_min into contact/penetration (the normal
    flips sign): assert the random sweep actually reaches min(h) < 0 for several configs,
    and that h/J equivalence holds on exactly those configs.
    """
    ik, fast, stock = _make()
    penetrating = 0
    for q in _random_configs(ik, 60):
        ik.configuration.update(q)
        h_stock = stock.compute_barrier(ik.configuration)
        if h_stock.min() < 0.0:
            penetrating += 1
            np.testing.assert_allclose(
                fast.compute_barrier(ik.configuration), h_stock, atol=_ATOL_H
            )
            np.testing.assert_allclose(
                fast.compute_jacobian(ik.configuration),
                stock.compute_jacobian(ik.configuration),
                atol=_ATOL_J,
            )
    assert penetrating >= 5, f"sweep only reached penetration in {penetrating} configs"


def test_real_example_pose_equivalent():
    """Equivalence on the measured hardware pose (torso_l3_0 / R_arm_l2_2 @ 2.00 cm)."""
    ik, fast, stock = _make()
    q = _joint_q(
        ik,
        torso=(7.7981311e-01, 1.5707964e00, -3.4906584e-04),
        left_arm=[ik.nominal_q()[ik._idx_q[n]] for n in LEFT_ARM_JOINTS],  # noqa: SLF001
        right_arm=(
            -1.272989,
            0.0807494,
            -0.85169446,
            -1.8787196,
            -2.0161536,
            -0.64398456,
            0.03259751,
        ),
        head=(0.0, 0.0, 0.0),
    )
    ik.configuration.update(q)
    np.testing.assert_allclose(
        fast.compute_barrier(ik.configuration),
        stock.compute_barrier(ik.configuration),
        atol=_ATOL_H,
    )
    np.testing.assert_allclose(
        fast.compute_jacobian(ik.configuration),
        stock.compute_jacobian(ik.configuration),
        atol=_ATOL_J,
    )


# --- narrowphase-free centre source (the standalone fast path) -----------------


def test_placement_only_centers_match_full_update():
    """``build_centers_from_placements`` (no narrowphase) == centres after a full update.

    Tier 2's actual speed win requires sourcing sphere centres from
    ``updateGeometryPlacements`` alone, skipping coal's ``computeDistances``. Confirm
    those centres equal the ones a full ``Configuration.update`` produces, so the
    barrier sees identical geometry either way.
    """
    ik, fast, _ = _make()  # placement-vs-full check uses `fast` centres only
    sphere_data = ik.collision_sphere_model.createData()
    for q in _random_configs(ik, 20):
        ik.configuration.update(q)
        full = fast._centers(ik.configuration)  # noqa: SLF001
        placement_only = build_centers_from_placements(
            ik.model, ik.model.createData(), ik.collision_sphere_model, sphere_data, q
        )
        np.testing.assert_allclose(placement_only, full, atol=1e-12)


# --- degenerate guards ---------------------------------------------------------


def test_non_sphere_model_rejected():
    """A model with a non-sphere geometry must raise (caller should keep stock barrier)."""
    gm = pin.GeometryModel()
    gm.addGeometryObject(
        pin.GeometryObject("box", 0, 0, pin.SE3.Identity(), _coal.Box(0.1, 0.1, 0.1))
    )
    raised = False
    try:
        FastSelfCollisionBarrier(gm, n_collision_pairs=1)
    except TypeError:
        raised = True
    assert raised


def _two_sphere_model(radius=0.05):
    """Minimal model: two translation joints, one sphere each, one collision pair.

    Lets a test drive the two sphere centres to EXACTLY coincide (q1 == q2), the
    degenerate case where the closed-form normal is undefined.
    """
    from pink.barriers import SelfCollisionBarrier  # noqa: PLC0415
    from pink.configuration import Configuration  # noqa: PLC0415

    model = pin.Model()
    for name in ("j1", "j2"):
        jid = model.addJoint(0, pin.JointModelTranslation(), pin.SE3.Identity(), name)
        model.appendBodyToJoint(jid, pin.Inertia.Identity(), pin.SE3.Identity())
    gm = pin.GeometryModel()
    i1 = gm.addGeometryObject(
        pin.GeometryObject("s1", 1, 0, pin.SE3.Identity(), _coal.Sphere(radius))
    )
    i2 = gm.addGeometryObject(
        pin.GeometryObject("s2", 2, 0, pin.SE3.Identity(), _coal.Sphere(radius))
    )
    gm.addCollisionPair(pin.CollisionPair(i1, i2))
    stock = SelfCollisionBarrier(
        n_collision_pairs=1, gain=1.0, safe_displacement_gain=0.0, d_min=0.02
    )
    fast = FastSelfCollisionBarrier(
        gm, n_collision_pairs=1, gain=1.0, safe_displacement_gain=0.0, d_min=0.02
    )
    return model, gm, stock, fast, Configuration


def test_coincident_centre_fallback_matches_stock():
    """Coincident sphere centres: the guarded fast row reproduces pink's coal row.

    Without the fallback the closed-form normal is 0/0 -> a zeroed row, while coal
    returns finite witness points and pink builds a (generally nonzero) row. The
    fallback reads the same coal witness points, so fast == stock even here.
    """
    model, gm, stock, fast, Configuration = _two_sphere_model(0.05)
    data = model.createData()
    cases = {
        "coincident": np.zeros(model.nq),  # centres identical (sep=0)
        "separated": np.array([0.0, 0, 0, 0.2, 0, 0]),  # 20 cm apart
        "penetrating": np.array([0.0, 0, 0, 0.04, 0, 0]),  # 4 cm < 2r => overlap
    }
    for q in cases.values():
        conf = Configuration(model, data, q, collision_model=gm, collision_data=gm.createData())
        np.testing.assert_allclose(
            fast.compute_barrier(conf), stock.compute_barrier(conf), atol=_ATOL_H
        )
        np.testing.assert_allclose(
            fast.compute_jacobian(conf), stock.compute_jacobian(conf), atol=_ATOL_J
        )


def test_pair_count_mutation_rejected():
    """Mutating the model's collision pairs after construction must fail loud."""
    model, gm, _stock, fast, Configuration = _two_sphere_model(0.05)
    data = model.createData()
    conf = Configuration(
        model,
        data,
        np.array([0.0, 0, 0, 0.2, 0, 0]),
        collision_model=gm,
        collision_data=gm.createData(),
    )
    fast.compute_barrier(conf)  # fine: pair count matches the cache
    gm.removeAllCollisionPairs()  # cache is now stale
    conf2 = Configuration(
        model,
        data,
        np.array([0.0, 0, 0, 0.2, 0, 0]),
        collision_model=gm,
        collision_data=gm.createData(),
    )
    raised = False
    try:
        fast.compute_barrier(conf2)
    except ValueError:
        raised = True
    assert raised


def test_near_tie_selection_matches_stock():
    """Near-tied clearances at the selection boundary still select coal's set.

    A left-right mirror pose makes (body,left)/(body,right) pairs share clearances,
    stressing the argpartition tie boundary. Because the fast clearance is bit-equal to
    coal's min_distance, the deterministic selection picks the same set (and rows match).
    """
    ik, fast, stock = _make()
    q = ik.nominal_q().copy()
    for ln, rn in zip(LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, strict=True):
        q[ik._idx_q[rn]] = -q[ik._idx_q[ln]]  # noqa: SLF001 -- enforce sagittal mirror
    ik.configuration.update(q)
    assert _selected_set(fast, ik.configuration) == _stock_selected_set(ik)
    np.testing.assert_allclose(
        fast.compute_jacobian(ik.configuration),
        stock.compute_jacobian(ik.configuration),
        atol=_ATOL_J,
    )


# --- equivalence: real robot joint trajectories --------------------------------


def test_real_robot_data_equivalence():
    """Frame-by-frame equivalence on recorded teleop joint trajectories.

    Loads ``obs/joint/*`` (+ ``obs/base/pose``) from the raw_data HDF5 takes on this
    workstation and asserts the Tier 2 barrier reproduces stock ``(h, J)`` on every
    sampled frame. No-ops when the data or h5py is absent (covered live by
    ``scripts/diagnostics/verify_fast_barrier_real_data.py`` instead).
    """
    try:
        import h5py  # noqa: PLC0415 -- optional dep, import-or-skip kept local
    except ImportError:
        return
    files = sorted(RAW_DATA_DIR.glob("*.hdf5")) if RAW_DATA_DIR.exists() else []
    if not files:
        return
    ik, fast, stock = _make()
    checked = 0
    for path in files:
        with h5py.File(path, "r") as f:
            if "obs/joint/torso" not in f:
                continue
            torso = f["obs/joint/torso"][:]
            larm = f["obs/joint/left_arm"][:]
            rarm = f["obs/joint/right_arm"][:]
            head = f["obs/joint/head"][:]
            base = f["obs/base/pose"][:] if "obs/base/pose" in f else None
            n = torso.shape[0]
            for t in range(0, n, max(1, n // 20)):  # ~20 frames per take
                q = _joint_q(
                    ik,
                    torso=torso[t],
                    left_arm=larm[t],
                    right_arm=rarm[t],
                    head=head[t],
                    base_pose=None if base is None else base[t],
                )
                ik.configuration.update(q)
                np.testing.assert_allclose(
                    fast.compute_barrier(ik.configuration),
                    stock.compute_barrier(ik.configuration),
                    atol=_ATOL_H,
                )
                np.testing.assert_allclose(
                    fast.compute_jacobian(ik.configuration),
                    stock.compute_jacobian(ik.configuration),
                    atol=_ATOL_J,
                )
                checked += 1
    assert checked > 0
