"""Whole-body differential IK for the Vega mobile manipulator.

Pink and Pinocchio coordinate a planar base, torso, head, and arms. The QP includes
joint limits, sphere-based collision barriers, and a CoM-over-base constraint.
A reactive safety gate rejects steps that worsen unsafe clearance or stability.
Control modes, task weights, and nominal posture are configured in wbik.yaml.
"""

from __future__ import annotations

import warnings as _warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pinocchio as pin
import qpsolvers
import yaml
from pink import Configuration, build_ik
from pink.barriers import SelfCollisionBarrier
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from pink.utils import get_root_joint_dim
from scipy.spatial import ConvexHull

try:
    import coal as _coal
except ImportError:  # pragma: no cover - compatibility with older Pinocchio stacks
    import hppfcl as _coal

from omniteleop.follower import wbc_safety
from omniteleop.follower.base_closed_loop import project_planar_twist_single_axis
from omniteleop.follower.fast_self_collision_barrier import FastSelfCollisionBarrier
from omniteleop.follower.gripper_collision import enclose_interval
from omniteleop.follower.wbc_safety import SafetyGate, collision_group

# Load solver defaults from the canonical YAML, shared with wbc_safety.
DEFAULT_CONFIG_PATH = Path(__file__).with_name("wbik.yaml")

# Other loaders own these YAML sections; exclude them from WBCConfig validation.
_NON_WBC_SECTIONS = frozenset({"vr_teleop", "joystick_teleop", "joystick_torso"})


def _load_config_yaml(path: Union[str, Path]) -> Dict:
    """Load a WBC YAML config into a plain dict (must be a top-level mapping)."""
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping, got {type(data).__name__}")
    # Resolve package placeholders and paths relative to this YAML, never the caller's cwd.
    for _key in ("urdf_path", "collision_spheres_urdf"):
        if isinstance(data.get(_key), str):
            value = data[_key]
            if value.startswith("${omniteleop}"):
                import omniteleop

                pkg_root = Path(omniteleop.__file__).parent
                value = str(pkg_root) + value[len("${omniteleop}") :]
            if value.startswith("${dexmate_urdf}"):
                import dexmate_urdf

                pkg_root = Path(dexmate_urdf.__file__).parent
                value = str(pkg_root) + value[len("${dexmate_urdf}") :]
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = config_path.parent / resolved
            data[_key] = str(resolved.resolve())
    return data


_DEFAULTS: Dict = _load_config_yaml(DEFAULT_CONFIG_PATH)

# Vega model constants (vega_no_effector.urdf convention)
DEFAULT_URDF = _DEFAULTS["urdf_path"]

WHEEL_JOINTS = [
    "B_wheel_j1",
    "B_wheel_j2",
    "R_wheel_j1",
    "R_wheel_j2",
    "L_wheel_j1",
    "L_wheel_j2",
]
TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]
LEFT_ARM_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
HEAD_JOINTS = ["head_j1", "head_j2", "head_j3"]

LEFT_EE_FRAME = "L_ee"
RIGHT_EE_FRAME = "R_ee"
LEFT_GRIPPER_FRAME = "L_robotiq"
RIGHT_GRIPPER_FRAME = "R_robotiq"
LEFT_WRIST_CAM_FRAME = "L_wrist_zed_mini"
RIGHT_WRIST_CAM_FRAME = "R_wrist_zed_mini"
HEAD_FRAME = "zed_depth_frame"
BASE_FRAME = "base"
BASE_DOF_MODES = ("xy_yaw", "xy")
# Torso-top x anchor frame: "arm_center" is the fixed mount at the top of torso_l3
# (torso_j3 is the link's lower joint; head_j1 attaches right next to it), carrying
# both arms and the head chain. The QP holds its base-frame x at the value it has at
# the config nominal posture (see solve()); its height -- and torso_j3 itself -- stay
# free, so the torso can still crouch and fold.
TORSO_TOP_FRAME = "arm_center"

# Open Robotiq proxies in each gripper link frame; +z points toward the fingertips.
# Fit tapered fingertips with padded half-slab spheres that cover both supported tip meshes.
ROBOTIQ_FINGERTIP_OPEN_SLABS = (
    (-0.08929, -0.07073, 0.08250, 0.01900),
    (-0.07073, -0.05217, 0.08250, 0.01900),
    (-0.05217, -0.03361, 0.08250, 0.01900),
    (-0.03361, -0.01505, 0.07557, 0.01290),
    (-0.01505, 0.00351, 0.07028, 0.01290),
    (0.00351, 0.02207, 0.06340, 0.01290),
    (0.02207, 0.04063, 0.05472, 0.01290),
)


def _robotiq_fingertip_proxy_spheres():
    """Enclose each OPEN fingertip half-slab with 2 mm padded bounding spheres.

    The previous sweep spanned x=0..x_max; the installed open mesh starts at
    x=0.0425. Runtime stroke envelopes are built from these tighter rest spheres.
    """
    spheres = []
    for row, (z_min, z_max, x_max, y_half) in enumerate(ROBOTIQ_FINGERTIP_OPEN_SLABS):
        x_min = 0.0425
        radius = float(np.linalg.norm([(x_max - x_min) / 4, y_half, (z_max - z_min) / 2])) + 0.002
        for column, fraction in enumerate((0.25, 0.75)):
            for side, sign, y in (("pos", 1, -0.00008), ("neg", -1, 0.00077)):
                xyz = (sign * (x_min + fraction * (x_max - x_min)), y, (z_min + z_max) / 2)
                spheres.append((f"tip_{row}_{column}_{side}", xyz, radius))
    return tuple(spheres)


ROBOTIQ_PROXY_SPHERES: tuple[tuple[str, tuple[float, float, float], float], ...] = (
    ("palm", (0.0, 0.0, -0.205), 0.055),
    ("finger_pos", (0.050, 0.0, -0.125), 0.055),
    ("finger_neg", (-0.050, 0.0, -0.125), 0.055),
) + _robotiq_fingertip_proxy_spheres()

# Fixed spheres cover the full opening sweep without measured-width updates.
ROBOTIQ_FIXED_PROXY_SPHERES = ROBOTIQ_PROXY_SPHERES[:3] + (
    ("tip_root_pos", (0.041, 0.0, -0.068), 0.055),
    ("tip_root_neg", (-0.041, 0.0, -0.068), 0.055),
    ("tip_mid_pos", (0.041, 0.0, -0.014), 0.055),
    ("tip_mid_neg", (-0.041, 0.0, -0.014), 0.055),
    ("tip_end_pos", (0.041, 0.0, 0.040), 0.055),
    ("tip_end_neg", (-0.041, 0.0, 0.040), 0.055),
)

# Wrist-camera proxy spheres in each L/R_wrist_zed_mini frame;
# they move with the arm and supplement the body-only collision-sphere URDF.
WRIST_CAM_PROXY_SPHERES: tuple[tuple[str, tuple[float, float, float], float], ...] = (
    ("lens_pos", (0.040, 0.009, 0.0), 0.026),
    ("body", (-0.002, 0.009, 0.0), 0.032),
    ("lens_neg", (-0.044, 0.008, 0.0), 0.032),
)


def _phone_mount_proxy_spheres() -> tuple[tuple[str, tuple[float, float, float], float], ...]:
    """Cover the installed mount's two solid envelopes with small overlapping spheres.

    Base-frame metres, from hardware/vega_iphone_mount/07_estop_tower_cradle.stl
    and mount_pose.json. The wide pedestal ends at z=334 mm; above it the cradle,
    cased phone and top stop occupy a thin 19 mm slab. The stop seats at z=524.8
    and extends to 532.8 mm. Fill both envelopes, including the phone pocket.

    Each sphere circumscribes a rectangular cell plus 2 mm of assembly padding.
    This covers the entire volume (not just sampled mesh vertices), while avoiding
    the old 75 mm-radius column's large bulge toward the left elbow. All geometry
    stays spherical for the shared QP barrier, safety gate, and distance probes.
    """
    envelopes = (
        ("base", (0.109, 0.190, 0.2817), (0.189, 0.290, 0.334), (2, 3, 2)),
        ("cradle", (0.134, 0.190, 0.334), (0.153, 0.290, 0.5328), (1, 3, 7)),
    )
    spheres = []
    for part, lower, upper, count in envelopes:
        lower = np.asarray(lower)
        step = (np.asarray(upper) - lower) / np.asarray(count)
        radius = float(np.linalg.norm(step / 2) + 0.002)
        for cell in np.ndindex(*count):
            center = lower + (np.asarray(cell) + 0.5) * step
            label = f"phone_mount_{part}_{cell[0]}_{cell[1]}_{cell[2]}"
            spheres.append((label, tuple(float(v) for v in center), radius))
    return tuple(spheres)


PHONE_MOUNT_PROXY_SPHERES = _phone_mount_proxy_spheres()

# Nominal URDF-convention posture; defaults come from wbik.yaml.
DEFAULT_NOMINAL_POSTURE: Dict[str, float] = dict(_DEFAULTS["nominal_posture"])

Pose = Union[np.ndarray, pin.SE3]


def _as_se3(pose: Pose) -> pin.SE3:
    """Accept a 4x4 homogeneous matrix or a ``pin.SE3`` and return ``pin.SE3``."""
    if isinstance(pose, pin.SE3):
        return pose
    arr = np.asarray(pose, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix or pin.SE3, got shape {arr.shape}")
    return pin.SE3(arr[:3, :3], arr[:3, 3])


def _support_polygon_from_model(model: pin.Model) -> np.ndarray:
    """Wheel ground-contact footprint (CCW polygon, base-frame xy) for tip-over checks.

    Computed from the full (un-reduced) model at the neutral configuration, where the
    planar base sits at the origin so the world frame coincides with the base frame.
    """
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    pts = np.array(
        [
            data.oMf[fid].translation[:2]
            for fid, frame in enumerate(model.frames)
            if "wheel" in frame.name.lower()
        ]
    )
    return pts[ConvexHull(pts).vertices]  # ConvexHull orders 2D vertices CCW


def _apply_collision_sphere_mesh_origin_offsets(
    robot_urdf_path: str,
    collision_spheres_urdf_path: str,
    sphere_model: pin.GeometryModel,
) -> None:
    """Translate sphere placements when their source URDF mesh origins differ."""
    offsets = wbc_safety.collision_sphere_mesh_origin_offsets(
        robot_urdf_path, collision_spheres_urdf_path
    )
    if not offsets:
        return
    for geom in sphere_model.geometryObjects:
        link_name = geom.name.rsplit("_", 1)[0]
        offset = offsets.get(link_name)
        if offset is not None:
            geom.placement.translation += offset


def _signed_dist_to_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> float:
    """Signed distance (m) from ``point`` to a CCW convex ``polygon`` boundary.

    Positive when the point is inside, negative when outside (the minimum over the
    edges of the inward-normal distance).
    """
    margin = np.inf
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        edge = polygon[(i + 1) % n] - a
        length = float(np.hypot(edge[0], edge[1]))
        if length < 1e-9:
            continue
        # Inward normal for a CCW polygon is (-edge_y, edge_x) / length.
        dist = (-edge[1] * (point[0] - a[0]) + edge[0] * (point[1] - a[1])) / length
        margin = min(margin, dist)
    return float(margin)


@dataclass
class WBCConfig:
    """Tunable parameters for :class:`VegaWholeBodyIK`.

    The default values below are **loaded from** the canonical ``wbik.yaml`` next to
    this module (see ``_DEFAULTS``) -- that file, not this dataclass, is the single
    source of truth for the values. This class is the typed schema and override surface:
    ``WBCConfig()`` reads the YAML, ``WBCConfig(ee_position_cost=...)`` overrides a key,
    and ``WBCConfig.from_yaml(path)`` loads an alternate file.

    Task-cost magnitudes preserve the priority hierarchy of the reference
    (end-effector >> posture >> base regularization).
    """

    urdf_path: str = DEFAULT_URDF

    # End-effector tracking (highest priority). Position and orientation weighted equally
    # (deps/rby1-wbc: ee_pos_cost == ee_ori_cost). The absolute magnitude (10000) is the
    # reference's literal value; it is only a *gauge* -- Pink's QP depends solely on the
    # ratios of the other costs to this one, so scaling every cost together is a no-op.
    # What matters is that base/posture/current below are set as ratios of it.
    ee_position_cost: float = _DEFAULTS["ee_position_cost"]
    ee_orientation_cost: float = _DEFAULTS["ee_orientation_cost"]
    lm_damping_ee: float = _DEFAULTS["lm_damping_ee"]

    # Damp planar base velocity by re-targeting its task to the current pose.
    # z/roll/pitch are fixed structurally by the planar root.
    base_position_cost: Sequence[float] = tuple(_DEFAULTS["base_position_cost"])
    base_orientation_cost: Sequence[float] = tuple(_DEFAULTS["base_orientation_cost"])
    lm_damping_base: float = _DEFAULTS["lm_damping_base"]  # reference's base_ground lm_damping

    # Penalize changes in (vx, vy, wz), leaving steady base velocity unpenalized.
    base_velocity_smoothing_cost: Sequence[float] = tuple(_DEFAULTS["base_velocity_smoothing_cost"])

    # Planar base DOF policy. "xy_yaw" leaves the full planar root available
    # (x, y, yaw). "xy" hard-pins the WBC yaw tangent DOF to zero, so the solver may
    # translate the base but cannot rotate it to satisfy EE/head tasks.
    base_dofs: str = _DEFAULTS["base_dofs"]

    # Pin all planar-root velocities in IK when the chassis is joystick-driven.
    # The base remains in FK, collision, and stability calculations.
    lock_base_in_ik: bool = False

    # Pin torso velocities while retaining torso FK/collision geometry.
    # External torso commands are supplied through solve(torso_joints=...).
    lock_torso_in_ik: bool = _DEFAULTS["lock_torso_in_ik"]

    # Allow only the dominant normalized base axis; apply before integration.
    enable_base_single_axis: bool = _DEFAULTS["enable_base_single_axis"]
    # Quiet floor on the dominant base axis, as a fraction of the base velocity cap
    # (|vx|,|vy| / base_xy_max_vel and |wz| / base_yaw_max_vel compare on this common scale):
    # below it the chassis is held still (all three axes zeroed). This is the solver's "is the
    # base meant to move at all" gate, distinct from the follower-loop vr_teleop deadbands
    # that shape the already-chosen command.
    base_single_axis_deadband: float = _DEFAULTS["base_single_axis_deadband"]
    # Follower-side quiet floor applied after measured-base PD and slew limiting. Keeping
    # it below the solver intent gate lets an accepted command ramp without being zeroed.
    base_dispatch_single_axis_deadband: float = _DEFAULTS["base_dispatch_single_axis_deadband"]
    # Relative hysteresis on the axis choice: the previously active axis is retained unless a
    # different axis's normalized magnitude beats it by this fraction, so two near-equal axes
    # do not chatter the chassis between (say) forward and turn at 100 Hz. 0 => per-tick argmax.
    base_single_axis_hysteresis_ratio: float = _DEFAULTS["base_single_axis_hysteresis_ratio"]

    # Pull redundant joints toward nominal posture with per-group weights.
    posture_cost_torso: float = _DEFAULTS["posture_cost_torso"]
    posture_cost_arm: float = _DEFAULTS["posture_cost_arm"]
    posture_cost_head: float = _DEFAULTS["posture_cost_head"]

    # Re-target a second posture task to current joints for velocity damping.
    current_posture_cost_main: float = _DEFAULTS["current_posture_cost_main"]
    current_posture_cost_head: float = _DEFAULTS["current_posture_cost_head"]

    # Head tracking uses solve_head() and QP pins; whole-body head tasks are
    # created only in head_mode="ik".

    # Head-tracking convergence gain (1/s) and damped-LS regularization.
    # The caller smooths target commands; body-motion compensation stays immediate.
    head_ik_gain: float = _DEFAULTS["head_ik_gain"]
    # Damped-least-squares regularization of the solve_head step. Pure conditioning
    # (the smoothing above comes from head_ik_gain): keeps the 2-DOF step bounded
    # when the pan/tilt axes degenerate against the orientation error near the
    # joint-range extremes.
    head_ik_damp: float = _DEFAULTS["head_ik_damp"]

    # Head modes: track uses external head joints, ik uses a Cartesian QP task,
    # and fixed holds nominal joints. Cartesian target orientation and position
    # are configured separately below.
    head_mode: str = _DEFAULTS["head_mode"]
    # Head FrameTask weights for head_mode "ik" (ignored in "track"). Follow rby1
    # head_pos_cost / head_ori_cost (10000). Roll about the camera forward axis is
    # unactuatable (planar-yaw base + pitch torso + pan/tilt head); the QP just leaves
    # it as residual (rby1 keeps roll weighted too), so zero the roll axis only if that
    # residual ever biases the solve.
    head_position_cost: float = _DEFAULTS["head_position_cost"]
    head_orientation_cost: Sequence[float] = tuple(_DEFAULTS["head_orientation_cost"])
    # Head-position cost in gravity-aligned world axes; zero weights free axes.
    # Unlike optical-frame weights, freeing z does not couple into fore/aft motion.
    head_world_position_cost: Sequence[float] = tuple(_DEFAULTS["head_world_position_cost"])
    # Head joints pinned in ik mode; unlisted joints remain QP variables.
    # Track/fixed modes pin all head joints independently of this list.
    head_ik_pinned_joints: Sequence[str] = tuple(_DEFAULTS["head_ik_pinned_joints"])

    # Hold arm_center at its nominal base-frame x with a hard equality.
    # Torso height and pitch remain free unless lock_torso_in_ik is enabled.
    enable_torso_top_x_anchor: bool = _DEFAULTS["enable_torso_top_x_anchor"]
    # Anchor recovery gain (1/s), clipped to half the row velocity headroom
    # to avoid making the QP infeasible during measured-state recovery.
    torso_top_x_anchor_gain: float = _DEFAULTS["torso_top_x_anchor_gain"]

    # Velocity limits. Arm/torso/head limits come from the URDF; the planar base
    # is unbounded in the URDF, so we cap it here (matches reference base limits).
    base_xy_max_vel: float = _DEFAULTS["base_xy_max_vel"]  # m/s
    base_yaw_max_vel: float = _DEFAULTS["base_yaw_max_vel"]  # rad/s
    # Global headroom factor applied to every velocity limit (base + joints), matching
    # the reference's ``velocity_limit_scale`` -- keeps a margin below the hard limits.
    velocity_limit_scale: float = _DEFAULTS["velocity_limit_scale"]

    # QP solver. ``daqp`` matches the reference (deps/rby1-wbc); the solve path mirrors
    # it too -- build_ik -> augment with the CoM-over-base inequality -> solve_problem.
    solver: str = _DEFAULTS["solver"]
    damping: float = _DEFAULTS["damping"]

    # Safety (for real-robot use); see omniteleop.follower.wbc_safety
    # Self-collision: a proactive SelfCollisionBarrier over the Dexmate collision-sphere
    # model plus a reactive hold gate (the hard guarantee). Tip-over: enforced
    # *proactively* as a hard QP inequality (the reference's
    # _add_com_over_base_xy_inequalities approach) that bounds a centroid proxy's
    # horizontal over-base offset within +/-com_safety_margin of its nominal value, so
    # the robot keeps moving while staying upright instead of stopping after the fact.
    # The barrier degrades gracefully to off (with a warning) if pink lacks barriers or
    # the sphere URDF is unavailable.
    enable_collision_avoidance: bool = _DEFAULTS["enable_collision_avoidance"]
    # Optional measured-width grippers and base-mounted phone envelope.
    dynamic_gripper_collision: bool = _DEFAULTS["dynamic_gripper_collision"]
    phone_mount_collision: bool = _DEFAULTS["phone_mount_collision"]
    enable_com_safety: bool = _DEFAULTS["enable_com_safety"]  # hard CoM-over-base inequality
    collision_spheres_urdf: str = _DEFAULTS["collision_spheres_urdf"]
    self_collision_safe_dist: float = _DEFAULTS["self_collision_safe_dist"]
    self_collision_floor: float = _DEFAULTS["self_collision_floor"]
    # Reactive WARN distance (m) for the gate/HUD only. Decoupled from the barrier's
    # security distance (self_collision_safe_dist).
    # Set above safe_dist to surface the approach as a warning (no hold added).
    self_collision_warn_dist: float = _DEFAULTS["self_collision_warn_dist"]
    nominal_pair_keep_dist: float = _DEFAULTS["nominal_pair_keep_dist"]
    n_collision_pairs: int = _DEFAULTS["n_collision_pairs"]
    collision_barrier_gain: float = _DEFAULTS["collision_barrier_gain"]
    # Zero safe-displacement gain keeps collision avoidance purely constrained;
    # a nonzero value adds Pink's repulsive objective near obstacles.
    collision_safe_displacement_gain: float = _DEFAULTS["collision_safe_displacement_gain"]
    # Centroid proxy whose horizontal over-base offset the tip-over inequality bounds:
    # top link frame "torso_l3", or "com" for the true whole-body centre of mass. The
    # torso-link frame origin is a jacobian-linearized lean indicator, exactly like the
    # reference; "com" swaps in the real CoM (jacobianCenterOfMass).
    com_centroid: str = _DEFAULTS["com_centroid"]
    # Half-width (m) of the box the centroid proxy's over-base offset may deviate from
    # its nominal value before the QP inequality clamps it (follows the reference's
    # com_over_base_xy_bounds [0.08, 0.08]).
    com_safety_margin: float = _DEFAULTS["com_safety_margin"]
    # Soft centroid-centering cost in base-frame xy; the hard stability box
    # remains active and vertical torso motion stays unpenalized.
    com_over_base_pos_cost: float = _DEFAULTS["com_over_base_pos_cost"]

    nominal_posture: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NOMINAL_POSTURE))

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "WBCConfig":
        """Load a config from a YAML file, the way the reference does.

        ``path`` defaults to the canonical ``wbik.yaml`` -- the same file the in-code
        defaults are loaded from -- so ``from_yaml() == WBCConfig()``. Each key overrides
        its default (an alternate file may set only a subset); unknown keys raise
        ``ValueError`` (typo guard). Mirrors ``deps/rby1-wbc``'s YAML-driven config.
        """
        data = _load_config_yaml(DEFAULT_CONFIG_PATH if path is None else path)
        # Drop sibling sections owned by other loaders (e.g. vr_teleop) so they are not
        # mistaken for typo'd WBCConfig keys; the remaining keys must map 1:1 to fields.
        data = {k: v for k, v in data.items() if k not in _NON_WBC_SECTIONS}
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys {sorted(unknown)}")
        # YAML sequences -> tuples, matching the dataclass defaults (which must be
        # immutable), so ``from_yaml() == WBCConfig()`` holds exactly.
        data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items()}
        return cls(**data)


@dataclass
class WBCResult:
    """Output of one :meth:`VegaWholeBodyIK.solve` call."""

    q: np.ndarray  # full configuration (nq,)
    velocity: np.ndarray  # joint velocity from the QP (nv,)
    base_pose: np.ndarray  # (x, y, yaw) of the mobile base, in world frame
    base_twist: np.ndarray  # (vx, vy, wz) -> maps to chassis velocity command
    torso: np.ndarray  # (3,)
    left_arm: np.ndarray  # (7,)
    right_arm: np.ndarray  # (7,)
    head: np.ndarray  # (3,) -- head_mode "track": echoes the external
    # head_joints command (head pinned, not an IK DOF).
    # head_mode "ik": the QP-solved head joints (see solve()).
    left_ee_error: float  # meters, target vs. achieved L_ee position
    right_ee_error: float  # meters
    success: bool  # False if the QP solve failed (configuration held)
    stability_margin: float  # signed dist (m) of true CoM ground-projection to the
    # wheel support polygon (diagnostic); the tip-over QP
    # bound keeps the robot upright so this stays positive
    # safety (see omniteleop.follower.wbc_safety)
    held: bool = False  # True => step vetoed; robot holds this pose
    min_self_distance: float = float("inf")  # closest cross-group self-collision dist (m)
    safety_status: str = "ok"  # "ok", or a "WARN:"/"HELD:" reason string


class VegaWholeBodyIK:
    """Whole-body differential IK solver for the Vega mobile manipulator.

    Typical use (kinematic loop)::

        ik = VegaWholeBodyIK()
        ik.reset()                                   # start from nominal posture
        left0 = ik.frame_pose(LEFT_EE_FRAME)         # current EE poses as targets
        right0 = ik.frame_pose(RIGHT_EE_FRAME)
        while True:
            result = ik.solve(left_target, right_target, dt)
            display(result.q)                        # e.g. Meshcat / SAPIEN
    """

    def __init__(
        self,
        config: Optional[WBCConfig] = None,
        *,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("pass either config or config_path, not both")
        if config is None and config_path is not None:
            config = WBCConfig.from_yaml(config_path)
        self.config = config or WBCConfig()
        if self.config.head_mode not in ("track", "ik", "fixed"):
            raise ValueError(
                f"head_mode must be 'track', 'ik' or 'fixed', got {self.config.head_mode!r}"
            )
        if self.config.base_dofs not in BASE_DOF_MODES:
            raise ValueError(
                f"base_dofs must be one of {BASE_DOF_MODES}, got {self.config.base_dofs!r}"
            )
        if self.config.solver not in qpsolvers.available_solvers:
            available = ", ".join(qpsolvers.available_solvers) or "none"
            raise RuntimeError(
                f"configured QP solver {self.config.solver!r} is not available to "
                f"qpsolvers (available: {available}). Install the solver package in "
                "this Python environment, or change the solver in "
                f"{DEFAULT_CONFIG_PATH}."
            )
        self._head_mode = self.config.head_mode
        self._build_model()
        self._build_tasks_and_limits()
        self.reset()

    # construction
    @staticmethod
    def _add_proxy_collision_spheres(
        model: pin.Model,
        geometry_model: pin.GeometryModel,
        *,
        dynamic_gripper_collision: bool = False,
    ) -> None:
        """Append selected Robotiq and fixed wrist-camera proxies, per frame present."""
        existing = {obj.name for obj in geometry_model.geometryObjects}
        gripper_spheres = (
            ROBOTIQ_PROXY_SPHERES if dynamic_gripper_collision else ROBOTIQ_FIXED_PROXY_SPHERES
        )
        for frame_name, spheres in (
            (LEFT_GRIPPER_FRAME, gripper_spheres),
            (RIGHT_GRIPPER_FRAME, gripper_spheres),
            (LEFT_WRIST_CAM_FRAME, WRIST_CAM_PROXY_SPHERES),
            (RIGHT_WRIST_CAM_FRAME, WRIST_CAM_PROXY_SPHERES),
        ):
            if not model.existFrame(frame_name):
                continue
            if any(name.startswith(frame_name) for name in existing):
                continue
            frame_id = model.getFrameId(frame_name)
            frame = model.frames[frame_id]
            for label, xyz, radius in spheres:
                name = f"{frame_name}_{label}"
                local_center = pin.SE3(np.eye(3), np.asarray(xyz, dtype=float))
                geom = pin.GeometryObject(
                    name,
                    frame.parentJoint,
                    frame_id,
                    frame.placement * local_center,
                    _coal.Sphere(radius),
                )
                geometry_model.addGeometryObject(geom)
                existing.add(name)

    @staticmethod
    def _add_phone_mount_spheres(
        model: pin.Model,
        geometry_model: pin.GeometryModel,
    ) -> None:
        """Append PHONE_MOUNT_PROXY_SPHERES to the base (the "body" collision group)."""
        frame_id = model.getFrameId(BASE_FRAME)
        frame = model.frames[frame_id]
        for label, xyz, radius in PHONE_MOUNT_PROXY_SPHERES:
            local_center = pin.SE3(np.eye(3), np.asarray(xyz, dtype=float))
            geometry_model.addGeometryObject(
                pin.GeometryObject(
                    f"{BASE_FRAME}_{label}",
                    frame.parentJoint,
                    frame_id,
                    frame.placement * local_center,
                    _coal.Sphere(radius),
                )
            )

    def _build_model(self) -> None:
        cfg = self.config
        urdf = cfg.urdf_path
        package_dir = str(Path(urdf).parent)

        # Planar root joint => mobile base (x, y, yaw) is part of the chain.
        model_full = pin.buildModelFromUrdf(urdf, pin.JointModelPlanar())
        visual_full = pin.buildGeomFromUrdf(
            model_full, urdf, pin.GeometryType.VISUAL, package_dirs=package_dir
        )
        collision_full = pin.buildGeomFromUrdf(
            model_full, urdf, pin.GeometryType.COLLISION, package_dirs=package_dir
        )

        # Self-collision geometry: the Dexmate collision-sphere model, augmented with
        # moving gripper proxies when the URDF has Robotiq links. Loaded as a separate
        # geometry model so the SelfCollisionBarrier uses clean sphere-sphere distances.
        # Optional/graceful.
        sphere_full = None
        if cfg.enable_collision_avoidance and SelfCollisionBarrier is not None:
            try:
                sphere_full = pin.buildGeomFromUrdf(
                    model_full, cfg.collision_spheres_urdf, pin.GeometryType.COLLISION
                )
                _apply_collision_sphere_mesh_origin_offsets(
                    urdf, cfg.collision_spheres_urdf, sphere_full
                )
            except Exception as exc:  # missing URDF / no hpp-fcl: disable, warn
                _warnings.warn(
                    f"self-collision avoidance disabled (could not load "
                    f"{cfg.collision_spheres_urdf}: {exc})",
                    stacklevel=2,
                )

        # Lock the passive wheel joints; they are not whole-body DOFs.
        lock_ids = [model_full.getJointId(j) for j in WHEEL_JOINTS if model_full.existJointName(j)]
        q_ref = pin.neutral(model_full)
        geom_models = [visual_full, collision_full]
        if sphere_full is not None:
            geom_models.append(sphere_full)
        self.model, reduced_geoms = pin.buildReducedModel(model_full, geom_models, lock_ids, q_ref)
        self.visual_model, self.collision_model = reduced_geoms[0], reduced_geoms[1]
        self.collision_sphere_model = reduced_geoms[2] if sphere_full is not None else None
        if self.collision_sphere_model is not None:
            self._add_proxy_collision_spheres(
                self.model,
                self.collision_sphere_model,
                dynamic_gripper_collision=cfg.dynamic_gripper_collision,
            )
            if cfg.phone_mount_collision:
                self._add_phone_mount_spheres(self.model, self.collision_sphere_model)

        # Cap mobile-base velocity (planar nv layout: 0=vx, 1=vy, 2=yaw-rate). Validate the
        # caps (and the headroom scale) here, before they enter both the QP VelocityLimit and
        # the single-axis projector's per-axis normalization (which would otherwise divide by a
        # zero/NaN cap).
        for name, value in (
            ("base_xy_max_vel", cfg.base_xy_max_vel),
            ("base_yaw_max_vel", cfg.base_yaw_max_vel),
            ("velocity_limit_scale", cfg.velocity_limit_scale),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value!r}")
        self.model.velocityLimit[0] = cfg.base_xy_max_vel
        self.model.velocityLimit[1] = cfg.base_xy_max_vel
        self.model.velocityLimit[2] = cfg.base_yaw_max_vel
        # Base-DOF availability in the QP. lock_base_in_ik pins ALL three planar-root
        # DOFs (base excluded from the whole-body IK -- joystick-teleop mode); otherwise
        # base_dofs "xy" pins only yaw and "xy_yaw" leaves the full planar root free. The
        # disabled DOFs below become QP equality rows (dq = 0) via _base_dof_pin_A and are
        # zeroed in the emitted twist by _apply_base_dof_mask, so a locked base has an
        # identically-zero base_twist and a fixed base_pose.
        self._base_dof_mask = np.ones(3, dtype=bool)
        if cfg.lock_base_in_ik:
            self._base_dof_mask[:] = False
        elif cfg.base_dofs == "xy":
            self._base_dof_mask[2] = False
        disabled_base = np.flatnonzero(~self._base_dof_mask)
        self._base_dof_pin_A = np.zeros((len(disabled_base), self.model.nv))
        for row, idx_v in enumerate(disabled_base):
            self._base_dof_pin_A[row, idx_v] = 1.0
        # Global velocity headroom (reference's velocity_limit_scale): scale every
        # limit -- base and URDF joint limits alike -- before VelocityLimit reads them.
        self.model.velocityLimit *= cfg.velocity_limit_scale

        self.data = self.model.createData()

        # Tip-over stability check: a CoM data buffer (separate from the Pink
        # configuration's data so it isn't disturbed) and the wheel support polygon
        # (from the full model, since the reduced model merges the locked wheels).
        self._com_data = self.model.createData()
        self._support_polygon = _support_polygon_from_model(model_full)

        # joint name -> first index in q (used to set posture and read out groups)
        self._idx_q: Dict[str, int] = {
            self.model.names[j]: self.model.idx_qs[j] for j in range(1, self.model.njoints)
        }

        # Fail fast on nominal_posture typos: nominal_q() applies only the keys present
        # in the reduced model, so a misspelled joint (e.g. "head_i1") would otherwise
        # silently no-op -- and the follower would home/anchor to a posture the operator
        # never configured. Every legitimate key (torso, arms, head) survives the wheel
        # lock above, so an unknown key is always an error.
        unknown_nominal = sorted(set(self.config.nominal_posture) - set(self._idx_q))
        if unknown_nominal:
            raise ValueError(
                f"nominal_posture keys {unknown_nominal} are not joints of the reduced "
                f"model (valid: {sorted(self._idx_q)})"
            )

        # Optional torso exclusion: retain the joints in the model so their fixed pose
        # still feeds arm/head FK and the safety geometry, but remove them as QP levers
        # with one hard dq = 0 row per joint. The explicit velocity mask below makes the
        # fixed-joint invariant exact even at the QP solver's numerical tolerance.
        self._torso_idx_v = [
            self.model.idx_vs[self.model.getJointId(name)] for name in TORSO_JOINTS
        ]
        self._torso_idx_q = [
            self.model.idx_qs[self.model.getJointId(name)] for name in TORSO_JOINTS
        ]
        n_torso_pins = len(self._torso_idx_v) if cfg.lock_torso_in_ik else 0
        self._torso_dof_pin_A = np.zeros((n_torso_pins, self.model.nv))
        for row, idx_v in enumerate(self._torso_idx_v[:n_torso_pins]):
            self._torso_dof_pin_A[row, idx_v] = 1.0

        # The head is teleoperated by its own damped IK (solve_head -> head_pos
        # command fed back via solve(head_joints=...)) and is never a *whole-body* IK
        # DOF: cache the head q/tangent indices and the QP equality rows that pin
        # dq_head = 0 (see solve()). The head joints stay in the model -- rather than
        # being locked like the wheels -- so the commanded head pose feeds FK, the
        # collision spheres, and the CoM/stability checks.
        self._head_idx_q = [self._idx_q[n] for n in HEAD_JOINTS]
        self._head_idx_v = [self.model.idx_vs[self.model.getJointId(n)] for n in HEAD_JOINTS]
        # solve_head actuates only head_j2/j3, exactly like vr_reader's head IK
        # (_HEAD_IK_JOINTS = {head_j2, head_j3}); head_j1 rides at its current value.
        self._head_ik_idx_q = self._head_idx_q[1:]
        self._head_ik_idx_v = self._head_idx_v[1:]
        self._head_pin_A = np.zeros((len(self._head_idx_v), self.model.nv))
        for row, idx_v in enumerate(self._head_idx_v):
            self._head_pin_A[row, idx_v] = 1.0
        # head_mode "ik": the head is a whole-body IK DOF, but this equality matrix pins
        # the head DOFs named in cfg.head_ik_pinned_joints (wbik.yaml; default
        # head_j1/head_j2) out of the QP -- forcing their motion through the base/body while
        # leaving the unlisted head joints (default head_j3) as free IK DOFs. Built from the
        # config so both VR followers pin identically; one row per pinned joint.
        idx_v_by_head = dict(zip(HEAD_JOINTS, self._head_idx_v, strict=True))
        pinned = list(cfg.head_ik_pinned_joints)
        unknown_pin = [n for n in pinned if n not in idx_v_by_head]
        if unknown_pin:
            raise ValueError(
                f"head_ik_pinned_joints {unknown_pin} are not head joints {HEAD_JOINTS}"
            )
        self._head_j1_pin_A = np.zeros((len(pinned), self.model.nv))
        for row, name in enumerate(pinned):
            self._head_j1_pin_A[row, idx_v_by_head[name]] = 1.0

        # Torso-top x anchor: resolve the anchored frame; the anchor value itself
        # (the mount's base-frame x at the config nominal posture) is set in reset().
        self._torso_top_fid: Optional[int] = None
        self._torso_top_x_target: Optional[float] = None
        if cfg.enable_torso_top_x_anchor:
            if not self.model.existFrame(TORSO_TOP_FRAME):
                raise ValueError(
                    f"enable_torso_top_x_anchor: frame {TORSO_TOP_FRAME!r} not "
                    f"found in {cfg.urdf_path}"
                )
            self._torso_top_fid = self.model.getFrameId(TORSO_TOP_FRAME)

        # Self-collision pairs (after _idx_q, since the nominal filter needs it).
        self._collision_enabled = False
        self.collision_sphere_data = None
        # Vectorized all-sphere reactive-gate distance: pair index arrays + per-geometry
        # radii, set by _setup_collision_pairs when every geometry is a sphere. None =>
        # _min_self_distance falls back to the exact coal per-pair read.
        self._sc_pair_i: Optional[np.ndarray] = None
        self._sc_pair_j: Optional[np.ndarray] = None
        self._sc_radius: Optional[np.ndarray] = None
        # Scratch collision data for self_collision_distance() -- a side-effect-free
        # distance probe at an arbitrary q (e.g. homing waypoints), allocated lazily so
        # the live gate/barrier collision_data is never disturbed.
        self._probe_collision_data = None
        # Scratch model/geometry data for self_collision_pair_distances() (lazy, same reason).
        self._probe_model_data = None
        self._probe_geom_data = None
        self._gripper_collision = {}
        if self.collision_sphere_model is not None:
            if cfg.dynamic_gripper_collision:
                self._init_gripper_collision_geometry()
            self._setup_collision_pairs()

        # Resolve the tip-over centroid proxy (a torso frame, or the true CoM).
        self._resolve_com_centroid()

    def _init_gripper_collision_geometry(self) -> None:
        """Attach the calibrated stroke to the installed open fingertip proxies."""
        sm = self.collision_sphere_model
        names = {g.name: i for i, g in enumerate(sm.geometryObjects)}
        tips = [p for p in ROBOTIQ_PROXY_SPHERES if p[0].startswith("tip_")]
        for side, prefix in (("left", "L"), ("right", "R")):
            ee_name, grip_name = f"{prefix}_ee", f"{prefix}_robotiq"
            if not self.model.existFrame(grip_name) or not self.model.existFrame(ee_name):
                continue
            if not all(f"{grip_name}_{p[0]}" in names for p in tips):
                continue
            ee = self.model.frames[self.model.getFrameId(ee_name)]
            grip = self.model.frames[self.model.getFrameId(grip_name)]
            if ee.parentJoint != grip.parentJoint:
                continue  # This stroke is for fixed Robotiq mounts, not actuated URDF fingers.
            ee_to_grip = ee.placement.inverse() * grip.placement
            points = np.array([ee_to_grip.act(np.asarray(p[1])) for p in tips])
            self._gripper_collision[side] = dict(
                indices=[names[f"{grip_name}_{p[0]}"] for p in tips],
                points=points,
                signs=np.array([1 if p[0].endswith("pos") else -1 for p in tips]),
                radii=np.array([p[2] for p in tips]),
                parent_placement=ee.placement,
            )
        # Without feedback, cover every calibrated opening, including axial motion.
        self.update_gripper_collision_geometry()

    def update_gripper_collision_geometry(
        self,
        left: Optional[tuple[float, float]] = None,
        right: Optional[tuple[float, float]] = None,
    ) -> None:
        """Set achieved raw-encoder intervals, independently per gripper.

        None (or an invalid interval) covers the full calibrated stroke. Pair indices
        never change; update both distance caches and Pink's same-q barrier cache.
        """
        if not self.config.dynamic_gripper_collision:
            return
        changed = False
        sm = self.collision_sphere_model
        for side, interval in (("left", left), ("right", right)):
            state = self._gripper_collision.get(side)
            if state is None:
                continue
            interval = (0.0, 1.0) if interval is None else tuple(interval)
            if state.get("interval") == interval:
                continue
            centers, radii = enclose_interval(
                side, state["points"], state["signs"], state["radii"], interval
            )
            for i, center, radius in zip(state["indices"], centers, radii):
                geom = sm.geometryObjects[i]
                geom.placement = state["parent_placement"] * pin.SE3(np.eye(3), center)
                geom.geometry.radius = float(radius)
                geom.geometry.computeLocalAABB()
                if self._sc_radius is not None:
                    self._sc_radius[i] = radius
            state["interval"] = interval
            changed = True
        if changed and hasattr(self, "configuration"):
            if isinstance(self.collision_barrier, FastSelfCollisionBarrier):
                self.collision_barrier.refresh_geometry(sm)
            elif self.collision_barrier is not None:
                # Stock Pink has no public invalidation API for geometry changes.
                self.collision_barrier._Barrier__q_cache = None
            self.configuration.update(self.configuration.q)

    def _setup_collision_pairs(self) -> None:
        """Add cross-group sphere collision pairs, dropping nominal overlaps.

        Pairs only the body / left-arm / right-arm groups against each other (never
        within a chain). Pairs that are closer than ``nominal_pair_keep_dist`` at the
        nominal posture -- the by-design shoulder-vs-torso overlaps -- are removed, so
        the barrier starts feasible and the reactive monitor doesn't false-trip.
        """
        cfg = self.config
        sm = self.collision_sphere_model
        group_of = [collision_group(sm.geometryObjects[i].name) for i in range(sm.ngeoms)]
        candidates = wbc_safety.cross_group_index_pairs(group_of)

        # Compute nominal-pose distances for every candidate pair in one batch.
        for i, j in candidates:
            sm.addCollisionPair(pin.CollisionPair(i, j))
        tmp_data = sm.createData()
        q_nom = self.nominal_q()
        Configuration(
            self.model, self.model.createData(), q_nom, collision_model=sm, collision_data=tmp_data
        )
        dist_map = {
            candidates[k]: tmp_data.distanceResults[k].min_distance for k in range(len(candidates))
        }
        kept = wbc_safety.filter_nominal_overlaps(
            candidates, lambda i, j: dist_map[(i, j)], cfg.nominal_pair_keep_dist
        )

        sm.removeAllCollisionPairs()
        for i, j in kept:
            sm.addCollisionPair(pin.CollisionPair(i, j))
        self.collision_sphere_data = sm.createData()
        self._collision_enabled = len(sm.collisionPairs) > 0

        # Precompute the vectorized self-collision distance arrays. The Dexmate sphere
        # model is all spheres, so the signed clearance of pair (i, j) is exactly
        # ||c_i - c_j|| - r_i - r_j (coal's sphere narrowphase). Caching the pair index
        # arrays + per-geometry radii lets _min_self_distance be one numpy reduction over
        # oMg instead of a ~2900-element Python loop over coal DistanceResults. If any
        # geometry is not a sphere, leave these None -> _min_self_distance uses coal.
        if self._collision_enabled and all(
            type(g.geometry).__name__ == "Sphere" for g in sm.geometryObjects
        ):
            pairs = sm.collisionPairs
            self._sc_pair_i = np.fromiter((p.first for p in pairs), dtype=np.intp, count=len(pairs))
            self._sc_pair_j = np.fromiter(
                (p.second for p in pairs), dtype=np.intp, count=len(pairs)
            )
            self._sc_radius = np.array(
                [float(g.geometry.radius) for g in sm.geometryObjects], dtype=float
            )

    def _resolve_com_centroid(self) -> None:
        """Resolve the tip-over centroid proxy from ``cfg.com_centroid``.

        Sets ``_com_use_com`` (true whole-body CoM) or ``_com_frame_id`` (the
        reference-style torso-link frame-origin lean proxy). The nominal over-base
        offset the box bound is centred on is computed later in :meth:`reset` from the
        config nominal posture, independent of the reset pose.
        """
        cfg = self.config
        self._com_use_com = False
        self._com_frame_id = -1
        self._com_over_base_target: Optional[np.ndarray] = None
        if not cfg.enable_com_safety:
            return
        name = cfg.com_centroid.strip()
        if name.lower() in ("com", "centerofmass", "center_of_mass"):
            self._com_use_com = True
        elif self.model.existFrame(name):
            self._com_frame_id = self.model.getFrameId(name)
        else:
            raise ValueError(
                f"com_centroid {name!r} is not a model frame; use a torso link name "
                f"(e.g. 'torso_l3') or 'com' for the whole-body centre of mass"
            )

    def _build_tasks_and_limits(self) -> None:
        cfg = self.config
        self.left_ee_task = FrameTask(
            LEFT_EE_FRAME,
            cfg.ee_position_cost,
            cfg.ee_orientation_cost,
            lm_damping=cfg.lm_damping_ee,
        )
        self.right_ee_task = FrameTask(
            RIGHT_EE_FRAME,
            cfg.ee_position_cost,
            cfg.ee_orientation_cost,
            lm_damping=cfg.lm_damping_ee,
        )
        self.base_task = FrameTask(
            BASE_FRAME,
            list(cfg.base_position_cost),
            list(cfg.base_orientation_cost),
            lm_damping=cfg.lm_damping_base,
        )
        # Base velocity-smoothing weights (vx, vy, wz), applied directly to the planar
        # root's QP objective in solve() (see _add_base_velocity_smoothing). Validated
        # once here; _prev_base_velocity holds the last commanded base twist (zeroed at
        # reset and after each solve) that the smoothing references.
        self._base_vel_smooth_cost = np.asarray(cfg.base_velocity_smoothing_cost, dtype=float)
        if (
            self._base_vel_smooth_cost.shape != (3,)
            or not np.all(np.isfinite(self._base_vel_smooth_cost))
            or np.any(self._base_vel_smooth_cost < 0.0)
        ):
            raise ValueError(
                "base_velocity_smoothing_cost must be 3 finite, non-negative values "
                f"(vx, vy, wz); got {cfg.base_velocity_smoothing_cost!r}"
            )
        self._prev_base_velocity = np.zeros(3)
        # Single-axis base projection state (see _project_base_twist_single_axis): retain
        # the active axis across quiet solves. Cleared at reset.
        if not np.isfinite(cfg.base_single_axis_deadband) or cfg.base_single_axis_deadband < 0.0:
            raise ValueError(
                "base_single_axis_deadband must be finite and >= 0, got "
                f"{cfg.base_single_axis_deadband!r}"
            )
        if (
            not np.isfinite(cfg.base_dispatch_single_axis_deadband)
            or cfg.base_dispatch_single_axis_deadband < 0.0
        ):
            raise ValueError(
                "base_dispatch_single_axis_deadband must be finite and >= 0, got "
                f"{cfg.base_dispatch_single_axis_deadband!r}"
            )
        if (
            not np.isfinite(cfg.base_single_axis_hysteresis_ratio)
            or cfg.base_single_axis_hysteresis_ratio < 0.0
        ):
            raise ValueError(
                "base_single_axis_hysteresis_ratio must be finite and >= 0, got "
                f"{cfg.base_single_axis_hysteresis_ratio!r}"
            )
        self._base_single_axis_idx: Optional[int] = None
        # PostureTask covers actuated joints only (the planar root is excluded by Pink),
        # so base regularization lives in base_task above. The cost is a per-DOF vector so
        # the torso / arms / head can be weighted independently (like the reference).
        self.posture_task = PostureTask(
            cost=self._posture_cost_vector(
                cfg.posture_cost_torso, cfg.posture_cost_arm, cfg.posture_cost_head
            )
        )

        self.tasks = [
            self.left_ee_task,
            self.right_ee_task,
            self.base_task,
            self.posture_task,
        ]

        # Second posture task, re-targeted to the *current* configuration each solve (zero
        # error => pure joint-space velocity damping; see WBCConfig). The analog of the
        # reference's current_posture_task. Enabled by default (_main = 50); the head DOFs
        # get current_posture_cost_head (0 by default, matching the reference).
        self.current_posture_task: Optional[PostureTask] = None
        if cfg.current_posture_cost_main > 0.0 or cfg.current_posture_cost_head > 0.0:
            self.current_posture_task = PostureTask(
                cost=self._posture_cost_vector(
                    cfg.current_posture_cost_main,
                    cfg.current_posture_cost_main,
                    cfg.current_posture_cost_head,
                )
            )
            self.tasks.append(self.current_posture_task)

        # Head task: in head_mode "ik" the head is a whole-body QP DOF and a FrameTask
        # tracks the headset pose (rby1-style: base + torso + head co-track it). In
        # "track" (default) the head is externally commanded via solve(head_joints=...)
        # and pinned, so the task is built (reset() may seed it) but NOT added to the QP.
        self.head_task = FrameTask(
            HEAD_FRAME,
            cfg.head_position_cost,
            list(cfg.head_orientation_cost),
            lm_damping=cfg.lm_damping_ee,
        )
        if self._head_mode == "ik":
            self.tasks.append(self.head_task)
        # WORLD-frame head-position soft objective weights (x, y, z), added to the QP in
        # solve() via _add_head_world_position_objective (head_mode "ik"). Validated once
        # here, mirroring base_velocity_smoothing_cost. All-zero (default) disables it.
        self._head_world_position_cost = np.asarray(cfg.head_world_position_cost, dtype=float)
        if (
            self._head_world_position_cost.shape != (3,)
            or not np.all(np.isfinite(self._head_world_position_cost))
            or np.any(self._head_world_position_cost < 0.0)
        ):
            raise ValueError(
                "head_world_position_cost must be 3 finite, non-negative values "
                f"(x, y, z); got {cfg.head_world_position_cost!r}"
            )

        self.config_limit = ConfigurationLimit(self.model)
        self.velocity_limit = VelocityLimit(self.model)
        self.limits = [self.config_limit, self.velocity_limit]

        # Proactive self-collision barrier (closest n pairs become QP rows). When the
        # collision model is all spheres (the same condition that arms the vectorized
        # gate metric, _sc_radius is not None), use FastSelfCollisionBarrier: it
        # reproduces pink's SelfCollisionBarrier (h, J) -- hence the exact QP -- from
        # closed-form sphere math instead of coal's per-pair narrowphase read, cutting
        # the barrier build ~11x (the bulk of the tick). Validated bit-for-bit in
        # tests/test_fast_self_collision_barrier.py. Falls back to stock pink for any
        # non-sphere model.
        self.collision_barrier = None
        self.barriers = None
        if self._collision_enabled and SelfCollisionBarrier is not None:
            n_pairs = len(self.collision_sphere_model.collisionPairs)
            dim = max(1, min(cfg.n_collision_pairs, n_pairs))
            barrier_kwargs = dict(
                n_collision_pairs=dim,
                gain=cfg.collision_barrier_gain,
                safe_displacement_gain=cfg.collision_safe_displacement_gain,
                d_min=cfg.self_collision_safe_dist,
            )
            if self._sc_radius is not None:
                self.collision_barrier = FastSelfCollisionBarrier(
                    self.collision_sphere_model, **barrier_kwargs
                )
            else:
                self.collision_barrier = SelfCollisionBarrier(**barrier_kwargs)
            self.barriers = [self.collision_barrier]

        # Reactive hold gate. Tip-over (CoM-over-base) is enforced proactively by the
        # QP inequality (_add_com_over_base_inequality); the gate's CoM branch
        # (enable_com=True) stays on as a reactive backstop that reverts any step whose
        # CoM margin still crosses com_safety_margin. Self-collision is gated here too
        # (the barrier is a soft QP cost, not a hard constraint). The Mink backend, by
        # contrast, has no QP inequality and relies on the gate alone for both.
        self._gate = SafetyGate(
            self_collision_floor=cfg.self_collision_floor,
            self_collision_warn=cfg.self_collision_warn_dist,
            enable_com=True,
            enable_collision=self._collision_enabled,
        )

    def _posture_cost_vector(self, torso: float, arm: float, head: float) -> np.ndarray:
        """Per-DOF PostureTask cost vector (``arm`` default, torso/head overridden).

        Pink's PostureTask error and Jacobian span the actuated joints only -- the planar
        root's ``nv_root`` tangent DOFs are dropped -- so the vector has length
        ``model.nv - nv_root`` and each joint maps to index ``idx_v - nv_root``.
        """
        nvr = get_root_joint_dim(self.model)[1]
        vec = np.full(self.model.nv - nvr, float(arm))
        for name in TORSO_JOINTS:
            vec[self.model.idx_vs[self.model.getJointId(name)] - nvr] = torso
        for name in HEAD_JOINTS:
            vec[self.model.idx_vs[self.model.getJointId(name)] - nvr] = head
        return vec

    # posture / state
    def nominal_q(self) -> np.ndarray:
        """Build the nominal configuration, clamped to the URDF joint limits."""
        q = pin.neutral(self.model)
        for name, value in self.config.nominal_posture.items():
            if name in self._idx_q:
                q[self._idx_q[name]] = value
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        finite = np.isfinite(lo) & np.isfinite(hi)
        q[finite] = np.clip(q[finite], lo[finite] + 1e-3, hi[finite] - 1e-3)
        return q

    def reset(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the solver to ``q`` (defaults to the nominal posture).

        The posture target is anchored here, so the arms return toward this posture
        when the end-effector targets do not require otherwise. The base target is
        *not* anchored: :meth:`solve` re-sets it to the current base pose each step
        (velocity damping), so the base stays where it is pushed rather than springing
        back here. The set below is just an initial seed. The tip-over box is anchored
        to the **config nominal** posture (not ``q``): resetting to a leaned pose keeps
        the safety bound centred on the canonical safe lean.
        """
        q0 = self.nominal_q() if q is None else np.asarray(q, dtype=float).copy()
        if self._collision_enabled:
            self.configuration = Configuration(
                self.model,
                self.data,
                q0,
                collision_model=self.collision_sphere_model,
                collision_data=self.collision_sphere_data,
            )
        else:
            self.configuration = Configuration(self.model, self.data, q0)
        self.posture_task.set_target(q0)
        if self.current_posture_task is not None:
            self.current_posture_task.set_target(q0)  # seed; solve() re-targets each step
        self.base_task.set_target(  # initial seed; solve() re-targets to current pose
            self.configuration.get_transform_frame_to_world(BASE_FRAME)
        )
        if self._head_mode == "ik":
            # Seed the head task at the current head pose so the first solve before a
            # head_target arrives holds the head where it is (no jump).
            self.head_task.set_target(self.configuration.get_transform_frame_to_world(HEAD_FRAME))
        self._nominal_q = q0.copy()
        # Clear the base velocity-smoothing history: the first solve after an engage/reset
        # then smooths against a standstill (no lurch into the previous take's twist).
        self._prev_base_velocity = np.zeros(3)
        # Clear the single-axis active-axis latch so the first post-reset solve picks the
        # dominant axis fresh (no carried hysteresis from a previous take).
        self._base_single_axis_idx = None
        # Anchor the tip-over box on the centroid proxy's over-base offset at the *config
        # nominal* posture -- NOT at q0 -- so resetting to a leaned pose cannot silently
        # re-center the safety box around an already-risky lean (the reference likewise
        # anchors once, from its nominal posture). Resetting outside the box is safe: the
        # inequality's recovery branch allows move-back-only rather than wedging.
        if self.config.enable_com_safety:
            self._com_over_base_target = self._centroid_over_base(self.nominal_q())[0]
        # The torso-top x anchor is likewise anchored at the *config nominal* (not
        # q0): resetting to a leaned pose must not re-anchor the hold around the lean.
        if self._torso_top_fid is not None:
            self._torso_top_x_target = self._torso_top_x_offset(self.nominal_q())[0]
        self._gate.reset(self.stability_margin(q0), self._min_self_distance())
        return q0

    def _min_self_distance(self) -> float:
        """Closest distance (m) over the kept cross-group collision pairs.

        Reads the live ``Configuration`` collision data (refreshed on every
        ``integrate``/``update``); ``inf`` when collision avoidance is disabled.

        The Dexmate sphere model is all spheres, so the signed clearance of pair
        ``(i, j)`` is exactly ``||c_i - c_j|| - r_i - r_j`` -- coal's sphere narrowphase.
        We reduce it as one vectorized numpy pass over the SAME ``oMg`` centres and pair
        set coal uses (bit-identical; see
        ``tests/test_wbc_safety.py::test_min_self_distance_matches_coal``) rather than a
        ~2900-element Python loop over ``distanceResults`` (~1.2 ms -> ~30 us/tick). Falls
        back to the exact coal read if any geometry is not a sphere (``_sc_radius`` None)
        or the cached pair count ever disagrees with the live model.
        """
        if not self._collision_enabled:
            return float("inf")
        cd = self.configuration.collision_data
        n = len(self.configuration.collision_model.collisionPairs)
        if n == 0:
            return float("inf")
        if self._sc_radius is not None and self._sc_pair_i.shape[0] == n:
            centers = np.asarray([cd.oMg[i].translation for i in range(len(cd.oMg))])
            diff = centers[self._sc_pair_i] - centers[self._sc_pair_j]
            d = np.sqrt(np.einsum("ij,ij->i", diff, diff)) - (
                self._sc_radius[self._sc_pair_i] + self._sc_radius[self._sc_pair_j]
            )
            return float(d.min())
        return float(min(cd.distanceResults[k].min_distance for k in range(n)))

    def self_collision_distance(self, q: np.ndarray) -> float:
        """Closest cross-group self-collision distance (m) at an ARBITRARY config ``q``.

        Side-effect-free counterpart to :meth:`_min_self_distance` (which reads the live
        solver Configuration): builds a throwaway Configuration over the same
        collision-sphere model and kept cross-group pairs to evaluate ``q`` without
        touching the solver's configuration, barrier, or gate state. Use it to vet a
        candidate posture before commanding it -- e.g. each homing waypoint in
        ``scripts/wbc_vr_robot.py`` -- against ``self_collision_floor``. Returns ``inf``
        when collision avoidance is disabled (no sphere model / no kept pairs).

        The distance is invariant to the planar base DOFs (``q[:4]``), so only the joint
        entries of ``q`` matter; the base block may be left at neutral.
        """
        if not self._collision_enabled:
            return float("inf")
        q = np.asarray(q, dtype=float)
        if q.shape != (self.model.nq,):
            raise ValueError(f"q has shape {q.shape}, expected ({self.model.nq},)")
        if not np.all(np.isfinite(q)):
            raise ValueError("q contains non-finite values")
        sm = self.collision_sphere_model
        if self._probe_collision_data is None:
            self._probe_collision_data = sm.createData()
        # Configuration copies `data` (copy_data defaults True) and computes the
        # collision-pair distances into the dedicated probe buffer on construction, so
        # neither self.data nor the live collision_data is mutated.
        Configuration(
            self.model,
            self.data,
            q,
            collision_model=sm,
            collision_data=self._probe_collision_data,
        )
        n = len(sm.collisionPairs)
        if n == 0:
            return float("inf")
        cd = self._probe_collision_data
        return float(min(cd.distanceResults[k].min_distance for k in range(n)))

    def self_collision_pair_distances(self, q: np.ndarray) -> np.ndarray:
        """Per-pair cross-group clearance (m) at an ARBITRARY ``q``, one entry per kept pair.

        The side-effect-free, per-tick counterpart of :meth:`self_collision_distance`: the
        same sphere-sphere clearance :meth:`_min_self_distance` reduces, evaluated at ``q``
        (~0.2 ms vs ~2 ms for the coal probe), for callers that vet a candidate posture
        every 100 Hz tick -- the joystick torso gate. The pair order is fixed for the
        solver's lifetime, so two calls compare pair by pair. Empty when collision
        avoidance is disabled.
        """
        if not self._collision_enabled:
            return np.zeros(0)
        q = np.asarray(q, dtype=float)
        if q.shape != (self.model.nq,):
            raise ValueError(f"q has shape {q.shape}, expected ({self.model.nq},)")
        if not np.all(np.isfinite(q)):
            raise ValueError("q contains non-finite values")
        sm = self.collision_sphere_model
        if self._sc_radius is None:  # non-sphere geometry: the exact (slow) coal read
            if self._probe_collision_data is None:
                self._probe_collision_data = sm.createData()
            Configuration(
                self.model,
                self.data,
                q,
                collision_model=sm,
                collision_data=self._probe_collision_data,
            )
            cd = self._probe_collision_data
            return np.array(
                [cd.distanceResults[k].min_distance for k in range(len(sm.collisionPairs))]
            )
        if self._probe_model_data is None:
            self._probe_model_data = self.model.createData()
            self._probe_geom_data = sm.createData()
        pin.updateGeometryPlacements(
            self.model, self._probe_model_data, sm, self._probe_geom_data, q
        )
        centers = np.asarray([g.translation for g in self._probe_geom_data.oMg])
        diff = centers[self._sc_pair_i] - centers[self._sc_pair_j]
        return np.sqrt(np.einsum("ij,ij->i", diff, diff)) - (
            self._sc_radius[self._sc_pair_i] + self._sc_radius[self._sc_pair_j]
        )

    @property
    def q(self) -> np.ndarray:
        """Current full configuration vector (copy)."""
        return self.configuration.q.copy()

    @property
    def collision_enabled(self) -> bool:
        """True if self-collision avoidance is active (sphere model loaded)."""
        return self._collision_enabled

    def frame_pose(self, frame: str, q: Optional[np.ndarray] = None) -> pin.SE3:
        """Forward kinematics: world pose of ``frame`` at ``q`` (or current state)."""
        if q is None:
            return self.configuration.get_transform_frame_to_world(frame)
        config = Configuration(self.model, self.data, np.asarray(q, dtype=float))
        return config.get_transform_frame_to_world(frame)

    # solve
    def solve_head(self, head_target: Pose, dt: float) -> np.ndarray:
        """Compute one head-tracking step against the live whole-body configuration.

        Only head_j2/j3 move; the target's optical-axis direction is tracked, not its
        position or unactuated roll. Progress is min(1, head_ik_gain * dt), with damped
        least squares and scaled URDF position/velocity limits.

        Returns a (3,) joint command without modifying solver state. Pass it to solve()
        as head_joints and dispatch it to the head servo. The caller smooths commands;
        body-motion compensation uses the current base and torso immediately.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and > 0, got {dt}")
        target = _as_se3(head_target)
        fk = self.configuration.get_transform_frame_to_world(HEAD_FRAME)
        # Viewing-direction error: the rotation (world frame) taking the current
        # optical axis onto the target's, then expressed in the LOCAL frame to
        # match the local Jacobian. Perpendicular to the roll axis by construction.
        cur_axis = fk.rotation[:, 2]
        tgt_axis = target.rotation[:, 2]
        cross = np.cross(cur_axis, tgt_axis)
        sin_a = float(np.linalg.norm(cross))
        angle = float(np.arctan2(sin_a, float(cur_axis @ tgt_axis)))
        if sin_a > 1e-12:
            err_world = (angle / sin_a) * cross
        elif angle > 1.0:  # axes antiparallel: any perpendicular recovery axis works
            perp = np.cross(cur_axis, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(cur_axis, np.array([0.0, 1.0, 0.0]))
            err_world = angle * perp / np.linalg.norm(perp)
        else:  # aligned: nothing to do
            err_world = np.zeros(3)
        err = fk.rotation.T @ err_world
        jac = self.configuration.get_frame_jacobian(HEAD_FRAME)  # 6 x nv, LOCAL
        jw = jac[3:6][:, self._head_ik_idx_v]  # angular rows, pan/tilt columns (3x2)
        lam = self.config.head_ik_damp
        dq = np.linalg.solve(jw.T @ jw + lam * np.eye(jw.shape[1]), jw.T @ err)
        step = min(1.0, self.config.head_ik_gain * dt) * dq

        q = self.configuration.q
        cmd = np.array([q[i] for i in self._head_idx_q])
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        for k, (idx_q, idx_v) in enumerate(
            zip(self._head_ik_idx_q, self._head_ik_idx_v, strict=True)
        ):
            v_max = self.model.velocityLimit[idx_v]  # already velocity_limit_scale'd
            delta = float(np.clip(step[k], -v_max * dt, v_max * dt))
            cmd[k + 1] = float(np.clip(q[idx_q] + delta, lo[idx_q], hi[idx_q]))
        return cmd

    def solve(
        self,
        left_target: Pose,
        right_target: Pose,
        dt: float,
        head_joints: Optional[Sequence[float]] = None,
        head_target: Optional[Pose] = None,
        current_q: Optional[np.ndarray] = None,
        torso_joints: Optional[Sequence[float]] = None,
    ) -> WBCResult:
        """Advance differential IK toward left/right 4x4 or pin.SE3 targets.

        dt is the integration interval in seconds. In track mode, head_joints supplies
        the external (3,) head command; in ik mode, head_target updates the Cartesian
        head task. Omitted commands retain the previous head state/target.

        current_q re-seeds from measured joints without resetting posture anchors or
        safety history. torso_joints supplies an external (3,) torso command and requires
        lock_torso_in_ik. External joint commands are clamped to URDF limits.

        Returns WBCResult with the configuration, component commands, and target errors.
        """
        if current_q is not None:
            self._update_from_measured(current_q)
        if self._head_mode == "ik":
            # Head is a whole-body QP DOF; the FrameTask (added in _build_tasks_and_limits)
            # tracks the headset pose. head_joints is ignored in this mode.
            if head_target is not None:
                self.head_task.set_target(_as_se3(head_target))
        elif self._head_mode == "track" and head_joints is not None:
            self._set_head_joints(head_joints)
        # "fixed": the head stays where reset() put it (nominal_posture); head_joints ignored.
        if torso_joints is not None:
            self._set_torso_joints(torso_joints)
        self.left_ee_task.set_target(_as_se3(left_target))
        self.right_ee_task.set_target(_as_se3(right_target))
        tasks = self.tasks
        # Damp the base at its current pose (re-targeted every solve), so it stays
        # where pushed instead of springing toward home (matches deps/rby1-wbc).
        self.base_task.set_target(self.configuration.get_transform_frame_to_world(BASE_FRAME))
        if self.current_posture_task is not None:
            # Zero-error target at the current q => per-joint velocity damping this step.
            self.current_posture_task.set_target(self.configuration.q)

        q_before = self.configuration.q.copy()
        success = True
        try:
            # Build the IK QP explicitly, augment it with the hard CoM-over-base
            # inequality, then solve -- the reference's build_ik -> add inequalities
            # -> solve_problem path (rather than calling pink.solve_ik directly).
            self.configuration.check_limits(safety_break=False)
            problem = build_ik(
                self.configuration,
                tasks,
                dt,
                self.config.damping,
                limits=self.limits,
                barriers=self.barriers,
            )
            # Velocity equalities (problem.A) -- the slot where the reference enforces
            # its 6-joint torso linkage (torso_0/4/5 == 0, torso_3 == -(torso_1+torso_2),
            # which keeps its link_torso_5 -- the arm/head mount -- upright). Two uses
            # here: (1) pin the head DOFs (dq_head = 0). The head is teleoperated
            # externally (head_joints above) and has no task, so without the pin the
            # damping-only head would be the QP's *cheapest* lever to satisfy
            # head-vs-arm collision rows -- silently steering a head it does not
            # command; pinned, the QP must move the arm away instead. (2) the
            # torso-top x anchor row -- the Vega analog of the reference's linkage --
            # holding the torso_l3 top mount (arms + head) at the base-frame x it has
            # at the config nominal, so EE load cannot lean the upper body fore-aft
            # over the base (the base drives instead).
            if self._head_mode == "ik":
                # Head is an IK DOF (tracked by head_task), but the head joints named in
                # cfg.head_ik_pinned_joints (wbik.yaml; default head_j1/head_j2) are pinned
                # -- one row per joint, built in __init__ -- so their motion resolves through
                # the base/body while the unlisted head joints (default head_j3) keep IK
                # authority (e.g. camera tilt). The torso-top anchor stacks onto these rows.
                problem.A = self._head_j1_pin_A
                problem.b = np.zeros(problem.A.shape[0])
            else:
                problem.A = self._head_pin_A
                problem.b = np.zeros(len(self._head_idx_v))
            self._add_base_dof_equalities(problem)
            self._add_torso_dof_equalities(problem)
            self._add_torso_top_x_equality(problem, dt)
            self._add_com_over_base_terms(problem)
            self._add_head_world_position_objective(problem)
            self._add_base_velocity_smoothing(problem, dt)
            result = qpsolvers.solve_problem(problem, solver=self.config.solver)
            if not result.found or result.x is None:
                raise RuntimeError("no QP solution")
            velocity = result.x / dt
        except Exception:  # infeasible QP / barrier / limit hit: hold the current pose
            velocity = np.zeros(self.model.nv)
            success = False

        velocity = self._apply_locked_dof_masks(velocity)

        # Constrain the chassis to ONE pure motion (forward XOR strafe XOR turn), zeroing
        # the non-dominant base axes BEFORE integration so the solver's own base pose and
        # the emitted base twist agree with what is commanded. (No-op on the QP-failure
        # path, where velocity is already all zero.)
        if self.config.enable_base_single_axis:
            velocity = self._project_base_twist_single_axis(velocity)

        self.configuration.integrate_inplace(velocity, dt)

        # Reactive safety gate: self-collision plus a CoM-over-base backstop layered on
        # the proactive QP inequality above (enable_com=True), re-checked here on the
        # integrated step. A gate hold freezes the joints (planar base nq=4, nv=3) and
        # lets the base keep tracking; only a QP failure (not success) freezes all.
        margin = self.stability_margin(self.configuration.q)
        self_dist = self._min_self_distance()
        status = self._gate.check(margin, self_dist)
        held = status.held or not success
        if held:
            if not success:
                self.configuration.update(q_before)  # QP failed: freeze all
                velocity = np.zeros(self.model.nv)
            else:
                q_hold = self.configuration.q.copy()
                q_hold[4:] = q_before[4:]  # freeze joints, keep base
                self.configuration.update(q_hold)
                velocity = velocity.copy()
                velocity[3:] = 0.0  # zero joint vel, keep base twist
            self_dist = self._min_self_distance()
        velocity = self._apply_locked_dof_masks(velocity)

        # Remember the base twist actually committed this tick (post-hold, so a frozen or
        # base-only-hold tick anchors smoothing on what the chassis really did) for the
        # next solve's base velocity-smoothing reference.
        self._prev_base_velocity = velocity[:3].copy()

        return self._decompose(
            velocity,
            left_target,
            right_target,
            success,
            held=held,
            min_self_distance=self_dist,
            safety_status=status.message,
        )

    def _update_from_measured(self, current_q: np.ndarray) -> None:
        """Re-seed the internal configuration from a measured ``q`` (closed loop).

        ``Configuration.update`` runs forward kinematics *and* recomputes the
        collision-pair distances, so the barrier, the gate metric, and the tip-over
        inequality all see the measured state. The planar-root rotation block
        ``q[2:4]`` must be a (cos, sin) pair; after validation it is renormalized to
        machine precision (Pinocchio integration assumes a unit pair).
        """
        q = np.asarray(current_q, dtype=float)
        if q.shape != (self.model.nq,):
            raise ValueError(f"current_q has shape {q.shape}, expected ({self.model.nq},)")
        if not np.all(np.isfinite(q)):
            raise ValueError("current_q contains non-finite values")
        norm = float(np.hypot(q[2], q[3]))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"current_q[2:4] (planar cos, sin) must be unit norm, got {norm:.6f}")
        q = q.copy()
        q[2:4] /= norm
        self.configuration.update(q)

    def _set_head_joints(self, head_joints: Sequence[float]) -> None:
        """Write an external head joint command into the model (vr_reader 'track').

        The head is not a whole-body IK DOF (its tangent DOFs are pinned in the QP),
        so this is the only way it moves. Values are clamped to the URDF position
        limits (solve_head already clamps; this also covers external publishers).
        ``Configuration.update`` re-runs FK and the collision-pair distances, so the
        barrier, the gate metric, and the CoM terms all see the commanded head.
        """
        arr = np.asarray(head_joints, dtype=float).reshape(-1)
        if arr.shape != (len(self._head_idx_q),):
            raise ValueError(
                f"head_joints has shape {arr.shape}, "
                f"expected ({len(self._head_idx_q)},) for joints {HEAD_JOINTS}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"head_joints contains non-finite values: {arr}")
        q = self.configuration.q.copy()
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        for idx_q, val in zip(self._head_idx_q, arr, strict=True):
            q[idx_q] = np.clip(val, lo[idx_q], hi[idx_q])
        self.configuration.update(q)

    def _set_torso_joints(self, torso_joints: Sequence[float]) -> None:
        """Write an external torso command into the model (the joystick torso command).

        Only valid with ``lock_torso_in_ik``: the torso DOFs are then pinned in the QP,
        so this is the only way the torso moves, exactly like ``_set_head_joints``.
        """
        if not self.config.lock_torso_in_ik:
            raise ValueError("torso_joints requires lock_torso_in_ik (the QP owns the torso)")
        arr = np.asarray(torso_joints, dtype=float).reshape(-1)
        if arr.shape != (len(self._torso_idx_q),):
            raise ValueError(
                f"torso_joints has shape {arr.shape}, "
                f"expected ({len(self._torso_idx_q)},) for joints {TORSO_JOINTS}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"torso_joints contains non-finite values: {arr}")
        q = self.configuration.q.copy()
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        target = np.clip(arr, lo[self._torso_idx_q], hi[self._torso_idx_q])
        if np.array_equal(q[self._torso_idx_q], target):
            return  # a resting torso: skip the FK/collision refresh
        q[self._torso_idx_q] = target
        self.configuration.update(q)

    def _torso_top_x_offset(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        """Base-frame x of the torso-top mount over the base, and its gradient row.

        Returns ``(x_off, row)`` where ``x_off`` (m) is the x-component, in the
        base frame, of ``TORSO_TOP_FRAME``'s position relative to the base (the
        quantity the anchor holds at its nominal value) and ``row`` (nv,) is its
        gradient w.r.t. the configuration displacement. Like
        :meth:`_centroid_over_base`, the offset is invariant to the planar base
        DOFs (translating/yawing the base carries the mount along), so those
        Jacobian columns are zeroed; with Vega's all-parallel-pitch torso the row
        is supported on the three torso joints only.
        """
        data = self._com_data
        pin.forwardKinematics(self.model, data, q)
        pin.computeJointJacobians(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        point = data.oMf[self._torso_top_fid].translation
        jac = pin.getFrameJacobian(self.model, data, self._torso_top_fid, pin.LOCAL_WORLD_ALIGNED)[
            :3
        ]
        jac = np.asarray(jac).copy()
        jac[:, :3] = 0.0  # invariant to the planar base DOFs (x, y, yaw)
        x, y, yaw = self.base_pose_from_q(q)
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)  # world -> base rotation
        x_off = cos_y * (point[0] - x) - sin_y * (point[1] - y)
        row = cos_y * jac[0] - sin_y * jac[1]
        return float(x_off), row

    def _add_torso_top_x_equality(self, problem: qpsolvers.Problem, dt: float) -> None:
        """Stack the torso-top x anchor equality row onto the QP.

        The Vega analog of the reference's hard torso rows in the same
        ``problem.A`` slot (its torso_3 = -(torso_1 + torso_2) coupling keeps the
        arm/head mount from pitching fore-aft): one row constrains the
        displacement of the top mount's base-frame x. The RHS servos
        ``min(1, torso_top_x_anchor_gain * dt)`` of the residual offset from the
        nominal anchor -- zero when on the anchor (a pure velocity constraint,
        the reference's b = 0), a recovery step when off it (measured
        ``current_q`` / custom reset poses) -- clipped to half the row's
        velocity-limit headroom so the equality alone can never make the QP
        infeasible.
        """
        if (
            self.config.lock_torso_in_ik
            or self._torso_top_fid is None
            or self._torso_top_x_target is None
        ):
            return
        x_off, row = self._torso_top_x_offset(self.configuration.q)
        err = x_off - self._torso_top_x_target
        delta = -err * min(1.0, self.config.torso_top_x_anchor_gain * dt)
        cap = 0.5 * float(np.abs(row) @ self.model.velocityLimit) * dt
        delta = float(np.clip(delta, -cap, cap))
        problem.A = np.vstack([problem.A, row[None, :]])
        problem.b = np.hstack([problem.b, [delta]])

    def _add_base_dof_equalities(self, problem: qpsolvers.Problem) -> None:
        """Stack base-DOF lock rows such as ``dq_yaw = 0`` onto the QP."""
        if self._base_dof_pin_A.shape[0] == 0:
            return
        problem.A = np.vstack([problem.A, self._base_dof_pin_A])
        problem.b = np.hstack([problem.b, np.zeros(self._base_dof_pin_A.shape[0])])

    def _add_torso_dof_equalities(self, problem: qpsolvers.Problem) -> None:
        """Stack ``dq_torso = 0`` rows when the torso is excluded from IK."""
        if self._torso_dof_pin_A.shape[0] == 0:
            return
        problem.A = np.vstack([problem.A, self._torso_dof_pin_A])
        problem.b = np.hstack([problem.b, np.zeros(self._torso_dof_pin_A.shape[0])])

    def _apply_locked_dof_masks(self, velocity: np.ndarray) -> np.ndarray:
        """Make configured base and torso locks exact in a full ``nv`` vector."""
        out = self._apply_base_dof_mask(velocity)
        if self.config.lock_torso_in_ik:
            out[self._torso_idx_v] = 0.0
        return out

    def _apply_base_dof_mask(self, velocity: np.ndarray) -> np.ndarray:
        """Zero disabled planar-root velocity components in a full ``nv`` vector."""
        out = np.asarray(velocity, dtype=float).copy()
        out[:3] = self._mask_base_twist(out[:3])
        return out

    def _mask_base_twist(self, twist: np.ndarray) -> np.ndarray:
        """Zero disabled planar base axes in a ``(vx, vy, wz)`` twist."""
        out = np.asarray(twist, dtype=float).copy()
        out[~self._base_dof_mask] = 0.0
        return out

    def _add_base_velocity_smoothing(self, problem: qpsolvers.Problem, dt: float) -> None:
        """Penalize changes in body-frame base twist without resisting steady motion.

        Add 0.5 * sum(cost_k**2 * (dq_k - previous_velocity_k * dt)**2) on the
        three planar-root DOFs. This changes only the QP objective, not safety constraints.
        """
        cost = self._base_vel_smooth_cost
        if not np.any(cost > 0.0):
            return
        v_prev = self._prev_base_velocity
        for k in range(3):  # planar root tangent DOFs: vx, vy, wz
            if not self._base_dof_mask[k]:
                continue
            c2 = float(cost[k]) ** 2
            if c2 <= 0.0:
                continue
            problem.P[k, k] += c2
            problem.q[k] += -c2 * float(v_prev[k]) * dt

    def _project_base_twist_single_axis(self, velocity: np.ndarray) -> np.ndarray:
        """Project the base command onto one axis before integrating solver state.

        Keep axis memory across quiet solves; reset clears it. Followers also project
        their post-feedback wheel commands, so both paths enforce the same axis policy.
        """
        out = velocity.copy()
        base = self._mask_base_twist(out[:3])
        prev_axis = self._base_single_axis_idx
        if prev_axis is not None and not self._base_dof_mask[prev_axis]:
            prev_axis = None
        out[:3], self._base_single_axis_idx = project_planar_twist_single_axis(
            base,
            xy_max_vel=self.config.base_xy_max_vel,
            yaw_max_vel=self.config.base_yaw_max_vel,
            deadband=self.config.base_single_axis_deadband,
            hysteresis_ratio=self.config.base_single_axis_hysteresis_ratio,
            prev_axis=prev_axis,
            retain_axis_on_quiet=True,
        )
        out[:3] = self._mask_base_twist(out[:3])
        return out

    def _centroid_over_base(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal offset of the tip-over centroid from the base, and its Jacobian.

        Returns ``(d_base_xy, jac_base_xy)`` where ``d_base_xy`` (2,) is the centroid's
        position relative to the base, expressed in the base frame and projected to the
        ground plane, and ``jac_base_xy`` (2 x nv) is ``d(d_base_xy)/dΔq``. The centroid
        is the true whole-body CoM (``com_centroid == "com"``) or a torso link frame
        origin (the reference-style lean proxy).

        The offset is invariant to the planar base DOFs -- translating or yawing the
        base moves the centroid and the base together -- so those Jacobian columns are
        zeroed; only joint motion can change it.
        """
        data = self._com_data
        if self._com_use_com:
            jac = pin.jacobianCenterOfMass(self.model, data, q)  # 3 x nv; sets data.com[0]
            point = data.com[0]
        else:
            pin.forwardKinematics(self.model, data, q)
            pin.computeJointJacobians(self.model, data, q)
            pin.updateFramePlacements(self.model, data)
            jac = pin.getFrameJacobian(
                self.model, data, self._com_frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3]
            point = data.oMf[self._com_frame_id].translation
        jac = np.asarray(jac).copy()
        jac[:, :3] = 0.0  # invariant to the planar base DOFs (x, y, yaw)
        x, y, yaw = self.base_pose_from_q(q)
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])  # world -> base (xy block)
        d_base = rot @ np.array([point[0] - x, point[1] - y])
        jac_base = rot @ jac[:2, :]
        return d_base, jac_base

    def _add_com_over_base_terms(self, problem: qpsolvers.Problem) -> None:
        """Augment the IK QP with the CoM-over-base safety terms.

        Computes the centroid proxy's horizontal over-base offset (and Jacobian)
        once, then adds both ports of the reference's CoM handling: the **hard**
        tip-over box inequality (:meth:`_add_com_over_base_inequality`) and the
        **soft** centering objective (:meth:`_add_com_over_base_soft_objective`).
        """
        if not self.config.enable_com_safety or self._com_over_base_target is None:
            return
        d_base, jac_base = self._centroid_over_base(self.configuration.q)
        err = d_base - self._com_over_base_target
        self._add_com_over_base_inequality(problem, err, jac_base)
        self._add_com_over_base_soft_objective(problem, err, jac_base)

    def _add_com_over_base_soft_objective(
        self, problem: qpsolvers.Problem, err: np.ndarray, jac_base: np.ndarray
    ) -> None:
        """Center the centroid proxy over its nominal base-frame xy offset.

        Add ||cost * (J * dq + error)||**2 using the hard stability box's xy Jacobian.
        Working in base axes keeps vertical motion free even when the torso is pitched;
        the hard box inequality remains the safety backstop.
        """
        c = self.config.com_over_base_pos_cost
        if c <= 0.0:
            return
        weighted_jac = c * jac_base
        problem.P = problem.P + weighted_jac.T @ weighted_jac
        problem.q = problem.q + (c * c) * (jac_base.T @ err)

    def _add_head_world_position_objective(self, problem: qpsolvers.Problem) -> None:
        """Track head translation in gravity-aligned world axes in head_mode="ik".

        Each active axis adds ||cost_k * (J_k * dq + error_k)||**2. A zero world-z
        weight frees torso height without coupling it to the pitched optical axis.
        The Cartesian target is shared with the head orientation task.
        """
        if self._head_mode != "ik":
            return
        costs = self._head_world_position_cost
        active = costs > 0.0
        if not np.any(active):
            return
        fk = self.configuration.get_transform_frame_to_world(HEAD_FRAME)
        target = self.head_task.transform_target_to_world  # pin.SE3, set in solve()/reset()
        # World-aligned translational Jacobian: get_frame_jacobian is LOCAL (see solve_head),
        # so rotate its linear block into the world frame (R @ J_local[:3]).
        jac_local = self.configuration.get_frame_jacobian(HEAD_FRAME)
        jac_world = fk.rotation @ jac_local[:3, :]  # 3 x nv
        err = fk.translation - target.translation  # (3,) world: current - target
        j = jac_world[active]  # (k, nv)
        c = costs[active]  # (k,)
        weighted = c[:, None] * j  # (k, nv)
        problem.P = problem.P + weighted.T @ weighted
        problem.q = problem.q + j.T @ ((c * c) * err[active])

    def _add_com_over_base_inequality(
        self, problem: qpsolvers.Problem, err: np.ndarray, jac_base: np.ndarray
    ) -> None:
        """Augment the IK QP with the hard tip-over (CoM-over-base) inequality.

        Ports the reference's ``_add_com_over_base_xy_inequalities``: rather than a
        reactive stop, the QP itself bounds the centroid proxy's horizontal over-base
        offset within ``+/-com_safety_margin`` of its nominal value, so the robot keeps
        moving while staying upright. With ``com_centroid`` a torso frame this is the
        reference's jacobian-linearized lean proxy; with ``"com"`` it is the true CoM.

        Per axis the offset ``d(q ⊕ Δq) ≈ d(q) + J Δq`` is held within
        ``[target - margin, target + margin]``. When already outside the box the row
        relaxes to "move back only" (``h = 0``), matching the reference's recovery
        branch, so the solver is never wedged.
        """
        bound = self.config.com_safety_margin

        g_rows: list[np.ndarray] = []
        h_rows: list[float] = []
        for k in range(2):  # x, y axes of the over-base offset
            row = jac_base[k]
            e = float(err[k])
            if abs(e) <= bound:  # inside the box: keep both sides within +/-bound
                g_rows += [row, -row]
                h_rows += [bound - e, bound + e]
            elif e > bound:  # past the +bound: allow moving back only
                g_rows += [row]
                h_rows += [0.0]
            else:  # past the -bound: allow moving back only
                g_rows += [-row]
                h_rows += [0.0]

        g_add = np.vstack(g_rows)
        h_add = np.asarray(h_rows, dtype=float)
        if problem.G is None:
            problem.G, problem.h = g_add, h_add
        else:
            problem.G = np.vstack([problem.G, g_add])
            problem.h = np.hstack([problem.h, h_add])

    # readout
    def _group(self, q: np.ndarray, names: Sequence[str]) -> np.ndarray:
        return np.array([q[self._idx_q[n]] for n in names])

    def base_pose_from_q(self, q: np.ndarray) -> np.ndarray:
        """Extract (x, y, yaw) of the mobile base from a configuration vector.

        The planar root joint stores its rotation as (cos, sin) at q[2:4].
        """
        return np.array([q[0], q[1], float(np.arctan2(q[3], q[2]))])

    def center_of_mass(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """World-frame center of mass at ``q`` (defaults to the current state)."""
        qq = self.configuration.q if q is None else np.asarray(q, dtype=float)
        return pin.centerOfMass(self.model, self._com_data, qq).copy()

    def stability_margin(self, q: Optional[np.ndarray] = None) -> float:
        """Signed distance (m) of the CoM ground projection to the support polygon.

        Positive = the CoM is inside the wheel footprint (stable); <= 0 means the
        CoM has left the support polygon and the robot is tipping. Joint limits do
        not capture this failure mode, so it is checked separately.
        """
        qq = self.configuration.q if q is None else np.asarray(q, dtype=float)
        com = self.center_of_mass(qq)
        x, y, yaw = self.base_pose_from_q(qq)
        # CoM in the base frame (rotate world -> base by -yaw, drop z).
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        dx, dy = com[0] - x, com[1] - y
        com_base = np.array([cos_y * dx - sin_y * dy, sin_y * dx + cos_y * dy])
        return _signed_dist_to_convex_polygon(com_base, self._support_polygon)

    def _decompose(
        self,
        velocity: np.ndarray,
        left_target: Pose,
        right_target: Pose,
        success: bool,
        held: bool = False,
        min_self_distance: float = float("inf"),
        safety_status: str = "ok",
    ) -> WBCResult:
        q = self.configuration.q.copy()
        left_err = float(
            np.linalg.norm(
                self.configuration.get_transform_frame_to_world(LEFT_EE_FRAME).translation
                - _as_se3(left_target).translation
            )
        )
        right_err = float(
            np.linalg.norm(
                self.configuration.get_transform_frame_to_world(RIGHT_EE_FRAME).translation
                - _as_se3(right_target).translation
            )
        )
        return WBCResult(
            q=q,
            velocity=velocity.copy(),
            base_pose=self.base_pose_from_q(q),
            base_twist=velocity[:3].copy(),  # (vx, vy, wz) -> chassis command
            torso=self._group(q, TORSO_JOINTS),
            left_arm=self._group(q, LEFT_ARM_JOINTS),
            right_arm=self._group(q, RIGHT_ARM_JOINTS),
            head=self._group(q, HEAD_JOINTS),
            left_ee_error=left_err,
            right_ee_error=right_err,
            success=success,
            stability_margin=self.stability_margin(q),
            held=held,
            min_self_distance=min_self_distance,
            safety_status=safety_status,
        )
