#!/usr/bin/env python3
"""Publish fixed-base-frame hand targets and fixed-speed joystick chassis commands.

Right stick drives; left stick turns. Optional --torso-stick or --torso-ik must
match the follower. Quest connection and calibration match wbc_vr_leader.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path
from typing import Optional

import numpy as np

from omniteleop.follower.base_closed_loop import (
    mask_planar_twist_for_base_dofs,
    project_planar_twist_single_axis_for_base_dofs,
)
from omniteleop.follower.joystick_base import fixed_speed_planar_twist
from omniteleop.follower.whole_body_ik import WBCConfig
from omniteleop.wbc_teleop import JoystickTeleopConfig, JoystickTorsoConfig, VRTeleopConfig

# Load the base leader as a module (scripts/ is not a package; the tests do the same via
# importlib). Executing it only binds module-level defaults -- main() is __main__-guarded --
# so this is a cheap, side-effect-free import of the class + helpers we extend.
_BASE_PATH = Path(__file__).resolve().parent / "wbc_vr_leader.py"
_spec = importlib.util.spec_from_file_location("wbc_vr_leader", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

WBCVRLeader = _base.WBCVRLeader
INVALID_LEFT_POSE = _base.INVALID_LEFT_POSE
INVALID_RIGHT_POSE = _base.INVALID_RIGHT_POSE
_is_tracked = _base._is_tracked
DEFAULT_CERT = _base.DEFAULT_CERT
DEFAULT_KEY = _base.DEFAULT_KEY
DEFAULT_EE_OFFSET_FILE = _base.DEFAULT_EE_OFFSET_FILE
DEFAULT_HEADSET_HUD_RATE = _base.DEFAULT_HEADSET_HUD_RATE
_DEFAULT_HUD_CAMERAS = _base._DEFAULT_HUD_CAMERAS


class JoystickVRLeader(WBCVRLeader):
    """VR leader with fixed-base-frame hands and joystick-driven base (see module docstring)."""

    def __init__(self, args: argparse.Namespace) -> None:
        cfg = getattr(args, "wbc_config", None)
        if cfg is None:
            cfg = WBCConfig.from_yaml(args.config)
        joystick_cfg = getattr(args, "joystick_config", None)
        if joystick_cfg is None:
            joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
        teleop_cfg = getattr(args, "vr_teleop_config", None)
        if teleop_cfg is None:
            teleop_cfg = VRTeleopConfig.from_yaml(args.config)
        self._joystick_base_dofs = cfg.base_dofs
        self._joystick_single_axis = cfg.enable_base_single_axis
        self._joystick_xy_max_vel = teleop_cfg.stick_max_vx
        self._joystick_yaw_max_vel = teleop_cfg.stick_max_wz
        self._joystick_axis_deadband = cfg.base_single_axis_deadband
        self._joystick_axis_hysteresis = joystick_cfg.single_axis_hysteresis_ratio
        self._joystick_translation_speed = joystick_cfg.translation_speed
        self._joystick_rotation_speed = joystick_cfg.rotation_speed
        self._joystick_axis: Optional[int] = None
        self._debug_sticks = bool(getattr(args, "debug_sticks", False))
        self._torso_stick = bool(getattr(args, "torso_stick", False))
        self._torso_mode = (
            "stick" if self._torso_stick else "ik" if getattr(args, "torso_ik", False) else ""
        )
        torso_cfg = getattr(args, "joystick_torso_config", None)
        if self._torso_stick and torso_cfg is None:
            torso_cfg = JoystickTorsoConfig.from_yaml(args.config)
        self._torso_speed = torso_cfg.speed if self._torso_stick else 0.0
        self._torso_vz = 0.0
        super().__init__(
            args,
            wbc_config=cfg,
            teleop_config=teleop_cfg,
            reference_target_frame="base",
        )

    def _calibrate(self, headset_pose: np.ndarray) -> None:
        # Engage is a fresh operator command sequence, matching the follower/IK latch reset.
        self._joystick_axis = None
        super()._calibrate(headset_pose)

    # hand mapping: fixed gravity-aligned base frame (NOT head-yaw relative)
    def _bprime_pos(self, side, vr_now, vr_head=None, head_target=None) -> np.ndarray:
        """Hand POSITION in the calibrated base frame (room-anchored, head-independent).

        ``robot_base_t_vr_base_eef`` is the gravity-aligned room->base calibration (yaw +
        translation) captured at engage. Mapping the controller through it places the hand
        at its operator-space position expressed in the robot base frame; the follower
        applies that in the robot's CURRENT base, so the hand rides a joystick base turn
        without any dependence on where the operator is looking. ``vr_head``/``head_target``
        are accepted for signature compatibility with the base class and ignored.
        """
        assert self.robot_base_t_vr_base_eef is not None
        return (self.robot_base_t_vr_base_eef @ np.asarray(vr_now, dtype=float))[:3, 3]

    def _eef_target(self, side, vr_now, *, vr_head=None, head_target=None, now=0.0) -> np.ndarray:
        """Base-frame EEF target: fixed-frame position (+ engage blend) and absolute orientation.

        Same structure as ``WBCVRLeader._eef_target`` minus the head-geometry requirement:
        the position comes from the fixed-frame ``_bprime_pos`` above (with the per-side
        engage offset ``_hf_offset_*`` ramped to zero over ``_hf_blend_dur`` so the arm does
        not lunge), and the orientation is the unchanged absolute map
        ``R_yaw @ vr_now_rot @ C`` (``C`` = the calibrated controller->gripper offset).
        """
        vr_now = np.asarray(vr_now, dtype=float)
        ee_offset = getattr(self, f"_ee_offset_{side}", None)
        if ee_offset is None:
            ee_offset = np.eye(3)
        off = self._hf_offset_left if side == "left" else self._hf_offset_right
        blend = (
            1.0
            if self._hf_engage_t is None
            else float(np.clip((now - self._hf_engage_t) / self._hf_blend_dur, 0.0, 1.0))
        )
        out = np.eye(4)
        out[:3, 3] = self._bprime_pos(side, vr_now) - (1.0 - blend) * off
        out[:3, :3] = self.robot_base_t_vr_base_eef[:3, :3] @ vr_now[:3, :3] @ ee_offset
        return out

    def _map_targets(self, vr_l, vr_r, *, vr_head=None, head_target=None, now=0.0):
        """Map controller poses to base-frame EEF targets, holding last-good if untracked.

        Unlike the base leader, the hands do NOT depend on ``head_target`` (the fixed-frame
        position needs no head geometry), so a lost headset frame updates the hands normally
        instead of freezing them. Each side holds its last-good target only when its own
        controller is untracked.
        """
        assert self.robot_base_t_vr_base_eef is not None
        if _is_tracked(vr_l, INVALID_LEFT_POSE):
            self.last_left_target = self._eef_target("left", vr_l, now=now)
        if _is_tracked(vr_r, INVALID_RIGHT_POSE):
            self.last_right_target = self._eef_target("right", vr_r, now=now)
        return self.last_left_target, self.last_right_target

    # base: RIGHT stick drives (fwd/strafe), LEFT stick turns
    def _thumbstick_to_chassis(self, transforms) -> tuple[float, float, float]:
        def dz(v: float) -> float:
            return v if abs(v) > self.stick_deadzone else 0.0

        left = transforms["left_thumbstick"]
        right = transforms["right_thumbstick"]
        # Quest stick y is negative when pushed up -> negate so forward is positive.
        vx = -dz(right[1]) * self.stick_max_vx  # right stick Y  -> forward/back
        vy = -dz(right[0]) * self.stick_max_vy  # right stick X  -> strafe left/right
        wz = -dz(left[0]) * self.stick_max_wz  # left stick X   -> yaw
        # left stick Y -> torso height (--torso-stick only). The left stick drives whichever
        # of its axes is pushed further, so a sideways push for a turn never moves the torso.
        torso_vz = 0.0
        if getattr(self, "_torso_stick", False) and abs(dz(left[1])) > abs(dz(left[0])):
            wz = 0.0
            torso_vz = -float(np.sign(left[1])) * self._torso_speed  # up (y < 0) raises
        self._torso_vz = torso_vz
        chassis = mask_planar_twist_for_base_dofs(
            np.array([vx, vy, wz], dtype=float),
            base_dofs=self._joystick_base_dofs,
            allow_yaw_hold=False,
        )
        if self._joystick_single_axis:
            chassis, self._joystick_axis = project_planar_twist_single_axis_for_base_dofs(
                chassis,
                base_dofs=self._joystick_base_dofs,
                allow_yaw_hold=False,
                xy_max_vel=self._joystick_xy_max_vel,
                yaw_max_vel=self._joystick_yaw_max_vel,
                deadband=self._joystick_axis_deadband,
                hysteresis_ratio=self._joystick_axis_hysteresis,
                prev_axis=self._joystick_axis,
            )
        else:
            self._joystick_axis = None
        chassis = fixed_speed_planar_twist(
            chassis,
            translation_speed=self._joystick_translation_speed,
            rotation_speed=self._joystick_rotation_speed,
        )
        vx, vy, wz = (float(v) for v in chassis)
        # Bring-up diagnostic, off unless --debug-sticks. Reads the RAW sticks WebXR delivers
        # beside the mapped chassis twist at ~2 Hz, which separates the two ways the base
        # stays still: sticks [0, 0] while you push them means the data is not reaching the
        # leader (browser/reader), non-zero sticks with a zero chassis means the deadzone or
        # the mapping.
        if getattr(self, "_debug_sticks", False):
            now = time.perf_counter()
            if now - getattr(self, "_dbg_stick_t", 0.0) > 0.5:
                self._dbg_stick_t = now
                print(
                    f"\n[joystick dbg] L_stick={[round(float(x), 3) for x in left]} "
                    f"R_stick={[round(float(x), 3) for x in right]} "
                    f"click(L,R)=({transforms.get('left_thumbstick_click')},"
                    f"{transforms.get('right_thumbstick_click')}) "
                    f"-> vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} torso_vz={torso_vz:+.2f}"
                )
        return vx, vy, wz

    def _extra_publish_fields(self) -> dict:
        # Consumed by one publish: the ones that skip _thumbstick_to_chassis (static, home,
        # sweep, episode end) then send zero instead of a stale torso command.
        torso_vz, self._torso_vz = getattr(self, "_torso_vz", 0.0), 0.0
        return {"torso_vz": float(torso_vz), "torso_mode": getattr(self, "_torso_mode", "")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--namespace", default="", help="Zenoh namespace (must match the follower; default empty)."
    )
    parser.add_argument("--host", default="0.0.0.0", help="WebXR bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5067, help="WebXR port (default 5067).")
    parser.add_argument("--cert", default=DEFAULT_CERT, help="TLS cert for the WebXR server.")
    parser.add_argument("--key", default=DEFAULT_KEY, help="TLS key for the WebXR server.")
    parser.add_argument(
        "--config",
        default=None,
        help="wbik.yaml path for publish rate, shared thumbstick mapping, and "
        "base single-axis policy. Default: canonical follower/wbik.yaml.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="right-grip hold time to calibrate/start (default 1.0).",
    )
    parser.add_argument(
        "--headset-hud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the robot camera/status HUD in the Quest (default: enabled).",
    )
    parser.add_argument(
        "--sim-cameras",
        action="store_true",
        help="read BEHAVIOR camera topics in --namespace for the headset HUD",
    )
    parser.add_argument(
        "--hud-rate",
        type=float,
        default=DEFAULT_HEADSET_HUD_RATE,
        help=f"headset HUD refresh rate in Hz (default {DEFAULT_HEADSET_HUD_RATE:g}).",
    )
    parser.add_argument(
        "--hud-cameras",
        nargs="+",
        choices=(
            "head_left_rgb",
            "head_right_rgb",
            "head_depth",
            "left_wrist_rgb",
            "right_wrist_rgb",
        ),
        default=list(_DEFAULT_HUD_CAMERAS),
        help="camera streams to read for the headset HUD.",
    )
    parser.add_argument(
        "--hud-center-camera",
        choices=_base.HUD_RGB_CAMERAS,
        default=None,
        help="RGB view to place in the middle of the headset HUD; must be "
        "included in --hud-cameras. Default: keep --hud-cameras order.",
    )
    parser.add_argument(
        "--align-reference",
        default=None,
        help="optional raw-data episode HDF5 defining the reference L/R EEF pose "
        "for the post-calibration align gate (see wbc_vr_leader). Omit to "
        "disable.",
    )
    parser.add_argument(
        "--ee-offset-file",
        default=DEFAULT_EE_OFFSET_FILE,
        help="YAML of per-side controller->gripper orientation offsets "
        f"(scipy [x,y,z,w] quats). Default: {DEFAULT_EE_OFFSET_FILE}.",
    )
    parser.add_argument(
        "--calibrate-ee-offset",
        action="store_true",
        help="one-time helper: capture the neutral-grip offset C and exit "
        "(same as wbc_vr_leader.py --calibrate-ee-offset).",
    )
    parser.add_argument(
        "--ee-capture-seconds",
        type=float,
        default=1.5,
        help="averaging window for --calibrate-ee-offset (default 1.5).",
    )
    parser.add_argument(
        "--debug-vr",
        dest="debug_vr",
        default=None,
        help="write a per-tick RAW-VR diagnostic HDF5 under this directory.",
    )
    parser.add_argument(
        "--debug-sticks",
        action="store_true",
        help="print the raw WebXR thumbstick values beside the mapped chassis "
        "twist at ~2 Hz. Bring-up aid for a base that will not drive; "
        "off by default.",
    )
    torso = parser.add_mutually_exclusive_group()
    torso.add_argument(
        "--torso-stick",
        action="store_true",
        help="LEFT stick Y raises/lowers the torso (fixed joystick_torso.speed "
        "in wbik.yaml); pass --torso-stick to the follower as well. Off by "
        "default: the torso stays at nominal.",
    )
    torso.add_argument(
        "--torso-ik",
        action="store_true",
        help="the follower's whole-body IK moves the torso to reach your hands "
        "(no head, base on the sticks); pass --torso-ik to the follower as "
        "well. Off by default: the torso stays at nominal.",
    )
    args = parser.parse_args()
    try:
        _base.order_hud_cameras(args.hud_cameras, args.hud_center_camera)
    except ValueError as exc:
        parser.error(str(exc))
    args.sim_camera_namespace = args.namespace if args.sim_cameras else None

    # Publish rate = follower cmd_rate from wbik.yaml (leader/follower cannot drift).
    cfg = WBCConfig.from_yaml(args.config)
    vt = VRTeleopConfig.from_yaml(args.config)
    joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
    args.wbc_config = cfg
    args.joystick_config = joystick_cfg
    args.vr_teleop_config = vt
    args.joystick_torso_config = (
        JoystickTorsoConfig.from_yaml(args.config) if args.torso_stick else None
    )
    args.rate = vt.cmd_rate
    if args.rate <= 0:
        raise ValueError(f"wbik.yaml vr_teleop.cmd_rate must be > 0, got {args.rate}")
    if args.headset_hud and (not np.isfinite(args.hud_rate) or args.hud_rate <= 0.0):
        parser.error("--hud-rate must be finite and > 0 when --headset-hud is enabled")

    leader = JoystickVRLeader(args)
    if args.calibrate_ee_offset:
        leader.calibrate_ee_offset(args.ee_capture_seconds, args.ee_offset_file)
    else:
        leader.run()


if __name__ == "__main__":
    main()
