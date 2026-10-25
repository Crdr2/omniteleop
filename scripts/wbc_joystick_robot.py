#!/usr/bin/env python3
"""Drive and record the real robot with joystick chassis control and arm IK.

Home with left index while idle before each engagement. Right stick drives and
left stick turns. Optional torso modes must match the leader. Gripper contact
latching and stale-feedback holds are enabled by default. See README.md.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from omniteleop.follower.joystick_base import JoystickBaseShaper
from omniteleop.follower.joystick_torso import JoystickTorsoCommand
from omniteleop.follower.whole_body_ik import VegaWholeBodyIK, WBCConfig
from omniteleop.wbc_policy_format import (
    JOYSTICK_FIXED_SPEED_MAPPING,
    JOYSTICK_POLICY_ACTION_SCHEMA,
    JOYSTICK_RAW_EPISODE_SCHEMA_V2,
    JOYSTICK_TORSO_IK_RAW_EPISODE_SCHEMA,
    JOYSTICK_TORSO_RAW_EPISODE_SCHEMA,
)
from omniteleop.wbc_robot_util import base_quiet_dispatch, parse_enable_mask
from omniteleop.wbc_teleop import (
    JoystickTeleopConfig,
    JoystickTorsoConfig,
    VRTeleopConfig,
    bind_joystick_teleop_args,
    bind_vr_teleop_args,
)

# Load the base real follower as a module (scripts/ is not a package; tests do this too).
# Importing binds module-level constants; main() is __main__-guarded, so nothing runs.
_BASE_PATH = Path(__file__).resolve().parent / "wbc_vr_robot.py"
_spec = importlib.util.spec_from_file_location("wbc_vr_robot", _BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

HardwareDriver = _base.HardwareDriver
run_loop = _base.run_loop
_TrajLog = _base._TrajLog
VRJointSubscriber = _base.VRJointSubscriber
ReplaySource = _base.ReplaySource
_publish_abort_status = _base._publish_abort_status
peek_next_episode_id = _base.peek_next_episode_id

_JOYSTICK_RAW_EPISODE_SCHEMA = JOYSTICK_RAW_EPISODE_SCHEMA_V2
# Raw schema of a take by torso mode: fixed (""), --torso-stick, --torso-ik.
_RAW_SCHEMA_BY_TORSO_MODE = {
    "": _JOYSTICK_RAW_EPISODE_SCHEMA,
    "stick": JOYSTICK_TORSO_RAW_EPISODE_SCHEMA,
    "ik": JOYSTICK_TORSO_IK_RAW_EPISODE_SCHEMA,
}
_TORSO_MODE_FLAG = {"": "no torso flag", "stick": "--torso-stick", "ik": "--torso-ik"}


class JoystickHardwareDriver(HardwareDriver):
    """``HardwareDriver`` whose base is joystick-driven (not from the whole-body QP).

    Everything except the base -- homing, the per-joint clamp/dispatch, grippers, odometry,
    recording -- is inherited unchanged. ``_drive_base`` is overridden to shape the leader's
    ``chassis_*`` joystick direction through :class:`JoystickBaseShaper` (whose integrated pose
    reference is odometry-PD tracked and single-axis projected exactly like
    the WBC base path) and dispatch it with the SAME swerve quiet-hold logic as the parent.

    With ``--torso-stick`` it also owns the torso command (``torso_targets`` /
    ``advance_torso``, called by the shared run_loop); with ``--torso-ik`` the solver owns
    the torso and only the recording contract differs. The class-level defaults keep drivers
    built without either (the policy rollout, tests) exactly fixed-torso.
    """

    _torso_mode = ""  # "", "stick" (--torso-stick) or "ik" (--torso-ik)
    _live_leader = False  # driven by a live leader, whose torso mode must match
    _torso_mode_warned = False
    _torso: Optional[JoystickTorsoCommand] = None
    _torso_replay = False  # episode replay: targets already carry the torso height
    _torso_row: Optional[dict] = None  # this tick's recorded torso fields
    _torso_engage_hold = False  # engaged with the torso off nominal: hold until homed
    _require_explicit_home = False
    _home_ready = False
    _home_session_active = False
    _gripper_speed = 1.0  # Robotiq speed/force; __init__ reads --gripper-speed/-force
    _gripper_force = 1.0
    _gripper_contact_latch = False
    _GRIPPER_RELEASE_THRESHOLD = 0.1
    _GRIPPER_LATCH_MAX_STATUS_AGE_S = 0.25
    _fixed_grippers_only = False  # hold tick: write only the fixed-position sides

    def __init__(self, args, ik, cfg, enable, source, traj=None) -> None:
        # The joystick twist is read from the live command source each base tick; the shaper
        # carries the integrator / slew / single-axis state. Set BEFORE super().__init__ so a
        # partial-init teardown (close()) never sees them unset.
        self._source = source
        self._require_explicit_home = source is not None and not getattr(args, "replay", None)
        self._live_leader = self._require_explicit_home
        self._torso_mode = (
            "stick"
            if getattr(args, "torso_stick", False)
            else "ik"
            if getattr(args, "torso_ik", False)
            else ""
        )
        self._home_ready = False
        self._home_session_active = False
        self._latched_source_sample = None
        self._shaper = JoystickBaseShaper.from_configs(cfg, args)
        if getattr(args, "torso_stick", False):
            self._torso = JoystickTorsoCommand.from_ik(ik, args.joystick_torso_speed)
            self._torso_replay = getattr(source, "replay_kind", "") == "joystick_episode"
        # The parent snapshots recording metadata during construction. Install these
        # fields first so it records the selected settings, not the class defaults.
        self._gripper_speed = float(getattr(args, "gripper_speed", 1.0))
        self._gripper_force = float(getattr(args, "gripper_force", 1.0))
        self._gripper_contact_latch = bool(getattr(args, "gripper_contact_latch", False))
        self._gripper_max_position = {
            side: float(getattr(args, f"{side}_gripper_max_position", 1.0))
            for side in ("left", "right")
        }
        self._gripper_fixed_position = {
            side: getattr(args, f"{side}_gripper_fixed_position", None)
            for side in ("left", "right")
        }
        self._gripper_hold_position: dict[str, Optional[float]] = {"left": None, "right": None}
        self._gripper_release_pending: set[str] = set()
        super().__init__(args, ik, cfg, enable, traj=traj)
        # Bind after the parent installs the builder. Rollouts without these flags keep
        # the existing full-speed/full-force behavior.
        self._build_hande = functools.partial(
            self._build_hande, speed=self._gripper_speed, force=self._gripper_force
        )

    def _gripper_status_poll_period_s(self) -> float:
        period = super()._gripper_status_poll_period_s()
        # Contact protection must not slow down when collecting at a low record rate.
        return min(period, 0.1) if self._gripper_contact_latch else period

    def _prepare_gripper_command(
        self, side: str, position: float, *, now_monotonic: float
    ) -> Optional[tuple[float, bytes]]:
        fixed = getattr(self, "_gripper_fixed_position", {}).get(side)
        if fixed is None and self._fixed_grippers_only:
            return None
        if fixed is not None:
            position = fixed
        cap = getattr(self, "_gripper_max_position", {}).get(side, 1.0)
        target = min(position, np.floor(cap * 255) / 255)
        if not self._gripper_contact_latch:
            return target, self._build_hande(target)

        hold = self._gripper_hold_position[side]
        last = self._last_gripper_wire_message[side]
        # A release caught during a bus guard remains pending until its OPEN is sent.
        # Compare the raw trigger, not a capped target (a small cap is not a release).
        released = (
            position == 0.0 if fixed is not None else position <= self._GRIPPER_RELEASE_THRESHOLD
        )
        if released and (hold is not None or last is None or last[7] != 0x09 or last[10] != 0):
            self._gripper_release_pending.add(side)
        if side in self._gripper_release_pending or released:
            return 0.0, self._build_hande(0.0)

        if hold is None:
            status = self._last_grip_status.get(side) or {}
            read_t = float(status.get("read_monotonic_ns", -1)) / 1e9
            request_t = float(status.get("request_send_monotonic_ns", -1)) / 1e9
            healthy = (
                request_t > 0.0
                and 0.0 <= read_t - request_t <= _base._GRIPPER_STATUS_REQUEST_TIMEOUT_S
                and 0.0 <= now_monotonic - read_t <= self._GRIPPER_LATCH_MAX_STATUS_AGE_S
                and status.get("gACT") == 1
                and status.get("gSTA_bits") == 3
                and status.get("gFLT") == 0
                and status.get("kFLT") == 0
                and 0 <= status.get("gPO", -1) <= 255
                and 0 <= status.get("gPR", -1) <= 255
            )
            if not healthy and last is None and self._fixed_grippers_only:
                # Startup hold: the only status is the warm-up read from before the
                # startup home. Wait for a fresh reply before the first fixed go-to
                # instead of latching a stop that a fixed side cannot release.
                return None
            contact = (
                healthy
                and last is not None
                and bool(last[7] & 0x08)
                and request_t >= self._last_gripper_wire_change_monotonic[side]
                and status.get("gGTO") == 1
                and status.get("gOBJ") == 2
                and status["gPR"] == last[10]
                and status["gPO"] < status["gPR"]
            )
            if contact or not healthy:
                # Stop at contact without issuing a go-to/backoff. When feedback is
                # unavailable, the target field is informational: rGTO=0 stops motion.
                hold = min(status["gPO"] / 255 if healthy else target, np.floor(cap * 255) / 255)
                self._gripper_hold_position[side] = hold
                reason = "closing contact" if contact else "missing/stale/invalid feedback"
                recovery = (
                    "Fixed position: restart to re-arm."
                    if fixed is not None
                    else "Release its trigger to <=0.1 to open and re-arm."
                )
                print(f"\n[wbc_joystick_robot] {side} gripper HOLD: {reason}. {recovery}")
        if hold is not None:
            # Clearing rGTO also prevents firmware re-grasp when force is nonzero.
            message = self._build_hande(hold, go_to=False)
            return None if message == last else (hold, message)
        return target, self._build_hande(target)

    def actuate(self, result, left_gripper, right_gripper, enable, hold, dt) -> None:
        super().actuate(result, left_gripper, right_gripper, enable, hold, dt)
        # A hold (e.g. before the left-trigger home) sends no gripper writes, so a fixed
        # side would sit wherever activation left it. Drive fixed sides there anyway;
        # trigger-driven sides stay untouched until teleop, as before.
        fixed = getattr(self, "_gripper_fixed_position", {})
        if hold and any(v is not None for v in fixed.values()):
            self._fixed_grippers_only = True
            try:
                self._command_grippers(left_gripper, right_gripper)
            finally:
                self._fixed_grippers_only = False

    def _gripper_command_sent(self, side: str, position: float, message: bytes) -> None:
        if self._gripper_contact_latch and side in self._gripper_release_pending:
            self._gripper_hold_position[side] = None
            self._gripper_release_pending.remove(side)

    def home_to_nominal(self) -> None:
        """Home enabled joints, then synchronize the joystick torso command."""
        self._home_ready = False
        super().home_to_nominal()
        # Only clear the command after the measured homing gate succeeds. A failed
        # or disabled torso home must not pretend that the torso reached nominal.
        if self._torso is not None and self.enable["torso"]:
            self._torso.reset()
            self._torso_row = None
            self._torso_engage_hold = False
        self._home_ready = True
        self._home_session_active = False

    def home_required(self) -> bool:
        """Live teleop needs an explicit successful home before each session."""
        return self._require_explicit_home and not self._home_ready

    def start_recording_if_teleop(self, vr) -> None:
        if not self.home_required():
            super().start_recording_if_teleop(vr)

    def engage_reset(self, ik, left0, right0, head0) -> None:
        """Zero the joystick reference in lock-step with the odom origin at engage."""
        super().engage_reset(ik, left0, right0, head0)  # resets odom origin to (0,0,0)
        self._shaper.reset(np.zeros(3))
        if self._torso is not None:
            # The torso command restarts at nominal. A torso left low by the last take
            # would jump back at full hardware speed, so hold until it is homed instead.
            # (Compared with the command, not ik.q: an aligned engage seeds the IK from the
            # measured pose, and the next solve still writes the command.)
            self._torso.reset()
            measured = self._read_measured_joints()["torso"]
            off = (
                np.inf
                if measured is None
                else float(np.max(np.abs(measured - self._torso.joints())))
            )
            self._torso_engage_hold = off > self.args.home_tol
            if self._torso_engage_hold:
                print(
                    f"\n[wbc_joystick_robot] torso is {off:.2f} rad from nominal: holding. "
                    "Disengage and home it (left trigger) before engaging."
                )

    def extra_hold(self, now, last_cmd_wall, estop) -> Optional[str]:
        """The parent's holds, plus the leader torso-mode match and the engage torso guard."""
        vr = self._latched_source_sample
        if self._live_leader and vr is not None:
            leader_mode = getattr(vr, "torso_mode", "")
            if leader_mode != self._torso_mode:
                if not self._torso_mode_warned:
                    self._torso_mode_warned = True
                    print(
                        f"\n[wbc_joystick_robot] the leader runs "
                        f"{_TORSO_MODE_FLAG.get(leader_mode, repr(leader_mode))} but this "
                        f"follower runs {_TORSO_MODE_FLAG[self._torso_mode]}: holding. Restart "
                        "the leader with the same torso flag."
                    )
                return "leader-torso-mode"
        if self.home_required():
            return "home-required"
        reason = super().extra_hold(now, last_cmd_wall, estop)
        if reason is None and self._torso_engage_hold:
            return "torso-not-homed"
        return reason

    def torso_targets(self, left_target, right_target):
        """Torso joints for this tick, and the live hand targets riding the torso height."""
        if self._torso is None:
            return left_target, right_target, None
        if not self._torso_replay:
            left_target = np.array(left_target, dtype=float)
            right_target = np.array(right_target, dtype=float)
            left_target[2, 3] += self._torso.height
            right_target[2, 3] += self._torso.height
        return left_target, right_target, self._torso.joints()

    def advance_torso(self, ik, hold, dt) -> None:
        """With --torso-stick, note this tick's torso fields, then step the height the next
        solve uses.
        """
        if self._torso is None:
            return
        vr = self._latched_source_sample
        height = self._torso.height  # the height this tick was solved (and recorded) at
        was_blocked, was_moving = self._torso.blocked, self._torso.intent != 0.0
        if hold:
            self._torso.hold()
        elif self._torso_replay and hasattr(vr, "torso_height"):
            self._torso.track(float(vr.torso_height), dt, ik.q, ik.self_collision_pair_distances)
        else:
            self._torso.step(
                float(getattr(vr, "torso_vz", 0.0)), dt, ik.q, ik.self_collision_pair_distances
            )
        self._torso_row = {
            "intent": self._torso.intent,
            "height": height,
            "blocked": self._torso.blocked,
        }
        if self._torso.blocked and not was_blocked:
            print(
                f"\n[wbc_joystick_robot] torso stopped at {self._torso.height:+.3f} m: going "
                f"on would bring an arm within {self._torso.gate_distance * 100:.0f} cm of "
                "the phone mount, base or body."
            )
        elif was_moving and self._torso.intent == 0.0:
            print(f"\n[wbc_joystick_robot] torso at {self._torso.height:+.3f} m.")

    def latch_source_sample(self, vr) -> None:
        """Latch the command object already selected by the shared arm/head loop."""
        self._latched_source_sample = vr
        if not self._require_explicit_home or vr is None:
            return
        if getattr(vr, "estop", True):
            if self._home_session_active:
                self._home_ready = False
            self._home_session_active = False
        elif not self._home_session_active:
            # Validate physical posture BEFORE the shared loop resets IK to nominal.
            # Never re-arm from a later good sample: another explicit home is required.
            if self._home_ready:
                measured = self._read_measured_joints()
                sweep = getattr(self, "_scenediff_session", None)
                reference = (
                    sweep.alignment_joints
                    if sweep is not None and sweep.state == "aligned"
                    else self._nominal
                )
                for group, names in self._joint_names.items():
                    enabled = self.enable["arms" if group.endswith("_arm") else group]
                    if not enabled:
                        continue
                    values = measured.get(group)
                    if (
                        values is None
                        or np.shape(values) != (len(names),)
                        or not np.all(np.isfinite(values))
                        or np.max(np.abs(values - reference[group])) > self.args.home_tol
                    ):
                        self._home_ready = False
                        break
            self._home_session_active = True

    def latch_chassis_intent(self, chassis) -> None:
        """Install one policy-produced chassis intent without fabricating a VR frame."""
        value = np.asarray(chassis, dtype=float).reshape(-1)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"chassis intent must be finite (3,), got {value!r}")
        self._latched_source_sample = argparse.Namespace(
            chassis_vx=float(value[0]),
            chassis_vy=float(value[1]),
            chassis_wz=float(value[2]),
        )

    def _read_joystick(self) -> np.ndarray:
        """Latched leader/policy body twist ``(vx, vy, wz)`` (zero before the first)."""
        vr = self._latched_source_sample
        if vr is None:
            return np.zeros(3)
        return np.array(
            [float(vr.chassis_vx), float(vr.chassis_vy), float(vr.chassis_wz)], dtype=float
        )

    def _drive_base(self, result, hold, enable, dt) -> None:
        # No chassis / base disabled: defer to the parent (sets the off-state /debug fields).
        if self._odom is None or not enable["base"]:
            self._dbg["rx_chassis"] = None
            self._dbg["projected_chassis"] = None
            self._dbg["joystick_axis"] = -1
            super()._drive_base(result, hold, enable, dt)
            return
        chassis = self.robot.chassis
        if hold:
            self._shaper.cancel_turn()
            # A hold must not demand a finite pose: under --arkit-base control the pose
            # is legitimately NaN before an engage captures the origin (extra_hold's
            # "arkit-no-origin"), and that hold only clears via engage_reset, which
            # re-zeros the shaper anyway. Read tolerantly here; the strict
            # policy_base_pose_snapshot() below stays reserved for active driving.
            snap = self._odom.snapshot()
            arkit = self._arkit.snapshot() if self._arkit is not None else None
            self._dbg["odom"] = snap
            self._dbg["arkit"] = arkit
            selected = arkit if self._arkit_drives_base() else snap
            measured_pose = np.asarray(selected["pose"], dtype=np.float64)
            # Park the joystick reference on the measured base so there is no lunge on
            # resume, and zero the wheels (the parent's hold behavior).
            if measured_pose.shape == (3,) and np.all(np.isfinite(measured_pose)):
                self._shaper.hold(measured_pose)
            chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0)
            self._prev_base_cmd = np.zeros(3)
            self._base_quiet_elapsed = 0.0
            self._dbg["base_pd_raw"] = None
            self._dbg["base_pd_err"] = None
            self._dbg["base_cmd"] = np.zeros(3)
            self._dbg["base_action"] = "safety_hold"
            self._dbg["rx_chassis"] = np.zeros(3)
            self._dbg["projected_chassis"] = np.zeros(3)
            self._dbg["joystick_axis"] = -1
            return
        # Active driving: extra_hold() has already forced a hold (and returned above) if
        # the pose source is stale, jumped, or has no origin, so the strict finiteness
        # check here can only trip on a genuine sensor fault.
        base_state = self.policy_base_pose_snapshot()
        measured_pose = base_state["pose"]
        self._dbg["odom"] = base_state["odom"]
        self._dbg["arkit"] = base_state["arkit"]
        rx_chassis = self._read_joystick()
        cmd, pd_raw, pd_err = self._shaper.step(rx_chassis, measured_pose, dt)
        if self.cfg.enable_base_single_axis and np.count_nonzero(np.abs(cmd) > 1e-9) > 1:
            raise RuntimeError(
                f"single-axis joystick shaper emitted multi-axis hardware command {cmd}"
            )
        self._prev_base_cmd = cmd
        # Dispatch via the SAME swerve quiet-hold path as HardwareDriver._drive_base: on a
        # quiet command hold the current steering with zero drive (avoid the lateral->forward
        # wheel snap set_velocity(0,0,0) causes on brief intra-motion dips); otherwise send
        # the twist. Only the shaping source differs (joystick, above) -- the dispatch and its
        # safety envelope (set_velocity / a zero-drive wheel hold) are unchanged.
        action, self._base_quiet_elapsed = base_quiet_dispatch(
            cmd, self._base_quiet_elapsed, dt, self.args.base_quiet_hold_s
        )
        if action == "hold":
            steer = np.asarray(chassis.steering_angle, dtype=float)
            if steer.shape == (2,) and np.all(np.isfinite(steer)):
                assert float(np.max(np.abs(cmd))) <= 1e-6  # base_quiet_dispatch invariant
                chassis.set_wheel_velocity(0.0)
            else:
                chassis.set_velocity(vx=0.0, vy=0.0, wz=0.0, wait_time=0.0)
                action = "recenter"
        else:
            chassis.set_velocity(
                vx=float(cmd[0]), vy=float(cmd[1]), wz=float(cmd[2]), wait_time=0.0
            )
        self._dbg["base_pd_raw"] = pd_raw
        self._dbg["base_pd_err"] = pd_err
        self._dbg["base_cmd"] = np.asarray(cmd, dtype=float)
        self._dbg["base_action"] = action
        self._dbg["rx_chassis"] = rx_chassis
        self._dbg["projected_chassis"] = self._shaper.last_projected_joystick.copy()
        self._dbg["joystick_axis"] = (
            -1 if self._shaper.last_intent_axis is None else self._shaper.last_intent_axis
        )

    def debug_row(self, result) -> dict:
        """Extend the inherited WBC row with the direct joystick reference chain."""
        row = super().debug_row(result)

        def vec(value):
            return (
                np.asarray(value, dtype=np.float32)
                if value is not None
                else np.full(3, np.nan, dtype=np.float32)
            )

        row["rx_chassis"] = vec(self._dbg.get("rx_chassis"))
        row["projected_chassis"] = vec(self._dbg.get("projected_chassis"))
        row["joystick_cmd_pose"] = np.asarray(self._shaper.cmd_pose, dtype=np.float32)
        row["joystick_axis"] = np.int64(self._dbg.get("joystick_axis", -1))
        if self._torso is not None:
            row["torso_height"] = np.float32(self._torso.height)
            row["torso_intent"] = np.float32(self._torso.intent)
            row["torso_blocked"] = np.bool_(self._torso.blocked)
        return row

    def _recording_extra_action(self) -> dict:
        """Keep policy intent and controller state beside the applied base command.

        ``action/chassis/intent_body`` is the 32-D joystick policy label: the
        follower-effective fixed-speed intent after DOF masking/single-axis projection and before
        integration, odometry feedback, slew limiting, or hardware dispatch. The parent
        already records the shaped command actually sent as ``action/joint/chassis_*``.
        """
        source = np.asarray(self._dbg.get("rx_chassis"), dtype=np.float32)
        intent = np.asarray(self._dbg.get("projected_chassis"), dtype=np.float32)
        pd_raw = np.asarray(self._dbg.get("base_pd_raw"), dtype=np.float32)
        pd_error = np.asarray(self._dbg.get("base_pd_err"), dtype=np.float32)
        values = {
            "source_body": source,
            "intent_body": intent,
            "command_pose": np.asarray(self._shaper.cmd_pose, dtype=np.float32),
            "pd_raw_body": pd_raw,
            "pd_error_body": pd_error,
            "prev_shaped_body": np.asarray(self._shaper.prev_shaped, dtype=np.float32),
        }
        bad = {
            name: value.shape
            for name, value in values.items()
            if value.shape != (3,) or not np.all(np.isfinite(value))
        }
        if bad:
            raise RuntimeError(f"cannot record malformed joystick action fields: {bad}")
        values["active_axis"] = np.int8(self._dbg.get("joystick_axis", -1))
        target_yaw = getattr(self._shaper, "turn_target_yaw", None)
        values["turn_active"] = np.bool_(target_yaw is not None)
        values["turn_target_yaw"] = np.float64(
            self._shaper.cmd_pose[2] if target_yaw is None else target_yaw
        )
        out = {"chassis": values}
        if self._gripper_contact_latch:
            # A stop has a position field but does not drive to it. Persist the actual
            # wire action bit so an offline reader can distinguish a hold from go-to.
            out["gripper_control"] = {
                side: {
                    "go_to": np.int8(
                        -1
                        if self._last_gripper_wire_message[side] is None
                        else bool(self._last_gripper_wire_message[side][7] & 0x08)
                    ),
                    "hold_latched": np.bool_(self._gripper_hold_position[side] is not None),
                    "release_pending": np.bool_(side in self._gripper_release_pending),
                }
                for side in ("left", "right")
            }
        if self._torso is None:
            return out
        # --torso-stick: the stick intent, the height this row was solved at (its hand
        # targets include it), and whether the collision gate refused the step.
        row = self._torso_row or {"intent": 0.0, "height": self._torso.height, "blocked": False}
        out["torso"] = {
            "intent": np.float32(row["intent"]),
            "height": np.float32(row["height"]),
            "blocked": np.bool_(row["blocked"]),
        }
        return out

    def _recording_control_metadata(self) -> dict:
        out = super()._recording_control_metadata()
        if getattr(self.args, "policy_path", None):
            demonstration_source = "policy_rollout"
        elif getattr(self.args, "replay_episode", None):
            demonstration_source = "policy_episode_replay"
        elif getattr(self.args, "replay", None):
            demonstration_source = "joystick_episode_replay"
        else:
            demonstration_source = "human_teleoperation"
        out.update(
            {
                "schema": np.bytes_(_JOYSTICK_RAW_EPISODE_SCHEMA),
                "control_mode": np.bytes_("joystick"),
                "demonstration_source": np.bytes_(demonstration_source),
                "policy_action_schema": np.bytes_(JOYSTICK_POLICY_ACTION_SCHEMA),
                "action_target_frame": np.bytes_("current_base"),
                "eef_target_frame": np.bytes_("current_base"),
                "head_target_frame": np.bytes_("current_base"),
                "head_target_semantics": np.bytes_("view_direction_pan_tilt_position_roll_ignored"),
                "base_reference_source": np.bytes_("joystick_integrator"),
                "action_base_pose_semantics": np.bytes_("joystick_command_reference"),
                "chassis_intent_path": np.bytes_("action/chassis/intent_body"),
                "chassis_intent_frame": np.bytes_("current_base_body"),
                "chassis_intent_units": np.bytes_("m_per_s,m_per_s,rad_per_s"),
                "chassis_intent_stage": np.bytes_("post_mask_projection_pre_controller"),
                "chassis_intent_temporal_semantics": np.bytes_("zero_order_hold"),
                "chassis_intent_mapping": np.bytes_(JOYSTICK_FIXED_SPEED_MAPPING),
                "chassis_translation_speed_mps": np.float64(self._shaper.translation_speed),
                "chassis_rotation_speed_radps": np.float64(self._shaper.rotation_speed),
                "chassis_reference_lead_m": np.float64(self._shaper.reference_lead_xy),
                "chassis_reference_lead_rad": np.float64(self._shaper.reference_lead_yaw),
                "applied_chassis_group": np.bytes_("action/joint"),
                "measured_chassis_group": np.bytes_("obs/joint"),
                "chassis_dataset_axes": np.asarray([b"chassis_vx", b"chassis_vy", b"chassis_wz"]),
            }
        )
        out["gripper_speed"] = np.float64(self._gripper_speed)
        out["gripper_force"] = np.float64(self._gripper_force)
        out["gripper_contact_latch"] = np.bool_(self._gripper_contact_latch)
        for side in ("left", "right"):
            fixed = getattr(self, "_gripper_fixed_position", {}).get(side)
            out[f"{side}_gripper_fixed_position_enabled"] = np.bool_(fixed is not None)
            if fixed is not None:
                out[f"{side}_gripper_fixed_position"] = np.float64(fixed)
            out[f"{side}_gripper_max_position"] = np.float64(
                getattr(self, "_gripper_max_position", {}).get(side, 1.0)
            )
        if self._gripper_contact_latch:
            out["gripper_release_threshold"] = np.float64(self._GRIPPER_RELEASE_THRESHOLD)
            out["gripper_latch_max_status_age_s"] = np.float64(self._GRIPPER_LATCH_MAX_STATUS_AGE_S)
            out["gripper_go_to_path"] = np.bytes_("action/gripper_control/{side}/go_to")
        out["chassis_turn_step_deg"] = np.float64(90 if self._shaper.turn_90 else 0)
        if self._shaper.turn_90:
            out["chassis_turn_stop_semantics"] = np.bytes_(
                "yaw_slew_to_zero_park_pose_until_new_intent/v1"
            )
            out["base_reference_source"] = np.bytes_("joystick_integrator_with_quarter_turn_goal")
            out["chassis_intent_stage"] = np.bytes_(
                "post_mask_projection_turn_expansion_pre_controller"
            )
            out["chassis_turn_tolerance_rad"] = np.float64(self._shaper.TURN_TOLERANCE_RAD)
            out["chassis_turn_trigger"] = np.bytes_(
                "neutral_then_yaw_no_repeat_until_neutral_after_turn"
            )
        if self._torso_mode == "ik":
            out.update(
                {
                    "schema": np.bytes_(JOYSTICK_TORSO_IK_RAW_EPISODE_SCHEMA),
                    # The QP moved the torso to reach action/eef/*; what it commanded (post
                    # clamp) is action/joint/torso. Its settings are in the wbik.yaml snapshot.
                    "torso_control": np.bytes_("whole_body_ik/v1"),
                    "torso_action_path": np.bytes_("action/joint/torso"),
                }
            )
        if self._torso is not None:
            path = self._torso.path
            out.update(
                {
                    "schema": np.bytes_(JOYSTICK_TORSO_RAW_EPISODE_SCHEMA),
                    # Hand targets (action/eef/*) already include the torso height.
                    "torso_control": np.bytes_("stick_height_hands_follow/v1"),
                    "torso_path": np.bytes_("torso_l3_vertical_translation/v1"),
                    "torso_height_path": np.bytes_("action/torso/height"),
                    "torso_intent_path": np.bytes_("action/torso/intent"),
                    "torso_speed_mps": np.float64(self._torso.speed),
                    "torso_height_range_m": np.asarray(
                        [path.min_height, path.max_height], dtype=np.float64
                    ),
                    "torso_gate_distance_m": np.float64(self._torso.gate_distance),
                }
            )
        return out

    def _recording_provenance_metadata(self) -> dict:
        """Extend inherited provenance with the actual joystick entry points/config."""
        out = super()._recording_provenance_metadata()
        out["schema"] = np.asarray(_RAW_SCHEMA_BY_TORSO_MODE[self._torso_mode].encode())
        repo = _BASE_PATH.parents[1]
        configured = getattr(self.args, "config", None)
        selected_config = (
            Path(configured).expanduser().resolve()
            if configured is not None
            else Path(_base.DEFAULT_CONFIG_PATH).resolve()
        )
        paths = {
            "wbc_joystick_robot": Path(__file__).resolve(),
            "wbc_joystick_leader": repo / "scripts/wbc_joystick_leader.py",
            "joystick_base": repo / "src/omniteleop/follower/joystick_base.py",
            "scenediff_collection": repo / "src/omniteleop/wbc_scenediff_collection.py",
            "wbc_record": repo / "src/omniteleop/wbc_record.py",
            "selected_wbik_yaml": selected_config,
        }
        if self._torso is not None:
            paths["joystick_torso"] = repo / "src/omniteleop/follower/joystick_torso.py"
        snapshots = out["source_snapshot"]
        for label, path in paths.items():
            if not path.is_file():
                raise RuntimeError(
                    f"[wbc_joystick_robot] provenance source file is missing: {path}"
                )
            snapshots[label] = {
                "path": np.asarray(
                    str(path.relative_to(repo) if path.is_relative_to(repo) else path).encode()
                ),
                "sha256": np.asarray(_base._sha256_file(path).encode()),
                "content": np.asarray(path.read_bytes()),
            }
        return out

    def record_tick(self, result, hold, *args, **kwargs):
        """Record the JOYSTICK reference pose as action/base/pose (not the locked origin).

        With the base excluded from the IK, ``result.base_pose`` is a constant origin, so it
        would make ``action/base/pose`` useless. Substitute the shaper's integrated command
        pose -- the base trajectory the operator commanded -- which is what downstream tools
        want for controller diagnostics (the 32-D policy label itself comes from
        ``action/chassis/intent_body``; ``vis_episode`` uses obs/base/pose from odometry).
        The semantics of this field therefore differ from the WBC follower's
        solver-commanded ``result.base_pose``.

        The substitution (and its allocation) is skipped whenever the base recorder would
        drop the frame anyway -- on ``hold`` or when not recording -- so this stays free on
        the 100 Hz loop in the common no-record case.
        """
        if not hold and self._episode is not None and self._episode.recording:
            result = replace(result, base_pose=np.asarray(self._shaper.cmd_pose, dtype=float))
        return super().record_tick(result, hold, *args, **kwargs)


def _build_joystick_ik(
    config: Optional[str],
    *,
    lock_torso_in_ik: Optional[bool] = None,
    torso_ik: bool = False,
) -> tuple[VegaWholeBodyIK, WBCConfig]:
    """Build the IK with joystick-mode overrides on top of the canonical wbik.yaml.

    ``torso_ik`` (``--torso-ik``, with ``lock_torso_in_ik=False``) fixes the head -- the
    operator's head maps to nothing.
    """
    cfg = WBCConfig.from_yaml(config)
    if lock_torso_in_ik is not None:
        cfg.lock_torso_in_ik = bool(lock_torso_in_ik)
    cfg = replace(
        cfg,
        lock_base_in_ik=True,  # base excluded from the whole-body IK (joystick-driven)
        # joystick_teleop.head_mode: "track" (neck pan/tilt follows the headset; the base
        # never follows it) or "fixed" (head held at nominal for the whole take). The
        # top-level head_mode ("ik" couples the base to the head) is deliberately ignored.
        head_mode=JoystickTeleopConfig.from_yaml(config).head_mode,
    )
    if torso_ik:
        cfg = replace(cfg, head_mode="fixed")
    ik = VegaWholeBodyIK(cfg)
    ik.reset()
    return ik, cfg


def _quarantine_failed_recording(driver, exc: BaseException) -> None:
    """Keep an exception-truncated joystick take out of the training-data glob."""
    if driver is None:
        return
    try:
        driver.abort_recording_episode(str(exc))
    except Exception as quarantine_exc:
        print(
            "[wbc_joystick_robot] WARNING: failed to quarantine the aborted "
            f"episode: {quarantine_exc}"
        )
        # Match the whole-body follower's last-resort behavior: if quarantine itself
        # fails, discard instead of letting close() publish the take as complete.
        episode = getattr(driver, "_episode", None)
        if episode is not None and getattr(episode, "recording", False):
            try:
                episode.discard()
            except Exception as discard_exc:
                print(
                    "[wbc_joystick_robot] WARNING: failed to discard the "
                    f"unquarantined take: {discard_exc}"
                )


def _run_joystick_ik_mode(args: argparse.Namespace, enable: dict) -> None:
    """Replay or live-drive the joystick follower (mirrors wbc_vr_robot._run_ik_mode)."""
    torso_stick = bool(getattr(args, "torso_stick", False))
    torso_ik = bool(getattr(args, "torso_ik", False))
    torso_mode = "stick" if torso_stick else "ik" if torso_ik else ""
    # The torso mode alone decides the torso lock: fixed and stick pin it, --torso-ik frees
    # it (see _build_joystick_ik for the rest of that mode).
    ik, cfg = _build_joystick_ik(args.config, lock_torso_in_ik=not torso_ik, torso_ik=torso_ik)
    print(
        f"[wbc_joystick_robot] model nq={ik.model.nq} nv={ik.model.nv} "
        f"head_mode={cfg.head_mode} base_dofs={cfg.base_dofs} "
        f"lock_base_in_ik={cfg.lock_base_in_ik} "
        f"lock_torso_in_ik={cfg.lock_torso_in_ik} "
        f"urdf={cfg.urdf_path.rsplit('/', 1)[-1]}"
    )
    replay = args.replay is not None
    clock: Callable[[], float] = time.perf_counter

    if replay:
        source = ReplaySource.from_joystick_hdf5(
            args.replay, args.speed, clock=clock, torso_mode=torso_mode
        )
        first = source._vr[0]
        if hasattr(first, "torso_height") and (abs(first.torso_height) > 0.01 or args.speed > 1.0):
            # The torso tracks recorded heights from nominal at no more than
            # joystick_torso.speed, while the recorded hand targets already carry them.
            print(
                f"[wbc_joystick_robot] WARNING: this torso take starts at "
                f"{first.torso_height:+.2f} m and replays at {args.speed:g}x: the torso lags "
                "the recorded height until it catches up, and the arms reach meanwhile."
            )
        print(
            f"[wbc_joystick_robot] replay {args.replay}: {len(source._vr)} frames, "
            f"{source.total_duration:.1f}s at {args.speed:g}x "
            f"({source.replay_kind}; head targets "
            f"{'already filtered' if source.head_targets_pre_filtered else 'filtered live'})"
        )
    else:
        source = VRJointSubscriber(args.namespace, name="wbc_joystick_robot")
        print(f"[wbc_joystick_robot] live on '{source.topic}'. Ctrl-C to stop.")

    enabled = [k for k, v in enable.items() if v]
    traj: Optional[_TrajLog] = None
    debug_start_id = 0
    if args.debug_dir is not None:
        import os

        os.makedirs(args.debug_dir, exist_ok=True)
        debug_start_id = (
            peek_next_episode_id(args.save_dir)
            if args.record
            else peek_next_episode_id(args.debug_dir, suffix="_debug")
        )
        meta = {
            "episode_id": int(debug_start_id),
            "mode": "replay" if replay else "live",
            "control": "joystick",
            "speed": float(args.speed) if replay else 1.0,
            "ik_rate": float(args.ik_rate),
            "cmd_rate": float(args.cmd_rate),
            "enable": ",".join(enabled),
            "urdf": cfg.urdf_path.rsplit("/", 1)[-1],
            "base_dofs": cfg.base_dofs,
            "lock_base_in_ik": True,
            "lock_torso_in_ik": bool(cfg.lock_torso_in_ik),
            "eef_target_frame": "current_base",
            "head_target_frame": "current_base",
            "policy_action_schema": JOYSTICK_POLICY_ACTION_SCHEMA,
            "base_reference_source": "joystick_integrator",
            "base_kp_xy": float(args.base_kp_xy),
            "base_kp_yaw": float(args.base_kp_yaw),
            "base_yaw_hold_in_xy": bool(args.base_yaw_hold_in_xy),
            "base_deadband": float(args.base_deadband),
            "base_accel": float(args.base_accel),
            "base_max_speed": float(args.base_max_speed),
            "base_quiet_hold_s": float(args.base_quiet_hold_s),
            "source_timeout": float(args.source_timeout),
            "max_joint_step": float(args.max_joint_step),
            "head_lpf_tau": float(args.head_lpf_tau),
            "joystick_stick_max_vx": float(args.joystick_stick_max_vx),
            "joystick_stick_max_vy": float(args.joystick_stick_max_vy),
            "joystick_stick_max_wz": float(args.joystick_stick_max_wz),
            "joystick_translation_speed": float(args.joystick_translation_speed),
            "joystick_rotation_speed": float(args.joystick_rotation_speed),
            "chassis_turn_step_deg": 90 if getattr(args, "turn_90", False) else 0,
            "joystick_single_axis_hysteresis_ratio": float(
                args.joystick_single_axis_hysteresis_ratio
            ),
            "joystick_reference_lead_xy": float(args.joystick_reference_lead_xy),
            "joystick_reference_lead_yaw": float(args.joystick_reference_lead_yaw),
            "torso_mode": torso_mode or "fixed",
            "replay_file": str(args.replay) if replay else "",
        }
        traj = _TrajLog(args.debug_dir, meta=meta)

    print("=" * 72)
    print(
        f"[wbc_joystick_robot] REAL ROBOT (JOYSTICK base). enabled={enabled} grippers=on | "
        f"{'REPLAY ' + format(args.speed, 'g') + 'x' if replay else 'LIVE'} | "
        f"fixed_speed={args.joystick_translation_speed:g}m/s,"
        f"{args.joystick_rotation_speed:g}rad/s | base_dofs={cfg.base_dofs} | "
        f"torso={torso_mode or 'fixed'} | "
        f"single_axis={cfg.enable_base_single_axis} | "
        f"gripper speed={getattr(args, 'gripper_speed', 1.0):g} "
        f"force={getattr(args, 'gripper_force', 1.0):g}"
    )
    if getattr(args, "gripper_contact_latch", False):
        print(
            "[wbc_joystick_robot] gripper contact latch ON: contact or invalid feedback "
            "stops and holds; release each trigger to <=0.1 to open and re-arm."
        )
    for side in ("left", "right"):
        cap = getattr(args, f"{side}_gripper_max_position", 1.0)
        fixed = getattr(args, f"{side}_gripper_fixed_position", None)
        if fixed is not None:
            print(
                f"[wbc_joystick_robot] {side} gripper FIXED at {min(fixed, cap):g} "
                "(trigger ignored; restart to re-arm after a contact/feedback latch)."
            )
        if cap < 1.0:
            print(
                f"[wbc_joystick_robot] {side} gripper closing limit: {cap:g} "
                f"(rPR <= {int(np.floor(cap * 255))})."
            )
    if getattr(args, "turn_90", False):
        print(
            "[wbc_joystick_robot] 90-degree turns: neutral then LEFT stick sideways; "
            "release lets the turn finish, translation/opposite yaw cancels. "
            "Measured endpoint tolerance=2deg; safety holds cancel pending turns."
        )
    if torso_stick:
        print(
            f"[wbc_joystick_robot] torso stick: LEFT stick Y moves the torso at "
            f"{args.joystick_torso_speed:g} m/s, hands riding along; steps toward the "
            f"phone mount/base/body are refused (phone_mount_collision="
            f"{cfg.phone_mount_collision}). Engage needs the torso homed."
        )
    if torso_ik:
        print(
            "[wbc_joystick_robot] torso IK: the whole-body IK moves the torso to reach the "
            "hands (arms first; lower the hands past their reach to crouch). Head fixed, "
            f"base on the sticks, phone_mount_collision={cfg.phone_mount_collision}."
        )
    if args.arkit_base != "off":
        arkit_role = (
            "closes the joystick base PD loop" if args.arkit_base == "control" else "recorded only"
        )
        print(
            f"  base pose -> iPhone/ARKit '{args.arkit_base}' "
            f"({arkit_role}"
            "; wheel odometry always recorded)."
        )
    if args.record:
        head_streams = "head_left_rgb" if args.no_head_depth else "head_left_rgb+head_depth"
        if args.head_right_rgb:
            head_streams += "+head_right_rgb"
        print(
            f"  recording -> {args.save_dir} @ {args.record_rate:g}Hz while engaged "
            f"({head_streams}+left_wrist_rgb+right_wrist_rgb, "
            f"{'streaming' if args.streaming_recorder else 'buffered'}, "
            "32-D policy action incl. chassis intent)."
        )
    print("  Robot homes to nominal, then moves. KEEP CLEAR. Ctrl-C aborts (zeros base).")
    print("=" * 72)

    driver = None
    try:
        driver = JoystickHardwareDriver(args, ik, cfg, enable, source=source, traj=traj)
        run_loop(
            source,
            ik,
            cfg,
            driver,
            args,
            replay=replay,
            clock=clock,
            realtime=True,
            enable=enable,
            traj=traj,
        )
    except KeyboardInterrupt:
        print("\n[wbc_joystick_robot] interrupted -- stopping.")
    except Exception as exc:
        _publish_abort_status(source, exc)
        _quarantine_failed_recording(driver, exc)
        raise
    finally:
        if hasattr(source, "close"):
            source.close()
        if driver is not None:
            driver.close()
        if traj is not None and not args.record and len(traj):
            print(f"\n[wbc_joystick_robot] summary: {traj.summary()}")
            traj.flush(debug_start_id)


def _bind_vr_teleop(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Bind the vr_teleop: tunables from wbik.yaml onto args (joystick-mode overrides applied).

    Mirrors wbc_vr_robot.main()'s post-parse binding, reading from ``--config`` (default
    canonical wbik.yaml). The head planar deadbands are forced to 0 for head-track (see the
    module docstring); everything else -- rates, base PD/slew/deadband/clamp -- is verbatim.
    """
    vt = VRTeleopConfig.from_yaml(args.config)
    joystick_cfg = JoystickTeleopConfig.from_yaml(args.config)
    bind_vr_teleop_args(args, vt, head_track=True)
    bind_joystick_teleop_args(args, vt, joystick_cfg)
    if args.base_translation_speed is not None:
        args.joystick_translation_speed = args.base_translation_speed
    if args.base_rotation_speed is not None:
        args.joystick_rotation_speed = args.base_rotation_speed
    if args.record and args.record_rate > args.ik_rate:
        parser.error("--record-rate must be <= the IK rate (ik_rate in wbik.yaml)")


def add_gripper_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared teleop/policy CLI: explicit overrides win; omitted values use teleop defaults."""
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=0.3,
        help="Robotiq closing/opening speed in [0,1], rSP = round(value*255) (default 0.3 -> 76).",
    )
    parser.add_argument(
        "--gripper-force",
        type=float,
        default=0.0,
        help="Robotiq grip force in [0,1], rFR = round(value*255) (default "
        "0.0 -> 0). 0 selects minimum force, not zero force.",
    )
    parser.add_argument(
        "--gripper-contact-latch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop and hold each gripper after closing contact; release "
        "its trigger to <=0.1 to open and re-arm. Polls feedback "
        "without --record too; stale/invalid feedback also holds. "
        "Enabled by default; --no-gripper-contact-latch disables it.",
    )
    for side in ("left", "right"):
        parser.add_argument(
            f"--{side}-gripper-max-position",
            type=float,
            default=0.843138,
            help=f"cap {side} gripper closing target in [0,1], rounded "
            "down to a register count (default 0.843138: register 215). "
            "For the eraser, calibrate a left limit near 0.573.",
        )
        parser.add_argument(
            f"--{side}-gripper-fixed-position",
            type=float,
            default=None,
            help="ignore this gripper's trigger and command a fixed position "
            "in [0,1] during teleop; capped by its max-position. "
            "Contact/feedback latches still stop motion; restart to re-arm.",
        )


def validate_gripper_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid gripper settings before any hardware is constructed."""
    for side in ("left", "right"):
        fixed = getattr(args, f"{side}_gripper_fixed_position")
        if fixed is not None and (not np.isfinite(fixed) or not 0.0 <= fixed <= 1.0):
            parser.error(f"--{side}-gripper-fixed-position must be finite and in [0,1]")
        if fixed is not None and (
            getattr(args, "replay", None) or getattr(args, "replay_episode", None)
        ):
            parser.error("fixed gripper positions are for live collection, not replay")
    for flag, val in (
        ("--gripper-speed", args.gripper_speed),
        ("--gripper-force", args.gripper_force),
        ("--left-gripper-max-position", args.left_gripper_max_position),
        ("--right-gripper-max-position", args.right_gripper_max_position),
    ):
        if not np.isfinite(val) or not 0.0 <= val <= 1.0:
            parser.error(f"{flag} must be finite and in [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--replay",
        metavar="FILE",
        help="re-drive either a raw joystick leader stream or an "
        "episode_<N>.hdf5 recorded by this script. Episode replay "
        "uses action/eef, action/head, action/gripper, and the exact "
        "action/chassis/intent_body labels; legacy joystick episodes "
        "without that intent field are rejected.",
    )
    mode.add_argument(
        "--source",
        choices=("live",),
        default=None,
        help="drive from the live leader (default when --replay is omitted).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=_base.DEFAULT_SPEED,
        help=f"--replay speed multiplier (default {_base.DEFAULT_SPEED:g}; <1 = slower).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="wbik.yaml path (default: canonical follower/wbik.yaml). "
        "base_dofs is read from it; lock_base_in_ik / head_mode:track / "
        "head deadbands are forced for joystick mode.",
    )
    parser.add_argument(
        "--enable",
        default="arms,torso,head,base",
        help="comma list of DOF groups to actuate (arms,torso,head,base or "
        "'all'/'none'). Default all. The base needs the chassis to move.",
    )
    torso = parser.add_mutually_exclusive_group()
    torso.add_argument(
        "--torso-stick",
        action="store_true",
        help="the leader's LEFT stick Y moves the torso (run the leader with "
        "--torso-stick too); the hands ride along and steps toward the "
        "phone mount are refused. Needs --enable torso. Off by default: "
        "the torso stays at nominal.",
    )
    torso.add_argument(
        "--torso-ik",
        action="store_true",
        help="the whole-body IK moves the torso to reach the hands, as the VR "
        "follower does with the torso unlocked (run the leader with "
        "--torso-ik too); head fixed, base on the sticks. Needs --enable "
        "torso. Off by default: the torso stays at nominal.",
    )
    parser.add_argument(
        "--namespace", default="", help="Zenoh namespace (must match the leader; default empty)."
    )
    add_gripper_arguments(parser)
    parser.add_argument(
        "--debug-dir",
        dest="debug_dir",
        default=None,
        help="write per-tick /debug HDF5 logs under this directory.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="record an episode while engaged (vr_reader EpisodeRecorder "
        "format). Needs --enable arms,torso,head,base.",
    )
    parser.add_argument(
        "--turn-90",
        action="store_true",
        help="live joystick: one measured +/-90deg turn per LEFT-stick "
        "gesture; release lets it finish, translation/opposite yaw "
        "cancels. Requires neutral before each turn. Leader unchanged.",
    )
    parser.add_argument(
        "--scenediff-sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="live --record: first right-grip hold parks arms "
        "and captures stereo; second hold aligns directly from the parked pose. "
        "Off unless passed.",
    )
    parser.add_argument(
        "--scenediff-sweep-dir",
        default=None,
        help="directory for SceneDiff sweep HDF5s (default: <save-dir>/sweep_for_scenediff).",
    )
    parser.add_argument(
        "--scenediff-park-reference", default="config/scenediff_park_reference.hdf5"
    )
    parser.add_argument(
        "--scenediff-arm-speed",
        type=float,
        default=0.50,
        help="arm parking speed in rad/s (default: 0.50); head sweep speed remains 0.10 rad/s",
    )
    parser.add_argument(
        "--streaming-recorder",
        action="store_true",
        help="write through the bounded streaming HDF5 recorder. Requires --record.",
    )
    parser.add_argument(
        "--no-head-depth",
        action="store_true",
        help="omit live head depth while recording. Requires --record.",
    )
    parser.add_argument(
        "--head-right-rgb",
        action="store_true",
        help="record the rectified head right eye for offline stereo. Requires --record.",
    )
    parser.add_argument(
        "--wrist-right-rgb",
        action="store_true",
        help="record both wrist right eyes for offline stereo. Requires --record.",
    )
    parser.add_argument(
        "--save-dir",
        default=_base.DEFAULT_SAVE_DIR,
        help=f"directory for recorded episodes (default {_base.DEFAULT_SAVE_DIR}).",
    )
    parser.add_argument(
        "--record-rate",
        type=float,
        default=_base.DEFAULT_RECORD_RATE,
        help=f"episode record cadence in Hz (default {_base.DEFAULT_RECORD_RATE:g}).",
    )
    parser.add_argument(
        "--record-stale-grace",
        type=float,
        default=1.0,
        help="seconds to retry stale cameras before aborting (default 1). "
        "No rows are written during the stall; camera age/skew "
        "limits still apply, and offline processing must handle gaps.",
    )
    parser.add_argument(
        "--record-startup-timeout",
        type=float,
        default=_base.DEFAULT_RECORD_STARTUP_TIMEOUT,
        help="seconds allowed for advancing, capture-aligned camera "
        "frames before recording frame zero.",
    )
    parser.add_argument(
        "--record-max-camera-skew-ms",
        type=float,
        default=_base.DEFAULT_RECORD_MAX_CAMERA_SKEW_MS,
        help="maximum corrected head-to-wrist capture skew per row.",
    )
    parser.add_argument(
        "--record-max-camera-age-ms",
        type=float,
        default=_base.DEFAULT_RECORD_MAX_CAMERA_AGE_MS,
        help="maximum corrected camera capture-to-selection age per row.",
    )
    hw = parser.add_argument_group("hardware")
    hw.add_argument("--home-tol", type=float, default=_base.DEFAULT_HOME_TOL)
    hw.add_argument("--home-settle", type=float, default=_base.DEFAULT_HOME_SETTLE)
    hw.add_argument(
        "--home-speed",
        type=float,
        default=None,
        help="pace nominal homing at this joint speed in rad/s (default: unpaced -- each 0.01 rad substep runs at the joint velocity limit). Lower = slower, easier to watch and abort.",
    )
    hw.add_argument("--max-joint-step", type=float, default=_base.DEFAULT_MAX_JOINT_STEP)
    hw.add_argument("--arm-cmd-lpf-tau", type=float, default=_base.DEFAULT_ARM_CMD_LPF_TAU)
    hw.add_argument("--source-timeout", type=float, default=_base.DEFAULT_SOURCE_TIMEOUT)
    hw.add_argument("--drive-state-mode", choices=("ms", "rad"), default="ms")
    hw.add_argument(
        "--arkit-base",
        dest="arkit_base",
        choices=("off", "record", "control"),
        default="off",
        help="use iPhone/ARKit base pose: record it, or also close the "
        "joystick command-pose PD loop with it. Needs --enable base.",
    )
    hw.add_argument("--base-quiet-hold-s", type=float, default=_base.DEFAULT_BASE_QUIET_HOLD_S)
    hw.add_argument(
        "--base-translation-speed",
        type=float,
        default=None,
        help="fixed joystick base translation speed (m/s; default "
        "joystick_teleop.translation_speed in wbik.yaml). Capped by base_max_speed.",
    )
    hw.add_argument(
        "--base-rotation-speed",
        type=float,
        default=None,
        help="fixed joystick base yaw speed (rad/s; default "
        "joystick_teleop.rotation_speed in wbik.yaml). Capped by 2x base_max_speed.",
    )
    args = parser.parse_args()
    for flag, val in (
        ("--base-translation-speed", args.base_translation_speed),
        ("--base-rotation-speed", args.base_rotation_speed),
    ):
        if val is not None and (not np.isfinite(val) or val <= 0.0):
            parser.error(f"{flag} must be finite and > 0")
    if args.scenediff_sweep_dir is None:
        args.scenediff_sweep_dir = str(Path(args.save_dir) / "sweep_for_scenediff")

    if not np.isfinite(args.scenediff_arm_speed) or args.scenediff_arm_speed <= 0:
        parser.error("--scenediff-arm-speed must be finite and > 0")
    args.scenediff_sweep = bool(args.scenediff_sweep and args.record and args.replay is None)
    if args.scenediff_sweep:
        args.head_right_rgb = True
        from omniteleop.wbc_scenediff_collection import load_park_pose

        load_park_pose(args.scenediff_park_reference)  # validate before connecting hardware

    _bind_vr_teleop(args, parser)
    for flag, val in (
        ("--home-tol", args.home_tol),
        ("--max-joint-step", args.max_joint_step),
        ("--arm-cmd-lpf-tau", args.arm_cmd_lpf_tau),
        ("--home-settle", args.home_settle),
        ("--record-stale-grace", args.record_stale_grace),
        ("--record-max-camera-skew-ms", args.record_max_camera_skew_ms),
    ):
        if not np.isfinite(val) or val < 0.0:
            parser.error(f"{flag} must be finite and >= 0")
    for flag, val in (
        ("--record-startup-timeout", args.record_startup_timeout),
        ("--record-max-camera-age-ms", args.record_max_camera_age_ms),
    ):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    validate_gripper_arguments(parser, args)
    if not np.isfinite(args.base_quiet_hold_s):
        parser.error("--base-quiet-hold-s must be finite")
    if args.home_speed is not None and (not np.isfinite(args.home_speed) or args.home_speed <= 0):
        parser.error("--home-speed must be finite and > 0")
    for flag, val in (
        ("--source-timeout", args.source_timeout),
        ("--record-rate", args.record_rate),
    ):
        if not np.isfinite(val) or val <= 0.0:
            parser.error(f"{flag} must be finite and > 0")
    for flag, enabled in (
        ("--streaming-recorder", args.streaming_recorder),
        ("--no-head-depth", args.no_head_depth),
        ("--head-right-rgb", args.head_right_rgb),
        ("--wrist-right-rgb", args.wrist_right_rgb),
    ):
        if enabled and not args.record:
            parser.error(f"{flag} only affects recording; pass it with --record")
    try:
        enable = parse_enable_mask(args.enable)
    except ValueError as exc:
        parser.error(str(exc))
    if args.arkit_base != "off" and not enable["base"]:
        parser.error("--arkit-base needs --enable base")
    if args.torso_ik and not enable["torso"]:
        parser.error("--torso-ik needs --enable torso (the IK moves the torso)")
    if args.torso_stick:
        if not enable["torso"]:
            parser.error("--torso-stick needs --enable torso (the hands ride the torso)")
        try:
            args.joystick_torso_speed = JoystickTorsoConfig.from_yaml(args.config).speed
        except ValueError as exc:
            parser.error(str(exc))
    if args.scenediff_sweep and not all(enable[g] for g in ("arms", "head", "base")):
        parser.error("--scenediff-sweep requires --enable arms,head,base (normally also torso)")
    if args.turn_90:
        if args.replay is not None:
            parser.error("--turn-90 is for live gestures, not replayed dense action labels")
        if not enable["base"]:
            parser.error("--turn-90 needs --enable base")
        try:
            JoystickBaseShaper.from_configs(WBCConfig.from_yaml(args.config), args)
        except ValueError as exc:
            parser.error(str(exc))

    _run_joystick_ik_mode(args, enable)


if __name__ == "__main__":
    main()
