# ruff: noqa: SLF001
"""Optional gripper guards exercised through the real scheduler with a fake wire."""

from __future__ import annotations

import functools
import importlib.util
import math
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import dexbot_utils
import dexcontrol.robot
import numpy as np
import pytest

from omniteleop.follower import robotiq
from omniteleop.follower.robotiq import build_hande_command, modbus_crc


@pytest.fixture
def mod():
    path = Path(__file__).resolve().parents[1] / "scripts/wbc_joystick_robot.py"
    spec = importlib.util.spec_from_file_location("joystick_gripper_guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _status(driver, now, side="left", **changes):
    status = dict(
        gACT=1,
        gSTA_bits=3,
        gFLT=0,
        kFLT=0,
        gGTO=1,
        gOBJ=0,
        gPR=255,
        gPO=100,
        request_send_monotonic_ns=round((now - 0.006) * 1e9),
        read_monotonic_ns=round((now - 0.001) * 1e9),
    )
    status.update(changes)
    driver._last_grip_status[side] = status


def _driver(mod, *, latch=True, left_cap=1.0, right_cap=1.0):
    driver = object.__new__(mod.JoystickHardwareDriver)
    driver._gripper_contact_latch = latch
    driver._gripper_max_position = {"left": left_cap, "right": right_cap}
    driver._gripper_hold_position = {"left": None, "right": None}
    driver._gripper_release_pending = set()
    driver._last_gripper_wire_message = {"left": None, "right": None}
    driver._last_gripper_wire_send_monotonic = {"left": -np.inf, "right": -np.inf}
    driver._last_grip_status = {}
    driver._last_gripper_cmd_left = driver._last_gripper_cmd_right = 0.0
    driver._build_hande = functools.partial(build_hande_command, speed=0.5, force=0.0)
    driver.robot = SimpleNamespace(
        left_arm=SimpleNamespace(side="left"), right_arm=SimpleNamespace(side="right")
    )
    driver._grip_monitors = {}
    driver.sent = []
    driver._send_ee_pass_through = lambda arm, message: driver.sent.append((arm.side, message))
    _status(driver, 10.0, gPR=0, gPO=3, gOBJ=3)
    _status(driver, 10.0, side="right", gPR=0, gPO=3, gOBJ=3)
    return driver


def _left_messages(driver):
    return [message for side, message in driver.sent if side == "left"]


def test_stop_command_preserves_activation_and_force_without_opening():
    message = build_hande_command(154 / 255, speed=0.5, force=0.7, go_to=False)
    assert message[7:13] == bytes([0x01, 0, 0, 154, 128, round(0.7 * 255)])
    assert message[-2:] == modbus_crc(message[:-2])
    assert build_hande_command(1.0)[7] == 0x09  # existing API remains go-to


@pytest.mark.parametrize("fixed", [0.0, 0.05, 0.6, 1.0])
def test_fixed_position_ignores_trigger_and_respects_cap(mod, fixed):
    driver = _driver(mod, left_cap=0.894118)
    driver._gripper_fixed_position = {"left": fixed}
    for trigger in [0, 1, 0.5, 0]:
        target, message = driver._prepare_gripper_command("left", trigger, now_monotonic=10)
        assert message[10] == round(min(fixed, 228 / 255) * 255)
        assert message[7] == 0x09
    # An unconfigured side still obeys its trigger.
    assert driver._prepare_gripper_command("right", 0, now_monotonic=10)[1][10] == 0


def test_fixed_position_feedback_stop_cannot_be_released_by_trigger(mod):
    driver = _driver(mod)
    driver._gripper_fixed_position = {"left": 0.6}
    driver._last_grip_status = {}
    _, message = driver._prepare_gripper_command("left", 1, now_monotonic=10)
    assert message[7] == 0x01
    _status(driver, 10)
    _, message = driver._prepare_gripper_command("left", 0, now_monotonic=10)
    assert message[7] == 0x01
    assert not driver._gripper_release_pending


def test_fixed_position_recording_uses_sent_target(mod):
    driver = _driver(mod, latch=False)
    driver._gripper_fixed_position = {"left": 0.6, "right": 0.4}
    driver._command_grippers(0, 1, now_monotonic=10)
    assert driver._last_gripper_cmd_left == 0.6
    assert driver._last_gripper_cmd_right == 0.4


def test_disabled_guards_preserve_analog_commands_and_keepalive(mod):
    driver = _driver(mod, latch=False)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    _status(driver, 10.1, gPO=154, gOBJ=2)
    driver._command_grippers(250 / 255, 0.0, now_monotonic=10.1)
    driver._command_grippers(250 / 255, 0.0, now_monotonic=11.2)
    messages = _left_messages(driver)
    assert [m[10] for m in messages] == [255, 250, 250]
    assert all(m[7] == 0x09 for m in messages)


def test_caps_independent_and_record_effective_sent_target(mod):
    driver = _driver(mod, latch=False, left_cap=0.573, right_cap=0.5)
    driver._command_grippers(1.0, 0.9, now_monotonic=10.0)
    assert driver.sent == [
        ("left", build_hande_command(146 / 255, speed=0.5, force=0)),
        ("right", build_hande_command(127 / 255, speed=0.5, force=0)),
    ]
    assert driver._last_gripper_cmd_left == pytest.approx(146 / 255)
    assert driver._last_gripper_cmd_right == pytest.approx(127 / 255)
    driver._command_grippers(0.0, 0.0, now_monotonic=10.1)
    assert all(m[10] == 0 for _, m in driver.sent[-2:])


def test_contact_stops_once_and_blocks_trigger_jitter_and_keepalives(mod):
    driver = _driver(mod)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    _status(driver, 10.1, gPO=154, gOBJ=2)
    driver._command_grippers(250 / 255, 0.0, now_monotonic=10.1)
    assert _left_messages(driver)[-1] == build_hande_command(
        154 / 255, speed=0.5, force=0.0, go_to=False
    )
    assert driver._last_gripper_cmd_left == pytest.approx(154 / 255)
    for now, trigger in ((10.2, 253 / 255), (11.2, 1.0), (12.3, 0.7)):
        # gOBJ loses its meaning after stop, and even missing feedback must not re-arm.
        driver._last_grip_status["left"] = None
        driver._command_grippers(trigger, 0.0, now_monotonic=now)
    assert len(_left_messages(driver)) == 2
    driver._command_grippers(0.1, 0.0, now_monotonic=12.4)
    assert _left_messages(driver)[-1][7:13] == bytes([9, 0, 0, 0, 128, 0])
    assert driver._gripper_hold_position["left"] is None
    _status(driver, 12.5, gPR=0, gPO=130, gOBJ=0)
    driver._command_grippers(1.0, 0.0, now_monotonic=12.5)
    assert _left_messages(driver)[-1][10] == 255


def test_hold_and_release_survive_bus_guards_and_old_contact_cannot_relatch(mod):
    driver = _driver(mod)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    monitor = SimpleNamespace(pending_request_age_s=lambda: 0.01)
    driver._grip_monitors = {"left": monitor}
    _status(driver, 10.1, gPO=154, gOBJ=2)
    driver._command_grippers(0.98, 0.0, now_monotonic=10.1)
    assert len(_left_messages(driver)) == 1
    assert driver._gripper_hold_position["left"] == pytest.approx(154 / 255)
    monitor.pending_request_age_s = lambda: None
    driver._last_grip_status["left"] = None
    driver._command_grippers(1.0, 0.0, now_monotonic=10.11)
    assert _left_messages(driver)[-1][7] == 1
    # Release is inside the command rate limit. Repress before the bus is available.
    driver._command_grippers(0.0, 0.0, now_monotonic=10.12)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.13)
    assert driver._gripper_hold_position["left"] is not None
    driver._command_grippers(1.0, 0.0, now_monotonic=10.17)
    assert _left_messages(driver)[-1][7] == 9
    assert _left_messages(driver)[-1][10] == 0
    assert driver._gripper_hold_position["left"] is None
    _status(driver, 10.18, gPO=154, gOBJ=2)  # delayed old echo, before new close
    driver._command_grippers(1.0, 0.0, now_monotonic=10.23)
    assert _left_messages(driver)[-1][10] == 255
    driver._command_grippers(0.98, 0.0, now_monotonic=10.29)
    assert _left_messages(driver)[-1][7] == 9
    assert driver._gripper_hold_position["left"] is None


def test_released_open_trigger_does_not_queue_a_redundant_open(mod):
    driver = _driver(mod)
    driver._command_grippers(0.0, 0.0, now_monotonic=10.0)
    driver._command_grippers(0.0, 0.0, now_monotonic=10.01)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.06)
    assert [m[10] for m in _left_messages(driver)] == [0, 255]


def test_keepalive_does_not_invalidate_contact_for_same_motion_command(mod):
    driver = _driver(mod)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    _status(driver, 11.05, gPO=154, gOBJ=0)
    driver._command_grippers(1.0, 0.0, now_monotonic=11.05)  # same-command keepalive
    # Contact's read began just before the keepalive; it still belongs to this close.
    _status(driver, 11.05, gPO=154, gOBJ=2)
    driver._command_grippers(0.98, 0.0, now_monotonic=11.06)
    assert driver._gripper_hold_position["left"] == pytest.approx(154 / 255)
    driver._command_grippers(0.98, 0.0, now_monotonic=11.12)
    assert _left_messages(driver)[-1][7] == 1


def test_right_contact_latch_is_independent_of_left_commands(mod):
    driver = _driver(mod)
    driver._command_grippers(1.0, 1.0, now_monotonic=10.0)
    _status(driver, 10.1, gPO=100, gOBJ=0)
    _status(driver, 10.1, side="right", gPO=120, gOBJ=2)
    driver._command_grippers(0.98, 0.98, now_monotonic=10.1)
    assert driver._last_gripper_wire_message["left"][7] == 9
    assert driver._last_gripper_wire_message["right"][7] == 1
    driver._command_grippers(0.0, 1.0, now_monotonic=10.2)
    assert driver._gripper_hold_position["right"] == pytest.approx(120 / 255)
    assert driver._last_gripper_wire_message["right"][7] == 1


def test_cap_below_release_threshold_does_not_release_held_raw_trigger(mod):
    driver = _driver(mod, left_cap=0.05)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    assert _left_messages(driver)[-1][10] == 12
    _status(driver, 10.1, gPO=6, gPR=12, gOBJ=2)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.1)
    assert _left_messages(driver)[-1][7] == 1


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {"gFLT": 5},
        {"kFLT": 1},
        {"gACT": 0},
        {"gSTA_bits": 1},
        {"read_monotonic_ns": 9_000_000_000, "request_send_monotonic_ns": 8_999_000_000},
        {"read_monotonic_ns": 11_000_000_000},
        {"request_send_monotonic_ns": 9_000_000_000},
    ],
)
def test_invalid_feedback_stops_and_holds_but_allows_explicit_open(mod, invalid):
    driver = _driver(mod)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    if invalid is None:
        driver._last_grip_status["left"] = None
    else:
        _status(driver, 10.1, **invalid)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.1)
    assert _left_messages(driver)[-1][7] == 1
    _status(driver, 10.2)  # recovering feedback does not resume squeezing on its own
    driver._command_grippers(1.0, 0.0, now_monotonic=10.2)
    assert len(_left_messages(driver)) == 2
    driver._command_grippers(0.0, 0.0, now_monotonic=10.3)
    assert _left_messages(driver)[-1][7] == 9
    assert _left_messages(driver)[-1][10] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"gOBJ": 0},
        {"gOBJ": 1},
        {"gOBJ": 3},
        {"gGTO": 0},
        {"gPR": 250},
        {"gPO": 255},
        {"request_send_monotonic_ns": 9_999_000_000, "read_monotonic_ns": 10_001_000_000},
    ],
)
def test_only_closing_contact_for_current_command_latches(mod, changes):
    driver = _driver(mod)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    _status(driver, 10.1, **(dict(gPO=154, gOBJ=2) | changes))
    driver._command_grippers(0.98, 0.0, now_monotonic=10.1)
    assert _left_messages(driver)[-1][7] == 9
    assert driver._gripper_hold_position["left"] is None


def test_cap_prevents_second_grasp_deep_targets_without_contact(mod):
    driver = _driver(mod, left_cap=0.573)
    # Recorded trigger samples around the second grasp; firmware has not detected contact.
    triggers = [0.588, 0.575, 0.602, 0.621, 0.628, 0.717, 0.793, 0.865]
    for i, trigger in enumerate(triggers):
        now = 10.0 + i * 0.11
        _status(driver, now, gPR=146, gPO=146, gOBJ=0)
        driver._command_grippers(trigger, 0.0, now_monotonic=now)
    assert max(m[10] for m in _left_messages(driver)) <= 146
    assert driver._gripper_hold_position["left"] is None


def _shaper():
    return SimpleNamespace(
        translation_speed=0.18,
        rotation_speed=0.3,
        reference_lead_xy=0.05,
        reference_lead_yaw=0.1,
        turn_90=False,
        cmd_pose=np.zeros(3),
        prev_shaped=np.zeros(3),
    )


@pytest.mark.parametrize("latch", [False, True])
def test_real_constructor_feedback_without_record_and_metadata(mod, monkeypatch, latch):
    events = []
    robot = SimpleNamespace(
        left_arm=object(),
        right_arm=object(),
        set_subscription_policy=lambda policy: None,
        get_component_status=lambda **kwargs: None,
    )
    monkeypatch.setattr(
        dexbot_utils, "RobotInfo", lambda: SimpleNamespace(has_torso=False, has_chassis=False)
    )
    monkeypatch.setattr(dexcontrol.robot, "Robot", lambda: robot)
    monkeypatch.setattr(robotiq, "send_activate", lambda arm: events.append("activate"))
    monkeypatch.setattr(mod._base.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod._base, "home_to_nominal", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.JoystickBaseShaper, "from_configs", lambda *args: _shaper())
    monkeypatch.setattr(mod.HardwareDriver, "_recording_control_metadata", lambda self: {})
    metadata = []
    monkeypatch.setattr(
        mod.HardwareDriver,
        "_init_recording",
        lambda self: metadata.append(self._recording_control_metadata()),
    )
    monkeypatch.setattr(
        mod.HardwareDriver, "_init_gripper_monitors", lambda self: events.append("monitor")
    )
    args = SimpleNamespace(
        record=False,
        gripper_contact_latch=latch,
        gripper_force=0.0,
        gripper_speed=0.5,
        left_gripper_max_position=0.573,
    )
    ik = SimpleNamespace(
        nominal_q=lambda: np.zeros(1),
        _idx_q=defaultdict(lambda: 0),
        config=SimpleNamespace(enable_collision_avoidance=True),
    )
    driver = mod.JoystickHardwareDriver(args, ik, None, {"base": False}, source=None)
    # Collision geometry now needs achieved position even when contact latching is off.
    assert events.count("monitor") == 1
    assert metadata[0]["gripper_force"] == 0.0
    assert metadata[0]["gripper_speed"] == 0.5
    assert metadata[0]["gripper_contact_latch"] == latch
    assert metadata[0]["left_gripper_max_position"] == 0.573
    assert metadata[0]["right_gripper_max_position"] == 1.0
    assert driver._build_hande(1.0)[11:13] == bytes([128, 0])
    driver._record_period = 0.5
    assert driver._gripper_status_poll_period_s() == 0.1


def test_recorded_stop_bit_matches_sent_command_even_when_release_is_pending(mod):
    driver = _driver(mod)
    driver._shaper = _shaper()
    driver._dbg = {
        key: np.zeros(3)
        for key in ("rx_chassis", "projected_chassis", "base_pd_raw", "base_pd_err")
    }
    driver._command_grippers(1.0, 0.0, now_monotonic=10.0)
    _status(driver, 10.1, gPO=154, gOBJ=2)
    driver._command_grippers(1.0, 0.0, now_monotonic=10.1)
    driver._command_grippers(0.0, 0.0, now_monotonic=10.11)
    row = driver._recording_extra_action()["gripper_control"]["left"]
    assert row == {"go_to": 0, "hold_latched": True, "release_pending": True}
    driver._command_grippers(1.0, 0.0, now_monotonic=10.17)
    row = driver._recording_extra_action()["gripper_control"]["left"]
    assert row == {"go_to": 1, "hold_latched": False, "release_pending": False}


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("value", ["-0.01", "1.01", "nan", "inf"])
def test_invalid_caps_rejected_before_hardware_startup(mod, monkeypatch, side, value):
    monkeypatch.setattr(
        "sys.argv", ["wbc_joystick_robot.py", f"--{side}-gripper-max-position", value]
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_default_closing_cap_is_register_215(mod, monkeypatch):
    seen = []
    monkeypatch.setattr(mod, "_run_joystick_ik_mode", lambda args, enable: seen.append(args))
    monkeypatch.setattr("sys.argv", ["wbc_joystick_robot.py"])
    mod.main()
    for side in ("left", "right"):
        cap = getattr(seen[0], f"{side}_gripper_max_position")
        assert math.floor(cap * 255) == 215


def test_fixed_position_is_commanded_during_hold_without_touching_trigger_sides(mod, monkeypatch):
    driver = _driver(mod, latch=False)
    driver._gripper_fixed_position = {"left": None, "right": 0.513726}
    driver._command_grippers = functools.partial(
        mod.HardwareDriver._command_grippers, driver, now_monotonic=10.0
    )
    # The parent's hold sends no gripper writes itself.
    monkeypatch.setattr(mod.HardwareDriver, "actuate", lambda *a, **k: None)
    driver.actuate(None, 1.0, 0.0, {}, True, 0.01)
    assert driver.sent == [("right", build_hande_command(131 / 255, speed=0.5, force=0))]
    assert not driver._fixed_grippers_only
    driver.sent.clear()
    driver.actuate(None, 1.0, 0.0, {}, False, 0.01)  # not a hold: the parent owns writes
    assert driver.sent == []


def test_fixed_position_waits_for_fresh_status_during_startup_hold(mod):
    driver = _driver(mod)
    driver._gripper_fixed_position = {"left": None, "right": 0.513726}
    driver._last_grip_status = {}  # only a stale warm-up read so far
    driver._fixed_grippers_only = True
    driver._command_grippers(0.0, 0.0, now_monotonic=10.0)
    assert driver.sent == []
    assert driver._gripper_hold_position["right"] is None
    _status(driver, 10.1, side="right", gPR=0, gPO=3, gOBJ=3)
    driver._command_grippers(0.0, 0.0, now_monotonic=10.1)
    assert driver.sent == [("right", build_hande_command(131 / 255, speed=0.5, force=0))]
    assert driver._gripper_hold_position["right"] is None
