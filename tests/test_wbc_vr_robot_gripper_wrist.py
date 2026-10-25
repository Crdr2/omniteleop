# ruff: noqa: SLF001
"""Unit tests for the gripper + wrist-camera recording wiring in wbc_vr_robot.py.

These exercise the pure data-plumbing added when the grippers and wrist ZED-M were
enabled -- no hardware: a HardwareDriver is built via ``object.__new__`` and fed fakes
(mirrors tests/test_wbc_vr_robot_closed_loop_q.py). They verify that:

  * ``_grab_wrist_image`` preserves the legacy left-eye policy input, while
    ``_grab_wrist_stereo`` validates and returns both rectified eyes with timestamps;
  * ``_poll_gripper_status_step`` folds the latest FC03 ``actual`` into the obs cache;
  * ``record_tick`` assembles a vr_reader-format frame with ``action/gripper``,
    ``obs/gripper``, and ``obs/images/left_wrist_rgb``; on a stale/missing camera frame
    it retries at the IK rate within ``--record-stale-grace`` (recording nothing, so a
    take never contains duplicates) and aborts once the grace is exceeded (the
    frame-freshness guard);
  * ``record_tick`` writes ``action/joint/*`` from the POST-CLAMP ``_dbg['sent_joints']``
    (not the raw WBC result) and stores the world-frame ``ik.solve()`` targets VERBATIM
    as ``action/eef/{left,right}`` / ``action/head`` plus ``action/base/pose``
    (PLAN.md Conventions), raising on a missing odom, missing sent joints, or a
    malformed target.
"""

from __future__ import annotations

import importlib.util
import threading
import types
from collections import deque
from pathlib import Path

import numpy as np
import pytest


def _load_wbc_vr_robot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_robot.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_robot_gripper_wrist_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_driver_updates_collision_geometry_from_feedback_before_solve(monkeypatch):
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver._grip_monitors = {"left": object()}
    driver._last_grip_status = {"left": None, "right": None}

    def drain(*, allow_request):
        assert not allow_request
        driver._last_grip_status["left"] = dict(
            gPO=204,
            gPR=228,
            gACT=1,
            gSTA_bits=3,
            gFLT=0,
            kFLT=0,
            request_send_monotonic_ns=9_990_000_000,
            read_monotonic_ns=9_995_000_000,
        )

    driver._poll_gripper_status_step = drain
    monkeypatch.setattr(mod.time, "monotonic", lambda: 10)
    intervals = {}
    ik = types.SimpleNamespace(update_gripper_collision_geometry=lambda **kw: intervals.update(kw))
    driver.update_gripper_collision_geometry(ik, 0.01)
    assert intervals["right"] == (0, 1)
    assert np.mean(intervals["left"]) == pytest.approx(204 / 255)


def _stamp(data, ts_ns: int, receive_ns: int | None = None) -> dict:
    """Wrap a raw frame like ``get_obs(include_timestamp=True)`` does over Zenoh."""
    return {
        "data": data,
        "timestamp_ns": ts_ns,
        "receive_time_ns": ts_ns + 1 if receive_ns is None else receive_ns,
        "sequence": 1,
    }


def _wrist_driver(mod, obs: dict, arm: str = "left"):
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = types.SimpleNamespace(
        sensors=types.SimpleNamespace(
            **{
                mod._WRIST_SENSOR_IDS[arm]: types.SimpleNamespace(
                    get_obs=lambda obs_keys, include_timestamp=False: obs
                )
            }
        )
    )
    return driver


def test_grab_wrist_image_returns_contiguous_uint8_and_timestamp(monkeypatch):
    mod = _load_wbc_vr_robot()
    # A non-contiguous, non-uint8 frame the publisher might hand us.
    frame = (np.arange(4 * 6 * 3).reshape(4, 6, 3) % 256).astype(np.int16)[:, ::-1, :]
    driver = _wrist_driver(mod, {mod._WRIST_OBS_KEY: _stamp(frame, 4242)})

    out, ts = mod.HardwareDriver._grab_wrist_image(driver, "left")

    assert out.dtype == np.uint8
    assert out.shape == (4, 6, 3)
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, frame.astype(np.uint8))
    assert ts == 4242  # publisher capture timestamp passed through for the freshness guard


def test_grab_wrist_image_sample_returns_exact_receive_timestamp():
    mod = _load_wbc_vr_robot()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    driver = _wrist_driver(
        mod,
        {mod._WRIST_OBS_KEY: _stamp(frame, 4242, receive_ns=5252)},
    )

    _out, capture_ns, receive_ns = mod.HardwareDriver._grab_wrist_image_sample(driver, "left")

    assert capture_ns == 4242
    assert receive_ns == 5252


def test_grab_wrist_image_tolerates_prefixed_key(monkeypatch):
    mod = _load_wbc_vr_robot()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    prefixed = f"{mod._WRIST_SENSOR_IDS['left']}_{mod._WRIST_OBS_KEY}"
    driver = _wrist_driver(mod, {prefixed: _stamp(frame, 7)})

    out, ts = mod.HardwareDriver._grab_wrist_image(driver, "left")

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert ts == 7


def test_grab_wrist_image_returns_none_on_missing_frame():
    mod = _load_wbc_vr_robot()
    driver = _wrist_driver(mod, {})  # publisher delivered nothing this poll

    # No cached reuse: a drop must surface as None so record_tick's freshness guard can act.
    assert mod.HardwareDriver._grab_wrist_image(driver, "left") is None


def test_grab_wrist_image_rejects_malformed_shape():
    mod = _load_wbc_vr_robot()
    # Missing channels, but a valid timestamp so we exercise the shape check, not the stamp one.
    driver = _wrist_driver(mod, {"left_rgb": _stamp(np.zeros((4, 6), dtype=np.uint8), 1)})

    with pytest.raises(ValueError, match="is not"):
        mod.HardwareDriver._grab_wrist_image(driver, "left")


def test_grab_wrist_image_rejects_missing_timestamp():
    mod = _load_wbc_vr_robot()
    # A frame without a valid publisher timestamp must fail loudly: the freshness guard
    # cannot be silently disabled (e.g. an RTC stream that cannot stamp frames).
    driver = _wrist_driver(mod, {mod._WRIST_OBS_KEY: {"data": np.zeros((2, 2, 3), np.uint8)}})

    with pytest.raises(ValueError, match="timestamp"):
        mod.HardwareDriver._grab_wrist_image(driver, "left")


def test_grab_wrist_stereo_returns_both_contiguous_eyes_and_timestamps():
    mod = _load_wbc_vr_robot()
    left = np.arange(3 * 4 * 3, dtype=np.int16).reshape(3, 4, 3)[:, ::-1]
    right = np.full((3, 4, 3), 271, dtype=np.int16)[:, ::-1]
    driver = _wrist_driver(
        mod,
        {
            "left_rgb": _stamp(left, 101),
            "right_rgb": _stamp(right, 101),
        },
    )

    left_out, left_ns, right_out, right_ns = mod.HardwareDriver._grab_wrist_stereo(driver, "left")

    assert left_out.dtype == right_out.dtype == np.uint8
    assert left_out.flags["C_CONTIGUOUS"] and right_out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(left_out, left.astype(np.uint8))
    np.testing.assert_array_equal(right_out, right.astype(np.uint8))
    assert left_ns == right_ns == 101


def test_grab_wrist_stereo_sample_preserves_each_eye_receive_timestamp():
    mod = _load_wbc_vr_robot()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    driver = _wrist_driver(
        mod,
        {
            "left_rgb": _stamp(frame, 101, receive_ns=201),
            "right_rgb": _stamp(frame, 101, receive_ns=202),
        },
    )

    sample = mod.HardwareDriver._grab_wrist_stereo_sample(driver, "left")

    assert sample is not None
    assert sample[1] == 101
    assert sample[2] == 201
    assert sample[4] == 101
    assert sample[5] == 202


def test_grab_wrist_stereo_tolerates_prefixed_keys_and_requires_both_eyes():
    mod = _load_wbc_vr_robot()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    sensor_id = mod._WRIST_SENSOR_IDS["left"]
    driver = _wrist_driver(
        mod,
        {
            f"{sensor_id}_left_rgb": _stamp(frame, 11),
            f"{sensor_id}_right_rgb": _stamp(frame, 11),
        },
    )
    assert mod.HardwareDriver._grab_wrist_stereo(driver, "left") is not None

    driver = _wrist_driver(mod, {"left_rgb": _stamp(frame, 11)})
    assert mod.HardwareDriver._grab_wrist_stereo(driver, "left") is None


def test_grab_wrist_stereo_rejects_different_eye_shapes():
    mod = _load_wbc_vr_robot()
    driver = _wrist_driver(
        mod,
        {
            "left_rgb": _stamp(np.zeros((2, 3, 3), dtype=np.uint8), 1),
            "right_rgb": _stamp(np.zeros((2, 4, 3), dtype=np.uint8), 1),
        },
    )

    with pytest.raises(ValueError, match="right_rgb shape"):
        mod.HardwareDriver._grab_wrist_stereo(driver, "left")


class _FakeMonitor:
    def __init__(self, events):
        self._events = list(events)
        self.requests = 0
        self.expired = 0
        self.pending = 0

    def drain_status_events(self):
        events, self._events = self._events, []
        if events:
            self.pending = 0
        return events

    def expire_timeouts(self, _timeout_s):
        self.expired += 1

    def send_status_request(self):
        self.requests += 1
        self.pending += 1

    def stats(self):
        return {"pending_status_requests": self.pending}


def test_poll_gripper_status_step_updates_obs_cache_from_latest_event():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver._last_obs_grip_left = float("nan")
    driver._last_obs_grip_right = float("nan")
    driver._last_grip_status = {"left": None, "right": None}
    driver._grip_status_event_count = {"left": 0, "right": 0}
    driver._recorded_grip_status_event_count = {"left": 0, "right": 0}
    driver._grip_monitors = {
        "left": _FakeMonitor([{"actual": 0.1}, {"actual": 0.42}]),  # latest wins
        "right": _FakeMonitor([]),  # no reply this step -> cache unchanged
    }

    mod.HardwareDriver._poll_gripper_status_step(driver)

    assert driver._last_obs_grip_left == pytest.approx(0.42)
    assert np.isnan(driver._last_obs_grip_right)
    # Every arm is re-requested and timeouts expired regardless of replies.
    for mon in driver._grip_monitors.values():
        assert mon.requests == 1
        assert mon.expired == 1

    assert driver._grip_status_event_count == {"left": 2, "right": 0}
    assert driver._last_grip_status["left"]["actual"] == pytest.approx(0.42)


def test_bounded_gripper_statuses_mark_zero_order_hold_per_arm_atomically():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver._last_grip_status = {
        "left": {"actual": 0.2},
        "right": {"actual": 0.7},
    }
    driver._grip_status_event_count = {"left": 1, "right": 0}
    driver._recorded_grip_status_event_count = {"left": 0, "right": 0}

    statuses = mod.HardwareDriver._fresh_gripper_statuses(driver)
    assert statuses is not None
    assert statuses["left"]["sample_reused"] is False
    assert statuses["right"]["sample_reused"] is True
    assert driver._recorded_grip_status_event_count == {"left": 1, "right": 0}

    # With no callbacks on the next row, both arms explicitly report a bounded hold.
    statuses = mod.HardwareDriver._fresh_gripper_statuses(driver)
    assert statuses is not None
    assert statuses["left"]["sample_reused"] is True
    assert statuses["right"]["sample_reused"] is True

    # A new right-arm event changes only that arm's provenance flag.
    driver._grip_status_event_count["right"] = 1
    statuses = mod.HardwareDriver._fresh_gripper_statuses(driver)
    assert statuses is not None
    assert statuses["left"]["sample_reused"] is True
    assert statuses["right"]["sample_reused"] is False
    assert driver._recorded_grip_status_event_count == {"left": 1, "right": 1}


def test_episode_start_drains_preengage_gripper_replies_before_baseline():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])

    class Recorder:
        recording = False

        def start(self):
            self.recording = True

    driver._episode = Recorder()
    driver._grip_monitors = {
        "left": _FakeMonitor([_gripper_status_event(request_id=0, gpr=0, gpo=3)]),
        "right": _FakeMonitor([_gripper_status_event(request_id=0, gpr=0, gpo=3)]),
    }

    assert driver.start_recording_episode()
    assert driver._grip_status_event_count == {"left": 1, "right": 1}
    assert driver._recorded_grip_status_event_count == {"left": 1, "right": 1}
    assert driver._fresh_gripper_statuses() is None

    # Only callbacks after the episode boundary can unlock frame zero.
    for monitor in driver._grip_monitors.values():
        event = _gripper_status_event(request_id=1, gpr=128, gpo=64)
        event["request_send_wall_ns"] = driver._grip_episode_start_wall_ns
        event["read_wall_ns"] = driver._grip_episode_start_wall_ns + 1
        monitor._events.append(event)
    driver._poll_gripper_status_step()

    statuses = driver._fresh_gripper_statuses()
    assert statuses is not None
    assert {side: status["status_request_id"] for side, status in statuses.items()} == {
        "left": 1,
        "right": 1,
    }


def test_episode_start_serializes_with_inflight_camera_poll():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver)

    class Recorder:
        recording = False

        def start(self):
            self.recording = True

    driver._episode = Recorder()
    entered_grab = threading.Event()
    release_grab = threading.Event()
    original_grab = driver._grab_head_alignment_sample

    def delayed_grab():
        entered_grab.set()
        assert release_grab.wait(timeout=2.0)
        return original_grab()

    driver._grab_head_alignment_sample = delayed_grab
    driver._grab_wrist_image_sample = lambda _arm: None
    poll = threading.Thread(target=driver._poll_camera_alignment_buffers)
    poll.start()
    assert entered_grab.wait(timeout=1.0)

    started = threading.Event()

    def start_episode():
        driver.start_recording_episode()
        started.set()

    boundary = threading.Thread(target=start_episode)
    boundary.start()
    assert not started.wait(timeout=0.05)
    release_grab.set()
    poll.join(timeout=1.0)
    boundary.join(timeout=1.0)

    assert started.is_set()
    # The in-flight poll appended before the serialized reset, never afterward.
    assert list(driver._head_alignment_buffer) == []
    assert all(not history for history in driver._wrist_alignment_buffers.values())
    assert driver._last_buffered_head_ns == -1
    assert driver._last_buffered_wrist_ns == {"left": -1, "right": -1}


def test_fresh_gripper_status_reserves_point_one_period_for_commit(
    monkeypatch,
):
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    now_ns = 10_000_000_000
    monkeypatch.setattr(mod.time, "time_ns", lambda: now_ns)
    driver._record_period = 0.1
    driver._grip_episode_start_wall_ns = 1
    driver._grip_status_event_count = {"left": 1, "right": 1}
    driver._recorded_grip_status_event_count = {"left": 0, "right": 0}
    driver._last_grip_status = {
        "left": {
            "actual": 0.2,
            "request_send_wall_ns": now_ns - 246_000_000,
            "read_wall_ns": now_ns - 241_000_000,
        },
        "right": {
            "actual": 0.7,
            "request_send_wall_ns": now_ns - 105_000_000,
            "read_wall_ns": now_ns - 100_000_000,
        },
    }

    assert driver._fresh_gripper_statuses() is None
    assert driver._recorded_grip_status_event_count == {"left": 0, "right": 0}

    driver._last_grip_status["left"]["read_wall_ns"] = now_ns - 239_000_000
    driver._last_grip_status["left"]["request_send_wall_ns"] = now_ns - 244_000_000
    assert driver._fresh_gripper_statuses() is not None


def test_gripper_status_commit_age_matches_offline_exact_limit():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver._record_period = 0.1
    commit_ns = 10_000_000_000
    statuses = {
        "left": {"read_wall_ns": commit_ns - 250_000_000},
        "right": {"read_wall_ns": commit_ns - 1},
    }

    driver._validate_gripper_status_commit_age(statuses, commit_ns)
    statuses["left"]["read_wall_ns"] -= 1
    with pytest.raises(RuntimeError, match="exact commit-age contract"):
        driver._validate_gripper_status_commit_age(statuses, commit_ns)


def test_actuate_commands_grippers_through_robotiq_timestamp_helper():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)

    class Arm:
        def __init__(self, side: str) -> None:
            self.side = side

        def send_ee_pass_through_message(self, _message: bytes) -> None:
            raise AssertionError("direct pass-through send should not be used")

    calls: list[tuple[str, bytes]] = []
    driver.robot = types.SimpleNamespace(left_arm=Arm("left"), right_arm=Arm("right"))
    driver.args = types.SimpleNamespace(max_joint_step=0.05)
    driver._dbg = {}
    driver._overstep_ticks = 0
    driver._build_hande = lambda pos: f"cmd:{pos:.3f}".encode()
    driver._send_ee_pass_through = lambda arm, message: calls.append((arm.side, message))
    driver._drive_base = lambda _result, _hold, _enable, _dt: None

    result = types.SimpleNamespace(
        torso=np.zeros(0),
        left_arm=np.zeros(0),
        right_arm=np.zeros(0),
        head=np.zeros(0),
    )

    mod.HardwareDriver.actuate(
        driver,
        result,
        left_gripper=1.2,
        right_gripper=-0.4,
        enable={"torso": False, "head": False, "arms": False, "base": False},
        hold=False,
        dt=0.01,
    )

    assert calls == [("left", b"cmd:1.000"), ("right", b"cmd:0.000")]
    assert driver._last_gripper_cmd_left == pytest.approx(1.0)
    assert driver._last_gripper_cmd_right == pytest.approx(0.0)


def test_gripper_scheduler_rate_limits_writes_and_protects_status_window():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)

    class Arm:
        def __init__(self, side: str) -> None:
            self.side = side

    class Monitor:
        pending_age = 0.0

        def pending_request_age_s(self):
            return self.pending_age

    monitors = {"left": Monitor(), "right": Monitor()}
    driver.robot = types.SimpleNamespace(left_arm=Arm("left"), right_arm=Arm("right"))
    driver._grip_monitors = monitors
    driver._build_hande = lambda pos: bytes([round(pos * 255)])
    calls: list[tuple[str, bytes]] = []
    driver._send_ee_pass_through = lambda arm, message: calls.append((arm.side, message))
    driver._last_gripper_cmd_left = 0.0
    driver._last_gripper_cmd_right = 0.0

    # FC03 has just gone out: no FC16 may enter its measured 75 ms transaction window.
    mod.HardwareDriver._command_grippers(driver, 1.0, 0.5, now_monotonic=10.0)
    assert calls == []
    assert driver._last_gripper_cmd_left == 0.0

    for monitor in monitors.values():
        monitor.pending_age = 0.080
    mod.HardwareDriver._command_grippers(driver, 1.0, 0.5, now_monotonic=10.080)
    assert calls == [("left", b"\xff"), ("right", bytes([128]))]
    assert driver._last_gripper_cmd_left == 1.0

    # A changed target waits until the 20 Hz limiter is due.
    for monitor in monitors.values():
        monitor.pending_age = None
    mod.HardwareDriver._command_grippers(driver, 0.0, 0.25, now_monotonic=10.090)
    assert len(calls) == 2
    mod.HardwareDriver._command_grippers(driver, 0.0, 0.25, now_monotonic=10.131)
    assert calls[-2:] == [("left", b"\x00"), ("right", bytes([64]))]

    # An unchanged target produces no 100 Hz flood, only the one-second keepalive.
    mod.HardwareDriver._command_grippers(driver, 0.0, 0.25, now_monotonic=10.50)
    assert len(calls) == 4
    mod.HardwareDriver._command_grippers(driver, 0.0, 0.25, now_monotonic=11.131)
    assert len(calls) == 6


def test_gripper_scheduler_rejects_monitor_without_pending_age_contract():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    arm = types.SimpleNamespace()
    driver.robot = types.SimpleNamespace(left_arm=arm, right_arm=arm)
    driver._grip_monitors = {
        "left": types.SimpleNamespace(stats=lambda: {"pending_status_requests": 1}),
        "right": types.SimpleNamespace(stats=lambda: {"pending_status_requests": 1}),
    }

    with pytest.raises(RuntimeError, match="pending_request_age_s"):
        driver._command_grippers(0.0, 0.0, now_monotonic=1.0)


def test_gripper_status_scheduler_prioritizes_one_read_per_record_period():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver._record_period = 0.1
    driver._last_obs_grip_left = float("nan")
    driver._last_obs_grip_right = float("nan")
    driver._last_grip_status = {"left": None, "right": None}
    driver._grip_status_event_count = {"left": 0, "right": 0}
    driver._next_grip_status_request_monotonic = {"left": 0.0, "right": 0.0}

    class Monitor:
        def __init__(self):
            self.pending = 0
            self.requests = 0
            self.timeouts = []

        def drain_status_events(self):
            return []

        def expire_timeouts(self, timeout_s):
            self.timeouts.append(timeout_s)

        def stats(self):
            return {"pending_status_requests": self.pending}

        def send_status_request(self):
            self.pending = 1
            self.requests += 1

    monitors = {"left": Monitor(), "right": Monitor()}
    driver._grip_monitors = monitors

    # A due service sends exactly one read per arm. Repeated 100 Hz service calls do
    # not flood requests, even if a fake callback clears pending early.
    driver._poll_gripper_status_step(now_monotonic=10.0)
    assert [m.requests for m in monitors.values()] == [1, 1]
    for monitor in monitors.values():
        monitor.pending = 0
    driver._poll_gripper_status_step(now_monotonic=10.01)
    assert [m.requests for m in monitors.values()] == [1, 1]
    driver._poll_gripper_status_step(now_monotonic=10.10)
    assert [m.requests for m in monitors.values()] == [2, 2]
    assert all(
        all(timeout == pytest.approx(0.075) for timeout in monitor.timeouts)
        for monitor in monitors.values()
    )


def test_gripper_scheduler_reserves_quiet_window_before_next_status_read():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)

    class Arm:
        def __init__(self, side):
            self.side = side

    class Monitor:
        def pending_request_age_s(self):
            return None

    driver.robot = types.SimpleNamespace(left_arm=Arm("left"), right_arm=Arm("right"))
    driver._grip_monitors = {"left": Monitor(), "right": Monitor()}
    driver._next_grip_status_request_monotonic = {"left": 20.0, "right": 20.0}
    driver._build_hande = lambda pos: bytes([round(pos * 255)])
    calls = []
    driver._send_ee_pass_through = lambda arm, msg: calls.append((arm.side, msg))

    # Thirty milliseconds before FC03 leaves enough separation for one changed write.
    driver._command_grippers(1.0, 1.0, now_monotonic=19.970)
    assert len(calls) == 2
    driver._last_gripper_wire_send_monotonic = {
        "left": float("-inf"),
        "right": float("-inf"),
    }
    driver._last_gripper_wire_message = {"left": None, "right": None}

    # Twenty milliseconds before FC03 is reserved; no write may precede the read.
    driver._command_grippers(0.0, 0.0, now_monotonic=19.980)
    assert len(calls) == 2


def _record_driver(mod, captured: list):
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(wrist_right_rgb=False)
    driver.enable = {"torso": True, "head": True, "arms": True, "base": True}
    driver._joint_names = {
        "torso": ["torso_j1", "torso_j2", "torso_j3"],
        "left_arm": [f"L_arm_j{i}" for i in range(1, 8)],
        "right_arm": [f"R_arm_j{i}" for i in range(1, 8)],
        "head": ["head_j1", "head_j2", "head_j3"],
    }
    measured = {
        "torso": np.zeros(3, dtype=np.float32),
        "left_arm": np.zeros(7, dtype=np.float32),
        "right_arm": np.zeros(7, dtype=np.float32),
        "head": np.zeros(3, dtype=np.float32),
    }
    driver._comp = lambda grp: types.SimpleNamespace(get_joint_pos=lambda: measured[grp])
    # Fake cameras hand back a strictly-newer publisher timestamp on each grab (a healthy
    # 15 fps > 10 fps record stream), so the freshness guard passes across ticks. Tests that
    # exercise a stall override these stubs.
    driver._cam_ns = {"head": 1000, "left": 2000, "right": 3000}

    def _grab_head():
        driver._cam_ns["head"] += 100
        return (
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint16),
            driver._cam_ns["head"],
            driver._cam_ns["head"] - 10,
            None,
            None,
        )

    # Distinct fill per arm so a left/right mix-up in record_tick cannot pass.
    driver._wrist = np.full((4, 4, 3), 5, dtype=np.uint8)
    driver._wrist_right = np.full((4, 4, 3), 9, dtype=np.uint8)

    def _grab_wrist(arm):
        driver._cam_ns[arm] += 100
        frame = driver._wrist if arm == "left" else driver._wrist_right
        return (frame, driver._cam_ns[arm])

    driver._grab_head_images = _grab_head
    driver._grab_wrist_image = _grab_wrist
    driver._head_extrinsic = lambda obs: np.eye(4, dtype=np.float32)
    driver._grip_monitors = {}  # poll path covered separately; skip here
    # Recording requires --enable base (odometry anchors the world-frame targets), so
    # the fake carries an odom stub by default; failure tests unset it.
    driver._odom = types.SimpleNamespace(
        snapshot=lambda: {
            "twist": np.array([0.2, -0.1, 0.05]),
            "pose": np.array([1.0, 2.0, 0.3]),
        }
    )
    # Post-clamp sent joints from this tick's actuate(); action/joint/* must record
    # THESE, not the raw WBC result. Values differ from _result()'s zeros on purpose.
    driver._dbg = {
        "sent_joints": {
            "torso": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "left_arm": np.full(7, 0.4, dtype=np.float32),
            "right_arm": np.full(7, 0.5, dtype=np.float32),
            "head": np.array([0.6, 0.7, 0.8], dtype=np.float32),
        }
    }
    driver._last_action_dispatch_start_wall_ns = 1
    driver._last_action_dispatch_end_wall_ns = 2
    driver._prev_base_cmd = np.array([0.2, -0.1, 0.05])
    driver._last_gripper_cmd_left = 0.3
    driver._last_gripper_cmd_right = 0.7
    driver._last_obs_grip_left = 0.4
    driver._last_obs_grip_right = 0.6
    driver._last_grip_status = {"left": None, "right": None}
    driver._grip_status_event_count = {"left": 0, "right": 0}
    driver._recorded_grip_status_event_count = {"left": 0, "right": 0}
    driver._grip_status_stale_since = None
    driver._last_rec_head_ns = -1  # no frame recorded yet (freshness guard is armed on first)
    driver._last_rec_wrist_ns = {"left": -1, "right": -1}
    driver._record_period = 0.01
    driver._next_record_t = 0.0
    # Stale-grace state (normally set by __init__/_init_recording): retry stale ticks at
    # the IK rate for up to _stale_grace before aborting; _frame_age_log feeds the abort
    # diagnostic with recent recorded-frame arrival ages.
    driver._stale_grace = 0.25
    driver._stale_since = None
    driver._frame_age_log = deque(maxlen=64)
    driver._right_unpaired_since = None
    driver._right_unpaired_streak = 0
    driver._rec_right_unpaired = 0
    driver._right_pair_retry = True
    driver._wrist_right_unpaired_since = {arm: None for arm in mod._WRIST_SENSOR_IDS}
    driver._wrist_right_unpaired_streak = {arm: 0 for arm in mod._WRIST_SENSOR_IDS}
    driver._rec_wrist_right_unpaired = {arm: 0 for arm in mod._WRIST_SENSOR_IDS}
    driver._wrist_right_pair_retry = {arm: True for arm in mod._WRIST_SENSOR_IDS}
    driver._arkit = None
    driver._episode = types.SimpleNamespace(recording=True, record=captured.append)
    return driver


def _enable_camera_alignment(mod, driver, *, max_skew_ms: float = 40.0) -> None:
    driver._record_max_camera_skew_ns = int(round(max_skew_ms * 1e6))
    driver._camera_clock_offsets_ns = {
        "head": 0,
        "left_wrist": 0,
        "right_wrist": 0,
    }
    driver._wrist_alignment_buffers = {
        arm: deque(maxlen=mod._WRIST_ALIGNMENT_BUFFER_FRAMES) for arm in mod._WRIST_SENSOR_IDS
    }
    driver._head_alignment_buffer = deque(maxlen=mod._HEAD_ALIGNMENT_BUFFER_FRAMES)
    driver._wrist_alignment_lock = threading.Lock()
    driver._wrist_alignment_poll_lock = threading.Lock()
    driver._last_buffered_head_ns = -1
    driver._last_buffered_wrist_ns = {arm: -1 for arm in mod._WRIST_SENSOR_IDS}
    driver._camera_alignment_stale_since = None

    def grab_head_alignment_sample():
        images = driver._grab_head_images()
        if images is None:
            return None
        left, depth, left_ns, depth_ns, right, right_ns = images
        receives = getattr(driver, "_last_camera_receive_ns", {})
        return mod._HeadAlignmentSample(
            left_rgb=left,
            depth_u16=depth,
            left_capture_ns=left_ns,
            depth_capture_ns=depth_ns,
            right_rgb=right,
            right_capture_ns=right_ns,
            left_receive_ns=receives.get("head_camera left_rgb", left_ns + 1),
            depth_receive_ns=(
                None if depth is None else receives.get("head_camera depth", int(depth_ns) + 1)
            ),
            right_receive_ns=(
                None if right is None else receives.get("head_camera right_rgb", int(right_ns) + 1)
            ),
        )

    # Unit drivers do not construct a dexcontrol Robot. Keep the head source dynamic
    # so tests that replace _grab_head_images after enabling alignment are honored.
    driver._grab_head_alignment_sample = grab_head_alignment_sample


def _gripper_status_event(*, request_id: int, gpr: int, gpo: int) -> dict:
    return {
        "actual": gpo / 255.0,
        "cmd_echo": gpr / 255.0,
        "gPO": gpo,
        "gPR": gpr,
        "gCU": 7,
        "gACT": 1,
        "gGTO": 1,
        "gSTA_bits": 3,
        "gOBJ": 3,
        "gFLT": 0,
        "kFLT": 0,
        "status_request_id": request_id,
        "request_send_wall_ns": 1_000 + request_id,
        "response_timestamp_ns": 2_000 + request_id,
        "response_sequence": 10 + request_id,
        "read_wall_ns": 3_000 + request_id,
    }


def _result(mod):
    return types.SimpleNamespace(
        torso=np.zeros(3, dtype=np.float32),
        left_arm=np.zeros(7, dtype=np.float32),
        right_arm=np.zeros(7, dtype=np.float32),
        head=np.zeros(3, dtype=np.float32),
        # Solver's commanded base pose (engage-origin world) -> action/base/pose.
        base_pose=np.array([0.9, 1.9, 0.25], dtype=np.float32),
    )


def _targets():
    """Three distinct finite world-frame 4x4 solver targets (recorded verbatim)."""
    left = np.eye(4)
    left[:3, 3] = [1.0, 2.3, 0.5]
    right = np.eye(4)
    right[:3, 3] = [1.0, 2.4, 0.5]
    head = np.eye(4)
    head[:3, 3] = [0.9, 2.2, 1.1]
    c, s = np.cos(0.3), np.sin(0.3)  # non-identity rotation: verbatim beyond translation
    head[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return left, right, head


def _tick(mod, driver, now: float, result=None) -> None:
    """record_tick with the standard world targets (the post-Task-3 signature)."""
    left, right, head = _targets()
    mod.HardwareDriver.record_tick(
        driver,
        _result(mod) if result is None else result,
        hold=False,
        now=now,
        left_target=left,
        right_target=right,
        head_target=head,
        source_timestamp_ns=10,
        source_receive_wall_ns=1,
    )


def test_record_tick_writes_gripper_and_wrist():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    _tick(mod, driver, now=100.0)

    assert len(captured) == 1
    frame = captured[0]
    # Gripper action = stashed clipped trigger command; obs = FC03 achieved position.
    assert frame["action"]["gripper"]["left"] == np.float32(0.3)
    assert frame["action"]["gripper"]["right"] == np.float32(0.7)
    assert frame["obs"]["gripper"]["left"] == np.float32(0.4)
    assert frame["obs"]["gripper"]["right"] == np.float32(0.6)
    # Both wrist frames recorded under obs/images, keyed by ARM and not swapped.
    assert frame["obs"]["images"]["left_wrist_rgb"] is driver._wrist
    assert frame["obs"]["images"]["right_wrist_rgb"] is driver._wrist_right
    assert frame["obs"]["images"]["left_wrist_frame_ns"] == np.int64(driver._cam_ns["left"])
    assert frame["obs"]["images"]["right_wrist_frame_ns"] == np.int64(driver._cam_ns["right"])
    assert "head_left_rgb" in frame["obs"]["images"]
    assert frame["obs"]["images"]["head_depth_frame_ns"] == np.int64(
        frame["obs"]["images"]["head_frame_ns"] - 10
    )
    # torso still added to action/joint (WBC commands the torso).
    assert "torso" in frame["action"]["joint"]
    # Recording requires --enable base, so obs/joint always carries the measured twist.
    assert set(frame["obs"]["joint"]) == {
        "left_arm",
        "right_arm",
        "head",
        "torso",
        "chassis_vx",
        "chassis_vy",
        "chassis_wz",
    }


def test_record_tick_buffers_at_ik_rate_and_selects_head_nearest_wrist_frames():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    _enable_camera_alignment(mod, driver)
    driver._record_period = 0.1
    driver._next_record_t = 100.1

    head_ns = 1_000_000_000
    head = np.full((4, 4, 3), 31, dtype=np.uint8)
    driver._grab_head_images = lambda: (
        head,
        np.zeros((4, 4), dtype=np.uint16),
        head_ns,
        head_ns - 10,
        None,
        None,
    )
    captures = {
        "left": [900_000_000, 995_000_000, 1_070_000_000],
        "right": [920_000_000, 1_005_000_000, 1_080_000_000],
    }
    values = {"left": [11, 12, 13], "right": [21, 22, 23]}

    def grab_wrist_sample(arm):
        capture_ns = captures[arm].pop(0)
        value = values[arm].pop(0)
        image = np.full((4, 4, 3), value, dtype=np.uint8)
        return image, capture_ns, capture_ns + 1_000_000

    driver._grab_wrist_image_sample = grab_wrist_sample
    driver._last_camera_receive_ns = {
        "head_camera left_rgb": head_ns + 2_000_000,
        "head_camera depth": head_ns + 2_100_000,
    }

    # These calls are below the 10 Hz record deadline, but each must still sample one
    # new wrist cache frame. The due call samples the third and then chooses the middle.
    _tick(mod, driver, now=100.02)
    _tick(mod, driver, now=100.04)
    assert captured == []
    _tick(mod, driver, now=100.1)

    assert len(captured) == 1
    images = captured[0]["obs"]["images"]
    assert np.all(images["left_wrist_rgb"] == 12)
    assert np.all(images["right_wrist_rgb"] == 22)
    assert images["left_wrist_frame_ns"] == np.int64(995_000_000)
    assert images["right_wrist_frame_ns"] == np.int64(1_005_000_000)
    # Receive stamps belong to the SELECTED historical samples, not the latest cache.
    assert images["left_wrist_receive_ns"] == np.int64(996_000_000)
    assert images["right_wrist_receive_ns"] == np.int64(1_006_000_000)
    # History stays available to the independent live-policy consumer; the recorder's
    # own last-used cursor prevents duplicate selection and the deque remains bounded.
    assert [sample[1] for sample in driver._wrist_alignment_buffers["left"]] == [
        900_000_000,
        995_000_000,
        1_070_000_000,
    ]
    assert [sample[1] for sample in driver._wrist_alignment_buffers["right"]] == [
        920_000_000,
        1_005_000_000,
        1_080_000_000,
    ]


def test_camera_alignment_uses_camera_ntp_corrected_capture_times():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver)
    # All three captures map to local t=1.0 s despite 200 ms of raw stamp spread.
    driver._camera_clock_offsets_ns = {
        "head": 100_000_000,
        "left_wrist": -100_000_000,
        "right_wrist": 50_000_000,
    }
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    driver._wrist_alignment_buffers["left"].append(
        (frame, 900_000_000, 910_000_000, None, None, None)
    )
    driver._wrist_alignment_buffers["right"].append(
        (frame, 1_050_000_000, 1_060_000_000, None, None, None)
    )

    selected, skew_ns = driver._select_aligned_wrist_samples(1_100_000_000)

    assert selected is not None
    assert skew_ns == {"left": 0, "right": 0}


def test_camera_alignment_falls_back_from_too_new_head_to_freshest_coherent_head():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver, max_skew_ms=40.0)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def head_sample(capture_ns: int, value: int):
        return mod._HeadAlignmentSample(
            left_rgb=np.full((1, 1, 3), value, dtype=np.uint8),
            depth_u16=None,
            left_capture_ns=capture_ns,
            depth_capture_ns=None,
            right_rgb=None,
            right_capture_ns=None,
            left_receive_ns=capture_ns + 50_000_000,
            depth_receive_ns=None,
            right_receive_ns=None,
        )

    # The newest 30 Hz head frame is 53--63 ms ahead of the currently arrived
    # wrists. The preceding head frame is coherent with both, so latest-only anchoring
    # would reject a row that the bounded history can safely record.
    older = head_sample(1_000_000_000, 10)
    newer = head_sample(1_033_333_333, 20)
    driver._head_alignment_buffer.extend((older, newer))
    driver._wrist_alignment_buffers["left"].append(
        (frame, 970_000_000, 1_040_000_000, None, None, None)
    )
    driver._wrist_alignment_buffers["right"].append(
        (frame, 980_000_000, 1_050_000_000, None, None, None)
    )

    head, wrists, skew_ns = driver._select_aligned_camera_samples()

    assert head is older
    assert wrists is not None
    assert skew_ns == {"left": 30_000_000, "right": 20_000_000}


def test_record_camera_startup_requires_advancing_three_camera_histories():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver, max_skew_ms=40.0)
    driver._record_camera_warmup_complete = False
    driver._record_camera_warmup_since = None
    driver._record_startup_timeout_s = 3.0
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    def head_sample(capture_ns: int):
        return mod._HeadAlignmentSample(
            left_rgb=frame,
            depth_u16=None,
            left_capture_ns=capture_ns,
            depth_capture_ns=None,
            right_rgb=None,
            right_capture_ns=None,
            left_receive_ns=capture_ns + 10_000_000,
            depth_receive_ns=None,
            right_receive_ns=None,
        )

    base = 1_000_000_000
    driver._head_alignment_buffer.extend(
        head_sample(base + delta) for delta in (0, 33_000_000, 66_000_000)
    )
    for arm in mod._WRIST_SENSOR_IDS:
        driver._wrist_alignment_buffers[arm].extend(
            (frame, base + delta, base + delta + 10_000_000, None, None, None)
            for delta in (0, 66_000_000)
        )

    assert not driver._record_camera_startup_ready(100.0)
    assert not driver._record_camera_warmup_complete
    assert driver._last_rec_head_ns == -1

    driver._head_alignment_buffer.append(head_sample(base + 99_000_000))
    for arm in mod._WRIST_SENSOR_IDS:
        driver._wrist_alignment_buffers[arm].append(
            (frame, base + 132_000_000, base + 142_000_000, None, None, None)
        )

    assert driver._record_camera_startup_ready(100.15)
    assert driver._record_camera_warmup_complete
    # Readiness is non-consuming; record_tick still owns frame-zero cursor commit.
    assert driver._last_rec_head_ns == -1
    assert driver._last_rec_wrist_ns == {"left": -1, "right": -1}


def test_record_camera_startup_timeout_aborts_before_frame_zero():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver, max_skew_ms=40.0)
    driver._record_camera_warmup_complete = False
    driver._record_camera_warmup_since = None
    driver._record_startup_timeout_s = 0.5
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    driver._head_alignment_buffer.append(
        mod._HeadAlignmentSample(
            left_rgb=frame,
            depth_u16=None,
            left_capture_ns=1_000_000_000,
            depth_capture_ns=None,
            right_rgb=None,
            right_capture_ns=None,
            left_receive_ns=1_010_000_000,
            depth_receive_ns=None,
            right_receive_ns=None,
        )
    )
    for arm in mod._WRIST_SENSOR_IDS:
        driver._wrist_alignment_buffers[arm].append(
            (frame, 1_000_000_000, 1_010_000_000, None, None, None)
        )

    assert not driver._record_camera_startup_ready(10.0)
    with pytest.raises(RuntimeError, match="camera startup never became stable"):
        driver._record_camera_startup_ready(10.51)
    assert driver._last_rec_head_ns == -1


def test_camera_age_uses_camera_ntp_corrected_capture_times():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver)
    driver._camera_clock_offsets_ns = {
        "head": 100_000_000,
        "left_wrist": -100_000_000,
        "right_wrist": 50_000_000,
    }

    ages = driver._camera_capture_ages_ns(
        1_120_000_000,
        head_ns=1_100_000_000,
        wrist_ns={"left": 900_000_000, "right": 1_050_000_000},
    )

    assert ages == {
        "head": 120_000_000,
        "left_wrist": 120_000_000,
        "right_wrist": 120_000_000,
    }


def test_camera_age_gate_retries_then_aborts():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    driver._record_max_camera_age_ns = 150_000_000
    driver._camera_age_stale_since = None
    driver._stale_grace = 0.2

    driver._retry_or_abort_camera_age(10.0, ages_ns={"head": 180_000_000, "left_wrist": 90_000_000})
    assert driver._next_record_t == 10.0
    with pytest.raises(RuntimeError, match="camera delivery/cache age"):
        driver._retry_or_abort_camera_age(
            10.21, ages_ns={"head": 180_000_000, "left_wrist": 90_000_000}
        )


def test_camera_age_diagnostics_distinguishes_stalled_wrist_from_selected_set(monkeypatch):
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver)
    monkeypatch.setattr(mod.time, "time_ns", lambda: 2_000_000_000)
    driver._camera_clock_offsets_ns = {
        "head": 10_000_000,
        "left_wrist": 0,
        "right_wrist": -20_000_000,
    }
    driver._head_alignment_buffer.append(types.SimpleNamespace(left_capture_ns=1_960_000_000))
    driver._wrist_alignment_buffers["left"].append((None, 1_940_000_000))
    driver._wrist_alignment_buffers["right"].append((None, 1_580_000_000))
    report = driver._camera_age_diagnostics({"head": 390_000_000, "right_wrist": 400_000_000})
    assert report["newest_coherent_age_ms"] == {"head": 50, "left_wrist": 60, "right_wrist": 400}
    assert report["selected_age_ms"]["head"] == 390
    assert driver._last_rec_head_ns == -1


def test_one_second_camera_grace_tolerates_half_second_stall_but_still_aborts():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    driver._record_max_camera_age_ns = 150_000_000
    driver._camera_age_stale_since = None
    driver._stale_grace = 1.0
    driver._retry_or_abort_camera_age(10.0, ages_ns={"right_wrist": 190_000_000})
    driver._retry_or_abort_camera_age(10.5, ages_ns={"right_wrist": 690_000_000})
    assert driver._next_record_t == 10.5
    assert driver._record_max_camera_age_ns == 150_000_000
    with pytest.raises(RuntimeError, match="camera delivery/cache age"):
        driver._retry_or_abort_camera_age(11.01, ages_ns={"right_wrist": 1_200_000_000})


def test_camera_alignment_rejects_row_beyond_hard_skew_limit():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver, max_skew_ms=40.0)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    for arm, capture_ns in (("left", 930_000_000), ("right", 1_060_000_000)):
        driver._wrist_alignment_buffers[arm].append(
            (frame, capture_ns, capture_ns + 1, None, None, None)
        )

    selected, skew_ns = driver._select_aligned_wrist_samples(1_000_000_000)

    assert selected is None
    assert skew_ns == {"left": 70_000_000, "right": -60_000_000}


def test_capture_nearest_alignment_sustains_10hz_across_15hz_phase_beats():
    """A 15 Hz history always supplies one <= half-period frame at every 10 Hz row."""
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver, max_skew_ms=40.0)
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    camera_period_ns = 66_666_667
    record_period_ns = 100_000_000
    # Head delivery is much older than either wrist, matching episode_2's measured
    # ~180 ms vs ~100 ms ages. Different capture phases exercise the sampling beat.
    latency_ns = {"head": 180_000_000, "left": 90_000_000, "right": 105_000_000}
    phase_ns = {"head": 0, "left": 17_000_000, "right": 41_000_000}
    next_capture_index = {"left": 0, "right": 0}
    used: dict[str, list[int]] = {"left": [], "right": []}
    all_skews: list[int] = []

    for row in range(100):
        local_now_ns = 1_000_000_000 + row * record_period_ns
        for arm in ("left", "right"):
            latest_arrived_capture = local_now_ns - latency_ns[arm]
            while True:
                capture_ns = phase_ns[arm] + next_capture_index[arm] * camera_period_ns
                if capture_ns > latest_arrived_capture:
                    break
                driver._wrist_alignment_buffers[arm].append(
                    (frame, capture_ns, local_now_ns, None, None, None)
                )
                driver._last_buffered_wrist_ns[arm] = capture_ns
                next_capture_index[arm] += 1

        head_latest = local_now_ns - latency_ns["head"]
        head_index = (head_latest - phase_ns["head"]) // camera_period_ns
        head_ns = phase_ns["head"] + head_index * camera_period_ns
        selected, skew_ns = driver._select_aligned_wrist_samples(head_ns)
        assert selected is not None, (row, head_ns, skew_ns)
        for arm in ("left", "right"):
            capture_ns = int(selected[arm][1])
            used[arm].append(capture_ns)
            driver._last_rec_wrist_ns[arm] = capture_ns
            all_skews.append(abs(skew_ns[arm]))

    assert all(np.all(np.diff(stamps) > 0) for stamps in used.values())
    assert max(all_skews) <= camera_period_ns // 2 + 1
    assert max(all_skews) < driver._record_max_camera_skew_ns


def test_wrist_alignment_buffer_aborts_on_backward_capture_clock():
    mod = _load_wbc_vr_robot()
    driver = _record_driver(mod, [])
    _enable_camera_alignment(mod, driver)
    stamps = {"left": [200, 199], "right": [300, 301]}
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    driver._grab_wrist_image_sample = lambda arm: (
        frame,
        stamps[arm].pop(0),
        400,
    )

    driver._poll_wrist_alignment_buffers()
    with pytest.raises(RuntimeError, match="BACKWARD CAMERA CLOCK STEP"):
        driver._poll_wrist_alignment_buffers()


def test_record_tick_writes_local_camera_receive_timestamps():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._last_camera_receive_ns = {
        "head_camera left_rgb": 101,
        f"{mod._WRIST_SENSOR_IDS['left']} left_rgb": 201,
        f"{mod._WRIST_SENSOR_IDS['right']} left_rgb": 301,
        "head_camera depth": 91,
    }

    _tick(mod, driver, now=100.0)

    images = captured[0]["obs"]["images"]
    assert images["head_receive_ns"] == 101
    assert images["head_depth_receive_ns"] == 91
    assert images["left_wrist_receive_ns"] == 201
    assert images["right_wrist_receive_ns"] == 301


def test_record_tick_allows_only_bounded_reuse_and_writes_provenance(monkeypatch):
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    read_ns = 10_000_000_000
    monkeypatch.setattr(mod.time, "time_ns", lambda: read_ns + 10_000_000)
    driver._grip_episode_start_wall_ns = read_ns - 1
    left_event = _gripper_status_event(request_id=4, gpr=255, gpo=80)
    right_event = _gripper_status_event(request_id=9, gpr=0, gpo=3)
    for event in (left_event, right_event):
        event["request_send_wall_ns"] = read_ns - 5_000_000
        event["read_wall_ns"] = read_ns
    driver._grip_monitors = {
        "left": _FakeMonitor([left_event]),
        "right": _FakeMonitor([right_event]),
    }

    _tick(mod, driver, now=100.0)

    assert len(captured) == 1
    obs = captured[0]["obs"]
    assert obs["gripper"]["left"] == pytest.approx(80 / 255.0)
    assert obs["gripper_status"]["left"]["gPO"] == 80
    assert obs["gripper_status"]["left"]["gPR"] == 255
    assert obs["gripper_status"]["left"]["status_request_id"] == 4
    assert obs["gripper_status"]["right"]["response_sequence"] == 19
    assert not bool(obs["gripper_status"]["left"]["sample_reused"])

    recorded_head_ns = driver._last_rec_head_ns
    recorded_wrist_ns = dict(driver._last_rec_wrist_ns)
    # No new status events: preserve the 10 Hz row with explicit, bounded reuse.
    _tick(mod, driver, now=100.01)
    assert len(captured) == 2
    assert driver._last_rec_head_ns > recorded_head_ns
    assert driver._last_rec_wrist_ns != recorded_wrist_ns
    assert bool(captured[-1]["obs"]["gripper_status"]["left"]["sample_reused"])
    assert driver._grip_status_stale_since is None

    # Once the callback exceeds the 2.4-period selection bound, rows stop; the spare
    # 0.1 period protects the porter's exact 2.5-period commit-time contract.
    monkeypatch.setattr(mod.time, "time_ns", lambda: read_ns + 251_000_000)
    _tick(mod, driver, now=100.021)
    assert len(captured) == 2
    assert driver._grip_status_stale_since == pytest.approx(100.021)
    with pytest.raises(RuntimeError, match="no valid bounded-age gripper status"):
        _tick(mod, driver, now=100.532)


def test_record_tick_writes_stereo_rgb_without_depth():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    right = np.full((4, 4, 3), 17, dtype=np.uint8)

    def grab_head_stereo():
        driver._cam_ns["head"] += 100
        frame_ns = driver._cam_ns["head"]
        return (
            np.zeros((4, 4, 3), dtype=np.uint8),
            None,
            frame_ns,
            None,
            right,
            frame_ns,
        )

    driver._grab_head_images = grab_head_stereo

    _tick(mod, driver, now=100.0)

    images = captured[0]["obs"]["images"]
    assert images["head_right_rgb"] is right
    assert images["head_right_frame_ns"] == images["head_frame_ns"]
    assert "head_depth" not in images
    assert "head_depth_frame_ns" not in images


def test_record_tick_writes_both_eyes_for_both_wrist_cameras():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver.args.wrist_right_rgb = True
    right_eyes = {
        "left": np.full((4, 4, 3), 21, dtype=np.uint8),
        "right": np.full((4, 4, 3), 22, dtype=np.uint8),
    }

    def grab_wrist_stereo(arm):
        driver._cam_ns[arm] += 100
        frame_ns = driver._cam_ns[arm]
        left_eye = driver._wrist if arm == "left" else driver._wrist_right
        return left_eye, frame_ns, right_eyes[arm], frame_ns

    driver._grab_wrist_stereo = grab_wrist_stereo

    _tick(mod, driver, now=100.0)

    images = captured[0]["obs"]["images"]
    assert images["left_wrist_rgb"] is driver._wrist
    assert images["right_wrist_rgb"] is driver._wrist_right
    assert images["left_wrist_right_rgb"] is right_eyes["left"]
    assert images["right_wrist_right_rgb"] is right_eyes["right"]
    assert images["left_wrist_right_frame_ns"] == images["left_wrist_frame_ns"]
    assert images["right_wrist_right_frame_ns"] == images["right_wrist_frame_ns"]


def test_record_tick_retries_temporarily_unpaired_wrist_eye():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver.args.wrist_right_rgb = True

    def grab_wrist_stereo(arm):
        driver._cam_ns[arm] += 100
        frame_ns = driver._cam_ns[arm]
        frame = driver._wrist if arm == "left" else driver._wrist_right
        right_ns = frame_ns - 100 if arm == "left" else frame_ns
        return frame, frame_ns, np.zeros_like(frame), right_ns

    driver._grab_wrist_stereo = grab_wrist_stereo

    _tick(mod, driver, now=100.0)

    assert captured == []
    assert driver._wrist_right_unpaired_since["left"] == pytest.approx(100.0)
    assert driver._next_record_t == pytest.approx(100.0)

    with pytest.raises(RuntimeError, match="left wrist stereo eyes stayed"):
        _tick(mod, driver, now=100.11)
    assert captured == []


def test_record_tick_retries_temporarily_unpaired_stereo_eye():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    def grab_unpaired():
        driver._cam_ns["head"] += 100
        frame_ns = driver._cam_ns["head"]
        return (
            np.zeros((4, 4, 3), dtype=np.uint8),
            None,
            frame_ns,
            None,
            np.zeros((4, 4, 3), dtype=np.uint8),
            frame_ns - 100,
        )

    driver._grab_head_images = grab_unpaired

    _tick(mod, driver, now=100.0)

    assert captured == []
    assert driver._right_unpaired_since == pytest.approx(100.0)
    assert driver._next_record_t == pytest.approx(100.0)

    with pytest.raises(RuntimeError, match="head stereo eyes stayed"):
        _tick(mod, driver, now=100.11)
    assert captured == []


def test_record_tick_writes_measured_base_twist_and_pose():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    _tick(mod, driver, now=100.0)

    assert len(captured) == 1
    obs = captured[0]["obs"]
    # obs/joint chassis_* = the MEASURED odometry body twist (achieved-velocity
    # counterpart to action/joint/chassis_*).
    assert obs["joint"]["chassis_vx"] == np.float32(0.2)
    assert obs["joint"]["chassis_vy"] == np.float32(-0.1)
    assert obs["joint"]["chassis_wz"] == np.float32(0.05)
    # obs/base/pose = the measured odometry pose in the engage-origin world frame.
    np.testing.assert_allclose(obs["base"]["pose"], [1.0, 2.0, 0.3])
    np.testing.assert_allclose(obs["base"]["pose_odom"], [1.0, 2.0, 0.3])


def test_arkit_control_makes_canonical_policy_pose_match_control_feedback():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver.args.arkit_base = "control"
    driver._arkit = types.SimpleNamespace(
        snapshot=lambda: {
            "pose": np.array([0.8, 1.7, 0.2]),
            "world_pose": np.array([4.0, 5.0, 0.4]),
        }
    )

    _tick(mod, driver, now=100.0)

    base = captured[0]["obs"]["base"]
    np.testing.assert_allclose(base["pose"], [0.8, 1.7, 0.2])
    np.testing.assert_allclose(base["pose_arkit"], base["pose"])
    np.testing.assert_allclose(base["pose_odom"], [1.0, 2.0, 0.3])
    assert driver.base_pose_source() == "arkit"


def test_policy_base_pose_snapshot_uses_the_same_source_as_control():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(arkit_base="control")
    driver._odom = types.SimpleNamespace(snapshot=lambda: {"pose": np.array([1.0, 2.0, 0.3])})
    driver._arkit = types.SimpleNamespace(snapshot=lambda: {"pose": np.array([0.8, 1.7, 0.2])})

    snapshot = driver.policy_base_pose_snapshot()

    assert snapshot["source"] == "arkit"
    np.testing.assert_allclose(snapshot["pose"], [0.8, 1.7, 0.2])
    np.testing.assert_allclose(snapshot["odom"]["pose"], [1.0, 2.0, 0.3])


def test_recording_metadata_declares_configured_arkit_before_tracker_starts():
    """Regression for episode 8's stale wheel_odometry label."""
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(
        arkit_base="control", policy_path=None, replay_episode=None, replay=None
    )
    driver._arkit = None  # _init_recording runs before _start_arkit_tracker
    driver._record_period = 0.1
    # The recorded head contract reads both of these off the real driver.
    driver.cfg = types.SimpleNamespace(head_mode="track")
    driver.enable = {"arms": True, "torso": True, "head": True, "base": True}

    metadata = driver._recording_control_metadata()

    assert metadata["obs_base_pose_source"].item() == b"arkit"
    assert metadata["base_control_pose_source"].item() == b"arkit"
    assert metadata["head_mode"].item() == b"track"
    assert bool(metadata["head_actuated"]) is True
    assert driver.base_pose_source() == "wheel_odometry"  # live source not ready yet


def test_episode_start_rejects_configured_source_without_live_tracker():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(arkit_base="control")
    driver._arkit = None
    driver._episode = types.SimpleNamespace(recording=False, starts=0)
    driver._episode.start = lambda: setattr(driver._episode, "starts", driver._episode.starts + 1)

    with pytest.raises(RuntimeError, match="configured canonical base source"):
        driver.start_recording_episode()

    assert driver._episode.starts == 0


def test_record_tick_writes_causal_timing_brackets():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    _tick(mod, driver, now=100.0)

    frame = captured[0]
    timing = frame["timing"]
    assert timing["source_timestamp_ns"] == np.int64(10)
    assert timing["source_receive_wall_ns"] == np.int64(1)
    assert timing["action_dispatch_start_wall_ns"] == np.int64(1)
    assert timing["action_dispatch_end_wall_ns"] == np.int64(2)
    assert timing["state_read_start_wall_ns"] >= timing["action_dispatch_end_wall_ns"]
    assert timing["state_read_end_wall_ns"] >= timing["state_read_start_wall_ns"]
    assert frame["timestamp_ns"] >= timing["state_read_end_wall_ns"]


def test_record_tick_rejects_missing_actuation_timing_bracket():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._last_action_dispatch_end_wall_ns = -1

    with pytest.raises(RuntimeError, match="completed actuation timestamp bracket"):
        _tick(mod, driver, now=100.0)
    assert captured == []


def test_record_tick_skips_during_warmup_when_wrist_not_ready():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._grab_wrist_image = lambda _arm: None  # publisher not up yet, before any recorded frame

    # Before the first recorded frame (freshness state == -1), a missing stream is a warmup
    # skip, NOT an abort -- the leader can reach teleop before the wrist publisher is up.
    _tick(mod, driver, now=100.0)

    assert captured == []  # no inconsistent-key frame written, and no abort


def test_record_tick_records_consecutive_fresh_frames():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)  # stubs return strictly-increasing timestamps

    _tick(mod, driver, now=100.0)
    _tick(mod, driver, now=100.05)

    assert len(captured) == 2  # both distinct-frame ticks recorded, no false stall
    assert len(driver._frame_age_log) == 2  # arrival-age history grows per recorded frame


def test_record_tick_retries_stale_head_frame_within_grace_then_aborts():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    # First tick records and arms the freshness guard.
    _tick(mod, driver, now=100.0)
    assert len(captured) == 1

    # Head delivery stalls: get_obs re-returns the SAME frame (timestamp stops advancing).
    frozen_ns = driver._cam_ns["head"]
    driver._grab_head_images = lambda: (
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint16),
        frozen_ns,
        frozen_ns - 10,
        None,
        None,
    )

    # Within the grace: retried (schedule pulled back to now), never recorded, no abort.
    _tick(mod, driver, now=100.05)
    assert len(captured) == 1
    assert driver._stale_since == pytest.approx(100.05)
    assert driver._next_record_t == pytest.approx(100.05)  # next IK tick retries immediately

    # Still stale past the grace -> abort; the duplicate frame is never recorded.
    with pytest.raises(RuntimeError, match="stale camera frame"):
        _tick(mod, driver, now=100.05 + driver._stale_grace + 0.05)
    assert len(captured) == 1


@pytest.mark.parametrize("stream", ["head", "left", "right"])
def test_record_tick_immediately_aborts_on_backward_camera_clock(stream):
    """A clock regression is never mistaken for a recoverable cached frame."""
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    _tick(mod, driver, now=100.0)
    assert len(captured) == 1

    if stream == "head":
        regressed_ns = driver._last_rec_head_ns - 1_000_000
        driver._grab_head_images = lambda: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint16),
            regressed_ns,
            regressed_ns - 10,
            None,
            None,
        )
        expected_name = "head"
    else:
        fresh_grab = driver._grab_wrist_image
        regressed_ns = driver._last_rec_wrist_ns[stream] - 1_000_000

        def grab_wrist(arm):
            if arm == stream:
                frame = driver._wrist if arm == "left" else driver._wrist_right
                return frame, regressed_ns
            return fresh_grab(arm)

        driver._grab_wrist_image = grab_wrist
        expected_name = f"{stream}_wrist"

    with pytest.raises(RuntimeError, match="BACKWARD CAMERA CLOCK STEP") as exc_info:
        _tick(mod, driver, now=100.05)

    assert expected_name in str(exc_info.value)
    assert "moved backward 1.000 ms" in str(exc_info.value)
    assert len(captured) == 1  # the regressed frame never enters the episode
    assert driver._stale_since is None  # immediate abort, not the retry-grace path


def test_record_tick_recovers_when_fresh_frames_resume_within_grace():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    fresh_grab = driver._grab_head_images

    _tick(mod, driver, now=100.0)
    assert len(captured) == 1

    frozen_ns = driver._cam_ns["head"]
    driver._grab_head_images = lambda: (
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros((4, 4), dtype=np.uint16),
        frozen_ns,
        frozen_ns - 10,
        None,
        None,
    )
    _tick(mod, driver, now=100.05)  # stale: retry, nothing recorded
    assert len(captured) == 1

    driver._grab_head_images = fresh_grab  # delivery resumes with a newer timestamp
    _tick(mod, driver, now=100.06)

    assert len(captured) == 2  # one late frame, take continues
    assert driver._stale_since is None  # streak state cleared for the next stall


def test_record_tick_aborts_when_stream_stays_dropped_past_grace():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    _tick(mod, driver, now=100.0)
    assert len(captured) == 1

    driver._grab_wrist_image = lambda _arm: None  # publisher drops after recording is underway

    _tick(mod, driver, now=100.05)  # within grace: retried, not fatal
    assert len(captured) == 1

    with pytest.raises(RuntimeError, match="left_wrist=MISSING"):
        _tick(mod, driver, now=100.05 + driver._stale_grace + 0.05)
    assert len(captured) == 1


def test_record_tick_prefers_post_clamp_sent_joints_for_action():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)

    # _result() carries zeros; _record_driver's sent_joints carry distinct values, so
    # this proves action/joint reads the POST-CLAMP sent command, not the WBC result.
    _tick(mod, driver, now=100.0)

    frame = captured[0]
    np.testing.assert_allclose(frame["action"]["joint"]["head"], [0.6, 0.7, 0.8])
    np.testing.assert_allclose(frame["action"]["joint"]["torso"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(frame["action"]["joint"]["left_arm"], np.full(7, 0.4))
    np.testing.assert_allclose(frame["action"]["joint"]["right_arm"], np.full(7, 0.5))


def test_record_tick_requires_post_clamp_sent_joints_for_action():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._dbg = {}  # actuate never ran (or ran on hold): no post-clamp commands

    with pytest.raises(RuntimeError, match="sent_joints"):
        _tick(mod, driver, now=100.0)
    assert captured == []


def test_record_tick_requires_every_enabled_group_in_sent_joints():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    del driver._dbg["sent_joints"]["head"]  # e.g. head missing from --enable

    with pytest.raises(RuntimeError, match="sent_joints\\['head'\\]"):
        _tick(mod, driver, now=100.0)
    assert captured == []


def test_record_tick_writes_world_frame_policy_targets():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    left, right, head = _targets()

    _tick(mod, driver, now=100.0)

    frame = captured[0]
    # The world-frame solver inputs are stored VERBATIM (only a float32 cast) --
    # no base composition, ever: offline values must equal online solver inputs.
    np.testing.assert_array_equal(frame["action"]["eef"]["left"], left.astype(np.float32))
    np.testing.assert_array_equal(frame["action"]["eef"]["right"], right.astype(np.float32))
    np.testing.assert_array_equal(frame["action"]["head"], head.astype(np.float32))
    # Commanded (solver) base pose rides along for offline analysis...
    np.testing.assert_allclose(frame["action"]["base"]["pose"], [0.9, 1.9, 0.25])
    # ...while the MEASURED odom pose stays the policy-state anchor.
    np.testing.assert_allclose(frame["obs"]["base"]["pose"], [1.0, 2.0, 0.3])


def test_record_tick_requires_odometry():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    driver._odom = None  # --enable base missing: no world anchor for the targets

    with pytest.raises(RuntimeError, match="enable base"):
        _tick(mod, driver, now=100.0)
    assert captured == []


def test_record_tick_rejects_non_finite_target():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    left, right, head = _targets()
    head[1, 3] = np.nan

    with pytest.raises(RuntimeError, match="non-finite"):
        mod.HardwareDriver.record_tick(
            driver,
            _result(mod),
            hold=False,
            now=100.0,
            left_target=left,
            right_target=right,
            head_target=head,
            source_timestamp_ns=10,
            source_receive_wall_ns=1,
        )
    assert captured == []


def test_record_tick_rejects_wrong_shape_target():
    mod = _load_wbc_vr_robot()
    captured: list = []
    driver = _record_driver(mod, captured)
    left, right, _ = _targets()

    with pytest.raises(RuntimeError, match="4x4"):
        mod.HardwareDriver.record_tick(
            driver,
            _result(mod),
            hold=False,
            now=100.0,
            left_target=left,
            right_target=right,
            head_target=np.eye(3),
            source_timestamp_ns=10,
            source_receive_wall_ns=1,
        )
    assert captured == []


def test_actuate_hold_keeps_commanding_the_head_at_its_held_position():
    """A hold sends nothing to torso/arms but keeps the head at its last command.

    dexcontrol 0.5 lets the neck pitch sag once commands stop (episode 0: head_j1
    drifted -0.25 -> -0.68 rad during the pre-engage hold, then swung back through the
    first 0.4 s of the take). Before any command the hold target is the nominal head.
    """
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)

    class Comp:
        def __init__(self) -> None:
            self.sent: list = []

        def set_joint_pos(self, q, wait_time):
            self.sent.append((list(q), wait_time))

        def get_joint_pos(self):
            return [0.0, 0.0, 0.0]

    comps = {name: Comp() for name in ("torso", "head", "left_arm", "right_arm")}
    driver.robot = types.SimpleNamespace(**comps)
    driver.args = types.SimpleNamespace(max_joint_step=0.05)
    driver._dbg = {}
    driver._overstep_ticks = 0
    driver._grip_monitors = {}
    driver._command_grippers = lambda left, right: None
    driver._drive_base = lambda _result, _hold, _enable, _dt: None
    driver._comp = lambda grp: comps[grp]
    driver._to_hw = lambda grp, joints: np.asarray(joints, dtype=float)
    driver._prev_cmd = {
        "torso": np.zeros(3),
        "head": None,
        "left_arm": np.zeros(7),
        "right_arm": np.zeros(7),
    }
    driver._nominal = {"head": np.array([-0.25, 0.0, 0.0])}
    result = types.SimpleNamespace(
        torso=np.zeros(3),
        left_arm=np.zeros(7),
        right_arm=np.zeros(7),
        head=np.array([0.4, 0.1, 0.1]),
    )
    enable = {"torso": True, "head": True, "arms": True, "base": False}

    mod.HardwareDriver.actuate(driver, result, 0.0, 0.0, enable, hold=True, dt=0.01)
    mod.HardwareDriver.actuate(driver, result, 0.0, 0.0, enable, hold=True, dt=0.01)

    assert comps["torso"].sent == [] and comps["left_arm"].sent == []
    assert comps["head"].sent == [([-0.25, 0.0, 0.0], 0.0)] * 2  # nominal, zero motion
    assert driver._prev_cmd["torso"] is None  # re-seeded on resume
    np.testing.assert_allclose(driver._prev_cmd["head"], [-0.25, 0.0, 0.0])


def test_record_head_startup_gate_waits_for_a_settled_head():
    """Frame zero waits until the measured head sits still on its command.

    Episode 0 committed 13 rows while the neck pitch was still converging from a sagged
    -0.68 rad to the -0.25 nominal; those rows carried a head-camera pose unlike the rest
    of the take. The gate passes only when every joint is within the tolerance of the last
    command AND barely moved since the previous tick; it times out like the camera gate.
    """
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    readings = iter(
        [
            [-0.60, 0.0, 0.0],  # far off
            [-0.30, 0.0, 0.0],  # inside tolerance? no (50 mrad)
            [-0.262, 0.0, 0.0],  # within 20 mrad but still moving (38 mrad step)
            [-0.251, 0.0, 0.0],  # 11 mrad step: still moving
            [-0.2505, 0.0, 0.0],  # 0.5 mrad step, 0.5 mrad error -> settled
        ]
    )
    driver._comp = lambda grp: types.SimpleNamespace(get_joint_pos=lambda: next(readings))
    driver._prev_cmd = {"head": np.array([-0.25, 0.0, 0.0])}
    driver._record_head_settled = False
    driver._record_head_prev_meas = None
    driver._record_head_since = None
    driver._record_startup_timeout_s = 3.0
    driver._next_record_t = 0.0

    verdicts = [driver._record_head_startup_ready(0.01 * k) for k in range(5)]
    assert verdicts == [False, False, False, False, True]
    assert driver._record_head_settled and driver._next_record_t == 0.04
    assert driver._record_head_startup_ready(1.0)  # latched: no further readback

    # Timeout: a head that never settles aborts the take instead of recording it.
    driver2 = object.__new__(mod.HardwareDriver)
    driver2._comp = lambda grp: types.SimpleNamespace(get_joint_pos=lambda: [-0.60, 0.0, 0.0])
    driver2._prev_cmd = {"head": np.array([-0.25, 0.0, 0.0])}
    driver2._record_head_settled = False
    driver2._record_head_prev_meas = None
    driver2._record_head_since = None
    driver2._record_startup_timeout_s = 3.0
    assert not driver2._record_head_startup_ready(10.0)
    with pytest.raises(RuntimeError, match="head never settled"):
        driver2._record_head_startup_ready(13.5)

    # Head disabled: start_recording_episode() marks the gate satisfied up front.
    driver3 = object.__new__(mod.HardwareDriver)
    driver3._record_head_settled = True
    assert driver3._record_head_startup_ready(0.0)
