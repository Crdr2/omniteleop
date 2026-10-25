"""Exercise both actual follower entry points without constructing robot objects."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name + "_native_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["wbc_vr_robot", "wbc_joystick_robot"])
def driver(request, monkeypatch):
    module = load_script(request.param)
    base = module if request.param == "wbc_vr_robot" else module._base
    monkeypatch.setattr(base, "_wait_for_recorder_save", lambda *a, **k: None)
    return (
        module.HardwareDriver if request.param == "wbc_vr_robot" else module.JoystickHardwareDriver
    )


def fixture(fail=False):
    events = []

    class Episode:
        episode_id = 7
        recording = False
        last_save_error = None

        def start(self):
            self.recording = True
            events.append("episode_start")

        def set_static(self, data):
            self.static = data

        def num_frames(self):
            return 2

        def stop(self):
            self.recording = False
            self.episode_id += 1
            events.append("episode_stop")
            return "episode_7.hdf5"

        def abort(self, reason):
            self.recording = False
            events.append(("episode_abort", reason))
            return "episode_7.hdf5.partial"

    class Native:
        def start(self, episode_id, metadata):
            events.append(("native_start", episode_id, metadata))
            if fail:
                raise RuntimeError("missing stream")
            return "native7.jsonl"

        def stop(self, **kwargs):
            events.append(("native_stop", kwargs))

    obj = SimpleNamespace(
        _episode=Episode(),
        _native_state_recorder=Native(),
        _record_static={"meta": {"ntp": {}, "camera_ntp": {"head": {}}}},
        configured_base_pose_source=lambda: "arkit",
        base_pose_source=lambda: "arkit",
        _refresh_camera_clock_offsets=lambda: None,
        _query_ntp_calibration=lambda: {"offset_ns": np.int64(5)},
        enable={"head": False},
        _frame_age_log=[],
        _grip_status_event_count={},
    )
    return obj, events


def test_native_starts_before_episode_and_stops_before_id_changes(driver):
    obj, events = fixture()
    assert driver.start_recording_episode(obj)
    assert events[0][0:2] == ("native_start", 7)
    assert events[0][2]["ntp"]["offset_ns"] == 5
    assert events[1] == "episode_start"
    assert obj._episode.static["meta"]["native_state"]["file"].item() == b"native7.jsonl"
    driver.stop_recording_episode(obj)
    assert events[-2] == ("native_stop", {"aborted": False}) and events[-1] == "episode_stop"


def test_missing_native_stream_prevents_episode_start(driver):
    obj, events = fixture(fail=True)
    with pytest.raises(RuntimeError, match="missing stream"):
        driver.start_recording_episode(obj)
    assert not obj._episode.recording and "episode_start" not in events


def test_aborting_follower_also_quarantines_native_recording(driver):
    obj, events = fixture()
    driver.start_recording_episode(obj)
    driver.abort_recording_episode(obj, "test stream failure")
    assert events[-2:] == [
        ("native_stop", {"aborted": True}),
        ("episode_abort", "test stream failure"),
    ]
    assert not obj._episode.recording


def test_joystick_leader_keeps_timestamped_intent_contract(monkeypatch):
    module = load_script("wbc_joystick_leader")
    sent = []
    leader = SimpleNamespace(stage="teleop", pub=SimpleNamespace(publish=sent.append))
    stamp = 1_800_000_000_123_456_789
    monkeypatch.setattr(module._base.time, "time_ns", lambda: stamp)
    module.JoystickVRLeader._publish(
        leader, np.eye(4), np.eye(4), np.eye(4), 0.2, 0.8, (0.18, 0.0, 0.0)
    )
    assert len(sent) == 1
    packet = sent[0]
    assert packet["timestamp_ns"] == stamp and packet["calib_stage"] == "teleop"
    assert not packet["estop"]
    assert packet["chassis_vx"] == 0.18 and packet["chassis_vy"] == packet["chassis_wz"] == 0
    assert packet["left_gripper"] == 0.2 and packet["right_gripper"] == 0.8
    assert packet["left_ee_pose"] == np.eye(4).reshape(-1).tolist()
