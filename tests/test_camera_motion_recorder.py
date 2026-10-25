"""Exercise sidecar boundaries, packet loss, and disk backpressure."""

import copy
import importlib.util
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "src/omniteleop/common/camera_motion_recorder.py"
    spec = importlib.util.spec_from_file_location("camera_motion_recorder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Node:
    def __init__(self):
        self.callbacks = {}

    def create_subscriber(self, topic, callback, **kwargs):
        self.callbacks[topic] = callback
        return type("Subscriber", (), {"shutdown": lambda self: None})()


@pytest.fixture
def recorded_packet():
    return {
        "schema_version": 1,
        "sensor_id": "head_camera",
        "session_id": "test_session",
        "sequence": 1,
        "image_timestamp_ns": 1_000_000_000,
        "pose": {
            "valid": True,
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
        },
        "imu": {"angular_velocity": [0.0, 0.0, 0.0], "linear_acceleration": [0.0, 0.0, 9.81]},
    }


def _send(node, packet, sequence):
    packet = copy.deepcopy(packet)
    packet["sequence"] = sequence
    node.callbacks["sensors/head_camera/motion"](packet)


def test_preroll_tail_and_numpy_metadata_roundtrip(tmp_path, recorded_packet):
    mod = _module()
    node = Node()
    recorder = mod.CameraMotionRecorder(node, tmp_path, {"head_camera": {}})
    _send(node, recorded_packet, 10)
    name = recorder.start(
        3, {"head": {"offset_ns": np.int64(123), "sensor_id": np.asarray(b"head_camera")}}
    )
    _send(node, recorded_packet, 11)
    recorder.stop()
    _send(node, recorded_packet, 12)
    recorder.close()
    rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert rows[0]["camera_clock_calibration"]["head"]["offset_ns"] == 123
    assert rows[0]["camera_clock_calibration"]["head"]["sensor_id"] == "head_camera"
    assert [r["packet"]["sequence"] for r in rows if r["type"] == "sample"] == [10, 11, 12]
    assert rows[-1]["complete"] is True
    assert rows[1]["packet"]["imu"] == recorded_packet["imu"]


@pytest.mark.parametrize("failure", ["gap", "restart", "abort"])
def test_unreliable_stream_is_not_published_complete(tmp_path, recorded_packet, failure):
    recorder = _module().CameraMotionRecorder(node := Node(), tmp_path, {"head_camera": {}})
    name = recorder.start(0, {})
    _send(node, recorded_packet, 10)
    changed = copy.deepcopy(recorded_packet)
    if failure == "restart":
        changed["session_id"] = "new_session"
    _send(node, changed, 12 if failure == "gap" else 11)
    recorder.stop(aborted=failure == "abort")
    recorder.close()
    assert not (tmp_path / name).exists()
    footer = json.loads((tmp_path / (name + ".partial")).read_text().splitlines()[-1])
    assert footer["complete"] is False


def test_slow_writer_never_blocks_callback_or_stop(tmp_path, recorded_packet, monkeypatch):
    mod = _module()
    release = threading.Event()
    real_write = mod.CameraMotionRecorder._write

    def blocked_write(self, job, header):
        release.wait(timeout=5)
        real_write(self, job, header)

    monkeypatch.setattr(mod.CameraMotionRecorder, "_write", blocked_write)
    recorder = mod.CameraMotionRecorder(node := Node(), tmp_path, {"head_camera": {}})
    name = recorder.start(0, {})
    callback = node.callbacks["sensors/head_camera/motion"]
    start = time.perf_counter()
    for _ in range(4200):
        callback(recorded_packet)
    recorder.stop()
    assert time.perf_counter() - start < 0.5
    assert recorder.jobs[0]["dropped"] > 0
    release.set()
    recorder.close()
    assert not (tmp_path / name).exists()
    assert (tmp_path / (name + ".partial")).exists()


def test_reused_episode_id_has_distinct_sidecars(tmp_path, recorded_packet):
    recorder = _module().CameraMotionRecorder(node := Node(), tmp_path, {"head_camera": {}})
    first = recorder.start(0, {})
    _send(node, recorded_packet, 10)
    recorder.stop(aborted=True)
    second = recorder.start(0, {})
    _send(node, recorded_packet, 11)
    recorder.stop()
    recorder.close()
    assert first != second
    assert (tmp_path / (first + ".partial")).exists()
    assert (tmp_path / second).exists()
