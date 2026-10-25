"""Hardware-free checks of publisher/subscriber/calibration compatibility."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniteleop.follower import arkit_base


def load(relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(path.stem + "_tracking_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibration(tmp_path):
    path = tmp_path / "calib_result.json"
    path.write_text(
        json.dumps(
            {
                "phoneCam_T_base": np.eye(4).tolist(),
                "R_fix_arkitcam_to_phonecam": np.eye(3).tolist(),
                "cam_in_base_mm": [0, 0, 0],
                "n_samples": 15,
                "gravity_check_deg": {"mean": 0.0},
            }
        )
    )
    return path


def test_quaternion_conversion_matches_scipy():
    publisher = load("scripts/base_tracking/base_pose_pub.py")
    for rotation in Rotation.from_rotvec([[0, 0, 0], [3.14, 0, 0], [0.2, 0.4, -1.0]]):
        matrix = publisher.quat_to_R(*rotation.as_quat())
        np.testing.assert_allclose(matrix, rotation.as_matrix(), atol=1e-12)
        np.testing.assert_allclose(
            Rotation.from_quat(publisher.R_to_quat(matrix)).as_matrix(), matrix, atol=1e-12
        )
    with pytest.raises(ValueError):
        publisher.quat_to_R(0, 0, 0, 0)


def test_calibration_composition_uses_inverse_camera_convention(tmp_path):
    publisher = load("scripts/base_tracking/base_pose_pub.py")
    path = calibration(tmp_path)
    data = json.loads(path.read_text())
    fixed = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    data["R_fix_arkitcam_to_phonecam"] = fixed.tolist()
    path.write_text(json.dumps(data))
    np.testing.assert_allclose(publisher.load_calib(path)[:3, :3], fixed.T)


def test_publisher_enables_latest_frames_and_marks_relocalization(tmp_path, monkeypatch):
    publisher = load("scripts/base_tracking/base_pose_pub.py")
    packets = []

    class Stream:
        latest_frame_only = False
        skipped_frames = 7

        @staticmethod
        def get_connected_devices():
            return [object()]

        def connect(self, device):
            assert self.latest_frame_only
            for x in (0.0, 0.01, 1.0):
                self.pose = SimpleNamespace(qx=0, qy=0, qz=0, qw=1, tx=x, ty=0, tz=0)
                self.on_new_frame()
            self.on_stream_stopped()
            return True

        def get_camera_pose(self):
            return self.pose

    class Node:
        def __init__(self, **kwargs):
            self.namespace = kwargs["namespace"]

        def create_publisher(self, topic, **kwargs):
            assert topic == arkit_base.TOPIC
            return SimpleNamespace(publish=packets.append)

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "record3d", SimpleNamespace(Record3DStream=Stream))
    monkeypatch.setattr(publisher, "Node", Node)
    monkeypatch.setenv("ZENOH_CONFIG", "test-config")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publisher",
            "--calib",
            str(calibration(tmp_path)),
            "--namespace",
            "dm/test",
            "--report",
            "0",
        ],
    )
    with pytest.raises(SystemExit, match="Record3D stream stopped"):
        publisher.main()
    assert [p["jumps"] for p in packets] == [0, 0, 1]
    assert packets[-1]["jump"]
    assert packets[-1]["drops"] == 7
    np.testing.assert_allclose(packets[-1]["xyz"], [1, 0, 0])
    np.testing.assert_allclose(packets[-1]["arkit_txyz"], [1, 0, 0])


def test_tracker_origin_and_jump_latch(monkeypatch):
    class Node:
        def __init__(self, **kwargs):
            pass

        def create_subscriber(self, *args, **kwargs):
            return SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(arkit_base, "Node", Node)
    tracker = arkit_base.ARKitBaseTracker("dm/test")
    packet = {"planar": [1, 2, np.pi / 2], "rp_rad": [0, 0], "seq": 1, "jumps": 0}
    tracker._on_msg(packet)
    tracker.reset_origin()
    tracker._on_msg(dict(packet, planar=[1, 3, np.pi / 2], seq=2))
    np.testing.assert_allclose(tracker.pose, [1, 0, 0], atol=1e-12)
    assert not tracker.jumped
    tracker._on_msg(dict(packet, seq=3, jumps=1))
    assert tracker.jumped
    tracker.reset_origin()
    assert not tracker.jumped


def test_capture_rejects_board_motion_and_accepts_reversed_detection(monkeypatch):
    capture = load("scripts/base_tracking/calibration/calib_capture.py")
    before = np.arange(176, dtype=np.float32).reshape(88, 2)
    assert not capture.head_board_moved(before, before[::-1])
    assert capture.head_board_moved(before, before + [3, 0])
    corners = np.zeros((88, 2), dtype=np.float32)
    results = iter([None, corners, corners.copy()])
    monkeypatch.setattr(capture, "head_frame", lambda: (np.zeros((1, 1, 3), np.uint8), 0))
    monkeypatch.setattr(capture, "detect_head_chessboard", lambda _: next(results))
    _, found = capture.acquire_head_chessboard(timeout=1, require_stable=True)
    np.testing.assert_array_equal(found, corners)
