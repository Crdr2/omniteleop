"""Pure tests for camera-host clock calibration helpers in wbc_vr_robot.py."""

# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_wbc_vr_robot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_robot.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_robot_camera_clock_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ntp_offset_rtt_math():
    mod = _load_wbc_vr_robot()

    offset_ns, rtt_ns = mod._ntp_offset_rtt_ns(t0=1_000, t1=2_100, t2=2_200, t3=1_300)

    assert offset_ns == 1_000
    assert rtt_ns == 300


def test_camera_clock_calibration_keeps_low_rtt_half():
    mod = _load_wbc_vr_robot()
    samples = [
        {"offset_ns": 1_000, "rtt_ns": 100},
        {"offset_ns": 1_100, "rtt_ns": 120},
        {"offset_ns": 100_000, "rtt_ns": 10_000},
    ]

    cal = mod._camera_clock_calibration_from_samples(
        samples,
        sensor_id="head_camera",
        source="zed_publisher_host",
        queried_at_ns=123_456,
    )

    assert int(cal["offset_ns"]) == 1_050
    assert int(cal["rtt_ns"]) == 110
    assert int(cal["queried_at_ns"]) == 123_456
    assert cal["sensor_id"].item() == b"head_camera"
    assert cal["source"].item() == b"zed_publisher_host"


def test_recording_ntp_requires_query_ntp():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = object()

    with pytest.raises(RuntimeError, match=r"requires Robot\.query_ntp"):
        driver._query_ntp_calibration()


def test_recording_camera_clock_requires_node():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = object()

    with pytest.raises(RuntimeError, match="requires a DexComm node"):
        driver._query_camera_clock_calibration("head", "head_camera")


def test_recording_camera_clock_requires_enough_samples():
    mod = _load_wbc_vr_robot()

    class FailingClient:
        def call(self, _request):
            raise TimeoutError("clock unavailable")

    class FakeNode:
        def create_service_client(self, **_kwargs):
            return FailingClient()

    driver = object.__new__(mod.HardwareDriver)
    driver.robot = type("Robot", (), {"_node": FakeNode()})()

    with pytest.raises(RuntimeError, match="0/15 usable samples"):
        driver._query_camera_clock_calibration("head", "head_camera")


def _head_info_node(response):
    class Client:
        def call(self, _request):
            return response

    class Node:
        def create_service_client(self, **_kwargs):
            return Client()

    return Node()


def test_head_stereo_calibration_uses_live_publisher_matrix():
    mod = _load_wbc_vr_robot()
    live_k = np.array(
        [[377.13342, 0.0, 489.7027], [0.0, 377.13342, 319.49548], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    response = {
        "streams": {"right_rgb": {"enabled": True}},
        "stereo": {
            "rectified": True,
            "baseline_m": 0.0499879,
            "left_K": live_k.tolist(),
            "right_K": live_k.tolist(),
        },
    }
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = types.SimpleNamespace(_node=_head_info_node(response))

    out = driver._query_head_stereo_calibration()

    np.testing.assert_array_equal(out["left_K"], live_k)
    np.testing.assert_array_equal(out["right_K"], live_k)
    assert out["baseline_m"] == pytest.approx(0.0499879)


def test_head_stereo_calibration_rejects_disabled_right_stream():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = types.SimpleNamespace(
        _node=_head_info_node({"streams": {"right_rgb": {"enabled": False}}})
    )

    with pytest.raises(RuntimeError, match="right_rgb disabled"):
        driver._query_head_stereo_calibration()


def test_head_stereo_calibration_names_info_service_when_block_missing():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = types.SimpleNamespace(
        _node=_head_info_node({"streams": {"right_rgb": {"enabled": True}}})
    )

    with pytest.raises(RuntimeError, match="sensors/head_camera/info.*stereo"):
        driver._query_head_stereo_calibration()


def test_init_recording_selects_streaming_and_stamps_live_intrinsic(tmp_path, monkeypatch):
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(
        record=True,
        record_stale_grace=0.2,
        record_rate=10.0,
        save_dir=str(tmp_path),
        streaming_recorder=True,
        no_head_depth=True,
        head_right_rgb=True,
        wrist_right_rgb=True,
        record_max_camera_skew_ms=40.0,
        record_max_camera_age_ms=150.0,
    )
    live_k = np.array(
        [[377.0, 0.0, 490.0], [0.0, 377.0, 320.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    stereo = {
        "baseline_m": np.float32(0.05),
        "left_K": live_k,
        "right_K": live_k.copy(),
    }
    driver._await_record_cameras = lambda: None
    driver._query_head_stereo_calibration = lambda: stereo
    driver._query_wrist_stereo_calibrations = lambda: {
        "left": stereo,
        "right": stereo,
    }
    driver._query_ntp_calibration = lambda: {"offset_ns": np.int64(0)}
    camera_ntp = {
        label: {
            "offset_ns": np.int64(offset),
            "rtt_ns": np.int64(1_000_000),
            "queried_at_ns": np.int64(123),
            "sensor_id": np.asarray(sensor_id.encode()),
            "source": np.asarray(b"zed_publisher_host"),
        }
        for label, sensor_id, offset in (
            ("head", "head_camera", 100),
            ("left_wrist", "left_wrist_zedm", 200),
            ("right_wrist", "right_wrist_zedm", 300),
        )
    }
    driver._query_camera_clock_calibrations = lambda: camera_ntp
    driver._recording_camera_metadata = lambda: {"head": {"serial_number": np.int64(1)}}
    driver._recording_provenance_metadata = lambda: {
        "schema": np.asarray(b"omniteleop_wbc_mobile_raw/v2")
    }
    driver._recording_control_metadata = dict

    import omniteleop.common.native_state_recorder as native_module

    monkeypatch.setattr(
        native_module.NativeStateRecorder, "start_clock_sampling", lambda self, queries: None
    )
    node = types.SimpleNamespace(
        create_subscriber=lambda *a, **kw: types.SimpleNamespace(shutdown=lambda: None)
    )
    driver.robot = types.SimpleNamespace(
        _node=node,
        **{
            g: types.SimpleNamespace(
                get_joint_name=lambda: ["j"], _control_pub_topic="command/" + g
            )
            for g in ("head", "torso", "left_arm", "right_arm")
        },
    )
    mod.HardwareDriver._init_recording(driver)

    assert isinstance(driver._episode, mod.StreamingEpisodeRecorder)
    np.testing.assert_array_equal(driver._episode._static["obs"]["images"]["intrinsic"], live_k)
    np.testing.assert_array_equal(driver._episode._static["meta"]["head_stereo"]["left_K"], live_k)
    for arm in ("left", "right"):
        np.testing.assert_array_equal(
            driver._episode._static["meta"]["wrist_stereo"][arm]["left_K"],
            live_k,
        )
    assert driver._camera_clock_offsets_ns == {
        "head": 100,
        "left_wrist": 200,
        "right_wrist": 300,
    }
    alignment = driver._episode._static["meta"]["camera_alignment"]
    assert alignment["mode"].item() == b"head_capture_nearest"
    assert alignment["max_abs_skew_ns"] == np.int64(40_000_000)
    assert alignment["max_camera_age_ns"] == np.int64(150_000_000)


def test_recording_camera_metadata_keeps_identity_geometry_and_config():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)

    def info(_label, sensor_id):
        serial = {
            "head_camera": 12930616,
            "left_wrist_zedm": 38085162,
            "right_wrist_zedm": 33046361,
        }[sensor_id]
        return {
            "camera_id": str(serial),
            "serial_number": serial,
            "model": "ZED",
            "firmware_version": "1523",
            "status": "running",
            "actual": {"width": 320, "height": 240, "fps": 15},
            "configured": {"resolution": "HD720", "resize_hw": [240, 320]},
            "streams": {"left_rgb": {"enabled": True}},
        }

    driver._query_camera_info_response = info

    out = driver._recording_camera_metadata()

    assert int(out["head"]["serial_number"]) == 12930616
    assert int(out["left_wrist"]["actual"]["width"]) == 320
    assert int(out["right_wrist"]["actual"]["height"]) == 240
    configured = json.loads(out["head"]["configured_json"].item().decode())
    assert configured == {"resize_hw": [240, 320], "resolution": "HD720"}
    assert out["right_wrist"]["capture"]["resolution"].item() == b"HD720"
    assert out["right_wrist"]["capture"]["width"] == 1280
    assert out["right_wrist"]["capture"]["height"] == 720
    assert out["right_wrist"]["output"]["width"] == 320


def test_recording_provenance_snapshots_dirty_source_and_config_content():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(
        namespace="test", record=True, replay=None, arkit_base="off"
    )
    driver.cfg = mod.WBCConfig()

    out = driver._recording_provenance_metadata()

    assert out["schema"].item() == b"omniteleop_wbc_mobile_raw/v5"
    robot_source = out["source_snapshot"]["wbc_vr_robot"]
    assert len(robot_source["sha256"].item().decode()) == 64
    assert b"class HardwareDriver" in robot_source["content"].item()
    assert b"vr_teleop:" in out["source_snapshot"]["wbik_yaml"]["content"].item()
    parsed = json.loads(out["parsed_args_json"].item().decode())
    assert parsed["namespace"] == "test"


def test_main_accepts_production_recording_mode_flags(monkeypatch, tmp_path):
    mod = _load_wbc_vr_robot()
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wbc_vr_robot.py",
            "--record",
            "--streaming-recorder",
            "--head-right-rgb",
            "--wrist-right-rgb",
            "--no-head-depth",
            "--save-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        mod,
        "_run_ik_mode",
        lambda args, enable: captured.update(args=args, enable=enable),
    )

    mod.main()

    assert captured["args"].record is True
    assert captured["args"].streaming_recorder is True
    assert captured["args"].head_right_rgb is True
    assert captured["args"].wrist_right_rgb is True
    assert captured["args"].no_head_depth is True


def test_grab_head_images_records_rgb_and_depth_timestamps_separately():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(no_head_depth=False, head_right_rgb=False, record=True)

    class HeadCamera:
        def __init__(self):
            self.depth_ns = 90

        def get_obs(self, *, obs_keys, include_timestamp):
            assert obs_keys == ["left_rgb", "depth"]
            assert include_timestamp is True
            return {
                "left_rgb": {
                    "data": np.zeros((4, 4, 3), dtype=np.uint8),
                    "timestamp_ns": 100,
                    "receive_time_ns": 110,
                },
                "depth": {
                    "data": np.zeros((4, 4), dtype=np.float32),
                    "timestamp_ns": self.depth_ns,
                    "receive_time_ns": 111,
                },
            }

    head_camera = HeadCamera()
    driver.robot = types.SimpleNamespace(sensors=types.SimpleNamespace(head_camera=head_camera))

    rgb, depth_u16, frame_ns, depth_ns, right_rgb, right_ns = driver._grab_head_images()
    assert rgb.shape == (4, 4, 3)
    assert depth_u16.dtype == np.uint16
    assert frame_ns == 100
    assert depth_ns == 90
    assert right_rgb is None
    assert right_ns is None
    assert driver._last_camera_receive_ns == {
        "head_camera left_rgb": 110,
        "head_camera depth": 111,
    }


def test_grab_head_images_can_select_stereo_rgb_without_depth():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(no_head_depth=True, head_right_rgb=True, record=True)

    class HeadCamera:
        def get_obs(self, *, obs_keys, include_timestamp):
            assert obs_keys == ["left_rgb", "right_rgb"]
            assert include_timestamp is True
            return {
                "left_rgb": {
                    "data": np.zeros((4, 5, 3), dtype=np.uint8),
                    "timestamp_ns": 101,
                    "receive_time_ns": 110,
                },
                "right_rgb": {
                    "data": np.ones((4, 5, 3), dtype=np.uint8),
                    "timestamp_ns": 101,
                    "receive_time_ns": 111,
                },
            }

    driver.robot = types.SimpleNamespace(sensors=types.SimpleNamespace(head_camera=HeadCamera()))

    rgb, depth, frame_ns, depth_ns, right_rgb, right_ns = driver._grab_head_images()

    assert rgb.shape == (4, 5, 3)
    assert depth is None
    assert depth_ns is None
    assert right_rgb.shape == rgb.shape
    assert right_ns == frame_ns == 101
    assert driver._last_camera_receive_ns == {
        "head_camera left_rgb": 110,
        "head_camera right_rgb": 111,
    }


def test_grab_head_rgb_does_not_fetch_or_convert_depth():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)

    class HeadCamera:
        def get_obs(self, *, obs_keys, include_timestamp):
            assert obs_keys == ["left_rgb"]
            assert include_timestamp is True
            return {
                "left_rgb": {
                    "data": np.zeros((4, 5, 3), dtype=np.uint8),
                    "timestamp_ns": 123,
                }
            }

    driver.robot = types.SimpleNamespace(sensors=types.SimpleNamespace(head_camera=HeadCamera()))

    rgb, frame_ns = driver._grab_head_rgb()

    assert rgb.shape == (4, 5, 3)
    assert rgb.dtype == np.uint8
    assert rgb.flags.c_contiguous
    assert frame_ns == 123


def test_grab_head_rgb_sample_returns_exact_receive_stamp():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.args = types.SimpleNamespace(record=True)

    class HeadCamera:
        @staticmethod
        def get_obs(*, obs_keys, include_timestamp):
            assert obs_keys == ["left_rgb"]
            assert include_timestamp is True
            return {
                "left_rgb": {
                    "data": np.zeros((3, 4, 3), dtype=np.uint8),
                    "timestamp_ns": 123,
                    "receive_time_ns": 456,
                }
            }

    driver.robot = types.SimpleNamespace(sensors=types.SimpleNamespace(head_camera=HeadCamera()))

    _rgb, capture_ns, receive_ns = driver._grab_head_rgb_sample()

    assert capture_ns == 123
    assert receive_ns == 456
