# ruff: noqa: SLF001

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import h5py
import numpy as np
import pytest


def _load_wbc_vr_robot():
    path = Path(__file__).resolve().parents[1] / "scripts" / "wbc_vr_robot.py"
    spec = importlib.util.spec_from_file_location("wbc_vr_robot_closed_loop_q_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hardware_driver_no_longer_exposes_measured_seed_hook():
    mod = _load_wbc_vr_robot()
    assert not hasattr(mod.HardwareDriver, "measured_q")


def test_engage_reset_does_not_start_recording_during_alignment():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.enable = {"torso": True, "head": True, "arms": True, "base": True}
    driver._arkit = None
    driver._odom = None
    driver._world_frame_epoch = 0
    driver._prev_cmd = {"torso": np.array([0.1]), "left_arm": np.array([0.2])}
    driver._episode = types.SimpleNamespace(recording=False, starts=0)
    driver._episode.start = lambda: (
        setattr(driver._episode, "recording", True),
        setattr(driver._episode, "starts", driver._episode.starts + 1),
    )
    driver._grip_status_event_count = {"left": 7, "right": 9}
    driver._recorded_grip_status_event_count = {"left": 1, "right": 2}
    driver._grip_status_stale_since = 8.0
    driver._last_rec_head_ns = 111
    driver._last_rec_wrist_ns = {"left": 222, "right": 333}
    driver._last_camera_receive_ns = {"head_left_rgb": 444}
    driver._stale_since = 7.0
    driver._frame_age_log = mod.deque([(1.0, 2.0, 3.0, 4.0)])
    driver._right_unpaired_since = 6.0
    driver._right_unpaired_streak = 5
    driver._rec_right_unpaired = 4
    driver._right_pair_retry = False
    driver._wrist_right_unpaired_since = {"left": 3.0, "right": 2.0}
    driver._wrist_right_unpaired_streak = {"left": 3, "right": 2}
    driver._rec_wrist_right_unpaired = {"left": 2, "right": 1}
    driver._wrist_right_pair_retry = {"left": False, "right": False}
    driver._next_record_t = 123.0

    mod.HardwareDriver.engage_reset(driver, None, None, None, None)

    assert not driver._episode.recording
    assert driver._episode.starts == 0
    assert driver._next_record_t == 123.0

    mod.HardwareDriver.start_recording_if_teleop(driver, types.SimpleNamespace(calib_stage="align"))

    assert not driver._episode.recording
    assert driver._episode.starts == 0

    mod.HardwareDriver.start_recording_if_teleop(
        driver, types.SimpleNamespace(calib_stage="teleop", estop=True)
    )

    assert not driver._episode.recording
    assert driver._episode.starts == 0

    mod.HardwareDriver.start_recording_if_teleop(
        driver, types.SimpleNamespace(calib_stage="teleop", estop=False)
    )

    assert driver._episode.recording
    assert driver._episode.starts == 1
    assert driver._next_record_t == 0.0
    assert driver._last_rec_head_ns == -1
    assert driver._last_rec_wrist_ns == {"left": -1, "right": -1}
    assert driver._last_camera_receive_ns == {}
    assert driver._stale_since is None
    assert list(driver._frame_age_log) == []
    assert driver._right_unpaired_since is None
    assert driver._right_unpaired_streak == driver._rec_right_unpaired == 0
    assert driver._right_pair_retry is True
    assert driver._wrist_right_unpaired_since == {"left": None, "right": None}
    assert driver._wrist_right_unpaired_streak == {"left": 0, "right": 0}
    assert driver._rec_wrist_right_unpaired == {"left": 0, "right": 0}
    assert driver._wrist_right_pair_retry == {"left": True, "right": True}
    assert driver._recorded_grip_status_event_count == {"left": 7, "right": 9}
    assert driver._grip_status_stale_since is None

    mod.HardwareDriver.start_recording_if_teleop(
        driver, types.SimpleNamespace(calib_stage="teleop")
    )

    assert driver._episode.starts == 1


def test_run_loop_stops_motion_immediately_on_leader_exit_request():
    mod = _load_wbc_vr_robot()
    pose = np.eye(4)
    ik = types.SimpleNamespace(frame_pose=lambda _frame: pose)
    cfg = types.SimpleNamespace(head_mode="ik")
    source = types.SimpleNamespace(
        latest=types.SimpleNamespace(exit_requested=True),
        done=False,
        start=lambda: None,
    )
    driver = types.SimpleNamespace(
        name="fake",
        stop_calls=0,
        actuate_calls=0,
        record_calls=0,
    )

    def stop_all_motion():
        driver.stop_calls += 1

    def actuate(*_args, **_kwargs):
        driver.actuate_calls += 1

    def record_tick(*_args, **_kwargs):
        driver.record_calls += 1

    driver.stop_all_motion = stop_all_motion
    driver.actuate = actuate
    driver.record_tick = record_tick
    args = types.SimpleNamespace(
        ik_rate=100.0,
        speed=1.0,
        cmd_rate=10.0,
        head_lpf_tau=0.0,
        head_planar_pos_deadband=0.0,
        head_planar_yaw_deadband=0.0,
    )
    traj = types.SimpleNamespace(append=lambda **_fields: None)

    mod.run_loop(
        source,
        ik,
        cfg,
        driver,
        args,
        replay=False,
        clock=lambda: 0.0,
        realtime=False,
        enable={"arms": True, "torso": True, "head": True, "base": True},
        traj=traj,
    )

    assert driver.stop_calls == 1
    assert driver.actuate_calls == 0
    assert driver.record_calls == 0


def test_stop_recording_episode_saves_without_shutdown(monkeypatch):
    mod = _load_wbc_vr_robot()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    driver = object.__new__(mod.HardwareDriver)
    episode = types.SimpleNamespace(recording=True, saving=False, stops=0)
    episode.num_frames = lambda: 3
    episode.stop = lambda: (
        setattr(episode, "recording", False),
        setattr(episode, "stops", episode.stops + 1),
        "/tmp/episode_0.hdf5",
    )[2]
    driver._episode = episode

    mod.HardwareDriver.stop_recording_episode(driver)

    assert episode.recording is False
    assert episode.stops == 1


def test_home_to_nominal_method_uses_existing_homing_helper(monkeypatch):
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.stop_all_motion = lambda: None
    calls = []
    monkeypatch.setattr(
        mod,
        "home_to_nominal",
        lambda drv, *, arm_home_step: calls.append((drv, arm_home_step)),
    )

    mod.HardwareDriver.home_to_nominal(driver)

    assert calls == [(driver, mod.ARM_HOME_STEP)]


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_run_loop_static_home_request_stops_recording_and_homes_once():
    mod = _load_wbc_vr_robot()
    pose = np.eye(4)
    result = types.SimpleNamespace(
        success=True,
        held=False,
        left_ee_error=0.0,
        right_ee_error=0.0,
        stability_margin=0.1,
        safety_status="ok",
    )
    events = []
    ik = types.SimpleNamespace(
        frame_pose=lambda _frame: pose,
        solve=lambda *_a, **_k: result,
        reset=lambda *_a, **_k: events.append("ik.reset"),
    )
    cfg = types.SimpleNamespace(head_mode="ik")
    frames = [
        types.SimpleNamespace(
            timestamp_ns=10,
            exit_requested=False,
            estop=True,
            calib_stage="static",
            home_requested=True,
            left_ee_pose=[],
            right_ee_pose=[],
            head_ee_pose=[],
            left_gripper=0.0,
            right_gripper=0.0,
        ),
        types.SimpleNamespace(
            timestamp_ns=10,
            exit_requested=False,
            estop=True,
            calib_stage="static",
            home_requested=True,
            left_ee_pose=[],
            right_ee_pose=[],
            head_ee_pose=[],
            left_gripper=0.0,
            right_gripper=0.0,
        ),
    ]

    class Source:
        segment_duration = None

        def __init__(self):
            self.i = 0
            self.done = False

        def start(self):
            pass

        @property
        def latest(self):
            frame = frames[min(self.i, len(frames) - 1)]
            self.i += 1
            if self.i >= len(frames):
                self.done = True
            return frame

    driver = types.SimpleNamespace(
        name="fake",
        _episode=types.SimpleNamespace(recording=True),
        stop_recording_calls=0,
        home_calls=0,
        actuate=lambda *_args, **_kwargs: None,
        extra_hold=lambda *_args, **_kwargs: None,
        stop_all_motion=lambda: events.append("motion-stop"),
    )

    def stop_recording_episode():
        if driver._episode.recording:
            events.append("record-stop")
        driver._episode.recording = False
        driver.stop_recording_calls += 1

    driver.stop_recording_episode = stop_recording_episode
    driver.home_to_nominal = lambda: (
        setattr(driver, "home_calls", driver.home_calls + 1),
        events.append("home"),
    )
    driver.engage_reset = lambda *_a, **_k: events.append("engage_reset")
    args = types.SimpleNamespace(
        ik_rate=100.0,
        speed=1.0,
        cmd_rate=10.0,
        head_lpf_tau=0.0,
        head_planar_pos_deadband=0.0,
        head_planar_yaw_deadband=0.0,
    )

    mod.run_loop(
        Source(),
        ik,
        cfg,
        driver,
        args,
        replay=True,
        clock=_FakeClock(),
        realtime=False,
        enable={"arms": True, "torso": True, "head": True, "base": True},
    )

    assert driver.stop_recording_calls >= 1
    assert driver.home_calls == 1
    # Homing puts the HARDWARE at nominal; the solver and the odometry origin must be
    # re-anchored there in the same breath (and only after the homing motion finished),
    # or the loop keeps solving the pre-home targets from a stale configuration.
    assert events == [
        "motion-stop",
        "record-stop",
        "home",
        "ik.reset",
        "engage_reset",
    ]


def test_build_follower_status_serializes_current_hold_and_safety_state():
    mod = _load_wbc_vr_robot()
    result = types.SimpleNamespace(
        success=False,
        held=True,
        left_ee_error=0.0123,
        right_ee_error=0.0456,
        stability_margin=-0.025,
        safety_status="HELD: self-collision -1.2cm",
    )
    vr = types.SimpleNamespace(calib_stage="teleop")

    status = mod._build_follower_status(
        vr,
        result,
        estop=False,
        hold=True,
        hold_reason="source_timeout",
        episode_id=7,
        timestamp_ns=123,
    )

    assert status.timestamp_ns == 123
    assert status.stage == "teleop"
    assert status.estop is False
    assert status.success is False
    assert status.held is True
    assert status.hold is True
    assert status.hold_reason == "source_timeout"
    assert status.safety_status == "HELD: self-collision -1.2cm"
    assert status.left_ee_error_mm == pytest.approx(12.3)
    assert status.right_ee_error_mm == pytest.approx(45.6)
    assert status.episode_id == 7
    # stability margin is no longer mirrored into the HUD status frame; it lives only
    # in the traj log / console print and (when dangerous) inside safety_status.
    assert not hasattr(status, "stability_margin_cm")


def test_recorder_episode_id_reports_next_save_or_minus_one():
    mod = _load_wbc_vr_robot()

    recording = types.SimpleNamespace(_episode=types.SimpleNamespace(episode_id=7))
    assert mod._recorder_episode_id(recording) == 7
    # --record off: the driver has no recorder, so the HUD gets the omit sentinel.
    assert mod._recorder_episode_id(types.SimpleNamespace()) == -1
    assert mod._recorder_episode_id(types.SimpleNamespace(_episode=None)) == -1


def test_stop_all_motion_stops_base_and_overwrites_arm_targets_with_measured_pose():
    mod = _load_wbc_vr_robot()

    class StopComponent:
        def __init__(self):
            self.stopped = 0

        def stop(self):
            self.stopped += 1

    class ArmComponent:
        def __init__(self, q):
            self.q = np.asarray(q, dtype=float)
            self.commands = []

        def get_joint_pos(self):
            return self.q

        def set_joint_pos(self, q, wait_time=0.0):
            self.commands.append((np.asarray(q, dtype=float), wait_time))

    chassis = StopComponent()
    torso = StopComponent()
    head = StopComponent()
    left_arm = ArmComponent([0.1, 0.2])
    right_arm = ArmComponent([0.3, 0.4])
    driver = object.__new__(mod.HardwareDriver)
    driver.robot = types.SimpleNamespace(
        chassis=chassis,
        torso=torso,
        head=head,
        left_arm=left_arm,
        right_arm=right_arm,
    )
    driver.has_chassis = True
    driver._prev_cmd = {"left_arm": np.ones(2), "right_arm": np.ones(2)}

    mod.HardwareDriver.stop_all_motion(driver)

    assert chassis.stopped == 1
    assert torso.stopped == 1
    assert head.stopped == 1
    np.testing.assert_allclose(left_arm.commands[-1][0], left_arm.q)
    np.testing.assert_allclose(right_arm.commands[-1][0], right_arm.q)
    assert left_arm.commands[-1][1] == 0.0
    assert right_arm.commands[-1][1] == 0.0
    assert all(v is None for v in driver._prev_cmd.values())


def test_build_ik_uses_yaml_config_without_base_dofs_override(monkeypatch):
    mod = _load_wbc_vr_robot()
    seen = {}

    class FakeIK:
        def __init__(self, cfg):
            seen["cfg"] = cfg

        def reset(self):
            seen["reset"] = True

    monkeypatch.setattr(mod, "VegaWholeBodyIK", FakeIK)

    ik, cfg = mod._build_ik()

    assert isinstance(ik, FakeIK)
    assert seen == {"cfg": cfg, "reset": True}


def test_hardware_driver_preserves_yaw_hold_feedback_in_xy_mode():
    mod = _load_wbc_vr_robot()
    driver = object.__new__(mod.HardwareDriver)
    driver.cfg = types.SimpleNamespace(base_dofs="xy")

    actual = mod.HardwareDriver._mask_base_twist(
        driver,
        np.array([0.12, -0.08, 0.31]),
        allow_yaw_hold=True,
    )

    np.testing.assert_allclose(actual, [0.12, -0.08, 0.31])


def test_trajlog_flush_writes_per_episode_file_and_frees_buffer(tmp_path):
    """flush(N) writes episode_<N>_debug.hdf5, frees the buffer, and skips empty flushes."""
    mod = _load_wbc_vr_robot()
    traj = mod._TrajLog(str(tmp_path), meta={"mode": "live"})
    row = dict(
        estop=False,
        success=True,
        held=False,
        left_ee_error=0.001,
        right_ee_error=0.002,
        base_pose=np.zeros(3, dtype=np.float32),
    )
    traj.append(**row)
    traj.append(**dict(row, base_pose=np.array([0.1, 0.0, 0.0], dtype=np.float32)))
    assert len(traj) == 2

    path = traj.flush(5)

    assert path == str(tmp_path / "episode_5_debug.hdf5")
    assert len(traj) == 0  # buffer released -> RAM stays bounded across episodes

    with h5py.File(path, "r") as f:
        g = f["debug"]
        assert int(g.attrs["n_frames"]) == 2
        assert int(g.attrs["episode_id"]) == 5
        assert g.attrs["mode"] == "live"
        assert g["left_ee_error"].shape == (2,)
        assert g["base_pose"].shape == (2, 3)

    # A second flush with an empty buffer writes no phantom episode_6_debug.hdf5.
    assert traj.flush(6) is None
    assert not (tmp_path / "episode_6_debug.hdf5").exists()


def test_trajlog_disabled_is_noop():
    """debug_dir=None (no --debug-dir): append/flush do nothing, so /debug costs nothing."""
    mod = _load_wbc_vr_robot()
    traj = mod._TrajLog(None)
    traj.append(
        estop=False,
        success=True,
        held=False,
        left_ee_error=0.0,
        right_ee_error=0.0,
        base_pose=np.zeros(3, dtype=np.float32),
    )
    assert len(traj) == 0
    assert traj.flush(0) is None


def test_stop_recording_episode_flushes_paired_debug_with_presave_id(monkeypatch):
    """stop_recording_episode flushes /debug keyed by the pre-increment id (episode_N)."""
    mod = _load_wbc_vr_robot()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    driver = object.__new__(mod.HardwareDriver)
    episode = types.SimpleNamespace(recording=True, saving=False, episode_id=7)
    episode.num_frames = lambda: 4

    def _stop():
        episode.recording = False
        episode.episode_id += 1  # the real EpisodeRecorder increments inside stop()
        return "/tmp/episode_7.hdf5"

    episode.stop = _stop
    driver._episode = episode
    flushed = []
    driver._traj = types.SimpleNamespace(flush=flushed.append)

    mod.HardwareDriver.stop_recording_episode(driver)

    assert flushed == [7]  # id captured pre-increment, not the post-stop 8


def test_abort_recording_episode_uses_incomplete_quarantine(monkeypatch):
    mod = _load_wbc_vr_robot()
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)
    driver = object.__new__(mod.HardwareDriver)
    calls = []
    episode = types.SimpleNamespace(
        recording=True,
        saving=False,
        num_frames=lambda: 3,
        last_save_error=None,
    )

    def abort(reason):
        calls.append(reason)
        episode.recording = False
        return "/tmp/episode_8.hdf5.partial"

    episode.abort = abort
    driver._episode = episode

    mod.HardwareDriver.abort_recording_episode(driver, "camera skew")

    assert calls == ["camera skew"]
    assert not episode.recording
