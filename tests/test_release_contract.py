"""Integration checks for portable assets, recording provenance, and torso modes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omniteleop.follower.joystick_torso import JoystickTorsoCommand
from omniteleop.follower.whole_body_ik import DEFAULT_CONFIG_PATH, WBCConfig


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name + "_release_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copied_config_still_finds_bundled_model(tmp_path):
    config = tmp_path / "custom.yaml"
    config.write_text(Path(DEFAULT_CONFIG_PATH).read_text())
    expected = WBCConfig()
    actual = WBCConfig.from_yaml(config)
    assert actual.urdf_path == expected.urdf_path
    assert Path(actual.urdf_path).is_file()
    assert Path(actual.collision_spheres_urdf).is_file()


def test_recording_provenance_contains_only_existing_release_files():
    module = script("wbc_joystick_robot")
    driver = object.__new__(module.JoystickHardwareDriver)
    driver.args = SimpleNamespace(config=None)
    driver.cfg = WBCConfig()
    metadata = driver._recording_provenance_metadata()
    snapshots = metadata["source_snapshot"]
    assert {
        "robot_urdf",
        "gripper_stroke",
        "wbc_joystick_robot",
        "head_zed_publisher",
    } <= snapshots.keys()
    for snapshot in snapshots.values():
        assert len(snapshot["sha256"].item()) == 64
        assert len(snapshot["content"].item()) > 0


@pytest.mark.parametrize("mode", ["fixed", "stick", "ik"])
def test_torso_modes_build_with_safety_and_base_locked(mode):
    module = script("wbc_joystick_robot")
    ik, cfg = module._build_joystick_ik(None, lock_torso_in_ik=mode != "ik", torso_ik=mode == "ik")
    assert cfg.lock_base_in_ik
    assert cfg.lock_torso_in_ik == (mode != "ik")
    assert ik.collision_sphere_model is not None
    assert len(ik.collision_sphere_model.collisionPairs) > 0
    assert ik.collision_barrier is not None
    assert ik.stability_margin(ik.configuration.q) > cfg.com_safety_margin
    if mode == "stick":
        command = JoystickTorsoCommand.from_ik(ik, speed=0.05)
        assert command.path.min_height < -0.1
        assert command.path.max_height > 0.0
        height = -0.1
        joints = command.path.joints(height)
        assert np.isfinite(joints).all()
