"""Simple data schemas for communication between components.

These dataclasses define the structure of messages passed via Zenoh.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VRJointData:
    """VR state and Cartesian targets streamed by the WebXR leader.

    The leader maps Quest 3 controller and headset poses into calibrated robot-frame
    targets. The follower subscribes and solves whole-body IK on its live state.
    """

    timestamp_ns: int
    # Gripper scale [0.0 = open, 1.0 = fully closed] mapped from controller trigger
    left_gripper: float = 0.0
    right_gripper: float = 0.0
    # Control flags
    estop: bool = True  # Default True (safe) until calibration completes
    exit_requested: bool = False
    home_requested: bool = False
    # Calibration stage published by the leader: static or teleop.
    calib_stage: str = "static"
    left_ee_pose: List[float] = field(default_factory=list)
    right_ee_pose: List[float] = field(default_factory=list)
    # Calibrated headset pose (flattened row-major 4x4) in the robot base frame at
    # calibration -- the head-frame (zed_depth_frame) teleop target. Published by
    # wbc_vr_leader; the WBC follower solves its own head IK against this from the
    # live whole-body configuration (so base yaw / torso lean are compensated).
    # Empty when not teleoperating.
    head_ee_pose: List[float] = field(default_factory=list)


# Terminal ``WBCFollowerStatus.stage`` published once by the follower when its control
# loop dies on an exception (record-abort guards, IK runaway abort, hardware errors).
# The leader HUD keys a persistent red FOLLOWER ABORTED banner on it; every other stage
# value mirrors the leader's live calib_stage.
WBC_FOLLOWER_STAGE_ABORTED = "aborted"


@dataclass
class WBCFollowerStatus:
    """Live whole-body follower status for the VR headset HUD."""

    timestamp_ns: int
    stage: str = "static"
    estop: bool = True
    success: bool = True
    held: bool = False
    hold: bool = True
    hold_reason: str = ""
    safety_status: str = "ok"
    left_ee_error_mm: float = 0.0
    right_ee_error_mm: float = 0.0
