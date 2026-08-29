"""Time-scaled replay of Cartesian targets from a recorded robot episode."""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from omniteleop.common.schemas import VRJointData

DEFAULT_REPLAY_GAP_LIMIT_MULTIPLE = 5.0

_EPISODE_POSE_DATASETS = {
    "left_ee_pose": "action/eef/left",
    "right_ee_pose": "action/eef/right",
    "head_ee_pose": "action/head",
}
_EPISODE_SCALAR_DATASETS = {
    "left_gripper": "action/gripper/left",
    "right_gripper": "action/gripper/right",
}
_EPISODE_TIME_DATASET = "timestamp_ns"


def _decode_episode_frames(arrays: dict[str, np.ndarray]) -> list[tuple[float, VRJointData]]:
    """Rebuild timestamped leader messages from an episode's solver inputs.

    Derived joint commands and observations are deliberately ignored: replay re-solves
    the original Cartesian intent with the current IK configuration.
    """
    timestamps = np.asarray(arrays[_EPISODE_TIME_DATASET], dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError(
            f"{_EPISODE_TIME_DATASET} must be a 1-D array of at least two frames; "
            f"got {timestamps.shape}"
        )
    if not np.all(np.diff(timestamps) > 0):
        bad = int(np.argmin(np.diff(timestamps)))
        raise ValueError(
            f"{_EPISODE_TIME_DATASET} must increase strictly; frames {bad} and "
            f"{bad + 1} contain {int(timestamps[bad])} and {int(timestamps[bad + 1])}"
        )

    frame_count = int(timestamps.size)
    poses: dict[str, np.ndarray] = {}
    for field, path in _EPISODE_POSE_DATASETS.items():
        matrices = np.asarray(arrays[path], dtype=np.float64)
        if matrices.shape != (frame_count, 4, 4):
            raise ValueError(f"{path} must have shape {(frame_count, 4, 4)}, got {matrices.shape}")
        if not np.all(np.isfinite(matrices)):
            raise ValueError(f"{path} contains non-finite values")
        if not np.allclose(matrices[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError(f"{path} contains invalid homogeneous transforms")
        poses[field] = matrices

    scalars: dict[str, np.ndarray] = {}
    for field, path in _EPISODE_SCALAR_DATASETS.items():
        values = np.asarray(arrays[path], dtype=np.float64)
        if values.shape != (frame_count,):
            raise ValueError(f"{path} must have shape {(frame_count,)}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains non-finite values")
        scalars[field] = values

    relative_times = (timestamps - timestamps[0]) * 1e-9
    frames: list[tuple[float, VRJointData]] = []
    for index in range(frame_count):
        fields: dict = {
            "timestamp_ns": int(timestamps[index]),
            "calib_stage": "teleop",
            "estop": False,
        }
        for field, matrices in poses.items():
            fields[field] = matrices[index].reshape(16).tolist()
        for field, values in scalars.items():
            fields[field] = float(values[index])
        frames.append((float(relative_times[index]), VRJointData(**fields)))
    return frames


class ReplaySource:
    """Release recorded target frames on a clock stretched by ``1 / speed``."""

    def __init__(
        self,
        frames: list[tuple[float, VRJointData]],
        speed: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError(f"speed must be finite and positive, got {speed}")
        if not frames:
            raise ValueError("cannot replay an empty episode")

        start_time = frames[0][0]
        self.speed = float(speed)
        self._clock = clock
        self._vr = [frame for _, frame in frames]
        self._release = [max(0.0, (timestamp - start_time) / self.speed) for timestamp, _ in frames]
        self._t0: Optional[float] = None
        self._cursor = -1
        self.segment_duration: Optional[float] = None

    def start(self) -> None:
        """Anchor the replay clock at the current time."""
        self._t0 = self._clock()
        self._cursor = -1
        self.segment_duration = None

    @property
    def started(self) -> bool:
        return self._t0 is not None

    @property
    def latest(self) -> Optional[VRJointData]:
        """Return the newest frame whose scaled release time has elapsed."""
        if self._t0 is None:
            self.start()
        assert self._t0 is not None
        elapsed = self._clock() - self._t0
        while self._cursor + 1 < len(self._vr) and self._release[self._cursor + 1] <= elapsed:
            previous = self._cursor
            self._cursor += 1
            self.segment_duration = (
                self._release[self._cursor] - self._release[previous] if previous >= 0 else None
            )
        return self._vr[self._cursor] if self._cursor >= 0 else None

    @property
    def done(self) -> bool:
        """Whether the final frame has been released."""
        return (
            self._t0 is not None
            and self._cursor >= len(self._vr) - 1
            and (self._clock() - self._t0) >= self._release[-1]
        )

    @property
    def total_duration(self) -> float:
        return self._release[-1]

    @classmethod
    def from_episode_hdf5(
        cls,
        path: str,
        speed: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> ReplaySource:
        """Load a robot episode and replay its original Cartesian targets."""
        import h5py

        wanted = (
            _EPISODE_TIME_DATASET,
            *_EPISODE_POSE_DATASETS.values(),
            *_EPISODE_SCALAR_DATASETS.values(),
        )
        with h5py.File(path, "r") as episode:
            missing = [key for key in wanted if key not in episode]
            if missing:
                raise ValueError(f"{path}: not a compatible robot episode; missing {missing}")
            arrays = {key: episode[key][()] for key in wanted}
        return cls(_decode_episode_frames(arrays), speed, clock)
