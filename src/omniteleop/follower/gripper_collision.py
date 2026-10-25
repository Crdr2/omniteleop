"""Measured Robotiq stroke for collision proxies (raw achieved encoder position).

The September 25 splat fit supplies motion, not the splat rest shape. Rebase its
EE-frame transforms to the URDF's open mesh, preserving the installed tip shape.
Unknown feedback encloses the entire stroke; feedback age expands a width interval.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STROKE = json.loads(Path(__file__).with_name("robotiq_stroke_0925.json").read_text())["sides"]
MAX_ENCODER_SPEED = 2.0  # raw gPO/255 per second; bound both travel directions
MAX_STATUS_AGE = 0.25
MAX_RESPONSE_TIME = 0.075


def feedback_interval(
    status: dict | None,
    now: float,
    horizon: float,
    *,
    command: bytes | None = None,
    command_change: float | None = None,
) -> tuple[float, float]:
    """Bound achieved width through the next IK step; reject stale/faulty FC03 data.

    Age starts at REQUEST send, so wire latency is included. Never use gPR or a
    trigger as evidence of achieved position, including while holding an object.
    """
    if not status:
        return (0.0, 1.0)
    try:
        request = float(status["request_send_monotonic_ns"]) / 1e9
        read = float(status["read_monotonic_ns"]) / 1e9
        position = float(status["gPO"]) / 255
        healthy = (
            np.isfinite([request, read, now, horizon, position]).all()
            and request > 0
            and 0 <= read - request <= MAX_RESPONSE_TIME
            and 0 <= now - read <= MAX_STATUS_AGE
            and read <= now
            and horizon >= 0
            and 0 <= position <= 1
            and status.get("gACT") == 1
            and status.get("gSTA_bits") == 3
            and status.get("gFLT") == 0
            and status.get("kFLT") == 0
        )
    except (KeyError, TypeError, ValueError):
        return (0.0, 1.0)
    if not healthy:
        return (0.0, 1.0)
    past_travel = MAX_ENCODER_SPEED * (now - request)
    lo, hi = position - past_travel, position + past_travel
    if (
        command is not None
        and len(command) >= 11
        and command_change is not None
        and np.isfinite(command_change)
        and request >= command_change
        and status.get("gPR") == command[10]
        and status.get("gGTO") == int(bool(command[7] & 0x08))
    ):
        # A post-command reply with the matching echo bounds motion toward its goal.
        # A latched rGTO=0 stop cannot spontaneously open while holding an object.
        # Include the entire route to gPR for go-to (even after closing contact).
        target = command[10] / 255 if command[7] & 0x08 else position
        lo = max(lo, min(position, target))
        hi = min(hi, max(position, target))
    # The next dispatch may change direction, so the future-step allowance is always
    # bilateral, even for a stopped gripper with a confirmed command echo.
    padding = MAX_ENCODER_SPEED * horizon + 2 / 255
    return (max(0.0, lo - padding), min(1.0, hi + padding))


def stroke_nodes(side: str) -> np.ndarray:
    return np.array(
        [[r[k] for k in ("w", "d_m", "z_m", "theta_deg")] for r in STROKE[side]["rows"]]
    )


def pose_open_points(side: str, points: np.ndarray, signs: np.ndarray, width: float) -> np.ndarray:
    """Pose open EE-frame points at raw encoder width using T(width) T(open)^-1."""
    table = stroke_nodes(side)
    d, z, theta = [np.interp(width, table[:, 0], table[:, i]) for i in (1, 2, 3)]
    d0, z0, theta0 = table[0, 1:]
    pivot = np.asarray(STROKE[side]["pivots"])[(signs < 0).astype(int)]
    delta = (
        points
        - pivot
        - np.column_stack((np.zeros(len(signs)), -signs * d0, np.full(len(signs), z0)))
    )
    angle = signs * np.deg2rad(theta - theta0)
    c, s = np.cos(angle), np.sin(angle)
    rotated = delta.copy()
    rotated[:, 1] = c * delta[:, 1] - s * delta[:, 2]
    rotated[:, 2] = s * delta[:, 1] + c * delta[:, 2]
    return (
        rotated
        + pivot
        + np.column_stack((np.zeros(len(signs)), -signs * d, np.full(len(signs), z)))
    )


def enclose_interval(side, points, signs, radii, interval):
    """Enclose each moving sphere throughout a width interval, including arc bulge.

    On each table segment translation and angle are linear. Distance to its endpoint
    chord is bounded by rho * delta_angle**2 / 8. Adding that bound covers continuous
    motion between nodes, rather than only the sampled encoder positions.
    """
    lo, hi = interval
    if not np.isfinite([lo, hi]).all() or not 0 <= lo <= hi <= 1:
        lo, hi = 0.0, 1.0
    table = stroke_nodes(side)
    widths = np.unique(np.r_[lo, table[(table[:, 0] > lo) & (table[:, 0] < hi), 0], hi])
    poses = np.array([pose_open_points(side, points, signs, w) for w in widths])
    centers = (poses.min(axis=0) + poses.max(axis=0)) / 2
    theta = np.deg2rad(np.interp(widths, table[:, 0], table[:, 3]))
    pivot = np.asarray(STROKE[side]["pivots"])[(signs < 0).astype(int)]
    d0, z0 = table[0, 1:3]
    delta = (
        points
        - pivot
        - np.column_stack((np.zeros(len(signs)), -signs * d0, np.full(len(signs), z0)))
    )
    rho = np.linalg.norm(delta[:, 1:], axis=1)
    arc = rho * (np.max(np.diff(theta) ** 2) / 8 if len(widths) > 1 else 0)
    return centers, radii + np.linalg.norm(poses - centers, axis=2).max(axis=0) + arc
