#!/usr/bin/env python
"""Publish the Vega base pose tracked by the mount-rigid iPhone (ARKit) over Zenoh.

Runs ON THE ROBOT (vega-1), where the phone is USB-attached and ``record3d`` is built:

    python scripts/base_tracking/base_pose_pub.py --calib /secure/path/calib_result.json

Per ARKit frame (~15 Hz) it applies the calibration from ``calibration/calib_solve.py``

    arworld_T_base = arworld_T_arkit . T(R_fix^-1) . phoneCam_T_base

and rotates ARKit's gravity-aligned world (+y up) into the Z-up world ``W`` the robot
stack uses, so the published pose is directly comparable to wheel odometry:

    W_T_base = T(R_W_ARWORLD) . arworld_T_base

The published frame is the ARKit SESSION world: its origin is wherever the phone was
when the Record3D stream started and it survives re-engages, so the consumer
(``omniteleop/follower/arkit_base.py``) is the one that re-zeros at engage.

Relocalization jumps are flagged, not smoothed: ARKit can teleport its world after a
tracking loss, which would silently corrupt every pose derived from an origin captured
before the jump. ``jump`` marks the offending sample and ``jumps`` counts them, leaving
the safety decision to the consumer.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time

import numpy as np
from dexcomm import Node
from dexcomm.codecs import DictDataCodec

TOPIC = "tracking/base_pose"

# ARKit world (right-handed, +y up) -> the Z-up world W the robot stack works in:
# W_x = AR_x, W_y = -AR_z, W_z = AR_y.
R_W_ARWORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

# Relocalization gate. The chassis tops out at 0.45 m/s (wbik.yaml base_max_speed), so
# these are ~3x the fastest real motion between two 15 Hz frames -- large enough never to
# fire on genuine driving, small enough to catch an ARKit world teleport.
MAX_STEP_M = 0.15
MAX_STEP_RAD = np.deg2rad(15.0)


def quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Rotation matrix of a (x, y, z, w) quaternion."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError(f"non-unit ARKit quaternion ({qx}, {qy}, {qz}, {qw})")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def R_to_quat(R: np.ndarray) -> list:
    """(x, y, z, w) quaternion of a rotation matrix (Shepperd's method)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return [float(v) for v in q]


def load_calib(path: str) -> np.ndarray:
    """``phoneCam_T_base`` premultiplied by ``T(R_fix^-1)``: arkitCam_T_base."""
    with open(path) as f:
        c = json.load(f)
    X = np.asarray(c["phoneCam_T_base"], dtype=np.float64)
    R_fix = np.asarray(c["R_fix_arkitcam_to_phonecam"], dtype=np.float64)
    if X.shape != (4, 4) or R_fix.shape != (3, 3):
        raise ValueError(
            f"{path}: phoneCam_T_base must be 4x4 and R_fix 3x3, got {X.shape} and {R_fix.shape}"
        )
    T_fix = np.eye(4)
    T_fix[:3, :3] = R_fix.T  # arkitCam_T_phoneCam = R_fix^-1
    print(
        f"[base_pose_pub] calibration {path}: phone camera at "
        f"{c['cam_in_base_mm']} mm in base, {c['n_samples']} samples, "
        f"gravity closes to {c['gravity_check_deg']['mean']:.2f} deg"
    )
    return T_fix @ X  # arkitCam_T_base


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--calib", required=True, help="calib_result.json written by calibration/calib_solve.py"
    )
    ap.add_argument(
        "--namespace",
        default=os.environ.get("ROBOT_NAME", ""),
        help="Zenoh namespace of this robot (default $ROBOT_NAME); the pose "
        f"is published on <namespace>/{TOPIC}",
    )
    ap.add_argument(
        "--report",
        type=float,
        default=10.0,
        help="seconds between console health lines (default 10; 0 disables)",
    )
    args = ap.parse_args()
    if not args.namespace:
        raise SystemExit("[base_pose_pub] --namespace is required (or export ROBOT_NAME)")
    if not os.environ.get("ZENOH_CONFIG"):
        raise SystemExit(
            "[base_pose_pub] export ZENOH_CONFIG first (~/.dexmate/comm/zenoh/<robot>.dzcfg)"
        )

    arkitCam_T_base = load_calib(args.calib)

    from record3d import Record3DStream

    devs = Record3DStream.get_connected_devices()
    if not devs:
        raise SystemExit(
            "[base_pose_pub] no iPhone on USB (check `lsusb | grep 12a8` "
            "and that Record3D is streaming)"
        )
    sess = Record3DStream()
    # Latest-wins delivery (patched record3d lib): whenever decode + this callback fall
    # behind the phone's frame rate, stale queued frames are skipped instead of replayed --
    # without this the stream lags real time WITHOUT BOUND (36+ s after a 55 min session)
    # while seq/age/jump all look healthy. AttributeError here = unpatched record3d build,
    # raised BEFORE the Zenoh node exists so nothing needs cleanup.
    sess.latest_frame_only = True
    node = Node(name="arkit_base_pose", namespace=args.namespace)
    pub = node.create_publisher(TOPIC, encoder=DictDataCodec.encode)
    state = {"seq": 0, "jumps": 0, "prev": None, "last_report": 0.0, "z0": None}
    stopped = threading.Event()

    def on_frame() -> None:
        p = sess.get_camera_pose()
        t_ns = time.time_ns()
        arw_T_arkit = np.eye(4)
        arw_T_arkit[:3, :3] = quat_to_R(p.qx, p.qy, p.qz, p.qw)
        arw_T_arkit[:3, 3] = (p.tx, p.ty, p.tz)
        T = arw_T_arkit @ arkitCam_T_base  # arworld_T_base
        T[:3, :3] = R_W_ARWORLD @ T[:3, :3]  # -> Z-up world W
        T[:3, 3] = R_W_ARWORLD @ T[:3, 3]
        R, xyz = T[:3, :3], T[:3, 3]
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        # Roll/pitch of the base in the gravity-aligned world: a constant on a flat floor,
        # so drift here is the honest health signal for the calibration + the mount.
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(-np.arcsin(np.clip(R[2, 0], -1.0, 1.0)))

        jump = False
        if state["prev"] is not None:
            d = xyz - state["prev"][:3]
            dyaw = abs((yaw - state["prev"][3] + np.pi) % (2 * np.pi) - np.pi)
            jump = bool(np.linalg.norm(d) > MAX_STEP_M or dyaw > MAX_STEP_RAD)
            if jump:
                state["jumps"] += 1
                print(
                    f"[base_pose_pub] ARKit JUMP #{state['jumps']}: "
                    f"{np.linalg.norm(d) * 1000:.0f} mm / {np.degrees(dyaw):.1f} deg "
                    "in one frame -- consumer must re-zero (re-engage) before trusting "
                    "the pose again."
                )
        state["prev"] = np.array([xyz[0], xyz[1], xyz[2], yaw])
        state["seq"] += 1
        pub.publish(
            {
                "seq": state["seq"],
                "t_ns": t_ns,
                "xyz": [float(v) for v in xyz],
                "quat_xyzw": R_to_quat(R),
                "planar": [float(xyz[0]), float(xyz[1]), yaw],
                "rp_rad": [roll, pitch],
                "jump": jump,
                "jumps": state["jumps"],
                # Cumulative stale frames skipped by latest_frame_only. A growing value means
                # the pipeline is saturated (lower the app's streaming resolution/FPS); a frozen
                # value with fresh seq means delivery is keeping up at the camera rate.
                "drops": int(sess.skipped_frames),
                # Raw ARKit device pose: the provenance that lets a REFINED calibration be
                # applied offline to data recorded against this one.
                "arkit_txyz": [float(p.tx), float(p.ty), float(p.tz)],
                "arkit_quat_xyzw": [float(p.qx), float(p.qy), float(p.qz), float(p.qw)],
            }
        )
        if args.report > 0 and time.time() - state["last_report"] > args.report:
            state["last_report"] = time.time()
            if state["z0"] is None:
                state["z0"] = xyz[2]
            print(
                f"[base_pose_pub] #{state['seq']} x={xyz[0]:+.3f} y={xyz[1]:+.3f} "
                f"yaw={np.degrees(yaw):+7.2f}deg | z drift {(xyz[2] - state['z0']) * 1000:+.0f}mm "
                f"roll/pitch {np.degrees(roll):+.2f}/{np.degrees(pitch):+.2f}deg "
                f"| jumps={state['jumps']} drops={sess.skipped_frames}"
            )

    sess.on_new_frame = on_frame
    sess.on_stream_stopped = stopped.set
    if not sess.connect(devs[0]):
        raise SystemExit(
            "[base_pose_pub] usbmuxd connection to the iPhone failed -- is "
            "Record3D streaming (USB mode) and no other client attached?"
        )
    print(
        f"[base_pose_pub] publishing {args.namespace}/{TOPIC} at the ARKit frame rate "
        "(~15 Hz). Ctrl-C to stop."
    )
    try:
        while not stopped.wait(0.5):
            pass
        raise SystemExit(
            "[base_pose_pub] Record3D stream stopped (phone unplugged / app "
            "closed) -- restart this publisher after restoring the stream."
        )
    except KeyboardInterrupt:
        print(f"\n[base_pose_pub] stopped after {state['seq']} frames, {state['jumps']} jumps.")
    finally:
        # Some dexcomm builds have no Node.close(); don't let shutdown bury the real exit
        # message (e.g. "Record3D stream stopped") under an AttributeError traceback.
        close = getattr(node, "close", None) or getattr(node, "shutdown", None)
        if close is not None:
            close()


if __name__ == "__main__":
    main()
