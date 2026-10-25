#!/usr/bin/env python
"""Capture phone-to-base calibration samples from a shared ChArUco board.

Bracket each remote phone grab with head-camera frames and measured head/torso
joints; reject samples where the board or robot moves. The head sweeps around a
downward aim between samples. See scripts/base_tracking/README.md for setup.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dexcomm import call_service, subscribe_once
from dexcomm.codecs import JsonDataCodec, RGBImageCodec

NS = os.environ.get("ROBOT_NAME", "")
SSH = []
SCP_HOST = ""
SCP_OPTS = []
ROBOT_PY = "python"
ROBOT_TMP = "/tmp/omniteleop_phone_calibration"
ROBOT_GRAB = ROBOT_TMP + "/phone_grab.py"

# OpenCV convention: squaresX runs along the board's LONG side here — the board
# is sold as "9x12", but with (9,12) all 54 markers detect and ZERO charuco
# corners interpolate; with (12,9) all 88 interpolate (verified on live frames).
SQUARES_XY = (12, 9)
SQUARE_M, MARKER_M = 0.030, 0.022
INNER = (8, 11)  # findChessboardCorners pattern size, verified on live frames
DICT_CANDIDATES = ["DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000", "DICT_5X5_50"]

DEFAULT_DOWN = np.radians(45.0)  # board is near the lowest torso joint
MIN_DOWN = np.radians(20.0)  # never aim near the horizon
HEAD_STABLE_PX = 0.5
BOARD_MOVED_PX = 1.0

# Per-sample head-pose offsets from the aim, in units of the --sweep amplitude and
# cycled in order: (pitch_down, yaw_left, split).  head_j1 (+) and head_j3 (-) both
# pitch the camera down, so pitch_down p and neck split s become the joint offsets
# (p/2 + s, yaw, -p/2 + s).  pitch_down is never negative, so every sample looks at
# least as far down as the aim and the board below stays in frame.  The last two
# entries only change the split: same view direction, different neck configuration,
# which turns any error in the head kinematics into visible solve residual.
HEAD_SWEEP = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),
    (1.0, -1.0, 0.0),
    (2.0, 0.7, 0.0),
    (2.0, -0.7, 0.0),
    (0.5, 0.0, 1.0),
    (1.5, 0.0, -1.0),
]
HEAD_SPEED = 0.6  # rad/s while sweeping (gentle)


# detections
def detect_head_chessboard(gray):
    """Classic chessboard detection with upscale fallback; subpix at native res.
    Returns (88,2) float32 corners or None."""
    for s in (1, 2, 3):
        g = gray if s == 1 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        ok, c = cv2.findChessboardCorners(
            g, INNER, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ok:
            c = (c / s).astype(np.float32)
            c = cv2.cornerSubPix(
                gray,
                c,
                (3, 3),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
            )
            return c.reshape(-1, 2)
    return None


def make_board(dict_name, legacy):
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(SQUARES_XY, SQUARE_M, MARKER_M, adict)
    board.setLegacyPattern(legacy)
    return board


def detect_phone_charuco(gray, cache):
    """ChArUco detection on the phone image; auto-identifies dictionary + legacy
    flag on first success and caches it. Returns (ids, corners) or (None, None)."""
    combos = (
        [tuple(cache["board"])]
        if cache.get("board")
        else [(d, legacy) for d in DICT_CANDIDATES for legacy in (False, True)]
    )
    best = (0, None, None, None)
    for dname, legacy in combos:
        det = cv2.aruco.CharucoDetector(make_board(dname, legacy))
        ccorners, cids, _, _ = det.detectBoard(gray)
        n = 0 if cids is None else len(cids)
        if n > best[0]:
            best = (n, cids, ccorners, (dname, legacy))
    n, cids, ccorners, combo = best
    if n >= 8 and not cache.get("board"):
        cache["board"] = list(combo)
        print(f"    phone board identified: {combo[0]} legacy={combo[1]}")
    return (cids.ravel(), ccorners.reshape(-1, 2)) if n >= 8 else (None, None)


def head_board_shift_px(before, after):
    """Robust board displacement, ignoring isolated corner-detector jitter."""
    if after is None:
        return float("inf")
    direct = np.linalg.norm(before - after, axis=1)
    reversed_ = np.linalg.norm(before - after[::-1], axis=1)
    return float(min(np.median(direct), np.median(reversed_)))


def head_board_moved(before, after):
    """Whether the head-camera chessboard coherently moved during capture."""
    return head_board_shift_px(before, after) > BOARD_MOVED_PX


# sources
def head_frame():
    msg = subscribe_once(
        f"{NS}/sensors/head_camera/left_rgb", timeout=10.0, decoder=RGBImageCodec.decode
    )
    if msg is None:
        sys.exit("no head frame in 10 s — is the head publisher running on vega-1?")
    return msg["data"], msg["timestamp_ns"]


def acquire_head_chessboard(timeout, require_stable):
    """Retry head detections, optionally waiting for two stable observations."""
    latest_frame = latest_corners = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame, _ = head_frame()
        corners = detect_head_chessboard(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        if corners is None:
            continue
        if not require_stable:
            return frame, corners
        stable = (
            latest_corners is not None
            and head_board_shift_px(latest_corners, corners) < HEAD_STABLE_PX
        )
        latest_frame, latest_corners = frame, corners
        if stable:
            break
    return latest_frame, latest_corners


def head_intrinsics():
    info = call_service(
        f"{NS}/sensors/head_camera/info",
        request={},
        timeout=5.0,
        request_encoder=JsonDataCodec.encode,
        response_decoder=JsonDataCodec.decode,
    )
    if not info:
        sys.exit("head_camera/info service did not answer")
    if not info["stereo"]["rectified"]:
        sys.exit("head stream is not rectified — solver assumes zero distortion")
    return np.asarray(info["stereo"]["left_K"], dtype=float), info


def phone_grab(local_dir: Path, tag: str):
    r = subprocess.run(
        SSH + [shlex.join([ROBOT_PY, ROBOT_GRAB, ROBOT_TMP, tag])],
        capture_output=True,
        text=True,
        timeout=45,
    )
    if r.returncode != 0:
        return None, (r.stdout + r.stderr).strip() or f"phone grab exited {r.returncode}"
    subprocess.run(
        [
            "scp",
            "-q",
            *SCP_OPTS,
            f"{SCP_HOST}:{ROBOT_TMP}/{tag}_rgb.npy",
            f"{SCP_HOST}:{ROBOT_TMP}/{tag}_meta.json",
            str(local_dir),
        ],
        check=True,
        timeout=60,
    )
    rgb = np.load(local_dir / f"{tag}_rgb.npy")
    meta = json.loads((local_dir / f"{tag}_meta.json").read_text())
    return (rgb, meta), None


def phone_present():
    r = subprocess.run(SSH + ["lsusb"], capture_output=True, text=True, timeout=15)
    return "05ac:12a8" in r.stdout or "iPhone" in r.stdout


# main
def main():
    global NS, SSH, SCP_HOST, ROBOT_PY, ROBOT_GRAB, ROBOT_TMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot-host", required=True, help="SSH host with the USB-connected iPhone")
    ap.add_argument(
        "--robot-python", default="python", help="remote Python with patched Record3D installed"
    )
    ap.add_argument(
        "--robot-work-dir", default=ROBOT_TMP, help="absolute remote temporary directory"
    )
    ap.add_argument("--namespace", default=NS, help="robot namespace (default $ROBOT_NAME)")
    ap.add_argument(
        "--out", type=Path, default=Path("data/calibration") / time.strftime("session_%Y%m%d_%H%M")
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="capture continuously whenever both detections pass and "
        "the board pose is new; move the board between beeps",
    )
    ap.add_argument("--target", type=int, default=15, help="samples to collect")
    ap.add_argument(
        "--sweep",
        type=float,
        default=0.12,
        help="head-joint sweep amplitude in rad per sample (default "
        "0.12 ~ 7 deg of view direction; 0 = hold the aim)",
    )
    ap.add_argument(
        "--head-pose",
        type=float,
        nargs=3,
        metavar=("J1", "J2", "J3"),
        default=None,
        help="override the automatic downward aim with these head joints "
        "(+j1 / -j3 pitch down, +j2 yaws left)",
    )
    ap.add_argument(
        "--keep-aim",
        action="store_true",
        help="start from the head's current pose instead of --head-pose",
    )
    args = ap.parse_args()
    if not args.namespace or not os.environ.get("ZENOH_CONFIG"):
        ap.error("set ROBOT_NAME (or --namespace) and ZENOH_CONFIG before capture")
    if not args.robot_work_dir.startswith("/") or args.robot_work_dir == "/":
        ap.error("--robot-work-dir must be an absolute directory below /")
    NS, SCP_HOST, ROBOT_PY = args.namespace, args.robot_host, args.robot_python
    SSH = ["ssh", "-o", "BatchMode=yes", SCP_HOST]
    ROBOT_TMP = args.robot_work_dir
    ROBOT_GRAB = ROBOT_TMP.rstrip("/") + "/phone_grab.py"
    args.out.mkdir(parents=True, exist_ok=True)

    subprocess.run(SSH + [shlex.join(["mkdir", "-p", ROBOT_TMP])], check=True)

    subprocess.run(
        [
            "scp",
            "-q",
            *SCP_OPTS,
            str(Path(__file__).parent / "phone_grab.py"),
            f"{SCP_HOST}:{ROBOT_GRAB}",
        ],
        check=True,
    )

    while not phone_present():
        print(
            "iPhone is NOT on the robot's USB bus — check the cable "
            "(lsusb on vega-1 must show 'Apple ... iPhone'). Retrying in 5 s ..."
        )
        time.sleep(5)

    K_head, info = head_intrinsics()
    print(
        f"head: {info['model']} sn {info['serial_number']} "
        f"{info['actual']['width']}x{info['actual']['height']} rectified, "
        f"fx={K_head[0, 0]:.2f}"
    )

    from dexcontrol.robot import Robot

    def enable_head_only(self):
        """Replaces Robot._set_default_state, whose last act is to drive the head
        to its "home" pose — camera level, board on the floor out of frame.  Only
        the head enable is kept (arm mode setup is not needed here)."""
        estop = getattr(self, "estop", None)
        if estop is not None and estop.is_software_estop_enabled():
            print("software E-Stop is active — the head will not move; release it before capturing")
            return
        if head := getattr(self, "head", None):
            head.set_mode("enable")

    Robot._set_default_state = enable_head_only
    robot = Robot()

    def joints():
        return (
            np.asarray(robot.head.get_joint_pos(), float),
            np.asarray(robot.torso.get_joint_pos(), float),
        )

    head_lim = robot.head.get_joint_pos_limit()  # (3, 2) or None

    def head_goto(target, timeout=2.5, settle=0.5):
        """Move the head there gently and wait until it has come to rest."""
        target = np.asarray(target, float)
        if head_lim is not None:
            target = np.clip(target, head_lim[:, 0] + 1e-3, head_lim[:, 1] - 1e-3)
        motion = target - joints()[0]
        norm = float(np.linalg.norm(motion))
        vel = motion / norm * HEAD_SPEED if norm > 1e-6 else np.zeros(3)
        robot.head.set_joint_pos_vel(
            target,
            joint_vel=vel,
            wait_time=timeout,
            exit_on_reach=True,
            exit_on_reach_kwargs={"tolerance": 8e-3},
        )
        time.sleep(settle)
        off = np.abs(joints()[0] - target).max()
        if off > 0.02:
            print(
                f"  head stopped {off:.3f} rad short of the commanded pose — "
                "is it enabled / E-Stopped?"
            )
        return target

    def down_rad(pose):
        """How far below horizontal the camera looks at this head pose.  The head is
        level when j1 - j3 equals the torso-pitch compensation dexcontrol applies to
        its own poses, so this stays right whatever the torso is doing."""
        level = robot.compensate_torso_pitch(np.zeros(3), "head")[0]
        return float(pose[0] - pose[2] - level)

    def sweep_pose(coef):
        """Head joints for one sweep entry: pitch down, yaw, and neck split."""
        pitch, yaw, split = np.asarray(coef, float) * args.sweep
        pose = aim + np.array([pitch / 2 + split, yaw, -pitch / 2 + split])
        short = MIN_DOWN - down_rad(pose)
        if short > 0:  # keep the camera pointed below horizontal
            pose += np.array([short / 2, 0.0, -short / 2])
        return pose

    if args.keep_aim:
        aim = joints()[0]
    elif args.head_pose is not None:
        aim = np.asarray(args.head_pose, float)
    else:
        level_j1 = robot.compensate_torso_pitch(np.zeros(3), "head")[0]
        aim = np.array([level_j1 + DEFAULT_DOWN, 0.0, 0.0])
    aim = head_goto(aim, timeout=6.0)
    tilt = down_rad(joints()[0])  # what the camera actually ended up at
    deepest = max(HEAD_SWEEP, key=lambda c: c[0])
    print(
        f"head aim: {np.round(aim, 3).tolist()} "
        f"({np.degrees(tilt):.1f} deg below horizontal)"
        + (
            f", sweeping to {np.degrees(down_rad(sweep_pose(deepest))):.1f} deg down "
            f"and +-{np.degrees(args.sweep):.1f} deg of yaw"
            if args.sweep > 0
            else ", sweep off"
        )
    )
    if tilt <= 0:
        print(
            "  WARNING: this aim points at or above the horizon — the board below "
            "will not be in frame.  Raise head_j1 (or lower head_j3) to look down."
        )
    if detect_head_chessboard(cv2.cvtColor(head_frame()[0], cv2.COLOR_RGB2GRAY)) is None:
        print(
            "  WARNING: the head camera does not see the full chessboard from this "
            "aim.\n  Move the board, or re-aim with --head-pose J1 J2 J3 "
            "(+j1 and -j3 both look further down)."
        )

    session = {
        "K_head": K_head.tolist(),
        "head_info": info,
        "board": None,
        "squares_xy": SQUARES_XY,
        "square_m": SQUARE_M,
        "marker_m": MARKER_M,
        "head_aim": aim.tolist(),
        "head_sweep_rad": args.sweep,
        "urdf_head_joints": ["head_j1", "head_j2", "head_j3"],
        "urdf_torso_joints": ["torso_j1", "torso_j2", "torso_j3"],
    }

    def wait_head_static(timeout=4.0):
        """Block until two joint reads 0.3 s apart agree to ~0.02 deg."""
        prev = joints()[0]
        end = time.time() + timeout
        while time.time() < end:
            time.sleep(0.3)
            cur = joints()[0]
            if np.abs(cur - prev).max() < 3e-4:
                return
            prev = cur
        print("  (head still settling after %.0f s — proceeding)" % timeout)

    n_ok, active_head_sample, last_centroid = 0, None, None
    try:
        while n_ok < args.target:
            # Head first, so the board can be placed against the final view.  A
            # rejected capture retries this same pose without commanding the neck
            # again; only a saved sample advances the sweep.
            tag = f"s{n_ok + 1:03d}"
            if args.sweep > 0 and active_head_sample != n_ok:
                head_goto(sweep_pose(HEAD_SWEEP[n_ok % len(HEAD_SWEEP)]))
                wait_head_static()
                active_head_sample = n_ok
            if args.auto:
                time.sleep(1.0)
            else:
                if (
                    input(
                        f"\n[{n_ok}/{args.target}] head is set — move & tilt "
                        "the board, hands off, then ENTER "
                        "(q + ENTER to finish): "
                    )
                    .strip()
                    .lower()
                    == "q"
                ):
                    break
            print("  checking the stationary board ...", flush=True)
            # The board was just placed by hand (and the neck may still be
            # creeping after its move) — open the capture bracket only once two
            # consecutive head detections agree to sub-pixel (or a deadline
            # passes and the bracket check below gets its say).
            h1, hc1 = acquire_head_chessboard(timeout=8.0, require_stable=True)
            if hc1 is None:
                print("  head camera does not see the full chessboard — skip")
                continue
            if (
                args.auto
                and last_centroid is not None
                and np.linalg.norm(hc1.mean(0) - last_centroid) < 8.0
            ):
                continue  # board hasn't moved — wait
            jh1, jt1 = joints()

            print("  board detected — capturing phone ...", flush=True)
            got, err = phone_grab(args.out, tag)
            if got is None:
                print(f"  phone grab failed: {err}")
                continue
            prgb, pmeta = got
            pg = cv2.cvtColor(prgb, cv2.COLOR_RGB2GRAY)
            pids, pcorners = detect_phone_charuco(pg, session)
            if pids is None:
                print("  phone does not see the board (or too few markers) — skip")
                continue

            h2, hc2 = acquire_head_chessboard(timeout=3.0, require_stable=False)
            jh2, jt2 = joints()
            if max(np.abs(jh1 - jh2).max(), np.abs(jt1 - jt2).max()) > 2e-3:
                print("  robot joints moved during capture — skip")
                continue
            shift_px = head_board_shift_px(hc1, hc2)
            if head_board_moved(hc1, hc2):
                if hc2 is None:
                    print("  head camera lost the full chessboard after capture — skip")
                else:
                    print(f"  board/head moved {shift_px:.2f} px during capture — skip")
                continue

            d = args.out / tag
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / "head.png"), cv2.cvtColor(h1, cv2.COLOR_RGB2BGR))
            cv2.imwrite(
                str(d / "phone.jpg"),
                cv2.cvtColor(prgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 97],
            )
            (d / "meta.json").write_text(
                json.dumps(
                    {
                        "head_joints": ((jh1 + jh2) / 2).tolist(),
                        "torso_joints": ((jt1 + jt2) / 2).tolist(),
                        "phone": pmeta,
                        "head_corners_px": hc1.tolist(),
                        "phone_charuco_ids": pids.tolist(),
                        "phone_charuco_px": pcorners.tolist(),
                    },
                    indent=1,
                )
            )
            os.remove(args.out / f"{tag}_rgb.npy")
            last_centroid, n_ok = hc1.mean(0), n_ok + 1
            print(
                f"  SAVED {tag}: head 88 corners, phone {len(pids)} charuco corners, "
                f"head_j={np.round(jh1, 3).tolist()}  \a"
            )
    finally:
        if args.sweep > 0:
            try:
                head_goto(aim, settle=0.0)
            except Exception as e:
                print(f"could not return the head to its aim pose: {e}")
        (args.out / "session.json").write_text(json.dumps(session, indent=1))
        robot.shutdown()
    print(
        f"\n{n_ok} samples in {args.out}\n"
        f"solve with: python scripts/base_tracking/calibration/calib_solve.py {args.out}"
    )


if __name__ == "__main__":
    main()
