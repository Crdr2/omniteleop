#!/usr/bin/env python
"""Solve the iPhone-camera -> robot-base extrinsic from calib_capture.py samples.

Per sample (all static):
    base_T_board  = FK(torso, head) * zedL_T_board      (head-side chessboard PnP)
    X_i           = phoneCam_T_board * inv(base_T_board)
X = robust average of X_i. phoneCam is the OpenCV frame of the streamed portrait
RGB image (x right, y down, z forward).

The head side uses plain chessboard corners (the ZED can't resolve the 5x5
markers at SVGA), which leaves a 180-degree ordering ambiguity plus the planar
IPPE two-fold ambiguity: all candidate poses are generated and the globally
consistent set is picked by mode clustering across samples.

Also searches the fixed rotation R_fix between ARKit's device camera frame and
phoneCam such that the derived base z-axis is vertical in the (gravity-aligned)
ARKit world; the runtime publisher needs it:

    arworld_T_base(t) = arworld_T_arkit(t) * T(R_fix^-1) * X

Usage: python scripts/base_tracking/calibration/calib_solve.py <session_dir>
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from omniteleop.follower.whole_body_ik import WBCConfig

CAD_PRIOR_M = np.array([0.151, 0.215, 0.478])  # nominal phone cam in base (+-20 mm)


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def rot_angle_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def pnp_planar(obj, img, K, max_err_px):
    """All IPPE solutions with reprojection error below the cap."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
        obj.astype(np.float64), img.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE
    )
    out = []
    for i in range(n):
        if errs[i][0] < max_err_px:
            out.append((T_from(cv2.Rodrigues(rvecs[i])[0], tvecs[i]), float(errs[i][0])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--config", default=None, help="same wbik.yaml used by the follower")
    args = parser.parse_args()
    sdir = args.session_dir
    S = json.loads((sdir / "session.json").read_text())
    if not S["board"]:
        sys.exit("session.json has no identified board (no phone detections?)")
    dname, legacy = S["board"]
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname))
    board = cv2.aruco.CharucoBoard(tuple(S["squares_xy"]), S["square_m"], S["marker_m"], adict)
    board.setLegacyPattern(legacy)
    OBJ = board.getChessboardCorners().astype(np.float64)  # (88,3) m
    # Head-side correspondence candidates: charuco corner ids run x-fastest over
    # an (sy-1) x (sx-1) grid, while findChessboardCorners walks its own pattern
    # rows in a scan direction of its choosing — transposed, reversed, and/or
    # REFLECTED relative to the id order.  A reflected correspondence still fits
    # planar PnP perfectly (it is the pose of the board seen from behind), so all
    # 8 axis-direction combinations are generated and the wrong ones are left to
    # the reprojection cap + cross-sample clustering + CAD-prior seed gate.
    # (Verified live: the true ordering here is transpose with x reversed.)
    sx, sy = S["squares_xy"]
    grid = OBJ.reshape(sy - 1, sx - 1, 3)
    OBJ_ORDERINGS = [
        g0[::fy, ::fx].reshape(-1, 3).copy()
        for g0 in (grid, grid.transpose(1, 0, 2))
        for fy in (1, -1)
        for fx in (1, -1)
    ]
    K_head = np.asarray(S["K_head"])

    model = pin.buildModelFromUrdf(WBCConfig.from_yaml(args.config).urdf_path)
    data = model.createData()
    fid = model.getFrameId("zed_left_camera")
    assert fid < model.nframes, "zed_left_camera frame not in URDF"

    def fk_base_T_zed(head_q, torso_q):
        q = pin.neutral(model)
        for names, vals in ((S["urdf_head_joints"], head_q), (S["urdf_torso_joints"], torso_q)):
            for nm, v in zip(names, vals):
                j = model.joints[model.getJointId(nm)]
                q[j.idx_q] = v
        pin.framesForwardKinematics(model, data, q)
        M = data.oMf[fid]
        return T_from(M.rotation, M.translation)

    samples = []
    for d in sorted(sdir.glob("s*")):
        if not (d / "meta.json").exists():
            continue
        m = json.loads((d / "meta.json").read_text())
        fx, fy, cx, cy = m["phone"]["K_fx_fy_cx_cy"]
        K_ph = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

        ids = np.asarray(m["phone_charuco_ids"], int)
        ph = pnp_planar(OBJ[ids], np.asarray(m["phone_charuco_px"]), K_ph, 3.0)
        if not ph:
            print(f"{d.name}: phone PnP failed — dropped")
            continue
        T_ph_board, ph_err = min(ph, key=lambda x: x[1])

        base_T_zed = fk_base_T_zed(m["head_joints"], m["torso_joints"])
        hc = np.asarray(m["head_corners_px"])
        cands = []
        for obj in OBJ_ORDERINGS:
            for T_z_b, err in pnp_planar(obj, hc, K_head, 3.0):
                base_T_board = base_T_zed @ T_z_b
                cands.append((T_ph_board @ np.linalg.inv(base_T_board), err))
        if not cands:
            print(f"{d.name}: head PnP failed — dropped")
            continue
        qx, qy, qz, qw, tx, ty, tz = m["phone"]["arkit_pose_qxyzw_txyz"]
        samples.append(
            {
                "name": d.name,
                "Xc": cands,
                "ph_err": ph_err,
                "T_arw_arkit": T_from(
                    Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), [tx, ty, tz]
                ),
            }
        )
    if len(samples) < 3:
        sys.exit(f"only {len(samples)} usable samples — capture more")

    # pick the globally consistent candidate per sample (mode clustering)
    def dist(A, B):
        return (
            rot_angle_deg(A[:3, :3].T @ B[:3, :3]) + np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000 / 4
        )  # 1deg ~ 4mm

    flat = [(i, X) for i, s in enumerate(samples) for X, _ in s["Xc"]]
    support = [sum(1 for j, Y in flat if j != i and dist(X, Y) < 15) for i, X in flat]

    # Seed from candidates near the CAD-designed mount pose when any exist —
    # mirror-correspondence clusters can be self-consistent too, but only the
    # real one puts the camera at the mount looking forward.  Falls back to the
    # global mode (and says so) if the phone is not where the CAD expects.
    def near_cad(X):
        cam = -X[:3, :3].T @ X[:3, 3]
        return np.linalg.norm(cam - CAD_PRIOR_M) < 0.25 and (X[:3, :3].T @ [0, 0, 1])[0] > 0.5

    gated = [k for k, (_, X) in enumerate(flat) if near_cad(X)]
    if gated:
        best_k = max(gated, key=lambda k: support[k])
    else:
        best_k = int(np.argmax(support))
        print(
            "WARNING: no candidate near the CAD mount pose — seeding from the "
            "global mode; check that the phone is seated in the cradle."
        )
    seed = flat[best_k][1]
    for _ in range(2):
        chosen = [min(s["Xc"], key=lambda c: dist(c[0], seed))[0] for s in samples]
        Rm = sum(X[:3, :3] for X in chosen)
        U, _, Vt = np.linalg.svd(Rm)
        R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        t = np.mean([X[:3, 3] for X in chosen], axis=0)
        seed = T_from(R, t)
    X = seed

    print(f"\n=== phoneCam_T_base from {len(samples)} samples ===")
    res_r, res_t = [], []
    for s, Xi in zip(samples, chosen):
        dr = rot_angle_deg(Xi[:3, :3].T @ X[:3, :3])
        dt = np.linalg.norm(Xi[:3, 3] - X[:3, 3]) * 1000
        res_r.append(dr)
        res_t.append(dt)
        print(
            f"  {s['name']}: residual {dr:5.2f} deg  {dt:6.1f} mm   "
            f"phone reproj {s['ph_err']:.2f} px"
        )
    print(
        f"  spread: rot {np.mean(res_r):.2f} deg mean / {np.max(res_r):.2f} max, "
        f"trans {np.mean(res_t):.1f} mm mean / {np.max(res_t):.1f} max"
    )

    cam_in_base = -X[:3, :3].T @ X[:3, 3]
    d_cad = np.linalg.norm(cam_in_base - CAD_PRIOR_M) * 1000
    print(
        f"\nphone camera in base frame: "
        f"({cam_in_base[0] * 1000:.1f}, {cam_in_base[1] * 1000:.1f}, "
        f"{cam_in_base[2] * 1000:.1f}) mm   [CAD prior (151, 215, 478) +-20; "
        f"off by {d_cad:.1f} mm]"
    )
    fwd = X[:3, :3].T @ [0, 0, 1]
    print(
        f"optical axis in base frame: ({fwd[0]:+.3f}, {fwd[1]:+.3f}, {fwd[2]:+.3f}) "
        f"[nominal (+1, 0, 0)]"
    )

    # R_fix: ARKit device-camera frame -> phoneCam (portrait OpenCV)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            Rf = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                Rf[r, c] = s
            if np.linalg.det(Rf) < 0:
                continue
            Tf = T_from(Rf.T, [0, 0, 0])  # arkit_T_phoneCam = Rf^-1
            ups = [(s["T_arw_arkit"] @ Tf @ X)[:3, 2] for s in samples]
            ang = [
                np.degrees(np.arccos(np.clip(u[1] / np.linalg.norm(u), -1, 1))) for u in ups
            ]  # ARKit +y is up
            if best is None or np.mean(ang) < best[0]:
                best = (float(np.mean(ang)), float(np.std(ang)), Rf)
    g_mean, g_std, Rf = best
    print(
        f"\nR_fix (arkit cam -> phoneCam): rows {Rf.astype(int).tolist()}   "
        f"base z vs gravity: {g_mean:.2f} deg mean, {g_std:.2f} deg std"
    )
    if g_mean > 3.0:
        print(
            "  WARNING: base axis not vertical — R_fix or X is off, "
            "or the floor/robot was not level. Inspect before trusting."
        )

    out = {
        "phoneCam_T_base": X.tolist(),
        "base_T_phoneCam": np.linalg.inv(X).tolist(),
        "cam_in_base_mm": (cam_in_base * 1000).round(2).tolist(),
        "R_fix_arkitcam_to_phonecam": Rf.tolist(),
        "gravity_check_deg": {"mean": g_mean, "std": g_std},
        "residuals": {"rot_deg": res_r, "trans_mm": res_t},
        "n_samples": len(samples),
        "board": S["board"],
        "runtime": "arworld_T_base(t) = arworld_T_arkit(t) @ T(R_fix^-1) @ phoneCam_T_base",
    }
    (sdir / "calib_result.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {sdir / 'calib_result.json'}")


if __name__ == "__main__":
    main()
