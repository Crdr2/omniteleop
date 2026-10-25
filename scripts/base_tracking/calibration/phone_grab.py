"""Capture one fresh phone RGB frame, intrinsics, and ARKit pose over USB."""

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("tag")
    args = parser.parse_args()
    from record3d import Record3DStream

    args.outdir.mkdir(parents=True, exist_ok=True)
    devices = Record3DStream.get_connected_devices()
    if not devices:
        raise SystemExit("NO DEVICE: usbmuxd sees no iPhone")
    session = Record3DStream()
    session.latest_frame_only = True
    got = threading.Event()
    output = {}

    def on_frame():
        if got.is_set():
            return
        pose = session.get_camera_pose()
        intrinsic = session.get_intrinsic_mat()
        output["rgb"] = np.asarray(session.get_rgb_frame()).copy()
        output["pose"] = [pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz]
        output["K"] = [intrinsic.fx, intrinsic.fy, intrinsic.tx, intrinsic.ty]
        got.set()

    session.on_new_frame = on_frame
    session.on_stream_stopped = got.set
    try:
        if not session.connect(devices[0]):
            raise SystemExit("Record3D could not connect to the iPhone")
        if not got.wait(15) or "rgb" not in output:
            raise SystemExit("NO FRAME in 15 s; check Record3D USB streaming")
        # Discard the first callback after connection, which can contain a queued frame.
        got.clear()
        time.sleep(0.4)
        if not got.wait(5):
            raise SystemExit("stream died after first frame")
        np.save(args.outdir / f"{args.tag}_rgb.npy", output["rgb"])
        metadata = {
            "K_fx_fy_cx_cy": output["K"],
            "arkit_pose_qxyzw_txyz": output["pose"],
            "shape": list(output["rgb"].shape),
            "host_time": time.time(),
        }
        (args.outdir / f"{args.tag}_meta.json").write_text(json.dumps(metadata))
        print(f"OK {args.tag} rgb {output['rgb'].shape}")
    finally:
        session.disconnect()


if __name__ == "__main__":
    main()
