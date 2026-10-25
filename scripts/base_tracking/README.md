# iPhone / ARKit base tracking

The USB-connected iPhone runs Record3D in USB Streaming mode. `base_pose_pub.py`
converts its ARKit camera pose through the measured phone-to-base extrinsic and
publishes `<ROBOT_NAME>/tracking/base_pose`. The follower subscriber is
`src/omniteleop/follower/arkit_base.py`.

Use the [current mount](../../hardware/vega_iphone_mount/README.md), then install
the [patched Record3D library](../../third_party/record3d/README.md) on the host
physically connected to the phone. Accept the phone's USB trust prompt, enable
USB Streaming in the Record3D app, and start streaming. Only one process may own
the phone stream; stop the base publisher before calibration.

## Calibrate once per installation

Start the head camera publisher and configure robot communication as in the root
README. Place a 12 x 9-square ChArUco board where both the phone and head camera
see it: 30 mm squares, 22 mm markers, `DICT_5X5` (the capture tool detects the
dictionary and legacy orientation). Keep the robot and board stationary during
each capture; move and tilt the board between samples.

Run on the workstation, with SSH access to the phone host and a Python executable
there that has Record3D installed:

```bash
python scripts/base_tracking/calibration/calib_capture.py \
  --robot-host <phone-host-ssh-alias> \
  --robot-python /absolute/path/to/arkit-env/bin/python \
  --out data/calibration/base_tracking

python scripts/base_tracking/calibration/calib_solve.py data/calibration/base_tracking
```

Capture moves the head through a small downward-looking sweep; clear the robot
workspace and keep the base, torso, and arms stationary. The helper is copied to
a temporary directory on the phone host over SSH. Use the same `--config` on the
solver as the follower if overriding the bundled model. The capture environment
needs OpenCV with `cv2.aruco.CharucoDetector`.

The solver writes `data/calibration/base_tracking/calib_result.json`, including
the rigid extrinsic, ARKit camera convention, and residual/gravity diagnostics.
Review these diagnostics and verify a known base translation/turn before using
the result for control. Copy only that JSON to a secure local path on the phone
host. It is installation-specific and should not be committed. Recalibrate after
changing phone, case, seating, or mount.

## Publish and select the tracking source

On the USB/phone host, export `ROBOT_NAME` and `ZENOH_CONFIG` from root README
section 2, activate the Record3D environment, and run from this checkout:

```bash
python scripts/base_tracking/base_pose_pub.py --calib /secure/path/calib_result.json
```

On the workstation, add `--arkit-base record` to the normal joystick follower
command to retain phone tracking alongside wheel odometry. After validating the
extrinsic and tracking, use `--arkit-base control` to close the chassis loop with
the phone pose. `off` remains the default. The leader command is unchanged.

Each engage establishes a fresh planar origin. Missing/stale tracking holds the
base; a relocalization jump latches a hold until re-engagement. ARKit can drift.
The packet's `t_ns` is the USB host callback time, not phone acquisition time;
the follower retains raw packets for offline analysis. Rising `drops` reports
USB/decoder backlog discarded by the patched receiver.
