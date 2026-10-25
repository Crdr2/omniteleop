# OmniTeleop

Quest 3 whole-body teleoperation for a Dexmate Vega-1  
with dual Robotiq 2F-85 grippers, one head ZED X Mini, and two wrist ZED Mini cameras.

## Pipeline

```text
Quest WebXR ──> wbc_joystick_leader ──> wbc_joystick_robot ──> Vega actuators
                                              │
head + 2 wrist ZED publishers ── RGB + clocks ──┴──> episode_N.hdf5 + sidecars
```

The joystick leader publishes calibrated hand targets and chassis intent at 10 Hz.
The follower runs arm IK and safety checks at 100 Hz and records at 10 Hz. Right stick
drives; left stick turns. The torso and head stay at nominal by default. Rates, speeds,
head tracking, and safety settings live in `src/omniteleop/follower/wbik.yaml`.
The original `wbc_vr_leader.py` / `wbc_vr_robot.py` pair remains available for
whole-body IK that derives base motion from Cartesian targets.

> Set up [1](#1-install-on-both-hosts) and [2](#2-configure-robot-communication) once.
> Run [3](#3-start-the-cameras)–[5](#5-record) every time.

## 1. Install on both hosts

The camera publishers normally run on the robot camera host; the VR leader and robot
follower run on a workstation. Clone this repository on both:

```bash
git clone https://github.com/Crdr2/omniteleop.git
cd omniteleop
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --locked
source .venv/bin/activate
```

The lockfile covers the Python stack except PyZED. On the camera host, install the
[ZED SDK and PyZED](https://docs.stereolabs.com/docs/development/api-languages/python)
into this environment:

```bash
source .venv/bin/activate
cd /usr/local/zed
python get_python_api.py
cd -
python -c "import pyzed.sl; print('PyZED ready')"
```

Run `uv sync --locked` before installing PyZED when recreating the environment.

## 2. Configure robot communication

Obtain the robot's `.dzcfg` certificate through Dexmate's normal secure channel. On each
host, unpack it once and set the same environment in every runtime terminal:

```bash
dextop cert unpack /secure/path/<profile>.dzcfg

export ROBOT_NAME='dm/<robot-id>'
export ROBOT_CONFIG='vega_1_gripper'
export ZENOH_CONFIG="$HOME/.dexmate/comm/zenoh/<profile>/zenoh_peer_config.json5"

dextop doctor check
```

## 3. Start the cameras

```bash
# I usually run them in tmux session on robot side

dextop node start

# Head: rectified stereo RGB, 15 Hz, no SDK depth.
python scripts/publish_head_camera.py --serial-number <head-serial>

# Wrists: left eye RGB, 320x240, 15 Hz.
python scripts/publish_wrist_camera.py \
  --sensor-id left_wrist_zedm --serial-number <left-wrist-serial>

python scripts/publish_wrist_camera.py \
  --sensor-id right_wrist_zedm --serial-number <right-wrist-serial>

# you can check serial number by
# python - <<'PY'
# import pyzed.sl as sl
# for camera in sl.Camera.get_device_list():
#   print(camera.serial_number, camera.camera_model)
# PY
```

## 4. Connect the Quest

Run the joystick leader with a certificate valid for the workstation address and a
private key stored outside the repository:

```bash
python scripts/wbc_joystick_leader.py \
  --cert /secure/path/webxr-cert.pem \
  --key /secure/path/webxr-key.pem
```

In the Quest browser, open `https://<workstation-ip>:5067` and select **Start XR**.
Use the workstation address reachable from the headset's Wi-Fi. LAN HTTP addresses
do not provide the secure context WebXR needs.

Alternatively, USB port forwarding makes the workstation a trusted `localhost` origin
without TLS:

```bash
adb reverse tcp:5067 tcp:5067
```

For USB, start the leader without `--cert` / `--key` and open `http://localhost:5067`
in the Quest. The headset must be USB-connected with ADB authorized.

Before first use, add `--calibrate-ee-offset --no-headset-hud` to the leader command.
Follow the neutral-grip prompt and hold right grip to save the local orientation
calibration in ignored `config/ee_offset.yaml`. Restart the leader normally afterward.

## 5. Record

```bash
python scripts/wbc_joystick_robot.py \
  --source live \
  --record \
  --streaming-recorder \
  --no-head-depth \
  --head-right-rgb \
  --save-dir data/raw
```

- While idle, press **left index** to home before each take. Startup homing alone does
  not arm a live joystick session.
- Hold **right grip** for at least one second to calibrate the room, engage, and start
  a take.
- Use the **right stick** for forward/backward or strafe and **left stick X** to turn.
  Past the deadzone, direction selects a fixed speed: 0.18 m/s translation and
  0.30 rad/s rotation by default. The dominant axis wins; releasing stops the command.
- Use the **index triggers** to command the Robotiq grippers.
- Press **X** to stop motion, finish/save the take, and return to the static stage.
- Home again before engaging the next take; press **Y** to stop both programs.

Optional modes:

- Add `--torso-stick` to both programs for left-stick Y torso height control, or
  `--torso-ik` to both for IK-assisted torso reaching. Modes cannot be combined;
  mismatched leader/follower modes hold motion. Recordings retain distinct schemas.
- Add `--turn-90` to the follower for one measured 90-degree turn per left-stick
  gesture. Return to neutral to re-arm; translation or opposite yaw cancels.
- Gripper defaults are `--gripper-speed 0.3 --gripper-force 0.0` (minimum force), with
  closing capped at register 215/255. Closing contact or stale feedback latches a stop;
  release the corresponding trigger to re-arm. Use per-side `--*-gripper-max-position`
  to adjust the closing limit. Fixed-position overrides are also available; they can
  command grippers while idle.

The default command records head stereo instead of SDK depth. To record ZED depth, start
the head publisher with `--enable-depth` and omit `--no-head-depth`; `--head-right-rgb`
may still be kept if both representations are useful.
For wrist stereo, start each wrist publisher with `--no-skip-right-rgb` and add
`--wrist-right-rgb` to the follower. Intrinsics come from the live publishers.

Recording waits for advancing, aligned camera frames before frame zero and rejects
rows above the configured capture-age/skew bounds (150 ms / 40 ms by default).
The joystick follower retries camera stalls for up to one second; no rows are written
during a stall, so downstream processing must handle gaps.

## Output

- `action/joint`, `action/eef`, `action/head`, `action/base`, `action/gripper`
- Joystick takes also store `action/chassis/intent_body` and the selected torso schema.
- `obs/joint`, `obs/base`, `obs/gripper`, `obs/images`
- `meta/ntp`, `meta/camera_ntp`, and `meta/head_stereo`
- Source/config snapshots, gripper settings, and native state/command sidecars accompany
  recordings. Keep the sidecars with their episode; incomplete writes retain `.partial`.

`timestamp_ns` is the workstation record time.

Optional iPhone base tracking includes the [mount CAD and printable models](hardware/vega_iphone_mount/README.md),
[publisher/calibration tools](scripts/base_tracking/README.md), and the
[Record3D USB latency patch](third_party/record3d/README.md). Start the publisher on the
USB-connected phone host, then add `--arkit-base record` to save tracking alongside wheel
odometry, or `--arkit-base control` after validating the extrinsic to use it for chassis
feedback. Calibration is specific to the installation and stays outside Git.

The optional pre-episode stereo sweep (`--scenediff-sweep`) requires a local
settled parking-reference HDF5 passed via `--scenediff-park-reference`. Neither is needed
for the basic pipeline above; task data and policy training are not bundled.

The bundled model includes wrist-camera frames, gripper collision proxies, and a
conservative phone-mount envelope. It has no render meshes. Gripper stroke calibration
and mount geometry describe the current hardware setup; review them for other mounts
or fingertips.

Robot credentials, calibration, recordings, caches, and generated artifacts are ignored
by Git. See [Dexmate documentation](https://docs.dexmate.ai/),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/), and the
[Stereolabs Python API guide](https://docs.stereolabs.com/docs/development/api-languages/python)
for vendor setup details.

## License

Code is AGPL-3.0-or-later. The stripped robot model carries separate Apache-2.0 and MIT
notices; see `THIRD_PARTY_NOTICES.md` and `src/omniteleop/assets/`.
