# OmniTeleop WBC recorder

Quest 3 whole-body teleoperation for a Dexmate Vega-1  
with dual Robotiq 2F-85 grippers, one head ZED X Mini, and two wrist ZED Mini cameras.

## Pipeline

```text
Quest WebXR ──> wbc_vr_leader ── vr/joints ──> wbc_vr_robot ──> Vega actuators
                                                    │
head + 2 wrist ZED publishers ── RGB/depth + clocks ┘──> episode_N.hdf5
```

The leader publishes calibrated Cartesian hand/head targets at 10 Hz. The follower
interpolates them into a 100 Hz whole-body IK and safety loop, controls the robot, and
records at 10 Hz. Rates and safety/IK parameters live in
`src/omniteleop/follower/wbik.yaml`.

> Set up [1](#1-install-on-both-hosts) and [2](#2-configure-robot-communication) once.
> Run [3](#3-start-the-cameras)–[5](#5-record) every time.

## 1. Install on both hosts

The camera publishers normally run on the robot camera host; the VR leader and robot
follower run on a workstation. Clone this repository on both:

```bash
git clone <repository-url> omniteleop
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

For another installation, run the public leader with an externally managed certificate
and key, :

```bash
python scripts/wbc_vr_leader.py \
  --cert /secure/path/webxr-cert.pem \
  --key /secure/path/webxr-key.pem
```

Then open `https://<workstation-ip>:5067`

Alternatively, USB port forwarding makes the workstation a trusted `localhost` origin
without TLS:

```bash
adb reverse tcp:5067 tcp:5067
```

Then open `http://localhost:5067` in the Quest. 

If you have issues with **Start XR** (Quest WebXR requires HTTPS when connecting to a workstation LAN address), you can try using TLS certificate and private key.

## 5. Record

```bash
python scripts/wbc_vr_robot.py \
  --source live \
  --record \
  --streaming-recorder \
  --no-head-depth \
  --head-right-rgb \
  --save-dir data/raw
```

- Hold **right grip** for at least one second to calibrate the room, engage, and start a
take.
- Use the **index triggers** to command the Robotiq grippers.
- Press **X** to stop motion, finish/save the take, and return to the static stage.
- Hold right grip again to start the next take.
- Press **left index** while static to request homing; press **Y** to stop both programs.

The default command records head stereo instead of SDK depth. To record ZED depth, start
the head publisher with `--enable-depth` and omit `--no-head-depth`; `--head-right-rgb`
may still be kept if both representations are useful.

## Output

- `action/joint`, `action/eef`, `action/head`, `action/base`, `action/gripper`
- `obs/joint`, `obs/base`, `obs/gripper`, `obs/images`
- `meta/ntp`, `meta/camera_ntp`, and `meta/head_stereo`

`timestamp_ns` is the workstation record time.

Robot credentials, calibration, recordings, caches, and generated artifacts are ignored
by Git. See [Dexmate documentation](https://docs.dexmate.ai/),
[uv installation](https://docs.astral.sh/uv/getting-started/installation/), and the
[Stereolabs Python API guide](https://docs.stereolabs.com/docs/development/api-languages/python)
for vendor setup details.

## License

Code is AGPL-3.0-or-later. The stripped robot model carries separate Apache-2.0 and MIT
notices; see `THIRD_PARTY_NOTICES.md` and `src/omniteleop/assets/`.