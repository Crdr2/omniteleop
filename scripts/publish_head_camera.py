#!/usr/bin/env python3
"""Publish cropped ZED X Mini stereo frames (and optional depth) over Zenoh.

Frames are cropped and resized before encoding so camera payloads do not congest the
control link. Depth uses nearest-neighbor resizing to avoid averaging across depth
discontinuities or mixing invalid pixels with valid values.

  pyzed (ZED X Mini, GMSL port 0)
     │
     ├─ grab → LEFT view + RIGHT view (+ DEPTH only if --enable-depth)
     │
     ├─ crop [top:bottom, left:right]  then resize → (resize_h, resize_w)
     │     RGB:   INTER_AREA   (anti-aliased downscale)
     │     DEPTH: INTER_NEAREST (never average across depth discontinuities /
     │            mix invalid 0/NaN pixels with valid ones)
     │
     ├─ publisher → sensors/<id>/left_rgb   (RGBImageCodec, RGB uint8)
     ├─ publisher → sensors/<id>/right_rgb  (RGBImageCodec, RGB uint8)
     ├─ publisher → sensors/<id>/depth      (DepthImageCodec, float32 m) [opt-in]
     ├─ service   → sensors/<id>/info       (JsonDataCodec)
     └─ service   → sensors/<id>/clock      (JsonDataCodec, publisher host clock)

Topic naming matches the head_camera convention (left_rgb / right_rgb / depth)
so dexcontrol's ``ZedCameraSensor`` subscribes to the same streams.

IMPORTANT — only one process can hold the camera. Stop dexsensor's head_camera
before running this::

    dexsensor stop head_camera   # or just don't launch --sensor head_camera

This is a long-running dependency, *not* a one-shot test — start it and leave it
running for the whole recording session (no ``--duration``).

Usage::

    # production: matches recording rate, stereo RGB, default crop/resize
    python scripts/publish_head_camera.py --serial-number <HEAD_SERIAL>
    # add the SDK depth stream
    python scripts/publish_head_camera.py --serial-number <HEAD_SERIAL> --enable-depth
    # debug round-trip fps / inspect cropped frames
    python scripts/publish_head_camera.py --serial-number <HEAD_SERIAL> --verify
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import time
from typing import Literal, Optional

import cv2
import numpy as np
import pyzed.sl as sl
import tyro
from dexcomm import Node
from dexcomm.codecs import DepthImageCodec, JsonDataCodec, RGBImageCodec
from loguru import logger

from omniteleop.common.head_camera import (
    HEAD_CROP_TBLR,
    HEAD_RESIZE_HW,
    crop_resize_intrinsics,
)

Resolution = Literal["SVGA", "HD1080", "HD1200"]
DepthMode = Literal["NEURAL", "NEURAL_LIGHT", "ULTRA", "QUALITY", "PERFORMANCE"]

# Camera payloads must yield to VR/actuator traffic if any link is congested.
# Best-effort + drop preserves the existing latest-frame semantics; data_low
# prevents large JPEG/PNG fragments from sharing control's interactive priority.
_CAMERA_QOS = {
    "reliability": "best_effort",
    "congestion_control": "drop",
    "priority": "data_low",
}


@dataclasses.dataclass
class Args:
    """CLI options for the standalone head ZED-X-Mini publisher."""

    serial_number: int = 0
    """Physical ZED serial. Required: selecting by index can silently open a wrist
    camera when USB enumeration changes."""

    sensor_id: str = "head_camera"
    """Sensor id used to derive topic names: sensors/<sensor_id>/{left_rgb,
    right_rgb,depth,info}. Must match what the follower subscribes to
    — keep it 'head_camera'."""

    namespace: str = ""
    """Zenoh namespace passed to dexcomm.Node."""

    resolution: Resolution = "SVGA"
    """ZED capture resolution. SVGA (960x600) is the SDK floor for ZED X Mini;
    the crop coordinates below assume SVGA."""

    rate: int = 15
    """Camera fps. The ZED X Mini at SVGA only supports a discrete set
    (15/30/60/120); any other value silently rounds DOWN to the nearest
    supported rate (e.g. 20 -> 15). Default 15 matches the downstream record
    cadence (the recorder defaults to 10 Hz) and minimises head-camera
    bandwidth. Bump to 30 only if grab latency, not bandwidth, is the limiter."""

    depth_mode: DepthMode = "NEURAL"

    enable_depth: bool = False
    """Compute + publish depth. Off by default — the SDK depth pass is the
    dominant grab-loop cost and the frames are the bandwidth hog, and stereo
    consumers reconstruct depth themselves from left+right. Turn on for
    consumers that want the SDK's depth directly."""

    skip_right_rgb: bool = False
    """Skip the right_rgb stream. Off by default — the pair is published so
    downstream stereo (FoundationStereo, via the meta/head_stereo fx+baseline)
    has both views. Turn on to halve RGB bandwidth when only left is consumed."""

    depth_min: float = 0.1
    """Minimum depth in metres (matches the dexsensor head_camera config)."""

    depth_max: float = 8.0
    """Maximum depth in metres (matches the dexsensor head_camera config)."""

    crop: tuple[int, int, int, int] = HEAD_CROP_TBLR
    """Crop applied to every stream as (top, bottom, left, right) pixel indices on
    the raw SVGA frame, i.e. img[top:bottom, left:right]. Default = the shared
    omniteleop.common.head_camera.HEAD_CROP_TBLR, the shared crop geometry used by
    publishers and recorders. Override on the CLI for a one-off capture."""

    resize_h: int = HEAD_RESIZE_HW[0]
    """Output height after the crop (default = head_camera.HEAD_RESIZE_HW[0]). Set
    to 0 (with resize_w=0) to skip resize."""

    resize_w: int = HEAD_RESIZE_HW[1]
    """Output width after the crop (default = head_camera.HEAD_RESIZE_HW[1]). Set
    to 0 (with resize_h=0) to skip resize."""

    verify: bool = False
    """Subscribe to our own topics each frame to print round-trip fps. Off by
    default — decoding every frame in-process throttles the publisher."""

    duration: float = 0.0
    """If > 0, stop after this many seconds. Leave at 0 for normal use."""

    save_dir: Optional[str] = None
    """If set (implies --verify), save the LAST decoded frame received by the
    subscriber once per second, proving the round-trip preserves data."""


def _configure_zenoh_like_dexcontrol() -> Optional[pathlib.Path]:
    """Mirror dexcontrol's default ZENOH_CONFIG discovery (see wrist publisher)."""
    existing = os.getenv("ZENOH_CONFIG")
    if existing:
        return pathlib.Path(existing).expanduser()

    base_dir = pathlib.Path("~/.dexmate/comm/zenoh").expanduser()
    for pattern in ("**/*.dzcfg", "**/zenoh*config*.json5"):
        matches = sorted(p for p in base_dir.glob(pattern) if p.is_file())
        if matches:
            os.environ["ZENOH_CONFIG"] = matches[0].as_posix()
            return matches[0]
    return None


def _validate_crop(crop: tuple[int, int, int, int], width: int, height: int) -> None:
    """Aggressively validate the crop window against the actual frame size."""
    top, bottom, left, right = crop
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ValueError(
            f"crop (top,bottom,left,right)={crop} is invalid for a {width}x{height} "
            f"(WxH) frame: need 0<=top<bottom<={height} and 0<=left<right<={width}."
        )


def _crop_resize(
    img: np.ndarray,
    crop: tuple[int, int, int, int],
    out_hw: Optional[tuple[int, int]],
    interpolation: int,
) -> np.ndarray:
    """Crop img[top:bottom, left:right], then resize to (h, w) if out_hw is set.

    out_hw is (height, width); cv2.resize wants (width, height). Works for both
    HxWx3 RGB and HxW float32 depth.
    """
    top, bottom, left, right = crop
    cropped = img[top:bottom, left:right]
    if out_hw is None:
        return np.ascontiguousarray(cropped)
    h, w = out_hw
    return cv2.resize(cropped, (w, h), interpolation=interpolation)


def _depth_to_color(depth_m: np.ndarray) -> np.ndarray:
    finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if len(finite) == 0:
        gray = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    else:
        mn, mx = finite.min(), np.percentile(finite, 95)
        gray = np.clip((depth_m - mn) / (mx - mn + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def _zed_image_timestamp_ns(zed: sl.Camera) -> int:
    """SDK IMAGE timestamp for the latest successful grab(), in UNIX ns."""
    timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
    ns = int(timestamp.get_nanoseconds())
    if ns <= 0:
        raise RuntimeError("ZED SDK returned no IMAGE timestamp after grab()")
    return ns


def _clock_response(sensor_id: str, sequence: int) -> dict:
    """NTP-style clock service response from this publisher host."""
    t1 = time.time_ns()
    return {
        "server_receive_time_ns": int(t1),
        "server_send_time_ns": int(time.time_ns()),
        "source": "zed_publisher_host",
        "sensor_id": str(sensor_id),
        "sequence": int(sequence),
    }


def main() -> None:
    args = tyro.cli(Args)
    if args.serial_number <= 0:
        raise ValueError("--serial-number must be the positive serial printed by the ZED SDK")
    verify = args.verify or args.save_dir is not None
    zenoh_config = _configure_zenoh_like_dexcontrol()
    if (args.resize_h > 0) != (args.resize_w > 0):
        raise ValueError(
            f"resize_h ({args.resize_h}) and resize_w ({args.resize_w}) must both "
            "be >0 (resize) or both 0 (skip resize)."
        )
    out_hw: Optional[tuple[int, int]] = (
        (args.resize_h, args.resize_w) if args.resize_h > 0 else None
    )

    # ── 1. Open ZED X Mini via SDK on the requested GMSL port ─────────────────
    depth_mode = (
        getattr(sl.DEPTH_MODE, args.depth_mode) if args.enable_depth else sl.DEPTH_MODE.NONE
    )
    init = sl.InitParameters(
        camera_resolution=getattr(sl.RESOLUTION, args.resolution),
        camera_fps=args.rate,
        depth_mode=depth_mode,
        coordinate_system=sl.COORDINATE_SYSTEM.IMAGE,
        coordinate_units=sl.UNIT.METER,
        depth_minimum_distance=args.depth_min,
        depth_maximum_distance=args.depth_max,
    )
    init.set_from_serial_number(args.serial_number)
    # Pin factory calibration so recorded intrinsics remain reproducible across opens.
    init.camera_disable_self_calib = True
    runtime = sl.RuntimeParameters()
    zed = sl.Camera()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(
            f"ZED open failed: {err}. Is dexsensor still holding the head_camera "
            f"(serial {args.serial_number})? Stop it first."
        )

    info = zed.get_camera_information()
    logger.info(
        f"Camera: {info.camera_model} sn={info.serial_number} "
        f"fw={info.camera_configuration.firmware_version}"
    )

    cam_res = info.camera_configuration.resolution
    raw_w, raw_h = cam_res.width, cam_res.height
    logger.info(f"Raw capture resolution: {raw_w}x{raw_h}")
    _validate_crop(args.crop, raw_w, raw_h)

    top, bottom, left, right = args.crop
    crop_h, crop_w = bottom - top, right - left
    out_h, out_w = out_hw if out_hw is not None else (crop_h, crop_w)
    logger.info(
        f"Pipeline: raw {raw_w}x{raw_h} → crop [{top}:{bottom},{left}:{right}] "
        f"({crop_w}x{crop_h}) → resize {out_w}x{out_h} (WxH)"
    )

    # ── 1c. Stereo calibration for the PUBLISHED frames ───────────────────────
    # FoundationStereo needs (left, right, fx, baseline) to turn disparity into
    # metric depth. `calibration_parameters` (not `_raw`) is the RECTIFIED pair,
    # which is what VIEW.LEFT/VIEW.RIGHT return, and left/right go through the
    # SAME _crop_resize below -- so the rows stay epipolar-aligned and only the
    # intrinsics need adjusting. The baseline is a physical distance and does
    # not scale with crop/resize.
    _calib = info.camera_configuration.calibration_parameters
    _baseline_m = float(_calib.get_camera_baseline())
    if not 0.01 < _baseline_m < 0.5:
        raise SystemExit(
            f"implausible ZED stereo baseline {_baseline_m} m; coordinate_units is "
            f"METER, so this should be the physical inter-lens distance"
        )

    def _sdk_k(cam) -> np.ndarray:
        return np.array(
            [[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    _stereo = {
        "baseline_m": _baseline_m,
        "rectified": True,
        "left_K": crop_resize_intrinsics(
            _sdk_k(_calib.left_cam), args.crop, (out_h, out_w)
        ).tolist(),
        "right_K": crop_resize_intrinsics(
            _sdk_k(_calib.right_cam), args.crop, (out_h, out_w)
        ).tolist(),
    }
    logger.info(
        f"Stereo calibration: baseline {_baseline_m * 1000:.2f} mm, "
        f"left fx {_stereo['left_K'][0][0]:.2f} px (published geometry)"
    )

    # ── 2. Set up Zenoh pub/sub via dexcomm ───────────────────────────────────
    left_topic = f"sensors/{args.sensor_id}/left_rgb"
    right_topic = f"sensors/{args.sensor_id}/right_rgb"
    depth_topic = f"sensors/{args.sensor_id}/depth"
    info_topic = f"sensors/{args.sensor_id}/info"
    clock_topic = f"sensors/{args.sensor_id}/clock"

    node = Node(name="head_zedx_pub", namespace=args.namespace)
    logger.info(
        f"Zenoh config: {zenoh_config if zenoh_config is not None else '<dexcomm default>'}"
    )
    logger.info(f"ROBOT_NAME={os.getenv('ROBOT_NAME')!r}; node.namespace={node.namespace!r}")
    left_pub = node.create_publisher(left_topic, encoder=RGBImageCodec.encode, qos=_CAMERA_QOS)
    right_pub = (
        node.create_publisher(right_topic, encoder=RGBImageCodec.encode, qos=_CAMERA_QOS)
        if not args.skip_right_rgb
        else None
    )
    depth_pub = (
        node.create_publisher(depth_topic, encoder=DepthImageCodec.encode, qos=_CAMERA_QOS)
        if args.enable_depth
        else None
    )
    left_sub = right_sub = depth_sub = None
    if verify:
        left_sub = node.create_subscriber(left_topic, decoder=RGBImageCodec.decode)
        if right_pub is not None:
            right_sub = node.create_subscriber(right_topic, decoder=RGBImageCodec.decode)
        if args.enable_depth:
            depth_sub = node.create_subscriber(depth_topic, decoder=DepthImageCodec.decode)

    frame_stats = {
        "left_rgb": {"published": 0, "last_timestamp_ns": 0},
        "right_rgb": {"published": 0, "last_timestamp_ns": 0},
        "depth": {"published": 0, "last_timestamp_ns": 0},
    }
    sequence_state = {"latest": 0}
    info_stats = {"queries": 0, "last_log_s": 0.0}

    def _camera_info(_request: bytes | None = None) -> dict:
        info_stats["queries"] += 1
        now_s = time.perf_counter()
        if info_stats["queries"] == 1 or now_s - info_stats["last_log_s"] >= 10.0:
            logger.info(
                f"Camera info queried on {node.resolve_topic(info_topic)!r} "
                f"(count={info_stats['queries']})"
            )
            info_stats["last_log_s"] = now_s
        return {
            "type": "ZED_CAMERA",
            "camera_id": str(info.serial_number),
            "status": "running",
            "model": str(info.camera_model),
            "serial_number": int(info.serial_number),
            "firmware_version": str(info.camera_configuration.firmware_version),
            "actual": {"width": int(out_w), "height": int(out_h), "fps": int(args.rate)},
            # Consumed by wbc_vr_robot.py --head-right-rgb so a stereo take
            # carries everything FoundationStereo needs (fx, baseline) without a
            # hardcoded constant anywhere downstream.
            "stereo": _stereo,
            "configured": {
                "resolution": args.resolution,
                "depth_mode": args.depth_mode,
                "depth_min": float(args.depth_min),
                "depth_max": float(args.depth_max),
                "crop": list(args.crop),
                "resize_hw": [int(out_h), int(out_w)],
            },
            "streams": {
                "left_rgb": {"enabled": True, "transport": "zenoh", "topic": left_topic},
                "right_rgb": {
                    "enabled": not args.skip_right_rgb,
                    "transport": "zenoh",
                    "topic": right_topic,
                },
                "depth": {"enabled": args.enable_depth, "transport": "zenoh", "topic": depth_topic},
            },
            "statistics": frame_stats,
        }

    def _clock(_request: bytes | None = None) -> dict:
        return _clock_response(args.sensor_id, sequence_state["latest"])

    info_service = node.create_service(
        info_topic, _camera_info, response_encoder=JsonDataCodec.encode
    )
    clock_service = node.create_service(clock_topic, _clock, response_encoder=JsonDataCodec.encode)
    pub_topics = f"'{left_topic}'"
    if right_pub is not None:
        pub_topics += f", '{right_topic}'"
    if args.enable_depth:
        pub_topics += f", '{depth_topic}'"
    logger.info(f"Publishing on {pub_topics}  (depth={'on' if args.enable_depth else 'off'})")
    logger.info(f"Serving camera info on '{info_topic}'")
    logger.info(f"Serving publisher clock on '{clock_topic}'")
    resolved = f"left={node.resolve_topic(left_topic)!r}"
    if right_pub is not None:
        resolved += f", right={node.resolve_topic(right_topic)!r}"
    if args.enable_depth:
        resolved += f", depth={node.resolve_topic(depth_topic)!r}"
    resolved += f", info={node.resolve_topic(info_topic)!r}"
    logger.info(f"Resolved head keys: {resolved}")
    if verify:
        logger.info("Subscribing to the same topics to verify round-trip")

    save_root: Optional[pathlib.Path] = None
    if args.save_dir:
        save_root = pathlib.Path(args.save_dir) / args.sensor_id
        save_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving 1 frame/s (post-subscriber decode) to {save_root}")

    # ── 3. Capture → crop+resize → publish → (optional) subscribe loop ────────
    left_mat = sl.Mat()
    right_mat = sl.Mat()
    depth_mat = sl.Mat()

    seq = 0
    pub_count = 0
    sub_left_count = sub_right_count = sub_depth_count = 0
    t_window = time.perf_counter()
    t_start = t_window
    t_last_save = 0.0
    first_logged = False

    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            # SDK timestamp of the grabbed image. This rides the codec-preserved
            # timestamp_ns field; extra payload keys are dropped by RGBImageCodec.
            ts = _zed_image_timestamp_ns(zed)

            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            # ZED returns BGRA; RGBImageCodec expects RGB.
            left_full = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2RGB)
            left_rgb = _crop_resize(left_full, args.crop, out_hw, cv2.INTER_AREA)

            seq += 1
            # Publish RGB immediately. NEURAL depth retrieval can take
            # most of a camera frame; putting RGB behind it makes the workstation select
            # an older head image than the wrist image at record ticks.
            left_pub.publish(
                {
                    "data": left_rgb,
                    "timestamp_ns": ts,
                    "sequence": seq,
                    "width": out_w,
                    "height": out_h,
                }
            )
            sequence_state["latest"] = seq
            frame_stats["left_rgb"]["published"] = seq
            frame_stats["left_rgb"]["last_timestamp_ns"] = ts

            right_rgb = None
            if right_pub is not None:
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
                right_full = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2RGB)
                right_rgb = _crop_resize(right_full, args.crop, out_hw, cv2.INTER_AREA)
                right_pub.publish(
                    {
                        "data": right_rgb,
                        "timestamp_ns": ts,
                        "sequence": seq,
                        "width": out_w,
                        "height": out_h,
                    }
                )

            depth_safe = None
            if args.enable_depth:
                zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                depth_full = np.ascontiguousarray(depth_mat.get_data())  # float32 m, may hold NaN
                # NEAREST so we never blend valid metres with NaN/Inf invalid pixels.
                depth_crop = _crop_resize(depth_full, args.crop, out_hw, cv2.INTER_NEAREST)
                depth_safe = np.nan_to_num(depth_crop, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32
                )

            if depth_pub is not None:
                depth_pub.publish(
                    {
                        "depth_values": depth_safe,
                        "timestamp_ns": ts,
                        "sequence": seq,
                        "width": out_w,
                        "height": out_h,
                    }
                )
            if right_pub is not None:
                frame_stats["right_rgb"]["published"] = seq
                frame_stats["right_rgb"]["last_timestamp_ns"] = ts
            if depth_pub is not None:
                frame_stats["depth"]["published"] = seq
                frame_stats["depth"]["last_timestamp_ns"] = ts
            pub_count += 1

            now = time.perf_counter()

            left_msg = right_msg = depth_msg = None
            if verify:
                left_msg = left_sub.get_latest()
                right_msg = right_sub.get_latest() if right_sub is not None else None
                depth_msg = depth_sub.get_latest() if depth_sub is not None else None

                rgb_ready = left_msg is not None and (right_sub is None or right_msg is not None)
                depth_ready = depth_sub is None or depth_msg is not None
                if rgb_ready and depth_ready and not first_logged:
                    msg = f"First decoded frame — left {left_msg['data'].shape} {left_msg['data'].dtype}"
                    assert left_msg["data"].shape == left_rgb.shape
                    if right_msg is not None:
                        msg += f", right {right_msg['data'].shape}"
                        assert right_msg["data"].shape == right_rgb.shape
                    if depth_msg is not None:
                        msg += (
                            f", depth {depth_msg['data'].shape} {depth_msg['data'].dtype}, "
                            f"finite={np.isfinite(depth_msg['data']).mean() * 100:.1f}%"
                        )
                        assert depth_msg["data"].shape == depth_safe.shape
                        assert depth_msg["data"].dtype == np.float32
                    logger.info(msg)
                    logger.info("Round-trip shape/dtype check: OK")
                    first_logged = True

                if left_msg is not None:
                    sub_left_count += 1
                if right_msg is not None:
                    sub_right_count += 1
                if depth_msg is not None:
                    sub_depth_count += 1

            if now - t_window >= 1.0:
                dt = now - t_window
                line = f"pub: {pub_count / dt:5.1f} fps"
                if verify:
                    line += (
                        f"  | sub L/R/D: {sub_left_count / dt:5.1f} / "
                        f"{sub_right_count / dt:5.1f} / {sub_depth_count / dt:5.1f} fps"
                    )
                logger.info(line)
                pub_count = sub_left_count = sub_right_count = sub_depth_count = 0
                t_window = now

            if save_root is not None and now - t_last_save >= 1.0 and left_msg is not None:
                idx = int(now - t_start)
                cv2.imwrite(
                    str(save_root / f"left_rgb_{idx:04d}.jpg"),
                    cv2.cvtColor(left_msg["data"], cv2.COLOR_RGB2BGR),
                )
                if right_msg is not None:
                    cv2.imwrite(
                        str(save_root / f"right_rgb_{idx:04d}.jpg"),
                        cv2.cvtColor(right_msg["data"], cv2.COLOR_RGB2BGR),
                    )
                if depth_msg is not None:
                    depth_mm = np.clip(depth_msg["data"] * 1000.0, 0, 65535).astype(np.uint16)
                    cv2.imwrite(str(save_root / f"depth_{idx:04d}.png"), depth_mm)
                    cv2.imwrite(
                        str(save_root / f"depth_vis_{idx:04d}.jpg"),
                        _depth_to_color(depth_msg["data"]),
                    )
                t_last_save = now

            if args.duration > 0 and now - t_start >= args.duration:
                break

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        zed.close()
        clock_service.shutdown()
        info_service.shutdown()
        node.shutdown()


if __name__ == "__main__":
    main()
