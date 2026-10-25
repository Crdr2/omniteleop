"""Explicit capture/output geometry derived from the publisher's recorded mode."""

import numpy as np

# ZED per-eye native dimensions. These are not the resized publication dimensions.
ZED_CAPTURE_WH = {
    "HD2K": (2208, 1242),
    "HD1080": (1920, 1080),
    "HD1200": (1920, 1200),
    "HD720": (1280, 720),
    "SVGA": (960, 600),
    "VGA": (672, 376),
}


def camera_resolution_metadata(configured: dict, actual: dict) -> dict:
    """Expose the saved mode without confusing native capture with output resizing.

    Unknown modes retain their label but omit native dimensions rather than guess.
    The source field distinguishes mode-derived dimensions from SDK measurements.
    """
    mode = str(configured.get("resolution", "unknown"))
    capture = {
        "resolution": np.asarray(mode.encode()),
        "source": np.asarray(b"configured_resolution"),
    }
    if mode in ZED_CAPTURE_WH:
        width, height = ZED_CAPTURE_WH[mode]
        capture.update(width=np.int64(width), height=np.int64(height))
    return {
        "capture": capture,
        "output": {"width": np.int64(actual["width"]), "height": np.int64(actual["height"])},
    }
