"""Head-camera crop/resize geometry shared with the ZED publisher."""

from __future__ import annotations

import numpy as np

# Raw ZED X Mini SVGA is 960x600. Keep the default full-frame so each recording
# preserves the complete view; reduce these values only after validating bandwidth.
HEAD_CROP_TBLR: tuple[int, int, int, int] = (0, 600, 0, 960)
HEAD_RESIZE_HW: tuple[int, int] = (600, 960)


def crop_resize_intrinsics(
    intrinsic: np.ndarray,
    crop_tblr: tuple[int, int, int, int],
    out_hw: tuple[int, int],
) -> np.ndarray:
    """Transform pinhole intrinsics after crop then resize."""
    top, bottom, left, right = crop_tblr
    out_h, out_w = out_hw
    if bottom <= top or right <= left or out_h <= 0 or out_w <= 0:
        raise ValueError("crop and output dimensions must be positive")
    sx = out_w / (right - left)
    sy = out_h / (bottom - top)
    adjusted = np.asarray(intrinsic, dtype=float).copy()
    if adjusted.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape (3, 3), got {adjusted.shape}")
    adjusted[0, 2] -= left
    adjusted[1, 2] -= top
    adjusted[0, (0, 2)] *= sx
    adjusted[1, (1, 2)] *= sy
    return adjusted
