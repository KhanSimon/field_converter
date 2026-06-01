from __future__ import annotations

from typing import Optional

import numpy as np


def filter_valid_mask_in_image(
    *,
    valid_mask: np.ndarray,
    valid_joints: np.ndarray,
    Y_2d_gt: np.ndarray,
    image_size: np.ndarray,
    min_in_image_joints_ratio: float,
) -> np.ndarray:
    """Keep frames where enough valid GT joints project inside the image."""

    image_size = np.asarray(image_size, dtype=np.float32).reshape(-1)
    if image_size.size != 2:
        raise ValueError(f"Expected image_size to have 2 values (W,H), got shape={image_size.shape}")
    W, H = float(image_size[0]), float(image_size[1])

    uv_finite = np.isfinite(Y_2d_gt).all(axis=-1)
    u = Y_2d_gt[..., 0]
    v = Y_2d_gt[..., 1]
    in_image = uv_finite & (u >= 0.0) & (u < W) & (v >= 0.0) & (v < H)

    in_image_valid = valid_joints & in_image
    num = in_image_valid.sum(axis=-1)
    den = np.maximum(valid_joints.sum(axis=-1), 1)
    ratio = num / den

    return valid_mask & (ratio >= float(min_in_image_joints_ratio))


def filter_valid_mask_bbox_geometry(
    *,
    valid_mask: np.ndarray,
    boxes_xyxy: np.ndarray,
    image_size: Optional[np.ndarray] = None,
    min_bbox_width_px: Optional[float] = None,
    min_bbox_height_px: Optional[float] = None,
    min_bbox_margin_px: Optional[float] = None,
) -> np.ndarray:
    """Keep frames whose bbox is large enough and sufficiently far from image edges.

    `min_bbox_margin_px` requires `image_size` and rejects boxes with any side
    closer than the margin to the image boundary.
    """

    if min_bbox_width_px is None and min_bbox_height_px is None and min_bbox_margin_px is None:
        return valid_mask

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"Expected boxes_xyxy to have shape (P,T,4), got shape={boxes.shape}")

    x1, y1, x2, y2 = [boxes[..., i] for i in range(4)]
    w = x2 - x1
    h = y2 - y1
    ok = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)

    if min_bbox_width_px is not None:
        ok &= w >= float(min_bbox_width_px)
    if min_bbox_height_px is not None:
        ok &= h >= float(min_bbox_height_px)

    if min_bbox_margin_px is not None:
        if image_size is None:
            raise ValueError("image_size is required when min_bbox_margin_px is set")
        image_size_arr = np.asarray(image_size, dtype=np.float32).reshape(-1)
        if image_size_arr.size != 2:
            raise ValueError(f"Expected image_size to have 2 values (W,H), got shape={image_size_arr.shape}")
        W, H = float(image_size_arr[0]), float(image_size_arr[1])
        margin = float(min_bbox_margin_px)
        ok &= (x1 >= margin) & (y1 >= margin) & ((W - x2) >= margin) & ((H - y2) >= margin)

    return valid_mask & ok
