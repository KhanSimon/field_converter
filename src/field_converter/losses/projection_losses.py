from __future__ import annotations

import torch
import torch.nn.functional as F

from field_converter.geometry.projection import project_cam_to_image
from field_converter.losses.root_losses import LossType, reconstruct_X_cam_pred
from field_converter.utils.normalization import TorchNormalizationStats


def loss_reprojection(
    *,
    x3d_sam_norm: torch.Tensor,
    root_pred_norm: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    Y_2d_gt: torch.Tensor,
    valid_joints: torch.Tensor,
    stats: TorchNormalizationStats,
    loss_type: LossType = "smoothl1",
) -> torch.Tensor:
    """Optional reprojection loss in pixel space (uses `valid_joints`)."""
    X_cam_pred = reconstruct_X_cam_pred(x3d_sam_norm=x3d_sam_norm, root_cam_norm=root_pred_norm, stats=stats)
    uv_pred = project_cam_to_image(X_cam_pred, K=K, k=k)

    if loss_type == "l1":
        per_coord = (uv_pred - Y_2d_gt).abs()
    elif loss_type == "smoothl1":
        per_coord = F.smooth_l1_loss(uv_pred, Y_2d_gt, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    per_joint = per_coord.mean(dim=-1)  # (B,25)
    mask = valid_joints.to(dtype=per_joint.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (per_joint * mask).sum() / denom
