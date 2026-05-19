from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

from field_converter.utils.normalization import TorchNormalizationStats


LossType = Literal["smoothl1", "l1"]


def reconstruct_X_cam_pred(
    *,
    x3d_sam_norm: torch.Tensor,
    root_cam_norm: torch.Tensor,
    stats: TorchNormalizationStats,
) -> torch.Tensor:
    """Reconstruct predicted camera-space joints.

    Uses the convention:
        X_cam_pred = X3D_sam_rel_denorm + root_cam_pred_denorm[:,None,:]

    Parameters
    ----------
    x3d_sam_norm:
        (B,25,3) normalized SAM3D relative joints.
    root_cam_norm:
        (B,3) normalized root translation.

    Returns
    -------
    X_cam_pred:
        (B,25,3) joints in camera coordinates, meters.
    """
    X_rel = stats.denorm_sam3d_rel(x3d_sam_norm)
    root = stats.denorm_root(root_cam_norm)
    return X_rel + root.unsqueeze(-2)


def loss_root_smooth_l1(root_pred_norm: torch.Tensor, root_gt_norm: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(root_pred_norm, root_gt_norm, reduction="mean")


def _masked_reduce(loss_per_joint: torch.Tensor, valid_joints: torch.Tensor) -> torch.Tensor:
    """Reduce a (B,J) tensor with a (B,J) boolean mask."""
    mask = valid_joints.to(dtype=loss_per_joint.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (loss_per_joint * mask).sum() / denom


def loss_cam3d(
    *,
    x3d_sam_norm: torch.Tensor,
    root_pred_norm: torch.Tensor,
    Y_cam_gt: torch.Tensor,
    valid_joints: torch.Tensor,
    stats: TorchNormalizationStats,
    loss_type: LossType = "smoothl1",
) -> torch.Tensor:
    """Optional 3D camera-space loss against `Y_cam_gt`.

    - Denormalizes `x3d_sam_norm` and `root_pred_norm`.
    - Reconstructs `X_cam_pred`.
    - Compares to `Y_cam_gt` using `valid_joints`.
    """
    X_cam_pred = reconstruct_X_cam_pred(x3d_sam_norm=x3d_sam_norm, root_cam_norm=root_pred_norm, stats=stats)

    diff = X_cam_pred - Y_cam_gt

    if loss_type == "l1":
        per_coord = diff.abs()
    elif loss_type == "smoothl1":
        per_coord = F.smooth_l1_loss(X_cam_pred, Y_cam_gt, reduction="none")
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    per_joint = per_coord.mean(dim=-1)  # (B,25)
    return _masked_reduce(per_joint, valid_joints)
