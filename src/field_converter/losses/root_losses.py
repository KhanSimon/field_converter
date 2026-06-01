from __future__ import annotations

from typing import Optional, Sequence, Literal

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


def _axis_weights_tensor(
    axis_weights: Optional[Sequence[float]],
    *,
    ref: torch.Tensor,
) -> torch.Tensor:
    if axis_weights is None:
        return ref.new_ones((3,))
    if len(axis_weights) != 3:
        raise ValueError(f"root axis weights must have length 3, got {len(axis_weights)}")
    w = ref.new_tensor([float(v) for v in axis_weights])
    if bool((w < 0).any()):
        raise ValueError("root axis weights must be >= 0")
    if float(w.sum().item()) <= 0.0:
        raise ValueError("at least one root axis weight must be > 0")
    return w


def loss_root_smooth_l1_axis(
    root_pred_norm: torch.Tensor,
    root_gt_norm: torch.Tensor,
) -> torch.Tensor:
    """Return unweighted SmoothL1 mean per root axis (x,y,z)."""
    if root_pred_norm.shape != root_gt_norm.shape:
        raise ValueError(f"root_pred_norm and root_gt_norm must match, got {root_pred_norm.shape} vs {root_gt_norm.shape}")
    if root_pred_norm.shape[-1] != 3:
        raise ValueError(f"Expected last dimension of root tensors to be 3, got {root_pred_norm.shape}")

    finite = torch.isfinite(root_pred_norm).all(dim=-1) & torch.isfinite(root_gt_norm).all(dim=-1)
    if not bool(finite.any()):
        return root_pred_norm.new_zeros((3,))
    per_coord = F.smooth_l1_loss(root_pred_norm[finite], root_gt_norm[finite], reduction="none")
    return per_coord.mean(dim=0)


def weighted_axis_loss(axis_loss: torch.Tensor, axis_weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    weights = _axis_weights_tensor(axis_weights, ref=axis_loss)
    return (axis_loss * weights).sum() / weights.sum().clamp(min=torch.finfo(axis_loss.dtype).eps)


def loss_root_smooth_l1(
    root_pred_norm: torch.Tensor,
    root_gt_norm: torch.Tensor,
    axis_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    return weighted_axis_loss(loss_root_smooth_l1_axis(root_pred_norm, root_gt_norm), axis_weights)


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
