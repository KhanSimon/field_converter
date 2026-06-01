from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from field_converter.losses.root_losses import weighted_axis_loss


def _masked_smooth_l1_mean(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    axis_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """SmoothL1 averaged over masked time steps.

    Parameters
    ----------
    pred, gt:
        (..., T, C) or (B, T, C)
    mask:
        (..., T) boolean mask. Only masked time steps contribute.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}")
    if mask.shape != pred.shape[:-1]:
        raise ValueError(f"mask must match pred without last dim, got mask={mask.shape} pred={pred.shape}")

    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
    valid = mask.bool() & finite
    if not bool(valid.any()):
        return pred.new_zeros(())

    per_coord = F.smooth_l1_loss(pred[valid], gt[valid], reduction="none")
    axis_loss = per_coord.mean(dim=0)
    return weighted_axis_loss(axis_loss, axis_weights)


def masked_smooth_l1_axis_mean(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return unweighted SmoothL1 mean per axis for masked time steps."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}")
    if mask.shape != pred.shape[:-1]:
        raise ValueError(f"mask must match pred without last dim, got mask={mask.shape} pred={pred.shape}")
    if pred.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {pred.shape}")

    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
    valid = mask.bool() & finite
    if not bool(valid.any()):
        return pred.new_zeros((3,))

    per_coord = F.smooth_l1_loss(pred[valid], gt[valid], reduction="none")
    return per_coord.mean(dim=0)


def loss_root_masked_smooth_l1(
    root_pred_norm: torch.Tensor,
    root_gt_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    axis_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Masked SmoothL1(root_pred_norm, root_gt_norm) over time.

    Shapes
    ------
    - root_pred_norm: (B,T,3)
    - root_gt_norm: (B,T,3)
    - valid_mask: (B,T)
    """
    return _masked_smooth_l1_mean(root_pred_norm, root_gt_norm, valid_mask, axis_weights=axis_weights)


def loss_root_velocity(
    root_pred_norm: torch.Tensor,
    root_gt_norm: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Velocity loss in normalized space.

    SmoothL1(Δroot_pred, Δroot_gt) over valid consecutive pairs.

    Shapes
    ------
    - root_pred_norm: (B,T,3)
    - root_gt_norm: (B,T,3)
    - valid_mask: (B,T)
    """
    if root_pred_norm.ndim != 3:
        raise ValueError(f"Expected (B,T,3) inputs, got {root_pred_norm.shape}")
    if root_pred_norm.shape != root_gt_norm.shape:
        raise ValueError("root_pred_norm and root_gt_norm must have same shape")
    if valid_mask.shape != root_pred_norm.shape[:-1]:
        raise ValueError("valid_mask must have shape (B,T)")

    if root_pred_norm.shape[1] < 2:
        return root_pred_norm.new_zeros(())

    vel_pred = root_pred_norm[:, 1:] - root_pred_norm[:, :-1]
    vel_gt = root_gt_norm[:, 1:] - root_gt_norm[:, :-1]
    pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    return _masked_smooth_l1_mean(vel_pred, vel_gt, pair_mask)


def loss_root_acceleration(
    root_pred_norm: torch.Tensor,
    root_gt_norm: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Acceleration loss in normalized space.

    SmoothL1(Δ²root_pred, Δ²root_gt) over valid consecutive triplets.
    """
    if root_pred_norm.shape[1] < 3:
        return root_pred_norm.new_zeros(())

    acc_pred = root_pred_norm[:, 2:] - 2.0 * root_pred_norm[:, 1:-1] + root_pred_norm[:, :-2]
    acc_gt = root_gt_norm[:, 2:] - 2.0 * root_gt_norm[:, 1:-1] + root_gt_norm[:, :-2]
    triplet_mask = valid_mask[:, 2:] & valid_mask[:, 1:-1] & valid_mask[:, :-2]
    return _masked_smooth_l1_mean(acc_pred, acc_gt, triplet_mask)
