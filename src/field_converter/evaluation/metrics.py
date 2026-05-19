from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from field_converter.geometry.projection import project_cam_to_image
from field_converter.geometry.transforms import cam_to_world
from field_converter.losses.root_losses import reconstruct_X_cam_pred
from field_converter.utils.normalization import TorchNormalizationStats


@dataclass
class MetricsAccumulator:
    """Accumulates evaluation metrics in de-normalized units."""

    # Root translation errors (per-sample)
    _root_errors: list[np.ndarray] = field(default_factory=list)
    _root_abs_axis_sum: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float64))
    _root_count: int = 0

    # MPJPE (streaming)
    _mpjpe_cam_sum: float = 0.0
    _mpjpe_cam_count: int = 0
    _mpjpe_world_sum: float = 0.0
    _mpjpe_world_count: int = 0

    # Reprojection errors (store valid values for median)
    _reproj_errors: list[np.ndarray] = field(default_factory=list)
    _reproj_sum: float = 0.0
    _reproj_count: int = 0

    def update(
        self,
        *,
        batch: Dict[str, Any],
        root_pred_norm: torch.Tensor,
        stats: TorchNormalizationStats,
    ) -> None:
        """Update metrics from one batch.

        Expected batch keys:
        - root_gt: (B,3) normalized
        - x3d_sam_norm: (B,25,3) normalized
        - valid_joints: (B,25) bool
        - K: (B,3,3)
        - R: (B,3,3)
        - t: (B,3)
        - k: (B,2)
        - Y_cam_gt: (B,25,3) meters
        - Y_2d_gt: (B,25,2) pixels
        """
        root_gt_norm = batch["root_gt"]
        valid_joints = batch["valid_joints"]
        x3d_sam_norm = batch["x3d_sam_norm"]

        K = batch["K"]
        R = batch["R"]
        t = batch["t"]
        k = batch["k"]

        Y_cam_gt = batch["Y_cam_gt"]
        Y_2d_gt = batch["Y_2d_gt"]

        # ---- Root errors (meters)
        root_pred_m = stats.denorm_root(root_pred_norm)
        root_gt_m = stats.denorm_root(root_gt_norm)

        diff_root = root_pred_m - root_gt_m
        err_root = torch.linalg.norm(diff_root, dim=-1)  # (B,)

        self._root_errors.append(err_root.detach().cpu().numpy())
        self._root_abs_axis_sum += diff_root.detach().abs().sum(dim=0).cpu().numpy().astype(np.float64)
        self._root_count += int(diff_root.shape[0])

        # ---- Joint reconstruction (camera, meters)
        X_cam_pred = reconstruct_X_cam_pred(x3d_sam_norm=x3d_sam_norm, root_cam_norm=root_pred_norm, stats=stats)

        # ---- MPJPE (camera)
        dist_cam = torch.linalg.norm(X_cam_pred - Y_cam_gt, dim=-1)  # (B,25)
        mask = valid_joints.bool()
        vals_cam = dist_cam[mask]
        if vals_cam.numel() > 0:
            self._mpjpe_cam_sum += float(vals_cam.sum().item())
            self._mpjpe_cam_count += int(vals_cam.numel())

        # ---- MPJPE (world)
        X_world_pred = cam_to_world(X_cam_pred, R=R, t=t)
        Y_world_gt = cam_to_world(Y_cam_gt, R=R, t=t)
        dist_world = torch.linalg.norm(X_world_pred - Y_world_gt, dim=-1)  # (B,25)
        vals_world = dist_world[mask]
        if vals_world.numel() > 0:
            self._mpjpe_world_sum += float(vals_world.sum().item())
            self._mpjpe_world_count += int(vals_world.numel())

        # ---- Reprojection error (pixels)
        uv_pred = project_cam_to_image(X_cam_pred, K=K, k=k)
        dist_px = torch.linalg.norm(uv_pred - Y_2d_gt, dim=-1)  # (B,25)
        vals_px = dist_px[mask]
        if vals_px.numel() > 0:
            self._reproj_sum += float(vals_px.sum().item())
            self._reproj_count += int(vals_px.numel())
            self._reproj_errors.append(vals_px.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        if self._root_count > 0 and self._root_errors:
            root_all = np.concatenate(self._root_errors, axis=0)
            metrics["root_error_mean_m"] = float(root_all.mean())
            metrics["root_error_median_m"] = float(np.median(root_all))
            metrics["root_error_p90_m"] = float(np.quantile(root_all, 0.90))

            axis_mean_abs = self._root_abs_axis_sum / float(self._root_count)
            metrics["root_error_x_m"] = float(axis_mean_abs[0])
            metrics["root_error_y_m"] = float(axis_mean_abs[1])
            metrics["root_error_z_m"] = float(axis_mean_abs[2])
        else:
            metrics["root_error_mean_m"] = float("nan")
            metrics["root_error_median_m"] = float("nan")
            metrics["root_error_p90_m"] = float("nan")
            metrics["root_error_x_m"] = float("nan")
            metrics["root_error_y_m"] = float("nan")
            metrics["root_error_z_m"] = float("nan")

        metrics["MPJPE_cam_m"] = (
            float(self._mpjpe_cam_sum / self._mpjpe_cam_count)
            if self._mpjpe_cam_count > 0
            else float("nan")
        )
        metrics["MPJPE_world_m"] = (
            float(self._mpjpe_world_sum / self._mpjpe_world_count)
            if self._mpjpe_world_count > 0
            else float("nan")
        )

        if self._reproj_count > 0 and self._reproj_errors:
            reproj_all = np.concatenate(self._reproj_errors, axis=0)
            metrics["reprojection_error_mean_px"] = float(self._reproj_sum / self._reproj_count)
            metrics["reprojection_error_median_px"] = float(np.median(reproj_all))
        else:
            metrics["reprojection_error_mean_px"] = float("nan")
            metrics["reprojection_error_median_px"] = float("nan")

        return metrics
