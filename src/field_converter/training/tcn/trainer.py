from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from field_converter.evaluation.diagnostics import (
    append_train_batch_diagnostic,
    build_train_batch_diagnostic,
    compute_grad_norm,
)
from field_converter.evaluation.temporal_evaluator import TemporalEvaluator
from field_converter.losses.projection_losses import loss_reprojection
from field_converter.losses.root_losses import loss_cam3d
from field_converter.losses.temporal_losses import (
    loss_root_acceleration,
    loss_root_masked_smooth_l1,
    loss_root_velocity,
    masked_smooth_l1_axis_mean,
)
from field_converter.utils.io import ensure_dir
from field_converter.utils.normalization import TorchNormalizationStats


@dataclass
class TrainerState:
    epoch: int = 0
    best_root_error_mean_m: float = float("inf")
    best_epoch: int = -1
    epochs_since_improve: int = 0


def _save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainerState,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": state.epoch,
            "best_root_error_mean_m": state.best_root_error_mean_m,
            "best_epoch": state.best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_tcn(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    stats: TorchNormalizationStats,
    epochs: int,
    grad_clip_norm: Optional[float],
    early_stopping_patience: int,
    w_root: float,
    root_axis_weights: Sequence[float],
    w_root_vel: float,
    w_root_acc: float,
    w_cam3d: float,
    w_proj: float,
    min_in_image_joints_ratio: Optional[float],
    min_bbox_width_px: Optional[float],
    min_bbox_height_px: Optional[float],
    min_bbox_margin_px: Optional[float],
    checkpoints_dir: Path,
    train_log_csv: Path,
) -> TrainerState:
    ensure_dir(checkpoints_dir)
    ensure_dir(train_log_csv.parent)
    diagnostics_dir = train_log_csv.parent / "diagnostics"

    model.to(device)
    stats = stats.to(device)

    # Aggregated validation (no predictions saved during training).
    evaluator = TemporalEvaluator(
        stats=stats.to("cpu"),
        device=device,
        save_predictions_npz=False,
        save_predictions_csv=False,
    )

    state = TrainerState()

    write_header = (not train_log_csv.exists()) or train_log_csv.stat().st_size == 0
    with train_log_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "train_loss_total",
                    "train_loss_root",
                    "train_loss_root_x",
                    "train_loss_root_y",
                    "train_loss_root_z",
                    "train_loss_root_vel",
                    "train_loss_root_acc",
                    "train_loss_cam3d",
                    "train_loss_proj",
                    "valid_loss_total",
                    "valid_loss_root",
                    "valid_loss_root_x",
                    "valid_loss_root_y",
                    "valid_loss_root_z",
                    "valid_loss_root_vel",
                    "valid_loss_root_acc",
                    "valid_root_error_mean_m",
                    "valid_MPJPE_cam_m",
                    "valid_MPJPE_world_m",
                    "valid_reprojection_error_mean_px",
                    "valid_root_velocity_error_mean_m",
                    "valid_root_acceleration_error_mean_m",
                ]
            )

        for epoch in range(1, int(epochs) + 1):
            state.epoch = epoch
            model.train()

            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            n_batches = 0
            loss_total_sum = 0.0
            loss_root_sum = 0.0
            loss_root_axis_sum = np.zeros((3,), dtype=np.float64)
            loss_vel_sum = 0.0
            loss_acc_sum = 0.0
            loss_cam3d_sum = 0.0
            loss_proj_sum = 0.0
            worst_batch_loss = float("-inf")
            worst_batch_diagnostic: Optional[dict[str, Any]] = None

            pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                batch = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                x = batch["x"].to(dtype=torch.float32)
                root_gt = batch["root_gt"].to(dtype=torch.float32)
                valid_mask = batch["valid_mask"].to(device=device)

                optimizer.zero_grad(set_to_none=True)
                root_pred = model(x).to(dtype=torch.float32)

                l_root_axis = masked_smooth_l1_axis_mean(root_pred, root_gt, valid_mask)
                l_root = loss_root_masked_smooth_l1(root_pred, root_gt, valid_mask, axis_weights=root_axis_weights)
                l_vel = loss_root_velocity(root_pred, root_gt, valid_mask)
                l_acc = loss_root_acceleration(root_pred, root_gt, valid_mask)

                if w_cam3d != 0.0:
                    B, W, _ = root_pred.shape
                    x3d = batch["x3d_sam_norm"].to(dtype=torch.float32).reshape(B * W, 25, 3)
                    Y_cam = batch["Y_cam_gt"].to(dtype=torch.float32).reshape(B * W, 25, 3)
                    vj = batch["valid_joints"].reshape(B * W, 25).bool()
                    vm = valid_mask.reshape(B * W).bool()
                    vj = vj & vm[:, None]
                    l_cam3d = loss_cam3d(
                        x3d_sam_norm=x3d,
                        root_pred_norm=root_pred.reshape(B * W, 3),
                        Y_cam_gt=Y_cam,
                        valid_joints=vj,
                        stats=stats,
                        loss_type="smoothl1",
                    )
                else:
                    l_cam3d = root_pred.new_zeros(())

                if w_proj != 0.0:
                    B, W, _ = root_pred.shape
                    x3d = batch["x3d_sam_norm"].to(dtype=torch.float32).reshape(B * W, 25, 3)
                    Y_2d = batch["Y_2d_gt"].to(dtype=torch.float32).reshape(B * W, 25, 2)
                    K = batch["K"].to(dtype=torch.float32).reshape(B * W, 3, 3)
                    k = batch["k"].to(dtype=torch.float32).reshape(B * W, 2)
                    vj = batch["valid_joints"].reshape(B * W, 25).bool()
                    vm = valid_mask.reshape(B * W).bool()
                    vj = vj & vm[:, None]
                    l_proj = loss_reprojection(
                        x3d_sam_norm=x3d,
                        root_pred_norm=root_pred.reshape(B * W, 3),
                        K=K,
                        k=k,
                        Y_2d_gt=Y_2d,
                        valid_joints=vj,
                        stats=stats,
                        loss_type="smoothl1",
                    )
                else:
                    l_proj = root_pred.new_zeros(())

                loss_total = (
                    w_root * l_root
                    + w_root_vel * l_vel
                    + w_root_acc * l_acc
                    + w_cam3d * l_cam3d
                    + w_proj * l_proj
                )

                loss_total.backward()
                grad_norm: Optional[float] = None
                if grad_clip_norm is not None:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                    grad_norm = float(grad_norm_t.item())
                else:
                    grad_norm = compute_grad_norm(model.parameters())
                optimizer.step()

                loss_total_value = float(loss_total.item())
                if loss_total_value > worst_batch_loss:
                    worst_batch_loss = loss_total_value
                    worst_batch_diagnostic = build_train_batch_diagnostic(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch=batch,
                        losses={
                            "loss_total": loss_total_value,
                            "loss_root": float(l_root.item()),
                            "loss_root_x": float(l_root_axis[0].item()),
                            "loss_root_y": float(l_root_axis[1].item()),
                            "loss_root_z": float(l_root_axis[2].item()),
                            "loss_root_vel": float(l_vel.item()),
                            "loss_root_acc": float(l_acc.item()),
                            "loss_cam3d": float(l_cam3d.item()),
                            "loss_proj": float(l_proj.item()),
                        },
                        root_pred_norm=root_pred,
                        root_gt_norm=root_gt,
                        stats=stats,
                        grad_norm=grad_norm,
                    )

                loss_total_sum += loss_total_value
                loss_root_sum += float(l_root.item())
                loss_root_axis_sum += np.array(
                    [float(l_root_axis[0].item()), float(l_root_axis[1].item()), float(l_root_axis[2].item())],
                    dtype=np.float64,
                )
                loss_vel_sum += float(l_vel.item())
                loss_acc_sum += float(l_acc.item())
                loss_cam3d_sum += float(l_cam3d.item())
                loss_proj_sum += float(l_proj.item())
                n_batches += 1

                pbar.set_postfix(
                    {
                        "loss": f"{loss_total_sum / max(1,n_batches):.4f}",
                        "root": f"{loss_root_sum / max(1,n_batches):.4f}",
                    }
                )

            train_loss_total = loss_total_sum / max(1, n_batches)
            train_loss_root = loss_root_sum / max(1, n_batches)
            train_loss_root_axis = loss_root_axis_sum / max(1, n_batches)
            train_loss_vel = loss_vel_sum / max(1, n_batches)
            train_loss_acc = loss_acc_sum / max(1, n_batches)
            train_loss_cam3d = loss_cam3d_sum / max(1, n_batches)
            train_loss_proj = loss_proj_sum / max(1, n_batches)

            # ---- Validation metrics after overlap aggregation
            model.eval()
            (valid_out, _extras) = evaluator.evaluate_split(
                model=model,
                dataloader=valid_loader,
                out_dir=checkpoints_dir,  # no files written (save_predictions=False)
                split_name="valid",
                min_in_image_joints_ratio=min_in_image_joints_ratio,
                min_bbox_width_px=min_bbox_width_px,
                min_bbox_height_px=min_bbox_height_px,
                min_bbox_margin_px=min_bbox_margin_px,
                root_axis_weights=root_axis_weights,
                diagnostics_dir=diagnostics_dir,
                diagnostics_prefix=f"valid_epoch_{epoch:03d}",
            )
            valid_metrics = valid_out.metrics
            valid_root_err = float(valid_metrics.get("root_error_mean_m", float("inf")))

            _save_checkpoint(
                path=checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                state=state,
            )

            improved = valid_root_err < (state.best_root_error_mean_m - 1e-9)
            if improved:
                state.best_root_error_mean_m = valid_root_err
                state.best_epoch = epoch
                state.epochs_since_improve = 0
                _save_checkpoint(
                    path=checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    state=state,
                )
            else:
                state.epochs_since_improve += 1

            writer.writerow(
                [
                    epoch,
                    train_loss_total,
                    train_loss_root,
                    float(train_loss_root_axis[0]),
                    float(train_loss_root_axis[1]),
                    float(train_loss_root_axis[2]),
                    train_loss_vel,
                    train_loss_acc,
                    train_loss_cam3d,
                    train_loss_proj,
                    float(valid_metrics.get("valid_loss_total", float("nan"))),
                    float(valid_metrics.get("valid_loss_root", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_x", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_y", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_z", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_vel", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_acc", float("nan"))),
                    float(valid_metrics.get("root_error_mean_m", float("nan"))),
                    float(valid_metrics.get("MPJPE_cam_m", float("nan"))),
                    float(valid_metrics.get("MPJPE_world_m", float("nan"))),
                    float(valid_metrics.get("reprojection_error_mean_px", float("nan"))),
                    float(valid_metrics.get("root_velocity_error_mean_m", float("nan"))),
                    float(valid_metrics.get("root_acceleration_error_mean_m", float("nan"))),
                ]
            )
            f.flush()

            if worst_batch_diagnostic is not None:
                append_train_batch_diagnostic(
                    diagnostics_dir / "train_worst_batches.csv",
                    worst_batch_diagnostic,
                )

            if early_stopping_patience > 0 and state.epochs_since_improve >= int(early_stopping_patience):
                break

    return state
