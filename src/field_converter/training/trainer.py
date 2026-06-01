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
    PredictionDiagnostics,
    append_train_batch_diagnostic,
    build_train_batch_diagnostic,
    compute_grad_norm,
)
from field_converter.evaluation.metrics import MetricsAccumulator
from field_converter.losses.projection_losses import loss_reprojection
from field_converter.losses.root_losses import loss_cam3d, loss_root_smooth_l1_axis, weighted_axis_loss
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


def _compute_losses(
    *,
    batch: Dict[str, Any],
    root_pred_norm: torch.Tensor,
    stats: TorchNormalizationStats,
    w_root: float,
    root_axis_weights: Sequence[float],
    w_cam3d: float,
    w_proj: float,
) -> Dict[str, torch.Tensor]:
    root_gt_norm = batch["root_gt"].to(dtype=torch.float32)

    l_root_axis = loss_root_smooth_l1_axis(root_pred_norm, root_gt_norm)
    l_root = weighted_axis_loss(l_root_axis, root_axis_weights)
    if w_cam3d != 0.0:
        l_cam3d = loss_cam3d(
            x3d_sam_norm=batch["x3d_sam_norm"].to(dtype=torch.float32),
            root_pred_norm=root_pred_norm,
            Y_cam_gt=batch["Y_cam_gt"].to(dtype=torch.float32),
            valid_joints=batch["valid_joints"],
            stats=stats,
            loss_type="smoothl1",
        )
    else:
        l_cam3d = root_pred_norm.new_zeros(())

    if w_proj != 0.0:
        l_proj = loss_reprojection(
            x3d_sam_norm=batch["x3d_sam_norm"].to(dtype=torch.float32),
            root_pred_norm=root_pred_norm,
            K=batch["K"].to(dtype=torch.float32),
            k=batch["k"].to(dtype=torch.float32),
            Y_2d_gt=batch["Y_2d_gt"].to(dtype=torch.float32),
            valid_joints=batch["valid_joints"],
            stats=stats,
            loss_type="smoothl1",
        )
    else:
        l_proj = root_pred_norm.new_zeros(())

    total = w_root * l_root + w_cam3d * l_cam3d + w_proj * l_proj
    return {
        "loss_total": total,
        "loss_root": l_root.detach(),
        "loss_root_x": l_root_axis[0].detach(),
        "loss_root_y": l_root_axis[1].detach(),
        "loss_root_z": l_root_axis[2].detach(),
        "loss_cam3d": l_cam3d.detach(),
        "loss_proj": l_proj.detach(),
    }


@torch.no_grad()
def _validate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    stats: TorchNormalizationStats,
    w_root: float,
    root_axis_weights: Sequence[float],
    w_cam3d: float,
    w_proj: float,
    diagnostics_dir: Optional[Path] = None,
    diagnostics_prefix: Optional[str] = None,
    diagnostics_top_k: int = 100,
) -> Dict[str, float]:
    model.eval()
    stats = stats.to(device)

    metrics_acc = MetricsAccumulator()
    seq_names = list(getattr(dataloader.dataset, "sequences", []))
    seq_to_id = {s: i for i, s in enumerate(seq_names)}
    diagnostics = (
        PredictionDiagnostics(seq_names=seq_names, top_k=diagnostics_top_k)
        if diagnostics_dir is not None
        else None
    )

    loss_total_sum = 0.0
    loss_root_sum = 0.0
    loss_root_axis_sum = np.zeros((3,), dtype=np.float64)
    loss_cam3d_sum = 0.0
    loss_proj_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        x = batch["x"].to(dtype=torch.float32)
        root_pred = model(x)

        losses = _compute_losses(
            batch=batch,
            root_pred_norm=root_pred,
            stats=stats,
            w_root=w_root,
            root_axis_weights=root_axis_weights,
            w_cam3d=w_cam3d,
            w_proj=w_proj,
        )

        loss_total_sum += float(losses["loss_total"].item())
        loss_root_sum += float(losses["loss_root"].item())
        loss_root_axis_sum += np.array(
            [
                float(losses["loss_root_x"].item()),
                float(losses["loss_root_y"].item()),
                float(losses["loss_root_z"].item()),
            ],
            dtype=np.float64,
        )
        loss_cam3d_sum += float(losses["loss_cam3d"].item())
        loss_proj_sum += float(losses["loss_proj"].item())
        n_batches += 1

        metrics_acc.update(batch=batch, root_pred_norm=root_pred, stats=stats)

        if diagnostics is not None:
            root_pred_m = stats.denorm_root(root_pred)
            root_gt_m = stats.denorm_root(batch["root_gt"].to(dtype=torch.float32))
            root_err_m = torch.linalg.norm(root_pred_m - root_gt_m, dim=-1)
            if "seq_name" in batch:
                seq_ids = np.array([seq_to_id.get(s, -1) for s in batch["seq_name"]], dtype=np.int32)
            else:
                seq_ids = np.full((root_pred.shape[0],), -1, dtype=np.int32)
            diagnostics.update(
                seq_id=seq_ids,
                person_idx=batch["person_idx"],
                frame_idx=batch["frame_idx"],
                root_pred_norm=root_pred,
                root_gt_norm=batch["root_gt"].to(dtype=torch.float32),
                root_pred_m=root_pred_m,
                root_gt_m=root_gt_m,
                root_error_m=root_err_m,
            )

    metrics = metrics_acc.compute()

    if n_batches > 0:
        metrics["valid_loss_total"] = loss_total_sum / n_batches
        metrics["valid_loss_root"] = loss_root_sum / n_batches
        metrics["valid_loss_root_x"] = float(loss_root_axis_sum[0] / n_batches)
        metrics["valid_loss_root_y"] = float(loss_root_axis_sum[1] / n_batches)
        metrics["valid_loss_root_z"] = float(loss_root_axis_sum[2] / n_batches)
        metrics["valid_loss_cam3d"] = loss_cam3d_sum / n_batches
        metrics["valid_loss_proj"] = loss_proj_sum / n_batches
    else:
        metrics["valid_loss_total"] = float("nan")
        metrics["valid_loss_root"] = float("nan")
        metrics["valid_loss_root_x"] = float("nan")
        metrics["valid_loss_root_y"] = float("nan")
        metrics["valid_loss_root_z"] = float("nan")
        metrics["valid_loss_cam3d"] = float("nan")
        metrics["valid_loss_proj"] = float("nan")

    if diagnostics is not None and diagnostics_dir is not None:
        diagnostics.write(
            out_dir=diagnostics_dir,
            prefix=diagnostics_prefix or "valid",
        )

    return metrics


def train(
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
    w_cam3d: float,
    w_proj: float,
    checkpoints_dir: Path,
    train_log_csv: Path,
) -> TrainerState:
    ensure_dir(checkpoints_dir)
    ensure_dir(train_log_csv.parent)
    diagnostics_dir = train_log_csv.parent / "diagnostics"

    model.to(device)
    stats = stats.to(device)

    state = TrainerState()

    # CSV logger
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
                    "train_loss_cam3d",
                    "train_loss_proj",
                    "valid_loss_total",
                    "valid_loss_root",
                    "valid_loss_root_x",
                    "valid_loss_root_y",
                    "valid_loss_root_z",
                    "valid_loss_cam3d",
                    "valid_loss_proj",
                    "valid_root_error_mean_m",
                    "valid_MPJPE_cam_m",
                    "valid_MPJPE_world_m",
                    "valid_reprojection_error_mean_px",
                ]
            )

        for epoch in range(1, int(epochs) + 1):
            state.epoch = epoch
            model.train()

            # Let custom samplers change their order each epoch.
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            loss_total_sum = 0.0
            loss_root_sum = 0.0
            loss_root_axis_sum = np.zeros((3,), dtype=np.float64)
            loss_cam3d_sum = 0.0
            loss_proj_sum = 0.0
            n_batches = 0
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

                optimizer.zero_grad(set_to_none=True)
                root_pred = model(x)

                l_root_axis = loss_root_smooth_l1_axis(root_pred, root_gt)
                l_root = weighted_axis_loss(l_root_axis, root_axis_weights)
                if w_cam3d != 0.0:
                    l_cam3d = loss_cam3d(
                        x3d_sam_norm=batch["x3d_sam_norm"].to(dtype=torch.float32),
                        root_pred_norm=root_pred,
                        Y_cam_gt=batch["Y_cam_gt"].to(dtype=torch.float32),
                        valid_joints=batch["valid_joints"],
                        stats=stats,
                        loss_type="smoothl1",
                    )
                else:
                    l_cam3d = root_pred.new_zeros(())

                if w_proj != 0.0:
                    l_proj = loss_reprojection(
                        x3d_sam_norm=batch["x3d_sam_norm"].to(dtype=torch.float32),
                        root_pred_norm=root_pred,
                        K=batch["K"].to(dtype=torch.float32),
                        k=batch["k"].to(dtype=torch.float32),
                        Y_2d_gt=batch["Y_2d_gt"].to(dtype=torch.float32),
                        valid_joints=batch["valid_joints"],
                        stats=stats,
                        loss_type="smoothl1",
                    )
                else:
                    l_proj = root_pred.new_zeros(())

                loss_total = w_root * l_root + w_cam3d * l_cam3d + w_proj * l_proj
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
            train_loss_cam3d = loss_cam3d_sum / max(1, n_batches)
            train_loss_proj = loss_proj_sum / max(1, n_batches)

            valid_metrics = _validate(
                model=model,
                dataloader=valid_loader,
                device=device,
                stats=stats,
                w_root=w_root,
                root_axis_weights=root_axis_weights,
                w_cam3d=w_cam3d,
                w_proj=w_proj,
                diagnostics_dir=diagnostics_dir,
                diagnostics_prefix=f"valid_epoch_{epoch:03d}",
            )

            valid_root_err = float(valid_metrics.get("root_error_mean_m", float("inf")))

            # Save last checkpoint
            _save_checkpoint(
                path=checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                state=state,
            )

            # Save best checkpoint (lowest root_error_mean_m)
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
                    train_loss_cam3d,
                    train_loss_proj,
                    float(valid_metrics.get("valid_loss_total", float("nan"))),
                    float(valid_metrics.get("valid_loss_root", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_x", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_y", float("nan"))),
                    float(valid_metrics.get("valid_loss_root_z", float("nan"))),
                    float(valid_metrics.get("valid_loss_cam3d", float("nan"))),
                    float(valid_metrics.get("valid_loss_proj", float("nan"))),
                    float(valid_metrics.get("root_error_mean_m", float("nan"))),
                    float(valid_metrics.get("MPJPE_cam_m", float("nan"))),
                    float(valid_metrics.get("MPJPE_world_m", float("nan"))),
                    float(valid_metrics.get("reprojection_error_mean_px", float("nan"))),
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
