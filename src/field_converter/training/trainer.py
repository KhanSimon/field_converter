from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from field_converter.evaluation.metrics import MetricsAccumulator
from field_converter.losses.projection_losses import loss_reprojection
from field_converter.losses.root_losses import loss_cam3d, loss_root_smooth_l1
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
    w_cam3d: float,
    w_proj: float,
) -> Dict[str, torch.Tensor]:
    root_gt_norm = batch["root_gt"].to(dtype=torch.float32)

    l_root = loss_root_smooth_l1(root_pred_norm, root_gt_norm)
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
    w_cam3d: float,
    w_proj: float,
) -> Dict[str, float]:
    model.eval()
    stats = stats.to(device)

    metrics_acc = MetricsAccumulator()

    loss_total_sum = 0.0
    loss_root_sum = 0.0
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
            w_cam3d=w_cam3d,
            w_proj=w_proj,
        )

        loss_total_sum += float(losses["loss_total"].item())
        loss_root_sum += float(losses["loss_root"].item())
        loss_cam3d_sum += float(losses["loss_cam3d"].item())
        loss_proj_sum += float(losses["loss_proj"].item())
        n_batches += 1

        metrics_acc.update(batch=batch, root_pred_norm=root_pred, stats=stats)

    metrics = metrics_acc.compute()

    if n_batches > 0:
        metrics["valid_loss_total"] = loss_total_sum / n_batches
        metrics["valid_loss_root"] = loss_root_sum / n_batches
        metrics["valid_loss_cam3d"] = loss_cam3d_sum / n_batches
        metrics["valid_loss_proj"] = loss_proj_sum / n_batches
    else:
        metrics["valid_loss_total"] = float("nan")
        metrics["valid_loss_root"] = float("nan")
        metrics["valid_loss_cam3d"] = float("nan")
        metrics["valid_loss_proj"] = float("nan")

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
    w_cam3d: float,
    w_proj: float,
    checkpoints_dir: Path,
    train_log_csv: Path,
) -> TrainerState:
    ensure_dir(checkpoints_dir)
    ensure_dir(train_log_csv.parent)

    model.to(device)
    stats = stats.to(device)

    state = TrainerState()

    # CSV logger
    write_header = not train_log_csv.exists()
    with train_log_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "train_loss_total",
                    "train_loss_root",
                    "train_loss_cam3d",
                    "train_loss_proj",
                    "valid_loss_total",
                    "valid_loss_root",
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
            loss_cam3d_sum = 0.0
            loss_proj_sum = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"train epoch {epoch}")
            for batch in pbar:
                batch = {
                    k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }

                x = batch["x"].to(dtype=torch.float32)
                root_gt = batch["root_gt"].to(dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                root_pred = model(x)

                l_root = loss_root_smooth_l1(root_pred, root_gt)
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

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))

                optimizer.step()

                loss_total_sum += float(loss_total.item())
                loss_root_sum += float(l_root.item())
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
            train_loss_cam3d = loss_cam3d_sum / max(1, n_batches)
            train_loss_proj = loss_proj_sum / max(1, n_batches)

            valid_metrics = _validate(
                model=model,
                dataloader=valid_loader,
                device=device,
                stats=stats,
                w_root=w_root,
                w_cam3d=w_cam3d,
                w_proj=w_proj,
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
                    train_loss_cam3d,
                    train_loss_proj,
                    float(valid_metrics.get("valid_loss_total", float("nan"))),
                    float(valid_metrics.get("valid_loss_root", float("nan"))),
                    float(valid_metrics.get("valid_loss_cam3d", float("nan"))),
                    float(valid_metrics.get("valid_loss_proj", float("nan"))),
                    float(valid_metrics.get("root_error_mean_m", float("nan"))),
                    float(valid_metrics.get("MPJPE_cam_m", float("nan"))),
                    float(valid_metrics.get("MPJPE_world_m", float("nan"))),
                    float(valid_metrics.get("reprojection_error_mean_px", float("nan"))),
                ]
            )
            f.flush()

            if early_stopping_patience > 0 and state.epochs_since_improve >= int(early_stopping_patience):
                break

    return state
