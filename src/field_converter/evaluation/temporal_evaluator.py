from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from field_converter.evaluation.diagnostics import PredictionDiagnostics
from field_converter.evaluation.evaluator import EvalOutputs
from field_converter.evaluation.metrics import MetricsAccumulator
from field_converter.geometry.transforms import cam_to_world
from field_converter.models.temporal import TemporalRootModel, forward_temporal_root_model
from field_converter.training.filters import filter_valid_mask_bbox_geometry, filter_valid_mask_in_image
from field_converter.utils.io import ensure_dir
from field_converter.utils.normalization import TorchNormalizationStats


@dataclass(frozen=True)
class TemporalEvalExtras:
    num_frames_total: int
    num_frames_covered: int
    num_frames_uncovered: int


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _masked_smooth_l1_sum_and_count(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, int]:
    """Return SmoothL1 sum/count over valid time steps only.

    Invalid steps may contain NaNs in the GT arrays. Index before computing the
    loss so NaN * 0 cannot contaminate the reduction.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}")
    if mask.shape != pred.shape[:-1]:
        raise ValueError(f"mask must match pred without last dim, got mask={mask.shape} pred={pred.shape}")

    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
    valid = mask.bool() & finite
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0

    per_step = torch.nn.functional.smooth_l1_loss(pred[valid], gt[valid], reduction="none").mean(dim=-1)
    return float(per_step.sum().item()), count


def _masked_smooth_l1_axis_sum_and_count(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[np.ndarray, int]:
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} vs {gt.shape}")
    if mask.shape != pred.shape[:-1]:
        raise ValueError(f"mask must match pred without last dim, got mask={mask.shape} pred={pred.shape}")
    if pred.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {pred.shape}")

    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
    valid = mask.bool() & finite
    count = int(valid.sum().item())
    if count == 0:
        return np.zeros((3,), dtype=np.float64), 0

    per_coord = torch.nn.functional.smooth_l1_loss(pred[valid], gt[valid], reduction="none")
    return _to_numpy(per_coord.sum(dim=0)).astype(np.float64), count


class TemporalEvaluator:
    """Evaluator for window-based temporal models.

    Runs the model on overlapping windows, aggregates per-frame predictions,
    then computes metrics on the aggregated per-frame outputs.

    Aggregation is currently a simple mean over overlaps.
    """

    def __init__(
        self,
        *,
        stats: TorchNormalizationStats,
        device: torch.device,
        save_predictions_npz: bool = True,
        save_predictions_csv: bool = False,
    ) -> None:
        self.stats = stats
        self.device = device
        self.save_predictions_npz = bool(save_predictions_npz)
        self.save_predictions_csv = bool(save_predictions_csv)

    @torch.no_grad()
    def evaluate_split(
        self,
        *,
        model: TemporalRootModel,
        dataloader: torch.utils.data.DataLoader,
        out_dir: Path,
        split_name: str,
        min_in_image_joints_ratio: Optional[float] = None,
        min_bbox_width_px: Optional[float] = None,
        min_bbox_height_px: Optional[float] = None,
        min_bbox_margin_px: Optional[float] = None,
        root_axis_weights: Optional[Sequence[float]] = None,
        diagnostics_dir: Optional[Path] = None,
        diagnostics_prefix: Optional[str] = None,
        diagnostics_top_k: int = 100,
    ) -> Tuple[EvalOutputs, TemporalEvalExtras]:
        """Evaluate a temporal model on one split.

        Returns
        -------
        (outputs, extras)
            - outputs.metrics: same keys as the frame-wise evaluator + temporal metrics
            - extras: coverage stats (covered/uncovered frames)
        """

        ensure_dir(out_dir)

        if isinstance(model, nn.Module):
            model.to(self.device)
            model.eval()

        stats_dev = self.stats.to(self.device)

        dataset = dataloader.dataset
        sequences: List[str] = list(getattr(dataset, "sequences", []))
        seq_to_id = {s: i for i, s in enumerate(sequences)}
        diagnostics = (
            PredictionDiagnostics(seq_names=sequences, top_k=diagnostics_top_k)
            if diagnostics_dir is not None
            else None
        )

        seq_lengths: Optional[List[int]] = getattr(dataset, "seq_lengths", None)

        # ---- Pass 1: aggregate root_pred_norm per frame
        sums: Dict[Tuple[int, int], np.ndarray] = {}
        counts: Dict[Tuple[int, int], np.ndarray] = {}

        for batch in dataloader:
            x = batch["x"].to(self.device, dtype=torch.float32)
            valid_mask = batch.get("valid_mask")
            if isinstance(valid_mask, torch.Tensor):
                valid_mask = valid_mask.to(self.device)
            else:
                valid_mask = None
            pred = forward_temporal_root_model(model, x, valid_mask=valid_mask).to(dtype=torch.float32)  # (B,W,3)

            pred_np = _to_numpy(pred)

            seq_names_b = batch["seq_name"]
            person_b = batch["person_idx"].detach().cpu().numpy().astype(np.int32)
            frame_indices_b = batch["frame_indices"].detach().cpu().numpy().astype(np.int64)

            B = int(pred_np.shape[0])
            for bi in range(B):
                sname = str(seq_names_b[bi])
                sid = int(seq_to_id.get(sname, -1))
                if sid < 0:
                    continue
                pid = int(person_b[bi])

                if (sid, pid) not in sums:
                    if seq_lengths is None:
                        # Fallback: infer from max seen index.
                        T_est = int(frame_indices_b[bi].max()) + 1
                    else:
                        T_est = int(seq_lengths[sid])
                    sums[(sid, pid)] = np.zeros((T_est, 3), dtype=np.float64)
                    counts[(sid, pid)] = np.zeros((T_est,), dtype=np.int64)

                frames = frame_indices_b[bi]
                ok = frames >= 0
                if not np.any(ok):
                    continue

                f = frames[ok].astype(np.int64, copy=False)
                p = pred_np[bi][ok].astype(np.float64, copy=False)

                np.add.at(sums[(sid, pid)], f, p)
                np.add.at(counts[(sid, pid)], f, 1)

        # Coverage stats
        covered = 0
        total = 0
        for key, c in counts.items():
            total += int(c.size)
            covered += int((c > 0).sum())
        uncovered = total - covered

        extras = TemporalEvalExtras(
            num_frames_total=int(total),
            num_frames_covered=int(covered),
            num_frames_uncovered=int(uncovered),
        )

        # ---- Pass 2: compute metrics on aggregated predictions
        metrics_acc = MetricsAccumulator()

        # Temporal errors in meters
        vel_err_sum = 0.0
        vel_err_count = 0
        acc_err_sum = 0.0
        acc_err_count = 0

        # Aggregated valid losses in normalized space
        valid_loss_root_axis = np.zeros((3,), dtype=np.float64)
        valid_loss_root_vel = 0.0
        valid_loss_root_acc = 0.0
        denom_root = 0
        denom_vel = 0
        denom_acc = 0

        # Collect predictions for saving.
        seq_id_chunks: list[np.ndarray] = []
        person_chunks: list[np.ndarray] = []
        frame_chunks: list[np.ndarray] = []

        root_pred_norm_chunks: list[np.ndarray] = []
        root_gt_norm_chunks: list[np.ndarray] = []

        root_pred_m_chunks: list[np.ndarray] = []
        root_gt_m_chunks: list[np.ndarray] = []
        root_err_m_chunks: list[np.ndarray] = []

        root_world_pred_chunks: list[np.ndarray] = []
        root_world_gt_chunks: list[np.ndarray] = []

        # We load each sequence file once for metrics/pred saving.
        data_dir = Path(getattr(dataset, "data_dir"))
        split = str(getattr(dataset, "split", split_name))

        for sid, seq_name in enumerate(sequences):
            seq_path = data_dir / split / f"{seq_name}.npz"
            if not seq_path.exists():
                continue

            with np.load(seq_path, allow_pickle=True) as npz:
                valid_mask = np.asarray(npz["valid_mask"], dtype=bool)  # (P,T)
                valid_joints = np.asarray(npz["valid_joints"], dtype=bool)  # (P,T,J)

                x3d_sam_norm = np.asarray(npz["skel_3d_sam3dbody_from_bbox_gt"], dtype=np.float32)  # (P,T,25,3)
                root_gt_norm = np.asarray(npz["Y_root_cam_gt"], dtype=np.float32)  # (P,T,3)

                K_all = np.asarray(npz["K"], dtype=np.float32)
                R_all = np.asarray(npz["R"], dtype=np.float32)
                t_all = np.asarray(npz["t"], dtype=np.float32)
                k_all = np.asarray(npz["k"], dtype=np.float32)

                Y_cam_gt = np.asarray(npz["Y_cam_gt"], dtype=np.float32)
                Y_2d_gt = np.asarray(npz["Y_2d_gt"], dtype=np.float32)

                if min_in_image_joints_ratio is not None:
                    image_size = np.asarray(npz["image_size"], dtype=np.float32)
                    valid_mask = filter_valid_mask_in_image(
                        valid_mask=valid_mask,
                        valid_joints=valid_joints,
                        Y_2d_gt=Y_2d_gt,
                        image_size=image_size,
                        min_in_image_joints_ratio=float(min_in_image_joints_ratio),
                    )
                if min_bbox_width_px is not None or min_bbox_height_px is not None or min_bbox_margin_px is not None:
                    boxes = np.asarray(npz["boxes_xyxy"], dtype=np.float32)
                    image_size = np.asarray(npz["image_size"], dtype=np.float32)
                    valid_mask = filter_valid_mask_bbox_geometry(
                        valid_mask=valid_mask,
                        boxes_xyxy=boxes,
                        image_size=image_size,
                        min_bbox_width_px=min_bbox_width_px,
                        min_bbox_height_px=min_bbox_height_px,
                        min_bbox_margin_px=min_bbox_margin_px,
                    )

            P, T = int(valid_mask.shape[0]), int(valid_mask.shape[1])

            for pid in range(P):
                key = (int(sid), int(pid))
                if key not in sums:
                    continue

                c = counts[key]
                s = sums[key]

                ok = c > 0
                pred_norm_full = np.zeros((T, 3), dtype=np.float32)
                if np.any(ok):
                    pred_norm_full[ok] = (s[ok] / c[ok, None]).astype(np.float32)

                # Guard against non-finite values (can happen if the model outputs NaNs/Infs).
                pred_finite = np.isfinite(pred_norm_full).all(axis=-1)
                usable = ok & pred_finite
                if not bool(pred_finite.all()):
                    pred_norm_full[~pred_finite] = 0.0

                # Only evaluate frames where root is valid *and* covered by at least one window.
                vm = valid_mask[pid]
                eval_mask = vm & usable
                frames = np.flatnonzero(eval_mask)
                if frames.size == 0:
                    continue

                # MetricsAccumulator expects per-frame samples (B, ...)
                root_pred_t = torch.from_numpy(pred_norm_full[frames]).to(self.device, dtype=torch.float32)
                root_gt_t = torch.from_numpy(root_gt_norm[pid, frames]).to(self.device, dtype=torch.float32)

                x3d_t = torch.from_numpy(x3d_sam_norm[pid, frames]).to(self.device, dtype=torch.float32)

                vj_np = valid_joints[pid, frames]
                vj_np = vj_np & vm[frames][:, None]
                vj_t = torch.from_numpy(vj_np).to(self.device)

                K_t = torch.from_numpy(K_all[frames]).to(self.device, dtype=torch.float32)
                R_t = torch.from_numpy(R_all[frames]).to(self.device, dtype=torch.float32)
                t_t = torch.from_numpy(t_all[frames]).to(self.device, dtype=torch.float32)
                k_t = torch.from_numpy(k_all[frames]).to(self.device, dtype=torch.float32)

                Y_cam_t = torch.from_numpy(Y_cam_gt[pid, frames]).to(self.device, dtype=torch.float32)
                Y_2d_t = torch.from_numpy(Y_2d_gt[pid, frames]).to(self.device, dtype=torch.float32)

                batch_t: Dict[str, Any] = {
                    "root_gt": root_gt_t,
                    "x3d_sam_norm": x3d_t,
                    "valid_joints": vj_t,
                    "K": K_t,
                    "R": R_t,
                    "t": t_t,
                    "k": k_t,
                    "Y_cam_gt": Y_cam_t,
                    "Y_2d_gt": Y_2d_t,
                }
                metrics_acc.update(batch=batch_t, root_pred_norm=root_pred_t, stats=stats_dev)

                root_pred_m_t = stats_dev.denorm_root(root_pred_t)
                root_gt_m_t = stats_dev.denorm_root(root_gt_t)
                root_err_m_t = torch.linalg.norm(root_pred_m_t - root_gt_m_t, dim=-1)

                if diagnostics is not None:
                    n_frames = int(frames.size)
                    diagnostics.update(
                        seq_id=np.full((n_frames,), int(sid), dtype=np.int32),
                        person_idx=np.full((n_frames,), int(pid), dtype=np.int32),
                        frame_idx=frames.astype(np.int32, copy=False),
                        root_pred_norm=root_pred_t,
                        root_gt_norm=root_gt_t,
                        root_pred_m=root_pred_m_t,
                        root_gt_m=root_gt_m_t,
                        root_error_m=root_err_m_t,
                    )

                # ---- Temporal errors (meters) on full timeline (masked)
                pred_m_full = stats_dev.denorm_root(torch.from_numpy(pred_norm_full).to(self.device, dtype=torch.float32))
                gt_m_full = stats_dev.denorm_root(
                    torch.from_numpy(root_gt_norm[pid]).to(self.device, dtype=torch.float32)
                )

                vm_t = torch.from_numpy(vm.astype(np.bool_)).to(self.device)
                usable_t = torch.from_numpy(usable.astype(np.bool_)).to(self.device)
                cov_t = vm_t & usable_t

                if T >= 2:
                    vel_err = torch.linalg.norm(
                        (pred_m_full[1:] - pred_m_full[:-1]) - (gt_m_full[1:] - gt_m_full[:-1]),
                        dim=-1,
                    )  # (T-1,)
                    pair = cov_t[1:] & cov_t[:-1]
                    vals = vel_err[pair]
                    if vals.numel() > 0:
                        vel_err_sum += float(vals.sum().item())
                        vel_err_count += int(vals.numel())

                if T >= 3:
                    acc_err = torch.linalg.norm(
                        (pred_m_full[2:] - 2.0 * pred_m_full[1:-1] + pred_m_full[:-2])
                        - (gt_m_full[2:] - 2.0 * gt_m_full[1:-1] + gt_m_full[:-2]),
                        dim=-1,
                    )  # (T-2,)
                    tri = cov_t[2:] & cov_t[1:-1] & cov_t[:-2]
                    vals = acc_err[tri]
                    if vals.numel() > 0:
                        acc_err_sum += float(vals.sum().item())
                        acc_err_count += int(vals.numel())

                # ---- Valid losses in normalized space (aggregated)
                pred_norm_bt = torch.from_numpy(pred_norm_full).to(self.device, dtype=torch.float32).unsqueeze(0)  # (1,T,3)
                gt_norm_bt = torch.from_numpy(root_gt_norm[pid]).to(self.device, dtype=torch.float32).unsqueeze(0)
                vm_bt = vm_t.unsqueeze(0)
                cov_bt = cov_t.unsqueeze(0)

                # We compute sums/counts so overall averages are correct. Index
                # before the loss because invalid GT frames can contain NaNs.
                axis_sum, count = _masked_smooth_l1_axis_sum_and_count(
                    pred_norm_bt,
                    gt_norm_bt,
                    vm_bt & cov_bt,
                )
                valid_loss_root_axis += axis_sum
                denom_root += count

                # Velocity loss
                if T >= 2:
                    vel_pred = pred_norm_bt[:, 1:] - pred_norm_bt[:, :-1]
                    vel_gt = gt_norm_bt[:, 1:] - gt_norm_bt[:, :-1]
                    pair = (vm_bt[:, 1:] & vm_bt[:, :-1]) & (cov_bt[:, 1:] & cov_bt[:, :-1])
                    loss_sum, count = _masked_smooth_l1_sum_and_count(vel_pred, vel_gt, pair)
                    valid_loss_root_vel += loss_sum
                    denom_vel += count

                # Acc loss
                if T >= 3:
                    acc_pred = pred_norm_bt[:, 2:] - 2.0 * pred_norm_bt[:, 1:-1] + pred_norm_bt[:, :-2]
                    acc_gt = gt_norm_bt[:, 2:] - 2.0 * gt_norm_bt[:, 1:-1] + gt_norm_bt[:, :-2]
                    tri = (
                        (vm_bt[:, 2:] & vm_bt[:, 1:-1] & vm_bt[:, :-2])
                        & (cov_bt[:, 2:] & cov_bt[:, 1:-1] & cov_bt[:, :-2])
                    )
                    loss_sum, count = _masked_smooth_l1_sum_and_count(acc_pred, acc_gt, tri)
                    valid_loss_root_acc += loss_sum
                    denom_acc += count

                # ---- Save predictions (only for valid frames)
                if self.save_predictions_npz or self.save_predictions_csv:
                    # Save only valid frames that are covered by at least one window.
                    frames_save = frames

                    R_save = torch.from_numpy(R_all[frames_save]).to(self.device, dtype=torch.float32)
                    t_save = torch.from_numpy(t_all[frames_save]).to(self.device, dtype=torch.float32)

                    root_world_pred_t = cam_to_world(root_pred_m_t, R=R_save, t=t_save)
                    root_world_gt_t = cam_to_world(root_gt_m_t, R=R_save, t=t_save)

                    n = int(frames_save.size)
                    seq_id_chunks.append(np.full((n,), int(sid), dtype=np.int32))
                    person_chunks.append(np.full((n,), int(pid), dtype=np.int32))
                    frame_chunks.append(frames_save.astype(np.int32, copy=False))

                    root_pred_norm_chunks.append(_to_numpy(root_pred_t).astype(np.float32))
                    root_gt_norm_chunks.append(_to_numpy(root_gt_t).astype(np.float32))

                    root_pred_m_chunks.append(_to_numpy(root_pred_m_t).astype(np.float32))
                    root_gt_m_chunks.append(_to_numpy(root_gt_m_t).astype(np.float32))
                    root_err_m_chunks.append(_to_numpy(root_err_m_t).astype(np.float32))

                    root_world_pred_chunks.append(_to_numpy(root_world_pred_t).astype(np.float32))
                    root_world_gt_chunks.append(_to_numpy(root_world_gt_t).astype(np.float32))

        metrics = metrics_acc.compute()

        metrics["root_velocity_error_mean_m"] = (
            float(vel_err_sum / vel_err_count) if vel_err_count > 0 else float("nan")
        )
        metrics["root_acceleration_error_mean_m"] = (
            float(acc_err_sum / acc_err_count) if acc_err_count > 0 else float("nan")
        )

        root_axis = valid_loss_root_axis / max(1, denom_root)
        root_weights = np.ones((3,), dtype=np.float64)
        if root_axis_weights is not None:
            root_weights = np.asarray([float(v) for v in root_axis_weights], dtype=np.float64)
            if root_weights.shape != (3,):
                raise ValueError(f"root_axis_weights must contain exactly 3 values, got {root_axis_weights}")
            if np.any(root_weights < 0.0) or float(root_weights.sum()) <= 0.0:
                raise ValueError("root_axis_weights must be >= 0 and contain at least one positive value")

        metrics["valid_loss_root_x"] = float(root_axis[0])
        metrics["valid_loss_root_y"] = float(root_axis[1])
        metrics["valid_loss_root_z"] = float(root_axis[2])
        metrics["valid_loss_root"] = float(np.sum(root_axis * root_weights) / max(float(root_weights.sum()), 1e-12))
        metrics["valid_loss_root_vel"] = float(valid_loss_root_vel / max(1, denom_vel))
        metrics["valid_loss_root_acc"] = float(valid_loss_root_acc / max(1, denom_acc))

        # Provide also a total valid loss (unweighted). Training can compute weighted separately if needed.
        metrics["valid_loss_total"] = float(
            metrics["valid_loss_root"] + metrics["valid_loss_root_vel"] + metrics["valid_loss_root_acc"]
        )

        predictions_npz_path: Optional[Path] = None
        predictions_csv_path: Optional[Path] = None

        if self.save_predictions_npz:
            predictions_npz_path = out_dir / f"{split_name}_predictions.npz"

            seq_id = np.concatenate(seq_id_chunks, axis=0) if seq_id_chunks else np.zeros((0,), dtype=np.int32)
            person_idx = np.concatenate(person_chunks, axis=0) if person_chunks else np.zeros((0,), dtype=np.int32)
            frame_idx = np.concatenate(frame_chunks, axis=0) if frame_chunks else np.zeros((0,), dtype=np.int32)

            root_pred_norm_all = (
                np.concatenate(root_pred_norm_chunks, axis=0)
                if root_pred_norm_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            root_gt_norm_all = (
                np.concatenate(root_gt_norm_chunks, axis=0)
                if root_gt_norm_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )

            root_pred_m_all = (
                np.concatenate(root_pred_m_chunks, axis=0)
                if root_pred_m_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            root_gt_m_all = (
                np.concatenate(root_gt_m_chunks, axis=0)
                if root_gt_m_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            root_err_m_all = (
                np.concatenate(root_err_m_chunks, axis=0)
                if root_err_m_chunks
                else np.zeros((0,), dtype=np.float32)
            )

            root_world_pred_all = (
                np.concatenate(root_world_pred_chunks, axis=0)
                if root_world_pred_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            root_world_gt_all = (
                np.concatenate(root_world_gt_chunks, axis=0)
                if root_world_gt_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )

            np.savez_compressed(
                predictions_npz_path,
                seq_names=np.array(sequences, dtype=object),
                seq_id=seq_id,
                person_idx=person_idx,
                frame_idx=frame_idx,
                root_pred_norm=root_pred_norm_all,
                root_gt_norm=root_gt_norm_all,
                root_pred_m=root_pred_m_all,
                root_gt_m=root_gt_m_all,
                root_error_m=root_err_m_all,
                root_world_pred_m=root_world_pred_all,
                root_world_gt_m=root_world_gt_all,
                metrics=json.dumps(metrics),
                temporal_extras=json.dumps(
                    {
                        "num_frames_total": extras.num_frames_total,
                        "num_frames_covered": extras.num_frames_covered,
                        "num_frames_uncovered": extras.num_frames_uncovered,
                    }
                ),
            )

        if self.save_predictions_csv:
            predictions_csv_path = out_dir / f"{split_name}_predictions.csv"

            seq_id = np.concatenate(seq_id_chunks, axis=0) if seq_id_chunks else np.zeros((0,), dtype=np.int32)
            person_idx = np.concatenate(person_chunks, axis=0) if person_chunks else np.zeros((0,), dtype=np.int32)
            frame_idx = np.concatenate(frame_chunks, axis=0) if frame_chunks else np.zeros((0,), dtype=np.int32)

            root_pred_m_all = (
                np.concatenate(root_pred_m_chunks, axis=0)
                if root_pred_m_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            root_gt_m_all = (
                np.concatenate(root_gt_m_chunks, axis=0)
                if root_gt_m_chunks
                else np.zeros((0, 3), dtype=np.float32)
            )
            root_err_m_all = (
                np.concatenate(root_err_m_chunks, axis=0)
                if root_err_m_chunks
                else np.zeros((0,), dtype=np.float32)
            )

            with predictions_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "seq_id",
                        "seq_name",
                        "person_idx",
                        "frame_idx",
                        "root_pred_x_m",
                        "root_pred_y_m",
                        "root_pred_z_m",
                        "root_gt_x_m",
                        "root_gt_y_m",
                        "root_gt_z_m",
                        "root_error_m",
                    ]
                )

                for i in range(int(root_err_m_all.shape[0])):
                    sid_i = int(seq_id[i])
                    sname = sequences[sid_i] if 0 <= sid_i < len(sequences) else ""
                    writer.writerow(
                        [
                            sid_i,
                            sname,
                            int(person_idx[i]),
                            int(frame_idx[i]),
                            float(root_pred_m_all[i, 0]),
                            float(root_pred_m_all[i, 1]),
                            float(root_pred_m_all[i, 2]),
                            float(root_gt_m_all[i, 0]),
                            float(root_gt_m_all[i, 1]),
                            float(root_gt_m_all[i, 2]),
                            float(root_err_m_all[i]),
                        ]
                    )

        if diagnostics is not None and diagnostics_dir is not None:
            diagnostics.write(
                out_dir=diagnostics_dir,
                prefix=diagnostics_prefix or split_name,
            )

        return (
            EvalOutputs(metrics=metrics, predictions_npz_path=predictions_npz_path, predictions_csv_path=predictions_csv_path),
            extras,
        )
