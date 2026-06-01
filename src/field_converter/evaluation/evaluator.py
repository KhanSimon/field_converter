from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np
import torch
from torch import nn

from field_converter.evaluation.diagnostics import PredictionDiagnostics
from field_converter.evaluation.metrics import MetricsAccumulator
from field_converter.geometry.transforms import cam_to_world
from field_converter.utils.normalization import TorchNormalizationStats


class RootModel(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # (B,D) -> (B,3)
        ...


@dataclass(frozen=True)
class EvalOutputs:
    metrics: Dict[str, float]
    predictions_npz_path: Optional[Path] = None
    predictions_csv_path: Optional[Path] = None


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class Evaluator:
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
        model: RootModel,
        dataloader: torch.utils.data.DataLoader,
        out_dir: Path,
        split_name: str,
        diagnostics_dir: Optional[Path] = None,
        diagnostics_prefix: Optional[str] = None,
        diagnostics_top_k: int = 100,
    ) -> EvalOutputs:
        """Evaluate a model/baseline on one split and optionally save predictions."""
        _ensure_dir(out_dir)

        # If the model is an nn.Module, ensure device + eval mode.
        if isinstance(model, nn.Module):
            model.to(self.device)
            model.eval()

        model_device = self.device
        stats = self.stats.to(model_device)

        metrics_acc = MetricsAccumulator()

        # Collect predictions for saving.
        seq_names: List[str] = getattr(dataloader.dataset, "sequences", [])
        seq_to_id = {s: i for i, s in enumerate(seq_names)}
        diagnostics = (
            PredictionDiagnostics(seq_names=seq_names, top_k=diagnostics_top_k)
            if diagnostics_dir is not None
            else None
        )

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

        for batch in dataloader:
            batch_dev = _move_to_device(batch, model_device)

            x = batch_dev["x"].to(dtype=torch.float32)
            root_gt_norm = batch_dev["root_gt"].to(dtype=torch.float32)

            root_pred_norm = model(x).to(dtype=torch.float32)

            metrics_acc.update(batch=batch_dev, root_pred_norm=root_pred_norm, stats=stats)

            # ---- Save predictions (denormalized)
            root_pred_m = stats.denorm_root(root_pred_norm)
            root_gt_m = stats.denorm_root(root_gt_norm)

            root_err_m = torch.linalg.norm(root_pred_m - root_gt_m, dim=-1)

            R = batch_dev["R"].to(dtype=torch.float32)
            t = batch_dev["t"].to(dtype=torch.float32)
            root_world_pred = cam_to_world(root_pred_m, R=R, t=t)
            root_world_gt = cam_to_world(root_gt_m, R=R, t=t)

            if "seq_name" in batch:
                seq_ids = np.array([seq_to_id.get(s, -1) for s in batch["seq_name"]], dtype=np.int32)
            else:
                seq_ids = np.full((x.shape[0],), -1, dtype=np.int32)

            if diagnostics is not None:
                diagnostics.update(
                    seq_id=seq_ids,
                    person_idx=batch_dev["person_idx"],
                    frame_idx=batch_dev["frame_idx"],
                    root_pred_norm=root_pred_norm,
                    root_gt_norm=root_gt_norm,
                    root_pred_m=root_pred_m,
                    root_gt_m=root_gt_m,
                    root_error_m=root_err_m,
                )

            if self.save_predictions_npz or self.save_predictions_csv:
                seq_id_chunks.append(seq_ids)
                person_chunks.append(_to_numpy(batch_dev["person_idx"]).astype(np.int32))
                frame_chunks.append(_to_numpy(batch_dev["frame_idx"]).astype(np.int32))

                root_pred_norm_chunks.append(_to_numpy(root_pred_norm).astype(np.float32))
                root_gt_norm_chunks.append(_to_numpy(root_gt_norm).astype(np.float32))

                root_pred_m_chunks.append(_to_numpy(root_pred_m).astype(np.float32))
                root_gt_m_chunks.append(_to_numpy(root_gt_m).astype(np.float32))
                root_err_m_chunks.append(_to_numpy(root_err_m).astype(np.float32))

                root_world_pred_chunks.append(_to_numpy(root_world_pred).astype(np.float32))
                root_world_gt_chunks.append(_to_numpy(root_world_gt).astype(np.float32))

        metrics = metrics_acc.compute()

        predictions_npz_path: Optional[Path] = None
        predictions_csv_path: Optional[Path] = None

        if self.save_predictions_npz:
            predictions_npz_path = out_dir / f"{split_name}_predictions.npz"

            seq_ids = np.concatenate(seq_id_chunks, axis=0) if seq_id_chunks else np.zeros((0,), dtype=np.int32)
            person_idx = np.concatenate(person_chunks, axis=0) if person_chunks else np.zeros((0,), dtype=np.int32)
            frame_idx = np.concatenate(frame_chunks, axis=0) if frame_chunks else np.zeros((0,), dtype=np.int32)

            root_pred_norm_all = (
                np.concatenate(root_pred_norm_chunks, axis=0) if root_pred_norm_chunks else np.zeros((0, 3), dtype=np.float32)
            )
            root_gt_norm_all = (
                np.concatenate(root_gt_norm_chunks, axis=0) if root_gt_norm_chunks else np.zeros((0, 3), dtype=np.float32)
            )

            root_pred_m_all = np.concatenate(root_pred_m_chunks, axis=0) if root_pred_m_chunks else np.zeros((0, 3), dtype=np.float32)
            root_gt_m_all = np.concatenate(root_gt_m_chunks, axis=0) if root_gt_m_chunks else np.zeros((0, 3), dtype=np.float32)
            root_err_m_all = np.concatenate(root_err_m_chunks, axis=0) if root_err_m_chunks else np.zeros((0,), dtype=np.float32)

            root_world_pred_all = (
                np.concatenate(root_world_pred_chunks, axis=0) if root_world_pred_chunks else np.zeros((0, 3), dtype=np.float32)
            )
            root_world_gt_all = (
                np.concatenate(root_world_gt_chunks, axis=0) if root_world_gt_chunks else np.zeros((0, 3), dtype=np.float32)
            )

            np.savez_compressed(
                predictions_npz_path,
                seq_names=np.array(seq_names, dtype=object),
                seq_id=seq_ids,
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
            )

        if self.save_predictions_csv:
            predictions_csv_path = out_dir / f"{split_name}_predictions.csv"
            seq_ids = np.concatenate(seq_id_chunks, axis=0) if seq_id_chunks else np.zeros((0,), dtype=np.int32)
            person_idx = np.concatenate(person_chunks, axis=0) if person_chunks else np.zeros((0,), dtype=np.int32)
            frame_idx = np.concatenate(frame_chunks, axis=0) if frame_chunks else np.zeros((0,), dtype=np.int32)
            root_pred_m_all = np.concatenate(root_pred_m_chunks, axis=0) if root_pred_m_chunks else np.zeros((0, 3), dtype=np.float32)
            root_gt_m_all = np.concatenate(root_gt_m_chunks, axis=0) if root_gt_m_chunks else np.zeros((0, 3), dtype=np.float32)
            root_err_m_all = np.concatenate(root_err_m_chunks, axis=0) if root_err_m_chunks else np.zeros((0,), dtype=np.float32)

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
                    sid = int(seq_ids[i])
                    sname = seq_names[sid] if 0 <= sid < len(seq_names) else ""
                    writer.writerow(
                        [
                            sid,
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

        return EvalOutputs(metrics=metrics, predictions_npz_path=predictions_npz_path, predictions_csv_path=predictions_csv_path)
