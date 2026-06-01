from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from field_converter.utils.io import ensure_dir
from field_converter.utils.normalization import TorchNormalizationStats


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _join_unique(values: Iterable[Any], *, max_items: int = 12) -> str:
    seen: list[str] = []
    for value in values:
        s = str(value)
        if s not in seen:
            seen.append(s)
        if len(seen) >= max_items:
            break
    return "|".join(seen)


class RunningArrayStats:
    def __init__(self) -> None:
        self.count = 0
        self.nonfinite_count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.abs_max = 0.0

    def update(self, values: torch.Tensor | np.ndarray) -> None:
        arr = np.asarray(_to_numpy(values), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return
        finite = np.isfinite(arr)
        self.nonfinite_count += int((~finite).sum())
        arr = arr[finite]
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.sum += float(arr.sum())
        self.sum_sq += float(np.square(arr).sum())
        self.min = min(self.min, float(arr.min()))
        self.max = max(self.max, float(arr.max()))
        self.abs_max = max(self.abs_max, float(np.abs(arr).max()))

    def as_dict(self) -> Dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "nonfinite_count": self.nonfinite_count,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "abs_max": float("nan"),
            }
        mean = self.sum / self.count
        var = max(0.0, self.sum_sq / self.count - mean * mean)
        return {
            "count": self.count,
            "nonfinite_count": self.nonfinite_count,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min,
            "max": self.max,
            "abs_max": self.abs_max,
        }


class PredictionDiagnostics:
    """Collect validation diagnostics without saving full prediction tensors."""

    def __init__(self, *, seq_names: Iterable[str], top_k: int = 100) -> None:
        self.seq_names = list(seq_names)
        self.top_k = int(top_k)
        self.root_pred_norm = RunningArrayStats()
        self.root_pred_m = RunningArrayStats()
        self.root_error_m = RunningArrayStats()
        self._top_rows: list[dict[str, Any]] = []
        self._seq_stats: dict[int, dict[str, Any]] = {}
        self._seq_person_stats: dict[tuple[int, int], dict[str, Any]] = {}

    def update(
        self,
        *,
        seq_id: torch.Tensor | np.ndarray,
        person_idx: torch.Tensor | np.ndarray,
        frame_idx: torch.Tensor | np.ndarray,
        root_pred_norm: torch.Tensor | np.ndarray,
        root_gt_norm: torch.Tensor | np.ndarray,
        root_pred_m: torch.Tensor | np.ndarray,
        root_gt_m: torch.Tensor | np.ndarray,
        root_error_m: torch.Tensor | np.ndarray,
    ) -> None:
        seq = np.asarray(_to_numpy(seq_id)).reshape(-1).astype(np.int64, copy=False)
        person = np.asarray(_to_numpy(person_idx)).reshape(-1).astype(np.int64, copy=False)
        frame = np.asarray(_to_numpy(frame_idx)).reshape(-1).astype(np.int64, copy=False)
        pred_norm = np.asarray(_to_numpy(root_pred_norm), dtype=np.float64).reshape(-1, 3)
        gt_norm = np.asarray(_to_numpy(root_gt_norm), dtype=np.float64).reshape(-1, 3)
        pred_m = np.asarray(_to_numpy(root_pred_m), dtype=np.float64).reshape(-1, 3)
        gt_m = np.asarray(_to_numpy(root_gt_m), dtype=np.float64).reshape(-1, 3)
        err = np.asarray(_to_numpy(root_error_m), dtype=np.float64).reshape(-1)

        n = min(seq.size, person.size, frame.size, pred_norm.shape[0], gt_norm.shape[0], pred_m.shape[0], gt_m.shape[0], err.size)
        if n <= 0:
            return

        seq = seq[:n]
        person = person[:n]
        frame = frame[:n]
        pred_norm = pred_norm[:n]
        gt_norm = gt_norm[:n]
        pred_m = pred_m[:n]
        gt_m = gt_m[:n]
        err = err[:n]

        self.root_pred_norm.update(pred_norm)
        self.root_pred_m.update(pred_m)
        self.root_error_m.update(err)

        finite_err = np.isfinite(err)
        for sid in np.unique(seq):
            mask = (seq == sid) & finite_err
            if not np.any(mask):
                continue
            stat = self._seq_stats.setdefault(int(sid), {"sum": 0.0, "count": 0, "max": float("-inf")})
            vals = err[mask]
            stat["sum"] += float(vals.sum())
            stat["count"] += int(vals.size)
            stat["max"] = max(float(stat["max"]), float(vals.max()))

        for sid, pid in set(zip(seq.tolist(), person.tolist())):
            mask = (seq == sid) & (person == pid) & finite_err
            if not np.any(mask):
                continue
            stat = self._seq_person_stats.setdefault((int(sid), int(pid)), {"sum": 0.0, "count": 0, "max": float("-inf")})
            vals = err[mask]
            stat["sum"] += float(vals.sum())
            stat["count"] += int(vals.size)
            stat["max"] = max(float(stat["max"]), float(vals.max()))

        score = np.where(np.isfinite(err), err, np.inf)
        k = min(self.top_k, int(score.size))
        if k <= 0:
            return
        idx = np.argpartition(-score, kth=k - 1)[:k]
        for i in idx:
            sid = int(seq[i])
            self._top_rows.append(
                {
                    "score": float(score[i]),
                    "seq_id": sid,
                    "seq_name": self.seq_names[sid] if 0 <= sid < len(self.seq_names) else "",
                    "person_idx": int(person[i]),
                    "frame_idx": int(frame[i]),
                    "root_error_m": float(err[i]),
                    "root_pred_norm_x": float(pred_norm[i, 0]),
                    "root_pred_norm_y": float(pred_norm[i, 1]),
                    "root_pred_norm_z": float(pred_norm[i, 2]),
                    "root_gt_norm_x": float(gt_norm[i, 0]),
                    "root_gt_norm_y": float(gt_norm[i, 1]),
                    "root_gt_norm_z": float(gt_norm[i, 2]),
                    "root_pred_x_m": float(pred_m[i, 0]),
                    "root_pred_y_m": float(pred_m[i, 1]),
                    "root_pred_z_m": float(pred_m[i, 2]),
                    "root_gt_x_m": float(gt_m[i, 0]),
                    "root_gt_y_m": float(gt_m[i, 1]),
                    "root_gt_z_m": float(gt_m[i, 2]),
                }
            )
        self._top_rows.sort(key=lambda r: _safe_float(r["score"]), reverse=True)
        del self._top_rows[self.top_k :]

    def write(self, *, out_dir: Path, prefix: str) -> None:
        ensure_dir(out_dir)

        stats_path = out_dir / f"{prefix}_prediction_stats.json"
        payload = {
            "root_pred_norm": self.root_pred_norm.as_dict(),
            "root_pred_m": self.root_pred_m.as_dict(),
            "root_error_m": self.root_error_m.as_dict(),
        }
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        top_path = out_dir / f"{prefix}_top_errors.csv"
        fieldnames = [
            "seq_id",
            "seq_name",
            "person_idx",
            "frame_idx",
            "root_error_m",
            "root_pred_norm_x",
            "root_pred_norm_y",
            "root_pred_norm_z",
            "root_gt_norm_x",
            "root_gt_norm_y",
            "root_gt_norm_z",
            "root_pred_x_m",
            "root_pred_y_m",
            "root_pred_z_m",
            "root_gt_x_m",
            "root_gt_y_m",
            "root_gt_z_m",
        ]
        with top_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._top_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        self._write_group_stats(out_dir / f"{prefix}_per_sequence.csv", self._seq_stats)
        self._write_group_stats(out_dir / f"{prefix}_per_sequence_person.csv", self._seq_person_stats)

    def _write_group_stats(self, path: Path, stats: dict[Any, dict[str, Any]]) -> None:
        fieldnames = ["seq_id", "seq_name", "person_idx", "count", "mean_root_error_m", "max_root_error_m"]
        rows: list[dict[str, Any]] = []
        for key, stat in stats.items():
            if isinstance(key, tuple):
                sid, pid = int(key[0]), int(key[1])
            else:
                sid, pid = int(key), ""
            count = int(stat["count"])
            if count <= 0:
                continue
            rows.append(
                {
                    "seq_id": sid,
                    "seq_name": self.seq_names[sid] if 0 <= sid < len(self.seq_names) else "",
                    "person_idx": pid,
                    "count": count,
                    "mean_root_error_m": float(stat["sum"]) / count,
                    "max_root_error_m": float(stat["max"]),
                }
            )
        rows.sort(key=lambda r: _safe_float(r["mean_root_error_m"]), reverse=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


TRAIN_BATCH_DIAGNOSTIC_COLUMNS = [
    "epoch",
    "batch_idx",
    "loss_total",
    "loss_root",
    "loss_root_x",
    "loss_root_y",
    "loss_root_z",
    "loss_root_vel",
    "loss_root_acc",
    "loss_cam3d",
    "loss_proj",
    "grad_norm",
    "root_error_m_mean",
    "root_error_m_max",
    "root_pred_norm_abs_max",
    "root_pred_m_abs_max",
    "num_valid_steps",
    "seq_names",
    "person_idx_min",
    "person_idx_max",
    "frame_idx_min",
    "frame_idx_max",
]


def build_train_batch_diagnostic(
    *,
    epoch: int,
    batch_idx: int,
    batch: dict[str, Any],
    losses: dict[str, Any],
    root_pred_norm: torch.Tensor,
    root_gt_norm: torch.Tensor,
    stats: TorchNormalizationStats,
    grad_norm: Optional[float],
) -> dict[str, Any]:
    pred = root_pred_norm.detach()
    gt = root_gt_norm.detach()
    valid_mask = batch.get("valid_mask")
    if isinstance(valid_mask, torch.Tensor) and valid_mask.shape == pred.shape[:-1]:
        mask = valid_mask.detach().bool()
    else:
        mask = torch.ones(pred.shape[:-1], dtype=torch.bool, device=pred.device)
    finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(gt).all(dim=-1)
    mask = mask & finite

    pred_m = stats.denorm_root(pred)
    gt_m = stats.denorm_root(gt)
    err = torch.linalg.norm(pred_m - gt_m, dim=-1)
    err_valid = err[mask]

    person = batch.get("person_idx")
    if isinstance(person, torch.Tensor) and person.numel() > 0:
        person_np = _to_numpy(person).reshape(-1)
        person_min: Any = int(np.min(person_np))
        person_max: Any = int(np.max(person_np))
    else:
        person_min = ""
        person_max = ""

    frame_values: np.ndarray
    if isinstance(batch.get("frame_idx"), torch.Tensor):
        frame_values = _to_numpy(batch["frame_idx"]).reshape(-1)
    elif isinstance(batch.get("frame_indices"), torch.Tensor):
        frame_values = _to_numpy(batch["frame_indices"]).reshape(-1)
        frame_values = frame_values[frame_values >= 0]
    else:
        frame_values = np.zeros((0,), dtype=np.int64)

    seq_names = batch.get("seq_name", [])
    if isinstance(seq_names, str):
        seq_names = [seq_names]

    return {
        "epoch": int(epoch),
        "batch_idx": int(batch_idx),
        "loss_total": _safe_float(losses.get("loss_total")),
        "loss_root": _safe_float(losses.get("loss_root")),
        "loss_root_vel": _safe_float(losses.get("loss_root_vel")),
        "loss_root_acc": _safe_float(losses.get("loss_root_acc")),
        "loss_cam3d": _safe_float(losses.get("loss_cam3d")),
        "loss_proj": _safe_float(losses.get("loss_proj")),
        "grad_norm": float(grad_norm) if grad_norm is not None else "",
        "root_error_m_mean": float(err_valid.mean().item()) if err_valid.numel() > 0 else float("nan"),
        "root_error_m_max": float(err_valid.max().item()) if err_valid.numel() > 0 else float("nan"),
        "root_pred_norm_abs_max": float(pred.detach().abs().max().item()) if pred.numel() > 0 else float("nan"),
        "root_pred_m_abs_max": float(pred_m.detach().abs().max().item()) if pred_m.numel() > 0 else float("nan"),
        "num_valid_steps": int(mask.sum().item()),
        "seq_names": _join_unique(seq_names),
        "person_idx_min": person_min,
        "person_idx_max": person_max,
        "frame_idx_min": int(frame_values.min()) if frame_values.size > 0 else "",
        "frame_idx_max": int(frame_values.max()) if frame_values.size > 0 else "",
    }


def append_train_batch_diagnostic(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_BATCH_DIAGNOSTIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TRAIN_BATCH_DIAGNOSTIC_COLUMNS})


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.detach()
        if grad.numel() == 0:
            continue
        norm = float(torch.linalg.norm(grad).item())
        if math.isfinite(norm):
            total_sq += norm * norm
    return math.sqrt(total_sq)
