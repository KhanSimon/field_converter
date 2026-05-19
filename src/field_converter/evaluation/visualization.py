from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib

# Use non-interactive backend for cluster runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from field_converter.geometry.projection import project_cam_to_image
from field_converter.utils.normalization import TorchNormalizationStats


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_predictions_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        payload = {k: npz[k] for k in npz.files}
    if "metrics" in payload and isinstance(payload["metrics"], np.ndarray) and payload["metrics"].shape == ():
        try:
            payload["metrics"] = json.loads(str(payload["metrics"].item()))
        except Exception:
            pass
    return payload


def _pick_seq_person(pred: Dict[str, Any], seq_name: Optional[str], person_idx: Optional[int]) -> Tuple[int, int, str]:
    seq_names = list(pred.get("seq_names", []))
    seq_id_arr = np.asarray(pred.get("seq_id"), dtype=np.int32)
    person_arr = np.asarray(pred.get("person_idx"), dtype=np.int32)

    if seq_name is not None:
        try:
            seq_id = seq_names.index(seq_name)
        except ValueError:
            raise ValueError(f"seq_name {seq_name!r} not found in predictions")
    else:
        # Most frequent sequence
        if seq_id_arr.size == 0:
            raise ValueError("Empty predictions")
        seq_id = int(np.bincount(seq_id_arr[seq_id_arr >= 0]).argmax())
        seq_name = str(seq_names[seq_id]) if 0 <= seq_id < len(seq_names) else ""

    if person_idx is None:
        mask = seq_id_arr == seq_id
        if not mask.any():
            raise ValueError("No samples for chosen sequence")
        person_vals = person_arr[mask]
        person_idx = int(np.bincount(person_vals).argmax())

    return int(seq_id), int(person_idx), str(seq_name)


def plot_root_timeseries(
    *,
    predictions_npz: Path,
    out_path: Path,
    seq_name: Optional[str] = None,
    person_idx: Optional[int] = None,
) -> None:
    pred = load_predictions_npz(predictions_npz)
    seq_id, pid, seq_name = _pick_seq_person(pred, seq_name, person_idx)

    seq_id_arr = np.asarray(pred["seq_id"], dtype=np.int32)
    person_arr = np.asarray(pred["person_idx"], dtype=np.int32)
    frame_arr = np.asarray(pred["frame_idx"], dtype=np.int32)

    root_pred_m = np.asarray(pred["root_pred_m"], dtype=np.float32)
    root_gt_m = np.asarray(pred["root_gt_m"], dtype=np.float32)

    mask = (seq_id_arr == seq_id) & (person_arr == pid)
    if not mask.any():
        raise ValueError(f"No samples for {seq_name} person={pid}")

    order = np.argsort(frame_arr[mask])
    frames = frame_arr[mask][order]
    rp = root_pred_m[mask][order]
    rg = root_gt_m[mask][order]

    _ensure_dir(out_path.parent)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(frames, rg[:, i], label="gt", linewidth=2)
        ax.plot(frames, rp[:, i], label="pred", linewidth=1)
        ax.set_ylabel(f"root_{labels[i]} (m)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    axes[0].set_title(f"Root components — {seq_name} person={pid}")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_world_trajectory_xy(
    *,
    predictions_npz: Path,
    out_path: Path,
    seq_name: Optional[str] = None,
    person_idx: Optional[int] = None,
) -> None:
    pred = load_predictions_npz(predictions_npz)
    seq_id, pid, seq_name = _pick_seq_person(pred, seq_name, person_idx)

    seq_id_arr = np.asarray(pred["seq_id"], dtype=np.int32)
    person_arr = np.asarray(pred["person_idx"], dtype=np.int32)
    frame_arr = np.asarray(pred["frame_idx"], dtype=np.int32)

    wp = np.asarray(pred["root_world_pred_m"], dtype=np.float32)
    wg = np.asarray(pred["root_world_gt_m"], dtype=np.float32)

    mask = (seq_id_arr == seq_id) & (person_arr == pid)
    if not mask.any():
        raise ValueError(f"No samples for {seq_name} person={pid}")

    order = np.argsort(frame_arr[mask])
    wp = wp[mask][order]
    wg = wg[mask][order]

    _ensure_dir(out_path.parent)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(wg[:, 0], wg[:, 1], label="gt", linewidth=2)
    ax.plot(wp[:, 0], wp[:, 1], label="pred", linewidth=1)
    ax.set_title(f"World root trajectory (X-Y) — {seq_name} person={pid}")
    ax.set_xlabel("X_world (m)")
    ax.set_ylabel("Y_world (m)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(
    *,
    train_log_csv: Path,
    out_path: Path,
) -> None:
    if not train_log_csv.exists():
        return

    rows = []
    with train_log_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    def col(name: str) -> np.ndarray:
        return np.array([float(r.get(name, "nan")) for r in rows], dtype=np.float64)

    epochs = np.array([int(r.get("epoch", 0)) for r in rows], dtype=np.int32)
    train_loss = col("train_loss_total")
    valid_loss = col("valid_loss_total")
    valid_root_err = col("valid_root_error_mean_m")

    _ensure_dir(out_path.parent)
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    ax1.plot(epochs, train_loss, label="train_loss")
    ax1.plot(epochs, valid_loss, label="valid_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(epochs, valid_root_err, label="valid_root_error_mean_m", color="tab:red")
    ax2.set_ylabel("root error (m)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_vs_baseline(
    *,
    model_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    out_path: Path,
    title: str = "Model vs Baseline (mean-root)",
) -> None:
    keys = [
        ("root_error_mean_m", "Root mean (m)"),
        ("MPJPE_cam_m", "MPJPE cam (m)"),
        ("MPJPE_world_m", "MPJPE world (m)"),
    ]

    model_vals = [float(model_metrics.get(k, float("nan"))) for k, _ in keys]
    base_vals = [float(baseline_metrics.get(k, float("nan"))) for k, _ in keys]
    labels = [lab for _, lab in keys]

    x = np.arange(len(labels))
    width = 0.38

    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(x - width / 2, base_vals, width, label="baseline")
    ax.bar(x + width / 2, model_vals, width, label="model")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reprojection_overlay(
    *,
    data_dir: Path,
    split: str,
    predictions_npz: Path,
    stats: TorchNormalizationStats,
    out_path: Path,
    seq_name: Optional[str] = None,
    person_idx: Optional[int] = None,
    num_frames: int = 6,
) -> None:
    """Overlay: GT 2D vs SAM 2D vs predicted reprojection on a few frames."""

    pred = load_predictions_npz(predictions_npz)
    seq_id, pid, seq_name = _pick_seq_person(pred, seq_name, person_idx)

    seq_id_arr = np.asarray(pred["seq_id"], dtype=np.int32)
    person_arr = np.asarray(pred["person_idx"], dtype=np.int32)
    frame_arr = np.asarray(pred["frame_idx"], dtype=np.int32)
    root_pred_m = np.asarray(pred["root_pred_m"], dtype=np.float32)

    mask = (seq_id_arr == seq_id) & (person_arr == pid)
    if not mask.any():
        raise ValueError(f"No samples for {seq_name} person={pid}")

    frames = frame_arr[mask]
    roots = root_pred_m[mask]

    # Pick evenly-spaced frames.
    order = np.argsort(frames)
    frames = frames[order]
    roots = roots[order]

    if frames.size == 0:
        return

    num_frames = int(max(1, num_frames))
    pick = np.linspace(0, frames.size - 1, num=min(num_frames, frames.size), dtype=int)

    # Load raw normalized sequence file.
    seq_path = Path(data_dir) / split / f"{seq_name}.npz"
    with np.load(seq_path, allow_pickle=True) as npz:
        image_size = np.asarray(npz["image_size"], dtype=np.float32)  # (2,) [W,H]
        W, H = float(image_size[0]), float(image_size[1])

        x2d_sam_img_norm = np.asarray(npz["skel_2d_sam3dbody_from_bbox_gt"][pid], dtype=np.float32)  # (T,25,2)
        y2d_gt = np.asarray(npz["Y_2d_gt"][pid], dtype=np.float32)  # (T,25,2)
        x3d_sam_norm = np.asarray(npz["skel_3d_sam3dbody_from_bbox_gt"][pid], dtype=np.float32)  # (T,25,3)
        K_all = np.asarray(npz["K"], dtype=np.float32)  # (T,3,3)
        k_all = np.asarray(npz["k"], dtype=np.float32)  # (T,2)
        valid_joints_all = np.asarray(npz["valid_joints"][pid], dtype=bool)  # (T,25)

    _ensure_dir(out_path.parent)

    cols = len(pick)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), squeeze=False)

    stats_cpu = stats.to("cpu")

    for j, idx0 in enumerate(pick):
        fr = int(frames[idx0])
        root_fr = roots[idx0]  # (3,) meters

        sam2d_px = x2d_sam_img_norm[fr] * np.array([W, H], dtype=np.float32)
        gt2d_px = y2d_gt[fr]

        X_rel_norm = torch.from_numpy(x3d_sam_norm[fr]).unsqueeze(0)  # (1,25,3)
        X_rel_m = stats_cpu.denorm_sam3d_rel(X_rel_norm).squeeze(0).numpy()

        X_cam_pred = X_rel_m + root_fr[None, :]
        uv_pred = project_cam_to_image(
            torch.from_numpy(X_cam_pred).unsqueeze(0),
            K=torch.from_numpy(K_all[fr]).unsqueeze(0),
            k=torch.from_numpy(k_all[fr]).unsqueeze(0),
        ).squeeze(0).numpy()

        vj = valid_joints_all[fr]

        ax = axes[0, j]
        ax.scatter(gt2d_px[vj, 0], gt2d_px[vj, 1], s=1, label="GT 2D")
        ax.scatter(sam2d_px[vj, 0], sam2d_px[vj, 1], s=1, label="SAM 2D")
        ax.scatter(uv_pred[vj, 0], uv_pred[vj, 1], s=1, label="Pred reproj")

        ax.set_title(f"frame {fr}")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # invert y
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"2D overlay — {seq_name} person={pid}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
