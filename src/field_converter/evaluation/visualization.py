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

    def _sniff_dialect(sample: str) -> csv.Dialect:
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t")
        except Exception:
            return csv.excel

    def _clean_key(k: Any) -> str:
        if k is None:
            return ""
        return str(k).strip().lstrip("\ufeff")

    def _clean_val(v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()

    def _looks_like_header(cells: list[str]) -> bool:
        if not cells:
            return False
        for c in cells:
            c = c.strip().lstrip("\ufeff")
            if c == "":
                continue
            # If there's any alphabetic character or underscore, it's almost certainly a header.
            if any(ch.isalpha() for ch in c) or ("_" in c):
                return True
        return False

    # Headerless logs exist in older runs, so keep schemas for all supported
    # train_log layouts instead of assuming only the newest one.
    mlp_header = [
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
    tcn_header = [
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
    legacy_mlp_header = [
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
    legacy_tcn_header = [
        "epoch",
        "train_loss_total",
        "train_loss_root",
        "train_loss_root_vel",
        "train_loss_root_acc",
        "train_loss_cam3d",
        "train_loss_proj",
        "valid_loss_total",
        "valid_loss_root",
        "valid_loss_root_vel",
        "valid_loss_root_acc",
        "valid_root_error_mean_m",
        "valid_MPJPE_cam_m",
        "valid_MPJPE_world_m",
        "valid_reprojection_error_mean_px",
        "valid_root_velocity_error_mean_m",
        "valid_root_acceleration_error_mean_m",
    ]
    header_by_len = {
        len(legacy_mlp_header): legacy_mlp_header,
        len(legacy_tcn_header): legacy_tcn_header,
        len(mlp_header): mlp_header,
        len(tcn_header): tcn_header,
    }

    rows: list[dict[str, str]] = []
    with train_log_csv.open("r", newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = _sniff_dialect(sample)
        reader = csv.reader(f, dialect=dialect)

        all_rows: list[list[str]] = []
        for r in reader:
            if not r:
                continue
            all_rows.append([_clean_val(v) for v in r])

    if not all_rows:
        return

    first = all_rows[0]
    if _looks_like_header(first):
        header = [_clean_key(c) for c in first]
        data_rows = all_rows[1:]
    else:
        # Headerless CSV: map by position if it matches the known schema.
        header = header_by_len.get(len(first), [f"col_{i}" for i in range(len(first))])
        data_rows = all_rows

    for r in data_rows:
        # Pad/truncate rows to header length.
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[: len(header)]
        rows.append({header[i]: r[i] for i in range(len(header))})

    if not rows:
        return

    available_cols = set().union(*(r.keys() for r in rows))
    lower_to_actual = {c.lower(): c for c in available_cols}

    def _pick_col(*names: str) -> str:
        for name in names:
            if name in available_cols:
                return name
            lowered = name.lower()
            if lowered in lower_to_actual:
                return lower_to_actual[lowered]
        return names[0]

    def _safe_float(s: str) -> float:
        s = s.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return float("nan")
        try:
            return float(s)
        except Exception:
            # Handle strings like 'tensor(0.123)'
            if "(" in s and s.endswith(")"):
                inner = s[s.find("(") + 1 : -1].strip()
                try:
                    return float(inner)
                except Exception:
                    return float("nan")
            return float("nan")

    def _safe_int(s: str) -> int:
        s = s.strip()
        if s == "":
            return 0
        try:
            return int(s)
        except Exception:
            try:
                return int(float(s))
            except Exception:
                return 0

    epoch_col = _pick_col("epoch", "Epoch", "step", "Step")
    train_loss_col = _pick_col("train_loss_total", "train_loss")
    valid_loss_col = _pick_col("valid_loss_total", "valid_loss")
    valid_root_err_col = _pick_col("valid_root_error_mean_m", "valid_root_error_mean_meters")

    def _maybe_col(name: str) -> Optional[str]:
        if name in available_cols:
            return name
        lowered = name.lower()
        if lowered in lower_to_actual:
            return lower_to_actual[lowered]
        return None

    extra_cols = {
        "train_loss_root": _maybe_col("train_loss_root"),
        "valid_loss_root": _maybe_col("valid_loss_root"),
        "train_loss_root_x": _maybe_col("train_loss_root_x"),
        "train_loss_root_y": _maybe_col("train_loss_root_y"),
        "train_loss_root_z": _maybe_col("train_loss_root_z"),
        "valid_loss_root_x": _maybe_col("valid_loss_root_x"),
        "valid_loss_root_y": _maybe_col("valid_loss_root_y"),
        "valid_loss_root_z": _maybe_col("valid_loss_root_z"),
        "train_loss_root_vel": _maybe_col("train_loss_root_vel"),
        "valid_loss_root_vel": _maybe_col("valid_loss_root_vel"),
        "train_loss_root_acc": _maybe_col("train_loss_root_acc"),
        "valid_loss_root_acc": _maybe_col("valid_loss_root_acc"),
    }

    epochs = np.array([_safe_int(r.get(epoch_col, "0")) for r in rows], dtype=np.int32)
    train_loss = np.array([_safe_float(r.get(train_loss_col, "nan")) for r in rows], dtype=np.float64)
    valid_loss = np.array([_safe_float(r.get(valid_loss_col, "nan")) for r in rows], dtype=np.float64)
    valid_root_err = np.array([_safe_float(r.get(valid_root_err_col, "nan")) for r in rows], dtype=np.float64)

    extra_series: dict[str, np.ndarray] = {}
    for key, col in extra_cols.items():
        if col is None:
            continue
        extra_series[key] = np.array([_safe_float(r.get(col, "nan")) for r in rows], dtype=np.float64)

    # Sort by epoch for cleaner plotting.
    order = np.argsort(epochs)
    epochs = epochs[order]
    train_loss = train_loss[order]
    valid_loss = valid_loss[order]
    valid_root_err = valid_root_err[order]
    for k in list(extra_series.keys()):
        extra_series[k] = extra_series[k][order]

    _ensure_dir(out_path.parent)
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

    m_train = np.isfinite(train_loss)
    m_valid = np.isfinite(valid_loss)
    m_root = np.isfinite(valid_root_err)

    # Use markers so a single point is visible.
    if m_train.any():
        ax1.plot(epochs[m_train], train_loss[m_train], label="train_loss", marker="o", markersize=3)
    else:
        ax1.plot([], [], label="train_loss")
    if m_valid.any():
        ax1.plot(epochs[m_valid], valid_loss[m_valid], label="valid_loss", marker="o", markersize=3)
    else:
        ax1.plot([], [], label="valid_loss")

    # Optional additional curves when present (e.g., temporal training logs).
    for key, label in [
        ("train_loss_root", "train_root"),
        ("valid_loss_root", "valid_root"),
        ("train_loss_root_x", "train_root_x"),
        ("train_loss_root_y", "train_root_y"),
        ("train_loss_root_z", "train_root_z"),
        ("valid_loss_root_x", "valid_root_x"),
        ("valid_loss_root_y", "valid_root_y"),
        ("valid_loss_root_z", "valid_root_z"),
        ("train_loss_root_vel", "train_root_vel"),
        ("valid_loss_root_vel", "valid_root_vel"),
        ("train_loss_root_acc", "train_root_acc"),
        ("valid_loss_root_acc", "valid_root_acc"),
    ]:
        s = extra_series.get(key)
        if s is None:
            continue
        m = np.isfinite(s)
        if m.any():
            ax1.plot(epochs[m], s[m], label=label, marker="o", markersize=2, linewidth=1)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    if m_root.any():
        ax2.plot(
            epochs[m_root],
            valid_root_err[m_root],
            label="valid_root_error_mean_m",
            color="tab:red",
            marker="o",
            markersize=3,
        )
    else:
        ax2.plot([], [], label="valid_root_error_mean_m", color="tab:red")
    ax2.set_ylabel("root error (m)")
    ax2.legend(loc="upper right")

    # Avoid tiny +/-0.04 axes when epochs collapse to a single value.
    if epochs.size > 0:
        xmin = int(np.min(epochs))
        xmax = int(np.max(epochs))
        if xmin == xmax:
            xmin -= 1
            xmax += 1
        ax1.set_xlim(xmin, xmax)

    if not (m_train.any() or m_valid.any() or m_root.any()):
        ax1.text(
            0.5,
            0.5,
            "No valid metrics found in train_log.csv\n(check delimiter/column names)",
            transform=ax1.transAxes,
            ha="center",
            va="center",
        )

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
    show_sam2d: bool = True,
    num_players_per_subplot: Optional[int] = 1,
) -> None:
    """Overlay: GT 2D vs optional SAM 2D vs predicted reprojection on a few frames."""

    pred = load_predictions_npz(predictions_npz)
    seq_names = list(pred.get("seq_names", []))

    seq_id_arr = np.asarray(pred["seq_id"], dtype=np.int32)
    person_arr = np.asarray(pred["person_idx"], dtype=np.int32)
    frame_arr = np.asarray(pred["frame_idx"], dtype=np.int32)
    root_pred_m = np.asarray(pred["root_pred_m"], dtype=np.float32)

    if seq_name is not None:
        try:
            seq_id = int(seq_names.index(seq_name))
        except ValueError:
            raise ValueError(f"seq_name {seq_name!r} not found in predictions")
    else:
        if seq_id_arr.size == 0:
            raise ValueError("Empty predictions")
        seq_id = int(np.bincount(seq_id_arr[seq_id_arr >= 0]).argmax())
        seq_name = str(seq_names[seq_id]) if 0 <= seq_id < len(seq_names) else ""

    seq_mask = seq_id_arr == seq_id
    if not seq_mask.any():
        raise ValueError(f"No samples for chosen sequence {seq_name}")

    if num_players_per_subplot is not None:
        num_players_per_subplot = int(num_players_per_subplot)
        if num_players_per_subplot <= 0:
            raise ValueError("num_players_per_subplot must be > 0 or None")

    root_lookup: dict[tuple[int, int], np.ndarray] = {}
    for p, f, r in zip(person_arr[seq_mask], frame_arr[seq_mask], root_pred_m[seq_mask]):
        root_lookup[(int(p), int(f))] = np.asarray(r, dtype=np.float32)

    if person_idx is not None:
        mask = seq_mask & (person_arr == int(person_idx))
        if not mask.any():
            raise ValueError(f"No samples for {seq_name} person={person_idx}")
        frames = frame_arr[mask]
    else:
        frames = np.unique(frame_arr[seq_mask])

    # Pick evenly-spaced frames.
    order = np.argsort(frames)
    frames = frames[order]

    if frames.size == 0:
        return

    num_frames = int(max(1, num_frames))
    pick = np.linspace(0, frames.size - 1, num=min(num_frames, frames.size), dtype=int)

    # Load raw normalized sequence file.
    seq_path = Path(data_dir) / split / f"{seq_name}.npz"
    with np.load(seq_path, allow_pickle=True) as npz:
        image_size = np.asarray(npz["image_size"], dtype=np.float32)  # (2,) [W,H]
        W, H = float(image_size[0]), float(image_size[1])

        x2d_sam_img_norm = np.asarray(npz["skel_2d_sam3dbody_from_bbox_gt"], dtype=np.float32)  # (P,T,25,2)
        y2d_gt = np.asarray(npz["Y_2d_gt"], dtype=np.float32)  # (P,T,25,2)
        x3d_sam_norm = np.asarray(npz["skel_3d_sam3dbody_from_bbox_gt"], dtype=np.float32)  # (P,T,25,3)
        K_all = np.asarray(npz["K"], dtype=np.float32)  # (T,3,3)
        k_all = np.asarray(npz["k"], dtype=np.float32)  # (T,2)
        valid_joints_all = np.asarray(npz["valid_joints"], dtype=bool)  # (P,T,25)
        valid_mask_all = np.asarray(npz["valid_mask"], dtype=bool)  # (P,T)

    _ensure_dir(out_path.parent)

    cols = len(pick)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), squeeze=False)

    stats_cpu = stats.to("cpu")

    P, T = int(valid_mask_all.shape[0]), int(valid_mask_all.shape[1])

    def visible_players_for_frame(fr: int) -> list[int]:
        if fr < 0 or fr >= T:
            return []

        pids_with_pred = sorted(
            {
                int(p)
                for p in person_arr[seq_mask & (frame_arr == fr)]
                if 0 <= int(p) < P and (int(p), fr) in root_lookup
            }
        )

        visible: list[tuple[int, int]] = []
        for p in pids_with_pred:
            if not bool(valid_mask_all[p, fr]):
                continue
            gt2d = y2d_gt[p, fr]
            finite = np.isfinite(gt2d).all(axis=-1)
            in_image = (gt2d[:, 0] >= 0.0) & (gt2d[:, 0] < W) & (gt2d[:, 1] >= 0.0) & (gt2d[:, 1] < H)
            joint_mask = valid_joints_all[p, fr] & finite & in_image
            if bool(joint_mask.any()):
                visible.append((p, int(joint_mask.sum())))

        visible.sort(key=lambda item: (-item[1], item[0]))
        players = [p for p, _ in visible]

        if person_idx is not None and 0 <= int(person_idx) < P and (int(person_idx), fr) in root_lookup:
            # Keep explicit person_idx as the first plotted player, preserving the
            # previous one-player behavior when num_players_per_subplot=1.
            players = [int(person_idx)] + [p for p in players if p != int(person_idx)]

        if num_players_per_subplot is None:
            return players
        return players[:num_players_per_subplot]

    for j, idx0 in enumerate(pick):
        fr = int(frames[idx0])
        ax = axes[0, j]
        plotted_players = visible_players_for_frame(fr)
        used_labels: set[str] = set()

        for p in plotted_players:
            root_fr = root_lookup.get((int(p), fr))
            if root_fr is None:
                continue

            sam2d_px = x2d_sam_img_norm[p, fr] * np.array([W, H], dtype=np.float32)
            gt2d_px = y2d_gt[p, fr]

            X_rel_norm = torch.from_numpy(x3d_sam_norm[p, fr]).unsqueeze(0)  # (1,25,3)
            X_rel_m = stats_cpu.denorm_sam3d_rel(X_rel_norm).squeeze(0).numpy()

            X_cam_pred = X_rel_m + root_fr[None, :]
            uv_pred = project_cam_to_image(
                torch.from_numpy(X_cam_pred).unsqueeze(0),
                K=torch.from_numpy(K_all[fr]).unsqueeze(0),
                k=torch.from_numpy(k_all[fr]).unsqueeze(0),
            ).squeeze(0).numpy()

            vj_gt = valid_joints_all[p, fr] & np.isfinite(gt2d_px).all(axis=-1)
            vj_sam = vj_gt & np.isfinite(sam2d_px).all(axis=-1)
            vj_pred = valid_joints_all[p, fr] & np.isfinite(uv_pred).all(axis=-1)

            label = "GT 2D"
            ax.scatter(
                gt2d_px[vj_gt, 0],
                gt2d_px[vj_gt, 1],
                s=1,
                color="tab:blue",
                label=label if label not in used_labels else None,
            )
            used_labels.add(label)

            if show_sam2d:
                label = "SAM 2D"
                ax.scatter(
                    sam2d_px[vj_sam, 0],
                    sam2d_px[vj_sam, 1],
                    s=1,
                    color="tab:green",
                    label=label if label not in used_labels else None,
                )
                used_labels.add(label)

            label = "Pred reproj"
            ax.scatter(
                uv_pred[vj_pred, 0],
                uv_pred[vj_pred, 1],
                s=1,
                color="tab:orange",
                label=label if label not in used_labels else None,
            )
            used_labels.add(label)

        ax.set_title(f"frame {fr} ({len(plotted_players)} players)")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)  # invert y
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)

    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="lower right")
            break

    if person_idx is not None and num_players_per_subplot == 1:
        player_desc = f"person={person_idx}"
    elif num_players_per_subplot is None:
        player_desc = "all visible players"
    else:
        player_desc = f"up to {num_players_per_subplot} players"
    fig.suptitle(f"2D overlay - {seq_name} ({player_desc})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
