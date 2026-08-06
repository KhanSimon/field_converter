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

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # -------------------------
    # Trajectoires
    # -------------------------
    ax.plot(wg[:, 0], wg[:, 1], label="gt", linewidth=2)
    ax.plot(wp[:, 0], wp[:, 1], label="pred", linewidth=1)

    # -------------------------
    # Terrain FIFA (105 x 68 m)
    # Coordonnées centrées en (0,0)
    # -------------------------
    FIELD_X = 52.5
    FIELD_Y = 34.0

    line_kw = dict(color="black", linewidth=1.0, alpha=0.5)

    # Contour
    ax.plot(
        [-FIELD_X, FIELD_X, FIELD_X, -FIELD_X, -FIELD_X],
        [-FIELD_Y, -FIELD_Y, FIELD_Y, FIELD_Y, -FIELD_Y],
        **line_kw,
    )

    # Ligne médiane
    ax.plot([0, 0], [-FIELD_Y, FIELD_Y], **line_kw)

    # Cercle central (rayon 9.15 m)
    center_circle = plt.Circle(
        (0, 0),
        9.15,
        fill=False,
        **line_kw,
    )
    ax.add_patch(center_circle)

    # Surface de réparation (16.5 m × 40.32 m)
    penalty_depth = 16.5
    penalty_half_width = 20.16

    # Gauche
    ax.plot(
        [-FIELD_X, -FIELD_X + penalty_depth,
         -FIELD_X + penalty_depth, -FIELD_X],
        [-penalty_half_width, -penalty_half_width,
         penalty_half_width, penalty_half_width],
        **line_kw,
    )

    # Droite
    ax.plot(
        [FIELD_X, FIELD_X - penalty_depth,
         FIELD_X - penalty_depth, FIELD_X],
        [-penalty_half_width, -penalty_half_width,
         penalty_half_width, penalty_half_width],
        **line_kw,
    )

    # Surface de but (5.5 m × 18.32 m)
    goal_depth = 5.5
    goal_half_width = 9.16

    # Gauche
    ax.plot(
        [-FIELD_X, -FIELD_X + goal_depth,
         -FIELD_X + goal_depth, -FIELD_X],
        [-goal_half_width, -goal_half_width,
         goal_half_width, goal_half_width],
        **line_kw,
    )

    # Droite
    ax.plot(
        [FIELD_X, FIELD_X - goal_depth,
         FIELD_X - goal_depth, FIELD_X],
        [-goal_half_width, -goal_half_width,
         goal_half_width, goal_half_width],
        **line_kw,
    )

    # -------------------------
    # Limites fixes
    # -------------------------
    ax.set_xlim(-52.5, 52.5)
    ax.set_ylim(-34, 34)

    ax.set_title(f"World root trajectory (X-Y) — {seq_name} person={pid}")
    ax.set_xlabel("X_world (m)")
    ax.set_ylabel("Y_world (m)")

    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _plot_diagnostic_scatter(
    *,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    x, y = _finite_xy(x, y)
    n = int(x.size)
    r = float("nan")

    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    if n > 0:
        ax.scatter(x, y, s=10, alpha=0.35, linewidths=0)

    if n >= 2 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.array([float(np.min(x)), float(np.max(x))], dtype=np.float64)
        ax.plot(xs, slope * xs + intercept, color="tab:red", linewidth=2, label="linear fit")
        r = float(np.corrcoef(x, y)[0, 1])
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"r = {r:.3f}\nN = {n}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    if n == 0:
        ax.text(0.5, 0.5, "No finite points", transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _project_root_center_distances(
    *,
    data_dir: Path,
    split: str,
    pred: Dict[str, Any],
) -> np.ndarray:
    seq_names = list(pred.get("seq_names", []))
    seq_id_arr = np.asarray(pred["seq_id"], dtype=np.int32)
    frame_arr = np.asarray(pred["frame_idx"], dtype=np.int32)
    root_gt_m = np.asarray(pred["root_gt_m"], dtype=np.float32)
    out = np.full((root_gt_m.shape[0],), np.nan, dtype=np.float64)

    for seq_id in sorted(int(s) for s in np.unique(seq_id_arr) if int(s) >= 0):
        if seq_id >= len(seq_names):
            continue
        seq_name = str(seq_names[seq_id])
        seq_path = Path(data_dir) / split / f"{seq_name}.npz"
        if not seq_path.exists():
            continue
        idx = np.flatnonzero(seq_id_arr == seq_id)
        if idx.size == 0:
            continue

        with np.load(seq_path, allow_pickle=True) as npz:
            image_size = np.asarray(npz["image_size"], dtype=np.float32).reshape(-1)
            K_all = np.asarray(npz["K"], dtype=np.float32)
            k_all = np.asarray(npz["k"], dtype=np.float32)

        if image_size.size != 2:
            continue
        W, H = float(image_size[0]), float(image_size[1])
        frames = frame_arr[idx]
        in_bounds = (frames >= 0) & (frames < K_all.shape[0])
        if not in_bounds.any() or W <= 0.0 or H <= 0.0:
            continue

        idx_valid = idx[in_bounds]
        frames_valid = frames[in_bounds]
        uv = project_cam_to_image(
            torch.from_numpy(root_gt_m[idx_valid, None, :]),
            K=torch.from_numpy(K_all[frames_valid]),
            k=torch.from_numpy(k_all[frames_valid]),
        ).squeeze(1).numpy()
        out[idx_valid] = np.sqrt(((uv[:, 0] / W) - 0.5) ** 2 + ((uv[:, 1] / H) - 0.5) ** 2)

    return out


def _root_world_speeds(pred: Dict[str, Any], *, speed_window: int) -> tuple[np.ndarray, np.ndarray]:
    seq_id_arr = np.asarray(pred["seq_id"], dtype=np.int32)
    person_arr = np.asarray(pred["person_idx"], dtype=np.int32)
    frame_arr = np.asarray(pred["frame_idx"], dtype=np.int32)
    root_world_gt = np.asarray(pred["root_world_gt_m"], dtype=np.float32)
    root_err = np.asarray(pred["root_error_m"], dtype=np.float32)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    step = int(max(1, speed_window))

    for seq_id in np.unique(seq_id_arr):
        for pid in np.unique(person_arr[seq_id_arr == seq_id]):
            mask = (seq_id_arr == seq_id) & (person_arr == pid)
            if int(mask.sum()) <= step:
                continue
            order = np.argsort(frame_arr[mask])
            frames = frame_arr[mask][order].astype(np.float64)
            roots = root_world_gt[mask][order].astype(np.float64)
            errs = root_err[mask][order].astype(np.float64)

            delta_frames = frames[step:] - frames[:-step]
            valid = delta_frames > 0.0
            if not valid.any():
                continue
            delta_pos = roots[step:] - roots[:-step]
            speed = np.linalg.norm(delta_pos, axis=-1) / delta_frames
            xs.append(speed[valid])
            ys.append(errs[step:][valid])

    if not xs:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def plot_root_diagnostic_plots(
    *,
    data_dir: Path,
    split: str,
    predictions_npz: Path,
    out_dir: Path,
    root_error_vs_camera_distance: bool = True,
    root_error_vs_image_center_distance: bool = True,
    root_error_vs_player_speed: bool = True,
    speed_window: int = 5,
) -> None:
    pred = load_predictions_npz(predictions_npz)
    root_gt_m = np.asarray(pred["root_gt_m"], dtype=np.float32)
    root_err = np.asarray(pred["root_error_m"], dtype=np.float32)

    if root_error_vs_camera_distance:
        camera_distance = np.linalg.norm(root_gt_m, axis=-1)
        _plot_diagnostic_scatter(
            x=camera_distance,
            y=root_err,
            out_path=out_dir / f"root_error_vs_camera_distance_{split}.png",
            xlabel="GT root camera distance ||root_cam_gt|| (m)",
            ylabel="root_error_3d (m)",
            title=f"Root error vs camera distance - {split}",
        )

    if root_error_vs_image_center_distance:
        center_distance = _project_root_center_distances(data_dir=data_dir, split=split, pred=pred)
        _plot_diagnostic_scatter(
            x=center_distance,
            y=root_err,
            out_path=out_dir / f"root_error_vs_image_center_distance_{split}.png",
            xlabel="normalized GT root distance to image center",
            ylabel="root_error_3d (m)",
            title=f"Root error vs image-center distance - {split}",
        )

    if root_error_vs_player_speed:
        speed, err = _root_world_speeds(pred, speed_window=speed_window)
        _plot_diagnostic_scatter(
            x=speed,
            y=err,
            out_path=out_dir / f"root_error_vs_player_speed_{split}.png",
            xlabel=f"GT root world speed over {int(max(1, speed_window))} frames (m/frame)",
            ylabel="root_error_3d (m)",
            title=f"Root error vs player speed - {split}",
        )


def _prediction_airborne_masks(
    *,
    data_dir: Path,
    split: str,
    pred: Dict[str, Any],
    airborne_threshold_m: float = 0.05,
    ground_reference_percentile: float = 20.0,
    ground_reference_window_frames: int = 125,
    min_airborne_frames: int = 2,
    max_ground_gap_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return airborne and classifiable masks aligned with prediction rows.

    Airborne phases use the same GT 3D definition as
    ``utils.analyze_airborne_root_error``: both BODY-25 feet must clear their
    local observed ground-contact reference by at least 5 cm.
    """
    from field_converter.utils.analyze_airborne_root_error import (
        DEFAULT_LEFT_FOOT_JOINTS,
        DEFAULT_RIGHT_FOOT_JOINTS,
        calibrate_foot_clearance,
        clean_airborne_mask,
        compute_foot_heights,
        fit_pitch_plane,
    )

    seq_names = [str(name) for name in pred.get("seq_names", [])]
    seq_ids = np.asarray(pred["seq_id"], dtype=np.int32).reshape(-1)
    person_indices = np.asarray(pred["person_idx"], dtype=np.int32).reshape(-1)
    frame_indices = np.asarray(pred["frame_idx"], dtype=np.int32).reshape(-1)
    if not (seq_ids.shape == person_indices.shape == frame_indices.shape):
        raise ValueError("seq_id, person_idx and frame_idx must have identical shapes")

    airborne_out = np.zeros(seq_ids.shape, dtype=bool)
    classifiable_out = np.zeros(seq_ids.shape, dtype=bool)

    for seq_id in sorted(int(value) for value in np.unique(seq_ids) if int(value) >= 0):
        if seq_id >= len(seq_names):
            raise ValueError(f"Prediction seq_id={seq_id} has no matching seq_names entry")
        sequence = seq_names[seq_id]
        feature_path = data_dir / split / f"{sequence}.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing GT features for airborne plot: {feature_path}")

        with np.load(feature_path, allow_pickle=True) as npz:
            required = ("Y_cam_gt", "Y_root_cam_gt", "R", "t", "valid_mask", "pitch_points_world")
            missing = [key for key in required if key not in npz.files]
            if missing:
                raise KeyError(f"{feature_path}: missing keys required by airborne plot: {missing}")
            Y_cam_gt = np.asarray(npz["Y_cam_gt"], dtype=np.float64)
            root_cam_gt = np.asarray(npz["Y_root_cam_gt"], dtype=np.float64)
            R = np.asarray(npz["R"], dtype=np.float64)
            t = np.asarray(npz["t"], dtype=np.float64)
            valid_mask = np.asarray(npz["valid_mask"], dtype=bool)
            pitch_points_world = np.asarray(npz["pitch_points_world"], dtype=np.float64)

        plane_normal, plane_offset = fit_pitch_plane(pitch_points_world)
        foot_joint_indices = (*DEFAULT_LEFT_FOOT_JOINTS, *DEFAULT_RIGHT_FOOT_JOINTS)
        Y_cam_feet_gt = Y_cam_gt[:, :, foot_joint_indices, :]
        left_plane_height, right_plane_height, _ = compute_foot_heights(
            # Transform only the six toe/heel joints needed by this plot,
            # instead of all 25 GT joints.
            Y_cam_gt=Y_cam_feet_gt,
            root_cam_gt=root_cam_gt,
            R=R,
            t=t,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
            left_foot_joints=(0, 1, 2),
            right_foot_joints=(3, 4, 5),
        )
        height_valid = valid_mask & np.isfinite(left_plane_height) & np.isfinite(right_plane_height)
        left_clearance, right_clearance, _, _, _ = calibrate_foot_clearance(
            left_plane_height,
            right_plane_height,
            height_valid,
            ground_reference_percentile=ground_reference_percentile,
            ground_reference_window_frames=ground_reference_window_frames,
        )
        gt_valid = height_valid & np.isfinite(left_clearance) & np.isfinite(right_clearance)
        sequence_airborne = np.zeros_like(gt_valid)

        prediction_rows = np.flatnonzero(seq_ids == seq_id)
        predicted_people = np.unique(person_indices[prediction_rows])
        for person_idx in predicted_people:
            if person_idx < 0 or person_idx >= gt_valid.shape[0]:
                continue
            candidate = (
                gt_valid[person_idx]
                & (left_clearance[person_idx] >= airborne_threshold_m)
                & (right_clearance[person_idx] >= airborne_threshold_m)
            )
            sequence_airborne[person_idx] = clean_airborne_mask(
                candidate,
                gt_valid[person_idx],
                min_airborne_frames=min_airborne_frames,
                max_ground_gap_frames=max_ground_gap_frames,
            )

        people = person_indices[prediction_rows]
        frames = frame_indices[prediction_rows]
        in_bounds = (
            (people >= 0)
            & (people < gt_valid.shape[0])
            & (frames >= 0)
            & (frames < gt_valid.shape[1])
        )
        bounded_rows = prediction_rows[in_bounds]
        bounded_people = people[in_bounds]
        bounded_frames = frames[in_bounds]
        row_is_valid = gt_valid[bounded_people, bounded_frames]
        valid_rows = bounded_rows[row_is_valid]
        valid_people = bounded_people[row_is_valid]
        valid_frames = bounded_frames[row_is_valid]
        classifiable_out[valid_rows] = True
        airborne_out[valid_rows] = sequence_airborne[valid_people, valid_frames]

    return airborne_out, classifiable_out


def plot_root_error_ground_vs_air_histogram(
    *,
    data_dir: Path,
    split: str,
    predictions_npz: Path,
    out_path: Path,
    density: bool = False,
    num_bins: int = 60,
) -> None:
    """Plot root-error distributions for ground and airborne GT frames.

    By default, the histogram shows actual frame counts (``density=False``),
    so the large ground/air class imbalance remains visible.
    """
    pred = load_predictions_npz(predictions_npz)
    root_error = np.asarray(pred["root_error_m"], dtype=np.float64).reshape(-1)
    airborne, classifiable = _prediction_airborne_masks(
        data_dir=Path(data_dir),
        split=split,
        pred=pred,
    )
    if root_error.shape != airborne.shape:
        raise ValueError(
            f"root_error_m and prediction metadata are misaligned: {root_error.shape} vs {airborne.shape}"
        )

    finite = classifiable & np.isfinite(root_error) & (root_error >= 0.0)
    ground_error = root_error[finite & ~airborne]
    airborne_error = root_error[finite & airborne]
    pooled = root_error[finite]

    _ensure_dir(out_path.parent)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    if pooled.size:
        upper = max(0.1, float(np.percentile(pooled, 99.5)))
        bins = np.linspace(0.0, upper, max(2, int(num_bins)) + 1)
        # Preserve every frame count while keeping a readable x-axis: the
        # upper 0.5% tail is accumulated in the final visible bin.
        clip_max = np.nextafter(upper, 0.0)
        ground_for_plot = np.minimum(ground_error, clip_max)
        airborne_for_plot = np.minimum(airborne_error, clip_max)
        if ground_for_plot.size:
            ax.hist(
                ground_for_plot,
                bins=bins,
                density=density,
                alpha=0.55,
                color="tab:gray",
                label=f"Ground (N frames={ground_error.size:,})",
            )
        if airborne_for_plot.size:
            ax.hist(
                airborne_for_plot,
                bins=bins,
                density=density,
                alpha=0.65,
                color="tab:orange",
                label=f"Airborne (N frames={airborne_error.size:,})",
            )
        ax.set_xlim(0.0, upper)
    else:
        ax.text(0.5, 0.5, "No classifiable finite frames", transform=ax.transAxes, ha="center", va="center")

    ax.set_xlabel(r"$\|root_{pred} - root_{GT}\|_2$ (m)")
    ax.set_ylabel("Density" if density else "Number of frames")
    ax.set_title(f"Root error on ground vs airborne frames - {split}")
    ax.grid(axis="y", alpha=0.25)
    if ground_error.size or airborne_error.size:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
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
        ("MPJPE_local_m", "MPJPE local (m)"),
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
