"""Canonicalize inference camera world axes to the training convention.

The root model can consume camera-center and camera-forward features in world
coordinates.  A camera exported from the opposite side of a pitch, or with a
different choice of world-axis signs, can therefore be geometrically valid but
far outside the training distribution.

Only origin-preserving 180-degree rotations are considered here.  They keep a
right-handed coordinate system, preserve pitch dimensions and preserve the
``z=0`` pitch plane.  Raw input files are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np


WorldAlignmentMode = Literal[
    "auto",
    "none",
    "rotate_x_180",
    "rotate_y_180",
    "rotate_z_180",
]

AUTO_OUTLIER_Z_THRESHOLD = 5.0
AUTO_MIN_SCORE_IMPROVEMENT = 0.25


WORLD_ROTATIONS: Dict[str, np.ndarray] = {
    "identity": np.eye(3, dtype=np.float64),
    # Row-vector equivalent: (x,y,z) -> (x,-y,-z).
    "rotate_x_180": np.diag([1.0, -1.0, -1.0]),
    # Row-vector equivalent: (x,y,z) -> (-x,y,-z).
    "rotate_y_180": np.diag([-1.0, 1.0, -1.0]),
    # Row-vector equivalent: (x,y,z) -> (-x,-y,z).
    "rotate_z_180": np.diag([-1.0, -1.0, 1.0]),
}


@dataclass(frozen=True)
class WorldAlignment:
    requested_mode: WorldAlignmentMode
    selected_transform: str
    applied: bool
    rotation_source_to_aligned: np.ndarray
    camera_center_source_median: np.ndarray
    camera_center_aligned_median: np.ndarray
    train_camera_center_mean: np.ndarray
    train_camera_center_std: np.ndarray
    source_zscore: np.ndarray
    aligned_zscore: np.ndarray
    source_score_rms: float
    aligned_score_rms: float
    score_improvement_fraction: float
    auto_outlier_z_threshold: float
    auto_min_score_improvement: float

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested_mode": self.requested_mode,
            "selected_transform": self.selected_transform,
            "applied": self.applied,
            "rotation_source_to_aligned": self.rotation_source_to_aligned.tolist(),
            "camera_center_source_median": self.camera_center_source_median.tolist(),
            "camera_center_aligned_median": self.camera_center_aligned_median.tolist(),
            "train_camera_center_mean": self.train_camera_center_mean.tolist(),
            "train_camera_center_std": self.train_camera_center_std.tolist(),
            "source_zscore": self.source_zscore.tolist(),
            "aligned_zscore": self.aligned_zscore.tolist(),
            "source_score_rms": self.source_score_rms,
            "aligned_score_rms": self.aligned_score_rms,
            "score_improvement_fraction": self.score_improvement_fraction,
            "auto_outlier_z_threshold": self.auto_outlier_z_threshold,
            "auto_min_score_improvement": self.auto_min_score_improvement,
        }


def camera_centers_world(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return camera centers for ``X_cam_col = R @ X_world_col + t``."""
    rotations = np.asarray(R, dtype=np.float64)
    translations = np.asarray(t, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (T,3,3), got {rotations.shape}")
    if translations.shape != (rotations.shape[0], 3):
        raise ValueError(f"t must have shape {(rotations.shape[0], 3)}, got {translations.shape}")
    return -np.einsum("tji,tj->ti", rotations, translations)


def _zscore(center: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.maximum(np.asarray(std, dtype=np.float64), 1e-8)
    return (np.asarray(center, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / safe_std


def _score_rms(zscore: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(zscore, dtype=np.float64)))))


def detect_world_alignment(
    R: np.ndarray,
    t: np.ndarray,
    *,
    train_mean_C: np.ndarray,
    train_std_C: np.ndarray,
    mode: WorldAlignmentMode = "auto",
    outlier_z_threshold: float = AUTO_OUTLIER_Z_THRESHOLD,
    min_score_improvement: float = AUTO_MIN_SCORE_IMPROVEMENT,
) -> WorldAlignment:
    """Select an origin-preserving world rotation using training camera stats.

    In ``auto`` mode a rotation is used only when the unaligned median camera
    center is an outlier and the best candidate improves the RMS z-score by at
    least ``min_score_improvement``. Explicit modes bypass that decision.
    """
    if mode not in {"auto", "none", "rotate_x_180", "rotate_y_180", "rotate_z_180"}:
        raise ValueError(f"Unsupported world alignment mode: {mode}")
    if outlier_z_threshold <= 0:
        raise ValueError("outlier_z_threshold must be > 0")
    if not 0.0 <= min_score_improvement < 1.0:
        raise ValueError("min_score_improvement must be in [0,1)")

    centers = camera_centers_world(R, t)
    finite_centers = centers[np.isfinite(centers).all(axis=-1)]
    if finite_centers.size == 0:
        raise ValueError("Cannot align world axes: no finite camera centers")

    source_median = np.median(finite_centers, axis=0)
    train_mean = np.asarray(train_mean_C, dtype=np.float64).reshape(3)
    train_std = np.asarray(train_std_C, dtype=np.float64).reshape(3)
    source_z = _zscore(source_median, train_mean, train_std)
    source_score = _score_rms(source_z)

    candidate_metrics: dict[str, Tuple[float, np.ndarray, np.ndarray]] = {}
    for name, rotation in WORLD_ROTATIONS.items():
        aligned_median = source_median @ rotation.T
        aligned_z = _zscore(aligned_median, train_mean, train_std)
        candidate_metrics[name] = (_score_rms(aligned_z), aligned_median, aligned_z)

    if mode == "none":
        selected = "identity"
    elif mode == "auto":
        best_name = min(candidate_metrics, key=lambda name: candidate_metrics[name][0])
        best_score = candidate_metrics[best_name][0]
        improvement = (source_score - best_score) / max(source_score, 1e-8)
        source_is_outlier = bool(np.max(np.abs(source_z)) >= float(outlier_z_threshold))
        selected = (
            best_name
            if best_name != "identity"
            and source_is_outlier
            and improvement >= float(min_score_improvement)
            else "identity"
        )
    else:
        selected = mode

    selected_score, aligned_median, aligned_z = candidate_metrics[selected]
    improvement = (source_score - selected_score) / max(source_score, 1e-8)
    return WorldAlignment(
        requested_mode=mode,
        selected_transform=selected,
        applied=selected != "identity",
        rotation_source_to_aligned=WORLD_ROTATIONS[selected].copy(),
        camera_center_source_median=source_median.astype(np.float64),
        camera_center_aligned_median=aligned_median.astype(np.float64),
        train_camera_center_mean=train_mean,
        train_camera_center_std=train_std,
        source_zscore=source_z.astype(np.float64),
        aligned_zscore=aligned_z.astype(np.float64),
        source_score_rms=source_score,
        aligned_score_rms=selected_score,
        score_improvement_fraction=float(improvement),
        auto_outlier_z_threshold=float(outlier_z_threshold),
        auto_min_score_improvement=float(min_score_improvement),
    )


def align_camera_extrinsics(
    R_source: np.ndarray,
    t_source: np.ndarray,
    alignment: WorldAlignment,
) -> Tuple[np.ndarray, np.ndarray]:
    """Express source camera extrinsics in the aligned world frame.

    For ``X_aligned = Q @ X_source`` and column-vector extrinsics, the aligned
    rotation is ``R_aligned = R_source @ Q.T``. Translation is unchanged
    because all supported rotations are around the pitch origin.
    """
    R = np.asarray(R_source, dtype=np.float64)
    t = np.asarray(t_source, dtype=np.float64)
    Q = np.asarray(alignment.rotation_source_to_aligned, dtype=np.float64)
    return np.einsum("tij,kj->tik", R, Q), t.copy()


def transform_world_points_source_to_aligned(
    points: np.ndarray,
    alignment: WorldAlignment,
) -> np.ndarray:
    """Transform row-vector world points from the source to aligned frame."""
    Q = np.asarray(alignment.rotation_source_to_aligned, dtype=np.float64)
    return np.asarray(points) @ Q.T


def transform_world_points_aligned_to_source(
    points: np.ndarray,
    alignment: WorldAlignment,
) -> np.ndarray:
    """Transform row-vector world points from the aligned back to source frame."""
    Q = np.asarray(alignment.rotation_source_to_aligned, dtype=np.float64)
    return np.asarray(points) @ Q
