from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from field_converter.ablation.common import (
    PROJECT_ROOT,
    campaign_output_dir,
    load_manifest,
    load_plan,
    resolve_project_path,
    write_json_atomic,
)

_CACHE_DIR = PROJECT_ROOT / "outputs" / "ablation" / ".publication_plot_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml  # type: ignore[import-untyped]

from field_converter.utils.analyze_airborne_root_error import (
    DEFAULT_LEFT_FOOT_JOINTS,
    DEFAULT_RIGHT_FOOT_JOINTS,
    _camera_roots_to_world,
    calibrate_foot_clearance,
    clean_airborne_mask,
    compute_foot_heights,
    fit_pitch_plane,
)


ARCHITECTURE_ORDER = ("mlp", "tcn", "transformer")
PUBLICATION_REFERENCE_VARIANTS = {
    "mlp": ("full",),
    "tcn": ("full", "window_201", "best_candidate"),
    "transformer": ("full", "window_41", "best_candidate"),
}
ADDITIONAL_PUBLICATION_RUNS: tuple[Mapping[str, Any], ...] = (
    {
        "index": 25,
        "run_name": "primary_tcn_best_candidate_no_pitch_ground_mask_s1235",
        "architecture": "tcn",
        "variant": "best_candidate",
        "fold": "primary",
        "seed": 1235,
        "window_size": 201,
    },
    {
        "index": 26,
        "run_name": "primary_transformer_best_candidate_no_pitch_ground_mask_s1235",
        "architecture": "transformer",
        "variant": "best_candidate",
        "fold": "primary",
        "seed": 1235,
        "window_size": 41,
    },
)
METHOD_COLORS = {
    "geometry": "#59636E",
    "mlp": "#E5A823",
    "tcn": "#007C83",
    "transformer": "#D1495B",
}
METHOD_MARKERS = {"geometry": "o", "mlp": "D", "tcn": "s", "transformer": "^"}
AXIS_LABELS = (r"$x^c$", r"$y^c$", r"$z^c$")


def _publication_runs(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    runs = list(plan["runs"])
    known_names = {str(run["run_name"]) for run in runs}
    runs.extend(
        run for run in ADDITIONAL_PUBLICATION_RUNS if str(run["run_name"]) not in known_names
    )
    return runs


@dataclass(frozen=True)
class PredictionSet:
    run_name: str
    architecture: str
    variant: str
    window_size: int | None
    sequence_names: tuple[str, ...]
    sequence_id: np.ndarray
    person_idx: np.ndarray
    frame_idx: np.ndarray
    sort_order: np.ndarray
    canonical_keys: np.ndarray
    root_pred_m: np.ndarray
    root_gt_m: np.ndarray
    root_error_m: np.ndarray
    root_world_pred_m: np.ndarray
    root_world_gt_m: np.ndarray


@dataclass(frozen=True)
class MethodSeries:
    key: str
    label: str
    table_label: str
    color: str
    marker: str
    root_error_m: np.ndarray
    component_error_m: np.ndarray
    run_name: str | None


@dataclass(frozen=True)
class DiagnosticData:
    sequence: np.ndarray
    cluster: np.ndarray
    person_idx: np.ndarray
    frame_idx: np.ndarray
    classifiable: np.ndarray
    airborne: np.ndarray
    lower_foot_clearance_m: np.ndarray
    camera_distance_m: np.ndarray
    image_center_distance_norm: np.ndarray
    bbox_height_ratio: np.ndarray
    world_speed_mps: np.ndarray
    pelvis_height_m: np.ndarray
    valid_joint_fraction: np.ndarray
    root_init_error_m: np.ndarray
    root_init_component_error_m: np.ndarray


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_path(output_dir: Path, run_name: str, split: str) -> Path:
    return output_dir / "predictions" / run_name / f"{split}_predictions.npz"


def _metrics_path(output_dir: Path, run_name: str) -> Path:
    return output_dir / "eval_reports" / run_name / "metrics.json"


def _input_config_flag(output_dir: Path, run_name: str, key: str) -> bool:
    path = output_dir / "eval_reports" / run_name / "config_used.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing run configuration for publication table: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    input_config = payload.get("input_config") if isinstance(payload, Mapping) else None
    if not isinstance(input_config, Mapping) or key not in input_config:
        raise KeyError(f"Missing input_config.{key} in {path}")
    value = input_config[key]
    if not isinstance(value, bool):
        raise TypeError(f"Expected boolean input_config.{key} in {path}, got {value!r}")
    return value


def _load_prediction(
    path: Path,
    *,
    run_name: str,
    architecture: str,
    variant: str,
    window_size: int | None,
) -> PredictionSet:
    with np.load(path, allow_pickle=True) as npz:
        sequence_names = tuple(str(value) for value in npz["seq_names"].tolist())
        sequence_id = np.asarray(npz["seq_id"], dtype=np.int32)
        person_idx = np.asarray(npz["person_idx"], dtype=np.int32)
        frame_idx = np.asarray(npz["frame_idx"], dtype=np.int32)
        root_pred_m = np.asarray(npz["root_pred_m"], dtype=np.float64)
        root_gt_m = np.asarray(npz["root_gt_m"], dtype=np.float64)
        root_error_m = np.asarray(npz["root_error_m"], dtype=np.float64)
        root_world_pred_m = np.asarray(npz["root_world_pred_m"], dtype=np.float64)
        root_world_gt_m = np.asarray(npz["root_world_gt_m"], dtype=np.float64)

    if not (sequence_id.shape == person_idx.shape == frame_idx.shape == root_error_m.shape):
        raise ValueError(f"Misaligned prediction metadata in {path}")
    if root_pred_m.shape != root_gt_m.shape or root_pred_m.shape != (root_error_m.size, 3):
        raise ValueError(f"Invalid root arrays in {path}")
    if root_world_pred_m.shape != root_world_gt_m.shape or root_world_pred_m.shape != root_pred_m.shape:
        raise ValueError(f"Invalid world-root arrays in {path}")

    sorted_names = tuple(sorted(sequence_names))
    name_rank = {name: index for index, name in enumerate(sorted_names)}
    sequence_rank = np.asarray([name_rank[sequence_names[int(value)]] for value in sequence_id], dtype=np.int32)
    order = np.lexsort((frame_idx, person_idx, sequence_rank))
    keys = np.column_stack((sequence_rank[order], person_idx[order], frame_idx[order]))
    return PredictionSet(
        run_name=run_name,
        architecture=architecture,
        variant=variant,
        window_size=window_size,
        sequence_names=sequence_names,
        sequence_id=sequence_id,
        person_idx=person_idx,
        frame_idx=frame_idx,
        sort_order=order,
        canonical_keys=keys,
        root_pred_m=root_pred_m[order],
        root_gt_m=root_gt_m[order],
        root_error_m=root_error_m[order],
        root_world_pred_m=root_world_pred_m[order],
        root_world_gt_m=root_world_gt_m[order],
    )


def _publication_run_by_architecture(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    fold: str,
    seed: int,
    split: str,
    reference_run_names: Mapping[str, str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    runs = _publication_runs(plan)
    runs_by_name = {str(run["run_name"]): run for run in runs}
    selected: dict[str, Mapping[str, Any]] = {}
    for architecture, run_name_value in (reference_run_names or {}).items():
        if architecture not in ARCHITECTURE_ORDER:
            raise ValueError(f"Unknown publication-reference architecture: {architecture!r}")
        run_name = str(run_name_value)
        run = runs_by_name.get(run_name)
        if run is None:
            raise KeyError(f"Publication-reference run is absent from the campaign plan: {run_name}")
        if str(run["architecture"]) != architecture or str(run["fold"]) != fold:
            raise ValueError(
                f"Invalid publication reference {run_name}: expected {architecture}/{fold}, "
                f"got {run['architecture']}/{run['fold']}"
            )
        missing = [
            path
            for path in (
                _prediction_path(output_dir, run_name, split),
                _metrics_path(output_dir, run_name),
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Publication-reference run {run_name} is incomplete; missing: "
                + ", ".join(str(path) for path in missing)
            )
        selected[architecture] = run

    candidates: dict[str, list[tuple[float, Mapping[str, Any]]]] = {
        architecture: [] for architecture in ARCHITECTURE_ORDER
    }
    for run in runs:
        architecture = str(run["architecture"])
        if architecture not in ARCHITECTURE_ORDER:
            continue
        if architecture in selected:
            continue
        if str(run["fold"]) != fold or int(run["seed"]) != seed:
            continue
        if str(run["variant"]) not in PUBLICATION_REFERENCE_VARIANTS[architecture]:
            continue
        run_name = str(run["run_name"])
        prediction_path = _prediction_path(output_dir, run_name, split)
        metrics_path = _metrics_path(output_dir, run_name)
        if not prediction_path.exists() or not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric = _metric_value(payload.get("splits", {}).get(split), "root_error_mean_m")
        if metric is not None:
            candidates[architecture].append((metric, run))
    selected.update(
        {
            architecture: min(values, key=lambda item: item[0])[1]
            for architecture, values in candidates.items()
            if values
        }
    )
    return selected


def _load_publication_predictions(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    fold: str,
    seed: int,
    split: str,
    reference_run_names: Mapping[str, str] | None = None,
) -> dict[str, PredictionSet]:
    predictions: dict[str, PredictionSet] = {}
    selected_runs = _publication_run_by_architecture(
        output_dir=output_dir,
        plan=plan,
        fold=fold,
        seed=seed,
        split=split,
        reference_run_names=reference_run_names,
    )
    for architecture, run in selected_runs.items():
        run_name = str(run["run_name"])
        prediction_path = _prediction_path(output_dir, run_name, split)
        predictions[architecture] = _load_prediction(
            prediction_path,
            run_name=run_name,
            architecture=architecture,
            variant=str(run["variant"]),
            window_size=(
                int(run["window_size"])
                if run.get("window_size") is not None
                else None
            ),
        )
    if not predictions:
        raise FileNotFoundError(
            f"No completed publication-reference predictions found for fold={fold}, split={split}"
        )
    return predictions


def _reference_prediction(predictions: Mapping[str, PredictionSet]) -> PredictionSet:
    for architecture in ("tcn", "transformer", "mlp"):
        if architecture in predictions:
            return predictions[architecture]
    raise ValueError("No reference prediction is available")


def _validate_prediction_alignment(reference: PredictionSet, prediction: PredictionSet) -> None:
    if reference.canonical_keys.shape != prediction.canonical_keys.shape or not np.array_equal(
        reference.canonical_keys, prediction.canonical_keys
    ):
        raise ValueError(
            f"Prediction identifiers differ between {reference.run_name} and {prediction.run_name}; "
            "publication diagnostics require identical evaluated frames"
        )
    reference_names = tuple(sorted(reference.sequence_names))
    prediction_names = tuple(sorted(prediction.sequence_names))
    if reference_names != prediction_names:
        raise ValueError(f"Prediction sequences differ between {reference.run_name} and {prediction.run_name}")


def _project_root_to_image_distance(
    root_cam: np.ndarray,
    K: np.ndarray,
    k: np.ndarray,
    *,
    width: float,
    height: float,
) -> np.ndarray:
    result = np.full(root_cam.shape[0], np.nan, dtype=np.float64)
    z = root_cam[:, 2]
    finite = np.isfinite(root_cam).all(axis=1) & np.isfinite(K).all(axis=(1, 2)) & (z > 1e-8)
    if not finite.any() or width <= 0.0 or height <= 0.0:
        return result
    xy = root_cam[finite, :2] / z[finite, None]
    r2 = np.sum(xy * xy, axis=1)
    radial = 1.0 + k[finite, 0] * r2 + k[finite, 1] * r2 * r2
    distorted = xy * radial[:, None]
    u = K[finite, 0, 0] * distorted[:, 0] + K[finite, 0, 2]
    v = K[finite, 1, 1] * distorted[:, 1] + K[finite, 1, 2]
    result[finite] = np.sqrt(((u / width) - 0.5) ** 2 + ((v / height) - 0.5) ** 2)
    return result


def _world_speed(
    root_world: np.ndarray,
    valid_mask: np.ndarray,
    *,
    fps: float,
    window_frames: int,
) -> np.ndarray:
    speed = np.full(valid_mask.shape, np.nan, dtype=np.float64)
    step = max(1, int(window_frames))
    if root_world.shape[1] <= step:
        return speed
    delta = root_world[:, step:] - root_world[:, :-step]
    pair_valid = (
        valid_mask[:, step:]
        & valid_mask[:, :-step]
        & np.isfinite(root_world[:, step:]).all(axis=-1)
        & np.isfinite(root_world[:, :-step]).all(axis=-1)
    )
    values = np.linalg.norm(delta, axis=-1) * float(fps) / float(step)
    speed[:, step:] = np.where(pair_valid, values, np.nan)
    return speed


def _load_root_init(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    with np.load(path, allow_pickle=True) as npz:
        for key in ("root_init_cam", "root_init", "arr_0"):
            if key in npz.files:
                return np.asarray(npz[key], dtype=np.float64)
    raise KeyError(f"No root-init array found in {path}")


def _build_diagnostics(
    *,
    reference: PredictionSet,
    raw_features_dir: Path,
    raw_root_init_dir: Path,
    fps: float,
    speed_window_frames: int,
    airborne_threshold_m: float,
    ground_reference_percentile: float,
    ground_reference_window_frames: int,
    min_airborne_frames: int,
    max_ground_gap_frames: int,
) -> DiagnosticData:
    count = reference.sequence_id.size
    float_arrays = {
        name: np.full(count, np.nan, dtype=np.float64)
        for name in (
            "lower_foot_clearance_m",
            "camera_distance_m",
            "image_center_distance_norm",
            "bbox_height_ratio",
            "world_speed_mps",
            "pelvis_height_m",
            "valid_joint_fraction",
            "root_init_error_m",
        )
    }
    root_init_component_error_m = np.full((count, 3), np.nan, dtype=np.float64)
    raw_root_gt_m = np.full((count, 3), np.nan, dtype=np.float64)
    classifiable = np.zeros(count, dtype=bool)
    airborne = np.zeros(count, dtype=bool)

    for sequence_id in sorted(int(value) for value in np.unique(reference.sequence_id)):
        sequence = reference.sequence_names[sequence_id]
        rows = np.flatnonzero(reference.sequence_id == sequence_id)
        people = reference.person_idx[rows]
        frames = reference.frame_idx[rows]
        feature_path = raw_features_dir / f"{sequence}.npz"
        root_init_path = raw_root_init_dir / f"{sequence}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing raw features: {feature_path}")
        if not root_init_path.exists():
            alternative = raw_root_init_dir / f"{sequence}.npz"
            if not alternative.exists():
                raise FileNotFoundError(f"Missing raw root initialization: {root_init_path}")
            root_init_path = alternative

        with np.load(feature_path, allow_pickle=True) as npz:
            required = (
                "Y_cam_gt",
                "Y_root_cam_gt",
                "R",
                "t",
                "K",
                "k",
                "valid_mask",
                "valid_joints",
                "boxes_xyxy",
                "image_size",
                "pitch_points_world",
            )
            missing = [key for key in required if key not in npz.files]
            if missing:
                raise KeyError(f"{feature_path}: missing keys {missing}")
            foot_indices = (*DEFAULT_LEFT_FOOT_JOINTS, *DEFAULT_RIGHT_FOOT_JOINTS)
            feet_cam_gt = np.asarray(npz["Y_cam_gt"][:, :, foot_indices, :], dtype=np.float64)
            root_cam_gt = np.asarray(npz["Y_root_cam_gt"], dtype=np.float64)
            R = np.asarray(npz["R"], dtype=np.float64)
            t = np.asarray(npz["t"], dtype=np.float64)
            K = np.asarray(npz["K"], dtype=np.float64)
            distortion = np.asarray(npz["k"], dtype=np.float64)
            valid_mask = np.asarray(npz["valid_mask"], dtype=bool)
            valid_joints = np.asarray(npz["valid_joints"], dtype=bool)
            boxes = np.asarray(npz["boxes_xyxy"], dtype=np.float64)
            image_size = np.asarray(npz["image_size"], dtype=np.float64).reshape(-1)
            pitch_points_world = np.asarray(npz["pitch_points_world"], dtype=np.float64)

        in_bounds = (
            (people >= 0)
            & (people < root_cam_gt.shape[0])
            & (frames >= 0)
            & (frames < root_cam_gt.shape[1])
        )
        if not in_bounds.all():
            raise ValueError(f"Out-of-bounds prediction metadata for {sequence}")

        plane_normal, plane_offset = fit_pitch_plane(pitch_points_world)
        root_world = _camera_roots_to_world(root_cam_gt, R, t)
        root_signed_height = np.einsum("ntc,c->nt", root_world, plane_normal) + plane_offset
        finite_root_height = root_signed_height[valid_mask & np.isfinite(root_signed_height)]
        if finite_root_height.size and float(np.median(finite_root_height)) < 0.0:
            plane_normal = -plane_normal
            plane_offset = -plane_offset
            root_signed_height = -root_signed_height

        left_plane, right_plane, _ = compute_foot_heights(
            Y_cam_gt=feet_cam_gt,
            root_cam_gt=root_cam_gt,
            R=R,
            t=t,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
            left_foot_joints=(0, 1, 2),
            right_foot_joints=(3, 4, 5),
        )
        foot_valid = valid_mask & np.isfinite(left_plane) & np.isfinite(right_plane)
        left_clearance, right_clearance, lower_clearance, _, _ = calibrate_foot_clearance(
            left_plane,
            right_plane,
            foot_valid,
            ground_reference_percentile=ground_reference_percentile,
            ground_reference_window_frames=ground_reference_window_frames,
        )
        foot_valid &= np.isfinite(left_clearance) & np.isfinite(right_clearance)
        sequence_airborne = np.zeros_like(valid_mask)
        for person in np.unique(people):
            candidate = (
                foot_valid[person]
                & (left_clearance[person] >= airborne_threshold_m)
                & (right_clearance[person] >= airborne_threshold_m)
            )
            sequence_airborne[person] = clean_airborne_mask(
                candidate,
                foot_valid[person],
                min_airborne_frames=min_airborne_frames,
                max_ground_gap_frames=max_ground_gap_frames,
            )

        root_init_cam = _load_root_init(root_init_path)
        if root_init_cam.shape != root_cam_gt.shape:
            raise ValueError(f"Root-init shape mismatch for {sequence}: {root_init_cam.shape} vs {root_cam_gt.shape}")
        speed = _world_speed(root_world, valid_mask, fps=fps, window_frames=speed_window_frames)

        row_root = root_cam_gt[people, frames]
        row_init = root_init_cam[people, frames]
        row_boxes = boxes[people, frames]
        row_K = K[frames]
        row_distortion = distortion[frames]
        width, height = float(image_size[0]), float(image_size[1])

        raw_root_gt_m[rows] = row_root
        float_arrays["camera_distance_m"][rows] = np.linalg.norm(row_root, axis=-1)
        float_arrays["image_center_distance_norm"][rows] = _project_root_to_image_distance(
            row_root,
            row_K,
            row_distortion,
            width=width,
            height=height,
        )
        float_arrays["bbox_height_ratio"][rows] = (row_boxes[:, 3] - row_boxes[:, 1]) / height
        float_arrays["world_speed_mps"][rows] = speed[people, frames]
        float_arrays["pelvis_height_m"][rows] = np.maximum(root_signed_height[people, frames], 0.0)
        float_arrays["valid_joint_fraction"][rows] = valid_joints[people, frames].mean(axis=-1)
        float_arrays["lower_foot_clearance_m"][rows] = lower_clearance[people, frames]
        root_init_delta = row_init - row_root
        float_arrays["root_init_error_m"][rows] = np.linalg.norm(root_init_delta, axis=-1)
        root_init_component_error_m[rows] = np.abs(root_init_delta)
        classifiable[rows] = foot_valid[people, frames]
        airborne[rows] = sequence_airborne[people, frames]

    raw_root_gt_sorted = raw_root_gt_m[reference.sort_order]
    finite = np.isfinite(raw_root_gt_sorted).all(axis=-1) & np.isfinite(reference.root_gt_m).all(axis=-1)
    if finite.any() and not np.allclose(raw_root_gt_sorted[finite], reference.root_gt_m[finite], atol=1e-4):
        max_difference = float(np.max(np.abs(raw_root_gt_sorted[finite] - reference.root_gt_m[finite])))
        raise ValueError(f"Raw GT and prediction GT are misaligned (max difference {max_difference:.6g} m)")

    sequence_unsorted = np.asarray(
        [reference.sequence_names[int(value)] for value in reference.sequence_id], dtype=object
    )
    sequence_sorted = sequence_unsorted[reference.sort_order]
    cluster_names = tuple(sorted(set(str(value) for value in sequence_sorted.tolist())))
    cluster_rank = {name: index for index, name in enumerate(cluster_names)}
    clusters = np.asarray([cluster_rank[str(value)] for value in sequence_sorted], dtype=np.int32)
    order = reference.sort_order
    return DiagnosticData(
        sequence=sequence_sorted,
        cluster=clusters,
        person_idx=reference.person_idx[order],
        frame_idx=reference.frame_idx[order],
        classifiable=classifiable[order],
        airborne=airborne[order],
        lower_foot_clearance_m=float_arrays["lower_foot_clearance_m"][order],
        camera_distance_m=float_arrays["camera_distance_m"][order],
        image_center_distance_norm=float_arrays["image_center_distance_norm"][order],
        bbox_height_ratio=float_arrays["bbox_height_ratio"][order],
        world_speed_mps=float_arrays["world_speed_mps"][order],
        pelvis_height_m=float_arrays["pelvis_height_m"][order],
        valid_joint_fraction=float_arrays["valid_joint_fraction"][order],
        root_init_error_m=float_arrays["root_init_error_m"][order],
        root_init_component_error_m=root_init_component_error_m[order],
    )


def _cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float, int, int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(clusters).reshape(-1)
    finite = np.isfinite(values)
    values = values[finite]
    clusters = clusters[finite]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), 0, 0
    unique, inverse = np.unique(clusters, return_inverse=True)
    mean = float(np.mean(values))
    if unique.size < 2 or samples <= 0:
        return mean, float("nan"), float("nan"), int(values.size), int(unique.size)
    sums = np.bincount(inverse, weights=values, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(samples, unique.size))
    bootstrap = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1.0)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return mean, float(low), float(high), int(values.size), int(unique.size)


def _paired_phase_contrast(
    values: np.ndarray,
    clusters: np.ndarray,
    ground_mask: np.ndarray,
    airborne_mask: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    cluster_ids = np.intersect1d(np.unique(clusters[ground_mask]), np.unique(clusters[airborne_mask]))
    if cluster_ids.size == 0:
        return float("nan"), float("nan"), float("nan")
    ground_sums = np.asarray([np.nansum(values[ground_mask & (clusters == key)]) for key in cluster_ids])
    ground_counts = np.asarray([np.isfinite(values[ground_mask & (clusters == key)]).sum() for key in cluster_ids])
    air_sums = np.asarray([np.nansum(values[airborne_mask & (clusters == key)]) for key in cluster_ids])
    air_counts = np.asarray([np.isfinite(values[airborne_mask & (clusters == key)]).sum() for key in cluster_ids])
    valid = (ground_counts > 0) & (air_counts > 0)
    ground_sums, ground_counts = ground_sums[valid], ground_counts[valid]
    air_sums, air_counts = air_sums[valid], air_counts[valid]
    if ground_counts.size == 0:
        return float("nan"), float("nan"), float("nan")
    estimate = float(air_sums.sum() / air_counts.sum() - ground_sums.sum() / ground_counts.sum())
    if ground_counts.size < 2 or samples <= 0:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, ground_counts.size, size=(samples, ground_counts.size))
    ground_boot = ground_sums[draws].sum(axis=1) / ground_counts[draws].sum(axis=1)
    air_boot = air_sums[draws].sum(axis=1) / air_counts[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(air_boot - ground_boot, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def _quantile_edges(values: np.ndarray, *, bins: int) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.asarray([], dtype=np.float64)
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, max(2, bins + 1))))
    if edges.size < 2:
        value = float(edges[0])
        epsilon = max(abs(value) * 1e-6, 1e-9)
        edges = np.asarray([value - epsilon, value + epsilon], dtype=np.float64)
    return edges


def _fixed_foot_edges(values: np.ndarray, *, bin_width_m: float) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite) & (finite >= 0.0)]
    upper = 0.30
    if finite.size:
        upper = min(0.60, max(0.30, float(np.quantile(finite, 0.995))))
    edges = np.arange(0.0, upper + bin_width_m, bin_width_m, dtype=np.float64)
    if edges[-1] < upper:
        edges = np.append(edges, upper)
    return edges


def _binned_rows(
    *,
    methods: Sequence[MethodSeries],
    x: np.ndarray,
    clusters: np.ndarray,
    edges: np.ndarray,
    variable: str,
    samples: int,
    confidence: float,
    seed: int,
    min_frames: int,
    value_override: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if edges.size < 2:
        return rows
    for method_index, method in enumerate(methods):
        y = value_override[method.key] if value_override is not None else method.root_error_m
        for bin_index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            interval = (x >= lower) & ((x < upper) if bin_index < edges.size - 2 else (x <= upper))
            mask = interval & np.isfinite(x) & np.isfinite(y)
            if int(mask.sum()) < min_frames:
                continue
            mean, low, high, frame_count, cluster_count = _cluster_bootstrap_mean(
                y[mask],
                clusters[mask],
                samples=samples,
                confidence=confidence,
                seed=seed + method_index * 1009 + bin_index * 37,
            )
            rows.append(
                {
                    "variable": variable,
                    "method": method.table_label,
                    "method_key": method.key,
                    "bin_index": bin_index,
                    "bin_start": float(lower),
                    "bin_end": float(upper),
                    "x_median": float(np.median(x[mask])),
                    "frame_count": frame_count,
                    "sequence_count": cluster_count,
                    "mean_value": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": confidence,
                }
            )
    return rows


def _plot_binned_curves(
    *,
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[MethodSeries],
    figures_dir: Path,
    name: str,
    xlabel: str,
    ylabel: str = "Mean root translation error (cm)",
    scale: float = 100.0,
    zero_line: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.35), constrained_layout=True)
    for method in methods:
        selected = [row for row in rows if row["method_key"] == method.key]
        selected.sort(key=lambda row: int(row["bin_index"]))
        if not selected:
            continue
        x = np.asarray([float(row["x_median"]) for row in selected])
        mean = scale * np.asarray([float(row["mean_value"]) for row in selected])
        low = scale * np.asarray([float(row["ci_low"]) for row in selected])
        high = scale * np.asarray([float(row["ci_high"]) for row in selected])
        ax.plot(
            x,
            mean,
            color=method.color,
            marker=method.marker,
            markersize=4.5,
            linewidth=1.8,
            label=method.label,
        )
        finite_ci = np.isfinite(low) & np.isfinite(high)
        if finite_ci.any():
            ax.fill_between(x[finite_ci], low[finite_ci], high[finite_ci], color=method.color, alpha=0.16)
    if zero_line:
        ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--")
    else:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.legend(frameon=False, ncol=min(2, len(methods)))
    _save_figure(fig, figures_dir, name)


def _plot_ground_vs_airborne(
    *,
    methods: Sequence[MethodSeries],
    diagnostics: DiagnosticData,
    figures_dir: Path,
    data_dir: Path,
    samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ground = diagnostics.classifiable & ~diagnostics.airborne
    air = diagnostics.classifiable & diagnostics.airborne
    categories = (("Ground / non-jump", ground), ("Airborne / jump", air))
    for method_index, method in enumerate(methods):
        for category_index, (category, mask) in enumerate(categories):
            mean, low, high, frame_count, cluster_count = _cluster_bootstrap_mean(
                method.root_error_m[mask],
                diagnostics.cluster[mask],
                samples=samples,
                confidence=confidence,
                seed=seed + method_index * 101 + category_index * 17,
            )
            rows.append(
                {
                    "method": method.table_label,
                    "method_key": method.key,
                    "phase": category,
                    "mean_root_error_m": mean,
                    "ci_low_m": low,
                    "ci_high_m": high,
                    "frame_count": frame_count,
                    "sequence_count": cluster_count,
                    "confidence_level": confidence,
                }
            )
        delta, delta_low, delta_high = _paired_phase_contrast(
            method.root_error_m,
            diagnostics.cluster,
            ground,
            air,
            samples=samples,
            confidence=confidence,
            seed=seed + method_index * 211,
        )
        rows.append(
            {
                "method": method.table_label,
                "method_key": method.key,
                "phase": "Airborne minus ground",
                "mean_root_error_m": delta,
                "ci_low_m": delta_low,
                "ci_high_m": delta_high,
                "frame_count": int(air.sum()),
                "sequence_count": int(np.unique(diagnostics.cluster[air]).size),
                "confidence_level": confidence,
            }
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.35), constrained_layout=True)
    x = np.arange(2, dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(methods)) if len(methods) > 1 else np.asarray([0.0])
    for offset, method in zip(offsets, methods):
        selected = [
            row for row in rows if row["method_key"] == method.key and row["phase"] != "Airborne minus ground"
        ]
        means = 100.0 * np.asarray([float(row["mean_root_error_m"]) for row in selected])
        lows = 100.0 * np.asarray([float(row["ci_low_m"]) for row in selected])
        highs = 100.0 * np.asarray([float(row["ci_high_m"]) for row in selected])
        errors = np.vstack((means - lows, highs - means))
        ax.errorbar(
            x + offset,
            means,
            yerr=errors,
            color=method.color,
            marker=method.marker,
            markersize=6,
            linewidth=1.5,
            capsize=3,
            label=method.label,
        )
    ground_count, air_count = int(ground.sum()), int(air.sum())
    ax.set_xticks(
        x,
        [f"Ground / non-jump\n$n={ground_count:,}$ frames", f"Airborne / jump\n$n={air_count:,}$ frames"],
    )
    ax.set_ylabel("Mean root translation error (cm)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir, "01_root_error_ground_vs_airborne")
    _write_csv(data_dir / "01_root_error_ground_vs_airborne.csv", rows)
    return rows


def _plot_condition_curves(
    *,
    methods: Sequence[MethodSeries],
    diagnostics: DiagnosticData,
    figures_dir: Path,
    data_dir: Path,
    samples: int,
    confidence: float,
    seed: int,
    foot_bin_width_m: float,
    quantile_bins: int,
    min_frames: int,
) -> dict[str, list[dict[str, Any]]]:
    classifiable_foot = np.where(
        diagnostics.classifiable,
        diagnostics.lower_foot_clearance_m,
        np.nan,
    )
    specifications = (
        (
            "02_root_error_vs_foot_clearance",
            "lower_foot_clearance_m",
            classifiable_foot,
            _fixed_foot_edges(classifiable_foot, bin_width_m=foot_bin_width_m),
            r"GT lower-foot clearance $h^{\mathrm{foot}}_{i,t}$ (m)",
        ),
        (
            "03_root_error_vs_camera_distance",
            "camera_distance_m",
            diagnostics.camera_distance_m,
            _quantile_edges(diagnostics.camera_distance_m, bins=quantile_bins),
            r"Camera-to-root distance $\|\mathbf{r}^{c,\mathrm{gt}}_{i,t}\|_2$ (m)",
        ),
        (
            "04_root_error_vs_image_center_distance",
            "image_center_distance_norm",
            diagnostics.image_center_distance_norm,
            _quantile_edges(diagnostics.image_center_distance_norm, bins=quantile_bins),
            "Normalized projected-root distance to image center",
        ),
        (
            "05_root_error_vs_bbox_height",
            "bbox_height_ratio",
            diagnostics.bbox_height_ratio,
            _quantile_edges(diagnostics.bbox_height_ratio, bins=quantile_bins),
            r"Bounding-box height $h_{i,t}/H$",
        ),
        (
            "06_root_error_vs_player_speed",
            "world_speed_mps",
            diagnostics.world_speed_mps,
            _quantile_edges(diagnostics.world_speed_mps, bins=quantile_bins),
            r"GT world-root speed $\|\mathbf{v}^{w,\mathrm{gt}}_{i,t}\|_2$ (m s$^{-1}$)",
        ),
        (
            "07_root_error_vs_pelvis_height",
            "pelvis_height_m",
            diagnostics.pelvis_height_m,
            _quantile_edges(diagnostics.pelvis_height_m, bins=quantile_bins),
            r"GT pelvis height above pitch plane $\pi$ (m)",
        ),
        (
            "08_root_error_vs_initial_error",
            "root_init_error_m",
            diagnostics.root_init_error_m,
            _quantile_edges(diagnostics.root_init_error_m, bins=quantile_bins),
            r"Initialization error $\|\mathbf{r}^{c,\mathrm{init}}_{i,t}-\mathbf{r}^{c,\mathrm{gt}}_{i,t}\|_2$ (m)",
        ),
    )
    outputs: dict[str, list[dict[str, Any]]] = {}
    for figure_index, (name, variable, values, edges, xlabel) in enumerate(specifications):
        rows = _binned_rows(
            methods=methods,
            x=values,
            clusters=diagnostics.cluster,
            edges=edges,
            variable=variable,
            samples=samples,
            confidence=confidence,
            seed=seed + figure_index * 10007,
            min_frames=min_frames,
        )
        _plot_binned_curves(
            rows=rows,
            methods=methods,
            figures_dir=figures_dir,
            name=name,
            xlabel=xlabel,
        )
        _write_csv(data_dir / f"{name}.csv", rows)
        outputs[name] = rows
    return outputs


def _plot_residual_gain(
    *,
    methods: Sequence[MethodSeries],
    diagnostics: DiagnosticData,
    figures_dir: Path,
    data_dir: Path,
    samples: int,
    confidence: float,
    seed: int,
    quantile_bins: int,
    min_frames: int,
) -> list[dict[str, Any]]:
    learned = [method for method in methods if method.key != "geometry"]
    gains = {
        method.key: diagnostics.root_init_error_m - method.root_error_m
        for method in learned
    }
    rows = _binned_rows(
        methods=learned,
        x=diagnostics.root_init_error_m,
        clusters=diagnostics.cluster,
        edges=_quantile_edges(diagnostics.root_init_error_m, bins=quantile_bins),
        variable="root_error_reduction_m",
        samples=samples,
        confidence=confidence,
        seed=seed,
        min_frames=min_frames,
        value_override=gains,
    )
    name = "09_refinement_gain_vs_initial_error"
    _plot_binned_curves(
        rows=rows,
        methods=learned,
        figures_dir=figures_dir,
        name=name,
        xlabel=r"Initialization error $\|\mathbf{r}^{c,\mathrm{init}}_{i,t}-\mathbf{r}^{c,\mathrm{gt}}_{i,t}\|_2$ (m)",
        ylabel="Mean error reduction after refinement (cm)",
        zero_line=True,
    )
    _write_csv(data_dir / f"{name}.csv", rows)
    return rows


def _plot_axis_errors(
    *,
    methods: Sequence[MethodSeries],
    diagnostics: DiagnosticData,
    figures_dir: Path,
    data_dir: Path,
    samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        for axis in range(3):
            mean, low, high, frame_count, cluster_count = _cluster_bootstrap_mean(
                method.component_error_m[:, axis],
                diagnostics.cluster,
                samples=samples,
                confidence=confidence,
                seed=seed + method_index * 101 + axis * 13,
            )
            rows.append(
                {
                    "method": method.table_label,
                    "method_key": method.key,
                    "axis": ("x", "y", "z")[axis],
                    "mean_absolute_error_m": mean,
                    "ci_low_m": low,
                    "ci_high_m": high,
                    "frame_count": frame_count,
                    "sequence_count": cluster_count,
                    "confidence_level": confidence,
                }
            )

    fig, ax = plt.subplots(figsize=(6.8, 4.25), constrained_layout=True)
    x = np.arange(3, dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(methods)) if len(methods) > 1 else np.asarray([0.0])
    for offset, method in zip(offsets, methods):
        selected = [row for row in rows if row["method_key"] == method.key]
        means = 100.0 * np.asarray([float(row["mean_absolute_error_m"]) for row in selected])
        lows = 100.0 * np.asarray([float(row["ci_low_m"]) for row in selected])
        highs = 100.0 * np.asarray([float(row["ci_high_m"]) for row in selected])
        ax.errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - lows, highs - means)),
            color=method.color,
            marker=method.marker,
            markersize=6,
            linewidth=1.5,
            capsize=3,
            label=method.label,
        )
    ax.set_xticks(x, AXIS_LABELS)
    ax.set_ylabel("Mean absolute root error (cm)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.legend(frameon=False)
    name = "10_root_error_by_camera_axis"
    _save_figure(fig, figures_dir, name)
    _write_csv(data_dir / f"{name}.csv", rows)
    return rows


def _plot_error_ecdf(
    *, methods: Sequence[MethodSeries], figures_dir: Path, data_dir: Path
) -> list[dict[str, Any]]:
    finite_values = [method.root_error_m[np.isfinite(method.root_error_m)] for method in methods]
    positive = np.concatenate([values[values > 0.0] for values in finite_values if values.size])
    if positive.size == 0:
        return []
    lower = max(0.1, 100.0 * float(np.quantile(positive, 0.0005)))
    upper = max(lower * 1.01, 100.0 * float(np.quantile(positive, 0.9995)))
    grid_cm = np.geomspace(lower, upper, 400)
    rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(6.8, 4.25), constrained_layout=True)
    for method in methods:
        values_cm = 100.0 * method.root_error_m[np.isfinite(method.root_error_m)]
        values_cm.sort()
        cdf = np.searchsorted(values_cm, grid_cm, side="right") / max(1, values_cm.size)
        ax.plot(grid_cm, cdf, color=method.color, linewidth=1.9, label=method.label)
        rows.extend(
            {
                "method": method.table_label,
                "method_key": method.key,
                "root_error_cm": float(x),
                "cumulative_fraction": float(y),
            }
            for x, y in zip(grid_cm, cdf)
        )
    ax.set_xscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(
        r"Root translation error "
        r"$\|\widehat{\mathbf{r}}^{c}_{i,t}-\mathbf{r}^{c,\mathrm{gt}}_{i,t}\|_2$ (cm)"
    )
    ax.set_ylabel("Cumulative fraction of evaluated frames")
    ax.grid(True, which="both", alpha=0.20, linewidth=0.6)
    ax.legend(frameon=False)
    name = "11_root_error_ecdf"
    _save_figure(fig, figures_dir, name)
    _write_csv(data_dir / f"{name}.csv", rows)
    return rows


def _plot_per_sequence_error(
    *,
    methods: Sequence[MethodSeries],
    diagnostics: DiagnosticData,
    figures_dir: Path,
    data_dir: Path,
) -> list[dict[str, Any]]:
    sequence_names = sorted(set(str(value) for value in diagnostics.sequence.tolist()))
    rows: list[dict[str, Any]] = []
    values_by_method: dict[str, list[float]] = {method.key: [] for method in methods}
    for sequence in sequence_names:
        mask = diagnostics.sequence == sequence
        for method in methods:
            value = float(np.nanmean(method.root_error_m[mask]))
            values_by_method[method.key].append(100.0 * value)
            rows.append(
                {
                    "sequence": sequence,
                    "method": method.table_label,
                    "method_key": method.key,
                    "mean_root_error_cm": 100.0 * value,
                    "frame_count": int(np.isfinite(method.root_error_m[mask]).sum()),
                }
            )

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    x = np.arange(len(methods), dtype=float)
    matrix = np.asarray([values_by_method[method.key] for method in methods]).T
    for sequence_values in matrix:
        ax.plot(x, sequence_values, color="#B9BDC2", alpha=0.45, linewidth=0.7, zorder=1)
    for method_index, method in enumerate(methods):
        values = np.asarray(values_by_method[method.key])
        ax.scatter(
            np.full(values.shape, x[method_index]),
            values,
            color=method.color,
            edgecolor="white",
            linewidth=0.45,
            s=28,
            zorder=3,
        )
        ax.scatter(
            [x[method_index]],
            [float(np.mean(values))],
            color="#111111",
            marker="_",
            s=170,
            linewidth=2.0,
            zorder=4,
        )
    ax.set_xticks(x, [method.table_label for method in methods])
    ax.set_ylabel("Per-sequence mean root error (cm)")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    name = "12_per_sequence_root_error"
    _save_figure(fig, figures_dir, name)
    _write_csv(data_dir / f"{name}.csv", rows)
    return rows


def _root_temporal_error_samples(
    prediction: PredictionSet,
    *,
    fps: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    order = prediction.sort_order
    sequence = prediction.sequence_id[order]
    person = prediction.person_idx[order]
    frame = prediction.frame_idx[order]
    residual = prediction.root_world_pred_m - prediction.root_world_gt_m

    contiguous = (
        (sequence[1:] == sequence[:-1])
        & (person[1:] == person[:-1])
        & (frame[1:] == frame[:-1] + 1)
    )
    acceleration_contiguous = contiguous[:-1] & contiguous[1:]
    velocity_error = np.linalg.norm(np.diff(residual, axis=0) * fps, axis=1)
    acceleration_error = np.linalg.norm(np.diff(residual, n=2, axis=0) * fps**2, axis=1)
    return {
        "position": (prediction.root_error_m, sequence),
        "velocity": (velocity_error[contiguous], sequence[1:][contiguous]),
        "acceleration": (
            acceleration_error[acceleration_contiguous],
            sequence[1:-1][acceleration_contiguous],
        ),
    }


def _format_ci_cell(
    row: Mapping[str, Any],
    prefix: str,
    *,
    digits: int,
    bold: bool,
) -> str:
    values = (
        _metric_value(row, f"{prefix}_mean"),
        _metric_value(row, f"{prefix}_ci_low"),
        _metric_value(row, f"{prefix}_ci_high"),
    )
    if any(value is None for value in values):
        return "--"
    mean, low, high = values
    cell = f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
    return rf"\textbf{{{cell}}}" if bold else cell


def _plot_temporal_consistency(
    *,
    predictions: Mapping[str, PredictionSet],
    figures_dir: Path,
    data_dir: Path,
    tables_dir: Path,
    fps: float,
    samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    display_names = {"mlp": "MLP", "tcn": "TCN", "transformer": "Transformer"}
    units = {"position": 100.0, "velocity": 100.0, "acceleration": 1.0}
    output_names = {
        "position": "root_error_cm",
        "velocity": "root_velocity_error_cm_s",
        "acceleration": "root_acceleration_error_m_s2",
    }
    for method_index, architecture in enumerate(ARCHITECTURE_ORDER):
        prediction = predictions.get(architecture)
        if prediction is None:
            continue
        label = display_names[architecture]
        if prediction.window_size is not None:
            label = f"{label} ({prediction.window_size} f)"
        temporal_samples = _root_temporal_error_samples(prediction, fps=fps)
        row: dict[str, Any] = {
            "model": label,
            "method_key": architecture,
            "run_name": prediction.run_name,
            "fps": fps,
            "confidence_level": confidence,
        }
        for metric_index, metric in enumerate(("position", "velocity", "acceleration")):
            values, clusters = temporal_samples[metric]
            scale = units[metric]
            mean, low, high, frame_count, sequence_count = _cluster_bootstrap_mean(
                values * scale,
                clusters,
                samples=samples,
                confidence=confidence,
                seed=seed + method_index * 1009 + metric_index * 97,
            )
            prefix = output_names[metric]
            row.update(
                {
                    f"{prefix}_mean": mean,
                    f"{prefix}_ci_low": low,
                    f"{prefix}_ci_high": high,
                    f"{prefix}_sample_count": frame_count,
                    f"{prefix}_sequence_count": sequence_count,
                }
            )
        rows.append(row)

    if not rows:
        return rows
    mlp_row = next((row for row in rows if row["method_key"] == "mlp"), None)
    for row in rows:
        for prefix in ("root_velocity_error_cm_s", "root_acceleration_error_m_s2"):
            baseline = _metric_value(mlp_row, f"{prefix}_mean")
            value = _metric_value(row, f"{prefix}_mean")
            row[f"{prefix}_reduction_vs_mlp_percent"] = (
                100.0 * (baseline - value) / baseline
                if baseline is not None and baseline > 0.0 and value is not None
                else None
            )

    name = "13_root_temporal_consistency"
    _write_csv(data_dir / f"{name}.csv", rows)
    figure_metrics = (
        ("root_velocity_error_cm_s", "(a) Root velocity error", r"Mean error (cm s$^{-1}$)"),
        ("root_acceleration_error_m_s2", "(b) Root acceleration error", r"Mean error (m s$^{-2}$)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.55), constrained_layout=True)
    x = np.arange(len(rows), dtype=float)
    labels = [str(row["model"]).replace(" (", "\n(") for row in rows]
    for ax, (prefix, title, ylabel) in zip(axes, figure_metrics):
        for index, row in enumerate(rows):
            mean = float(row[f"{prefix}_mean"])
            low = float(row[f"{prefix}_ci_low"])
            high = float(row[f"{prefix}_ci_high"])
            key = str(row["method_key"])
            ax.errorbar(
                [x[index]],
                [mean],
                yerr=[[mean - low], [high - mean]],
                fmt=METHOD_MARKERS[key],
                markersize=7,
                color=METHOD_COLORS[key],
                markeredgecolor="white",
                markeredgewidth=0.6,
                elinewidth=1.4,
                capsize=3.0,
                zorder=3,
            )
        ax.set_xticks(x, labels)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    _save_figure(fig, figures_dir, name)

    best = {
        prefix: min(float(row[f"{prefix}_mean"]) for row in rows)
        for prefix in (
            "root_error_cm",
            "root_velocity_error_cm_s",
            "root_acceleration_error_m_s2",
        )
    }
    _write_latex_table(
        tables_dir / "table_temporal_consistency.tex",
        alignment="lrrr",
        headers=(
            "Model",
            r"Root error $\downarrow$ (cm)",
            r"Root velocity error $\downarrow$ (cm s$^{-1}$)",
            r"Root acceleration error $\downarrow$ (m s$^{-2}$)",
        ),
        rows=[
            [
                _latex_escape(str(row["model"])),
                _format_ci_cell(
                    row,
                    "root_error_cm",
                    digits=2,
                    bold=_is_best(float(row["root_error_cm_mean"]), best["root_error_cm"]),
                ),
                _format_ci_cell(
                    row,
                    "root_velocity_error_cm_s",
                    digits=1,
                    bold=_is_best(
                        float(row["root_velocity_error_cm_s_mean"]),
                        best["root_velocity_error_cm_s"],
                    ),
                ),
                _format_ci_cell(
                    row,
                    "root_acceleration_error_m_s2",
                    digits=1,
                    bold=_is_best(
                        float(row["root_acceleration_error_m_s2_mean"]),
                        best["root_acceleration_error_m_s2"],
                    ),
                ),
            ]
            for row in rows
        ],
    )
    _write_csv(tables_dir / "table_temporal_consistency.csv", rows)
    return rows


def _cluster_bootstrap_relative_gain(
    mlp_error: np.ndarray,
    temporal_error: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float]:
    mlp_error = np.asarray(mlp_error, dtype=np.float64).reshape(-1)
    temporal_error = np.asarray(temporal_error, dtype=np.float64).reshape(-1)
    clusters = np.asarray(clusters).reshape(-1)
    finite = np.isfinite(mlp_error) & np.isfinite(temporal_error)
    mlp_error = mlp_error[finite]
    temporal_error = temporal_error[finite]
    clusters = clusters[finite]
    if mlp_error.size == 0 or float(mlp_error.sum()) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    gain = mlp_error - temporal_error
    estimate = float(100.0 * gain.sum() / mlp_error.sum())
    unique, inverse = np.unique(clusters, return_inverse=True)
    if unique.size < 2 or samples <= 0:
        return estimate, float("nan"), float("nan")
    mlp_sums = np.bincount(inverse, weights=mlp_error, minlength=unique.size)
    gain_sums = np.bincount(inverse, weights=gain, minlength=unique.size)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(samples, unique.size))
    denominators = mlp_sums[draws].sum(axis=1)
    bootstrap = 100.0 * gain_sums[draws].sum(axis=1) / np.maximum(denominators, 1e-12)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap, [alpha, 1.0 - alpha])
    return estimate, float(low), float(high)


def _plot_temporal_gain_by_regime(
    *,
    methods: Sequence[MethodSeries],
    diagnostics: DiagnosticData,
    figures_dir: Path,
    data_dir: Path,
    tables_dir: Path,
    samples: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    by_key = {method.key: method for method in methods}
    mlp = by_key.get("mlp")
    temporal_methods = [
        by_key[key] for key in ("tcn", "transformer") if key in by_key
    ]
    if mlp is None or not temporal_methods:
        return []

    finite_speed = np.isfinite(diagnostics.world_speed_mps)
    finite_initial = np.isfinite(diagnostics.root_init_error_m)
    speed_threshold = float(np.quantile(diagnostics.world_speed_mps[finite_speed], 0.75))
    initial_threshold = float(np.quantile(diagnostics.root_init_error_m[finite_initial], 0.75))
    grounded = diagnostics.classifiable & ~diagnostics.airborne
    airborne = diagnostics.classifiable & diagnostics.airborne
    high_speed = finite_speed & (diagnostics.world_speed_mps >= speed_threshold)
    high_initial_error = finite_initial & (diagnostics.root_init_error_m >= initial_threshold)
    regimes = (
        ("overall", "All frames", "All frames", np.ones(mlp.root_error_m.shape, dtype=bool), None),
        ("grounded", "Grounded", "Grounded", grounded, None),
        ("airborne", "Airborne", "Airborne", airborne, grounded),
        (
            "high_speed",
            f"High speed (top 25%; threshold {speed_threshold:.2f} m/s)",
            "High speed (top 25%)",
            high_speed,
            finite_speed & ~high_speed,
        ),
        (
            "large_initial_error",
            f"Large initial error (top 25%; threshold {initial_threshold * 100.0:.1f} cm)",
            "Large init. error (top 25%)",
            high_initial_error,
            finite_initial & ~high_initial_error,
        ),
    )

    rows: list[dict[str, Any]] = []
    for regime_index, (regime_key, regime_label, plot_label, regime_mask, complement) in enumerate(regimes):
        for method_index, method in enumerate(temporal_methods):
            valid = (
                regime_mask
                & np.isfinite(mlp.root_error_m)
                & np.isfinite(method.root_error_m)
            )
            mlp_error_cm = mlp.root_error_m[valid] * 100.0
            temporal_error_cm = method.root_error_m[valid] * 100.0
            clusters = diagnostics.cluster[valid]
            mlp_mean, mlp_low, mlp_high, frame_count, sequence_count = _cluster_bootstrap_mean(
                mlp_error_cm,
                clusters,
                samples=samples,
                confidence=confidence,
                seed=seed + regime_index * 101,
            )
            temporal_mean, temporal_low, temporal_high, _, _ = _cluster_bootstrap_mean(
                temporal_error_cm,
                clusters,
                samples=samples,
                confidence=confidence,
                seed=seed + regime_index * 101 + method_index * 1009 + 17,
            )
            gain_mean, gain_low, gain_high, _, _ = _cluster_bootstrap_mean(
                mlp_error_cm - temporal_error_cm,
                clusters,
                samples=samples,
                confidence=confidence,
                seed=seed + regime_index * 101 + method_index * 1009 + 31,
            )
            relative_mean, relative_low, relative_high = _cluster_bootstrap_relative_gain(
                mlp_error_cm,
                temporal_error_cm,
                clusters,
                samples=samples,
                confidence=confidence,
                seed=seed + regime_index * 101 + method_index * 1009 + 47,
            )
            contrast_mean = contrast_low = contrast_high = None
            if complement is not None:
                full_gain_cm = (mlp.root_error_m - method.root_error_m) * 100.0
                contrast_mean, contrast_low, contrast_high = _paired_phase_contrast(
                    full_gain_cm,
                    diagnostics.cluster,
                    complement,
                    regime_mask,
                    samples=samples,
                    confidence=confidence,
                    seed=seed + regime_index * 101 + method_index * 1009 + 61,
                )
            rows.append(
                {
                    "regime": regime_label,
                    "regime_key": regime_key,
                    "plot_label": plot_label,
                    "method": method.table_label,
                    "method_key": method.key,
                    "frame_count": frame_count,
                    "sequence_count": sequence_count,
                    "mlp_root_error_cm_mean": mlp_mean,
                    "mlp_root_error_cm_ci_low": mlp_low,
                    "mlp_root_error_cm_ci_high": mlp_high,
                    "temporal_root_error_cm_mean": temporal_mean,
                    "temporal_root_error_cm_ci_low": temporal_low,
                    "temporal_root_error_cm_ci_high": temporal_high,
                    "absolute_gain_cm_mean": gain_mean,
                    "absolute_gain_cm_ci_low": gain_low,
                    "absolute_gain_cm_ci_high": gain_high,
                    "relative_gain_percent_mean": relative_mean,
                    "relative_gain_percent_ci_low": relative_low,
                    "relative_gain_percent_ci_high": relative_high,
                    "gain_contrast_vs_complement_cm_mean": contrast_mean,
                    "gain_contrast_vs_complement_cm_ci_low": contrast_low,
                    "gain_contrast_vs_complement_cm_ci_high": contrast_high,
                    "speed_threshold_mps": speed_threshold,
                    "initial_error_threshold_cm": initial_threshold * 100.0,
                    "confidence_level": confidence,
                }
            )

    name = "14_temporal_gain_by_regime"
    _write_csv(data_dir / f"{name}.csv", rows)
    regime_keys = [regime[0] for regime in regimes]
    plot_labels = [regime[2].replace("\n", " ") for regime in regimes]
    y = np.arange(len(regime_keys), dtype=float)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 4.25),
        sharey=True,
        constrained_layout=True,
    )
    plot_metrics = (
        ("absolute_gain_cm", "(a) Absolute gain", "Root-error reduction vs. MLP (cm)"),
        ("relative_gain_percent", "(b) Relative gain", "Root-error reduction vs. MLP (%)"),
    )
    offsets = np.linspace(-0.10, 0.10, len(temporal_methods))
    for ax, (prefix, title, xlabel) in zip(axes, plot_metrics):
        for method_index, method in enumerate(temporal_methods):
            method_rows = {
                str(row["regime_key"]): row
                for row in rows
                if row["method_key"] == method.key
            }
            for regime_index, regime_key in enumerate(regime_keys):
                row = method_rows[regime_key]
                mean = float(row[f"{prefix}_mean"])
                low = float(row[f"{prefix}_ci_low"])
                high = float(row[f"{prefix}_ci_high"])
                ax.errorbar(
                    [mean],
                    [y[regime_index] + offsets[method_index]],
                    xerr=[[mean - low], [high - mean]],
                    fmt=method.marker,
                    markersize=6.5,
                    color=method.color,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    elinewidth=1.3,
                    capsize=2.5,
                    label=(
                        method.table_label.replace(" residual", "")
                        if regime_index == 0
                        else None
                    ),
                    zorder=3,
                )
        ax.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_yticks(y, plot_labels)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=max(1, len(legend_labels)),
    )
    _save_figure(fig, figures_dir, name)

    row_lookup = {
        (str(row["regime_key"]), str(row["method_key"])): row for row in rows
    }

    def gain_cell(row: Mapping[str, Any] | None) -> str:
        if row is None:
            return "--"
        return (
            f"{float(row['absolute_gain_cm_mean']):.2f} "
            f"[{float(row['absolute_gain_cm_ci_low']):.2f}, "
            f"{float(row['absolute_gain_cm_ci_high']):.2f}] / "
            f"{float(row['relative_gain_percent_mean']):.1f}\%"
        )

    latex_rows: list[list[str]] = []
    for regime_key, regime_label, _, _, _ in regimes:
        tcn = row_lookup.get((regime_key, "tcn"))
        transformer = row_lookup.get((regime_key, "transformer"))
        reference = tcn or transformer
        latex_rows.append(
            [
                _latex_escape(regime_label),
                str(reference["frame_count"]) if reference is not None else "--",
                _latex_number(
                    _metric_value(reference, "mlp_root_error_cm_mean")
                    if reference is not None
                    else None
                ),
                _latex_number(_metric_value(tcn, "temporal_root_error_cm_mean")),
                gain_cell(tcn),
                _latex_number(_metric_value(transformer, "temporal_root_error_cm_mean")),
                gain_cell(transformer),
            ]
        )
    _write_latex_table(
        tables_dir / "table_temporal_gain_by_regime.tex",
        alignment="lrrrrrr",
        headers=(
            "Regime",
            "Frames",
            "MLP (cm)",
            "TCN (cm)",
            r"TCN gain (cm / \%)",
            "Transformer (cm)",
            r"Transformer gain (cm / \%)",
        ),
        rows=latex_rows,
    )
    _write_csv(tables_dir / "table_temporal_gain_by_regime.csv", rows)
    return rows


def _load_completed_metric_rows(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    fold: str,
    seed: int,
    split: str,
    selected_run_names: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_names = set((selected_run_names or {}).values())
    for run in _publication_runs(plan):
        run_name = str(run["run_name"])
        if str(run["fold"]) != fold:
            continue
        if int(run["seed"]) != seed and run_name not in selected_names:
            continue
        path = _metrics_path(output_dir, run_name)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload.get("splits", {}).get(split)
        if not isinstance(metrics, dict):
            continue
        rows.append({**dict(run), **metrics})
    return rows


def _load_geometry_metrics(output_dir: Path, *, fold: str, split: str) -> dict[str, Any] | None:
    path = output_dir / "baselines" / fold / "geometry_metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("splits", {}).get(split)
    return dict(metrics) if isinstance(metrics, dict) else None


def _checkpoint_parameter_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state, Mapping):
            return None
        return int(sum(int(tensor.numel()) for tensor in state.values() if hasattr(tensor, "numel")))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _metric_value(row: Mapping[str, Any] | None, key: str, *, scale: float = 1.0) -> float | None:
    if row is None:
        return None
    try:
        value = float(row[key]) * scale
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_number(value: float | None, *, digits: int = 2, bold: bool = False, suffix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "--"
    result = f"{value:.{digits}f}{suffix}"
    return rf"\textbf{{{result}}}" if bold else result


def _write_latex_table(
    path: Path,
    *,
    alignment: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    lines = [rf"\begin{{tabular}}{{{alignment}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _is_best(value: float | None, best: float | None) -> bool:
    return value is not None and best is not None and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12)


def _generate_main_table(
    *,
    tables_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    geometry_metrics: Mapping[str, Any] | None,
    selected_run_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_run_name = {str(row["run_name"]): row for row in metric_rows}
    by_architecture = {
        architecture: by_run_name[run_name]
        for architecture, run_name in selected_run_names.items()
        if run_name in by_run_name
    }

    def method_label(architecture: str, prefix: str) -> str:
        metrics = by_architecture.get(architecture)
        if metrics is None or architecture == "mlp":
            return prefix
        return f"{prefix} ({int(metrics['window_size'])} frames)"

    entries: list[tuple[str, Mapping[str, Any] | None, str]] = [
        ("Geometry initialization", geometry_metrics, "complete" if geometry_metrics else "pending"),
        (
            method_label("mlp", "MLP residual"),
            by_architecture.get("mlp"),
            "complete" if "mlp" in by_architecture else "pending",
        ),
        (
            method_label("tcn", "TCN residual"),
            by_architecture.get("tcn"),
            "complete" if "tcn" in by_architecture else "pending",
        ),
        (
            method_label("transformer", "Transformer residual"),
            by_architecture.get("transformer"),
            "complete" if "transformer" in by_architecture else "pending",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for method, metrics, status in entries:
        rows.append(
            {
                "method": method,
                "root_error_cm": _metric_value(metrics, "root_error_mean_m", scale=100.0),
                "world_mpjpe_cm": _metric_value(metrics, "MPJPE_world_m", scale=100.0),
                "local_mpjpe_cm": _metric_value(metrics, "MPJPE_local_m", scale=100.0),
                "reprojection_error_px": _metric_value(metrics, "reprojection_error_mean_px"),
                "status": status,
                "run_name": str(metrics.get("run_name", "")) if metrics else "",
            }
        )
    _write_csv(tables_dir / "table_main_results.csv", rows)
    complete = [row for row in rows if row["status"] == "complete"]
    best = {
        key: min(float(row[key]) for row in complete if row[key] is not None)
        for key in ("root_error_cm", "world_mpjpe_cm", "local_mpjpe_cm", "reprojection_error_px")
    }
    latex_rows = []
    for row in rows:
        latex_rows.append(
            [
                _latex_escape(str(row["method"])),
                _latex_number(row["root_error_cm"], bold=_is_best(row["root_error_cm"], best["root_error_cm"])),
                _latex_number(row["world_mpjpe_cm"], bold=_is_best(row["world_mpjpe_cm"], best["world_mpjpe_cm"])),
                _latex_number(row["local_mpjpe_cm"], bold=_is_best(row["local_mpjpe_cm"], best["local_mpjpe_cm"])),
                _latex_number(
                    row["reprojection_error_px"],
                    bold=_is_best(row["reprojection_error_px"], best["reprojection_error_px"]),
                ),
            ]
        )
    _write_latex_table(
        tables_dir / "table_main_results.tex",
        alignment="lrrrr",
        headers=(
            "Method",
            r"Root error $\downarrow$ (cm)",
            r"World MPJPE $\downarrow$ (cm)",
            r"Local MPJPE $\downarrow$ (cm)",
            r"Reproj. $\downarrow$ (px)",
        ),
        rows=latex_rows,
    )
    return rows


def _generate_architecture_table(
    *,
    tables_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    selected_run_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_run_name = {str(row["run_name"]): row for row in metric_rows}
    by_architecture = {
        architecture: by_run_name[run_name]
        for architecture, run_name in selected_run_names.items()
        if run_name in by_run_name
    }
    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_ORDER:
        metrics = by_architecture.get(architecture)
        run_name = str(metrics["run_name"]) if metrics else ""
        model = architecture.upper() if architecture != "transformer" else "Transformer"
        if metrics is not None and architecture != "mlp":
            model = f"{model} ({int(metrics['window_size'])} f)"
        parameter_count = (
            _checkpoint_parameter_count(output_dir / "checkpoints" / run_name / "best.pt") if run_name else None
        )
        rows.append(
            {
                "model": model,
                "root_error_cm": _metric_value(metrics, "root_error_mean_m", scale=100.0),
                "world_mpjpe_cm": _metric_value(metrics, "MPJPE_world_m", scale=100.0),
                "parameters": parameter_count,
                "parameters_millions": parameter_count / 1e6 if parameter_count is not None else None,
                "window_size_frames": (
                    int(metrics["window_size"])
                    if metrics is not None and metrics.get("window_size") is not None
                    else None
                ),
                "status": "complete" if metrics else "pending",
                "run_name": run_name,
            }
        )
    _write_csv(tables_dir / "table_architecture.csv", rows)
    complete = [row for row in rows if row["status"] == "complete"]
    best_root = min(
        (float(row["root_error_cm"]) for row in complete if row["root_error_cm"] is not None),
        default=None,
    )
    best_world = min(
        (float(row["world_mpjpe_cm"]) for row in complete if row["world_mpjpe_cm"] is not None),
        default=None,
    )
    latex_rows = [
        [
            _latex_escape(str(row["model"])),
            _latex_number(row["root_error_cm"], bold=_is_best(row["root_error_cm"], best_root)),
            _latex_number(row["world_mpjpe_cm"], bold=_is_best(row["world_mpjpe_cm"], best_world)),
            _latex_number(row["parameters_millions"], digits=3),
        ]
        for row in rows
    ]
    _write_latex_table(
        tables_dir / "table_architecture.tex",
        alignment="lrrr",
        headers=("Model", r"Root error $\downarrow$ (cm)", r"World MPJPE $\downarrow$ (cm)", "Params. (M)"),
        rows=latex_rows,
    )
    return rows


def _generate_input_ablation_table(
    *,
    tables_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    selected_run_names: Mapping[str, str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    variant_labels = {
        "no_sam3d": r"w/o relative 3D pose $\mathbf{f}^{3D}$",
        "no_player_2d": r"w/o 2D pose and box cues $\mathbf{f}^{2D},\mathbf{f}^{\mathrm{box}}$",
        "no_camera": r"w/o camera descriptor $\mathbf{f}^{cam}$",
    }
    variant_order = ("full", *variant_labels, "valid_joints_toggle")
    lookup = {(str(row["architecture"]), str(row["variant"])): row for row in metric_rows}
    by_run_name = {str(row["run_name"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for architecture in ("tcn", "transformer"):
        reference = by_run_name.get(selected_run_names.get(architecture, ""))
        reference_error = _metric_value(reference, "root_error_mean_m", scale=100.0)
        for variant in variant_order:
            metrics = reference if variant == "full" else lookup.get((architecture, variant))
            if metrics is None:
                continue
            row_mask_enabled = _input_config_flag(
                output_dir,
                str(metrics["run_name"]),
                "use_valid_joints_as_input",
            )
            if variant == "full":
                label = f"Full ({'with' if row_mask_enabled else 'without'} valid-joint mask)"
            elif variant == "valid_joints_toggle":
                label = f"{'With' if row_mask_enabled else 'Without'} valid-joint mask"
            else:
                label = variant_labels[variant]
            root_error = _metric_value(metrics, "root_error_mean_m", scale=100.0)
            rows.append(
                {
                    "model": architecture.upper() if architecture == "tcn" else "Transformer",
                    "variant": variant,
                    "source_variant": str(metrics["variant"]),
                    "input_configuration": label,
                    "valid_joint_mask": row_mask_enabled,
                    "root_error_cm": root_error,
                    "delta_root_error_cm": (
                        root_error - reference_error
                        if root_error is not None and reference_error is not None
                        else None
                    ),
                    "world_mpjpe_cm": _metric_value(metrics, "MPJPE_world_m", scale=100.0),
                    "local_mpjpe_cm": _metric_value(metrics, "MPJPE_local_m", scale=100.0),
                    "reprojection_error_px": _metric_value(metrics, "reprojection_error_mean_px"),
                    "seed": int(metrics["seed"]),
                    "window_size_frames": (
                        int(metrics["window_size"])
                        if metrics.get("window_size") is not None
                        else None
                    ),
                    "run_name": str(metrics["run_name"]),
                }
            )
    _write_csv(tables_dir / "table_input_ablation.csv", rows)
    latex_rows = [
        [
            _latex_escape(str(row["model"])),
            str(row["input_configuration"]),
            _latex_number(row["root_error_cm"]),
            _latex_number(row["delta_root_error_cm"], suffix=""),
            _latex_number(row["world_mpjpe_cm"]),
            _latex_number(row["reprojection_error_px"]),
        ]
        for row in rows
    ]
    _write_latex_table(
        tables_dir / "table_input_ablation.tex",
        alignment="llrrrr",
        headers=(
            "Model",
            "Input configuration",
            r"Root $\downarrow$ (cm)",
            r"$\Delta$ Root (cm)",
            r"World MPJPE $\downarrow$ (cm)",
            r"Reproj. $\downarrow$ (px)",
        ),
        rows=latex_rows,
    )
    return rows


def _generate_formulation_table(
    *,
    tables_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    selected_run_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    labels = {
        "full": r"Residual, $\mathbf{r}^{c,\mathrm{init}}+\Delta\widehat{\mathbf{r}}^{c}$",
        "absolute": r"Absolute root, $\widehat{\mathbf{r}}^{c}$",
    }
    lookup = {(str(row["architecture"]), str(row["variant"])): row for row in metric_rows}
    by_run_name = {str(row["run_name"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_ORDER:
        for variant, label in labels.items():
            metrics = (
                by_run_name.get(selected_run_names.get(architecture, ""))
                if variant == "full"
                else lookup.get((architecture, variant))
            )
            if metrics is None:
                continue
            rows.append(
                {
                    "model": (
                        architecture.upper()
                        if architecture in {"mlp", "tcn"}
                        else "Transformer"
                    ),
                    "variant": variant,
                    "source_variant": str(metrics["variant"]),
                    "formulation": label,
                    "root_error_cm": _metric_value(metrics, "root_error_mean_m", scale=100.0),
                    "world_mpjpe_cm": _metric_value(metrics, "MPJPE_world_m", scale=100.0),
                    "local_mpjpe_cm": _metric_value(metrics, "MPJPE_local_m", scale=100.0),
                    "reprojection_error_px": _metric_value(metrics, "reprojection_error_mean_px"),
                    "run_name": str(metrics["run_name"]),
                }
            )
    _write_csv(tables_dir / "table_root_formulation.csv", rows)
    _write_latex_table(
        tables_dir / "table_root_formulation.tex",
        alignment="llrrrr",
        headers=(
            "Model",
            "Root formulation",
            r"Root $\downarrow$ (cm)",
            r"World MPJPE $\downarrow$ (cm)",
            r"Local MPJPE $\downarrow$ (cm)",
            r"Reproj. $\downarrow$ (px)",
        ),
        rows=[
            [
                _latex_escape(str(row["model"])),
                str(row["formulation"]),
                _latex_number(row["root_error_cm"]),
                _latex_number(row["world_mpjpe_cm"]),
                _latex_number(row["local_mpjpe_cm"]),
                _latex_number(row["reprojection_error_px"]),
            ]
            for row in rows
        ],
    )
    return rows


def _generate_temporal_table(
    *,
    tables_dir: Path,
    metric_rows: Sequence[Mapping[str, Any]],
    selected_run_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in metric_rows
        if str(row["architecture"]) in {"tcn", "transformer"}
        and str(row["variant"]) in {"full", "window_201", "window_41"}
    ]
    selected: list[Mapping[str, Any]] = []
    for architecture in ("tcn", "transformer"):
        architecture_rows = [
            row for row in candidates if str(row["architecture"]) == architecture
        ]
        by_window: dict[int, Mapping[str, Any]] = {}
        for row in architecture_rows:
            window_size = int(row["window_size"])
            current = by_window.get(window_size)
            is_reference = str(row["run_name"]) == selected_run_names.get(architecture)
            current_is_reference = (
                current is not None
                and str(current["run_name"]) == selected_run_names.get(architecture)
            )
            if current is None or (is_reference and not current_is_reference):
                by_window[window_size] = row
        selected.extend(by_window[window] for window in sorted(by_window))
    rows = [
        {
            "model": str(row["architecture"]).upper()
            if str(row["architecture"]) == "tcn"
            else "Transformer",
            "variant": str(row["variant"]),
            "reference": str(row["run_name"]) == selected_run_names.get(str(row["architecture"])),
            "window_size_frames": int(row["window_size"]),
            "root_error_cm": _metric_value(row, "root_error_mean_m", scale=100.0),
            "world_mpjpe_cm": _metric_value(row, "MPJPE_world_m", scale=100.0),
            "reprojection_error_px": _metric_value(row, "reprojection_error_mean_px"),
            "run_name": str(row["run_name"]),
        }
        for row in selected
    ]
    _write_csv(tables_dir / "table_temporal_context.csv", rows)
    _write_latex_table(
        tables_dir / "table_temporal_context.tex",
        alignment="lrrrr",
        headers=(
            "Model",
            "Window (frames)",
            r"Root $\downarrow$ (cm)",
            r"World MPJPE $\downarrow$ (cm)",
            r"Reproj. $\downarrow$ (px)",
        ),
        rows=[
            [
                _latex_escape(str(row["model"])),
                str(row["window_size_frames"]),
                _latex_number(row["root_error_cm"]),
                _latex_number(row["world_mpjpe_cm"]),
                _latex_number(row["reprojection_error_px"]),
            ]
            for row in rows
        ],
    )
    return rows


def _build_method_series(
    predictions: Mapping[str, PredictionSet],
    diagnostics: DiagnosticData,
) -> list[MethodSeries]:
    reference = _reference_prediction(predictions)
    methods = [
        MethodSeries(
            key="geometry",
            label=r"Geometry init. $\mathbf{r}^{c,\mathrm{init}}$",
            table_label="Geometry initialization",
            color=METHOD_COLORS["geometry"],
            marker=METHOD_MARKERS["geometry"],
            root_error_m=diagnostics.root_init_error_m,
            component_error_m=diagnostics.root_init_component_error_m,
            run_name=None,
        )
    ]
    base_labels = {
        "mlp": "MLP residual",
        "tcn": "TCN residual",
        "transformer": "Transformer residual",
    }
    for architecture in ARCHITECTURE_ORDER:
        prediction = predictions.get(architecture)
        if prediction is None:
            continue
        _validate_prediction_alignment(reference, prediction)
        label = base_labels[architecture]
        if architecture != "mlp":
            label = f"{label} ({prediction.window_size} f)"
        methods.append(
            MethodSeries(
                key=architecture,
                label=label,
                table_label=label,
                color=METHOD_COLORS[architecture],
                marker=METHOD_MARKERS[architecture],
                root_error_m=prediction.root_error_m,
                component_error_m=np.abs(prediction.root_pred_m - prediction.root_gt_m),
                run_name=prediction.run_name,
            )
        )
    return methods


def _save_diagnostic_arrays(
    path: Path,
    *,
    diagnostics: DiagnosticData,
    methods: Sequence[MethodSeries],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "sequence": diagnostics.sequence,
        "sequence_cluster": diagnostics.cluster,
        "person_idx": diagnostics.person_idx,
        "frame_idx": diagnostics.frame_idx,
        "airborne_classifiable": diagnostics.classifiable,
        "airborne": diagnostics.airborne,
        "lower_foot_clearance_m": diagnostics.lower_foot_clearance_m,
        "camera_distance_m": diagnostics.camera_distance_m,
        "image_center_distance_norm": diagnostics.image_center_distance_norm,
        "bbox_height_ratio": diagnostics.bbox_height_ratio,
        "world_speed_mps": diagnostics.world_speed_mps,
        "pelvis_height_m": diagnostics.pelvis_height_m,
        "valid_joint_fraction": diagnostics.valid_joint_fraction,
    }
    for method in methods:
        arrays[f"root_error_{method.key}_m"] = method.root_error_m
        arrays[f"root_component_error_{method.key}_m"] = method.component_error_m
    np.savez_compressed(path, **arrays)


def _format_mean_ci_cm(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "--"
    mean = _metric_value(row, "mean_root_error_m", scale=100.0)
    low = _metric_value(row, "ci_low_m", scale=100.0)
    high = _metric_value(row, "ci_high_m", scale=100.0)
    if mean is None or low is None or high is None:
        return "--"
    return f"{mean:.2f} [{low:.2f}, {high:.2f}]"


def _generate_phase_table(
    *,
    tables_dir: Path,
    phase_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in phase_rows))
    rows: list[dict[str, Any]] = []
    for method in methods:
        phase_lookup = {
            str(row["phase"]): row for row in phase_rows if str(row["method"]) == method
        }
        ground = phase_lookup.get("Ground / non-jump")
        airborne = phase_lookup.get("Airborne / jump")
        contrast = phase_lookup.get("Airborne minus ground")
        rows.append(
            {
                "method": method,
                "ground_mean_cm": _metric_value(ground, "mean_root_error_m", scale=100.0),
                "ground_ci_low_cm": _metric_value(ground, "ci_low_m", scale=100.0),
                "ground_ci_high_cm": _metric_value(ground, "ci_high_m", scale=100.0),
                "airborne_mean_cm": _metric_value(airborne, "mean_root_error_m", scale=100.0),
                "airborne_ci_low_cm": _metric_value(airborne, "ci_low_m", scale=100.0),
                "airborne_ci_high_cm": _metric_value(airborne, "ci_high_m", scale=100.0),
                "airborne_minus_ground_cm": _metric_value(
                    contrast, "mean_root_error_m", scale=100.0
                ),
                "contrast_ci_low_cm": _metric_value(contrast, "ci_low_m", scale=100.0),
                "contrast_ci_high_cm": _metric_value(contrast, "ci_high_m", scale=100.0),
            }
        )
    _write_csv(tables_dir / "table_ground_airborne.csv", rows)
    phase_lookup_by_method = {
        method: {
            str(row["phase"]): row for row in phase_rows if str(row["method"]) == method
        }
        for method in methods
    }
    _write_latex_table(
        tables_dir / "table_ground_airborne.tex",
        alignment="lrrr",
        headers=(
            "Method",
            r"Ground (cm)",
            r"Airborne (cm)",
            r"Airborne $-$ ground (cm)",
        ),
        rows=[
            [
                _latex_escape(method),
                _format_mean_ci_cm(phase_lookup_by_method[method].get("Ground / non-jump")),
                _format_mean_ci_cm(phase_lookup_by_method[method].get("Airborne / jump")),
                _format_mean_ci_cm(phase_lookup_by_method[method].get("Airborne minus ground")),
            ]
            for method in methods
        ],
    )
    return rows


def _write_completed_runs_table(
    path: Path,
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    rows = [
        {
            "run_index": row.get("index"),
            "run_name": row.get("run_name"),
            "fold": row.get("fold"),
            "model": row.get("architecture"),
            "variant": row.get("variant"),
            "seed": row.get("seed"),
            "window_size_frames": row.get("window_size"),
            "root_error_cm": _metric_value(row, "root_error_mean_m", scale=100.0),
            "root_error_median_cm": _metric_value(row, "root_error_median_m", scale=100.0),
            "root_error_p90_cm": _metric_value(row, "root_error_p90_m", scale=100.0),
            "world_mpjpe_cm": _metric_value(row, "MPJPE_world_m", scale=100.0),
            "local_mpjpe_cm": _metric_value(row, "MPJPE_local_m", scale=100.0),
            "reprojection_error_px": _metric_value(row, "reprojection_error_mean_px"),
        }
        for row in sorted(metric_rows, key=lambda value: int(value.get("index", 0)))
    ]
    _write_csv(path, rows)


def _write_publication_readme(
    path: Path,
    *,
    summary: Mapping[str, Any],
) -> None:
    methods = ", ".join(str(value) for value in summary["methods"])
    text = f"""# Publication figures: unseen-match root localization

This directory is a reproducible snapshot of the completed portion of the ablation campaign.
At generation time, **{summary['completed_runs']}/{summary['planned_runs']} publication runs** were complete.
The evaluated split is the held-out match **{summary['test_match']}** ({summary['sequence_count']} clips,
{summary['frame_count']:,} evaluated player-frames). Available methods: {methods}.

## Statistical protocol

- Curves show frame-weighted means in fixed or quantile bins.
- Shaded regions and error bars are {summary['confidence_level']:.0%} confidence intervals from
  {summary['bootstrap_samples']:,} sequence-cluster bootstrap resamples. A clip is the resampling unit;
  frames are not treated as independent replicates.
- A frame is airborne when both calibrated feet are at least
  {summary['airborne_threshold_m'] * 100:.0f} cm above the pitch plane $\\pi$. Episodes shorter than
  {summary['min_airborne_frames']} frames are removed. Ground and airborne are the only phase classes;
  there is no small/large-jump subdivision.
- `02_root_error_vs_foot_clearance` uses the lower of the two calibrated foot clearances as jump height.
- `13_root_temporal_consistency` reports first- and second-order finite-difference errors of
  $\widehat{{\mathbf{{r}}}}^w_t$ at {summary['fps']:.0f} fps. Only consecutive frames from the same
  player track are used; derivatives are evaluated in world coordinates against ground truth.
- `14_temporal_gain_by_regime` reports paired root-error reductions relative to the MLP. High
  speed and large initialization error denote the upper quartile of their respective test-set
  distributions. These diagnostic subsets may overlap with the grounded/airborne partition.
- Continuous-condition plots use equal-frequency bins except foot clearance, which uses fixed
  {summary['foot_bin_width_m'] * 100:.1f} cm bins.

## Outputs

- `figures/`: publication figures in vector PDF and 300-dpi PNG.
- `figure_data/`: compact CSV values used for every figure, including confidence bounds.
- `figure_data/diagnostic_frame_data.npz`: aligned per-frame diagnostics and method errors.
- `tables/`: CSV source data and `booktabs`-compatible LaTeX snippets.
- `summary.json`: exact snapshot and analysis settings.

Main figures and tables use the explicitly frozen runs stored in `summary.json`; completed jobs do
not change the references automatically. The input ablation table omits `no_pitch_points` and
`no_ground_intersection`, includes both valid-joint-mask states, and rebases every displayed
difference to the selected architecture reference. Its CSV records the source run, seed, temporal
window, and mask state for traceability. The root-formulation table intentionally omits the
absolute model conditioned on root initialization as an input feature.

A cumulative ablation table should only be reported after training an explicitly ordered, nested
sequence of configurations; otherwise its conclusions depend on an arbitrary removal order and
interactions are confounded.

Regenerate after additional array jobs finish; completed models are discovered automatically.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _remove_obsolete_outputs(figures_dir: Path, figure_data_dir: Path) -> None:
    obsolete = "08_root_error_vs_valid_joint_fraction"
    for path in (
        figures_dir / f"{obsolete}.png",
        figures_dir / f"{obsolete}.pdf",
        figure_data_dir / f"{obsolete}.csv",
    ):
        path.unlink(missing_ok=True)


def generate_publication_package(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    fold: str | None = None,
    split: str = "test",
    fps: float = 25.0,
    speed_window_frames: int = 5,
    airborne_threshold_m: float = 0.05,
    ground_reference_percentile: float = 20.0,
    ground_reference_window_s: float = 5.0,
    min_airborne_frames: int = 2,
    max_ground_gap_frames: int = 1,
    foot_bin_width_m: float = 0.025,
    quantile_bins: int = 10,
    min_frames_per_bin: int = 100,
    bootstrap_samples: int | None = None,
    confidence_level: float | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    manifest_file, manifest = load_manifest(manifest_path)
    plan = load_plan(manifest)
    campaign_dir = campaign_output_dir(manifest)
    selected_fold = fold or str(manifest.get("design", {}).get("primary_fold", "primary"))
    if selected_fold not in manifest["folds"]:
        raise KeyError(f"Unknown fold {selected_fold!r}; available folds: {sorted(manifest['folds'])}")
    selected_seed = int(manifest.get("design", {}).get("base_seed", plan.get("base_seed", 0)))
    publication_config = manifest.get("publication", {}) or {}
    reference_runs_by_fold = publication_config.get("reference_runs", {}) or {}
    configured_reference_runs = reference_runs_by_fold.get(selected_fold, {}) or {}
    if not isinstance(configured_reference_runs, Mapping):
        raise ValueError(
            f"publication.reference_runs.{selected_fold} must be a mapping of architecture to run name"
        )
    statistics = manifest.get("statistics", {}) or {}
    samples = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else statistics.get("bootstrap_samples", 2000)
    )
    confidence = float(
        confidence_level
        if confidence_level is not None
        else statistics.get("confidence_level", 0.95)
    )
    seed = int(
        bootstrap_seed
        if bootstrap_seed is not None
        else statistics.get("bootstrap_seed", 314159)
    )
    if fps <= 0.0 or speed_window_frames < 1:
        raise ValueError("fps must be positive and speed_window_frames must be >= 1")
    if not 0.0 < confidence < 1.0 or samples < 1:
        raise ValueError("confidence_level must be in (0, 1) and bootstrap_samples must be >= 1")
    if foot_bin_width_m <= 0.0 or quantile_bins < 2 or min_frames_per_bin < 1:
        raise ValueError("Invalid binning configuration")

    publication_dir = (
        resolve_project_path(output_dir)
        if output_dir is not None
        else campaign_dir / "publication"
    )
    figures_dir = publication_dir / "figures"
    figure_data_dir = publication_dir / "figure_data"
    tables_dir = publication_dir / "tables"
    for directory in (figures_dir, figure_data_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_outputs(figures_dir, figure_data_dir)

    predictions = _load_publication_predictions(
        output_dir=campaign_dir,
        plan=plan,
        fold=selected_fold,
        seed=selected_seed,
        split=split,
        reference_run_names={
            str(architecture): str(run_name)
            for architecture, run_name in configured_reference_runs.items()
        },
    )
    selected_run_names = {
        architecture: prediction.run_name for architecture, prediction in predictions.items()
    }
    reference = _reference_prediction(predictions)
    data_root = resolve_project_path(str(manifest["data_dir"]))
    preprocessing = manifest.get("preprocessing", {}) or {}
    raw_features_dir = data_root / str(preprocessing.get("raw_features_dirname", "features"))
    raw_root_init_dir = data_root / str(
        preprocessing.get("raw_root_init_dirname", "root_init_cam")
    )
    diagnostics = _build_diagnostics(
        reference=reference,
        raw_features_dir=raw_features_dir,
        raw_root_init_dir=raw_root_init_dir,
        fps=fps,
        speed_window_frames=speed_window_frames,
        airborne_threshold_m=airborne_threshold_m,
        ground_reference_percentile=ground_reference_percentile,
        ground_reference_window_frames=max(0, int(round(ground_reference_window_s * fps))),
        min_airborne_frames=min_airborne_frames,
        max_ground_gap_frames=max_ground_gap_frames,
    )
    methods = _build_method_series(predictions, diagnostics)
    _save_diagnostic_arrays(
        figure_data_dir / "diagnostic_frame_data.npz",
        diagnostics=diagnostics,
        methods=methods,
    )

    _configure_style()
    phase_rows = _plot_ground_vs_airborne(
        methods=methods,
        diagnostics=diagnostics,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    _plot_condition_curves(
        methods=methods,
        diagnostics=diagnostics,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
        samples=samples,
        confidence=confidence,
        seed=seed + 101,
        foot_bin_width_m=foot_bin_width_m,
        quantile_bins=quantile_bins,
        min_frames=min_frames_per_bin,
    )
    _plot_residual_gain(
        methods=methods,
        diagnostics=diagnostics,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
        samples=samples,
        confidence=confidence,
        seed=seed + 202,
        quantile_bins=quantile_bins,
        min_frames=min_frames_per_bin,
    )
    _plot_axis_errors(
        methods=methods,
        diagnostics=diagnostics,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
        samples=samples,
        confidence=confidence,
        seed=seed + 303,
    )
    _plot_error_ecdf(methods=methods, figures_dir=figures_dir, data_dir=figure_data_dir)
    _plot_per_sequence_error(
        methods=methods,
        diagnostics=diagnostics,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
    )
    temporal_consistency_rows = _plot_temporal_consistency(
        predictions=predictions,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
        tables_dir=tables_dir,
        fps=fps,
        samples=samples,
        confidence=confidence,
        seed=seed + 404,
    )
    temporal_gain_rows = _plot_temporal_gain_by_regime(
        methods=methods,
        diagnostics=diagnostics,
        figures_dir=figures_dir,
        data_dir=figure_data_dir,
        tables_dir=tables_dir,
        samples=samples,
        confidence=confidence,
        seed=seed + 505,
    )

    metric_rows = _load_completed_metric_rows(
        output_dir=campaign_dir,
        plan=plan,
        fold=selected_fold,
        seed=selected_seed,
        split=split,
        selected_run_names=selected_run_names,
    )
    geometry_metrics = _load_geometry_metrics(campaign_dir, fold=selected_fold, split=split)
    _write_completed_runs_table(tables_dir / "all_completed_runs.csv", metric_rows)
    _generate_main_table(
        tables_dir=tables_dir,
        metric_rows=metric_rows,
        geometry_metrics=geometry_metrics,
        selected_run_names=selected_run_names,
    )
    _generate_architecture_table(
        tables_dir=tables_dir,
        metric_rows=metric_rows,
        output_dir=campaign_dir,
        selected_run_names=selected_run_names,
    )
    _generate_input_ablation_table(
        tables_dir=tables_dir,
        metric_rows=metric_rows,
        selected_run_names=selected_run_names,
        output_dir=campaign_dir,
    )
    _generate_formulation_table(
        tables_dir=tables_dir,
        metric_rows=metric_rows,
        selected_run_names=selected_run_names,
    )
    _generate_temporal_table(
        tables_dir=tables_dir,
        metric_rows=metric_rows,
        selected_run_names=selected_run_names,
    )
    _generate_phase_table(tables_dir=tables_dir, phase_rows=phase_rows)

    classifiable = diagnostics.classifiable
    airborne = classifiable & diagnostics.airborne
    planned_runs = len(_publication_runs(plan))
    completed_run_names = sorted({str(row["run_name"]) for row in metric_rows})
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_file),
        "campaign_output_dir": str(campaign_dir),
        "publication_output_dir": str(publication_dir),
        "fold": selected_fold,
        "split": split,
        "test_match": str(manifest["folds"][selected_fold][f"{split}_match"]),
        "planned_runs": planned_runs,
        "completed_runs": len(completed_run_names),
        "completed_run_names": completed_run_names,
        "methods": [method.table_label for method in methods],
        "method_runs": {method.key: method.run_name for method in methods},
        "reference_selection": "explicit publication.reference_runs mapping from the campaign manifest",
        "input_ablation_reference_runs": selected_run_names,
        "excluded_input_ablation_variants": [
            "no_pitch_points",
            "no_ground_intersection",
        ],
        "included_input_ablation_variants": [
            "full",
            "no_sam3d",
            "no_player_2d",
            "no_camera",
            "valid_joints_toggle",
        ],
        "excluded_root_formulation_variants": ["absolute_root_init_input"],
        "sequence_count": int(np.unique(diagnostics.sequence).size),
        "frame_count": int(diagnostics.sequence.size),
        "classifiable_frame_count": int(classifiable.sum()),
        "ground_frame_count": int((classifiable & ~diagnostics.airborne).sum()),
        "airborne_frame_count": int(airborne.sum()),
        "fps": fps,
        "speed_window_frames": speed_window_frames,
        "airborne_threshold_m": airborne_threshold_m,
        "ground_reference_percentile": ground_reference_percentile,
        "ground_reference_window_s": ground_reference_window_s,
        "min_airborne_frames": min_airborne_frames,
        "max_ground_gap_frames": max_ground_gap_frames,
        "foot_bin_width_m": foot_bin_width_m,
        "quantile_bins": quantile_bins,
        "min_frames_per_bin": min_frames_per_bin,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "confidence_level": confidence,
        "temporal_consistency_methods": [
            str(row["model"]) for row in temporal_consistency_rows
        ],
        "temporal_gain_regimes": list(
            dict.fromkeys(str(row["regime"]) for row in temporal_gain_rows)
        ),
    }
    write_json_atomic(publication_dir / "summary.json", summary)
    _write_publication_readme(publication_dir / "README.md", summary=summary)
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication figures and tables from the completed ablation runs."
    )
    parser.add_argument("--manifest", default="configs/ablation/unseen_match_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fold", default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--speed-window-frames", type=int, default=5)
    parser.add_argument("--airborne-threshold-m", type=float, default=0.05)
    parser.add_argument("--ground-reference-percentile", type=float, default=20.0)
    parser.add_argument("--ground-reference-window-s", type=float, default=5.0)
    parser.add_argument("--min-airborne-frames", type=int, default=2)
    parser.add_argument("--max-ground-gap-frames", type=int, default=1)
    parser.add_argument("--foot-bin-width-m", type=float, default=0.025)
    parser.add_argument("--quantile-bins", type=int, default=10)
    parser.add_argument("--min-frames-per-bin", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--confidence-level", type=float, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    summary = generate_publication_package(
        args.manifest,
        output_dir=args.output_dir,
        fold=args.fold,
        split=args.split,
        fps=args.fps,
        speed_window_frames=args.speed_window_frames,
        airborne_threshold_m=args.airborne_threshold_m,
        ground_reference_percentile=args.ground_reference_percentile,
        ground_reference_window_s=args.ground_reference_window_s,
        min_airborne_frames=args.min_airborne_frames,
        max_ground_gap_frames=args.max_ground_gap_frames,
        foot_bin_width_m=args.foot_bin_width_m,
        quantile_bins=args.quantile_bins,
        min_frames_per_bin=args.min_frames_per_bin,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
