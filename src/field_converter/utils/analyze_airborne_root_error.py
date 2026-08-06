"""Analyse des phases aeriennes GT 3D et de l'erreur du root initial.

Le script utilise les articulations BODY-25 des orteils et des talons dans
``Y_cam_gt``, les repasse dans le repere monde, puis mesure leur distance au
plan du terrain. Pour la detection, le biais statique des centres de joints
BODY-25 est retire par joueur et par pied. Une frame est aerienne lorsque le
pied gauche ET le pied droit sont au-dessus du seuil configure. Les plots de
hauteur conservent, eux, la distance geometrique GT brute au plan.

Exemple
-------
PYTHONPATH=src python -m field_converter.utils.analyze_airborne_root_error \
    --data-dir data \
    --output-dir outputs/airborne_root_analysis

Toutes les figures sont sauvegardees en PNG avec le backend non interactif
``Agg``. Le script n'appelle jamais ``plt.show()``.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Mapping OpenPose BODY-25, indices zero-based.
DEFAULT_LEFT_FOOT_JOINTS = (19, 20, 21)  # LBigToe, LSmallToe, LHeel
DEFAULT_RIGHT_FOOT_JOINTS = (22, 23, 24)  # RBigToe, RSmallToe, RHeel


@dataclass(frozen=True)
class JumpPhase:
    sequence: str
    person_idx: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    peak_foot_clearance_m: float
    mean_foot_clearance_m: float
    category: str


def _parse_joint_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Les indices doivent etre des entiers separes par des virgules.") from exc
    if not indices or min(indices) < 0:
        raise argparse.ArgumentTypeError("Il faut au moins un indice de joint positif ou nul.")
    return indices


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ``[start, end)`` runs of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"mask doit etre 1D, shape recue: {mask.shape}")
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def clean_airborne_mask(
    candidate: np.ndarray,
    valid: np.ndarray,
    *,
    min_airborne_frames: int,
    max_ground_gap_frames: int,
) -> np.ndarray:
    """Bridge short valid gaps, then remove very short airborne detections."""
    candidate = np.asarray(candidate, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if candidate.shape != valid.shape or candidate.ndim != 1:
        raise ValueError("candidate et valid doivent etre des masques 1D de meme shape")
    if min_airborne_frames < 1 or max_ground_gap_frames < 0:
        raise ValueError("min_airborne_frames >= 1 et max_ground_gap_frames >= 0 requis")

    airborne = candidate & valid
    if max_ground_gap_frames:
        for start, end in _contiguous_runs(~airborne):
            bounded_by_air = start > 0 and end < airborne.size and airborne[start - 1] and airborne[end]
            valid_gap = bool(np.all(valid[start:end]))
            if bounded_by_air and valid_gap and (end - start) <= max_ground_gap_frames:
                airborne[start:end] = True

    for start, end in _contiguous_runs(airborne):
        if (end - start) < min_airborne_frames:
            airborne[start:end] = False
    return airborne & valid


def fit_pitch_plane(points_world: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit ``normal . X + offset = 0`` with a unit normal."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=-1)]
    if points.shape[0] < 3:
        raise ValueError("Au moins trois points terrain finis sont necessaires.")
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("Impossible d'estimer un plan terrain valide.")
    normal = normal / norm
    return normal, -float(normal @ centroid)


def _camera_joints_to_world(Y_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Invert ``X_cam = X_world @ R.T + t`` for arrays shaped ``(N,T,J,3)``."""
    return np.einsum(
        "ntjc,tcw->ntjw",
        np.asarray(Y_cam, dtype=np.float64) - np.asarray(t, dtype=np.float64)[None, :, None, :],
        np.asarray(R, dtype=np.float64),
    )


def _camera_roots_to_world(root_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Invert ``X_cam = X_world @ R.T + t`` for arrays shaped ``(N,T,3)``."""
    return np.einsum(
        "ntc,tcw->ntw",
        np.asarray(root_cam, dtype=np.float64) - np.asarray(t, dtype=np.float64)[None, :, :],
        np.asarray(R, dtype=np.float64),
    )


def _minimum_group_height(
    joints_world: np.ndarray,
    joint_indices: Sequence[int],
    normal: np.ndarray,
    offset: float,
) -> np.ndarray:
    selected = joints_world[:, :, joint_indices, :]
    distances = np.einsum("ntjc,c->ntj", selected, normal) + offset
    finite = np.isfinite(distances)
    minimum = np.min(np.where(finite, distances, np.inf), axis=-1)
    minimum[~finite.any(axis=-1)] = np.nan
    # A negative signed distance is an annotation/calibration artefact, not a
    # physical height below the pitch.
    return np.maximum(minimum, 0.0)


def compute_foot_heights(
    *,
    Y_cam_gt: np.ndarray,
    root_cam_gt: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    left_foot_joints: Sequence[int],
    right_foot_joints: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return left, right and lower-of-both physical foot heights in metres."""
    joints_world = _camera_joints_to_world(Y_cam_gt, R, t)
    roots_world = _camera_roots_to_world(root_cam_gt, R, t)

    normal = np.asarray(plane_normal, dtype=np.float64).copy()
    offset = float(plane_offset)
    root_signed_distance = np.einsum("ntc,c->nt", roots_world, normal) + offset
    finite_root = root_signed_distance[np.isfinite(root_signed_distance)]
    if finite_root.size and float(np.median(finite_root)) < 0.0:
        normal = -normal
        offset = -offset

    left = _minimum_group_height(joints_world, left_foot_joints, normal, offset)
    right = _minimum_group_height(joints_world, right_foot_joints, normal, offset)
    lower_foot = np.minimum(left, right)
    return left, right, lower_foot


def calibrate_foot_clearance(
    left_height: np.ndarray,
    right_height: np.ndarray,
    valid: np.ndarray,
    *,
    ground_reference_percentile: float,
    ground_reference_window_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove the static BODY-25 foot-keypoint offset for each player track.

    Toe/heel keypoints describe joint centres rather than the shoe sole, and a
    few GT tracks have an additional constant vertical offset. The low
    percentile of each foot is therefore used as its observed contact level.
    With a positive window, sparse local percentiles are interpolated over
    time so that slow GT drift is corrected without following short jumps.
    A percentile of zero disables the correction and keeps raw plane heights.
    """
    left_height = np.asarray(left_height, dtype=np.float64)
    right_height = np.asarray(right_height, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if left_height.shape != right_height.shape or valid.shape != left_height.shape:
        raise ValueError("Shapes incompatibles pour la calibration des pieds")

    def reference_for(foot_height: np.ndarray) -> np.ndarray:
        if ground_reference_percentile == 0.0:
            return np.zeros_like(foot_height)

        values = np.where(valid & np.isfinite(foot_height), foot_height, np.nan)
        N, T = values.shape
        if ground_reference_window_frames <= 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                track_reference = np.nanpercentile(values, ground_reference_percentile, axis=1)
            return np.broadcast_to(track_reference[:, None], values.shape)

        window = min(int(ground_reference_window_frames), T)
        half_window = window // 2
        anchor_stride = max(1, window // 4)
        anchors = np.unique(np.append(np.arange(0, T, anchor_stride), T - 1)).astype(np.int64)
        anchor_references = np.full((N, anchors.size), np.nan, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for anchor_idx, center in enumerate(anchors):
                start = max(0, int(center) - half_window)
                end = min(T, int(center) + half_window + 1)
                anchor_references[:, anchor_idx] = np.nanpercentile(
                    values[:, start:end],
                    ground_reference_percentile,
                    axis=1,
                )

        reference = np.full_like(values, np.nan)
        frame_axis = np.arange(T)
        for person_idx in range(N):
            finite = np.isfinite(anchor_references[person_idx])
            if finite.any():
                reference[person_idx] = np.interp(
                    frame_axis,
                    anchors[finite],
                    anchor_references[person_idx, finite],
                )
        return reference

    left_reference = reference_for(left_height)
    right_reference = reference_for(right_height)
    left_clearance = np.maximum(left_height - left_reference, 0.0)
    right_clearance = np.maximum(right_height - right_reference, 0.0)
    lower_clearance = np.minimum(left_clearance, right_clearance)
    return left_clearance, right_clearance, lower_clearance, left_reference, right_reference


def _load_root_init(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64)
    with np.load(path, allow_pickle=True) as npz:
        for key in ("root_init_cam", "root_init", "arr_0"):
            if key in npz.files:
                return np.asarray(npz[key], dtype=np.float64)
    raise KeyError(f"{path}: cle root_init_cam, root_init ou arr_0 absente")


def _resolve_sequence_path(folder: Path, sequence: str, suffixes: Iterable[str], split: str) -> Path:
    candidates: list[Path] = []
    for suffix in suffixes:
        candidates.append(folder / f"{sequence}{suffix}")
        if split != "all":
            candidates.append(folder / split / f"{sequence}{suffix}")
    for path in candidates:
        if path.exists():
            return path
    looked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Fichier introuvable pour {sequence}. Chemins testes: {looked}")


def _read_sequences(
    *,
    features_dir: Path,
    data_dir: Path,
    split: str,
    split_json: Path,
    requested_sequences: Sequence[str] | None,
) -> list[str]:
    if requested_sequences:
        sequences = list(dict.fromkeys(requested_sequences))
    elif split != "all":
        if not split_json.exists():
            raise FileNotFoundError(f"split.json absent: {split_json}")
        payload = json.loads(split_json.read_text(encoding="utf-8"))
        sequences = payload.get(split, [])
        if not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences):
            raise ValueError(f"Contenu invalide pour le split {split!r} dans {split_json}")
    else:
        sequences_file = data_dir / "sequences_gt.txt"
        if sequences_file.exists():
            sequences = [
                line.strip()
                for line in sequences_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        else:
            sequences = sorted(path.stem for path in features_dir.glob("*.npz"))

    if not sequences:
        raise ValueError("Aucune sequence a analyser.")
    return sequences


def _describe(values: np.ndarray) -> dict[str, int | float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _save_jump_histogram(
    *,
    out_path: Path,
    ground_phase_count: int,
    small_jump_count: int,
    large_jump_count: int,
    duration_frames: np.ndarray,
    fps: float,
) -> None:
    labels = ["Pas de saut\n(phases au sol)", "Petit saut", "Grand saut"]
    counts = [ground_phase_count, small_jump_count, large_jump_count]
    colors = ["tab:gray", "tab:orange", "tab:red"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.85)
    ax.bar_label(bars, labels=[f"{value:,}".replace(",", " ") for value in counts], padding=3)
    ax.set_ylabel("Nombre de phases")
    ax.set_title("Phases au sol et phases aeriennes detectees avec la GT 3D")
    ax.grid(axis="y", alpha=0.25)

    if duration_frames.size:
        mean_frames = float(np.mean(duration_frames))
        median_frames = float(np.median(duration_frames))
        text = (
            f"Duree des sauts — moyenne : {mean_frames:.1f} frames ({mean_frames / fps:.3f} s)\n"
            f"mediane : {median_frames:.1f} frames ({median_frames / fps:.3f} s)"
        )
    else:
        text = "Aucune phase aerienne detectee"
    ax.text(
        0.98,
        0.96,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_limits(height: np.ndarray, error: np.ndarray, configured_height_max: float | None) -> tuple[float, float]:
    height_finite = height[np.isfinite(height)]
    error_finite = error[np.isfinite(error)]
    if configured_height_max is not None:
        height_max = configured_height_max
    elif height_finite.size:
        height_max = max(0.35, float(np.percentile(height_finite, 99.9)))
    else:
        height_max = 0.5
    error_max = max(0.1, float(np.percentile(error_finite, 99.5))) if error_finite.size else 1.0
    return height_max, error_max


def _save_scatter(
    *,
    out_path: Path,
    height: np.ndarray,
    error: np.ndarray,
    max_points: int,
    height_max: float,
    error_max: float,
    seed: int,
) -> None:
    finite = np.isfinite(height) & np.isfinite(error)
    within_plot = finite & (height <= height_max) & (error <= error_max)
    indices = np.flatnonzero(within_plot)
    total_within = int(indices.size)
    if indices.size > max_points:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))

    fig, ax = plt.subplots(figsize=(8, 6))
    if indices.size:
        ax.scatter(height[indices], error[indices], s=5, alpha=0.18, linewidths=0, color="tab:blue")
    ax.set_xlabel("Hauteur GT du pied le plus bas au-dessus du sol (m)")
    ax.set_ylabel(r"$\|\Delta root\|_2$ (m)")
    ax.set_title(r"Erreur du root initial vs hauteur GT des pieds")
    ax.set_xlim(0.0, height_max)
    ax.set_ylim(0.0, error_max)
    ax.grid(True, alpha=0.25)

    corr_mask = finite
    correlation = None
    if int(corr_mask.sum()) >= 2:
        h = height[corr_mask]
        e = error[corr_mask]
        if float(np.std(h)) > 0.0 and float(np.std(e)) > 0.0:
            correlation = float(np.corrcoef(h, e)[0, 1])
    corr_text = "n/a" if correlation is None else f"{correlation:.3f}"
    ax.text(
        0.02,
        0.98,
        f"Pearson r (toutes les frames) = {corr_text}\n"
        f"Affichees : {indices.size:,}/{total_within:,} frames dans les limites".replace(",", " "),
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_ground_air_histogram(
    *,
    out_path: Path,
    ground_error: np.ndarray,
    airborne_error: np.ndarray,
) -> None:
    pooled = np.concatenate([ground_error, airborne_error])
    finite_pooled = pooled[np.isfinite(pooled)]
    upper = max(0.1, float(np.percentile(finite_pooled, 99.5))) if finite_pooled.size else 1.0
    bins = np.linspace(0.0, upper, 61)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if ground_error.size:
        ax.hist(
            ground_error[ground_error <= upper],
            bins=bins,
            density=False,
            alpha=0.55,
            color="tab:gray",
            label=f"Au sol (N={ground_error.size:,})".replace(",", " "),
        )
    if airborne_error.size:
        ax.hist(
            airborne_error[airborne_error <= upper],
            bins=bins,
            density=False,
            alpha=0.55,
            color="tab:orange",
            label=f"En l'air (N={airborne_error.size:,})".replace(",", " "),
        )
    ax.set_xlabel(r"$\|\Delta root\|_2$ (m)")
    ax.set_ylabel("Densite")
    ax.set_title("Distribution de l'erreur du root initial")
    ax.set_xlim(0.0, upper)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _binned_root_error(
    *,
    height: np.ndarray,
    error: np.ndarray,
    bin_width: float,
    height_max: float,
) -> list[dict[str, int | float]]:
    edges = np.arange(0.0, height_max + bin_width, bin_width, dtype=np.float64)
    if edges[-1] < height_max:
        edges = np.append(edges, height_max)
    bin_ids = np.digitize(height, edges, right=False) - 1
    rows: list[dict[str, int | float]] = []
    for index in range(edges.size - 1):
        mask = (bin_ids == index) & np.isfinite(error)
        values = error[mask]
        rows.append(
            {
                "height_start_m": float(edges[index]),
                "height_end_m": float(edges[index + 1]),
                "height_center_m": float((edges[index] + edges[index + 1]) / 2.0),
                "frame_count": int(values.size),
                "mean_root_error_m": float(np.mean(values)) if values.size else float("nan"),
                "median_root_error_m": float(np.median(values)) if values.size else float("nan"),
            }
        )
    return rows


def _save_mean_error_curve(
    *,
    out_path: Path,
    rows: Sequence[dict[str, int | float]],
    min_frames_per_bin: int,
) -> None:
    selected = [
        row
        for row in rows
        if int(row["frame_count"]) >= min_frames_per_bin
        and np.isfinite(float(row["mean_root_error_m"]))
    ]
    x = np.asarray([row["height_center_m"] for row in selected], dtype=np.float64)
    mean = np.asarray([row["mean_root_error_m"] for row in selected], dtype=np.float64)
    median = np.asarray([row["median_root_error_m"] for row in selected], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if x.size:
        ax.plot(x, mean, marker="o", markersize=4, linewidth=2, label="Erreur moyenne")
        ax.plot(x, median, linewidth=1.5, linestyle="--", alpha=0.8, label="Erreur mediane")
    ax.set_xlabel("Hauteur GT du pied le plus bas au-dessus du sol (m)")
    ax.set_ylabel(r"$\|\Delta root\|_2$ (m)")
    ax.set_title("Erreur du root en fonction de la hauteur GT des pieds")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_ylim(bottom=0.0)
    if not x.size:
        ax.text(0.5, 0.5, "Aucun bin suffisamment peuple", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    features_dir = Path(args.features_dir) if args.features_dir else data_dir / "features"
    root_init_dir = Path(args.root_init_dir) if args.root_init_dir else data_dir / "root_init_cam"
    split_json = Path(args.split_json) if args.split_json else data_dir / "features_normalized" / "split.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = _read_sequences(
        features_dir=features_dir,
        data_dir=data_dir,
        split=args.split,
        split_json=split_json,
        requested_sequences=args.sequences,
    )
    pitch_points_path = Path(args.pitch_points) if args.pitch_points else data_dir / "pitch_points.txt"
    if not pitch_points_path.exists():
        raise FileNotFoundError(f"Points terrain absents: {pitch_points_path}")
    pitch_points = np.loadtxt(pitch_points_path, dtype=np.float64)
    plane_normal, plane_offset = fit_pitch_plane(pitch_points)

    all_height: list[np.ndarray] = []
    all_error: list[np.ndarray] = []
    ground_errors: list[np.ndarray] = []
    airborne_errors: list[np.ndarray] = []
    jump_phases: list[JumpPhase] = []
    ground_phase_count = 0
    valid_gt_frame_count = 0
    airborne_gt_frame_count = 0
    track_count = 0
    left_ground_references: list[np.ndarray] = []
    right_ground_references: list[np.ndarray] = []
    frame_sequence_indices: list[np.ndarray] = []
    frame_person_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    frame_root_deltas: list[np.ndarray] = []
    frame_left_plane_heights: list[np.ndarray] = []
    frame_right_plane_heights: list[np.ndarray] = []
    frame_left_clearances: list[np.ndarray] = []
    frame_right_clearances: list[np.ndarray] = []
    frame_airborne: list[np.ndarray] = []
    frame_jump_categories: list[np.ndarray] = []

    required_keys = ("Y_cam_gt", "Y_root_cam_gt", "R", "t", "valid_mask")

    for sequence_index, sequence in enumerate(sequences, start=1):
        feature_path = _resolve_sequence_path(features_dir, sequence, (".npz",), args.split)
        root_init_path = _resolve_sequence_path(root_init_dir, sequence, (".npy", ".npz"), args.split)
        print(f"[{sequence_index:03d}/{len(sequences):03d}] {sequence}")

        with np.load(feature_path, allow_pickle=True) as npz:
            missing = [key for key in required_keys if key not in npz.files]
            if missing:
                raise KeyError(f"{feature_path}: cles manquantes {missing}")
            Y_cam_gt = np.asarray(npz["Y_cam_gt"], dtype=np.float64)
            root_cam_gt = np.asarray(npz["Y_root_cam_gt"], dtype=np.float64)
            R = np.asarray(npz["R"], dtype=np.float64)
            t = np.asarray(npz["t"], dtype=np.float64)
            valid_mask = np.asarray(npz["valid_mask"], dtype=bool)

        if Y_cam_gt.ndim != 4 or Y_cam_gt.shape[-1] != 3:
            raise ValueError(f"{sequence}: Y_cam_gt doit avoir la shape (N,T,J,3), recu {Y_cam_gt.shape}")
        max_joint = max((*args.left_foot_joints, *args.right_foot_joints))
        if max_joint >= Y_cam_gt.shape[2]:
            raise ValueError(f"{sequence}: joint {max_joint} absent de Y_cam_gt (J={Y_cam_gt.shape[2]})")
        if root_cam_gt.shape != Y_cam_gt.shape[:2] + (3,) or valid_mask.shape != Y_cam_gt.shape[:2]:
            raise ValueError(f"{sequence}: shapes GT/root/valid incompatibles")

        root_init_cam = _load_root_init(root_init_path)
        if root_init_cam.shape != root_cam_gt.shape:
            raise ValueError(
                f"{sequence}: root init {root_init_cam.shape} incompatible avec root GT {root_cam_gt.shape}"
            )

        left_plane_height, right_plane_height, plane_foot_height = compute_foot_heights(
            Y_cam_gt=Y_cam_gt,
            root_cam_gt=root_cam_gt,
            R=R,
            t=t,
            plane_normal=plane_normal,
            plane_offset=plane_offset,
            left_foot_joints=args.left_foot_joints,
            right_foot_joints=args.right_foot_joints,
        )
        plane_height_valid = valid_mask & np.isfinite(left_plane_height) & np.isfinite(right_plane_height)
        reference_window_frames = (
            max(1, int(round(args.ground_reference_window_s * args.fps)))
            if args.ground_reference_window_s > 0.0
            else 0
        )
        left_height, right_height, foot_height, left_reference, right_reference = calibrate_foot_clearance(
            left_plane_height,
            right_plane_height,
            plane_height_valid,
            ground_reference_percentile=args.ground_reference_percentile,
            ground_reference_window_frames=reference_window_frames,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            left_track_references = np.nanmedian(np.where(plane_height_valid, left_reference, np.nan), axis=1)
            right_track_references = np.nanmedian(np.where(plane_height_valid, right_reference, np.nan), axis=1)
        left_ground_references.append(left_track_references[np.isfinite(left_track_references)])
        right_ground_references.append(right_track_references[np.isfinite(right_track_references)])
        gt_valid = plane_height_valid & np.isfinite(left_height) & np.isfinite(right_height)
        root_delta = root_cam_gt - root_init_cam
        root_error = np.linalg.norm(root_delta, axis=-1)
        error_valid = gt_valid & np.isfinite(root_error)

        N, _ = gt_valid.shape
        track_count += N
        sequence_airborne = np.zeros_like(gt_valid)
        sequence_jump_category = np.zeros_like(gt_valid, dtype=np.int8)

        for person_idx in range(N):
            candidate = (
                gt_valid[person_idx]
                & (left_height[person_idx] >= args.airborne_threshold_m)
                & (right_height[person_idx] >= args.airborne_threshold_m)
            )
            airborne = clean_airborne_mask(
                candidate,
                gt_valid[person_idx],
                min_airborne_frames=args.min_airborne_frames,
                max_ground_gap_frames=args.max_ground_gap_frames,
            )
            sequence_airborne[person_idx] = airborne
            ground_phase_count += len(_contiguous_runs(gt_valid[person_idx] & ~airborne))

            for start, end in _contiguous_runs(airborne):
                phase_height = foot_height[person_idx, start:end]
                peak_height = float(np.max(phase_height))
                category = "grand_saut" if peak_height >= args.large_jump_threshold_m else "petit_saut"
                sequence_jump_category[person_idx, start:end] = 2 if category == "grand_saut" else 1
                duration_frames = end - start
                jump_phases.append(
                    JumpPhase(
                        sequence=sequence,
                        person_idx=person_idx,
                        start_frame=start,
                        end_frame=end - 1,
                        duration_frames=duration_frames,
                        duration_seconds=duration_frames / args.fps,
                        peak_foot_clearance_m=peak_height,
                        mean_foot_clearance_m=float(np.mean(phase_height)),
                        category=category,
                    )
                )

        valid_gt_frame_count += int(gt_valid.sum())
        airborne_gt_frame_count += int(sequence_airborne.sum())
        all_height.append(plane_foot_height[error_valid])
        all_error.append(root_error[error_valid])
        ground_errors.append(root_error[error_valid & ~sequence_airborne])
        airborne_errors.append(root_error[error_valid & sequence_airborne])

        person_indices, local_frame_indices = np.nonzero(error_valid)
        frame_sequence_indices.append(
            np.full(person_indices.size, sequence_index - 1, dtype=np.int16)
        )
        frame_person_indices.append(person_indices.astype(np.int16, copy=False))
        frame_indices.append(local_frame_indices.astype(np.int32, copy=False))
        frame_root_deltas.append(root_delta[error_valid].astype(np.float32, copy=False))
        frame_left_plane_heights.append(left_plane_height[error_valid].astype(np.float32, copy=False))
        frame_right_plane_heights.append(right_plane_height[error_valid].astype(np.float32, copy=False))
        frame_left_clearances.append(left_height[error_valid].astype(np.float32, copy=False))
        frame_right_clearances.append(right_height[error_valid].astype(np.float32, copy=False))
        frame_airborne.append(sequence_airborne[error_valid])
        frame_jump_categories.append(sequence_jump_category[error_valid])

    height = np.concatenate(all_height) if all_height else np.empty(0, dtype=np.float64)
    error = np.concatenate(all_error) if all_error else np.empty(0, dtype=np.float64)
    ground_error = np.concatenate(ground_errors) if ground_errors else np.empty(0, dtype=np.float64)
    airborne_error = np.concatenate(airborne_errors) if airborne_errors else np.empty(0, dtype=np.float64)

    small_phases = [phase for phase in jump_phases if phase.category == "petit_saut"]
    large_phases = [phase for phase in jump_phases if phase.category == "grand_saut"]
    durations_frames = np.asarray([phase.duration_frames for phase in jump_phases], dtype=np.float64)
    durations_seconds = durations_frames / args.fps
    left_reference_values = (
        np.concatenate(left_ground_references) if left_ground_references else np.empty(0, dtype=np.float64)
    )
    right_reference_values = (
        np.concatenate(right_ground_references) if right_ground_references else np.empty(0, dtype=np.float64)
    )
    height_max, error_max = _plot_limits(height, error, args.plot_max_height_m)

    curve_rows = _binned_root_error(
        height=height,
        error=error,
        bin_width=args.height_bin_width_m,
        height_max=height_max,
    )

    _save_jump_histogram(
        out_path=output_dir / "jump_phase_histogram.png",
        ground_phase_count=ground_phase_count,
        small_jump_count=len(small_phases),
        large_jump_count=len(large_phases),
        duration_frames=durations_frames,
        fps=args.fps,
    )
    _save_scatter(
        out_path=output_dir / "root_error_vs_foot_height_scatter.png",
        height=height,
        error=error,
        max_points=args.max_scatter_points,
        height_max=height_max,
        error_max=error_max,
        seed=args.seed,
    )
    _save_ground_air_histogram(
        out_path=output_dir / "root_error_ground_vs_air_histogram.png",
        ground_error=ground_error,
        airborne_error=airborne_error,
    )
    _save_mean_error_curve(
        out_path=output_dir / "mean_root_error_vs_foot_height.png",
        rows=curve_rows,
        min_frames_per_bin=args.min_frames_per_height_bin,
    )

    phase_rows = [asdict(phase) for phase in jump_phases]
    _write_csv(
        output_dir / "jump_phases.csv",
        phase_rows,
        (
            "sequence",
            "person_idx",
            "start_frame",
            "end_frame",
            "duration_frames",
            "duration_seconds",
            "peak_foot_clearance_m",
            "mean_foot_clearance_m",
            "category",
        ),
    )
    _write_csv(
        output_dir / "root_error_by_foot_height.csv",
        curve_rows,
        (
            "height_start_m",
            "height_end_m",
            "height_center_m",
            "frame_count",
            "mean_root_error_m",
            "median_root_error_m",
        ),
    )
    np.savez_compressed(
        output_dir / "frame_metrics.npz",
        sequence_names=np.asarray(sequences, dtype=str),
        sequence_idx=np.concatenate(frame_sequence_indices),
        person_idx=np.concatenate(frame_person_indices),
        frame_idx=np.concatenate(frame_indices),
        root_delta_cam_m=np.concatenate(frame_root_deltas, axis=0),
        root_error_m=error.astype(np.float32, copy=False),
        left_foot_plane_height_m=np.concatenate(frame_left_plane_heights),
        right_foot_plane_height_m=np.concatenate(frame_right_plane_heights),
        lower_foot_plane_height_m=height.astype(np.float32, copy=False),
        left_foot_clearance_m=np.concatenate(frame_left_clearances),
        right_foot_clearance_m=np.concatenate(frame_right_clearances),
        is_airborne=np.concatenate(frame_airborne),
        jump_category=np.concatenate(frame_jump_categories),
        jump_category_labels=np.asarray(["ground", "small_jump", "large_jump"], dtype=str),
    )

    correlation = None
    if height.size >= 2 and float(np.std(height)) > 0.0 and float(np.std(error)) > 0.0:
        correlation = float(np.corrcoef(height, error)[0, 1])
    summary: dict[str, object] = {
        "definition": {
            "foot_height": (
                "height used by scatter/curve: minimum(left foot, right foot), each foot=min(toe/heel GT "
                "distances to pitch plane), clipped at 0 m"
            ),
            "foot_clearance_for_detection": (
                "raw foot plane height minus its per-track observed ground-contact reference, clipped at 0 m"
            ),
            "ground_contact_reference": (
                "local per-player/per-foot percentile of valid raw plane heights, interpolated in time; "
                "percentile 0 disables this BODY-25 bias correction"
            ),
            "airborne": "left foot clearance >= threshold AND right foot clearance >= threshold",
            "small_vs_large": "classification by maximum lower-foot clearance during the airborne phase",
            "root_delta": "Y_root_cam_gt - root_init_cam",
            "no_jump_bar": "number of contiguous valid ground phases",
        },
        "settings": {
            "split": args.split,
            "fps": args.fps,
            "airborne_threshold_m": args.airborne_threshold_m,
            "large_jump_threshold_m": args.large_jump_threshold_m,
            "min_airborne_frames": args.min_airborne_frames,
            "max_ground_gap_frames": args.max_ground_gap_frames,
            "ground_reference_percentile": args.ground_reference_percentile,
            "ground_reference_window_s": args.ground_reference_window_s,
            "left_foot_joints": list(args.left_foot_joints),
            "right_foot_joints": list(args.right_foot_joints),
            "height_bin_width_m": args.height_bin_width_m,
        },
        "counts": {
            "sequences": len(sequences),
            "player_tracks": track_count,
            "valid_gt_frames": valid_gt_frame_count,
            "frames_with_root_error": int(error.size),
            "ground_frames_with_root_error": int(ground_error.size),
            "airborne_frames_with_root_error": int(airborne_error.size),
            "ground_phases": ground_phase_count,
            "small_jump_phases": len(small_phases),
            "large_jump_phases": len(large_phases),
            "all_jump_phases": len(jump_phases),
            "airborne_gt_frames": airborne_gt_frame_count,
        },
        "jump_duration_frames": _describe(durations_frames),
        "jump_duration_seconds": _describe(durations_seconds),
        "ground_contact_reference_m": {
            "left_foot": _describe(left_reference_values),
            "right_foot": _describe(right_reference_values),
        },
        "root_error_m": {
            "all": _describe(error),
            "ground": _describe(ground_error),
            "airborne": _describe(airborne_error),
        },
        "pearson_root_error_vs_foot_height": correlation,
        "outputs": {
            "jump_phase_histogram": "jump_phase_histogram.png",
            "scatter": "root_error_vs_foot_height_scatter.png",
            "ground_vs_air_histogram": "root_error_ground_vs_air_histogram.png",
            "mean_error_curve": "mean_root_error_vs_foot_height.png",
            "jump_phases_csv": "jump_phases.csv",
            "height_bins_csv": "root_error_by_foot_height.csv",
            "per_frame_metrics": "frame_metrics.npz",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detecte les phases aeriennes GT 3D et analyse ||root_cam_gt - root_cam_init||."
    )
    parser.add_argument("--data-dir", default="data", help="Racine du dataset (defaut: data).")
    parser.add_argument("--features-dir", default=None, help="Dossier des features brutes (defaut: DATA_DIR/features).")
    parser.add_argument("--root-init-dir", default=None, help="Roots initiaux bruts en metres (defaut: DATA_DIR/root_init_cam).")
    parser.add_argument("--output-dir", default="outputs/airborne_root_analysis", help="Dossier de sortie.")
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="all")
    parser.add_argument("--split-json", default=None, help="split.json (defaut: DATA_DIR/features_normalized/split.json).")
    parser.add_argument("--sequences", nargs="+", default=None, help="Sous-ensemble explicite de sequences.")
    parser.add_argument("--pitch-points", default=None, help="Fichier de points 3D du terrain.")
    parser.add_argument("--fps", type=float, default=25.0, help="Frequence video utilisee pour les durees en secondes.")
    parser.add_argument(
        "--airborne-threshold-m",
        type=float,
        default=0.05,
        help="Degagement minimal de chacun des deux pieds pour etre en l'air (defaut: 0.05 m).",
    )
    parser.add_argument(
        "--large-jump-threshold-m",
        type=float,
        default=0.20,
        help="Hauteur maximale du pied le plus bas separant petit/grand saut (defaut: 0.20 m).",
    )
    parser.add_argument("--min-airborne-frames", type=int, default=2, help="Duree minimale d'une phase aerienne.")
    parser.add_argument(
        "--max-ground-gap-frames",
        type=int,
        default=1,
        help="Petit trou au sol comble entre deux detections aeriennes (defaut: 1 frame).",
    )
    parser.add_argument(
        "--left-foot-joints",
        type=_parse_joint_indices,
        default=DEFAULT_LEFT_FOOT_JOINTS,
        help="Indices BODY-25 du pied gauche, separes par des virgules.",
    )
    parser.add_argument(
        "--right-foot-joints",
        type=_parse_joint_indices,
        default=DEFAULT_RIGHT_FOOT_JOINTS,
        help="Indices BODY-25 du pied droit, separes par des virgules.",
    )
    parser.add_argument(
        "--ground-reference-percentile",
        type=float,
        default=20.0,
        help=(
            "Percentile bas par joueur/pied utilise comme contact au sol pour corriger le biais des keypoints "
            "BODY-25 (defaut: 20; 0 desactive la correction)."
        ),
    )
    parser.add_argument(
        "--ground-reference-window-s",
        type=float,
        default=5.0,
        help=(
            "Fenetre temporelle de la reference locale de contact (defaut: 5 s; "
            "0 utilise une reference constante sur toute la sequence)."
        ),
    )
    parser.add_argument("--height-bin-width-m", type=float, default=0.025, help="Largeur des bins de la courbe.")
    parser.add_argument("--min-frames-per-height-bin", type=int, default=20)
    parser.add_argument("--plot-max-height-m", type=float, default=None, help="Limite x des plots; auto si omise.")
    parser.add_argument("--max-scatter-points", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=12345, help="Seed du sous-echantillonnage du scatter.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps doit etre > 0")
    if args.airborne_threshold_m < 0:
        raise ValueError("--airborne-threshold-m doit etre >= 0")
    if args.large_jump_threshold_m <= args.airborne_threshold_m:
        raise ValueError("--large-jump-threshold-m doit etre superieur a --airborne-threshold-m")
    if not (0.0 <= args.ground_reference_percentile <= 50.0):
        raise ValueError("--ground-reference-percentile doit etre compris entre 0 et 50")
    if args.ground_reference_window_s < 0.0:
        raise ValueError("--ground-reference-window-s doit etre >= 0")
    if args.min_airborne_frames < 1 or args.max_ground_gap_frames < 0:
        raise ValueError("Durees de filtrage invalides")
    if args.height_bin_width_m <= 0 or args.min_frames_per_height_bin < 1:
        raise ValueError("Configuration des bins invalide")
    if args.plot_max_height_m is not None and args.plot_max_height_m <= 0:
        raise ValueError("--plot-max-height-m doit etre > 0")
    if args.max_scatter_points < 1:
        raise ValueError("--max-scatter-points doit etre >= 1")


def main() -> None:
    args = _build_argparser().parse_args()
    _validate_args(args)
    summary = analyze(args)
    counts = summary["counts"]
    duration = summary["jump_duration_seconds"]
    print(
        "Termine: "
        f"{counts['all_jump_phases']} sauts "
        f"({counts['small_jump_phases']} petits, {counts['large_jump_phases']} grands), "
        f"duree moyenne={duration['mean']} s, mediane={duration['median']} s."
    )
    print(f"Sorties: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
#PYTHONPATH=src python -m field_converter.utils.analyze_airborne_root_error --data-dir data --output-dir outputs/airborne_root_analysis