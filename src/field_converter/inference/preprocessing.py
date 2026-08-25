"""Preprocess qualitative datasets for root-refiner inference without GT labels.

The raw inference layout is intentionally independent from the training layout::

    data/data_inference/
      boxes/<sequence>.npy
      cameras/<sequence>.npz
      skel_2d/<sequence>.npy
      skel_3d_relative/<sequence>.npy
      frames/<sequence>/*.jpg                 # optional

Raw person arrays may be stored as ``(T,N,...)`` or ``(N,T,...)``.  This
module converts them to the training convention ``(N,T,...)``, builds the same
features as :mod:`field_converter.data_preparation.features_creation`, and
applies the train-only statistics produced by ``normalize.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from field_converter.data_preparation.features_creation import FeatureCreator
from field_converter.data_preparation.generate_root_init import compute_root_init_cam
from field_converter.data_preparation.normalize import NormalizationStats
from field_converter.data_preparation.normalize_root_init import normalize_one
from field_converter.inference.resampling import (
    build_resampling_plan,
    resample_continuous,
    resample_rotations,
)
from field_converter.inference.world_alignment import (
    WorldAlignment,
    WorldAlignmentMode,
    align_camera_extrinsics,
    camera_centers_world,
    detect_world_alignment,
)


NUM_JOINTS = 25
NUM_PITCH_POINTS = 50


@dataclass(frozen=True)
class PreparedSequence:
    sequence: str
    raw: Dict[str, np.ndarray]
    normalized: Dict[str, np.ndarray]
    world_alignment: WorldAlignment
    raw_features_path: Path
    normalized_features_path: Path
    root_init_path: Path
    root_init_normalized_path: Path


def discover_sequences(input_dir: Path | str) -> list[str]:
    """Return sequences for which all four required raw inputs exist."""
    root = Path(input_dir)
    folders_and_suffixes = (
        ("boxes", ".npy"),
        ("cameras", ".npz"),
        ("skel_2d", ".npy"),
        ("skel_3d_relative", ".npy"),
    )
    stem_sets: list[set[str]] = []
    for dirname, suffix in folders_and_suffixes:
        folder = root / dirname
        if not folder.exists():
            raise FileNotFoundError(f"Missing inference input folder: {folder}")
        stem_sets.append({path.stem for path in folder.glob(f"*{suffix}")})

    sequences = sorted(set.intersection(*stem_sets)) if stem_sets else []
    if not sequences:
        raise FileNotFoundError(
            f"No complete inference sequence found under {root}; expected boxes, cameras, "
            "skel_2d and skel_3d_relative files with matching stems"
        )
    return sequences


def select_sequences(input_dir: Path | str, requested: Optional[Iterable[str]]) -> list[str]:
    available = discover_sequences(input_dir)
    if requested is None:
        return available
    requested_list = [str(value) for value in requested]
    missing = sorted(set(requested_list) - set(available))
    if missing:
        raise FileNotFoundError(f"Incomplete or unknown inference sequences: {missing}; available={available}")
    return requested_list


def _ensure_nt(arr: np.ndarray, *, T: int, trailing_shape: Tuple[int, ...], name: str) -> np.ndarray:
    """Align a raw inference array to ``(N,T,...)`` using camera length T."""
    value = np.asarray(arr)
    expected_ndim = 2 + len(trailing_shape)
    if value.ndim != expected_ndim or tuple(value.shape[-len(trailing_shape) :]) != trailing_shape:
        raise ValueError(
            f"{name}: expected (T,N,{','.join(map(str, trailing_shape))}) or "
            f"(N,T,{','.join(map(str, trailing_shape))}), got {value.shape}"
        )

    # The documented raw inference layout is (T,N,...), so prefer that
    # interpretation in the rare ambiguous case N == T.
    if value.shape[0] == T:
        return np.swapaxes(value, 0, 1)
    if value.shape[1] == T:
        return value
    raise ValueError(f"{name}: neither of the first two dimensions matches camera T={T}: {value.shape}")


def _load_camera(path: Path, *, k_to_zero: bool) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as npz:
        missing = [key for key in ("K", "R", "t") if key not in npz.files]
        if missing:
            raise KeyError(f"{path}: missing camera keys {missing}")
        K = np.asarray(npz["K"], dtype=np.float64)
        R = np.asarray(npz["R"], dtype=np.float64)
        t = np.asarray(npz["t"], dtype=np.float64)
        k = (
            np.asarray(npz["k"], dtype=np.float64)
            if "k" in npz.files
            else np.zeros((K.shape[0], 2), dtype=np.float64)
        )

    T = int(K.shape[0])
    if K.shape != (T, 3, 3) or R.shape != (T, 3, 3) or t.shape != (T, 3):
        raise ValueError(f"Invalid camera shapes in {path}: K={K.shape}, R={R.shape}, t={t.shape}")
    if k.ndim != 2 or k.shape[0] != T or k.shape[1] < 2:
        raise ValueError(f"Invalid distortion shape in {path}: k={k.shape}, expected (T,>=2)")

    # The project radial model uses the first two columns (k1, k2); camera
    # files commonly contain three additional coefficients that are ignored.
    k = k[:, :2].copy()
    if k_to_zero:
        k[:, :2] = 0.0
    return {"K": K, "R": R, "t": t, "k": k}


def _infer_image_size(K: np.ndarray, explicit: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if explicit is not None:
        W, H = (int(explicit[0]), int(explicit[1]))
    else:
        W = int(round(2.0 * float(np.nanmedian(K[:, 0, 2]))))
        H = int(round(2.0 * float(np.nanmedian(K[:, 1, 2]))))
    if W <= 0 or H <= 0:
        raise ValueError(f"Invalid inferred image size (W,H)=({W},{H}); pass an explicit image size")
    return W, H


def _normalize_sam2d_box(
    sam2d: np.ndarray,
    boxes: np.ndarray,
    *,
    min_bbox_size_px: float,
) -> np.ndarray:
    x1, y1, x2, y2 = [boxes[..., i] for i in range(4)]
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    width = x2 - x1
    height = y2 - y1
    valid_box = (
        np.isfinite(boxes).all(axis=-1)
        & (width >= float(min_bbox_size_px))
        & (height >= float(min_bbox_size_px))
    )

    out = np.zeros_like(sam2d, dtype=np.float64)
    out[..., 0] = np.divide(
        sam2d[..., 0] - cx[..., None],
        width[..., None],
        out=np.full_like(sam2d[..., 0], np.nan, dtype=np.float64),
        where=valid_box[..., None],
    )
    out[..., 1] = np.divide(
        sam2d[..., 1] - cy[..., None],
        height[..., None],
        out=np.full_like(sam2d[..., 1], np.nan, dtype=np.float64),
        where=valid_box[..., None],
    )
    out[~valid_box] = 0.0
    return out.astype(np.float32)


def _bbox_log_ratio(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    out = np.asarray(features, dtype=np.float64).copy()
    ratio = out[..., 4]
    valid = np.isfinite(ratio) & (ratio > eps)
    logged = np.zeros_like(ratio)
    np.log(ratio, out=logged, where=valid)
    out[..., 4] = logged
    return out.astype(np.float32)


def _normalize_pitch_points(
    pitch_points_2d: np.ndarray,
    valid_pitch_points: np.ndarray,
    *,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    W, H = image_size
    points = np.asarray(pitch_points_2d, dtype=np.float64)
    valid = np.asarray(valid_pitch_points, dtype=bool).copy()
    valid &= np.isfinite(points).all(axis=-1)
    valid &= (points[..., 0] >= 0.0) & (points[..., 0] < float(W))
    valid &= (points[..., 1] >= 0.0) & (points[..., 1] < float(H))

    out = np.zeros_like(points)
    out[..., 0] = np.where(valid, points[..., 0] / max(float(W), 1e-8), 0.0)
    out[..., 1] = np.where(valid, points[..., 1] / max(float(H), 1e-8), 0.0)
    return out.astype(np.float32), valid


def _frame_numbers(input_dir: Path, sequence: str, T: int) -> np.ndarray:
    frame_dir = input_dir / "frames" / sequence
    if not frame_dir.exists():
        return np.arange(T, dtype=np.int64)
    paths = sorted(path for path in frame_dir.iterdir() if path.is_file())
    if len(paths) != T:
        return np.arange(T, dtype=np.int64)
    numbers: list[int] = []
    for path in paths:
        matches = re.findall(r"\d+", path.stem)
        if not matches:
            return np.arange(T, dtype=np.int64)
        numbers.append(int(matches[-1]))
    return np.asarray(numbers, dtype=np.int64)


def _save_npz_atomic(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(path)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.npy")
    np.save(tmp_path, array)
    tmp_path.replace(path)


def prepare_sequence(
    *,
    input_dir: Path | str,
    sequence: str,
    stats_path: Path | str,
    pitch_points_path: Path | str,
    image_size: Optional[Tuple[int, int]] = None,
    k_to_zero: bool = False,
    sam3d_sign: int = 1,
    world_alignment_mode: WorldAlignmentMode = "auto",
    min_bbox_size_px: float = 4.0,
    box_normalization_min_size_px: float = 10.0,
    source_fps: Optional[float] = None,
    target_fps: float = 50.0,
    save_intermediate: bool = True,
) -> PreparedSequence:
    """Build raw and train-normalized inference features for one sequence."""
    root = Path(input_dir)
    stats_file = Path(stats_path)
    pitch_file = Path(pitch_points_path)
    if not stats_file.exists():
        raise FileNotFoundError(f"Missing training normalization statistics: {stats_file}")
    if not pitch_file.exists():
        raise FileNotFoundError(f"Missing pitch point file: {pitch_file}")
    if pitch_file.name != "pitch_points.txt":
        raise ValueError(
            f"pitch_points_path must point to a file named pitch_points.txt for FeatureCreator reuse: {pitch_file}"
        )
    if sam3d_sign not in {-1, 1}:
        raise ValueError(f"sam3d_sign must be +1 or -1, got {sam3d_sign}")

    stats = NormalizationStats.load(stats_file)
    camera = _load_camera(root / "cameras" / f"{sequence}.npz", k_to_zero=k_to_zero)
    K_source_timeline = camera["K"]
    R_source_timeline = camera["R"]
    t_source_timeline = camera["t"]
    k_source_timeline = camera["k"]
    source_T = int(K_source_timeline.shape[0])
    world_alignment = detect_world_alignment(
        R_source_timeline,
        t_source_timeline,
        train_mean_C=stats.mean_C,
        train_std_C=stats.std_C,
        mode=world_alignment_mode,
    )

    boxes = _ensure_nt(
        np.load(root / "boxes" / f"{sequence}.npy"),
        T=source_T,
        trailing_shape=(4,),
        name=f"{sequence} boxes",
    ).astype(np.float32)
    sam2d = _ensure_nt(
        np.load(root / "skel_2d" / f"{sequence}.npy"),
        T=source_T,
        trailing_shape=(NUM_JOINTS, 2),
        name=f"{sequence} SAM2D",
    ).astype(np.float32)
    sam3d = _ensure_nt(
        np.load(root / "skel_3d_relative" / f"{sequence}.npy"),
        T=source_T,
        trailing_shape=(NUM_JOINTS, 3),
        name=f"{sequence} SAM3D",
    ).astype(np.float32)
    sam3d = (float(sam3d_sign) * sam3d).astype(np.float32, copy=False)

    if boxes.shape[:2] != sam2d.shape[:2] or sam2d.shape[:3] != sam3d.shape[:3]:
        raise ValueError(
            f"{sequence}: person/time mismatch boxes={boxes.shape}, SAM2D={sam2d.shape}, SAM3D={sam3d.shape}"
        )

    source_frame_numbers = _frame_numbers(root, sequence, source_T)
    if source_fps is None:
        K = K_source_timeline
        R_source = R_source_timeline
        t_source = t_source_timeline
        k = k_source_timeline
        T = source_T
        frame_numbers = source_frame_numbers
        source_frame_indices = np.arange(T, dtype=np.int64)
        source_frame_positions = source_frame_indices.astype(np.float64)
        source_frame_numbers_float = source_frame_numbers.astype(np.float64)
        timestamps_s = np.full((T,), np.nan, dtype=np.float64)
        output_fps = np.nan
        resampling_meta: dict[str, object] = {
            "requested": False,
            "applied": False,
            "mode": "disabled",
            "source_fps": None,
            "target_fps": float(target_fps),
            "source_num_frames": source_T,
            "target_num_frames": T,
            "note": "provide --source-fps to enable temporal resampling",
        }
    else:
        plan = build_resampling_plan(
            source_T,
            source_fps=float(source_fps),
            target_fps=float(target_fps),
        )
        K = resample_continuous(K_source_timeline, plan, time_axis=0)
        R_source = resample_rotations(R_source_timeline, plan)
        t_source = resample_continuous(t_source_timeline, plan, time_axis=0)
        k = resample_continuous(k_source_timeline, plan, time_axis=0)
        boxes = resample_continuous(boxes, plan, time_axis=1)
        sam2d = resample_continuous(sam2d, plan, time_axis=1)
        sam3d = resample_continuous(sam3d, plan, time_axis=1)
        T = plan.target_num_frames
        source_frame_indices = plan.nearest_source_indices.copy()
        source_frame_positions = plan.source_positions.copy()
        frame_numbers = source_frame_numbers[source_frame_indices]
        source_frame_numbers_float = np.interp(
            source_frame_positions,
            np.arange(source_T, dtype=np.float64),
            source_frame_numbers.astype(np.float64),
        )
        timestamps_s = plan.target_times_s.copy()
        output_fps = plan.target_fps
        resampling_meta = plan.to_metadata()

    # Alignment is selected from the original camera trajectory, then applied
    # after temporal resampling so R == R_source @ Q.T remains exact.
    R, t = align_camera_extrinsics(R_source, t_source, world_alignment)
    N = int(boxes.shape[0])
    W, H = _infer_image_size(K, image_size)

    creator = FeatureCreator(
        data_dir=pitch_file.parent,
        image_size=(W, H),
        pelvis_mode=stats.pelvis_mode,
        num_pitch_points=NUM_PITCH_POINTS,
        k_to_zero=k_to_zero,
    )

    valid_joints = np.isfinite(sam2d).all(axis=-1) & np.isfinite(sam3d).all(axis=-1)
    box_width = boxes[..., 2] - boxes[..., 0]
    box_height = boxes[..., 3] - boxes[..., 1]
    valid_box = (
        np.isfinite(boxes).all(axis=-1)
        & (box_width >= float(min_bbox_size_px))
        & (box_height >= float(min_bbox_size_px))
    )
    valid_mask = valid_box & valid_joints.any(axis=-1)

    bbox_feat_clean = creator.make_bbox_features(boxes, (W, H), min_size_px=min_bbox_size_px)
    cam_base, cam_boosted = creator.make_camera_features(K, R, t, k, (W, H))
    pitch_world, pitch_2d, valid_pitch = creator.project_pitch_points_to_image(
        K,
        R,
        t,
        k,
        image_size=(W, H),
    )
    ground = creator.compute_ground_intersections_from_sam(sam2d, sam3d, K, R, t, k)

    root_payload = {
        "skel_2d_sam3dbody_from_bbox_gt": sam2d,
        "skel_3d_sam3dbody_from_bbox_gt": sam3d,
        "K": K,
        "R": R,
        "t": t,
        "k": k,
    }
    root_init_cam = compute_root_init_cam(root_payload, creator, sequence=sequence)
    root_init_norm = normalize_one(root_init_cam, mean_root=stats.mean_root, std_root=stats.std_root)
    root_init_valid = np.isfinite(root_init_cam).all(axis=-1)

    camera_centers = camera_centers_world(R, t).astype(np.float32)
    camera_centers_source = camera_centers_world(R_source, t_source).astype(np.float32)
    raw_meta = {
        "sequence": sequence,
        "layout": "(N,T,...)",
        "source_layout": "auto-detected from camera T; (T,N,...) preferred if ambiguous",
        "image_size": [W, H],
        "pelvis_mode": stats.pelvis_mode,
        "sam3d_sign": int(sam3d_sign),
        "k_to_zero": bool(k_to_zero),
        "distortion_coefficients_all_zero": bool(np.allclose(k[:, :2], 0.0)),
        "min_bbox_size_px": float(min_bbox_size_px),
        "box_normalization_min_size_px": float(box_normalization_min_size_px),
        "pitch_points_path": str(pitch_file.resolve()),
        "stats_path": str(stats_file.resolve()),
        "num_persons": N,
        "num_frames": T,
        "num_source_frames": source_T,
        "resampling": resampling_meta,
        "world_alignment": world_alignment.to_metadata(),
        "world_frame": "training-aligned after optional origin-preserving 180-degree rotation",
        "pitch_points_frame": "training-aligned canonical pitch frame",
    }

    raw: Dict[str, np.ndarray] = {
        "bbox_feat": bbox_feat_clean.astype(np.float32),
        "bbox_feat_clean": bbox_feat_clean.astype(np.float32),
        "cam_feat_base_clean": cam_base.astype(np.float32),
        "cam_feat_base_noisy": cam_base.astype(np.float32),
        "cam_feat_boosted_clean": cam_boosted.astype(np.float32),
        "cam_feat_boosted_noisy": cam_boosted.astype(np.float32),
        "valid_mask": valid_mask.astype(bool),
        "valid_joints": valid_joints.astype(bool),
        "root_init_valid": root_init_valid.astype(bool),
        "boxes_xyxy": boxes.astype(np.float32),
        "K": K.astype(np.float32),
        "R": R.astype(np.float32),
        "t": t.astype(np.float32),
        "R_source": R_source.astype(np.float32),
        "t_source": t_source.astype(np.float32),
        "k": k.astype(np.float32),
        "image_size": np.asarray([W, H], dtype=np.int32),
        "camera_center_world": camera_centers,
        "camera_center_world_source": camera_centers_source,
        "world_alignment_rotation": world_alignment.rotation_source_to_aligned.astype(np.float32),
        "world_alignment_applied": np.asarray(world_alignment.applied, dtype=bool),
        "pitch_points_world": pitch_world.astype(np.float32),
        "pitch_points_2d": pitch_2d.astype(np.float32),
        "valid_pitch_points": valid_pitch.astype(bool),
        "skel_2d_sam3dbody_from_bbox_gt": sam2d.astype(np.float32),
        "skel_3d_sam3dbody_from_bbox_gt": sam3d.astype(np.float32),
        "ground_intersection": ground.astype(np.float32),
        "root_init_cam": root_init_cam.astype(np.float32),
        "sam3d_orientation_sign": np.asarray(int(sam3d_sign), dtype=np.int8),
        "frame_numbers": frame_numbers,
        "source_num_frames": np.asarray(source_T, dtype=np.int64),
        "source_frame_indices": source_frame_indices,
        "source_frame_positions": source_frame_positions,
        "source_frame_numbers_float": source_frame_numbers_float,
        "timestamps_s": timestamps_s,
        "source_fps": np.asarray(np.nan if source_fps is None else float(source_fps), dtype=np.float64),
        "output_fps": np.asarray(output_fps, dtype=np.float64),
        "meta_json": np.array(json.dumps(raw_meta), dtype=object),
    }

    pelvis = creator.compute_pelvis(sam3d, mode=stats.pelvis_mode)
    sam3d_rel = np.asarray(sam3d, dtype=np.float64) - np.asarray(pelvis, dtype=np.float64)[..., None, :]
    sam3d_norm = (
        (sam3d_rel - stats.mean_sam3d_rel[None, None, :, :])
        / stats.std_sam3d_rel[None, None, :, :]
    ).astype(np.float32)

    sam2d_norm = np.asarray(sam2d, dtype=np.float64).copy()
    sam2d_norm[..., 0] /= float(W)
    sam2d_norm[..., 1] /= float(H)
    sam2d_box = _normalize_sam2d_box(
        sam2d,
        boxes,
        min_bbox_size_px=box_normalization_min_size_px,
    )
    pitch_norm, valid_pitch_norm = _normalize_pitch_points(pitch_2d, valid_pitch, image_size=(W, H))

    cam_boosted_norm = np.asarray(cam_boosted, dtype=np.float64).copy()
    cam_boosted_norm[:, 6:9] = (
        (cam_boosted_norm[:, 6:9] - stats.mean_C[None, :]) / stats.std_C[None, :]
    )
    ground_norm = (
        (np.asarray(ground, dtype=np.float64) - stats.mean_ground_intersection[None, None, :])
        / stats.std_ground_intersection[None, None, :]
    ).astype(np.float32)

    normalized: Dict[str, np.ndarray] = dict(raw)
    normalized.update(
        {
            "bbox_feat": _bbox_log_ratio(bbox_feat_clean),
            "bbox_feat_clean": _bbox_log_ratio(bbox_feat_clean),
            "cam_feat_boosted_clean": cam_boosted_norm.astype(np.float32),
            "cam_feat_boosted_noisy": cam_boosted_norm.astype(np.float32),
            "skel_3d_sam3dbody_from_bbox_gt": sam3d_norm,
            "skel_3d_sam3dbody_from_bbox_gt_rel_m": sam3d_rel.astype(np.float32),
            "skel_2d_sam3dbody_from_bbox_gt": sam2d_norm.astype(np.float32),
            "skel_2d_sam3dbody_from_bbox_gt_box": sam2d_box,
            "skel_2d_sam3dbody_from_bbox_gt_pixels": sam2d.astype(np.float32),
            "pitch_points_2d": pitch_norm,
            "valid_pitch_points": valid_pitch_norm.astype(bool),
            "ground_intersection": ground_norm,
            "root_init_norm": root_init_norm.astype(np.float32),
            "meta_norm_json": np.array(
                json.dumps(
                    {
                        "normalized": True,
                        "stats_path": str(stats_file.resolve()),
                        "noisy_features_at_inference": "aliased to observed clean features; no artificial noise added",
                        "root_init": "(root_init_cam - mean_root) / std_root",
                    }
                ),
                dtype=object,
            ),
        }
    )

    raw_path = root / "features" / f"{sequence}.npz"
    normalized_path = root / "features_normalized" / f"{sequence}.npz"
    root_init_path = root / "root_init_cam" / f"{sequence}.npy"
    root_init_norm_path = root / "root_init_cam_normalized" / f"{sequence}.npy"
    if save_intermediate:
        _save_npz_atomic(raw_path, raw)
        _save_npz_atomic(normalized_path, normalized)
        _save_npy_atomic(root / "ground_intersection" / f"{sequence}.npy", ground.astype(np.float32))
        _save_npy_atomic(root_init_path, root_init_cam.astype(np.float32))
        _save_npy_atomic(root_init_norm_path, root_init_norm.astype(np.float32))

    return PreparedSequence(
        sequence=sequence,
        raw=raw,
        normalized=normalized,
        world_alignment=world_alignment,
        raw_features_path=raw_path,
        normalized_features_path=normalized_path,
        root_init_path=root_init_path,
        root_init_normalized_path=root_init_norm_path,
    )
