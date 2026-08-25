"""Model input assembly, prediction and qualitative export for inference data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from field_converter.data_preparation.features_creation import FeatureCreator
from field_converter.data_preparation.normalize import NormalizationStats
from field_converter.inference.modeling import InferenceModel
from field_converter.inference.preprocessing import NUM_JOINTS, NUM_PITCH_POINTS, PreparedSequence
from field_converter.inference.world_alignment import transform_world_points_aligned_to_source
from field_converter.models.temporal import forward_temporal_root_model
from field_converter.training.dataset import infer_input_dim
from field_converter.training.filters import filter_valid_mask_bbox_geometry, filter_valid_mask_in_image
from field_converter.training.prediction import apply_prediction_mode
from field_converter.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class DensePrediction:
    root_pred_norm: np.ndarray
    model_output_norm: np.ndarray
    valid_mask: np.ndarray
    coverage_count: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    sequence: str
    predictions_path: Path
    csv_path: Optional[Path]
    summary_path: Path
    summary: Dict[str, Any]


def _nan_to_num(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(value, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _broadcast_time_feature(value: np.ndarray, N: int, T: int, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] != T:
        raise ValueError(f"{name}: expected time dimension T={T}, got {arr.shape}")
    return np.broadcast_to(arr[None, ...], (N,) + arr.shape)


def build_input_matrix(payload: Dict[str, np.ndarray], input_config: Any) -> np.ndarray:
    """Build ``(N,T,D)`` inputs in the exact order used by training datasets."""
    valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must have shape (N,T), got {valid_mask.shape}")
    N, T = (int(valid_mask.shape[0]), int(valid_mask.shape[1]))
    parts: list[np.ndarray] = []

    if input_config.use_x3d_sam_rel:
        value = np.asarray(payload["skel_3d_sam3dbody_from_bbox_gt"], dtype=np.float32)
        parts.append(value.reshape(N, T, NUM_JOINTS * 3))

    if input_config.use_x2d_img:
        value = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt"], dtype=np.float32)
        parts.append(value.reshape(N, T, NUM_JOINTS * 2))

    if input_config.use_x2d_box:
        value = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt_box"], dtype=np.float32)
        parts.append(value.reshape(N, T, NUM_JOINTS * 2))

    if input_config.use_pitch_points_2d:
        pitch = np.asarray(payload["pitch_points_2d"], dtype=np.float32)
        pitch_valid = np.asarray(payload["valid_pitch_points"], dtype=bool)
        if pitch.shape != (T, NUM_PITCH_POINTS, 2) or pitch_valid.shape != (T, NUM_PITCH_POINTS):
            raise ValueError(
                f"Expected pitch shapes {(T, NUM_PITCH_POINTS, 2)} and {(T, NUM_PITCH_POINTS)}, "
                f"got {pitch.shape} and {pitch_valid.shape}"
            )
        parts.append(_broadcast_time_feature(pitch, N, T, name="pitch_points_2d").reshape(N, T, -1))
        parts.append(
            _broadcast_time_feature(pitch_valid.astype(np.float32), N, T, name="valid_pitch_points")
        )

    if input_config.use_bbox_feat:
        bbox_key = "bbox_feat_clean" if input_config.bbox_clean_or_noisy == "clean" else "bbox_feat"
        parts.append(np.asarray(payload[bbox_key], dtype=np.float32).reshape(N, T, 5))

    if input_config.use_cam_feat:
        camera_keys = {
            "base_clean": "cam_feat_base_clean",
            "base_noisy": "cam_feat_base_noisy",
            "boosted_clean": "cam_feat_boosted_clean",
            "boosted_noisy": "cam_feat_boosted_noisy",
        }
        camera_key = camera_keys[input_config.cam_feat_type]
        camera = np.asarray(payload[camera_key], dtype=np.float32)
        parts.append(_broadcast_time_feature(camera, N, T, name=camera_key))

    if input_config.use_ground_intersection:
        parts.append(np.asarray(payload["ground_intersection"], dtype=np.float32).reshape(N, T, 3))

    if input_config.use_valid_joints_as_input:
        parts.append(np.asarray(payload["valid_joints"], dtype=np.float32).reshape(N, T, NUM_JOINTS))

    if not parts:
        raise RuntimeError("No model input features selected")
    inputs = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    _nan_to_num(inputs)

    expected_dim = infer_input_dim(input_config)
    if inputs.shape != (N, T, expected_dim):
        raise ValueError(f"Input shape mismatch: got {inputs.shape}, expected {(N, T, expected_dim)}")
    return inputs


def effective_valid_mask(prepared: PreparedSequence, runtime: InferenceModel) -> np.ndarray:
    """Apply inference-safe equivalents of the training dataset filters."""
    payload = prepared.normalized
    valid = np.asarray(payload["valid_mask"], dtype=bool).copy()
    dataset_cfg = runtime.config.dataset
    image_size = np.asarray(payload["image_size"], dtype=np.float32)

    if getattr(dataset_cfg, "min_in_image_joints_ratio", None) is not None:
        valid = filter_valid_mask_in_image(
            valid_mask=valid,
            valid_joints=np.asarray(payload["valid_joints"], dtype=bool),
            # There is no GT at inference; SAM2D is the observed proxy.
            Y_2d_gt=np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt_pixels"], dtype=np.float32),
            image_size=image_size,
            min_in_image_joints_ratio=float(dataset_cfg.min_in_image_joints_ratio),
        )

    valid = filter_valid_mask_bbox_geometry(
        valid_mask=valid,
        boxes_xyxy=np.asarray(payload["boxes_xyxy"], dtype=np.float32),
        image_size=image_size,
        min_bbox_width_px=getattr(dataset_cfg, "min_bbox_width_px", None),
        min_bbox_height_px=getattr(dataset_cfg, "min_bbox_height_px", None),
        min_bbox_margin_px=getattr(dataset_cfg, "min_bbox_margin_px", None),
    )

    if runtime.config.prediction_mode == "delta":
        # Never turn an invalid ray/ground hit into a plausible zero root.
        valid &= np.asarray(payload["root_init_valid"], dtype=bool)
        valid &= np.isfinite(np.asarray(payload["root_init_norm"])).all(axis=-1)
    return valid


@torch.no_grad()
def _predict_mlp(
    *,
    runtime: InferenceModel,
    inputs: np.ndarray,
    valid_mask: np.ndarray,
    root_init_norm: np.ndarray,
) -> DensePrediction:
    N, T, _ = inputs.shape
    root_pred = np.full((N, T, 3), np.nan, dtype=np.float32)
    model_output_full = np.full((N, T, 3), np.nan, dtype=np.float32)
    coverage = np.zeros((N, T), dtype=np.int32)
    person_idx, frame_idx = np.nonzero(valid_mask)

    for start in range(0, int(person_idx.size), runtime.batch_size):
        stop = min(start + runtime.batch_size, int(person_idx.size))
        p = person_idx[start:stop]
        f = frame_idx[start:stop]
        x = torch.from_numpy(inputs[p, f]).to(runtime.device, dtype=torch.float32)
        model_output = runtime.model(x).to(dtype=torch.float32)
        batch: Dict[str, Any] = {}
        if runtime.config.prediction_mode == "delta":
            batch["root_init_norm"] = torch.from_numpy(root_init_norm[p, f]).to(
                runtime.device,
                dtype=torch.float32,
            )
        prediction = apply_prediction_mode(
            model_output,
            batch,
            prediction_mode=runtime.config.prediction_mode,
        )
        output_np = model_output.detach().cpu().numpy().astype(np.float32)
        prediction_np = prediction.detach().cpu().numpy().astype(np.float32)
        finite = np.isfinite(output_np).all(axis=-1) & np.isfinite(prediction_np).all(axis=-1)
        if np.any(finite):
            root_pred[p[finite], f[finite]] = prediction_np[finite]
            model_output_full[p[finite], f[finite]] = output_np[finite]
            coverage[p[finite], f[finite]] = 1

    usable = valid_mask & (coverage > 0)
    return DensePrediction(root_pred, model_output_full, usable, coverage)


def _window_starts(T: int, window_size: int, stride: int) -> list[int]:
    if T < window_size:
        return [0]
    starts = list(range(0, T - window_size + 1, stride))
    last = T - window_size
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


def _window_specs(valid_mask: np.ndarray, window_size: int, stride: int) -> list[Tuple[int, np.ndarray]]:
    N, T = valid_mask.shape
    specs: list[Tuple[int, np.ndarray]] = []
    for person_idx in range(N):
        for start in _window_starts(T, window_size, stride):
            frames = np.full((window_size,), -1, dtype=np.int64)
            if T >= window_size:
                frames[:] = np.arange(start, start + window_size, dtype=np.int64)
            else:
                frames[:T] = np.arange(T, dtype=np.int64)
            real = frames >= 0
            if np.any(real) and np.any(valid_mask[person_idx, frames[real]]):
                specs.append((person_idx, frames))
    return specs


def _iter_batches(values: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


@torch.no_grad()
def _predict_temporal(
    *,
    runtime: InferenceModel,
    inputs: np.ndarray,
    valid_mask: np.ndarray,
    root_init_norm: np.ndarray,
) -> DensePrediction:
    if runtime.window_size is None or runtime.stride is None:
        raise RuntimeError("Temporal runtime is missing window_size/stride")
    N, T, D = inputs.shape
    W = int(runtime.window_size)
    if T < W and runtime.pad_mode == "none":
        raise ValueError(f"Sequence T={T} is shorter than window_size={W} with pad_mode='none'")

    masked_inputs = inputs.copy()
    masked_inputs[~valid_mask] = 0.0
    root_init_safe = np.nan_to_num(root_init_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    root_init_safe[~valid_mask] = 0.0

    prediction_sums = np.zeros((N, T, 3), dtype=np.float64)
    output_sums = np.zeros((N, T, 3), dtype=np.float64)
    coverage = np.zeros((N, T), dtype=np.int32)
    specs = _window_specs(valid_mask, W, int(runtime.stride))

    for specs_batch in _iter_batches(specs, runtime.batch_size):
        B = len(specs_batch)
        x_batch = np.zeros((B, W, D), dtype=np.float32)
        valid_batch = np.zeros((B, W), dtype=bool)
        root_init_batch = np.zeros((B, W, 3), dtype=np.float32)

        for batch_idx, (person_idx, frames) in enumerate(specs_batch):
            real = frames >= 0
            f = frames[real]
            x_batch[batch_idx, real] = masked_inputs[person_idx, f]
            valid_batch[batch_idx, real] = valid_mask[person_idx, f]
            root_init_batch[batch_idx, real] = root_init_safe[person_idx, f]

        x_t = torch.from_numpy(x_batch).to(runtime.device, dtype=torch.float32)
        valid_t = torch.from_numpy(valid_batch).to(runtime.device)
        model_output = forward_temporal_root_model(runtime.model, x_t, valid_mask=valid_t).to(dtype=torch.float32)
        batch: Dict[str, Any] = {}
        if runtime.config.prediction_mode == "delta":
            batch["root_init_norm"] = torch.from_numpy(root_init_batch).to(
                runtime.device,
                dtype=torch.float32,
            )
        prediction = apply_prediction_mode(
            model_output,
            batch,
            prediction_mode=runtime.config.prediction_mode,
        )
        output_np = model_output.detach().cpu().numpy().astype(np.float32)
        prediction_np = prediction.detach().cpu().numpy().astype(np.float32)

        for batch_idx, (person_idx, frames) in enumerate(specs_batch):
            real_positions = np.flatnonzero(frames >= 0)
            f = frames[real_positions]
            keep = valid_mask[person_idx, f]
            keep &= np.isfinite(output_np[batch_idx, real_positions]).all(axis=-1)
            keep &= np.isfinite(prediction_np[batch_idx, real_positions]).all(axis=-1)
            if not np.any(keep):
                continue
            positions = real_positions[keep]
            frame_keep = f[keep]
            prediction_sums[person_idx, frame_keep] += prediction_np[batch_idx, positions]
            output_sums[person_idx, frame_keep] += output_np[batch_idx, positions]
            coverage[person_idx, frame_keep] += 1

    root_pred = np.full((N, T, 3), np.nan, dtype=np.float32)
    model_output_full = np.full((N, T, 3), np.nan, dtype=np.float32)
    usable = valid_mask & (coverage > 0)
    denominator = coverage[usable][:, None]
    root_pred[usable] = (prediction_sums[usable] / denominator).astype(np.float32)
    model_output_full[usable] = (output_sums[usable] / denominator).astype(np.float32)
    return DensePrediction(root_pred, model_output_full, usable, coverage)


def predict_prepared_sequence(prepared: PreparedSequence, runtime: InferenceModel) -> DensePrediction:
    inputs = build_input_matrix(prepared.normalized, runtime.config.input_config)
    if inputs.shape[-1] != runtime.input_dim:
        raise ValueError(f"Prepared input D={inputs.shape[-1]} does not match model D={runtime.input_dim}")
    valid = effective_valid_mask(prepared, runtime)
    root_init = np.asarray(prepared.normalized["root_init_norm"], dtype=np.float32)
    if runtime.temporal:
        return _predict_temporal(
            runtime=runtime,
            inputs=inputs,
            valid_mask=valid,
            root_init_norm=root_init,
        )
    return _predict_mlp(
        runtime=runtime,
        inputs=inputs,
        valid_mask=valid,
        root_init_norm=root_init,
    )


def _cam_points_to_world(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    if points.ndim == 3:
        return np.einsum("ntc,tcw->ntw", points - t[None, :, :], R)
    if points.ndim == 4:
        return np.einsum("ntjc,tcw->ntjw", points - t[None, :, None, :], R)
    raise ValueError(f"Expected camera points with shape (N,T,3) or (N,T,J,3), got {points.shape}")


def _proxy_reprojection_stats(
    projected: np.ndarray,
    observed: np.ndarray,
    valid_joints: np.ndarray,
    valid_frames: np.ndarray,
) -> Dict[str, Optional[float]]:
    mask = np.asarray(valid_joints, dtype=bool) & np.asarray(valid_frames, dtype=bool)[..., None]
    mask &= np.isfinite(projected).all(axis=-1) & np.isfinite(observed).all(axis=-1)
    if not np.any(mask):
        return {"mean_px": None, "median_px": None, "p95_px": None, "count": 0}
    errors = np.linalg.norm(projected - observed, axis=-1)[mask]
    return {
        "mean_px": float(np.mean(errors)),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95.0)),
        "count": int(errors.size),
    }


def _finite_vector_stats(values: np.ndarray, mask: np.ndarray) -> Dict[str, Optional[float]]:
    usable = np.asarray(mask, dtype=bool) & np.isfinite(values).all(axis=-1)
    if not np.any(usable):
        return {"mean_m": None, "median_m": None, "p95_m": None}
    norms = np.linalg.norm(values[usable], axis=-1)
    return {
        "mean_m": float(np.mean(norms)),
        "median_m": float(np.median(norms)),
        "p95_m": float(np.percentile(norms, 95.0)),
    }


def _write_csv(
    path: Path,
    *,
    sequence: str,
    frame_numbers: np.ndarray,
    source_frame_positions: np.ndarray,
    timestamps_s: np.ndarray,
    valid_mask: np.ndarray,
    coverage: np.ndarray,
    root_pred_norm: np.ndarray,
    root_pred_cam: np.ndarray,
    root_pred_world: np.ndarray,
    root_init_cam: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence",
                "person_idx",
                "frame_idx",
                "source_frame_number",
                "source_frame_position",
                "timestamp_s",
                "overlap_count",
                "root_pred_norm_x",
                "root_pred_norm_y",
                "root_pred_norm_z",
                "root_pred_cam_x_m",
                "root_pred_cam_y_m",
                "root_pred_cam_z_m",
                "root_pred_world_x_m",
                "root_pred_world_y_m",
                "root_pred_world_z_m",
                "root_init_cam_x_m",
                "root_init_cam_y_m",
                "root_init_cam_z_m",
            ]
        )
        persons, frames = np.nonzero(valid_mask)
        for person_idx, frame_idx in zip(persons.tolist(), frames.tolist()):
            writer.writerow(
                [
                    sequence,
                    person_idx,
                    frame_idx,
                    int(frame_numbers[frame_idx]),
                    float(source_frame_positions[frame_idx]),
                    float(timestamps_s[frame_idx]),
                    int(coverage[person_idx, frame_idx]),
                    *root_pred_norm[person_idx, frame_idx].tolist(),
                    *root_pred_cam[person_idx, frame_idx].tolist(),
                    *root_pred_world[person_idx, frame_idx].tolist(),
                    *root_init_cam[person_idx, frame_idx].tolist(),
                ]
            )


def save_inference_result(
    *,
    prepared: PreparedSequence,
    prediction: DensePrediction,
    runtime: InferenceModel,
    output_root: Path | str,
    save_csv: bool = True,
) -> InferenceResult:
    payload = prepared.normalized
    stats = NormalizationStats.load(runtime.config.normalization_stats_path)
    valid = np.asarray(prediction.valid_mask, dtype=bool)
    root_pred_norm = np.asarray(prediction.root_pred_norm, dtype=np.float32)
    model_output_norm = np.asarray(prediction.model_output_norm, dtype=np.float32)

    root_pred_cam = (
        root_pred_norm * stats.std_root[None, None, :] + stats.mean_root[None, None, :]
    ).astype(np.float32)
    root_pred_cam[~valid] = np.nan

    R = np.asarray(payload["R"], dtype=np.float64)
    t = np.asarray(payload["t"], dtype=np.float64)
    K = np.asarray(payload["K"], dtype=np.float64)
    k = np.asarray(payload["k"], dtype=np.float64)
    root_pred_world = _cam_points_to_world(root_pred_cam.astype(np.float64), R, t).astype(np.float32)
    root_pred_world[~valid] = np.nan
    R_source = np.asarray(payload["R_source"], dtype=np.float64)
    t_source = np.asarray(payload["t_source"], dtype=np.float64)
    root_pred_world_source = _cam_points_to_world(
        root_pred_cam.astype(np.float64), R_source, t_source
    ).astype(np.float32)
    root_pred_world_source[~valid] = np.nan

    sam_rel_m = np.asarray(payload["skel_3d_sam3dbody_from_bbox_gt_rel_m"], dtype=np.float32)
    joints_pred_cam = sam_rel_m + root_pred_cam[..., None, :]
    joints_pred_cam[~valid] = np.nan
    joints_pred_world = _cam_points_to_world(joints_pred_cam.astype(np.float64), R, t).astype(np.float32)
    joints_pred_world[~valid] = np.nan
    joints_pred_world_source = _cam_points_to_world(
        joints_pred_cam.astype(np.float64), R_source, t_source
    ).astype(np.float32)
    joints_pred_world_source[~valid] = np.nan
    joints_pred_2d, projected_valid = FeatureCreator.project_camera_to_image(joints_pred_cam, K, k)
    joints_pred_2d[~valid] = np.nan
    projected_valid &= valid[..., None]

    root_init_norm = np.asarray(payload["root_init_norm"], dtype=np.float32)
    root_init_cam = np.asarray(payload["root_init_cam"], dtype=np.float32)
    root_init_world = _cam_points_to_world(root_init_cam.astype(np.float64), R, t).astype(np.float32)
    root_init_world_source = _cam_points_to_world(
        root_init_cam.astype(np.float64), R_source, t_source
    ).astype(np.float32)
    root_init_valid = np.asarray(payload["root_init_valid"], dtype=bool)
    root_init_world[~root_init_valid] = np.nan
    root_init_world_source[~root_init_valid] = np.nan

    output_dir = Path(output_root) / runtime.config.run_name / prepared.sequence
    ensure_dir(output_dir)
    predictions_path = output_dir / "predictions.npz"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "root_predictions.csv" if save_csv else None

    effective_valid = effective_valid_mask(prepared, runtime)
    camera_center_used = bool(
        runtime.config.input_config.use_cam_feat
        and str(runtime.config.input_config.cam_feat_type).startswith("boosted")
    )
    camera_center_zscore = np.asarray(payload["cam_feat_boosted_clean"], dtype=np.float32)[:, 6:9]
    camera_ood_max = float(np.nanmax(np.abs(camera_center_zscore)))
    camera_ood_warning = bool(camera_center_used and camera_ood_max > 5.0)
    proxy_reprojection = _proxy_reprojection_stats(
        joints_pred_2d,
        np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt_pixels"], dtype=np.float32),
        np.asarray(payload["valid_joints"], dtype=bool) & projected_valid,
        valid,
    )
    correction_cam = root_pred_cam - root_init_cam
    correction_stats = (
        _finite_vector_stats(correction_cam, valid)
        if runtime.config.prediction_mode == "delta"
        else {"mean_m": None, "median_m": None, "p95_m": None}
    )
    preprocessing_meta = json.loads(str(np.asarray(prepared.raw["meta_json"]).item()))

    summary: Dict[str, Any] = {
        "sequence": prepared.sequence,
        "run_name": runtime.config.run_name,
        "model_type": runtime.model_type,
        "prediction_mode": runtime.config.prediction_mode,
        "config": str(runtime.config_path.resolve()),
        "checkpoint": str(runtime.checkpoint_path.resolve()),
        "checkpoint_metadata": runtime.checkpoint_metadata,
        "device": str(runtime.device),
        "input_dim": int(runtime.input_dim),
        "num_persons": int(valid.shape[0]),
        "num_frames": int(valid.shape[1]),
        "num_source_frames": int(np.asarray(payload["source_num_frames"]).item()),
        "resampling": preprocessing_meta["resampling"],
        "image_size_width_height": np.asarray(payload["image_size"], dtype=int).tolist(),
        "sam3d_orientation_sign": int(np.asarray(payload["sam3d_orientation_sign"]).item()),
        "num_base_valid_positions": int(np.asarray(payload["valid_mask"], dtype=bool).sum()),
        "num_root_init_valid_positions": int(root_init_valid.sum()),
        "num_effective_valid_positions": int(effective_valid.sum()),
        "num_predicted_positions": int(valid.sum()),
        "num_uncovered_valid_positions": int((effective_valid & ~valid).sum()),
        "overlap_count_min": int(prediction.coverage_count[valid].min()) if np.any(valid) else 0,
        "overlap_count_max": int(prediction.coverage_count[valid].max()) if np.any(valid) else 0,
        "proxy_reprojection_to_observed_sam2d": proxy_reprojection,
        "delta_correction_camera": correction_stats,
        "distortion_coefficients_all_zero": bool(np.allclose(k[:, :2], 0.0)),
        "camera_center_feature_used": camera_center_used,
        "world_alignment": prepared.world_alignment.to_metadata(),
        "camera_center_max_abs_train_zscore_before_alignment": float(
            np.max(np.abs(prepared.world_alignment.source_zscore))
        ),
        "camera_center_max_abs_train_zscore": camera_ood_max,
        "camera_center_out_of_distribution_warning": camera_ood_warning,
        "metrics_note": "No GT is available. Reprojection is only a proxy against observed SAM2D, not a GT metric.",
    }

    meta = {
        "summary": summary,
        "array_layout": "dense (N,T,...) unless documented otherwise",
        "frame_idx": "zero-based index into inference arrays",
        "frame_numbers": "nearest source number parsed from frames/<sequence> filenames, or index fallback",
        "source_frame_positions": "fractional zero-based position on the original input timeline",
        "timestamps_s": "time from the beginning of the sequence; NaN when source FPS was not provided",
        "camera_extrinsics": "X_cam_col = R @ X_world_col + t",
        "world_frame": "training-aligned; source-world copies are also exported",
        "world_units": "metres",
    }
    output_payload: Dict[str, np.ndarray] = {
        "valid_mask": valid,
        "base_valid_mask": np.asarray(payload["valid_mask"], dtype=bool),
        "root_init_valid": root_init_valid,
        "overlap_count": np.asarray(prediction.coverage_count, dtype=np.int32),
        "frame_idx": np.arange(valid.shape[1], dtype=np.int64),
        "frame_numbers": np.asarray(payload["frame_numbers"], dtype=np.int64),
        "source_num_frames": np.asarray(payload["source_num_frames"], dtype=np.int64),
        "source_frame_indices": np.asarray(payload["source_frame_indices"], dtype=np.int64),
        "source_frame_positions": np.asarray(payload["source_frame_positions"], dtype=np.float64),
        "source_frame_numbers_float": np.asarray(
            payload["source_frame_numbers_float"], dtype=np.float64
        ),
        "timestamps_s": np.asarray(payload["timestamps_s"], dtype=np.float64),
        "source_fps": np.asarray(payload["source_fps"], dtype=np.float64),
        "output_fps": np.asarray(payload["output_fps"], dtype=np.float64),
        "image_size": np.asarray(payload["image_size"], dtype=np.int32),
        "boxes_xyxy": np.asarray(payload["boxes_xyxy"], dtype=np.float32),
        "K": K.astype(np.float32),
        "R": R.astype(np.float32),
        "t": t.astype(np.float32),
        "R_source": R_source.astype(np.float32),
        "t_source": t_source.astype(np.float32),
        "k": k.astype(np.float32),
        "camera_center_world_m": np.asarray(prepared.raw["camera_center_world"], dtype=np.float32),
        "camera_center_source_world_m": np.asarray(
            prepared.raw["camera_center_world_source"], dtype=np.float32
        ),
        "world_alignment_rotation": np.asarray(
            prepared.raw["world_alignment_rotation"], dtype=np.float32
        ),
        "pitch_points_world_m": np.asarray(prepared.raw["pitch_points_world"], dtype=np.float32),
        "ground_intersection_world_m": np.asarray(
            prepared.raw["ground_intersection"], dtype=np.float32
        ),
        "ground_intersection_source_world_m": transform_world_points_aligned_to_source(
            np.asarray(prepared.raw["ground_intersection"], dtype=np.float32),
            prepared.world_alignment,
        ).astype(np.float32),
        "root_pred_norm": root_pred_norm,
        "root_pred_m": root_pred_cam,
        "root_world_pred_m": root_pred_world,
        "root_source_world_pred_m": root_pred_world_source,
        "joints_pred_cam_m": joints_pred_cam.astype(np.float32),
        "joints_pred_world_m": joints_pred_world.astype(np.float32),
        "joints_pred_source_world_m": joints_pred_world_source.astype(np.float32),
        "joints_pred_2d": joints_pred_2d.astype(np.float32),
        "valid_projected_joints": projected_valid.astype(bool),
        "root_init_norm": root_init_norm,
        "root_init_cam_m": root_init_cam,
        "root_init_world_m": root_init_world,
        "root_init_source_world_m": root_init_world_source,
        "meta_json": np.array(json.dumps(meta), dtype=object),
    }
    if runtime.config.prediction_mode == "delta":
        output_payload["root_delta_pred_norm"] = model_output_norm
        output_payload["root_delta_pred_cam_m"] = (
            model_output_norm * stats.std_root[None, None, :]
        ).astype(np.float32)
    else:
        output_payload["model_output_norm"] = model_output_norm

    tmp_predictions = predictions_path.with_name("predictions.tmp.npz")
    np.savez_compressed(tmp_predictions, **output_payload)
    tmp_predictions.replace(predictions_path)
    write_json(summary_path, summary)

    if csv_path is not None:
        _write_csv(
            csv_path,
            sequence=prepared.sequence,
            frame_numbers=np.asarray(payload["frame_numbers"], dtype=np.int64),
            source_frame_positions=np.asarray(payload["source_frame_positions"], dtype=np.float64),
            timestamps_s=np.asarray(payload["timestamps_s"], dtype=np.float64),
            valid_mask=valid,
            coverage=np.asarray(prediction.coverage_count, dtype=np.int32),
            root_pred_norm=root_pred_norm,
            root_pred_cam=root_pred_cam,
            root_pred_world=root_pred_world,
            root_init_cam=root_init_cam,
        )

    return InferenceResult(
        sequence=prepared.sequence,
        predictions_path=predictions_path,
        csv_path=csv_path,
        summary_path=summary_path,
        summary=summary,
    )
