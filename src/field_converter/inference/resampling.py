"""Temporal resampling utilities for qualitative inference inputs.

Upsampling uses local, shape-preserving cubic interpolation for continuous
values and spherical interpolation (SLERP) for camera rotations. Missing
observations are never bridged: an inserted value is finite only when both
source frames surrounding it are finite. Downsampling deliberately selects
the nearest source sample instead of smoothing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ResamplingMode = Literal["identity", "interpolate", "sample"]


@dataclass(frozen=True)
class ResamplingPlan:
    """Mapping from a regularly sampled source timeline to a target one."""

    source_fps: float
    target_fps: float
    source_num_frames: int
    target_num_frames: int
    mode: ResamplingMode
    target_times_s: np.ndarray
    source_positions: np.ndarray
    nearest_source_indices: np.ndarray
    left_source_indices: np.ndarray
    right_source_indices: np.ndarray
    interpolation_alpha: np.ndarray

    @property
    def applied(self) -> bool:
        return self.mode != "identity"

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": True,
            "applied": self.applied,
            "mode": self.mode,
            "source_fps": self.source_fps,
            "target_fps": self.target_fps,
            "source_num_frames": self.source_num_frames,
            "target_num_frames": self.target_num_frames,
            "source_duration_s": (
                (self.source_num_frames - 1) / self.source_fps
                if self.source_num_frames > 1
                else 0.0
            ),
            "continuous_interpolation": (
                "local shape-preserving cubic (linear fallback at boundaries/missing neighbours)"
                if self.mode == "interpolate"
                else None
            ),
            "rotation_interpolation": "quaternion SLERP" if self.mode == "interpolate" else None,
            "downsampling": "nearest source sample" if self.mode == "sample" else None,
            "missing_data_policy": "do not interpolate across a missing source endpoint",
        }


def build_resampling_plan(
    source_num_frames: int,
    *,
    source_fps: float,
    target_fps: float = 50.0,
) -> ResamplingPlan:
    """Create an endpoint-safe regular timeline.

    The target timeline starts at zero and never extends beyond the last source
    timestamp. For example, 1000 samples at 25 FPS span timestamps 0..39.96 s
    and therefore produce 1999 samples at 50 FPS, not an extrapolated 2000th
    sample at 39.98 s.
    """
    source_num_frames = int(source_num_frames)
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    if source_num_frames <= 0:
        raise ValueError(f"source_num_frames must be > 0, got {source_num_frames}")
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"source_fps must be a finite value > 0, got {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError(f"target_fps must be a finite value > 0, got {target_fps}")

    if np.isclose(source_fps, target_fps, rtol=1e-9, atol=1e-9):
        mode: ResamplingMode = "identity"
        target_num_frames = source_num_frames
    elif source_fps < target_fps:
        mode = "interpolate"
        duration_s = (source_num_frames - 1) / source_fps
        target_num_frames = int(np.floor(duration_s * target_fps + 1e-9)) + 1
    else:
        mode = "sample"
        duration_s = (source_num_frames - 1) / source_fps
        target_num_frames = int(np.floor(duration_s * target_fps + 1e-9)) + 1

    target_times_s = np.arange(target_num_frames, dtype=np.float64) / target_fps
    source_positions = target_times_s * source_fps
    source_positions = np.clip(source_positions, 0.0, float(source_num_frames - 1))

    # Avoid turning mathematically integral positions into 1.9999999998 because
    # of floating-point arithmetic.
    rounded = np.rint(source_positions)
    source_positions = np.where(
        np.abs(source_positions - rounded) <= 1e-10,
        rounded,
        source_positions,
    )
    left = np.floor(source_positions).astype(np.int64)
    right = np.minimum(left + 1, source_num_frames - 1)
    alpha = source_positions - left.astype(np.float64)
    nearest = np.minimum(
        np.floor(source_positions + 0.5).astype(np.int64),
        source_num_frames - 1,
    )

    return ResamplingPlan(
        source_fps=source_fps,
        target_fps=target_fps,
        source_num_frames=source_num_frames,
        target_num_frames=target_num_frames,
        mode=mode,
        target_times_s=target_times_s,
        source_positions=source_positions,
        nearest_source_indices=nearest,
        left_source_indices=left,
        right_source_indices=right,
        interpolation_alpha=alpha,
    )


def _pchip_tangent(left_slope: np.ndarray, right_slope: np.ndarray) -> np.ndarray:
    """Uniform-grid PCHIP tangent, evaluated component by component."""
    left = np.asarray(left_slope)
    right = np.asarray(right_slope)
    tangent = np.zeros_like(left)
    same_direction = (left * right) > 0.0
    denominator = left + right
    usable = same_direction & np.isfinite(denominator) & (np.abs(denominator) > 1e-15)
    np.divide(2.0 * left * right, denominator, out=tangent, where=usable)
    return tangent


def resample_continuous(
    values: np.ndarray,
    plan: ResamplingPlan,
    *,
    time_axis: int = 0,
) -> np.ndarray:
    """Resample floating-point values along ``time_axis``.

    Interpolation is local to one source interval. Cubic tangents use adjacent
    samples when available, falling back to the interval's linear slope. This
    preserves exact source samples and avoids filling long detection gaps.
    """
    value = np.asarray(values)
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"continuous resampling requires floating values, got {value.dtype}")
    axis = int(time_axis)
    if axis < 0:
        axis += value.ndim
    if axis < 0 or axis >= value.ndim:
        raise ValueError(f"time_axis={time_axis} is invalid for an array with {value.ndim} dimensions")
    moved = np.moveaxis(value, axis, 0)
    if moved.shape[0] != plan.source_num_frames:
        raise ValueError(
            f"time dimension has {moved.shape[0]} frames, expected {plan.source_num_frames}"
        )

    if plan.mode == "identity":
        return value.copy()
    if plan.mode == "sample":
        sampled = np.take(moved, plan.nearest_source_indices, axis=0)
        return np.moveaxis(sampled, 0, axis)

    out = np.full(
        (plan.target_num_frames,) + moved.shape[1:],
        np.nan,
        dtype=value.dtype,
    )
    for target_idx, (left_idx, right_idx, alpha) in enumerate(
        zip(
            plan.left_source_indices.tolist(),
            plan.right_source_indices.tolist(),
            plan.interpolation_alpha.tolist(),
        )
    ):
        if left_idx == right_idx or alpha <= 1e-12:
            out[target_idx] = moved[left_idx]
            continue

        y0 = moved[left_idx]
        y1 = moved[right_idx]
        endpoints_finite = np.isfinite(y0) & np.isfinite(y1)
        slope = y1 - y0
        tangent0 = slope
        tangent1 = slope

        if left_idx > 0:
            previous = moved[left_idx - 1]
            previous_slope = y0 - previous
            previous_finite = np.isfinite(previous) & endpoints_finite
            cubic_tangent0 = _pchip_tangent(previous_slope, slope)
            tangent0 = np.where(previous_finite, cubic_tangent0, tangent0)
        if right_idx + 1 < plan.source_num_frames:
            following = moved[right_idx + 1]
            following_slope = following - y1
            following_finite = np.isfinite(following) & endpoints_finite
            cubic_tangent1 = _pchip_tangent(slope, following_slope)
            tangent1 = np.where(following_finite, cubic_tangent1, tangent1)

        a = float(alpha)
        a2 = a * a
        a3 = a2 * a
        interpolated = (
            (2.0 * a3 - 3.0 * a2 + 1.0) * y0
            + (a3 - 2.0 * a2 + a) * tangent0
            + (-2.0 * a3 + 3.0 * a2) * y1
            + (a3 - a2) * tangent1
        )
        # This is both a guard against numerical overshoot and important for
        # box coordinates, whose independent edges should stay between their
        # two observed positions.
        interpolated = np.clip(interpolated, np.minimum(y0, y1), np.maximum(y0, y1))
        out[target_idx] = np.where(endpoints_finite, interpolated, np.nan)

    return np.moveaxis(out, 0, axis)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert one 3x3 rotation matrix to a normalized [w,x,y,z] quaternion."""
    R = np.asarray(matrix, dtype=np.float64)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        return np.full((4,), np.nan, dtype=np.float64)

    trace = float(np.trace(R))
    if trace > 0.0:
        scale = np.sqrt(max(trace + 1.0, 0.0)) * 2.0
        quaternion = np.array(
            [0.25 * scale, (R[2, 1] - R[1, 2]) / scale, (R[0, 2] - R[2, 0]) / scale,
             (R[1, 0] - R[0, 1]) / scale],
            dtype=np.float64,
        )
    else:
        diagonal_idx = int(np.argmax(np.diag(R)))
        if diagonal_idx == 0:
            scale = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [(R[2, 1] - R[1, 2]) / scale, 0.25 * scale,
                 (R[0, 1] + R[1, 0]) / scale, (R[0, 2] + R[2, 0]) / scale],
                dtype=np.float64,
            )
        elif diagonal_idx == 1:
            scale = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 0.0)) * 2.0
            quaternion = np.array(
                [(R[0, 2] - R[2, 0]) / scale, (R[0, 1] + R[1, 0]) / scale,
                 0.25 * scale, (R[1, 2] + R[2, 1]) / scale],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 0.0)) * 2.0
            quaternion = np.array(
                [(R[1, 0] - R[0, 1]) / scale, (R[0, 2] + R[2, 0]) / scale,
                 (R[1, 2] + R[2, 1]) / scale, 0.25 * scale],
                dtype=np.float64,
            )

    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-15:
        return np.full((4,), np.nan, dtype=np.float64)
    return quaternion / norm


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if q.shape != (4,) or not np.isfinite(norm) or norm <= 1e-15:
        return np.full((3, 3), np.nan, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        return np.full((4,), np.nan, dtype=np.float64)
    first = np.asarray(q0, dtype=np.float64)
    second = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = first + float(alpha) * (second - first)
        return blended / np.linalg.norm(blended)
    angle = float(np.arccos(dot))
    sin_angle = float(np.sin(angle))
    return (
        np.sin((1.0 - float(alpha)) * angle) / sin_angle * first
        + np.sin(float(alpha) * angle) / sin_angle * second
    )


def resample_rotations(rotations: np.ndarray, plan: ResamplingPlan) -> np.ndarray:
    """Resample ``(T,3,3)`` camera rotations with quaternion SLERP."""
    value = np.asarray(rotations)
    if value.shape != (plan.source_num_frames, 3, 3):
        raise ValueError(
            f"rotations must have shape {(plan.source_num_frames, 3, 3)}, got {value.shape}"
        )
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f"rotation resampling requires floating values, got {value.dtype}")
    if plan.mode == "identity":
        return value.copy()
    if plan.mode == "sample":
        return value[plan.nearest_source_indices].copy()

    quaternions = np.stack([_matrix_to_quaternion(matrix) for matrix in value], axis=0)
    # Keep a continuous quaternion sign along valid runs. q and -q encode the
    # same rotation, but inconsistent signs would take the long interpolation arc.
    for idx in range(1, quaternions.shape[0]):
        if np.isfinite(quaternions[idx - 1]).all() and np.isfinite(quaternions[idx]).all():
            if float(np.dot(quaternions[idx - 1], quaternions[idx])) < 0.0:
                quaternions[idx] *= -1.0

    out = np.full((plan.target_num_frames, 3, 3), np.nan, dtype=value.dtype)
    for target_idx, (left_idx, right_idx, alpha) in enumerate(
        zip(
            plan.left_source_indices.tolist(),
            plan.right_source_indices.tolist(),
            plan.interpolation_alpha.tolist(),
        )
    ):
        if left_idx == right_idx or alpha <= 1e-12:
            out[target_idx] = value[left_idx]
            continue
        quaternion = _slerp(quaternions[left_idx], quaternions[right_idx], float(alpha))
        out[target_idx] = _quaternion_to_matrix(quaternion).astype(value.dtype, copy=False)
    return out
