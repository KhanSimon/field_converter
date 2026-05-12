#!/usr/bin/env python3
"""
check_rotation_conventions.py
=============================

Script de diagnostic pour vérifier les conventions caméra / rotation / projection.

Il charge les données depuis::

    from field_converter import pathseeker as ps
    DATA_DIR = ps.DATA_DIR

Par défaut, il attend::

    DATA_DIR/
      sequences_gt.txt
      cameras_gt/{sequence}.npz       # K, R, t, k
      joints_3d_gt/{sequence}.npy|npz # joints GT monde, 25 joints
      boxes_gt/{sequence}.npy|npz     # boxes xyxy, optionnel mais recommandé

Pour chaque séquence, il affiche pour chaque test :

    - ce qui est attendu
    - ce qui est observé
    - PASS / WARN / FAIL

Convention principale testée
----------------------------

Convention math vecteur colonne::

    X_cam = R @ X_world + t

Implémentation numpy avec points en lignes (..., 3)::

    X_cam = X_world @ R.T + t

Inverse::

    X_world = R.T @ (X_cam - t)

Implémentation numpy row-vector::

    X_world = (X_cam - t) @ R

Usage
-----

Depuis ton repo::

    python check_rotation_conventions.py

Exemples::

    python check_rotation_conventions.py --sequence ARG_CRO_225412
    python check_rotation_conventions.py --image-size 1920 1080
    python check_rotation_conventions.py --max-sequences 5 --num-frames 10
    python check_rotation_conventions.py --save-overlays --image-root images

Notes
-----

- Le test le plus discriminant est : GT 3D monde -> image, puis comparaison avec les boxes.
- Les tests purement algébriques, comme R.T @ R ≈ I, vérifient la validité de R,
  mais ne prouvent pas à eux seuls que la convention monde->caméra est la bonne.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


from field_converter import pathseeker as ps



# -----------------------------------------------------------------------------
# Configuration / reporting
# -----------------------------------------------------------------------------


@dataclass
class Thresholds:
    """Seuils utilisés pour PASS / WARN / FAIL."""

    rotation_ortho_pass: float = 1e-4
    rotation_ortho_warn: float = 1e-2
    rotation_det_pass_abs_err: float = 1e-3
    rotation_det_warn_abs_err: float = 5e-2
    camera_center_pass: float = 1e-4
    camera_center_warn: float = 1e-2
    inverse_pass: float = 1e-4
    inverse_warn: float = 1e-2
    z_positive_pass: float = 0.98
    z_positive_warn: float = 0.90
    bbox_inside_pass: float = 0.85
    bbox_inside_warn: float = 0.60
    image_inside_pass: float = 0.85
    image_inside_warn: float = 0.60

    # SAM3D-vs-GT diagnostics. These values are intentionally loose because
    # SAM3D predictions are model outputs, not deterministic GT projections.
    sam2d_median_px_pass: float = 25.0
    sam2d_median_px_warn: float = 80.0
    sam_axis_mpjpe_pass: float = 0.25
    sam_axis_mpjpe_warn: float = 0.60
    sam_procrustes_gain_warn: float = 0.35
    sam_reproj_median_px_pass: float = 35.0
    sam_reproj_median_px_warn: float = 100.0
    bone_ratio_median_pass_abs_log: float = 0.25
    bone_ratio_median_warn_abs_log: float = 0.60


@dataclass
class TestResult:
    name: str
    expected: str
    observed: str
    status: str
    value: Optional[float] = None


def status_from_value(
    value: float,
    pass_cond: bool,
    warn_cond: bool,
) -> str:
    if pass_cond:
        return "PASS"
    if warn_cond:
        return "WARN"
    return "FAIL"


def print_result(result: TestResult, indent: str = "  ") -> None:
    print(f"{indent}[{result.status}] {result.name}")
    print(f"{indent}  attendu : {result.expected}")
    print(f"{indent}  observé : {result.observed}")


# -----------------------------------------------------------------------------
# Loading utilities
# -----------------------------------------------------------------------------


def read_sequences(data_dir: Path, sequences_file: str) -> list[str]:
    path = data_dir / sequences_file
    if not path.exists():
        raise FileNotFoundError(f"Fichier de séquences introuvable : {path}")

    sequences: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sequences.append(line)
    return sequences


def load_np_array(path_without_ext: Path, possible_keys: Iterable[str]) -> np.ndarray:
    """Load .npy or .npz using several possible keys."""
    npy = path_without_ext.with_suffix(".npy")
    npz = path_without_ext.with_suffix(".npz")

    if npy.exists():
        return np.asarray(np.load(npy))

    if npz.exists():
        data = np.load(npz)
        for key in possible_keys:
            if key in data:
                return np.asarray(data[key])
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]])
        raise KeyError(
            f"Aucune clé parmi {tuple(possible_keys)} trouvée dans {npz}. "
            f"Clés disponibles : {data.files}"
        )

    raise FileNotFoundError(f"Aucun fichier .npy/.npz trouvé pour : {path_without_ext}")


def load_camera(camera_path: Path) -> Dict[str, np.ndarray]:
    if not camera_path.exists():
        raise FileNotFoundError(f"Fichier caméra introuvable : {camera_path}")
    data = dict(np.load(camera_path))
    required = ("K", "R", "t", "k")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{camera_path} ne contient pas les clés requises : {missing}")
    return {key: np.asarray(data[key]) for key in required}


def ensure_camera_shapes(camera: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    K = np.asarray(camera["K"], dtype=np.float64)
    R = np.asarray(camera["R"], dtype=np.float64)
    t = np.asarray(camera["t"], dtype=np.float64)
    k = np.asarray(camera["k"], dtype=np.float64)

    if K.ndim == 2:
        K = K[None]
    if R.ndim == 2:
        R = R[None]
    if t.ndim == 1:
        t = t[None]
    if k.ndim == 1:
        k = k[None]

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K doit être (T,3,3), reçu {K.shape}")
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R doit être (T,3,3), reçu {R.shape}")
    if t.shape[-1] != 3:
        raise ValueError(f"t doit être (T,3), reçu {t.shape}")
    if k.shape[-1] < 2:
        raise ValueError(f"k doit contenir au moins k1,k2, reçu {k.shape}")

    # Keep only k1,k2.
    k = k[..., :2]

    T = K.shape[0]
    for name, arr in (("R", R), ("t", t), ("k", k)):
        if arr.shape[0] not in (1, T):
            raise ValueError(f"{name} a T={arr.shape[0]} mais K a T={T}")

    # Broadcast frame-constant camera arrays if needed.
    if R.shape[0] == 1 and T > 1:
        R = np.repeat(R, T, axis=0)
    if t.shape[0] == 1 and T > 1:
        t = np.repeat(t, T, axis=0)
    if k.shape[0] == 1 and T > 1:
        k = np.repeat(k, T, axis=0)

    return {"K": K, "R": R, "t": t, "k": k}


def ensure_joints_shape(joints: np.ndarray, T: int) -> np.ndarray:
    """
    Return joints as (N, T, J, 3).

    Accepted:
      - (N,T,J,3)
      - (T,N,J,3)
      - (T,J,3), one player
    """
    joints = np.asarray(joints, dtype=np.float64)

    if joints.ndim == 4 and joints.shape[-1] == 3:
        if joints.shape[1] == T:
            return joints
        if joints.shape[0] == T:
            return joints.transpose(1, 0, 2, 3)

    if joints.ndim == 3 and joints.shape[-1] == 3 and joints.shape[0] == T:
        return joints[None]

    raise ValueError(
        f"Joints attendus en (N,T,J,3), (T,N,J,3), ou (T,J,3). Reçu {joints.shape}, T={T}."
    )


def ensure_boxes_shape(boxes: np.ndarray, T: int, N: int) -> np.ndarray:
    """
    Return boxes as (N, T, 4).

    Accepted:
      - (N,T,4)
      - (T,N,4)
      - (T,4), one player
    """
    boxes = np.asarray(boxes, dtype=np.float64)

    if boxes.ndim == 3 and boxes.shape[-1] == 4:
        if boxes.shape[0] == N and boxes.shape[1] == T:
            return boxes
        if boxes.shape[0] == T and boxes.shape[1] == N:
            return boxes.transpose(1, 0, 2)

    if boxes.ndim == 2 and boxes.shape == (T, 4) and N == 1:
        return boxes[None]

    raise ValueError(
        f"Boxes attendues en (N,T,4), (T,N,4), ou (T,4) si N=1. "
        f"Reçu {boxes.shape}, attendu N={N}, T={T}."
    )


def ensure_sam3d_shape(arr: np.ndarray, T: int, N: int, target_joints: Optional[int] = None) -> np.ndarray:
    """
    Return SAM 3D joints as (N, T, J, 3).

    Accepted:
      - (N,T,J,3)
      - (T,N,J,3)
      - (T,J,3), one player

    If target_joints is provided and arr has more joints, the first target_joints
    are kept. Prefer explicit mapping upstream for serious experiments.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        if arr.shape[0] == N and arr.shape[1] == T:
            out = arr
        elif arr.shape[0] == T and arr.shape[1] == N:
            out = arr.transpose(1, 0, 2, 3)
        else:
            raise ValueError(f"SAM3D shape incompatible avec N={N}, T={T}: {arr.shape}")
    elif arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] == T and N == 1:
        out = arr[None]
    else:
        raise ValueError(f"SAM3D attendu en (N,T,J,3), (T,N,J,3), ou (T,J,3). Reçu {arr.shape}")

    if target_joints is not None:
        if out.shape[2] < target_joints:
            raise ValueError(f"SAM3D a seulement J={out.shape[2]} joints, impossible de garder {target_joints} joints")
        out = out[:, :, :target_joints]
    return out


def ensure_sam2d_shape(arr: np.ndarray, T: int, N: int, target_joints: Optional[int] = None) -> np.ndarray:
    """
    Return SAM 2D joints as (N, T, J, 2).

    Accepted:
      - (N,T,J,2)
      - (T,N,J,2)
      - (T,J,2), one player
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[-1] == 2:
        if arr.shape[0] == N and arr.shape[1] == T:
            out = arr
        elif arr.shape[0] == T and arr.shape[1] == N:
            out = arr.transpose(1, 0, 2, 3)
        else:
            raise ValueError(f"SAM2D shape incompatible avec N={N}, T={T}: {arr.shape}")
    elif arr.ndim == 3 and arr.shape[-1] == 2 and arr.shape[0] == T and N == 1:
        out = arr[None]
    else:
        raise ValueError(f"SAM2D attendu en (N,T,J,2), (T,N,J,2), ou (T,J,2). Reçu {arr.shape}")

    if target_joints is not None:
        if out.shape[2] < target_joints:
            raise ValueError(f"SAM2D a seulement J={out.shape[2]} joints, impossible de garder {target_joints} joints")
        out = out[:, :, :target_joints]
    return out


def infer_image_size(K: np.ndarray, image_size_arg: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if image_size_arg is not None:
        return image_size_arg
    cx = float(np.nanmedian(K[:, 0, 2]))
    cy = float(np.nanmedian(K[:, 1, 2]))
    W = int(round(2.0 * cx))
    H = int(round(2.0 * cy))
    if W <= 0 or H <= 0:
        raise ValueError(
            "Impossible d'inférer la taille image depuis K. Utilise --image-size W H."
        )
    return W, H


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


def world_to_camera_A(X_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convention A, attendue : X_cam_col = R @ X_world_col + t.
    Row-vector numpy: X_cam = X_world @ R.T + t.

    X_world: (N,T,J,3), R: (T,3,3), t: (T,3)
    """
    return np.einsum("ntjw,tcw->ntjc", X_world, R) + t[None, :, None, :]


def camera_to_world_A(X_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Inverse of convention A.

    Column: X_world = R.T @ (X_cam - t)
    Row-vector: X_world = (X_cam - t) @ R
    """
    return np.einsum("ntjc,tcw->ntjw", X_cam - t[None, :, None, :], R)


def world_to_camera_B_row_R(X_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Alternative fréquente, souvent fausse si R est world->cam:
    X_cam = X_world @ R + t.
    """
    return np.einsum("ntjw,twc->ntjc", X_world, R) + t[None, :, None, :]


def world_to_camera_C_t_as_camera_center(X_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Alternative si t était en fait le centre caméra C dans le monde:
    X_cam = (X_world - C) @ R.T.

    Ce n'est pas la convention attendue ici, mais on la teste pour diagnostic.
    """
    return np.einsum("ntjw,tcw->ntjc", X_world - t[None, :, None, :], R)


def project_camera_to_image(
    X_cam: np.ndarray,
    K: np.ndarray,
    k: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project camera-space points to pixels using k1,k2 radial distortion.

    X_cam: (N,T,J,3)
    K: (T,3,3)
    k: (T,2)
    """
    N, T, J, _ = X_cam.shape
    X_img = np.full((N, T, J, 2), np.nan, dtype=np.float64)

    finite = np.isfinite(X_cam).all(axis=-1)
    Z = X_cam[..., 2]
    valid = finite & (Z > eps)

    if not np.any(valid):
        return X_img, valid

    x = np.full((N, T, J), np.nan, dtype=np.float64)
    y = np.full((N, T, J), np.nan, dtype=np.float64)
    x[valid] = X_cam[..., 0][valid] / Z[valid]
    y[valid] = X_cam[..., 1][valid] / Z[valid]

    k1 = k[:, 0][None, :, None]
    k2 = k[:, 1][None, :, None]
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    x_d = x * factor
    y_d = y * factor

    fx = K[:, 0, 0][None, :, None]
    fy = K[:, 1, 1][None, :, None]
    cx = K[:, 0, 2][None, :, None]
    cy = K[:, 1, 2][None, :, None]

    X_img[..., 0] = fx * x_d + cx
    X_img[..., 1] = fy * y_d + cy
    X_img[~valid] = np.nan
    return X_img, valid


def camera_centers_from_extrinsics(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Camera center in world coordinates.

    Convention:
        X_cam_col = R @ X_world_col + t

    Then:
        C_world = -R.T @ t
    """
    return -np.einsum("tcw,tc->tw", R, t)


def camera_forward_world(R: np.ndarray) -> np.ndarray:
    """
    Camera forward direction in world coordinates.

    If camera +Z is forward:
        forward_world = R.T @ [0, 0, 1]
    """
    z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = np.einsum("tcw,c->tw", R, z_cam)
    forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-12)
    return forward


# -----------------------------------------------------------------------------
# Metrics / tests
# -----------------------------------------------------------------------------


def nanmean_norm(x: np.ndarray, axis: int = -1) -> float:
    return float(np.nanmean(np.linalg.norm(x, axis=axis)))


def test_shapes(sequence: str, K: np.ndarray, R: np.ndarray, t: np.ndarray, k: np.ndarray, X_world: np.ndarray, boxes: Optional[np.ndarray]) -> list[TestResult]:
    N, T, J, _ = X_world.shape
    results: list[TestResult] = []

    expected = "K=(T,3,3), R=(T,3,3), t=(T,3), k=(T,2), joints=(N,T,25,3)"
    observed = f"K={K.shape}, R={R.shape}, t={t.shape}, k={k.shape}, joints={X_world.shape}"
    status = "PASS" if (K.shape == (T, 3, 3) and R.shape == (T, 3, 3) and t.shape == (T, 3) and k.shape == (T, 2) and J == 25) else "WARN"
    results.append(TestResult("Shapes caméra / joints", expected, observed, status))

    if boxes is not None:
        expected = "boxes=(N,T,4), même N et T que les joints"
        observed = f"boxes={boxes.shape}, joints N,T=({N},{T})"
        status = "PASS" if boxes.shape == (N, T, 4) else "FAIL"
        results.append(TestResult("Shape boxes", expected, observed, status))

    return results


def test_rotation_matrices(R: np.ndarray, th: Thresholds) -> list[TestResult]:
    I = np.eye(3)
    ortho_err = np.linalg.norm(np.einsum("tji,tjk->tik", R, R) - I[None], axis=(1, 2))
    det = np.linalg.det(R)

    max_ortho = float(np.nanmax(ortho_err))
    mean_ortho = float(np.nanmean(ortho_err))
    max_det_err = float(np.nanmax(np.abs(det - 1.0)))
    mean_det = float(np.nanmean(det))

    status_ortho = status_from_value(
        max_ortho,
        max_ortho <= th.rotation_ortho_pass,
        max_ortho <= th.rotation_ortho_warn,
    )
    status_det = status_from_value(
        max_det_err,
        max_det_err <= th.rotation_det_pass_abs_err,
        max_det_err <= th.rotation_det_warn_abs_err,
    )

    return [
        TestResult(
            "R orthonormale",
            f"max ||R.T@R - I|| <= {th.rotation_ortho_pass:g}",
            f"mean={mean_ortho:.3e}, max={max_ortho:.3e}",
            status_ortho,
            max_ortho,
        ),
        TestResult(
            "det(R)",
            f"det(R) proche de 1, max |det-1| <= {th.rotation_det_pass_abs_err:g}",
            f"mean det={mean_det:.6f}, max |det-1|={max_det_err:.3e}",
            status_det,
            max_det_err,
        ),
    ]


def test_camera_center(R: np.ndarray, t: np.ndarray, th: Thresholds) -> list[TestResult]:
    C = camera_centers_from_extrinsics(R, t)
    residual = np.einsum("tcw,tw->tc", R, C) + t
    residual_norm = np.linalg.norm(residual, axis=-1)
    max_err = float(np.nanmax(residual_norm))
    mean_err = float(np.nanmean(residual_norm))

    status = status_from_value(
        max_err,
        max_err <= th.camera_center_pass,
        max_err <= th.camera_center_warn,
    )

    return [
        TestResult(
            "Centre caméra C = -R.T @ t",
            f"R @ C + t ≈ 0, max erreur <= {th.camera_center_pass:g}",
            f"mean ||R@C+t||={mean_err:.3e}, max={max_err:.3e}; C min={np.nanmin(C, axis=0)}, C max={np.nanmax(C, axis=0)}",
            status,
            max_err,
        )
    ]


def test_inverse_roundtrip(X_world: np.ndarray, R: np.ndarray, t: np.ndarray, th: Thresholds) -> list[TestResult]:
    X_cam = world_to_camera_A(X_world, R, t)
    X_back = camera_to_world_A(X_cam, R, t)
    finite = np.isfinite(X_world).all(axis=-1)
    err = np.linalg.norm(X_back - X_world, axis=-1)
    value = float(np.nanmean(err[finite])) if np.any(finite) else float("nan")

    status = status_from_value(
        value,
        value <= th.inverse_pass,
        value <= th.inverse_warn,
    )

    return [
        TestResult(
            "Round-trip world -> cam -> world",
            f"erreur moyenne <= {th.inverse_pass:g}",
            f"mean L2={value:.3e}",
            status,
            value,
        )
    ]


def test_depth(X_world: np.ndarray, R: np.ndarray, t: np.ndarray, th: Thresholds) -> Tuple[list[TestResult], np.ndarray]:
    X_cam = world_to_camera_A(X_world, R, t)
    finite = np.isfinite(X_cam).all(axis=-1)
    Z = X_cam[..., 2]
    ratio = float(np.mean((Z > 1e-6)[finite])) if np.any(finite) else float("nan")
    z_min = float(np.nanmin(Z[finite])) if np.any(finite) else float("nan")
    z_med = float(np.nanmedian(Z[finite])) if np.any(finite) else float("nan")
    z_max = float(np.nanmax(Z[finite])) if np.any(finite) else float("nan")

    status = status_from_value(
        ratio,
        ratio >= th.z_positive_pass,
        ratio >= th.z_positive_warn,
    )

    return [
        TestResult(
            "Profondeur caméra positive",
            f"la majorité des joints visibles doit avoir Z_cam > 0, ratio >= {th.z_positive_pass:.2f}",
            f"ratio={ratio:.4f}, Z min/median/max=({z_min:.3f}, {z_med:.3f}, {z_max:.3f})",
            status,
            ratio,
        )
    ], X_cam


def points_inside_image(X_img: np.ndarray, valid: np.ndarray, W: int, H: int) -> Tuple[float, int]:
    finite = valid & np.isfinite(X_img).all(axis=-1)
    if not np.any(finite):
        return float("nan"), 0
    u = X_img[..., 0]
    v = X_img[..., 1]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return float(np.mean(inside[finite])), int(np.sum(finite))


def points_inside_boxes(X_img: np.ndarray, valid: np.ndarray, boxes: np.ndarray) -> Tuple[float, int]:
    """Ratio of valid projected joints inside their player/frame xyxy box."""
    finite = valid & np.isfinite(X_img).all(axis=-1) & np.isfinite(boxes).all(axis=-1)[..., None]
    if not np.any(finite):
        return float("nan"), 0

    x1 = boxes[..., 0][..., None]
    y1 = boxes[..., 1][..., None]
    x2 = boxes[..., 2][..., None]
    y2 = boxes[..., 3][..., None]
    u = X_img[..., 0]
    v = X_img[..., 1]
    inside = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return float(np.mean(inside[finite])), int(np.sum(finite))


def bbox_from_projected_joints(X_img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Create tight xyxy boxes from projected joints, shape (N,T,4)."""
    N, T, _, _ = X_img.shape
    boxes = np.full((N, T, 4), np.nan, dtype=np.float64)
    for n in range(N):
        for tt in range(T):
            mask = valid[n, tt] & np.isfinite(X_img[n, tt]).all(axis=-1)
            if not np.any(mask):
                continue
            pts = X_img[n, tt, mask]
            boxes[n, tt, 0:2] = pts.min(axis=0)
            boxes[n, tt, 2:4] = pts.max(axis=0)
    return boxes


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU for boxes shaped (...,4)."""
    valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    out = np.full(a.shape[:-1], np.nan, dtype=np.float64)
    if not np.any(valid):
        return out

    ax1, ay1, ax2, ay2 = [a[..., i] for i in range(4)]
    bx1, by1, bx2, by2 = [b[..., i] for i in range(4)]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = np.maximum(ax2 - ax1, 0.0) * np.maximum(ay2 - ay1, 0.0)
    area_b = np.maximum(bx2 - bx1, 0.0) * np.maximum(by2 - by1, 0.0)
    union = area_a + area_b - inter
    good = valid & (union > 1e-9)
    out[good] = inter[good] / union[good]
    return out


def test_projection_and_boxes(
    X_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    k: np.ndarray,
    boxes: Optional[np.ndarray],
    image_size: Tuple[int, int],
    th: Thresholds,
) -> Tuple[list[TestResult], Dict[str, Any]]:
    W, H = image_size
    X_cam = world_to_camera_A(X_world, R, t)
    X_img, valid = project_camera_to_image(X_cam, K, k)

    image_ratio, count = points_inside_image(X_img, valid, W, H)
    status_img = status_from_value(
        image_ratio,
        image_ratio >= th.image_inside_pass,
        image_ratio >= th.image_inside_warn,
    )

    results = [
        TestResult(
            "Projection dans l'image",
            f"une grande majorité des joints projetés doit être dans [0,{W})x[0,{H}), ratio >= {th.image_inside_pass:.2f}",
            f"ratio={image_ratio:.4f}, nb joints valides projetés={count}, image_size=(W={W}, H={H})",
            status_img,
            image_ratio,
        )
    ]

    debug = {"X_img": X_img, "valid": valid}

    if boxes is not None:
        bbox_ratio, bbox_count = points_inside_boxes(X_img, valid, boxes)
        status_bbox = status_from_value(
            bbox_ratio,
            bbox_ratio >= th.bbox_inside_pass,
            bbox_ratio >= th.bbox_inside_warn,
        )
        results.append(
            TestResult(
                "Projection GT dans les boxes",
                f"les joints projetés doivent majoritairement tomber dans leur bbox, ratio >= {th.bbox_inside_pass:.2f}",
                f"ratio={bbox_ratio:.4f}, nb joints testés={bbox_count}",
                status_bbox,
                bbox_ratio,
            )
        )

        proj_boxes = bbox_from_projected_joints(X_img, valid)
        iou = bbox_iou_xyxy(proj_boxes, boxes)
        mean_iou = float(np.nanmean(iou)) if np.isfinite(iou).any() else float("nan")
        med_iou = float(np.nanmedian(iou)) if np.isfinite(iou).any() else float("nan")
        results.append(
            TestResult(
                "IoU bbox projetée vs bbox_gt",
                "IoU raisonnable si boxes_gt viennent bien des joueurs correspondants; valeur typique attendue > 0.3-0.5 selon marge",
                f"mean IoU={mean_iou:.4f}, median IoU={med_iou:.4f}",
                "PASS" if mean_iou >= 0.30 else ("WARN" if mean_iou >= 0.10 else "FAIL"),
                mean_iou,
            )
        )
        debug["proj_boxes"] = proj_boxes
        debug["bbox_iou"] = iou

    return results, debug


def test_alternative_conventions(
    X_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    k: np.ndarray,
    boxes: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> list[TestResult]:
    """
    Compare A against a few common wrong alternatives using bbox-inside ratio.

    This does not replace a visual check, but helps diagnose transpose/t-as-center mistakes.
    """
    W, H = image_size
    conventions = {
        "A attendu: X_cam = X_world @ R.T + t": world_to_camera_A,
        "B alternatif: X_cam = X_world @ R + t": world_to_camera_B_row_R,
        "C alternatif: t comme centre C, X_cam = (X_world - t) @ R.T": world_to_camera_C_t_as_camera_center,
    }

    rows = []
    for name, fn in conventions.items():
        try:
            X_cam = fn(X_world, R, t)
            X_img, valid = project_camera_to_image(X_cam, K, k)
            img_ratio, _ = points_inside_image(X_img, valid, W, H)
            if boxes is not None:
                box_ratio, _ = points_inside_boxes(X_img, valid, boxes)
            else:
                box_ratio = float("nan")
            z_finite = np.isfinite(X_cam).all(axis=-1)
            z_ratio = float(np.mean((X_cam[..., 2] > 1e-6)[z_finite])) if np.any(z_finite) else float("nan")
            rows.append((name, z_ratio, img_ratio, box_ratio))
        except Exception as exc:
            rows.append((name, float("nan"), float("nan"), float("nan")))

    # Pick best by bbox ratio if boxes exist, otherwise image ratio.
    key_idx = 3 if boxes is not None else 2
    valid_rows = [r for r in rows if np.isfinite(r[key_idx])]
    if valid_rows:
        best = max(valid_rows, key=lambda r: r[key_idx])
        best_name = best[0]
    else:
        best_name = "indéterminé"

    observed = "".join(
        f"      - {name}: Z+={z_ratio:.4f}, in_image={img_ratio:.4f}, in_bbox={box_ratio:.4f}"
        for name, z_ratio, img_ratio, box_ratio in rows
    )
    status = "PASS" if best_name.startswith("A attendu") else "WARN"

    return [
        TestResult(
            "Comparaison conventions alternatives",
            "La convention A attendue doit être la meilleure ou très proche de la meilleure",
            f"meilleure={best_name}{observed}",
            status,
            None,
        )
    ]


# -----------------------------------------------------------------------------
# SAM3D vs GT convention diagnostics
# -----------------------------------------------------------------------------


def pelvis_from_joints(X: np.ndarray, mode: str = "hips_mean") -> np.ndarray:
    """Return pelvis/root from (..., J, C) joints."""
    if mode == "hips_mean":
        return 0.5 * (X[..., 9, :] + X[..., 12, :])
    if mode == "joint8":
        return X[..., 8, :]
    raise ValueError(f"Unknown pelvis mode: {mode}")


def center_at_pelvis(X: np.ndarray, mode: str = "hips_mean") -> Tuple[np.ndarray, np.ndarray]:
    root = pelvis_from_joints(X, mode=mode)
    return X - root[..., None, :], root


def finite_joint_mask(*arrays: np.ndarray) -> np.ndarray:
    """Joint-level mask where all arrays are finite."""
    mask = None
    for arr in arrays:
        m = np.isfinite(arr).all(axis=-1)
        mask = m if mask is None else (mask & m)
    assert mask is not None
    return mask


def masked_l2_mean(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
    err = np.linalg.norm(A - B, axis=-1)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmean(err[mask]))


def masked_l2_median(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
    err = np.linalg.norm(A - B, axis=-1)
    if not np.any(mask):
        return float("nan")
    return float(np.nanmedian(err[mask]))


def best_similarity_scale(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
    """Least-squares scalar s minimizing ||s*A - B|| over valid joints."""
    valid = mask & np.isfinite(A).all(axis=-1) & np.isfinite(B).all(axis=-1)
    if not np.any(valid):
        return 1.0
    a = A[valid].reshape(-1, 3)
    b = B[valid].reshape(-1, 3)
    denom = float(np.sum(a * a))
    if denom < 1e-12:
        return 1.0
    return float(np.sum(a * b) / denom)


def apply_axis_transform(X: np.ndarray, perm: Tuple[int, int, int], signs: Tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
    Y = X[..., list(perm)]
    return scale * Y * np.array(signs, dtype=np.float64)


def enumerate_axis_alignments(
    X_sam_rel: np.ndarray,
    X_gt_rel_cam: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Search all axis permutations and sign flips, with best scalar scale.

    Finds: X_gt_rel_cam ≈ scale * X_sam_rel[..., perm] * signs
    """
    import itertools

    best: Optional[Dict[str, Any]] = None
    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((-1, 1), repeat=3):
            X_tmp = apply_axis_transform(X_sam_rel, perm, signs, scale=1.0)
            scale = best_similarity_scale(X_tmp, X_gt_rel_cam, mask)
            X_aligned = scale * X_tmp
            mpjpe = masked_l2_mean(X_aligned, X_gt_rel_cam, mask)
            med = masked_l2_median(X_aligned, X_gt_rel_cam, mask)
            item = {
                "perm": tuple(int(p) for p in perm),
                "signs": tuple(int(s) for s in signs),
                "scale": float(scale),
                "mpjpe": float(mpjpe),
                "median": float(med),
                "X_aligned": X_aligned,
            }
            if best is None or item["mpjpe"] < best["mpjpe"]:
                best = item
    assert best is not None
    return best


def procrustes_align_row_vectors(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Global similarity Procrustes alignment A -> B over all valid points.

    Returns scale, rotation M such that A @ M * scale ≈ B.
    This is only diagnostic: if Procrustes is much better than axis/sign search,
    SAM3D differs from GT by a general rotation, not just an axis convention.
    """
    valid = mask & np.isfinite(A).all(axis=-1) & np.isfinite(B).all(axis=-1)
    if not np.any(valid):
        return {"scale": 1.0, "R": np.eye(3), "mpjpe": float("nan"), "median": float("nan")}

    a = A[valid].reshape(-1, 3)
    b = B[valid].reshape(-1, 3)
    # A and B are already pelvis-centered per frame, but remove any global mean for safety.
    a0 = a - np.mean(a, axis=0, keepdims=True)
    b0 = b - np.mean(b, axis=0, keepdims=True)
    H = a0.T @ b0
    U, S, Vt = np.linalg.svd(H)
    M = U @ Vt
    if np.linalg.det(M) < 0:
        Vt[-1, :] *= -1
        M = U @ Vt
    denom = np.sum(a0 * a0)
    scale = float(np.sum(S) / max(denom, 1e-12))
    A_aligned = scale * ((A - np.mean(a, axis=0)) @ M + np.mean(b, axis=0))
    return {
        "scale": scale,
        "R": M,
        "mpjpe": masked_l2_mean(A_aligned, B, mask),
        "median": masked_l2_median(A_aligned, B, mask),
        "X_aligned": A_aligned,
    }


def bone_length_stats(
    X_a: np.ndarray,
    X_b: np.ndarray,
    bones: list[Tuple[int, int]],
    mask: np.ndarray,
) -> Dict[str, float]:
    """Compare bone lengths via log-ratio log(len_a / len_b)."""
    logs = []
    for i, j in bones:
        if i >= X_a.shape[-2] or j >= X_a.shape[-2]:
            continue
        m = mask[..., i] & mask[..., j]
        if not np.any(m):
            continue
        la = np.linalg.norm(X_a[..., i, :] - X_a[..., j, :], axis=-1)
        lb = np.linalg.norm(X_b[..., i, :] - X_b[..., j, :], axis=-1)
        good = m & np.isfinite(la) & np.isfinite(lb) & (la > 1e-8) & (lb > 1e-8)
        if np.any(good):
            logs.append(np.log(la[good] / lb[good]))
    if not logs:
        return {"median_abs_log_ratio": float("nan"), "mean_abs_log_ratio": float("nan"), "count": 0}
    vals = np.concatenate([v.reshape(-1) for v in logs])
    return {
        "median_abs_log_ratio": float(np.median(np.abs(vals))),
        "mean_abs_log_ratio": float(np.mean(np.abs(vals))),
        "count": int(vals.size),
    }


def left_right_swap_diagnostic(
    X_sam_aligned: np.ndarray,
    X_gt_rel_cam: np.ndarray,
    mask: np.ndarray,
    swap_pairs: list[Tuple[int, int]],
) -> Dict[str, float]:
    """Check if swapping left/right joints improves MPJPE."""
    base = masked_l2_mean(X_sam_aligned, X_gt_rel_cam, mask)
    swapped = X_sam_aligned.copy()
    for a, b in swap_pairs:
        if a < swapped.shape[-2] and b < swapped.shape[-2]:
            swapped[..., [a, b], :] = swapped[..., [b, a], :]
    sw = masked_l2_mean(swapped, X_gt_rel_cam, mask)
    return {"mpjpe_no_swap": float(base), "mpjpe_with_lr_swap": float(sw), "improvement": float(base - sw)}


def test_sam_vs_gt_conventions(
    X_world_gt: np.ndarray,
    X2d_gt: np.ndarray,
    valid_gt_proj: np.ndarray,
    X3d_sam: np.ndarray,
    X2d_sam: Optional[np.ndarray],
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    k: np.ndarray,
    pelvis_mode: str,
    image_size: Tuple[int, int],
    th: Thresholds,
    bones: Optional[list[Tuple[int, int]]] = None,
    lr_pairs: Optional[list[Tuple[int, int]]] = None,
) -> Tuple[list[TestResult], Dict[str, Any]]:
    """
    Diagnostic SAM3D vs GT.

    Tests:
      1. SAM2D vs GT projected 2D.
      2. Best axis permutation/sign/scale from SAM3D relative to GT camera-relative.
      3. Full Procrustes diagnostic.
      4. Bone length ratio diagnostic.
      5. Left/right swap diagnostic.
      6. Reproject aligned SAM3D using GT root_cam.
    """
    X_cam_gt = world_to_camera_A(X_world_gt, R, t)
    X_gt_rel_cam, root_gt_cam = center_at_pelvis(X_cam_gt, mode=pelvis_mode)
    X_sam_rel, _ = center_at_pelvis(X3d_sam, mode=pelvis_mode)

    mask3d = finite_joint_mask(X_gt_rel_cam, X_sam_rel) & valid_gt_proj
    results: list[TestResult] = []
    debug: Dict[str, Any] = {}

    if X2d_sam is not None:
        mask2d = finite_joint_mask(X2d_sam, X2d_gt) & valid_gt_proj
        med2d = masked_l2_median(X2d_sam, X2d_gt, mask2d)
        mean2d = masked_l2_mean(X2d_sam, X2d_gt, mask2d)
        status = status_from_value(
            med2d,
            med2d <= th.sam2d_median_px_pass,
            med2d <= th.sam2d_median_px_warn,
        )
        results.append(TestResult(
            "SAM2D vs GT projetée",
            f"median error <= {th.sam2d_median_px_pass:g}px si mapping/frame/person IDs cohérents",
            f"mean={mean2d:.2f}px, median={med2d:.2f}px, joints comparés={int(np.sum(mask2d))}",
            status,
            med2d,
        ))

    raw_mpjpe = masked_l2_mean(X_sam_rel, X_gt_rel_cam, mask3d)
    raw_median = masked_l2_median(X_sam_rel, X_gt_rel_cam, mask3d)
    best_axis = enumerate_axis_alignments(X_sam_rel, X_gt_rel_cam, mask3d)
    X_axis = best_axis["X_aligned"]

    status_axis = status_from_value(
        best_axis["mpjpe"],
        best_axis["mpjpe"] <= th.sam_axis_mpjpe_pass,
        best_axis["mpjpe"] <= th.sam_axis_mpjpe_warn,
    )
    results.append(TestResult(
        "Meilleur alignement axes/signs SAM3D -> GT caméra",
        "Une permutation/sign/scale fixe doit réduire clairement l'erreur si le problème est une convention d'axes",
        (
            f"raw mean={raw_mpjpe:.4f}, raw median={raw_median:.4f}; "
            f"best mean={best_axis['mpjpe']:.4f}, median={best_axis['median']:.4f}, "
            f"perm={best_axis['perm']}, signs={best_axis['signs']}, scale={best_axis['scale']:.4f}"
        ),
        status_axis,
        best_axis["mpjpe"],
    ))

    proc = procrustes_align_row_vectors(X_sam_rel, X_gt_rel_cam, mask3d)
    gain = (best_axis["mpjpe"] - proc["mpjpe"]) / max(best_axis["mpjpe"], 1e-8)
    status_proc = "PASS" if gain <= th.sam_procrustes_gain_warn else "WARN"
    results.append(TestResult(
        "Alignement Procrustes complet",
        "Procrustes ne doit pas être beaucoup meilleur que axes/signs; sinon différence de rotation générale ou mapping douteux",
        f"procrustes mean={proc['mpjpe']:.4f}, median={proc['median']:.4f}, scale={proc['scale']:.4f}, gain_vs_axis={100*gain:.1f}%",
        status_proc,
        proc["mpjpe"],
    ))

    if bones is None:
        # OpenPose/body25-ish common bones. Adjust if your 25-joint mapping differs.
        bones = [
            (1, 2), (2, 3), (3, 4),
            (1, 5), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11),
            (8, 12), (12, 13), (13, 14),
            (1, 8), (1, 0),
        ]
    bone_stats = bone_length_stats(X_axis, X_gt_rel_cam, bones, mask3d)
    med_log = bone_stats["median_abs_log_ratio"]
    status_bone = status_from_value(
        med_log,
        med_log <= th.bone_ratio_median_pass_abs_log,
        med_log <= th.bone_ratio_median_warn_abs_log,
    )
    results.append(TestResult(
        "Cohérence longueurs d'os SAM aligné vs GT",
        f"median |log(len_sam/len_gt)| <= {th.bone_ratio_median_pass_abs_log:g}; grand écart = mapping joints suspect",
        f"median_abs_log_ratio={med_log:.4f}, mean_abs_log_ratio={bone_stats['mean_abs_log_ratio']:.4f}, count={bone_stats['count']}",
        status_bone,
        med_log,
    ))

    if lr_pairs is None:
        lr_pairs = [(2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14)]
    lr = left_right_swap_diagnostic(X_axis, X_gt_rel_cam, mask3d, lr_pairs)
    status_lr = "WARN" if lr["improvement"] > 0.05 else "PASS"
    results.append(TestResult(
        "Heuristique inversion gauche/droite",
        "Le swap gauche/droite ne doit pas améliorer fortement le MPJPE",
        f"no_swap={lr['mpjpe_no_swap']:.4f}, with_swap={lr['mpjpe_with_lr_swap']:.4f}, improvement={lr['improvement']:.4f}",
        status_lr,
        lr["improvement"],
    ))

    # Reproject SAM aligned using GT root. If this is bad while axis MPJPE is good,
    # the problem may be scale/depth/projection or pelvis definition.
    X_sam_cam_with_gt_root = X_axis + root_gt_cam[..., None, :]
    X_sam_reproj, valid_sam_reproj = project_camera_to_image(X_sam_cam_with_gt_root, K, k)
    mask_reproj = finite_joint_mask(X_sam_reproj, X2d_gt) & valid_sam_reproj & valid_gt_proj
    reproj_med = masked_l2_median(X_sam_reproj, X2d_gt, mask_reproj)
    reproj_mean = masked_l2_mean(X_sam_reproj, X2d_gt, mask_reproj)
    status_reproj = status_from_value(
        reproj_med,
        reproj_med <= th.sam_reproj_median_px_pass,
        reproj_med <= th.sam_reproj_median_px_warn,
    )
    results.append(TestResult(
        "Reprojection SAM3D aligné + root GT",
        f"median reproj <= {th.sam_reproj_median_px_pass:g}px si axes/scale/pelvis/mapping corrects",
        f"mean={reproj_mean:.2f}px, median={reproj_med:.2f}px, joints comparés={int(np.sum(mask_reproj))}",
        status_reproj,
        reproj_med,
    ))

    debug.update({
        "best_axis_perm": np.array(best_axis["perm"], dtype=np.int64),
        "best_axis_signs": np.array(best_axis["signs"], dtype=np.int64),
        "best_axis_scale": np.array(best_axis["scale"], dtype=np.float64),
        "X_sam_axis_aligned": X_axis,
        "X_sam_reproj": X_sam_reproj,
        "valid_sam_reproj": valid_sam_reproj,
    })
    return results, debug


# -----------------------------------------------------------------------------
# Optional overlays
# -----------------------------------------------------------------------------


def maybe_save_overlay(
    sequence,
    data_dir,
    image_root,
    output_dir,
    X_img,
    boxes,
    image_indices,
):
    import cv2

    img_dir = data_dir / image_root / sequence
    image_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    out_seq = output_dir / sequence
    out_seq.mkdir(parents=True, exist_ok=True)

    N, T_local, J, _ = X_img.shape
    

    for local_idx, image_idx in enumerate(image_indices):
        print(local_idx, image_idx)
        img = cv2.imread(str(image_files[image_idx]))
        if img is None:
            continue

        for n in range(N):
            pts = X_img[n, local_idx]

            for j, p in enumerate(pts):
                if not np.isfinite(p).all():
                    continue
                u, v = int(round(p[0])), int(round(p[1]))
                cv2.circle(img, (u, v), 3, (0, 255, 0), -1)
                """cv2.putText(
                    img,
                    str(j),
                    (u + 2, v - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                )"""

            if boxes is not None and np.isfinite(boxes[n, local_idx]).all():
                x1, y1, x2, y2 = boxes[n, local_idx].astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        out_path = out_seq / f"frame_{image_idx:06d}.jpg"
        cv2.imwrite(str(out_path), img)


# -----------------------------------------------------------------------------
# Sequence runner
# -----------------------------------------------------------------------------


def select_frames(T: int, num_frames: int) -> np.ndarray:
    if num_frames <= 0 or num_frames >= T:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, num_frames).round().astype(int))


def subset_time(arr: np.ndarray, frames: np.ndarray, time_axis: int) -> np.ndarray:
    return np.take(arr, frames, axis=time_axis)


def run_sequence_checks(
    sequence: str,
    args: argparse.Namespace,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    data_dir = Path(ps.DATA_DIR)
    camera_path = data_dir / args.camera_dir / f"{sequence}.npz"
    joints_base = data_dir / args.joints_dir / sequence
    boxes_base = data_dir / args.boxes_dir / sequence

    camera = ensure_camera_shapes(load_camera(camera_path))
    K, R, t, k = camera["K"], camera["R"], camera["t"], camera["k"]
    T = K.shape[0]

    joints_raw = load_np_array(joints_base, possible_keys=("joints_3d", "joints3d", "X_world", "X_world_gt", "arr_0"))
    X_world = ensure_joints_shape(joints_raw, T=T)
    N, T, J, _ = X_world.shape

    boxes: Optional[np.ndarray] = None
    if (boxes_base.with_suffix(".npy")).exists() or (boxes_base.with_suffix(".npz")).exists():
        boxes_raw = load_np_array(boxes_base, possible_keys=("boxes", "bbox", "bboxes", "arr_0"))
        boxes = ensure_boxes_shape(boxes_raw, T=T, N=N)
    else:
        print(f"  [WARN] Pas de boxes pour {sequence}; tests bbox ignorés.")

    X3d_sam_full: Optional[np.ndarray] = None
    X2d_sam_full: Optional[np.ndarray] = None
    if getattr(args, "check_sam", False):
        sam3d_base = data_dir / args.sam3d_dir / sequence
        sam2d_base = data_dir / args.sam2d_dir / sequence
        X3d_sam_raw = load_np_array(
            sam3d_base,
            possible_keys=("skel_3d", "skels_3d", "joints_3d", "pred_keypoints_3d", "arr_0"),
        )
        X3d_sam_full = ensure_sam3d_shape(X3d_sam_raw, T=T, N=N, target_joints=J if args.sam_keep_first_joints else None)
        if X3d_sam_full.shape[2] != J:
            raise ValueError(
                f"SAM3D J={X3d_sam_full.shape[2]} mais GT J={J}. "
                "Ajoute un mapping explicite ou utilise --sam-keep-first-joints si adapté."
            )

        if (sam2d_base.with_suffix(".npy")).exists() or (sam2d_base.with_suffix(".npz")).exists():
            X2d_sam_raw = load_np_array(
                sam2d_base,
                possible_keys=("skel_2d", "skels_2d", "joints_2d", "pred_keypoints_2d", "arr_0"),
            )
            X2d_sam_full = ensure_sam2d_shape(X2d_sam_raw, T=T, N=N, target_joints=J if args.sam_keep_first_joints else None)
            if X2d_sam_full.shape[2] != J:
                raise ValueError(
                    f"SAM2D J={X2d_sam_full.shape[2]} mais GT J={J}. "
                    "Ajoute un mapping explicite ou utilise --sam-keep-first-joints si adapté."
                )
        else:
            print(f"  [WARN] Pas de SAM2D pour {sequence}; test SAM2D vs GT ignoré.")

    image_size = infer_image_size(K, tuple(args.image_size) if args.image_size is not None else None)

    # Subsample frames for speed if requested.
    frames = select_frames(T, args.num_frames)
    if len(frames) < T:
        K_s = K[frames]
        R_s = R[frames]
        t_s = t[frames]
        k_s = k[frames]
        X_s = X_world[:, frames]
        boxes_s = boxes[:, frames] if boxes is not None else None
        X3d_sam_s = X3d_sam_full[:, frames] if X3d_sam_full is not None else None
        X2d_sam_s = X2d_sam_full[:, frames] if X2d_sam_full is not None else None
    else:
        K_s, R_s, t_s, k_s, X_s, boxes_s = K, R, t, k, X_world, boxes
        X3d_sam_s = X3d_sam_full
        X2d_sam_s = X2d_sam_full

    print("\n" + "=" * 100)
    print(f"SEQUENCE: {sequence}")
    print("=" * 100)
    print(f"DATA_DIR : {data_dir}")
    print(f"frames testées : {len(frames)} / {T}")
    print(f"image_size utilisée : W={image_size[0]}, H={image_size[1]}")

    all_results: list[TestResult] = []
    all_results.extend(test_shapes(sequence, K_s, R_s, t_s, k_s, X_s, boxes_s))
    all_results.extend(test_rotation_matrices(R_s, thresholds))
    all_results.extend(test_camera_center(R_s, t_s, thresholds))
    all_results.extend(test_inverse_roundtrip(X_s, R_s, t_s, thresholds))

    depth_results, _ = test_depth(X_s, R_s, t_s, thresholds)
    all_results.extend(depth_results)

    proj_results, debug = test_projection_and_boxes(X_s, K_s, R_s, t_s, k_s, boxes_s, image_size, thresholds)
    all_results.extend(proj_results)
    all_results.extend(test_alternative_conventions(X_s, K_s, R_s, t_s, k_s, boxes_s, image_size))

    if getattr(args, "check_sam", False):
        if X3d_sam_s is None:
            raise RuntimeError("--check-sam activé mais X3d_sam_s est None")
        sam_results, sam_debug = test_sam_vs_gt_conventions(
            X_world_gt=X_s,
            X2d_gt=debug["X_img"],
            valid_gt_proj=debug["valid"],
            X3d_sam=X3d_sam_s,
            X2d_sam=X2d_sam_s,
            K=K_s,
            R=R_s,
            t=t_s,
            k=k_s,
            pelvis_mode=args.pelvis_mode,
            image_size=image_size,
            th=thresholds,
        )
        all_results.extend(sam_results)
        debug.update({f"sam_{key}": val for key, val in sam_debug.items()})

    for result in all_results:
        print_result(result)

    if args.save_overlays:
        overlay_frames_local = select_frames(len(frames), min(args.overlay_count, len(frames)))

        image_indices = frames[overlay_frames_local]
        annotation_offset = 0 #Tested -2, -1, 0, +1, +2 to detect potential misalignments; best is 0. Little offset is due to rolling shutter or motion blur
        annotation_indices = image_indices + annotation_offset

        # Garde uniquement les indices valides
        valid = annotation_indices < X_world.shape[1]
        image_indices = image_indices[valid]
        annotation_indices = annotation_indices[valid]

        # Points/boxes viennent de annotation_indices
        X_cam_full = world_to_camera_A(
            X_world[:, annotation_indices],
            R[annotation_indices],
            t[annotation_indices],
        )
        X_img_full, _ = project_camera_to_image(
            X_cam_full,
            K[annotation_indices],
            k[annotation_indices],
        )
        boxes_full = boxes[:, annotation_indices] if boxes is not None else None

        # Images viennent de image_indices
        maybe_save_overlay(
            sequence=sequence,
            data_dir=data_dir,
            image_root=args.image_root,
            output_dir=Path(args.overlay_dir),
            X_img=X_img_full,
            boxes=boxes_full,
            image_indices=image_indices,
        )

        print(f"  overlays sauvegardés dans : {Path(args.overlay_dir) / sequence}")

    status_counts: Dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in all_results:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    return {
        "sequence": sequence,
        "status_counts": status_counts,
        "results": [asdict(r) for r in all_results],
        "image_size": image_size,
        "num_frames_tested": int(len(frames)),
        "num_frames_total": int(T),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vérifie les conventions caméra/rotation/projection sur les données GT.")
    parser.add_argument("--sequences-file", type=str, default="sequences_gt.txt")
    parser.add_argument("--sequence", type=str, default=None, help="Nom d'une seule séquence à tester.")
    parser.add_argument("--max-sequences", type=int, default=0, help="Limiter le nombre de séquences; 0 = toutes.")
    parser.add_argument("--num-frames", type=int, default=20, help="Nombre de frames échantillonnées par séquence; 0 = toutes.")

    parser.add_argument("--camera-dir", type=str, default="cameras_gt")
    parser.add_argument("--joints-dir", type=str, default="joints_3d_gt")
    parser.add_argument("--boxes-dir", type=str, default="boxes_gt")

    parser.add_argument("--check-sam", action="store_true", help="Active les diagnostics de convention SAM3D vs GT projetée/caméra.")
    parser.add_argument("--sam3d-dir", type=str, default="skel_3d_sam3dbody", help="Dossier DATA_DIR/skel_3d_sam3dbody/{sequence}.npy|npz contenant les keypoints 3D SAM.")
    parser.add_argument("--sam2d-dir", type=str, default="skel_2d_sam3dbody", help="Dossier DATA_DIR/skel_2d_sam3dbody/{sequence}.npy|npz contenant les keypoints 2D SAM.")
    parser.add_argument("--sam-keep-first-joints", action="store_true", help="Si SAM a plus de joints que la GT, garde les J premiers. À remplacer par un mapping explicite dès que possible.")
    parser.add_argument("--pelvis-mode", type=str, default="hips_mean", choices=("hips_mean", "joint8"), help="Définition du pelvis pour les comparaisons relatives.")

    parser.add_argument("--image-size", type=int, nargs=2, default=(None), metavar=("W", "H"), help="Taille image réelle. Si absent, inférée depuis K.")

    parser.add_argument("--save-report", type=str, default="debug_folder/rotation_convention_report.json")
    parser.add_argument("--save-overlays", action="store_true", help="Sauvegarde des overlays 2D sur images si disponibles.")
    parser.add_argument("--image-root", type=str, default="images_gt", help="Dossier DATA_DIR/image_gt/{sequence}/*.jpg pour overlays.")
    parser.add_argument("--overlay-dir", type=str, default="debug_folder/debug_rotation_overlays")
    parser.add_argument("--overlay-count", type=int, default=5, help="Nombre d'images à sauvegarder avec overlay parmi le nombre de frames echantillonées.")

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    data_dir = Path(ps.DATA_DIR)
    thresholds = Thresholds()

    if args.sequence is not None:
        sequences = [args.sequence]
    else:
        sequences = read_sequences(data_dir, args.sequences_file)
        if args.max_sequences and args.max_sequences > 0:
            sequences = sequences[: args.max_sequences]

    print("#" * 100)
    print("CHECK ROTATION / CAMERA CONVENTIONS")
    print("#" * 100)
    print(f"DATA_DIR: {data_dir}")
    print(f"Nombre de séquences: {len(sequences)}")
    print("Convention attendue:")
    print("  Math colonne : X_cam = R @ X_world + t")
    print("  NumPy lignes : X_cam = X_world @ R.T + t")
    print("  Inverse      : X_world = (X_cam - t) @ R")
    print("Seuils:")
    print(json.dumps(asdict(thresholds), indent=2))

    report = {
        "data_dir": str(data_dir),
        "expected_convention": {
            "column_math": "X_cam = R @ X_world + t",
            "numpy_row": "X_cam = X_world @ R.T + t",
            "inverse_numpy_row": "X_world = (X_cam - t) @ R",
        },
        "thresholds": asdict(thresholds),
        "sequences": [],
    }

    global_counts: Dict[str, int] = {"PASS": 0, "WARN": 0, "FAIL": 0}
    had_exception = False

    for sequence in sequences:
        try:
            seq_report = run_sequence_checks(sequence, args, thresholds)
            report["sequences"].append(seq_report)
            for key, val in seq_report["status_counts"].items():
                global_counts[key] = global_counts.get(key, 0) + int(val)
        except Exception as exc:
            had_exception = True
            print("\n" + "=" * 100)
            print(f"SEQUENCE: {sequence}")
            print("=" * 100)
            print(f"  [FAIL] exception pendant les tests : {type(exc).__name__}: {exc}")
            report["sequences"].append(
                {
                    "sequence": sequence,
                    "exception": f"{type(exc).__name__}: {exc}",
                    "status_counts": {"PASS": 0, "WARN": 0, "FAIL": 1},
                    "results": [],
                }
            )
            global_counts["FAIL"] += 1

    print("\n" + "#" * 100)
    print("SUMMARY")
    print("#" * 100)
    print(f"PASS: {global_counts.get('PASS', 0)}")
    print(f"WARN: {global_counts.get('WARN', 0)}")
    print(f"FAIL: {global_counts.get('FAIL', 0)}")

    report["global_counts"] = global_counts
    report_path = Path(args.save_report)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Rapport JSON sauvegardé : {report_path}")

    if had_exception or global_counts.get("FAIL", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
