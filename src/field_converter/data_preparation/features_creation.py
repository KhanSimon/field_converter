

"""
FeatureCreator
==============

Utility class to build training features and labels for a 3D pose refiner.

Expected dataset layout under ``ps.DATA_DIR`` or ``data_dir``::

    data/
      sequences_gt.txt
      cameras_gt/{sequence}.npz       # keys: K, R, t, k
      boxes_gt/{sequence}.npy|npz     # boxes, usually xyxy pixels
      joints_3d_gt/{sequence}.npy|npz # GT world joints, 25 joints
      poses_gt/{sequence}.npz         # optional; not required for the v1 feature set

Generated outputs::

    data/bbox_feat/{sequence}.npy
    data/bbox_feat_clean/{sequence}.npy
    data/cam_feat_base_clean/{sequence}.npy
    data/cam_feat_base_noisy/{sequence}.npy
    data/cam_feat_boosted_clean/{sequence}.npy
    data/cam_feat_boosted_noisy/{sequence}.npy
    data/valid_mask/{sequence}.npy
    data/valid_joints/{sequence}.npy
    data/Y_rel_cam_gt/{sequence}.npy
    data/Y_cam_gt/{sequence}.npy
    data/Y_root_cam_gt/{sequence}.npy
    data/Y_2d_gt/{sequence}.npy
    data/features/{sequence}.npz

Optional SAM3DBody inputs are folded directly into ``data/features/{sequence}.npz``:

    data/skel_2d_sam3dbody_from_bbox_gt/{sequence}.npy
    data/skel_3d_sam3dbody_from_bbox_gt/{sequence}.npy

Conventions
-----------

This code assumes the common extrinsic convention:

    X_cam_col = R @ X_world_col + t

For numpy row-vector arrays shaped (..., 3), this is implemented as:

    X_cam = X_world @ R.T + t

The input GT world joints are normalized internally to shape:

    (N, T, J, 3)

where N is number of players, T number of frames, J=25 joints.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple
import hashlib
import json
import warnings

import numpy as np


PelvisMode = Literal["hips_mean", "joint8"]
FOLDER = "features2"


@dataclass
class NoiseConfig:
    """Noise configuration used to make training features robust to test-time errors."""

    # Bounding-box augmentation in pixel space.
    bbox_center_std: float = 0.03  # relative to box width/height
    bbox_scale_std: float = 0.05   # log-normal-ish multiplicative scale std
    bbox_min_size_px: float = 4.0

    # Camera intrinsic / distortion noise.
    focal_rel_std: float = 0.01       # 1% noise on fx/fy
    principal_rel_std: float = 0.005  # relative to image width/height
    distortion_std: float = 1e-4

    # Camera extrinsic noise.
    rotation_deg_std: float = 0.25    # std of small angle perturbation in degrees
    translation_std: float = 0.05     # std added to extrinsic t, in same units as t


@dataclass
class FeatureCreator:
    """
    Build refiner features and labels from GT joints, boxes and calibrated cameras.

    Parameters
    ----------
    data_dir:
        Dataset root. If None, imports ``ps`` and uses ``ps.DATA_DIR``.
    image_size:
        Optional image size as ``(W, H)``. If None, it is inferred per sequence from
        the principal point as roughly ``W=2*cx`` and ``H=2*cy``.
    sequences_file:
        Name of the text file listing training sequences.
    pelvis_mode:
        ``"hips_mean"`` uses the mean of joints 9 and 12 as pelvis.
        In that mode, GT joint 8 is also replaced by the same hips mean before
        creating labels and projections.
        ``"joint8"`` uses joint 8 directly.
        Indices are assumed to be zero-based.
    margin_for_boxes:
        Optional margin applied only if you call ``boxes_from_2d_joints``.
        Existing ``boxes_gt`` are not enlarged unless ``rebuild_boxes_from_gt=True``.
    noise:
        Noise config for bbox/camera feature augmentation.
    seed:
        Global random seed. Each sequence gets a deterministic derived seed.
    """

    data_dir: Optional[Path | str] = None
    image_size: Optional[Tuple[int, int]] = None
    sequences_file: str = "sequences_gt.txt"
    pelvis_mode: PelvisMode = "joint8"
    margin_for_boxes: float = 0.15
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    seed: int = 12345

    def __post_init__(self) -> None:
        if self.data_dir is None:
            try:
                from field_converter import pathseeker as ps 
            except ImportError as exc:
                raise ImportError(
                    "data_dir is None, but importing ps failed. "
                    "Pass data_dir explicitly or ensure ps.DATA_DIR exists."
                ) from exc
            self.data_dir = Path(ps.DATA_DIR)
        else:
            self.data_dir = Path(self.data_dir)

        self.dirs = {
            "cameras": self.data_dir / "cameras_gt",
            "boxes": self.data_dir / "boxes_gt",
            "joints3d": self.data_dir / "joints_3d_gt",
            "poses": self.data_dir / "poses_gt",
            "bbox_feat": self.data_dir / "bbox_feat",
            "bbox_feat_clean": self.data_dir / "bbox_feat_clean",
            "cam_feat_base_clean": self.data_dir / "cam_feat_base_clean",
            "cam_feat_base_noisy": self.data_dir / "cam_feat_base_noisy",
            "cam_feat_boosted_clean": self.data_dir / "cam_feat_boosted_clean",
            "cam_feat_boosted_noisy": self.data_dir / "cam_feat_boosted_noisy",
            "valid_mask": self.data_dir / "valid_mask",
            "valid_joints": self.data_dir / "valid_joints",
            "Y_rel_cam_gt": self.data_dir / "Y_rel_cam_gt",
            "Y_cam_gt": self.data_dir / "Y_cam_gt",
            "Y_root_cam_gt": self.data_dir / "Y_root_cam_gt",
            "Y_2d_gt": self.data_dir / "Y_2d_gt",
            "features": self.data_dir / FOLDER,
        }
        self.sam3dbody_from_bbox_gt_dirs = {
            "skel_2d_sam3dbody_from_bbox_gt": self.data_dir / "skel_2d_sam3dbody_from_bbox_gt",
            "skel_3d_sam3dbody_from_bbox_gt": self.data_dir / "skel_3d_sam3dbody_from_bbox_gt",
        }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def read_sequences(self) -> list[str]:
        """Read the sequence list from ``data/sequences_gt.txt``."""
        path = self.data_dir / self.sequences_file
        if not path.exists():
            raise FileNotFoundError(f"Sequence file not found: {path}")

        sequences: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sequences.append(line)
        return sequences

    def create_all(
        self,
        overwrite: bool = False,
        rebuild_boxes_from_gt: bool = False,
        save_individual_folders: bool = True,
        save_npz: bool = True,
    ) -> None:
        """
        Create features for all sequences in ``sequences_gt.txt``.

        Parameters
        ----------
        overwrite:
            If False, existing consolidated ``data/features/{seq}.npz`` files are skipped.
        rebuild_boxes_from_gt:
            If True, ignore ``boxes_gt`` and rebuild boxes by projecting GT joints.
            If False, use ``boxes_gt`` as the base boxes.
        save_individual_folders:
            Save each feature in its own subfolder.
        save_npz:
            Save a consolidated ``.npz`` per sequence under ``data/features``.
        """
        self._mkdirs()
        sequences = self.read_sequences()

        for seq in sequences:
            out_path = self.dirs["features"] / f"{seq}.npz"
            if out_path.exists() and not overwrite and save_npz:
                print(f"[skip] {seq} exists: {out_path}")
                continue

            print(f"[create] {seq}")
            features = self.create_sequence_features(
                seq,
                rebuild_boxes_from_gt=rebuild_boxes_from_gt,
            )

            if save_individual_folders:
                self.save_individual_features(seq, features, overwrite=overwrite)
            if save_npz:
                self.save_feature_npz(seq, features, overwrite=overwrite)

    def create_sequence_features(
        self,
        sequence: str,
        rebuild_boxes_from_gt: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Build all features/labels for one sequence.

        Returns
        -------
        dict
            Contains both noisy/clean features and GT labels. Main shapes:

            - ``bbox_feat``: ``(N, T, 5)`` noisy normalized bbox features
            - ``bbox_feat_clean``: ``(N, T, 5)`` clean normalized bbox features
            - ``cam_feat_base_clean``: ``(T, 6)`` = fx/W, fy/H, cx/W, cy/H, k1, k2
            - ``cam_feat_base_noisy``: ``(T, 6)`` noisy version
            - ``cam_feat_boosted_clean``: ``(T, 12)`` base + camera_center + camera_forward
            - ``cam_feat_boosted_noisy``: ``(T, 12)`` noisy version
            - ``valid_mask``: ``(N, T)`` player-frame validity
            - ``valid_joints``: ``(N, T, J)`` joint-level validity
            - ``Y_cam_gt``: ``(N, T, J, 3)`` GT camera-space joints
            - ``Y_rel_cam_gt``: ``(N, T, J, 3)`` GT root-relative camera-space joints
            - ``Y_root_cam_gt``: ``(N, T, 3)`` GT pelvis/root in camera-space
            - ``Y_2d_gt``: ``(N, T, J, 2)`` projected GT 2D joints in pixels
            - ``skel_2d_sam3dbody_from_bbox_gt``: optional ``(N, T, J, 2)`` SAM2D pixels
            - ``skel_3d_sam3dbody_from_bbox_gt``: optional ``(N, T, J, 3)`` SAM3D camera convention
        """
        rng = self._rng_for_sequence(sequence)

        camera = self.load_camera(sequence)
        K, R, t, k = camera["K"], camera["R"], camera["t"], camera["k"]
        T = K.shape[0]

        W, H = self.get_image_size(K)

        X_world_gt = self.load_joints3d(sequence)  # (N,T,J,3)
        X_world_gt = self._ensure_ntj3(X_world_gt, T=T, name=f"{sequence} joints_3d_gt")
        N, T_j, J, _ = X_world_gt.shape
        if T_j != T:
            raise ValueError(
                f"Time mismatch for {sequence}: joints T={T_j}, camera T={T}"
            )
        if J != 25:
            warnings.warn(
                f"{sequence}: expected 25 joints, got {J}. Code will still run, "
                "but verify your mapping."
            )
        if self.pelvis_mode == "hips_mean":
            X_world_gt = self.replace_joint8_with_hips_mean(X_world_gt)

        X_cam_gt = self.world_to_camera(X_world_gt, R, t)  # (N,T,J,3)
        root_cam_gt = self.compute_pelvis(X_cam_gt, mode=self.pelvis_mode)  # (N,T,3)
        X_rel_cam_gt = X_cam_gt - root_cam_gt[:, :, None, :]

        Y_2d_gt, valid_joints_proj = self.project_world_to_image(X_world_gt, K, R, t, k)

        if rebuild_boxes_from_gt:
            boxes_xyxy = self.boxes_from_2d_joints_batch(
                Y_2d_gt,
                valid_joints_proj,
                image_size=(W, H),
                margin=self.margin_for_boxes,
            )
        else:
            boxes_xyxy = self.load_boxes(sequence)
            boxes_xyxy = self._ensure_nt4(boxes_xyxy, T=T, N=N, name=f"{sequence} boxes_gt")

        bbox_feat_clean = self.make_bbox_features(boxes_xyxy, image_size=(W, H))
        boxes_noisy = self.add_bbox_noise(boxes_xyxy, image_size=(W, H), rng=rng)
        bbox_feat_noisy = self.make_bbox_features(boxes_noisy, image_size=(W, H))

        noisy_camera = self.add_camera_noise(K, R, t, k, image_size=(W, H), rng=rng)
        cam_feat_base_clean, cam_feat_boosted_clean = self.make_camera_features(K, R, t, k, image_size=(W, H))
        cam_feat_base_noisy, cam_feat_boosted_noisy = self.make_camera_features(
            noisy_camera["K"], noisy_camera["R"], noisy_camera["t"], noisy_camera["k"], image_size=(W, H)
        )

        valid_box = np.isfinite(boxes_xyxy).all(axis=-1)  # (N,T)
        valid_joints_3d = np.isfinite(X_world_gt).all(axis=-1)  # (N,T,J)
        valid_depth = X_cam_gt[..., 2] > 1e-6
        valid_joints = valid_joints_3d & valid_depth & valid_joints_proj
        valid_mask = valid_box & valid_joints.any(axis=-1) & np.isfinite(root_cam_gt).all(axis=-1)

        features = {
            "bbox_feat": bbox_feat_noisy.astype(np.float32),
            "bbox_feat_clean": bbox_feat_clean.astype(np.float32),
            "cam_feat_base_clean": cam_feat_base_clean.astype(np.float32),
            "cam_feat_base_noisy": cam_feat_base_noisy.astype(np.float32),
            "cam_feat_boosted_clean": cam_feat_boosted_clean.astype(np.float32),
            "cam_feat_boosted_noisy": cam_feat_boosted_noisy.astype(np.float32),
            "valid_mask": valid_mask.astype(bool),
            "valid_joints": valid_joints.astype(bool),
            "Y_rel_cam_gt": X_rel_cam_gt.astype(np.float32),
            "Y_cam_gt": X_cam_gt.astype(np.float32),
            "Y_root_cam_gt": root_cam_gt.astype(np.float32),
            "Y_2d_gt": Y_2d_gt.astype(np.float32),
            "boxes_xyxy": boxes_xyxy.astype(np.float32),
            "boxes_xyxy_noisy": boxes_noisy.astype(np.float32),
            "K": K.astype(np.float32),
            "R": R.astype(np.float32),
            "t": t.astype(np.float32),
            "k": self._truncate_k(k).astype(np.float32),
            "image_size": np.array([W, H], dtype=np.int32),
        }
        self.add_sam3dbody_from_bbox_gt_features(sequence, features, T=T)
        return features

    def save_individual_features(
        self,
        sequence: str,
        features: Dict[str, np.ndarray],
        overwrite: bool = False,
    ) -> None:
        """Save requested feature arrays into their dedicated subfolders."""
        key_to_dir = {
            "bbox_feat": "bbox_feat",
            "bbox_feat_clean": "bbox_feat_clean",
            "cam_feat_base_clean": "cam_feat_base_clean",
            "cam_feat_base_noisy": "cam_feat_base_noisy",
            "cam_feat_boosted_clean": "cam_feat_boosted_clean",
            "cam_feat_boosted_noisy": "cam_feat_boosted_noisy",
            "valid_mask": "valid_mask",
            "valid_joints": "valid_joints",
            "Y_rel_cam_gt": "Y_rel_cam_gt",
            "Y_cam_gt": "Y_cam_gt",
            "Y_root_cam_gt": "Y_root_cam_gt",
            "Y_2d_gt": "Y_2d_gt",
        }
        for key, dirname in key_to_dir.items():
            out_path = self.dirs[dirname] / f"{sequence}.npy"
            if out_path.exists() and not overwrite:
                continue
            np.save(out_path, features[key])

    def save_feature_npz(
        self,
        sequence: str,
        features: Dict[str, np.ndarray],
        overwrite: bool = False,
    ) -> Path:
        """Save all features for one sequence into ``data/features/{sequence}.npz``."""
        out_path = self.dirs["features"] / f"{sequence}.npz"
        if out_path.exists() and not overwrite:
            return out_path

        meta = {
            "pelvis_mode": self.pelvis_mode,
            "bbox_feat": "cx/W, cy/H, w/W, h/H, w/h",
            "cam_feat_base": "fx/W, fy/H, cx/W, cy/H, k1, k2",
            "cam_feat_boosted": "base + camera_center_world(3) + camera_forward_world(3)",
            "gt_joint8": (
                "replaced by mean(joint9, joint12) before label creation "
                "when pelvis_mode='hips_mean'"
            ),
            "k": "stored with the last 3 distortion columns removed when present",
            "sam3dbody_from_bbox_gt": {
                "skel_2d_sam3dbody_from_bbox_gt": "(N,T,J,2), image coordinates, no sign flip",
                "skel_3d_sam3dbody_from_bbox_gt": (
                    "(N,T,J,3), orientation checked against Y_rel_cam_gt; "
                    "applied sign is stored in sam3d_orientation_sign"
                ),
                "sam3d_orientation_sign": self._meta_scalar(features, "sam3d_orientation_sign", default=None),
                "sam3d_orientation_mpjpe_keep": self._meta_scalar(
                    features,
                    "sam3d_orientation_mpjpe_keep",
                    default=None,
                ),
                "sam3d_orientation_mpjpe_flip": self._meta_scalar(
                    features,
                    "sam3d_orientation_mpjpe_flip",
                    default=None,
                ),
            },
            "extrinsic_convention": "X_cam_col = R @ X_world_col + t",
        }

        # np.savez cannot store nested dict nicely without pickle; store JSON string.
        payload = dict(features)
        payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
        np.savez_compressed(out_path, **payload)
        return out_path

    # ---------------------------------------------------------------------
    # Loading
    # ---------------------------------------------------------------------

    def load_camera(self, sequence: str) -> Dict[str, np.ndarray]:
        """Load camera file ``data/cameras_gt/{sequence}.npz``."""
        path = self.dirs["cameras"] / f"{sequence}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Camera file not found: {path}")
        data = dict(np.load(path))
        for key in ("K", "R", "t", "k"):
            if key not in data:
                raise KeyError(f"Camera file {path} missing key '{key}'")
        return {key: np.asarray(data[key]) for key in ("K", "R", "t", "k")}

    def load_boxes(self, sequence: str) -> np.ndarray:
        """Load boxes from ``data/boxes_gt``. Supports ``.npy`` and ``.npz``."""
        return self._load_array_from_folder(self.dirs["boxes"], sequence, keys=("boxes", "bbox", "bboxes"))

    def load_joints3d(self, sequence: str) -> np.ndarray:
        """Load GT world joints from ``data/joints_3d_gt``. Supports ``.npy`` and ``.npz``."""
        return self._load_array_from_folder(
            self.dirs["joints3d"],
            sequence,
            keys=("joints_3d", "joints3d", "X_world", "X_world_gt", "arr_0"),
        )

    def add_sam3dbody_from_bbox_gt_features(
        self,
        sequence: str,
        features: Dict[str, np.ndarray],
        T: int,
    ) -> None:
        """Add optional SAM3DBody arrays from bbox GT folders to the consolidated payload."""
        sam2d_key = "skel_2d_sam3dbody_from_bbox_gt"
        sam3d_key = "skel_3d_sam3dbody_from_bbox_gt"

        sam2d = self._load_optional_sequence_array(
            self.sam3dbody_from_bbox_gt_dirs[sam2d_key],
            sequence,
        )
        if sam2d is not None:
            features[sam2d_key] = self._ensure_ntjc(
                sam2d,
                T=T,
                C=2,
                name=f"{sequence} {sam2d_key}",
            ).astype(np.float32)
            features["sam2d_layout_fixed"] = np.array(True, dtype=np.bool_)

        sam3d = self._load_optional_sequence_array(
            self.sam3dbody_from_bbox_gt_dirs[sam3d_key],
            sequence,
        )
        if sam3d is not None:
            sam3d_ntj3 = self._ensure_ntjc(
                sam3d,
                T=T,
                C=3,
                name=f"{sequence} {sam3d_key}",
            )
            sam3d_oriented, sign, keep_mpjpe, flip_mpjpe = self.ensure_sam3d_camera_orientation(
                sam3d_ntj3,
                gt_rel_cam=features.get("Y_rel_cam_gt"),
                valid_joints=features.get("valid_joints"),
            )
            features[sam3d_key] = sam3d_oriented.astype(np.float32)
            features["sam3d_orientation_sign"] = np.array(sign, dtype=np.int8)
            features["sam3d_orientation_mpjpe_keep"] = np.array(keep_mpjpe, dtype=np.float32)
            features["sam3d_orientation_mpjpe_flip"] = np.array(flip_mpjpe, dtype=np.float32)
            features["sam3d_convention_fixed"] = np.array(sign == -1, dtype=np.bool_)

    def ensure_sam3d_camera_orientation(
        self,
        sam3d_ntj3: np.ndarray,
        *,
        gt_rel_cam: Optional[np.ndarray],
        valid_joints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int, float, float]:
        """
        Keep or globally flip SAM3D so its relative pose matches GT camera-relative joints.

        Some SAM3D exports can differ by a global sign flip. We avoid a hard-coded
        convention by comparing pelvis-centered SAM against ``Y_rel_cam_gt`` and
        choosing the lower MPJPE between ``sam`` and ``-sam``.
        """
        if gt_rel_cam is None:
            warnings.warn("Y_rel_cam_gt missing; keeping SAM3D orientation unchanged.")
            return sam3d_ntj3, 1, float("nan"), float("nan")

        gt_rel = np.asarray(gt_rel_cam, dtype=np.float64)
        sam = np.asarray(sam3d_ntj3, dtype=np.float64)
        if gt_rel.shape != sam.shape:
            warnings.warn(
                f"Cannot check SAM3D orientation: shape mismatch sam={sam.shape}, gt_rel={gt_rel.shape}. "
                "Keeping SAM3D orientation unchanged."
            )
            return sam3d_ntj3, 1, float("nan"), float("nan")

        sam_root = self.compute_pelvis(sam, mode=self.pelvis_mode)
        sam_rel = sam - sam_root[:, :, None, :]

        mask = np.isfinite(sam_rel).all(axis=-1) & np.isfinite(gt_rel).all(axis=-1)
        if valid_joints is not None:
            vj = np.asarray(valid_joints, dtype=bool)
            if vj.shape == mask.shape:
                mask &= vj

        if not np.any(mask):
            warnings.warn("Cannot check SAM3D orientation: no finite valid joints. Keeping orientation unchanged.")
            return sam3d_ntj3, 1, float("nan"), float("nan")

        keep_mpjpe = self._masked_mpjpe(sam_rel, gt_rel, mask)
        flip_mpjpe = self._masked_mpjpe(-sam_rel, gt_rel, mask)
        sign = -1 if flip_mpjpe < keep_mpjpe else 1
        return (sign * sam3d_ntj3).astype(np.float32), sign, keep_mpjpe, flip_mpjpe

    @staticmethod
    def _masked_mpjpe(A: np.ndarray, B: np.ndarray, mask: np.ndarray) -> float:
        """Mean Euclidean joint error over a boolean ``(N,T,J)`` mask."""
        diff = np.asarray(A, dtype=np.float64) - np.asarray(B, dtype=np.float64)
        err = np.linalg.norm(diff, axis=-1)
        return float(err[np.asarray(mask, dtype=bool)].mean())

    @staticmethod
    def _meta_scalar(features: Dict[str, np.ndarray], key: str, default: Any = None) -> Any:
        """Return a JSON-serializable scalar from a feature payload."""
        if key not in features:
            return default
        value = np.asarray(features[key])
        if value.shape != ():
            return default
        item = value.item()
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    # ---------------------------------------------------------------------
    # Core geometry
    # ---------------------------------------------------------------------

    @staticmethod
    def world_to_camera(X_world: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Convert world points to camera coordinates.

        Parameters
        ----------
        X_world:
            ``(N,T,J,3)`` row-vector points.
        R:
            ``(T,3,3)`` world-to-camera rotation.
        t:
            ``(T,3)`` world-to-camera translation.

        Returns
        -------
        np.ndarray
            ``(N,T,J,3)`` camera-space points.
        """
        return np.einsum("ntjw,tcw->ntjc", X_world, R) + t[None, :, None, :]

    @staticmethod
    def camera_to_world(X_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Convert camera points to world coordinates.

        Row-vector equivalent of ``X_world_col = R.T @ (X_cam_col - t)``.
        """
        return np.einsum("ntjc,twc->ntjw", X_cam - t[None, :, None, :], R)

    def compute_pelvis(self, X: np.ndarray, mode: Optional[PelvisMode] = None) -> np.ndarray:
        """
        Compute pelvis/root from joints.

        Parameters
        ----------
        X:
            Joint array ``(..., J, 3)``.
        mode:
            ``"hips_mean"`` = mean of joints 9 and 12.
            ``"joint8"`` = joint 8.

        Returns
        -------
        np.ndarray
            Root array ``(..., 3)``.
        """
        mode = mode or self.pelvis_mode
        if mode == "hips_mean":
            return 0.5 * (X[..., 9, :] + X[..., 12, :])
        if mode == "joint8":
            return X[..., 8, :]
        raise ValueError(f"Unknown pelvis mode: {mode}")

    @staticmethod
    def replace_joint8_with_hips_mean(X: np.ndarray) -> np.ndarray:
        """Return a copy where joint 8 is replaced by mean(joint 9, joint 12)."""
        X = np.asarray(X)
        if X.shape[-2] <= 12:
            raise ValueError(
                f"hips_mean requires joints 8, 9 and 12, but got joint dimension {X.shape[-2]}"
            )
        out = X.copy()
        out[..., 8, :] = 0.5 * (out[..., 9, :] + out[..., 12, :])
        return out

    def project_world_to_image(
        self,
        X_world: np.ndarray,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        k: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project world joints to image pixels with radial distortion k1/k2.

        Parameters
        ----------
        X_world:
            ``(N,T,J,3)`` world coordinates.
        K:
            ``(T,3,3)`` intrinsics.
        R:
            ``(T,3,3)`` world-to-camera rotation.
        t:
            ``(T,3)`` world-to-camera translation.
        k:
            ``(T,2)`` radial distortion coefficients ``k1,k2``.

        Returns
        -------
        X_img:
            ``(N,T,J,2)`` pixel coordinates. Invalid points are NaN.
        valid:
            ``(N,T,J)`` bool mask.
        """
        X_cam = self.world_to_camera(X_world, R, t)
        return self.project_camera_to_image(X_cam, K, k, eps=eps)

    @staticmethod
    def project_camera_to_image(
        X_cam: np.ndarray,
        K: np.ndarray,
        k: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project camera-space joints to image pixels with radial distortion k1/k2.

        Parameters
        ----------
        X_cam:
            ``(N,T,J,3)`` camera coordinates.
        K:
            ``(T,3,3)`` intrinsics.
        k:
            ``(T,2)`` radial distortion coefficients.
        """
        N, T, J, _ = X_cam.shape
        X_img = np.full((N, T, J, 2), np.nan, dtype=np.float32)

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
        factor = 1.0 + k1 * r2 + k2 * (r2 ** 2)
        x_d = x * factor
        y_d = y * factor

        fx = K[:, 0, 0][None, :, None]
        fy = K[:, 1, 1][None, :, None]
        cx = K[:, 0, 2][None, :, None]
        cy = K[:, 1, 2][None, :, None]

        u = fx * x_d + cx
        v = fy * y_d + cy

        X_img[..., 0] = u.astype(np.float32)
        X_img[..., 1] = v.astype(np.float32)
        X_img[~valid] = np.nan
        return X_img, valid

    # ---------------------------------------------------------------------
    # Feature creation
    # ---------------------------------------------------------------------

    @staticmethod
    def make_bbox_features(boxes_xyxy: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Convert xyxy boxes to normalized features.

        Returns ``(N,T,5)`` with:

            ``cx/W, cy/H, w/W, h/H, w/h``
        """
        W, H = image_size
        boxes = boxes_xyxy.astype(np.float64)
        x1, y1, x2, y2 = [boxes[..., i] for i in range(4)]
        w = x2 - x1
        h = y2 - y1
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        feat = np.stack(
            [
                cx / W,
                cy / H,
                w / W,
                h / H,
                w / np.maximum(h, 1e-6),
            ],
            axis=-1,
        )
        feat[~np.isfinite(boxes).all(axis=-1)] = np.nan
        return feat.astype(np.float32)

    def make_camera_features(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        k: np.ndarray,
        image_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build base and boosted camera features.

        Base feature shape ``(T,6)``:

            ``fx/W, fy/H, cx/W, cy/H, k1, k2``

        Boosted feature shape ``(T,12)``:

            ``base + camera_center_world(3) + camera_forward_world(3)``

        Camera center:

            ``C = -R.T @ t``

        Camera forward direction, assuming camera +Z is forward:

            ``forward = R.T @ [0,0,1]``
        """
        W, H = image_size
        fx = K[:, 0, 0] / W
        fy = K[:, 1, 1] / H
        cx = K[:, 0, 2] / W
        cy = K[:, 1, 2] / H
        base = np.stack([fx, fy, cx, cy, k[:, 0], k[:, 1]], axis=-1)

        C = -np.einsum("tji,tj->ti", R, t)  # R.T @ t per frame, negated
        forward_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        forward = np.einsum("tji,j->ti", R, forward_cam)  # R.T @ [0,0,1]
        forward /= np.maximum(np.linalg.norm(forward, axis=-1, keepdims=True), 1e-8)

        boosted = np.concatenate([base, C, forward], axis=-1)
        return base.astype(np.float32), boosted.astype(np.float32)

    def add_bbox_noise(
        self,
        boxes_xyxy: np.ndarray,
        image_size: Tuple[int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add realistic random shift/scale noise to xyxy boxes in pixel space."""
        W, H = image_size
        boxes = boxes_xyxy.astype(np.float64).copy()
        valid = np.isfinite(boxes).all(axis=-1)
        if not np.any(valid):
            return boxes.astype(np.float32)

        x1, y1, x2, y2 = [boxes[..., i] for i in range(4)]
        bw = np.maximum(x2 - x1, self.noise.bbox_min_size_px)
        bh = np.maximum(y2 - y1, self.noise.bbox_min_size_px)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        shift_x = rng.normal(0.0, self.noise.bbox_center_std, size=cx.shape) * bw
        shift_y = rng.normal(0.0, self.noise.bbox_center_std, size=cy.shape) * bh
        scale_w = np.exp(rng.normal(0.0, self.noise.bbox_scale_std, size=bw.shape))
        scale_h = np.exp(rng.normal(0.0, self.noise.bbox_scale_std, size=bh.shape))

        cx_n = cx + shift_x
        cy_n = cy + shift_y
        bw_n = bw * scale_w
        bh_n = bh * scale_h

        noisy = np.stack(
            [
                cx_n - 0.5 * bw_n,
                cy_n - 0.5 * bh_n,
                cx_n + 0.5 * bw_n,
                cy_n + 0.5 * bh_n,
            ],
            axis=-1,
        )

        noisy[..., 0] = np.clip(noisy[..., 0], 0, W - 1)
        noisy[..., 2] = np.clip(noisy[..., 2], 0, W - 1)
        noisy[..., 1] = np.clip(noisy[..., 1], 0, H - 1)
        noisy[..., 3] = np.clip(noisy[..., 3], 0, H - 1)
        noisy[~valid] = np.nan
        return noisy.astype(np.float32)

    def add_camera_noise(
        self,
        K: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        k: np.ndarray,
        image_size: Tuple[int, int],
        rng: np.random.Generator,
    ) -> Dict[str, np.ndarray]:
        """Create noisy camera parameters, then camera features can be computed from them."""
        W, H = image_size
        K_n = K.astype(np.float64).copy()
        R_n = R.astype(np.float64).copy()
        t_n = t.astype(np.float64).copy()
        k_n = k.astype(np.float64).copy()

        T = K.shape[0]

        # Intrinsics.
        K_n[:, 0, 0] *= 1.0 + rng.normal(0.0, self.noise.focal_rel_std, size=T)
        K_n[:, 1, 1] *= 1.0 + rng.normal(0.0, self.noise.focal_rel_std, size=T)
        K_n[:, 0, 2] += rng.normal(0.0, self.noise.principal_rel_std * W, size=T)
        K_n[:, 1, 2] += rng.normal(0.0, self.noise.principal_rel_std * H, size=T)
        k_n += rng.normal(0.0, self.noise.distortion_std, size=k_n.shape)

        # Extrinsics: small random rotation left-multiplied in camera/world-to-camera convention.
        rot_std_rad = np.deg2rad(self.noise.rotation_deg_std)
        rotvec = rng.normal(0.0, rot_std_rad, size=(T, 3))
        dR = self.axis_angle_to_matrix(rotvec)
        R_n = dR @ R_n
        t_n = t_n + rng.normal(0.0, self.noise.translation_std, size=t_n.shape)

        return {
            "K": K_n.astype(np.float32),
            "R": R_n.astype(np.float32),
            "t": t_n.astype(np.float32),
            "k": k_n.astype(np.float32),
        }

    # ---------------------------------------------------------------------
    # Box generation from 2D joints
    # ---------------------------------------------------------------------

    def boxes_from_2d_joints_batch(
        self,
        joints_2d: np.ndarray,
        valid: np.ndarray,
        image_size: Tuple[int, int],
        margin: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build xyxy boxes from projected joints.

        Parameters
        ----------
        joints_2d:
            ``(N,T,J,2)`` projected pixels.
        valid:
            ``(N,T,J)`` joint validity.
        image_size:
            ``(W,H)``.
        margin:
            Relative margin based on max(width,height). If None, uses
            ``self.margin_for_boxes``.
        """
        margin = self.margin_for_boxes if margin is None else margin
        W, H = image_size
        N, T, _, _ = joints_2d.shape
        boxes = np.full((N, T, 4), np.nan, dtype=np.float32)

        for n in range(N):
            for tt in range(T):
                mask = valid[n, tt] & np.isfinite(joints_2d[n, tt]).all(axis=-1)
                if not np.any(mask):
                    continue
                pts = joints_2d[n, tt, mask]
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                bw = x2 - x1
                bh = y2 - y1
                m = margin * max(float(bw), float(bh), 1.0)
                x1 = np.clip(x1 - m, 0, W - 1)
                y1 = np.clip(y1 - m, 0, H - 1)
                x2 = np.clip(x2 + m, 0, W - 1)
                y2 = np.clip(y2 + m, 0, H - 1)
                boxes[n, tt] = np.array([x1, y1, x2, y2], dtype=np.float32)
        return boxes

    # ---------------------------------------------------------------------
    # Rotation utility
    # ---------------------------------------------------------------------

    @staticmethod
    def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
        """Convert axis-angle vectors ``(...,3)`` to rotation matrices ``(...,3,3)``."""
        rotvec = np.asarray(rotvec, dtype=np.float64)
        theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
        axis = rotvec / np.maximum(theta, 1e-12)
        x, y, z = np.moveaxis(axis, -1, 0)
        zeros = np.zeros_like(x)

        K = np.stack(
            [
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros,
            ],
            axis=-1,
        ).reshape(*rotvec.shape[:-1], 3, 3)

        I = np.broadcast_to(np.eye(3), K.shape)
        theta_m = theta[..., None]
        R = I + np.sin(theta_m) * K + (1.0 - np.cos(theta_m)) * (K @ K)

        # For very small angles, Rodrigues with arbitrary axis is numerically okay here,
        # but set exact identity when theta is essentially zero.
        small = theta[..., 0] < 1e-12
        if np.any(small):
            R = R.copy()
            R[small] = np.eye(3)
        return R

    # ---------------------------------------------------------------------
    # Shape / file helpers
    # ---------------------------------------------------------------------

    def get_image_size(self, K: np.ndarray) -> Tuple[int, int]:
        """
        Return image size ``(W,H)``.

        If ``self.image_size`` is not provided, infer it from the principal point:
        ``W≈2*cx`` and ``H≈2*cy``. This is a fallback; passing the true size is safer.
        """
        if self.image_size is not None:
            W, H = self.image_size
            return int(W), int(H)

        cx = np.nanmedian(K[:, 0, 2])
        cy = np.nanmedian(K[:, 1, 2])
        W = int(round(2.0 * cx))
        H = int(round(2.0 * cy))
        if W <= 0 or H <= 0:
            raise ValueError(
                "Could not infer image size from K. Pass image_size=(W,H) to FeatureCreator."
            )
        warnings.warn(
            f"image_size not provided; inferred (W,H)=({W},{H}) from principal point. "
            "Pass the true size if available."
        )
        return W, H

    def _mkdirs(self) -> None:
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def _rng_for_sequence(self, sequence: str) -> np.random.Generator:
        token = f"{self.seed}:{sequence}".encode("utf-8")
        digest = hashlib.sha256(token).hexdigest()
        seed_int = int(digest[:8], 16)
        return np.random.default_rng(seed_int)

    @staticmethod
    def _load_array_from_folder(folder: Path, sequence: str, keys: Iterable[str]) -> np.ndarray:
        npy = folder / f"{sequence}.npy"
        npz = folder / f"{sequence}.npz"
        if npy.exists():
            return np.asarray(np.load(npy))
        if npz.exists():
            data = np.load(npz)
            for key in keys:
                if key in data:
                    return np.asarray(data[key])
            if len(data.files) == 1:
                return np.asarray(data[data.files[0]])
            raise KeyError(f"Could not find any of keys {tuple(keys)} in {npz}. Keys={data.files}")
        raise FileNotFoundError(f"No .npy or .npz file found for {sequence} in {folder}")

    @staticmethod
    def _load_optional_sequence_array(folder: Path, sequence: str) -> Optional[np.ndarray]:
        npy = folder / f"{sequence}.npy"
        npz = folder / f"{sequence}.npz"
        if npy.exists():
            return np.asarray(np.load(npy, allow_pickle=True))
        if npz.exists():
            with np.load(npz, allow_pickle=True) as data:
                if "arr_0" in data.files:
                    return np.asarray(data["arr_0"])
                if len(data.files) == 1:
                    return np.asarray(data[data.files[0]])
                raise KeyError(
                    f"Ambiguous npz content for {npz}; keys={list(data.files)}"
                )
        return None

    @staticmethod
    def _truncate_k(arr: np.ndarray) -> np.ndarray:
        """Remove the last 3 columns of k when the array has distortion extras."""
        arr = np.asarray(arr)
        if arr.ndim != 2:
            return arr
        if arr.shape[1] < 3:
            return arr
        return arr[:, :-3]

    @staticmethod
    def _ensure_ntj3(arr: np.ndarray, T: int, name: str) -> np.ndarray:
        """
        Ensure joints are shaped ``(N,T,J,3)``.

        Accepts common alternatives:
        - ``(N,T,J,3)``
        - ``(T,N,J,3)``
        - ``(T,J,3)`` for a single player
        """
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            if arr.shape[1] == T:
                return arr.astype(np.float32)
            if arr.shape[0] == T:
                return arr.transpose(1, 0, 2, 3).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3 and arr.shape[0] == T:
            return arr[None].astype(np.float32)
        raise ValueError(
            f"{name}: expected (N,T,J,3), (T,N,J,3), or (T,J,3); got {arr.shape}"
        )

    @staticmethod
    def _ensure_ntjc(arr: np.ndarray, T: int, C: int, name: str) -> np.ndarray:
        """
        Ensure skeleton arrays are shaped ``(N,T,J,C)``.

        Accepts common alternatives:
        - ``(N,T,J,C)``
        - ``(T,N,J,C)``
        - ``(T,J,C)`` for a single player
        """
        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[-1] == C:
            if arr.shape[1] == T:
                return arr.astype(np.float32)
            if arr.shape[0] == T:
                return arr.transpose(1, 0, 2, 3).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == C and arr.shape[0] == T:
            return arr[None].astype(np.float32)
        raise ValueError(
            f"{name}: expected (N,T,J,{C}), (T,N,J,{C}), or (T,J,{C}); got {arr.shape}"
        )

    @staticmethod
    def _ensure_nt4(arr: np.ndarray, T: int, N: int, name: str) -> np.ndarray:
        """
        Ensure boxes are shaped ``(N,T,4)``.

        Accepts common alternatives:
        - ``(N,T,4)``
        - ``(T,N,4)``
        - ``(T,4)`` for a single player
        """
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            if arr.shape[0] == N and arr.shape[1] == T:
                return arr.astype(np.float32)
            if arr.shape[0] == T and arr.shape[1] == N:
                return arr.transpose(1, 0, 2).astype(np.float32)
        if arr.ndim == 2 and arr.shape == (T, 4) and N == 1:
            return arr[None].astype(np.float32)
        raise ValueError(
            f"{name}: expected (N,T,4), (T,N,4), or (T,4) for N=1; got {arr.shape}, expected N={N}, T={T}"
        )


if __name__ == "__main__":
    # Example usage:
    #
    #   sbatch scripts/run_feature_creation.sh
    #
    # If ps.DATA_DIR is not importable, instantiate with data_dir="/path/to/data".
    creator = FeatureCreator(
        data_dir=None,
        image_size=(1920, 1080),          # safer: set explicitly, e.g. image_size=(1920, 1080)
        pelvis_mode="hips_mean", # or "joint8"
        seed=12345,
    )
    creator.create_all(
        overwrite=True,
        rebuild_boxes_from_gt=False,
        save_individual_folders=True,
        save_npz=True,
    )
