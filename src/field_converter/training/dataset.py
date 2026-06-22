from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from field_converter.training.config import CamFeatTypeStr, InputConfig
from field_converter.training.filters import filter_valid_mask_bbox_geometry, filter_valid_mask_in_image


SplitStr = Literal["train", "valid", "test"]


def infer_input_dim(cfg: InputConfig) -> int:
    dim = 0
    if cfg.use_x3d_sam_rel:
        dim += 25 * 3
    if cfg.use_x2d_img:
        dim += 25 * 2
    if cfg.use_x2d_box:
        dim += 25 * 2
    if cfg.use_bbox_feat:
        dim += 5
    if cfg.use_cam_feat:
        dim += 6 if cfg.cam_feat_type.startswith("base_") else 12
    if cfg.use_ground_intersection:
        dim += 3
    if cfg.use_valid_joints_as_input:
        dim += 25
    return dim


def _stable_hash_int(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def _nan_to_num_inplace(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _cam_feat_key(cam_feat_type: CamFeatTypeStr) -> str:
    mapping = {
        "base_clean": "cam_feat_base_clean",
        "base_noisy": "cam_feat_base_noisy",
        "boosted_clean": "cam_feat_boosted_clean",
        "boosted_noisy": "cam_feat_boosted_noisy",
    }
    return mapping[cam_feat_type]


def _bbox_feat_key(clean_or_noisy: Literal["clean", "noisy"]) -> str:
    return "bbox_feat_clean" if clean_or_noisy == "clean" else "bbox_feat"


class NormalizedFrameDataset(Dataset[Dict[str, Any]]):
    """Frame-wise dataset over normalized per-sequence .npz files.

    Each sample corresponds to one valid (person, frame) pair where `valid_mask=True`.

    Notes
    -----
    - Inputs are flattened per-frame according to `InputConfig`.
    - Any NaN/Inf found in input features is replaced by 0. Masks are returned
      separately (e.g. `valid_joints`).
    - The dataset returns additional fields required for evaluation/visualization.
    """

    def __init__(
        self,
        *,
        data_dir: Path | str,
        split: SplitStr,
        input_config: InputConfig,
        seed: int = 0,
        max_sequences: Optional[int] = None,
        max_samples_per_sequence: Optional[int] = None,
        subsample_stride: int = 1,
        min_in_image_joints_ratio: Optional[float] = None,
        min_bbox_width_px: Optional[float] = None,
        min_bbox_height_px: Optional[float] = None,
        min_bbox_margin_px: Optional[float] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_config = input_config
        self.seed = int(seed)
        self.max_sequences = max_sequences
        self.max_samples_per_sequence = max_samples_per_sequence
        self.subsample_stride = int(subsample_stride)
        self.min_in_image_joints_ratio = (
            None if min_in_image_joints_ratio is None else float(min_in_image_joints_ratio)
        )
        self.min_bbox_width_px = None if min_bbox_width_px is None else float(min_bbox_width_px)
        self.min_bbox_height_px = None if min_bbox_height_px is None else float(min_bbox_height_px)
        self.min_bbox_margin_px = None if min_bbox_margin_px is None else float(min_bbox_margin_px)

        if self.min_in_image_joints_ratio is not None and not (0.0 <= self.min_in_image_joints_ratio <= 1.0):
            raise ValueError("min_in_image_joints_ratio must be in [0,1] or None")

        if self.min_bbox_width_px is not None and self.min_bbox_width_px <= 0:
            raise ValueError("min_bbox_width_px must be > 0 or None")
        if self.min_bbox_height_px is not None and self.min_bbox_height_px <= 0:
            raise ValueError("min_bbox_height_px must be > 0 or None")
        if self.min_bbox_margin_px is not None and self.min_bbox_margin_px < 0:
            raise ValueError("min_bbox_margin_px must be >= 0 or None")

        if self.subsample_stride < 1:
            raise ValueError("subsample_stride must be >= 1")

        self.sequences = self._load_split_sequences(self.data_dir / "split.json", split)
        if self.max_sequences is not None:
            self.sequences = self.sequences[: int(self.max_sequences)]

        self.index = self._build_index()

        # Loading entire per-sequence .npz payloads in each DataLoader worker can
        # blow up CPU RAM. We therefore only load the keys actually needed by
        # __getitem__ for the current InputConfig.
        self._payload_keys_required = self._infer_required_payload_keys()

        self._cache_seq_idx: Optional[int] = None
        self._cache_payload: Optional[Dict[str, np.ndarray]] = None

    def _infer_required_payload_keys(self) -> set[str]:
        keys: set[str] = {
            # Always required for training/eval and visualization.
            "valid_joints",
            "skel_3d_sam3dbody_from_bbox_gt",
            "K",
            "R",
            "t",
            "k",
            "Y_cam_gt",
            "Y_2d_gt",
            "Y_root_cam_gt",
        }

        # Optional inputs.
        if self.input_config.use_x2d_img:
            keys.add("skel_2d_sam3dbody_from_bbox_gt")
        if self.input_config.use_x2d_box:
            keys.add("skel_2d_sam3dbody_from_bbox_gt_box")
        if self.input_config.use_bbox_feat:
            keys.add(_bbox_feat_key(self.input_config.bbox_clean_or_noisy))
        if self.input_config.use_cam_feat:
            keys.add(_cam_feat_key(self.input_config.cam_feat_type))
        if self.input_config.use_ground_intersection:
            keys.add("ground_intersection")

        # Common optional fields used for visualization/debug.
        keys.update({"image_size", "boxes_xyxy"})
        return keys

    # ------------------------------------------------------------------
    # Split/indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_split_sequences(split_json_path: Path, split: SplitStr) -> list[str]:
        if not split_json_path.exists():
            raise FileNotFoundError(f"Missing split.json: {split_json_path}")
        payload = json.loads(split_json_path.read_text(encoding="utf-8"))
        seqs = payload.get(split)
        if not isinstance(seqs, list) or not all(isinstance(s, str) for s in seqs):
            raise ValueError(f"Invalid split.json content for split={split}")
        return list(seqs)

    def _seq_path(self, seq_name: str) -> Path:
        return self.data_dir / self.split / f"{seq_name}.npz"

    def _build_index(self) -> np.ndarray:
        chunks: list[np.ndarray] = []

        for seq_idx, seq_name in enumerate(self.sequences):
            path = self._seq_path(seq_name)
            if not path.exists():
                raise FileNotFoundError(f"Missing sequence file: {path}")

            with np.load(path, allow_pickle=True) as npz:
                valid_mask = np.asarray(npz["valid_mask"], dtype=bool)

                if (
                    self.min_bbox_width_px is not None
                    or self.min_bbox_height_px is not None
                    or self.min_bbox_margin_px is not None
                ):
                    boxes = np.asarray(npz["boxes_xyxy"], dtype=np.float32)  # (N,T,4)
                    image_size = np.asarray(npz["image_size"], dtype=np.float32)
                    valid_mask = filter_valid_mask_bbox_geometry(
                        valid_mask=valid_mask,
                        boxes_xyxy=boxes,
                        image_size=image_size,
                        min_bbox_width_px=self.min_bbox_width_px,
                        min_bbox_height_px=self.min_bbox_height_px,
                        min_bbox_margin_px=self.min_bbox_margin_px,
                    )

                if self.min_in_image_joints_ratio is not None:
                    valid_joints = np.asarray(npz["valid_joints"], dtype=bool)  # (N,T,J)
                    Y_2d_gt = np.asarray(npz["Y_2d_gt"], dtype=np.float32)  # (N,T,J,2)
                    image_size = np.asarray(npz["image_size"], dtype=np.float32)
                    valid_mask = filter_valid_mask_in_image(
                        valid_mask=valid_mask,
                        valid_joints=valid_joints,
                        Y_2d_gt=Y_2d_gt,
                        image_size=image_size,
                        min_in_image_joints_ratio=float(self.min_in_image_joints_ratio),
                    )

            # pf: (K,2) with columns (person_idx, frame_idx)
            pf = np.argwhere(valid_mask)

            if self.subsample_stride > 1:
                pf = pf[:: self.subsample_stride]

            if self.max_samples_per_sequence is not None and pf.shape[0] > self.max_samples_per_sequence:
                rng = np.random.default_rng(self.seed ^ _stable_hash_int(seq_name))
                choice = rng.choice(pf.shape[0], size=int(self.max_samples_per_sequence), replace=False)
                pf = pf[choice]

            if pf.size == 0:
                continue

            seq_col = np.full((pf.shape[0], 1), int(seq_idx), dtype=np.int32)
            triples = np.concatenate([seq_col, pf.astype(np.int32)], axis=1)  # (K,3)
            chunks.append(triples)

        if not chunks:
            return np.zeros((0, 3), dtype=np.int32)

        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_payload(self, seq_idx: int) -> Dict[str, np.ndarray]:
        seq_name = self.sequences[int(seq_idx)]
        path = self._seq_path(seq_name)
        payload: Dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=True) as npz:
            available = set(npz.files)
            missing_required = sorted(k for k in self._payload_keys_required if k not in available)
            if missing_required:
                raise KeyError(
                    "Missing required keys in sequence payload: "
                    f"seq={seq_name} split={self.split} missing={missing_required}"
                )

            # Only load the keys we actually need to keep per-worker RAM bounded.
            for k in sorted(self._payload_keys_required):
                if k in available:
                    payload[k] = npz[k]
        return payload

    def _get_payload(self, seq_idx: int) -> Dict[str, np.ndarray]:
        if self._cache_seq_idx != int(seq_idx) or self._cache_payload is None:
            self._cache_seq_idx = int(seq_idx)
            self._cache_payload = self._load_payload(int(seq_idx))
        return self._cache_payload

    # ------------------------------------------------------------------
    # PyTorch Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401
        return int(self.index.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_idx, person_idx, frame_idx = (int(v) for v in self.index[int(idx)])
        seq_name = self.sequences[seq_idx]

        payload = self._get_payload(seq_idx)

        valid_joints_np = np.asarray(payload["valid_joints"][person_idx, frame_idx], dtype=bool)  # (25,)
        valid_joints = torch.from_numpy(valid_joints_np)

        # Core tensors needed for eval/visualization.
        x3d_sam = np.asarray(payload["skel_3d_sam3dbody_from_bbox_gt"][person_idx, frame_idx], dtype=np.float32)  # (25,3)
        _nan_to_num_inplace(x3d_sam)

        K = np.asarray(payload["K"][frame_idx], dtype=np.float32)
        R = np.asarray(payload["R"][frame_idx], dtype=np.float32)
        t = np.asarray(payload["t"][frame_idx], dtype=np.float32)
        k = np.asarray(payload["k"][frame_idx], dtype=np.float32)

        Y_cam_gt = np.asarray(payload["Y_cam_gt"][person_idx, frame_idx], dtype=np.float32)  # (25,3)
        Y_2d_gt = np.asarray(payload["Y_2d_gt"][person_idx, frame_idx], dtype=np.float32)  # (25,2)
        _nan_to_num_inplace(Y_cam_gt)
        _nan_to_num_inplace(Y_2d_gt)

        root_gt = np.asarray(payload["Y_root_cam_gt"][person_idx, frame_idx], dtype=np.float32)  # (3,) normalized
        _nan_to_num_inplace(root_gt)

        # ------------------------------------------------------------------
        # Build input vector x
        # ------------------------------------------------------------------
        parts: list[np.ndarray] = []

        if self.input_config.use_x3d_sam_rel:
            parts.append(x3d_sam.reshape(-1))

        if self.input_config.use_x2d_img:
            x2d_img = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt"][person_idx, frame_idx], dtype=np.float32)  # (25,2)
            _nan_to_num_inplace(x2d_img)
            parts.append(x2d_img.reshape(-1))

        if self.input_config.use_x2d_box:
            x2d_box = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt_box"][person_idx, frame_idx], dtype=np.float32)  # (25,2)
            _nan_to_num_inplace(x2d_box)
            parts.append(x2d_box.reshape(-1))

        if self.input_config.use_bbox_feat:
            bbox_key = _bbox_feat_key(self.input_config.bbox_clean_or_noisy)
            bbox_feat = np.asarray(payload[bbox_key][person_idx, frame_idx], dtype=np.float32)  # (5,)
            _nan_to_num_inplace(bbox_feat)
            parts.append(bbox_feat.reshape(-1))

        if self.input_config.use_cam_feat:
            cam_key = _cam_feat_key(self.input_config.cam_feat_type)
            cam_feat = np.asarray(payload[cam_key][frame_idx], dtype=np.float32)  # (6|12,)
            _nan_to_num_inplace(cam_feat)
            parts.append(cam_feat.reshape(-1))

        if self.input_config.use_ground_intersection:
            ground = np.asarray(payload["ground_intersection"][person_idx, frame_idx], dtype=np.float32)  # (3,)
            _nan_to_num_inplace(ground)
            parts.append(ground.reshape(-1))

        if self.input_config.use_valid_joints_as_input:
            parts.append(valid_joints_np.astype(np.float32).reshape(-1))

        if not parts:
            raise RuntimeError("No input parts selected")

        x = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        _nan_to_num_inplace(x)

        sample: Dict[str, Any] = {
            "x": torch.from_numpy(x),
            "root_gt": torch.from_numpy(root_gt),
            "seq_name": seq_name,
            "person_idx": person_idx,
            "frame_idx": frame_idx,
            "valid_joints": valid_joints,
            "x3d_sam_norm": torch.from_numpy(x3d_sam),
            "K": torch.from_numpy(K),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),
            "k": torch.from_numpy(k),
            "Y_cam_gt": torch.from_numpy(Y_cam_gt),
            "Y_2d_gt": torch.from_numpy(Y_2d_gt),
        }

        # Optional fields often useful for visualization.
        if "image_size" in payload:
            sample["image_size"] = torch.from_numpy(np.asarray(payload["image_size"], dtype=np.int64))
        if "boxes_xyxy" in payload:
            boxes = np.asarray(payload["boxes_xyxy"][person_idx, frame_idx], dtype=np.float32)
            _nan_to_num_inplace(boxes)
            sample["boxes_xyxy"] = torch.from_numpy(boxes)

        if "skel_2d_sam3dbody_from_bbox_gt" in payload:
            x2d_img = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt"][person_idx, frame_idx], dtype=np.float32)
            _nan_to_num_inplace(x2d_img)
            sample["x2d_sam_img_norm"] = torch.from_numpy(x2d_img)

        return sample
