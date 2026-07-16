from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from field_converter.training.config import CamFeatTypeStr, InputConfig, PredictionModeStr
from field_converter.training.filters import filter_valid_mask_bbox_geometry, filter_valid_mask_in_image
from field_converter.training.root_init import default_root_init_dir, load_root_init_sequence
from field_converter.training.tcn.config import PadModeStr


SplitStr = Literal["train", "valid", "test"]
NUM_PITCH_POINTS = 50


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


class NormalizedWindowDataset(Dataset[Dict[str, Any]]):
    """Temporal window dataset over normalized per-sequence .npz files.

    Each sample corresponds to one (sequence, person, window_start) triple.

    Notes
    -----
    - Inputs are concatenated per frame according to `InputConfig`.
    - Any NaN/Inf found in input features is replaced by 0.
    - The dataset returns per-frame `valid_mask` (root validity) and `valid_joints`
      for masking losses/metrics.
    - To reduce BeeGFS thrashing, it keeps a per-worker cache of the last loaded
      sequence payload and only loads required keys.
    """

    def __init__(
        self,
        *,
        data_dir: Path | str,
        split: SplitStr,
        input_config: InputConfig,
        prediction_mode: PredictionModeStr = "absolute",
        root_init_dir: Optional[Path | str] = None,
        window_size: int,
        stride: int,
        min_valid_ratio: float,
        pad_mode: PadModeStr = "edge",
        seed: int = 0,
        max_sequences: Optional[int] = None,
        max_windows_per_sequence: Optional[int] = None,
        min_in_image_joints_ratio: Optional[float] = None,
        min_bbox_width_px: Optional[float] = None,
        min_bbox_height_px: Optional[float] = None,
        min_bbox_margin_px: Optional[float] = None,
        filter_by_min_valid_ratio: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.input_config = input_config
        self.prediction_mode = prediction_mode
        if self.prediction_mode not in {"absolute", "delta"}:
            raise ValueError(f"prediction_mode must be 'absolute' or 'delta' (got {self.prediction_mode!r})")
        self.root_init_dir = Path(root_init_dir) if root_init_dir is not None else default_root_init_dir(self.data_dir)

        self.window_size = int(window_size)
        self.stride = int(stride)
        self.min_valid_ratio = float(min_valid_ratio)
        self.pad_mode = pad_mode

        self.seed = int(seed)
        self.max_sequences = max_sequences
        self.max_windows_per_sequence = max_windows_per_sequence
        self.min_in_image_joints_ratio = (
            None if min_in_image_joints_ratio is None else float(min_in_image_joints_ratio)
        )
        self.min_bbox_width_px = None if min_bbox_width_px is None else float(min_bbox_width_px)
        self.min_bbox_height_px = None if min_bbox_height_px is None else float(min_bbox_height_px)
        self.min_bbox_margin_px = None if min_bbox_margin_px is None else float(min_bbox_margin_px)
        self.filter_by_min_valid_ratio = bool(filter_by_min_valid_ratio)

        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if not (0.0 <= self.min_valid_ratio <= 1.0):
            raise ValueError("min_valid_ratio must be in [0,1]")
        if self.pad_mode not in {"edge", "zero", "none"}:
            raise ValueError("pad_mode must be one of: edge, zero, none")
        if self.min_in_image_joints_ratio is not None and not (0.0 <= self.min_in_image_joints_ratio <= 1.0):
            raise ValueError("min_in_image_joints_ratio must be in [0,1] or None")

        if self.min_bbox_width_px is not None and self.min_bbox_width_px <= 0:
            raise ValueError("min_bbox_width_px must be > 0 or None")
        if self.min_bbox_height_px is not None and self.min_bbox_height_px <= 0:
            raise ValueError("min_bbox_height_px must be > 0 or None")
        if self.min_bbox_margin_px is not None and self.min_bbox_margin_px < 0:
            raise ValueError("min_bbox_margin_px must be >= 0 or None")

        self.sequences = self._load_split_sequences(self.data_dir / "split.json", split)
        if self.max_sequences is not None:
            self.sequences = self.sequences[: int(self.max_sequences)]

        # Per-sequence metadata used by evaluators/samplers.
        self.seq_lengths: list[int] = []
        self.seq_num_persons: list[int] = []

        self.index = self._build_index()

        self._payload_keys_required = self._infer_required_payload_keys()

        self._cache_seq_idx: Optional[int] = None
        self._cache_payload: Optional[Dict[str, np.ndarray]] = None

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

    def _infer_required_payload_keys(self) -> set[str]:
        keys: set[str] = {
            # Always required for training/eval.
            "valid_mask",
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

        if self.min_in_image_joints_ratio is not None:
            keys.add("image_size")

        if (
            self.min_bbox_width_px is not None
            or self.min_bbox_height_px is not None
            or self.min_bbox_margin_px is not None
        ):
            keys.add("boxes_xyxy")
            keys.add("image_size")

        # Optional inputs.
        if self.input_config.use_x2d_img:
            keys.add("skel_2d_sam3dbody_from_bbox_gt")
        if self.input_config.use_x2d_box:
            keys.add("skel_2d_sam3dbody_from_bbox_gt_box")
        if self.input_config.use_pitch_points_2d:
            keys.update({"pitch_points_2d", "valid_pitch_points"})
        if self.input_config.use_bbox_feat:
            keys.add(_bbox_feat_key(self.input_config.bbox_clean_or_noisy))
        if self.input_config.use_cam_feat:
            keys.add(_cam_feat_key(self.input_config.cam_feat_type))
        if self.input_config.use_ground_intersection:
            keys.add("ground_intersection")

        return keys

    def _build_index(self) -> np.ndarray:
        chunks: list[np.ndarray] = []

        self.seq_lengths = [0 for _ in self.sequences]
        self.seq_num_persons = [0 for _ in self.sequences]

        for seq_idx, seq_name in enumerate(self.sequences):
            path = self._seq_path(seq_name)
            if not path.exists():
                raise FileNotFoundError(f"Missing sequence file: {path}")

            with np.load(path, allow_pickle=True) as npz:
                valid_mask = np.asarray(npz["valid_mask"], dtype=bool)  # (P,T)
                if valid_mask.ndim != 2:
                    raise ValueError(f"Expected valid_mask with shape (P,T), got {valid_mask.shape} in {path}")

                if (
                    self.min_bbox_width_px is not None
                    or self.min_bbox_height_px is not None
                    or self.min_bbox_margin_px is not None
                ):
                    boxes = np.asarray(npz["boxes_xyxy"], dtype=np.float32)
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
                    valid_joints = np.asarray(npz["valid_joints"], dtype=bool)  # (P,T,J)
                    Y_2d_gt = np.asarray(npz["Y_2d_gt"], dtype=np.float32)  # (P,T,J,2)
                    image_size = np.asarray(npz["image_size"], dtype=np.float32)
                    valid_mask = filter_valid_mask_in_image(
                        valid_mask=valid_mask,
                        valid_joints=valid_joints,
                        Y_2d_gt=Y_2d_gt,
                        image_size=image_size,
                        min_in_image_joints_ratio=float(self.min_in_image_joints_ratio),
                    )

            P, T = int(valid_mask.shape[0]), int(valid_mask.shape[1])
            self.seq_lengths[int(seq_idx)] = int(T)
            self.seq_num_persons[int(seq_idx)] = int(P)

            if T <= 0 or P <= 0:
                continue

            if T >= self.window_size:
                starts = list(range(0, T - self.window_size + 1, self.stride))
                last = T - self.window_size
                if not starts:
                    starts = [0]
                if starts[-1] != last:
                    starts.append(last)
            else:
                starts = [0]

            seq_windows: list[list[int]] = []
            for person_idx in range(P):
                for start in starts:
                    if T >= self.window_size:
                        wmask = valid_mask[person_idx, start : start + self.window_size]
                    else:
                        wmask = np.zeros((self.window_size,), dtype=bool)
                        wmask[:T] = valid_mask[person_idx, :T]

                    ratio = float(wmask.mean()) if wmask.size > 0 else 0.0
                    if self.filter_by_min_valid_ratio and ratio < self.min_valid_ratio:
                        continue

                    seq_windows.append([int(seq_idx), int(person_idx), int(start)])

            if self.max_windows_per_sequence is not None and len(seq_windows) > int(self.max_windows_per_sequence):
                rng = np.random.default_rng(self.seed ^ _stable_hash_int(seq_name))
                choice = rng.choice(len(seq_windows), size=int(self.max_windows_per_sequence), replace=False)
                seq_windows = [seq_windows[int(i)] for i in choice]

            if not seq_windows:
                continue

            chunks.append(np.asarray(seq_windows, dtype=np.int32))

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

            for k in sorted(self._payload_keys_required):
                payload[k] = npz[k]

        if self.prediction_mode == "delta":
            root_init = load_root_init_sequence(self.root_init_dir, self.split, seq_name)
            root_shape = np.asarray(payload["Y_root_cam_gt"]).shape
            if root_init.shape != root_shape:
                raise ValueError(
                    f"root_init shape mismatch for seq={seq_name}: root_init={root_init.shape}, "
                    f"Y_root_cam_gt={root_shape}"
                )
            payload["root_init_norm"] = root_init

        # Precompute the effective valid_mask used for losses/metrics.
        valid_mask = np.asarray(payload["valid_mask"], dtype=bool)

        if (
            self.min_bbox_width_px is not None
            or self.min_bbox_height_px is not None
            or self.min_bbox_margin_px is not None
        ):
            boxes = np.asarray(payload["boxes_xyxy"], dtype=np.float32)
            image_size = np.asarray(payload["image_size"], dtype=np.float32)
            valid_mask = filter_valid_mask_bbox_geometry(
                valid_mask=valid_mask,
                boxes_xyxy=boxes,
                image_size=image_size,
                min_bbox_width_px=self.min_bbox_width_px,
                min_bbox_height_px=self.min_bbox_height_px,
                min_bbox_margin_px=self.min_bbox_margin_px,
            )

        if self.min_in_image_joints_ratio is not None:
            valid_joints = np.asarray(payload["valid_joints"], dtype=bool)
            Y_2d_gt = np.asarray(payload["Y_2d_gt"], dtype=np.float32)
            image_size = np.asarray(payload["image_size"], dtype=np.float32)
            valid_mask = filter_valid_mask_in_image(
                valid_mask=valid_mask,
                valid_joints=valid_joints,
                Y_2d_gt=Y_2d_gt,
                image_size=image_size,
                min_in_image_joints_ratio=float(self.min_in_image_joints_ratio),
            )
        payload["valid_mask_used"] = valid_mask

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
        seq_idx, person_idx, frame_start = (int(v) for v in self.index[int(idx)])
        seq_name = self.sequences[seq_idx]

        payload = self._get_payload(seq_idx)

        valid_mask_used = np.asarray(payload.get("valid_mask_used", payload["valid_mask"]), dtype=bool)  # (P,T)
        P, T = int(valid_mask_used.shape[0]), int(valid_mask_used.shape[1])
        if not (0 <= person_idx < P):
            raise IndexError(f"person_idx out of bounds: {person_idx} (P={P})")

        # Frame indices for this window.
        if T >= self.window_size:
            frames = np.arange(frame_start, frame_start + self.window_size, dtype=np.int64)
            in_bounds = np.ones((self.window_size,), dtype=bool)
            frames_fetch = frames
            frame_indices = frames
        else:
            if self.pad_mode == "none":
                raise ValueError(
                    f"Sequence shorter than window_size with pad_mode='none': seq={seq_name} T={T} window_size={self.window_size}"
                )

            raw = np.arange(self.window_size, dtype=np.int64)
            in_bounds = raw < T
            frame_indices = np.where(in_bounds, raw, -1)

            if self.pad_mode == "edge":
                frames_fetch = np.clip(raw, 0, max(0, T - 1))
            else:  # zero
                frames_fetch = raw  # will be handled by explicit zero-padding below

        # ------------------------------------------------------------------
        # Gather per-frame tensors
        # ------------------------------------------------------------------

        def _alloc_zeros(shape: tuple[int, ...], dtype: Any) -> np.ndarray:
            return np.zeros(shape, dtype=dtype)

        if T >= self.window_size or self.pad_mode == "edge":
            # Edge padding uses clamped indices to fetch.
            x3d_sam = np.asarray(payload["skel_3d_sam3dbody_from_bbox_gt"][person_idx, frames_fetch], dtype=np.float32)  # (W,25,3)
            _nan_to_num_inplace(x3d_sam)

            root_gt = np.asarray(payload["Y_root_cam_gt"][person_idx, frames_fetch], dtype=np.float32)  # (W,3)
            _nan_to_num_inplace(root_gt)
            if self.prediction_mode == "delta":
                root_init = np.asarray(payload["root_init_norm"][person_idx, frames_fetch], dtype=np.float32)
                _nan_to_num_inplace(root_init)

            valid_mask = np.asarray(valid_mask_used[person_idx, frames_fetch], dtype=bool)  # (W,)
            valid_joints = np.asarray(payload["valid_joints"][person_idx, frames_fetch], dtype=bool)  # (W,25)

            K = np.asarray(payload["K"][frames_fetch], dtype=np.float32)  # (W,3,3)
            R = np.asarray(payload["R"][frames_fetch], dtype=np.float32)  # (W,3,3)
            t = np.asarray(payload["t"][frames_fetch], dtype=np.float32)  # (W,3)
            k = np.asarray(payload["k"][frames_fetch], dtype=np.float32)  # (W,2)

            Y_cam_gt = np.asarray(payload["Y_cam_gt"][person_idx, frames_fetch], dtype=np.float32)  # (W,25,3)
            Y_2d_gt = np.asarray(payload["Y_2d_gt"][person_idx, frames_fetch], dtype=np.float32)  # (W,25,2)
            _nan_to_num_inplace(Y_cam_gt)
            _nan_to_num_inplace(Y_2d_gt)

            # Mask out padded positions explicitly.
            if not in_bounds.all():
                valid_mask = valid_mask & in_bounds
                valid_joints = valid_joints & in_bounds[:, None]

                root_gt = root_gt.copy()
                root_gt[~in_bounds] = 0.0
                if self.prediction_mode == "delta":
                    root_init = root_init.copy()
                    root_init[~in_bounds] = 0.0

                Y_cam_gt = Y_cam_gt.copy()
                Y_cam_gt[~in_bounds] = 0.0

                Y_2d_gt = Y_2d_gt.copy()
                Y_2d_gt[~in_bounds] = 0.0

        else:
            # Zero-padding: fill first T frames, rest zeros.
            W = self.window_size

            x3d_sam = _alloc_zeros((W, 25, 3), np.float32)
            root_gt = _alloc_zeros((W, 3), np.float32)
            root_init = _alloc_zeros((W, 3), np.float32)
            valid_mask = _alloc_zeros((W,), bool)
            valid_joints = _alloc_zeros((W, 25), bool)

            K = _alloc_zeros((W, 3, 3), np.float32)
            R = _alloc_zeros((W, 3, 3), np.float32)
            t = _alloc_zeros((W, 3), np.float32)
            k = _alloc_zeros((W, 2), np.float32)

            Y_cam_gt = _alloc_zeros((W, 25, 3), np.float32)
            Y_2d_gt = _alloc_zeros((W, 25, 2), np.float32)

            if T > 0:
                x3d_sam[:T] = np.asarray(
                    payload["skel_3d_sam3dbody_from_bbox_gt"][person_idx, :T], dtype=np.float32
                )
                _nan_to_num_inplace(x3d_sam[:T])

                root_gt[:T] = np.asarray(payload["Y_root_cam_gt"][person_idx, :T], dtype=np.float32)
                _nan_to_num_inplace(root_gt[:T])
                if self.prediction_mode == "delta":
                    root_init[:T] = np.asarray(payload["root_init_norm"][person_idx, :T], dtype=np.float32)
                    _nan_to_num_inplace(root_init[:T])

                valid_mask[:T] = np.asarray(valid_mask_used[person_idx, :T], dtype=bool)
                valid_joints[:T] = np.asarray(payload["valid_joints"][person_idx, :T], dtype=bool)

                K[:T] = np.asarray(payload["K"][:T], dtype=np.float32)
                R[:T] = np.asarray(payload["R"][:T], dtype=np.float32)
                t[:T] = np.asarray(payload["t"][:T], dtype=np.float32)
                k[:T] = np.asarray(payload["k"][:T], dtype=np.float32)

                Y_cam_gt[:T] = np.asarray(payload["Y_cam_gt"][person_idx, :T], dtype=np.float32)
                Y_2d_gt[:T] = np.asarray(payload["Y_2d_gt"][person_idx, :T], dtype=np.float32)
                _nan_to_num_inplace(Y_cam_gt[:T])
                _nan_to_num_inplace(Y_2d_gt[:T])

        invalid_frame_mask = ~valid_mask

        def _zero_invalid_frames(x: np.ndarray) -> np.ndarray:
            if not invalid_frame_mask.any():
                return x
            x = x.copy()
            x[invalid_frame_mask] = 0.0
            return x

        # Frames rejected by dataset filters can still be present as temporal
        # context in a kept window. Zero them so invalid bbox/joint features do
        # not drive the TCN activations while losses still use valid_mask.
        if invalid_frame_mask.any():
            x3d_sam = _zero_invalid_frames(x3d_sam)
            root_gt = _zero_invalid_frames(root_gt)
            if self.prediction_mode == "delta":
                root_init = _zero_invalid_frames(root_init)
            Y_cam_gt = _zero_invalid_frames(Y_cam_gt)
            Y_2d_gt = _zero_invalid_frames(Y_2d_gt)
            valid_joints = valid_joints.copy()
            valid_joints[invalid_frame_mask] = False

        # ------------------------------------------------------------------
        # Build per-frame input matrix x: (W, D)
        # ------------------------------------------------------------------
        parts: list[np.ndarray] = []

        if self.input_config.use_x3d_sam_rel:
            parts.append(x3d_sam.reshape(self.window_size, -1))

        if self.input_config.use_x2d_img:
            if T >= self.window_size or self.pad_mode == "edge":
                x2d_img = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt"][person_idx, frames_fetch], dtype=np.float32)
                _nan_to_num_inplace(x2d_img)
                if not in_bounds.all():
                    x2d_img = x2d_img.copy()
                    x2d_img[~in_bounds] = 0.0
            else:
                x2d_img = np.zeros((self.window_size, 25, 2), dtype=np.float32)
                if T > 0:
                    x2d_img[:T] = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt"][person_idx, :T], dtype=np.float32)
                    _nan_to_num_inplace(x2d_img[:T])
            x2d_img = _zero_invalid_frames(x2d_img)
            parts.append(x2d_img.reshape(self.window_size, -1))

        if self.input_config.use_x2d_box:
            if T >= self.window_size or self.pad_mode == "edge":
                x2d_box = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt_box"][person_idx, frames_fetch], dtype=np.float32)
                _nan_to_num_inplace(x2d_box)
                if not in_bounds.all():
                    x2d_box = x2d_box.copy()
                    x2d_box[~in_bounds] = 0.0
            else:
                x2d_box = np.zeros((self.window_size, 25, 2), dtype=np.float32)
                if T > 0:
                    x2d_box[:T] = np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt_box"][person_idx, :T], dtype=np.float32)
                    _nan_to_num_inplace(x2d_box[:T])
            x2d_box = _zero_invalid_frames(x2d_box)
            parts.append(x2d_box.reshape(self.window_size, -1))

        if self.input_config.use_pitch_points_2d:
            if T >= self.window_size or self.pad_mode == "edge":
                pitch_2d = np.asarray(payload["pitch_points_2d"][frames_fetch], dtype=np.float32)
                valid_pitch = np.asarray(payload["valid_pitch_points"][frames_fetch], dtype=bool)
                _nan_to_num_inplace(pitch_2d)
                if not in_bounds.all():
                    pitch_2d = pitch_2d.copy()
                    valid_pitch = valid_pitch.copy()
                    pitch_2d[~in_bounds] = 0.0
                    valid_pitch[~in_bounds] = False
            else:
                pitch_2d = np.zeros((self.window_size, NUM_PITCH_POINTS, 2), dtype=np.float32)
                valid_pitch = np.zeros((self.window_size, NUM_PITCH_POINTS), dtype=bool)
                if T > 0:
                    pitch_2d[:T] = np.asarray(payload["pitch_points_2d"][:T], dtype=np.float32)
                    valid_pitch[:T] = np.asarray(payload["valid_pitch_points"][:T], dtype=bool)
                    _nan_to_num_inplace(pitch_2d[:T])

            if pitch_2d.shape != (self.window_size, NUM_PITCH_POINTS, 2) or valid_pitch.shape != (
                self.window_size,
                NUM_PITCH_POINTS,
            ):
                raise ValueError(
                    f"Expected {NUM_PITCH_POINTS} pitch points for seq={seq_name}; "
                    f"got pitch_points_2d={pitch_2d.shape}, valid_pitch_points={valid_pitch.shape}"
                )
            pitch_2d = _zero_invalid_frames(pitch_2d)
            valid_pitch = _zero_invalid_frames(valid_pitch)
            parts.append(pitch_2d.reshape(self.window_size, -1))
            parts.append(valid_pitch.astype(np.float32))

        if self.input_config.use_bbox_feat:
            bbox_key = _bbox_feat_key(self.input_config.bbox_clean_or_noisy)
            if T >= self.window_size or self.pad_mode == "edge":
                bbox_feat = np.asarray(payload[bbox_key][person_idx, frames_fetch], dtype=np.float32)
                _nan_to_num_inplace(bbox_feat)
                if not in_bounds.all():
                    bbox_feat = bbox_feat.copy()
                    bbox_feat[~in_bounds] = 0.0
            else:
                bbox_feat = np.zeros((self.window_size, 5), dtype=np.float32)
                if T > 0:
                    bbox_feat[:T] = np.asarray(payload[bbox_key][person_idx, :T], dtype=np.float32)
                    _nan_to_num_inplace(bbox_feat[:T])
            bbox_feat = _zero_invalid_frames(bbox_feat)
            parts.append(bbox_feat.reshape(self.window_size, -1))

        if self.input_config.use_cam_feat:
            cam_key = _cam_feat_key(self.input_config.cam_feat_type)
            cam_dim = 6 if self.input_config.cam_feat_type.startswith("base_") else 12
            if T >= self.window_size or self.pad_mode == "edge":
                cam_feat = np.asarray(payload[cam_key][frames_fetch], dtype=np.float32)
                _nan_to_num_inplace(cam_feat)
                if not in_bounds.all():
                    cam_feat = cam_feat.copy()
                    cam_feat[~in_bounds] = 0.0
            else:
                cam_feat = np.zeros((self.window_size, cam_dim), dtype=np.float32)
                if T > 0:
                    cam_feat[:T] = np.asarray(payload[cam_key][:T], dtype=np.float32)
                    _nan_to_num_inplace(cam_feat[:T])
            cam_feat = _zero_invalid_frames(cam_feat)
            parts.append(cam_feat.reshape(self.window_size, -1))

        if self.input_config.use_ground_intersection:
            if T >= self.window_size or self.pad_mode == "edge":
                ground = np.asarray(payload["ground_intersection"][person_idx, frames_fetch], dtype=np.float32)
                _nan_to_num_inplace(ground)
                if not in_bounds.all():
                    ground = ground.copy()
                    ground[~in_bounds] = 0.0
            else:
                ground = np.zeros((self.window_size, 3), dtype=np.float32)
                if T > 0:
                    ground[:T] = np.asarray(payload["ground_intersection"][person_idx, :T], dtype=np.float32)
                    _nan_to_num_inplace(ground[:T])
            ground = _zero_invalid_frames(ground)
            parts.append(ground.reshape(self.window_size, -1))

        if self.input_config.use_valid_joints_as_input:
            parts.append(valid_joints.astype(np.float32).reshape(self.window_size, -1))

        if not parts:
            raise RuntimeError("No input parts selected")

        x = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
        _nan_to_num_inplace(x)

        sample: Dict[str, Any] = {
            "x": torch.from_numpy(x),
            "root_gt": torch.from_numpy(root_gt),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.bool_)),
            "valid_joints": torch.from_numpy(valid_joints.astype(np.bool_)),
            "x3d_sam_norm": torch.from_numpy(x3d_sam),
            "K": torch.from_numpy(K),
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),
            "k": torch.from_numpy(k),
            "Y_cam_gt": torch.from_numpy(Y_cam_gt),
            "Y_2d_gt": torch.from_numpy(Y_2d_gt),
            "seq_name": seq_name,
            "person_idx": person_idx,
            "frame_start": frame_start,
            "frame_indices": torch.from_numpy(frame_indices.astype(np.int64)),
        }
        if self.prediction_mode == "delta":
            sample["root_init_norm"] = torch.from_numpy(root_init)

        return sample
