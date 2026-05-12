"""field_converter.training.data_preparation.normalize

Create a normalized copy of consolidated feature files under ``data/features``.

This module builds ``data/features_normalized/{train,valid,test}`` and computes
normalization statistics **only on the training split** to avoid data leakage.

Main normalizations implemented (as requested):

- 2D SAM skeleton (pixels) ->
	- image-normalized: ``x/W``, ``y/H``
	- box-normalized: ``(x-cx_box)/w_box``, ``(y-cy_box)/h_box``
- 3D SAM skeleton -> pelvis centered ("hips_mean" or "joint8"), then standardized
  with train mean/std.
- ``Y_rel_cam_gt`` standardized with train mean/std, with de-normalization helper.
- ``Y_root_cam_gt`` standardized with train mean/std, with de-normalization helper.
- camera center ``C`` (meters) standardized with train mean/std; camera forward is
  already unit length and is left untouched.
- bbox feature ratio: replace ``w/h`` by ``log(w/h)`` (ratios often have more stable
  distributions in log space).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

import numpy as np

# Allow running this script without installing the package (e.g. without
# `pip install -e .`). When `field_converter` is importable, we reuse its
# pathseeker; otherwise we infer the project root from this file location.

from field_converter import pathseeker as ps  

_DEFAULT_DATA_DIR = ps.DATA_DIR




PelvisMode = Literal["hips_mean", "joint8"]


def _load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
	payload: Dict[str, np.ndarray] = {}
	with np.load(path, allow_pickle=True) as npz:
		for key in npz.files:
			payload[key] = npz[key]
	return payload


def _save_npz_atomic(path: Path, payload: Dict[str, np.ndarray]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = path.with_name(f"{path.stem}.tmp.npz")
	np.savez_compressed(tmp_path, **payload)
	tmp_path.replace(path)


def _safe_std_from_sums(
	sum_: np.ndarray,
	sumsq: np.ndarray,
	count: np.ndarray,
	eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute mean/std from sum/sumsq/count with numerical safeguards."""
	count_safe = np.maximum(count, 1.0)
	mean = sum_ / count_safe
	var = sumsq / count_safe - mean**2
	var = np.maximum(var, eps)
	std = np.sqrt(var)
	return mean, std


def _ensure_ntjc(
	arr: np.ndarray,
	T: int,
	name: str,
) -> Tuple[np.ndarray, bool]:
	"""Ensure array is in (N,T,J,C) order.

	Some arrays (SAM skeletons) are stored as (T,N,J,C). This helper converts them
	to (N,T,J,C) and returns whether a transpose was applied.
	"""
	if arr.ndim != 4:
		raise ValueError(f"{name}: expected 4D array, got shape {arr.shape}")

	if arr.shape[0] == T and arr.shape[1] != T:
		# (T,N,J,C) -> (N,T,J,C)
		return arr.transpose(1, 0, 2, 3), True

	return arr, False


@dataclass(frozen=True)
class NormalizationStats:
	pelvis_mode: PelvisMode

	mean_sam3d_rel: np.ndarray  # (J,3)
	std_sam3d_rel: np.ndarray   # (J,3)

	mean_y_rel: np.ndarray      # (J,3)
	std_y_rel: np.ndarray       # (J,3)

	mean_root: np.ndarray       # (3,)
	std_root: np.ndarray        # (3,)

	mean_C: np.ndarray          # (3,)
	std_C: np.ndarray           # (3,)

	train_sequences: Tuple[str, ...]
	valid_sequences: Tuple[str, ...]
	test_sequences: Tuple[str, ...]
	seed: int

	def save(self, out_dir: Path) -> None:
		out_dir.mkdir(parents=True, exist_ok=True)

		npz_path = out_dir / "normalization_stats.npz"
		np.savez_compressed(
			npz_path,
			pelvis_mode=np.array(self.pelvis_mode, dtype=object),
			mean_sam3d_rel=self.mean_sam3d_rel.astype(np.float32),
			std_sam3d_rel=self.std_sam3d_rel.astype(np.float32),
			mean_y_rel=self.mean_y_rel.astype(np.float32),
			std_y_rel=self.std_y_rel.astype(np.float32),
			mean_root=self.mean_root.astype(np.float32),
			std_root=self.std_root.astype(np.float32),
			mean_C=self.mean_C.astype(np.float32),
			std_C=self.std_C.astype(np.float32),
			train_sequences=np.array(self.train_sequences, dtype=object),
			valid_sequences=np.array(self.valid_sequences, dtype=object),
			test_sequences=np.array(self.test_sequences, dtype=object),
			seed=np.array(self.seed, dtype=np.int64),
		)

		json_path = out_dir / "normalization_stats.json"
		json_payload = {
			"pelvis_mode": self.pelvis_mode,
			"seed": int(self.seed),
			"train_sequences": list(self.train_sequences),
			"valid_sequences": list(self.valid_sequences),
			"test_sequences": list(self.test_sequences),
			"mean_sam3d_rel": self.mean_sam3d_rel.tolist(),
			"std_sam3d_rel": self.std_sam3d_rel.tolist(),
			"mean_y_rel": self.mean_y_rel.tolist(),
			"std_y_rel": self.std_y_rel.tolist(),
			"mean_root": self.mean_root.tolist(),
			"std_root": self.std_root.tolist(),
			"mean_C": self.mean_C.tolist(),
			"std_C": self.std_C.tolist(),
		}
		json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

	@staticmethod
	def load(npz_path: Path) -> "NormalizationStats":
		with np.load(npz_path, allow_pickle=True) as npz:
			pelvis_mode = str(npz["pelvis_mode"].item())
			return NormalizationStats(
				pelvis_mode=pelvis_mode,  # type: ignore[assignment]
				mean_sam3d_rel=np.asarray(npz["mean_sam3d_rel"], dtype=np.float64),
				std_sam3d_rel=np.asarray(npz["std_sam3d_rel"], dtype=np.float64),
				mean_y_rel=np.asarray(npz["mean_y_rel"], dtype=np.float64),
				std_y_rel=np.asarray(npz["std_y_rel"], dtype=np.float64),
				mean_root=np.asarray(npz["mean_root"], dtype=np.float64),
				std_root=np.asarray(npz["std_root"], dtype=np.float64),
				mean_C=np.asarray(npz["mean_C"], dtype=np.float64),
				std_C=np.asarray(npz["std_C"], dtype=np.float64),
				train_sequences=tuple(npz["train_sequences"].tolist()),
				valid_sequences=tuple(npz["valid_sequences"].tolist()),
				test_sequences=tuple(npz["test_sequences"].tolist()),
				seed=int(npz["seed"].item()),
			)


class Normalizer:
	def __init__(
		self,
		*,
		data_dir: Optional[Path | str] = None,
		in_features_dirname: str = "features",
		out_features_dirname: str = "features_normalized",
		sequences_file: str = "sequences_gt.txt",
		pelvis_mode: PelvisMode = "hips_mean",
		eps: float = 1e-8,
	) -> None:
		self.data_dir = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
		self.in_dir = self.data_dir / in_features_dirname
		self.out_dir = self.data_dir / out_features_dirname
		self.sequences_file = sequences_file
		self.pelvis_mode = pelvis_mode
		self.eps = float(eps)

	# ------------------------------------------------------------------
	# Split
	# ------------------------------------------------------------------

	def read_sequences(self) -> list[str]:
		path = self.data_dir / self.sequences_file
		if not path.exists():
			raise FileNotFoundError(f"Missing sequences file: {path}")
		sequences = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
		sequences = [s for s in sequences if s]
		return sequences

	@staticmethod
	def split_sequences(
		sequences: list[str],
		train_n: int,
		valid_n: int,
		test_n: int,
		seed: int,
	) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
		if train_n < 0 or valid_n < 0 or test_n < 0:
			raise ValueError("Split sizes must be non-negative")
		total = train_n + valid_n + test_n
		if total != len(sequences):
			raise ValueError(
				f"Split sizes must sum to {len(sequences)} (got {train_n}+{valid_n}+{test_n}={total})"
			)

		rng = np.random.default_rng(seed)
		perm = rng.permutation(len(sequences)).tolist()
		seq_perm = [sequences[i] for i in perm]

		train = tuple(seq_perm[:train_n])
		valid = tuple(seq_perm[train_n : train_n + valid_n])
		test = tuple(seq_perm[train_n + valid_n :])
		return train, valid, test

	# ------------------------------------------------------------------
	# Core math helpers
	# ------------------------------------------------------------------

	@staticmethod
	def compute_pelvis(X: np.ndarray, mode: PelvisMode) -> np.ndarray:
		"""Compute pelvis from joints array shaped (..., J, 3)."""
		if X.shape[-2] < 13 or X.shape[-1] != 3:
			raise ValueError(f"Expected (...,J,3) with J>=13, got {X.shape}")
		if mode == "hips_mean":
			return 0.5 * (X[..., 9, :] + X[..., 12, :])
		if mode == "joint8":
			return X[..., 8, :]
		raise ValueError(f"Unknown pelvis mode: {mode}")

	def _bbox_log_ratio_inplace(self, bbox_feat: np.ndarray) -> np.ndarray:
		"""Replace w/h by log(w/h) in-place (returns same array)."""
		if bbox_feat.shape[-1] != 5:
			return bbox_feat
		ratio = bbox_feat[..., 4]
		bbox_feat[..., 4] = np.log(np.maximum(ratio, self.eps))
		return bbox_feat

	# ------------------------------------------------------------------
	# Stats computation (train only)
	# ------------------------------------------------------------------

	def compute_train_stats(
		self,
		train_sequences: Iterable[str],
		*,
		seed: int,
		valid_sequences: Tuple[str, ...],
		test_sequences: Tuple[str, ...],
	) -> NormalizationStats:
		sum_sam = np.zeros((25, 3), dtype=np.float64)
		sumsq_sam = np.zeros((25, 3), dtype=np.float64)
		count_sam = np.zeros((25, 1), dtype=np.float64)

		sum_rel = np.zeros((25, 3), dtype=np.float64)
		sumsq_rel = np.zeros((25, 3), dtype=np.float64)
		count_rel = np.zeros((25, 1), dtype=np.float64)

		sum_root = np.zeros((3,), dtype=np.float64)
		sumsq_root = np.zeros((3,), dtype=np.float64)
		count_root = 0.0

		sum_C = np.zeros((3,), dtype=np.float64)
		sumsq_C = np.zeros((3,), dtype=np.float64)
		count_C = 0.0

		train_sequences = tuple(train_sequences)

		for seq in train_sequences:
			path = self.in_dir / f"{seq}.npz"
			payload = _load_npz_payload(path)

			# Basic shapes
			T = int(payload["K"].shape[0])
			valid_mask = np.asarray(payload.get("valid_mask"), dtype=bool)  # (N,T)
			valid_joints = np.asarray(payload.get("valid_joints"), dtype=bool)  # (N,T,J)

			# --- SAM 3D: pelvis center then stats ---
			X_sam3d = np.asarray(payload["skel_3d_sam3dbody_from_bbox_gt"], dtype=np.float64)
			X_sam3d_ntjc, transposed = _ensure_ntjc(X_sam3d, T=T, name=f"{seq} skel_3d")
			# Apply GT valid mask to avoid counting missing player-frames.
			if valid_mask.shape[:2] == X_sam3d_ntjc.shape[:2]:
				frame_mask = valid_mask[:, :, None]  # (N,T,1)
			else:
				frame_mask = np.ones(X_sam3d_ntjc.shape[:2] + (1,), dtype=bool)

			pelvis = self.compute_pelvis(X_sam3d_ntjc, mode=self.pelvis_mode)  # (N,T,3)
			X_sam3d_rel = X_sam3d_ntjc - pelvis[:, :, None, :]

			finite = np.isfinite(X_sam3d_rel).all(axis=-1)  # (N,T,J)
			m = frame_mask & finite  # (N,T,J)
			if np.any(m):
				# IMPORTANT: do not multiply by the mask, because NaN * 0 == NaN in numpy.
				# Use np.where to zero-out invalid entries without contaminating sums.
				x = np.where(m[..., None], X_sam3d_rel, 0.0)
				sum_sam += x.sum(axis=(0, 1))
				sumsq_sam += (x**2).sum(axis=(0, 1))
				count_sam += m.sum(axis=(0, 1))[:, None]

			# --- Y_rel_cam_gt ---
			Y_rel = np.asarray(payload["Y_rel_cam_gt"], dtype=np.float64)  # (N,T,J,3)
			finite_rel = np.isfinite(Y_rel).all(axis=-1)
			if valid_joints.shape == finite_rel.shape:
				m_rel = valid_joints & finite_rel
			else:
				m_rel = finite_rel
			if valid_mask.shape == m_rel.shape[:2]:
				m_rel = m_rel & valid_mask[:, :, None]

			if np.any(m_rel):
				y = np.where(m_rel[..., None], Y_rel, 0.0)
				sum_rel += y.sum(axis=(0, 1))
				sumsq_rel += (y**2).sum(axis=(0, 1))
				count_rel += m_rel.sum(axis=(0, 1))[:, None]

			# --- Y_root_cam_gt ---
			root = np.asarray(payload["Y_root_cam_gt"], dtype=np.float64)  # (N,T,3)
			finite_root = np.isfinite(root).all(axis=-1)
			m_root = finite_root
			if valid_mask.shape == m_root.shape:
				m_root = m_root & valid_mask
			if np.any(m_root):
				r = np.where(m_root[..., None], root, 0.0)
				sum_root += r.sum(axis=(0, 1))
				sumsq_root += (r**2).sum(axis=(0, 1))
				count_root += float(m_root.sum())

			# --- Camera center C (from boosted features) ---
			cam_boost = np.asarray(payload["cam_feat_boosted_clean"], dtype=np.float64)  # (T,12)
			C = cam_boost[:, 6:9]
			finite_C = np.isfinite(C).all(axis=-1)
			if np.any(finite_C):
				sum_C += C[finite_C].sum(axis=0)
				sumsq_C += (C[finite_C] ** 2).sum(axis=0)
				count_C += float(finite_C.sum())

		mean_sam, std_sam = _safe_std_from_sums(sum_sam, sumsq_sam, count_sam, eps=self.eps)
		mean_rel, std_rel = _safe_std_from_sums(sum_rel, sumsq_rel, count_rel, eps=self.eps)
		mean_root, std_root = _safe_std_from_sums(sum_root, sumsq_root, np.array(count_root), eps=self.eps)
		mean_C, std_C = _safe_std_from_sums(sum_C, sumsq_C, np.array(count_C), eps=self.eps)

		return NormalizationStats(
			pelvis_mode=self.pelvis_mode,
			mean_sam3d_rel=mean_sam,
			std_sam3d_rel=std_sam,
			mean_y_rel=mean_rel,
			std_y_rel=std_rel,
			mean_root=np.asarray(mean_root, dtype=np.float64),
			std_root=np.asarray(std_root, dtype=np.float64),
			mean_C=np.asarray(mean_C, dtype=np.float64),
			std_C=np.asarray(std_C, dtype=np.float64),
			train_sequences=train_sequences,
			valid_sequences=valid_sequences,
			test_sequences=test_sequences,
			seed=int(seed),
		)

	# ------------------------------------------------------------------
	# De-normalization helpers (for inference)
	# ------------------------------------------------------------------

	@staticmethod
	def denormalize_y_rel_cam(
		Y_rel_cam_pred_norm: np.ndarray,
		*,
		mean_rel: np.ndarray,
		std_rel: np.ndarray,
	) -> np.ndarray:
		"""Y_rel_cam_pred = Y_rel_cam_pred_norm * std_rel + mean_rel."""
		return Y_rel_cam_pred_norm * std_rel[None, None, :, :] + mean_rel[None, None, :, :]

	@staticmethod
	def denormalize_root_cam(
		root_cam_pred_norm: np.ndarray,
		*,
		mean_root: np.ndarray,
		std_root: np.ndarray,
	) -> np.ndarray:
		"""root_cam_pred = root_cam_pred_norm * std_root + mean_root."""
		return root_cam_pred_norm * std_root[None, None, :] + mean_root[None, None, :]

	# ------------------------------------------------------------------
	# Sequence normalization
	# ------------------------------------------------------------------

	def normalize_sequence(self, sequence: str, stats: NormalizationStats) -> Dict[str, np.ndarray]:
		path = self.in_dir / f"{sequence}.npz"
		payload = _load_npz_payload(path)

		T = int(payload["K"].shape[0])
		image_size = np.asarray(payload.get("image_size"), dtype=np.float64)
		if image_size.shape != (2,):
			raise ValueError(f"{sequence}: invalid image_size shape {image_size.shape}")
		W, H = float(image_size[0]), float(image_size[1])

		out: Dict[str, np.ndarray] = dict(payload)

		# --- bbox: log ratio ---
		if "bbox_feat" in out:
			out["bbox_feat"] = self._bbox_log_ratio_inplace(np.asarray(out["bbox_feat"]).copy())
		if "bbox_feat_clean" in out:
			out["bbox_feat_clean"] = self._bbox_log_ratio_inplace(np.asarray(out["bbox_feat_clean"]).copy())

		# --- camera boosted: standardize camera_center only ---
		for key in ("cam_feat_boosted_clean", "cam_feat_boosted_noisy"):
			if key not in out:
				continue
			cam = np.asarray(out[key], dtype=np.float64).copy()
			if cam.shape[-1] != 12:
				out[key] = cam.astype(np.float32)
				continue
			C = cam[:, 6:9]
			cam[:, 6:9] = (C - stats.mean_C[None, :]) / stats.std_C[None, :]
			out[key] = cam.astype(np.float32)

		# --- Y_rel_cam_gt standardization ---
		Y_rel = np.asarray(out["Y_rel_cam_gt"], dtype=np.float64)
		out["Y_rel_cam_gt"] = (
			(Y_rel - stats.mean_y_rel[None, None, :, :]) / stats.std_y_rel[None, None, :, :]
		).astype(np.float32)

		# --- Y_root_cam_gt standardization ---
		root = np.asarray(out["Y_root_cam_gt"], dtype=np.float64)
		out["Y_root_cam_gt"] = ((root - stats.mean_root[None, None, :]) / stats.std_root[None, None, :]).astype(
			np.float32
		)

		# --- SAM 3D: pelvis-center then standardize ---
		X_sam3d = np.asarray(out["skel_3d_sam3dbody_from_bbox_gt"], dtype=np.float64)
		X_ntjc, _ = _ensure_ntjc(X_sam3d, T=T, name=f"{sequence} skel_3d")
		pelvis = self.compute_pelvis(X_ntjc, mode=self.pelvis_mode)  # (N,T,3)
		X_rel = X_ntjc - pelvis[:, :, None, :]
		X_norm = (X_rel - stats.mean_sam3d_rel[None, None, :, :]) / stats.std_sam3d_rel[None, None, :, :]
		out["skel_3d_sam3dbody_from_bbox_gt"] = X_norm.astype(np.float32)

		# --- SAM 2D: image norm + box norm ---
		X_sam2d_raw = np.asarray(out["skel_2d_sam3dbody_from_bbox_gt"], dtype=np.float64)
		X_sam2d_ntjc, _ = _ensure_ntjc(X_sam2d_raw, T=T, name=f"{sequence} skel_2d")
		if X_sam2d_ntjc.ndim != 4 or X_sam2d_ntjc.shape[-1] != 2:
			raise ValueError(f"{sequence}: invalid skel_2d shape {X_sam2d_ntjc.shape}")

		# Image normalization (always written as (N,T,J,2))
		X_img_norm = X_sam2d_ntjc.copy()
		X_img_norm[..., 0] = X_img_norm[..., 0] / max(W, self.eps)
		X_img_norm[..., 1] = X_img_norm[..., 1] / max(H, self.eps)
		out["skel_2d_sam3dbody_from_bbox_gt"] = X_img_norm.astype(np.float32)

		# Box normalization (always written as (N,T,J,2))
		boxes = np.asarray(out.get("boxes_xyxy"), dtype=np.float64)  # (N,T,4)
		if boxes.ndim == 3 and boxes.shape[-1] == 4:
			x1, y1, x2, y2 = [boxes[..., i] for i in range(4)]
			cx = 0.5 * (x1 + x2)
			cy = 0.5 * (y1 + y2)
			bw = np.maximum(x2 - x1, self.eps)
			bh = np.maximum(y2 - y1, self.eps)

			X_box = X_sam2d_ntjc.copy()
			X_box[..., 0] = (X_box[..., 0] - cx[..., None]) / bw[..., None]
			X_box[..., 1] = (X_box[..., 1] - cy[..., None]) / bh[..., None]
			out["skel_2d_sam3dbody_from_bbox_gt_box"] = X_box.astype(np.float32)
		else:
			out.pop("skel_2d_sam3dbody_from_bbox_gt_box", None)

		# Attach normalization meta.
		meta_norm = {
			"normalized": True,
			"pelvis_mode": self.pelvis_mode,
			"bbox_ratio": "log(w/h)",
			"skel2d": {
				"main": "x/W, y/H",
				"extra": "skel_2d_sam3dbody_from_bbox_gt_box = (x-cx_box)/w_box, (y-cy_box)/h_box",
			},
			"labels": {
				"Y_rel_cam_gt": "standardized using train mean/std",
				"Y_root_cam_gt": "standardized using train mean/std",
			},
			"camera": {
				"camera_center": "standardized using train mean/std",
				"camera_forward": "unchanged (unit vector)",
			},
		}
		out["meta_norm_json"] = np.array(json.dumps(meta_norm), dtype=object)

		return out

	# ------------------------------------------------------------------
	# Orchestration
	# ------------------------------------------------------------------

	def run(
		self,
		*,
		train_n: int = 65,
		valid_n: int = 12,
		test_n: int = 12,
		seed: int = 12345,
		overwrite: bool = False,
	) -> NormalizationStats:
		sequences = self.read_sequences()
		train, valid, test = self.split_sequences(sequences, train_n, valid_n, test_n, seed)

		# Compute train-only stats.
		stats = self.compute_train_stats(train, seed=seed, valid_sequences=valid, test_sequences=test)

		# Save stats.
		self.out_dir.mkdir(parents=True, exist_ok=True)
		stats.save(self.out_dir)

		# Write normalized feature files.
		split_to_seqs = {"train": train, "valid": valid, "test": test}
		written = 0

		for split_name, seqs in split_to_seqs.items():
			out_split_dir = self.out_dir / split_name
			out_split_dir.mkdir(parents=True, exist_ok=True)

			for seq in seqs:
				out_path = out_split_dir / f"{seq}.npz"
				if out_path.exists() and not overwrite:
					continue
				norm_payload = self.normalize_sequence(seq, stats)
				_save_npz_atomic(out_path, norm_payload)
				written += 1

		# Save the split for convenience.
		split_path = self.out_dir / "split.json"
		split_path.write_text(
			json.dumps({"train": list(train), "valid": list(valid), "test": list(test), "seed": int(seed)}, indent=2),
			encoding="utf-8",
		)

		print(
			f"Done. Wrote {written} normalized .npz files under {self.out_dir} "
			f"(train={len(train)}, valid={len(valid)}, test={len(test)})."
		)
		return stats


def _build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		description=(
			"Normalize data/features/*.npz into data/features_normalized/{train,valid,test} "
			"using train-only mean/std to avoid data leakage."
		)
	)

	p.add_argument("train_n", nargs="?", type=int, default=65, help="Number of train sequences (default: 65)")
	p.add_argument("valid_n", nargs="?", type=int, default=12, help="Number of valid sequences (default: 12)")
	p.add_argument("test_n", nargs="?", type=int, default=12, help="Number of test sequences (default: 12)")

	p.add_argument("--seed", type=int, default=12345, help="RNG seed for reproducible split")
	p.add_argument(
		"--pelvis-mode",
		type=str,
		default="hips_mean",
		choices=["hips_mean", "joint8"],
		help="Pelvis centering mode for SAM3D (default: hips_mean)",
	)
	p.add_argument(
		"--data-dir",
		type=str,
		default=None,
		help="Dataset root (default: field_converter/pathseeker.py::DATA_DIR)",
	)
	p.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized .npz files")

	return p


def main() -> None:
	args = _build_argparser().parse_args()
	normalizer = Normalizer(data_dir=args.data_dir, pelvis_mode=args.pelvis_mode)
	normalizer.run(
		train_n=args.train_n,
		valid_n=args.valid_n,
		test_n=args.test_n,
		seed=args.seed,
		overwrite=args.overwrite,
	)


if __name__ == "__main__":
	main()

