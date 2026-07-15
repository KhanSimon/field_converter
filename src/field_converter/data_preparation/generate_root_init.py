"""Generate offline root initialization from SAM3DBody and ground intersections.

Outputs one raw camera-space array per sequence:

    data/root_init_cam/{sequence}.npy

Each array has shape (N,T,3). Values are NaN when the SAM skeleton, SAM2D
pixel, camera ray, or ground intersection is invalid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from field_converter import pathseeker as ps
from field_converter.data_preparation.features_creation import FeatureCreator


def _load_npz_payload(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: npz[key] for key in npz.files}


def _read_sequences(data_dir: Path, sequences_file: str) -> list[str]:
    path = data_dir / sequences_file
    if not path.exists():
        raise FileNotFoundError(f"Missing sequences file: {path}")
    sequences: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sequences.append(line)
    return sequences


def _ensure_ntjc(arr: np.ndarray, *, T: int, C: int, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 4 or arr.shape[-1] != C:
        raise ValueError(f"{name}: expected 4D array with last dim {C}, got {arr.shape}")
    if arr.shape[0] == T and arr.shape[1] != T:
        return arr.transpose(1, 0, 2, 3)
    return arr


def _lowest_joint_indices(sam2d_ntj2: np.ndarray, sam3d_ntj3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return lowest SAM joint indices and validity mask.

    SAM3DBody uses an inverted y axis in this dataset: larger relative y values
    are lower on the body. This matches FeatureCreator.compute_ground_intersections_from_sam.
    """
    finite_3d = np.isfinite(sam3d_ntj3).all(axis=-1)
    finite_2d = np.isfinite(sam2d_ntj2).all(axis=-1)
    selectable = finite_3d & finite_2d
    has_joint = selectable.any(axis=-1)
    y_for_argmax = np.where(selectable, sam3d_ntj3[..., 1], -np.inf)
    return np.argmax(y_for_argmax, axis=-1), has_joint


def _world_to_camera_points(points_nt3: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.einsum("ntw,tcw->ntc", points_nt3, R) + t[None, :, :]


def compute_root_init_cam(payload: Dict[str, np.ndarray], creator: FeatureCreator, *, sequence: str) -> np.ndarray:
    required = ["skel_2d_sam3dbody_from_bbox_gt", "skel_3d_sam3dbody_from_bbox_gt", "K", "R", "t", "k"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"{sequence}: missing required keys for root init: {missing}")

    K = np.asarray(payload["K"], dtype=np.float64)
    R = np.asarray(payload["R"], dtype=np.float64)
    t = np.asarray(payload["t"], dtype=np.float64)
    k = np.asarray(payload["k"], dtype=np.float64)
    T = int(K.shape[0])

    sam2d = _ensure_ntjc(
        np.asarray(payload["skel_2d_sam3dbody_from_bbox_gt"], dtype=np.float64),
        T=T,
        C=2,
        name=f"{sequence} skel_2d_sam3dbody_from_bbox_gt",
    )
    sam3d = _ensure_ntjc(
        np.asarray(payload["skel_3d_sam3dbody_from_bbox_gt"], dtype=np.float64),
        T=T,
        C=3,
        name=f"{sequence} skel_3d_sam3dbody_from_bbox_gt",
    )
    if sam2d.shape[:3] != sam3d.shape[:3]:
        raise ValueError(f"{sequence}: SAM2D/SAM3D shape mismatch: {sam2d.shape} vs {sam3d.shape}")

    lowest_idx, has_joint = _lowest_joint_indices(sam2d, sam3d)
    n_idx = np.arange(sam3d.shape[0])[:, None]
    t_idx = np.arange(sam3d.shape[1])[None, :]
    lowest_rel_cam = sam3d[n_idx, t_idx, lowest_idx]

    lowest_world = creator.compute_ground_intersections_from_sam(sam2d, sam3d, K, R, t, k)
    lowest_cam = _world_to_camera_points(np.asarray(lowest_world, dtype=np.float64), R, t)

    root_init_cam = lowest_cam - lowest_rel_cam
    valid = has_joint & np.isfinite(lowest_world).all(axis=-1) & np.isfinite(lowest_rel_cam).all(axis=-1)
    root_init_cam[~valid] = np.nan
    return root_init_cam.astype(np.float32)


def generate_all(
    *,
    data_dir: Path,
    features_dirname: str,
    out_dirname: str,
    sequences: Iterable[str],
    overwrite: bool,
) -> int:
    creator = FeatureCreator(data_dir=data_dir)
    features_dir = data_dir / features_dirname
    out_dir = data_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for sequence in sequences:
        in_path = features_dir / f"{sequence}.npz"
        out_path = out_dir / f"{sequence}.npy"
        if out_path.exists() and not overwrite:
            print(f"[skip] {sequence}: {out_path}")
            continue
        if not in_path.exists():
            raise FileNotFoundError(f"Missing feature file for {sequence}: {in_path}")

        print(f"[root-init] {sequence}")
        payload = _load_npz_payload(in_path)
        root_init = compute_root_init_cam(payload, creator, sequence=sequence)
        np.save(out_path, root_init)
        written += 1

    return written


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate data/root_init_cam from consolidated feature files.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root (default: pathseeker.DATA_DIR)")
    parser.add_argument("--features-dirname", type=str, default="features", help="Input consolidated features folder")
    parser.add_argument("--out-dirname", type=str, default="root_init_cam", help="Output folder under data-dir")
    parser.add_argument("--sequences-file", type=str, default="sequences_gt.txt", help="Sequence list under data-dir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing root init .npy files")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    data_dir = Path(args.data_dir) if args.data_dir is not None else Path(ps.DATA_DIR)
    sequences = _read_sequences(data_dir, args.sequences_file)
    written = generate_all(
        data_dir=data_dir,
        features_dirname=args.features_dirname,
        out_dirname=args.out_dirname,
        sequences=sequences,
        overwrite=bool(args.overwrite),
    )
    print(f"Done. Wrote {written} root-init files under {data_dir / args.out_dirname}.")


if __name__ == "__main__":
    main()
