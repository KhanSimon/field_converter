"""Normalize offline root initialization arrays with root training statistics.

Inputs:

    data/root_init_cam/{sequence}.npy

Outputs:

    data/root_init_cam_normalized/{train,valid,test}/{sequence}.npy

The normalization uses mean_root/std_root from data/features_normalized by
default, so root_init_norm lives in the same normalized space as Y_root_cam_gt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from field_converter import pathseeker as ps
from field_converter.data_preparation.normalize import NormalizationStats


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.npy")
    np.save(tmp_path, array)
    tmp_path.replace(path)


def _load_split(split_json_path: Path) -> Dict[str, list[str]]:
    if not split_json_path.exists():
        raise FileNotFoundError(f"Missing split JSON: {split_json_path}")
    payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    out: Dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        seqs = payload.get(split)
        if not isinstance(seqs, list) or not all(isinstance(s, str) for s in seqs):
            raise ValueError(f"Invalid split.json content for split={split}: {split_json_path}")
        out[split] = list(seqs)
    return out


def normalize_one(root_init_cam: np.ndarray, *, mean_root: np.ndarray, std_root: np.ndarray) -> np.ndarray:
    root = np.asarray(root_init_cam, dtype=np.float64)
    return ((root - mean_root[None, None, :]) / std_root[None, None, :]).astype(np.float32)


def normalize_all(
    *,
    data_dir: Path,
    in_dirname: str,
    out_dirname: str,
    split_to_sequences: Dict[str, Iterable[str]],
    stats: NormalizationStats,
    overwrite: bool,
) -> int:
    in_dir = data_dir / in_dirname
    out_dir = data_dir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for split, sequences in split_to_sequences.items():
        for sequence in sequences:
            in_path = in_dir / f"{sequence}.npy"
            out_path = out_dir / split / f"{sequence}.npy"
            if out_path.exists() and not overwrite:
                continue
            if not in_path.exists():
                raise FileNotFoundError(f"Missing root init file for {sequence}: {in_path}")

            root_init = np.load(in_path)
            root_init_norm = normalize_one(root_init, mean_root=stats.mean_root, std_root=stats.std_root)
            _save_npy_atomic(out_path, root_init_norm)
            written += 1

    split_json_out = out_dir / "split.json"
    split_json_out.write_text(json.dumps(split_to_sequences, indent=2), encoding="utf-8")

    meta = {
        "source": str(in_dir),
        "normalization": "root_init_norm = (root_init_cam - mean_root) / std_root",
        "mean_root": stats.mean_root.tolist(),
        "std_root": stats.std_root.tolist(),
        "train_sequences": list(stats.train_sequences),
        "valid_sequences": list(stats.valid_sequences),
        "test_sequences": list(stats.test_sequences),
    }
    (out_dir / "root_init_normalization_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return written


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize data/root_init_cam with root mean/std statistics.")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root (default: pathseeker.DATA_DIR)")
    parser.add_argument("--in-dirname", type=str, default="root_init_cam", help="Input root-init folder under data-dir")
    parser.add_argument(
        "--out-dirname",
        type=str,
        default="root_init_cam_normalized",
        help="Output normalized root-init folder under data-dir",
    )
    parser.add_argument(
        "--features-normalized-dirname",
        type=str,
        default="features_normalized",
        help="Normalized features folder containing split.json and normalization_stats.npz",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized root-init files")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    data_dir = Path(args.data_dir) if args.data_dir is not None else Path(ps.DATA_DIR)
    features_norm_dir = data_dir / args.features_normalized_dirname
    stats = NormalizationStats.load(features_norm_dir / "normalization_stats.npz")
    split_to_sequences = _load_split(features_norm_dir / "split.json")

    written = normalize_all(
        data_dir=data_dir,
        in_dirname=args.in_dirname,
        out_dirname=args.out_dirname,
        split_to_sequences=split_to_sequences,
        stats=stats,
        overwrite=bool(args.overwrite),
    )
    print(f"Done. Wrote {written} normalized root-init files under {data_dir / args.out_dirname}.")


if __name__ == "__main__":
    main()
