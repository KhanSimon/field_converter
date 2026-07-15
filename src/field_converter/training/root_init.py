from __future__ import annotations

from pathlib import Path

import numpy as np


def default_root_init_dir(data_dir: Path | str) -> Path:
    return Path(data_dir).parent / "root_init_cam_normalized"


def resolve_root_init_path(root_init_dir: Path | str, split: str, sequence: str) -> Path:
    root = Path(root_init_dir)
    candidates = [
        root / split / f"{sequence}.npy",
        root / f"{sequence}.npy",
        root / split / f"{sequence}.npz",
        root / f"{sequence}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing normalized root init for seq={sequence!r} split={split!r}. "
        f"Looked under: {root}"
    )


def load_root_init_sequence(root_init_dir: Path | str, split: str, sequence: str) -> np.ndarray:
    path = resolve_root_init_path(root_init_dir, split, sequence)
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        with np.load(path, allow_pickle=True) as npz:
            for key in ("root_init_norm", "root_init_cam_norm", "arr_0"):
                if key in npz.files:
                    arr = npz[key]
                    break
            else:
                raise KeyError(f"{path} must contain one of: root_init_norm, root_init_cam_norm, arr_0")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"root init must have shape (N,T,3), got {arr.shape} from {path}")
    return arr
