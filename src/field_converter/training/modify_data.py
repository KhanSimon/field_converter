from __future__ import annotations

from pathlib import Path

import numpy as np

from field_converter import pathseeker as ps


def _truncate_k(arr: np.ndarray) -> np.ndarray:
    """Remove the last 3 columns of k.

    Expected shape is (N, C) with C >= 3.
    """
    if arr.ndim != 2:
        return arr
    if arr.shape[1] < 3:
        return arr
    return arr[:, :-3]


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=True) as npz:
        for key in npz.files:
            payload[key] = npz[key]
    return payload


def _save_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(tmp_path, **payload)
    tmp_path.replace(path)


def _load_sequence_array(folder: Path, sequence: str) -> np.ndarray:
    npy_path = folder / f"{sequence}.npy"
    if npy_path.exists():
        return np.load(npy_path, allow_pickle=True)

    npz_path = folder / f"{sequence}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as npz:
            if "arr_0" in npz.files:
                return npz["arr_0"]
            if len(npz.files) == 1:
                return npz[npz.files[0]]
            raise KeyError(
                f"Ambiguous npz content for {npz_path}; keys={list(npz.files)}"
            )

    raise FileNotFoundError(f"Missing {sequence}.npy|npz in {folder}")


def add_sam3dbody_from_bbox_gt_features(
    sequence: str,
    payload: dict[str, np.ndarray],
    *,
    overwrite: bool = False,
) -> bool:
    """Add SAM3DBody skeleton arrays stored under data/*_from_bbox_gt into features payload."""
    changed = False

    key_to_folder = {
        "skel_2d_sam3dbody_from_bbox_gt": ps.DATA_DIR / "skel_2d_sam3dbody_from_bbox_gt",
        "skel_3d_sam3dbody_from_bbox_gt": ps.DATA_DIR / "skel_3d_sam3dbody_from_bbox_gt",
    }

    for key, folder in key_to_folder.items():
        if (not overwrite) and (key in payload):
            continue
        try:
            payload[key] = _load_sequence_array(folder, sequence)
        except FileNotFoundError:
            continue
        changed = True

    return changed


def truncate_k_in_features(payload: dict[str, np.ndarray]) -> bool:
    if "k" not in payload:
        return False
    arr = payload["k"]
    new_arr = _truncate_k(arr)
    if new_arr.shape == arr.shape:
        return False
    payload["k"] = new_arr
    return True


def main() -> None:
    features_dir = ps.DATA_DIR / "features"
    paths = sorted(features_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz found in {features_dir}")

    modified = 0
    for p in paths:
        seq = p.stem
        payload = _load_npz_payload(p)

        changed = False
        #changed = truncate_k_in_features(payload) or changed
        changed = add_sam3dbody_from_bbox_gt_features(seq, payload) or changed

        if changed:
            _save_npz_atomic(p, payload)
            modified += 1
            print(f"[updated] {p.name}")

    print(f"Done. Updated {modified}/{len(paths)} files.")


if __name__ == "__main__":
    main()

