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


def _payload_flag_is_true(payload: dict[str, np.ndarray], key: str) -> bool:
    """Return True if payload contains a truthy scalar flag.

    When saving flags into .npz, they are commonly stored as 0-d numpy arrays.
    Checking `payload.get(key) is True` will never work in that case.
    """
    if key not in payload:
        return False
    val = payload[key]
    try:
        if isinstance(val, np.ndarray):
            if val.size != 1:
                return False
            return bool(val.reshape(()).item())
        return bool(val)
    except Exception:
        return False


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


def fix_sam3d_convention_in_features(
    payload: dict[str, np.ndarray],
    *,
    key: str = "skel_3d_sam3dbody_from_bbox_gt",
) -> bool:
    """Fix SAM3D convention to match the rest of the dataset.

    The SAM3D skeleton stored in the dataset is expected to be in the same camera
    coordinate convention as other 3D quantities (e.g. ``Y_rel_cam_gt``).

    Empirically, SAM3D may differ by a global sign flip (i.e. X,Y,Z inverted).
    We correct it by multiplying coordinates by -1.

    Additionally, we align tensor layout to match the rest of the consolidated
    features:

    - expected layout for most arrays is (N, T, ...)
    - SAM skeletons are commonly stored as (T, N, J, 3)

    This function converts SAM3D to (N, T, J, 3) when needed.

    This is safe for either storage order (T,N,J,3) or (N,T,J,3) as we only touch
    the last dimension.
    """
    if _payload_flag_is_true(payload, "sam3d_convention_fixed"):
        return False
    if key not in payload:
        return False
    arr = payload[key]
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim != 4 or arr.shape[-1] != 3:
        return False

    changed = False

    # 1) Layout: (T,N,J,3) -> (N,T,J,3) using K.shape[0] as T.
    # We do this because most other arrays are (N,T,...) in consolidated features.
    if "K" in payload and isinstance(payload["K"], np.ndarray) and payload["K"].ndim >= 1:
        T = int(payload["K"].shape[0])
        if arr.shape[0] == T and arr.shape[1] != T:
            arr = arr.transpose(1, 0, 2, 3)
            changed = True

    # 2) Convention: global sign flip to match camera convention.
    arr2 = (-arr).astype(arr.dtype, copy=False)
    if arr2 is not arr:
        changed = True
    payload[key] = arr2
    if changed:
        payload["sam3d_convention_fixed"] = np.array(True, dtype=np.bool_)

    return changed


def fix_sam2d_layout_in_features(
    payload: dict[str, np.ndarray],
    *,
    key: str = "skel_2d_sam3dbody_from_bbox_gt",
) -> bool:
    """Align SAM2D layout to match the consolidated features convention.

    Most consolidated arrays use (N, T, ...) layout, but SAM2D may be stored as
    (T, N, J, 2). This function transposes SAM2D to (N, T, J, 2) when needed.

    Note: unlike SAM3D, we do NOT apply any sign flip on 2D image coordinates.
    """
    if _payload_flag_is_true(payload, "sam2d_layout_fixed"):
        return False
    if key not in payload:
        return False
    arr = payload[key]
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim != 4 or arr.shape[-1] != 2:
        return False
    if "K" not in payload or not isinstance(payload["K"], np.ndarray):
        return False
    T = int(payload["K"].shape[0])
    if arr.shape[0] == T and arr.shape[1] != T:
        payload[key] = arr.transpose(1, 0, 2, 3)
        payload["sam2d_layout_fixed"] = np.array(True, dtype=np.bool_)
        return True
    return False


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
        changed = fix_sam2d_layout_in_features(payload) or changed
        changed = fix_sam3d_convention_in_features(payload) or changed

        if changed:
            _save_npz_atomic(p, payload)
            modified += 1
            print(f"[updated] {p.name}")

    print(f"Done. Updated {modified}/{len(paths)} files.")


if __name__ == "__main__":
    main()

