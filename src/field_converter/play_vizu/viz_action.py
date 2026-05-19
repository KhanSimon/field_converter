from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import matplotlib

# Non-interactive backend (cluster friendly)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


@dataclass(frozen=True)
class SequencePaths:
    images_dir: Path
    boxes_path: Path
    y2d_gt_path: Path
    sam2d_path: Path
    valid_mask_path: Path
    valid_joints_path: Path
    cameras_path: Path


def _repo_root() -> Path:
    # .../field_converter/src/field_converter/play_vizu/viz_action.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _default_data_dir() -> Path:
    return _repo_root() / "data"


def _default_out_dir() -> Path:
    return _repo_root() / "play_vizu"


def _frame_filename(frame_idx: int) -> str:
    return f"{int(frame_idx):05d}.jpg"


def _load_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return np.load(path, mmap_mode="r")


def _ensure_nt(arr: np.ndarray, valid_mask: np.ndarray, *, name: str) -> np.ndarray:
    """Return array aligned as (N,T,...) based on valid_mask shape (N,T)."""

    if arr.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dims, got shape={arr.shape}")

    n, t = valid_mask.shape

    if arr.shape[0] == n and arr.shape[1] == t:
        return arr

    if arr.shape[0] == t and arr.shape[1] == n:
        return np.swapaxes(arr, 0, 1)

    raise ValueError(
        f"Cannot align {name} to (N,T,...) using valid_mask (N={n},T={t}). "
        f"Got shape={arr.shape}"
    )


def sequence_paths(data_dir: Path, seq_name: str) -> SequencePaths:
    data_dir = Path(data_dir)
    return SequencePaths(
        images_dir=data_dir / "images_gt" / seq_name,
        boxes_path=data_dir / "boxes_gt" / f"{seq_name}.npy",
        y2d_gt_path=data_dir / "Y_2d_gt" / f"{seq_name}.npy",
        sam2d_path=data_dir / "skel_2d_sam3dbody_from_bbox_gt" / f"{seq_name}.npy",
        valid_mask_path=data_dir / "valid_mask" / f"{seq_name}.npy",
        valid_joints_path=data_dir / "valid_joints" / f"{seq_name}.npy",
        cameras_path=data_dir / "cameras_gt" / f"{seq_name}.npz",
    )


def load_camera_info(*, data_dir: Path, seq_name: str, frame_idx: int) -> Dict[str, Any]:
    """Load camera GT info for a given frame.

    Returns a dict with at least: K, R, t, k, camera_center_world.
    """

    p = sequence_paths(data_dir, seq_name)
    if not p.cameras_path.exists():
        return {"available": False, "path": str(p.cameras_path)}

    with np.load(p.cameras_path, allow_pickle=True) as z:
        K_all = np.asarray(z["K"], dtype=np.float64)
        R_all = np.asarray(z["R"], dtype=np.float64)
        t_all = np.asarray(z["t"], dtype=np.float64)
        k_all = np.asarray(z["k"], dtype=np.float64) if "k" in z.files else None

    if frame_idx < 0 or frame_idx >= K_all.shape[0]:
        raise IndexError(f"frame_idx out of range for cameras_gt: {frame_idx} (T={K_all.shape[0]})")

    K = K_all[frame_idx]
    R = R_all[frame_idx]
    t = t_all[frame_idx]
    k = k_all[frame_idx] if k_all is not None else None

    # Convention validated in data/README.md:
    # X_cam = R X_world + t  =>  camera center in world: C = -R^T t
    C = -(R.T @ t)

    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    out: Dict[str, Any] = {
        "available": True,
        "path": str(p.cameras_path),
        "K": K,
        "R": R,
        "t": t,
        "k": k,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "camera_center_world": C,
    }
    return out


def _bbox_summary(bbox_xyxy: np.ndarray) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy, dtype=np.float32).tolist()]
    w = float(max(0.0, x2 - x1))
    h = float(max(0.0, y2 - y1))
    cx = float(x1 + 0.5 * w)
    cy = float(y1 + 0.5 * h)
    area = float(w * h)
    aspect = float(w / h) if h > 0 else float("inf")
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "area": area,
        "aspect": aspect,
    }


def print_run_info(
    *,
    seq_name: str,
    frame_idx: int,
    player_idx: int,
    image_path: Path,
    img_shape_hw: Tuple[int, int],
    bbox_xyxy: np.ndarray,
    is_valid_person: bool,
    valid_joints: Optional[np.ndarray],
    cam: Dict[str, Any],
) -> None:
    H, W = int(img_shape_hw[0]), int(img_shape_hw[1])
    bb = _bbox_summary(bbox_xyxy)

    print("--- viz_action info ---")
    print(f"seq_name:   {seq_name}")
    print(f"frame_idx:  {int(frame_idx)}")
    print(f"player_idx: {int(player_idx)}")
    print(f"image:      {image_path}")
    print(f"image_size: W={W} H={H}")
    print(f"valid_mask: {bool(is_valid_person)}")
    if valid_joints is not None:
        vj = np.asarray(valid_joints, dtype=bool)
        print(f"valid_joints: {int(vj.sum())}/{int(vj.size)}")
    print(
        "bbox_xyxy:  "
        f"x1={bb['x1']:.1f} y1={bb['y1']:.1f} x2={bb['x2']:.1f} y2={bb['y2']:.1f}"
    )
    print(
        "bbox_xywh:  "
        f"x={bb['x1']:.1f} y={bb['y1']:.1f} w={bb['w']:.1f} h={bb['h']:.1f} "
        f"(cx={bb['cx']:.1f} cy={bb['cy']:.1f} aspect={bb['aspect']:.3f})"
    )

    if cam.get("available"):
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        print(f"camera_gt:  {cam.get('path','')}")
        print(f"K:          fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        k = cam.get("k")
        if k is not None:
            k = np.asarray(k, dtype=np.float64).reshape(-1)
            k_str = " ".join([f"{float(v):.6g}" for v in k.tolist()])
            print(f"dist(k):    [{k_str}]")
        t = np.asarray(cam["t"], dtype=np.float64).reshape(-1)
        C = np.asarray(cam["camera_center_world"], dtype=np.float64).reshape(-1)
        print(f"t:          [{t[0]:.6g} {t[1]:.6g} {t[2]:.6g}]")
        print(f"C_world:    [{C[0]:.6g} {C[1]:.6g} {C[2]:.6g}]")
    else:
        print(f"camera_gt:  unavailable ({cam.get('path','')})")


def load_sample(
    *,
    data_dir: Path,
    seq_name: str,
    frame_idx: int,
    player_idx: int,
) -> Tuple[Path, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
    """Load (image_path, bbox_xyxy, sam2d, gt2d, valid_joints, is_valid_person)."""

    p = sequence_paths(data_dir, seq_name)

    valid_mask = _load_npy(p.valid_mask_path).astype(bool)
    valid_mask_nt = _ensure_nt(valid_mask, valid_mask, name="valid_mask")

    if player_idx < 0 or player_idx >= valid_mask_nt.shape[0]:
        raise IndexError(f"player_idx out of range: {player_idx} (N={valid_mask_nt.shape[0]})")
    if frame_idx < 0 or frame_idx >= valid_mask_nt.shape[1]:
        raise IndexError(f"frame_idx out of range: {frame_idx} (T={valid_mask_nt.shape[1]})")

    is_valid_person = bool(valid_mask_nt[player_idx, frame_idx])

    boxes = _ensure_nt(_load_npy(p.boxes_path), valid_mask_nt, name="boxes_gt")  # (N,T,4)
    bbox = np.asarray(boxes[player_idx, frame_idx], dtype=np.float32)

    y2d = _ensure_nt(_load_npy(p.y2d_gt_path), valid_mask_nt, name="Y_2d_gt")  # (N,T,25,2)
    gt2d = np.asarray(y2d[player_idx, frame_idx], dtype=np.float32)

    sam2d_raw = _ensure_nt(
        _load_npy(p.sam2d_path), valid_mask_nt, name="skel_2d_sam3dbody_from_bbox_gt"
    )  # (N,T,25,2)
    sam2d = np.asarray(sam2d_raw[player_idx, frame_idx], dtype=np.float32)

    vj = _ensure_nt(_load_npy(p.valid_joints_path).astype(bool), valid_mask_nt, name="valid_joints")
    valid_joints = np.asarray(vj[player_idx, frame_idx], dtype=bool)

    image_path = p.images_dir / _frame_filename(frame_idx)
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    return image_path, bbox, sam2d, gt2d, valid_joints, is_valid_person


def render_action(
    *,
    image_path: Path,
    bbox_xyxy: np.ndarray,
    out_path: Path,
    title: str,
    sam2d: Optional[np.ndarray] = None,
    gt2d: Optional[np.ndarray] = None,
    valid_joints: Optional[np.ndarray] = None,
    show_sam2d: bool = False,
    show_gt2d: bool = False,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = plt.imread(str(image_path))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(img)

    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    rect = Rectangle((x1, y1), max(0.0, x2 - x1), max(0.0, y2 - y1), fill=False, linewidth=2)
    ax.add_patch(rect)

    vmask = None
    if valid_joints is not None:
        vmask = np.asarray(valid_joints, dtype=bool)

    if show_gt2d and gt2d is not None:
        pts = np.asarray(gt2d, dtype=np.float32)
        if vmask is not None and vmask.shape[0] == pts.shape[0]:
            pts = pts[vmask]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, label="GT 2D")

    if show_sam2d and sam2d is not None:
        pts = np.asarray(sam2d, dtype=np.float32)
        if vmask is not None and vmask.shape[0] == pts.shape[0]:
            pts = pts[vmask]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, label="SAM 2D")

    if (show_gt2d and gt2d is not None) or (show_sam2d and sam2d is not None):
        ax.legend(loc="lower right")

    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize one frame/player for a given sequence")
    parser.add_argument("--seq_name", type=str, required=True)
    parser.add_argument("--frame_idx", type=int, required=True, help="0-based frame index (as in test_prediction.csv)")
    parser.add_argument(
        "--player_idx", type=int, required=True, help="0-based player/person index (as in test_prediction.csv)"
    )
    parser.add_argument("--data_dir", type=Path, default=_default_data_dir())
    parser.add_argument("--out_dir", type=Path, default=_default_out_dir())
    parser.add_argument("--show_sam2d", action="store_true", help="Overlay SAM3DBody 2D points")
    parser.add_argument("--show_gt2d", action="store_true", help="Overlay GT 2D points")

    args = parser.parse_args(argv)

    image_path, bbox, sam2d, gt2d, valid_joints, is_valid_person = load_sample(
        data_dir=args.data_dir,
        seq_name=args.seq_name,
        frame_idx=args.frame_idx,
        player_idx=args.player_idx,
    )

    cam = load_camera_info(data_dir=args.data_dir, seq_name=args.seq_name, frame_idx=int(args.frame_idx))

    # Read once to print size; render_action reads again but keeps the implementation simple.
    img = plt.imread(str(image_path))
    if img.ndim < 2:
        raise ValueError(f"Unexpected image array shape: {img.shape}")
    img_h, img_w = int(img.shape[0]), int(img.shape[1])

    print_run_info(
        seq_name=args.seq_name,
        frame_idx=int(args.frame_idx),
        player_idx=int(args.player_idx),
        image_path=image_path,
        img_shape_hw=(img_h, img_w),
        bbox_xyxy=bbox,
        is_valid_person=is_valid_person,
        valid_joints=valid_joints,
        cam=cam,
    )

    out_name = f"{args.seq_name}_f{int(args.frame_idx):05d}_p{int(args.player_idx):03d}.jpg"
    out_path = Path(args.out_dir) / out_name

    title = f"{args.seq_name} frame={args.frame_idx} player={args.player_idx} valid={is_valid_person}"

    render_action(
        image_path=image_path,
        bbox_xyxy=bbox,
        out_path=out_path,
        title=title,
        sam2d=sam2d,
        gt2d=gt2d,
        valid_joints=valid_joints,
        show_sam2d=bool(args.show_sam2d),
        show_gt2d=bool(args.show_gt2d),
    )

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
