from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/field_converter_matplotlib")

import matplotlib

# Non-interactive backend (cluster friendly)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


Mode = Literal["gt", "sam", "both"]


@dataclass(frozen=True)
class NormalizationStats:
    mean_sam3d_rel: np.ndarray
    std_sam3d_rel: np.ndarray
    mean_root: np.ndarray
    std_root: np.ndarray


@dataclass(frozen=True)
class PlayerSample:
    seq_path: Path
    split: str
    gt_cam: Optional[np.ndarray]
    sam_cam: Optional[np.ndarray]
    valid_joints: Optional[np.ndarray]
    valid_person: Optional[bool]


VIEW_PRESETS: Dict[str, Tuple[float, float]] = {
    "face": (5.0, -90.0),
    "front": (5.0, -90.0),
    "dos": (5.0, 90.0),
    "back": (5.0, 90.0),
    "gauche": (5.0, 180.0),
    "left": (5.0, 180.0),
    "droite": (5.0, 0.0),
    "right": (5.0, 0.0),
    "top": (90.0, -90.0),
    "dessus": (90.0, -90.0),
    "iso": (20.0, -60.0),
}


def _repo_root() -> Path:
    # .../field_converter/src/field_converter/play_vizu/viz_player.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def _default_features_dir() -> Path:
    return _repo_root() / "data" / "features_normalized"


def _default_out_dir() -> Path:
    return _repo_root() / "data" / "play_vizu"


def _resolve_features_dir(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path

    # Be tolerant with the singular name used in some notes/commands.
    if path.name == "feature_normalized":
        plural = path.with_name("features_normalized")
        if plural.exists():
            return plural

    raise FileNotFoundError(f"features directory not found: {path}")


def _find_sequence_path(features_dir: Path, seq_name: str) -> Tuple[Path, str]:
    candidates = [(features_dir / f"{seq_name}.npz", "")]
    candidates.extend((features_dir / split / f"{seq_name}.npz", split) for split in ("train", "valid", "test"))

    for path, split in candidates:
        if path.exists():
            return path, split

    tried = "\n  ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(f"sequence {seq_name!r} not found. Tried:\n  {tried}")


def _load_stats(features_dir: Path) -> NormalizationStats:
    stats_path = features_dir / "normalization_stats.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"missing normalization stats: {stats_path}")

    with np.load(stats_path, allow_pickle=True) as npz:
        return NormalizationStats(
            mean_sam3d_rel=np.asarray(npz["mean_sam3d_rel"], dtype=np.float32),
            std_sam3d_rel=np.asarray(npz["std_sam3d_rel"], dtype=np.float32),
            mean_root=np.asarray(npz["mean_root"], dtype=np.float32),
            std_root=np.asarray(npz["std_root"], dtype=np.float32),
        )


def _check_index(name: str, idx: int, size: int) -> None:
    if idx < 0 or idx >= size:
        raise IndexError(f"{name} out of range: {idx} (valid: 0..{size - 1})")


def _denorm_sam_cam(sam_norm: np.ndarray, root_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    sam_rel_m = sam_norm * stats.std_sam3d_rel + stats.mean_sam3d_rel
    root_m = root_norm * stats.std_root + stats.mean_root
    return sam_rel_m + root_m[None, :]


def load_player_sample(
    *,
    features_dir: Path,
    seq_name: str,
    frame_idx: int,
    player_idx: int,
    mode: Mode,
) -> PlayerSample:
    features_dir = _resolve_features_dir(features_dir)
    seq_path, split = _find_sequence_path(features_dir, seq_name)
    stats = _load_stats(features_dir)

    with np.load(seq_path, allow_pickle=True) as npz:
        valid_mask = np.asarray(npz["valid_mask"], dtype=bool) if "valid_mask" in npz.files else None
        valid_joints_all = np.asarray(npz["valid_joints"], dtype=bool) if "valid_joints" in npz.files else None

        shape_ref = npz["Y_cam_gt"] if "Y_cam_gt" in npz.files else npz["skel_3d_sam3dbody_from_bbox_gt"]
        _check_index("player_idx", int(player_idx), int(shape_ref.shape[0]))
        _check_index("frame_idx", int(frame_idx), int(shape_ref.shape[1]))

        valid_person = None
        if valid_mask is not None:
            valid_person = bool(valid_mask[player_idx, frame_idx])

        valid_joints = None
        if valid_joints_all is not None:
            valid_joints = np.asarray(valid_joints_all[player_idx, frame_idx], dtype=bool)

        gt_cam = None
        if mode in ("gt", "both"):
            if "Y_cam_gt" not in npz.files:
                raise KeyError(f"{seq_path} does not contain Y_cam_gt")
            gt_cam = np.asarray(npz["Y_cam_gt"][player_idx, frame_idx], dtype=np.float32)

        sam_cam = None
        if mode in ("sam", "both"):
            if "skel_3d_sam3dbody_from_bbox_gt" not in npz.files:
                raise KeyError(f"{seq_path} does not contain skel_3d_sam3dbody_from_bbox_gt")
            if "Y_root_cam_gt" not in npz.files:
                raise KeyError(f"{seq_path} does not contain Y_root_cam_gt, needed to place SAM in camera space")
            sam_norm = np.asarray(npz["skel_3d_sam3dbody_from_bbox_gt"][player_idx, frame_idx], dtype=np.float32)
            root_norm = np.asarray(npz["Y_root_cam_gt"][player_idx, frame_idx], dtype=np.float32)
            sam_cam = _denorm_sam_cam(sam_norm, root_norm, stats).astype(np.float32)

    return PlayerSample(
        seq_path=seq_path,
        split=split,
        gt_cam=gt_cam,
        sam_cam=sam_cam,
        valid_joints=valid_joints,
        valid_person=valid_person,
    )


def _finite_points(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def _set_axes_equal(ax: plt.Axes, arrays: list[np.ndarray]) -> None:
    finite_arrays = []
    for arr in arrays:
        finite = _finite_points(arr)
        if np.any(finite):
            finite_arrays.append(arr[finite])

    if not finite_arrays:
        return

    pts = np.concatenate(finite_arrays, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius, 0.1)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_points(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str,
    valid_joints: Optional[np.ndarray],
) -> None:
    points = np.asarray(points, dtype=np.float32)
    finite = _finite_points(points)
    mask = finite
    if valid_joints is not None and valid_joints.shape[0] == points.shape[0]:
        mask = mask & np.asarray(valid_joints, dtype=bool)

    invalid = finite & ~mask
    if np.any(invalid):
        ax.scatter(
            points[invalid, 0],
            points[invalid, 1],
            points[invalid, 2],
            s=24,
            color=color,
            marker=marker,
            alpha=0.18,
        )

    if np.any(mask):
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=42,
            color=color,
            marker=marker,
            label=label,
            depthshade=False,
        )

    for joint_idx, xyz in enumerate(points):
        if not finite[joint_idx]:
            continue
        alpha = 0.95 if mask[joint_idx] else 0.25
        ax.text(
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            f" {joint_idx}",
            color=color,
            fontsize=8,
            alpha=alpha,
        )


def render_player_plot(
    *,
    sample: PlayerSample,
    seq_name: str,
    frame_idx: int,
    player_idx: int,
    mode: Mode,
    view: str,
    out_path: Path,
    elev: Optional[float] = None,
    azim: Optional[float] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    view_key = view.lower()
    if elev is None or azim is None:
        if view_key not in VIEW_PRESETS:
            allowed = ", ".join(sorted(VIEW_PRESETS))
            raise ValueError(f"unknown view {view!r}. Allowed: {allowed}, or pass --elev and --azim")
        preset_elev, preset_azim = VIEW_PRESETS[view_key]
        elev = preset_elev if elev is None else elev
        azim = preset_azim if azim is None else azim

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    arrays_for_limits = []
    if sample.gt_cam is not None:
        _plot_points(
            ax,
            sample.gt_cam,
            label="GT 3D (camera)",
            color="#1f77b4",
            marker="o",
            valid_joints=sample.valid_joints,
        )
        arrays_for_limits.append(sample.gt_cam)

    if sample.sam_cam is not None:
        _plot_points(
            ax,
            sample.sam_cam,
            label="SAM3DBody 3D (denorm + GT root)",
            color="#d62728",
            marker="^",
            valid_joints=sample.valid_joints,
        )
        arrays_for_limits.append(sample.sam_cam)

    _set_axes_equal(ax, arrays_for_limits)
    ax.view_init(elev=float(elev), azim=float(azim))

    ax.set_xlabel("X cam (m)")
    ax.set_ylabel("Y cam (m)")
    ax.set_zlabel("Z cam (m)")
    ax.legend(loc="upper right")

    valid_msg = "unknown" if sample.valid_person is None else str(bool(sample.valid_person))
    split_msg = f" split={sample.split}" if sample.split else ""
    ax.set_title(
        f"{seq_name}{split_msg} | frame={frame_idx} player={player_idx} "
        f"| mode={mode} | view={view} | valid={valid_msg}"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Save a 3D plot of one player's 25 joints from data/features_normalized."
    )
    parser.add_argument("--seq_name", "--sequence", dest="seq_name", type=str, required=True)
    parser.add_argument("--frame_idx", "--frame", dest="frame_idx", type=int, required=True)
    parser.add_argument("--player_idx", "--player", dest="player_idx", type=int, required=True)
    parser.add_argument("--mode", choices=("gt", "sam", "both"), default="both")
    parser.add_argument(
        "--view",
        "--angle",
        dest="view",
        type=str,
        default="iso",
        help="View preset: face, dos/back, gauche/left, droite/right, top/dessus, iso.",
    )
    parser.add_argument("--elev", type=float, default=None, help="Override matplotlib elevation angle.")
    parser.add_argument("--azim", type=float, default=None, help="Override matplotlib azimuth angle.")
    parser.add_argument("--features_dir", type=Path, default=_default_features_dir())
    parser.add_argument("--out_dir", type=Path, default=_default_out_dir())
    args = parser.parse_args(argv)

    sample = load_player_sample(
        features_dir=args.features_dir,
        seq_name=args.seq_name,
        frame_idx=int(args.frame_idx),
        player_idx=int(args.player_idx),
        mode=args.mode,
    )

    out_name = (
        f"{args.seq_name}_f{int(args.frame_idx):05d}_p{int(args.player_idx):03d}"
        f"_{args.mode}_{args.view.lower()}.png"
    )
    out_path = Path(args.out_dir) / out_name

    render_player_plot(
        sample=sample,
        seq_name=args.seq_name,
        frame_idx=int(args.frame_idx),
        player_idx=int(args.player_idx),
        mode=args.mode,
        view=args.view,
        out_path=out_path,
        elev=args.elev,
        azim=args.azim,
    )

    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())