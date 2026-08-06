"""Evaluate the naive ground-intersection root baseline.

The baseline is the root initialization used by delta models:

1. select the lowest valid SAM3DBody joint;
2. cast the camera ray through the corresponding 2D joint;
3. intersect that ray with the ground plane;
4. translate the relative SAM skeleton so that its lowest joint lies on the
   intersection.

The script recomputes this camera-space root from raw features with an explicit
pelvis centring step. It therefore cannot silently evaluate legacy root-init
files generated before that correction. Evaluation uses the same splits,
filters and metrics as a Transformer run, without loading a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from field_converter.data_preparation.features_creation import FeatureCreator
from field_converter.data_preparation.generate_root_init import compute_root_init_cam
from field_converter.data_preparation.normalize_root_init import normalize_one
from field_converter.evaluation.temporal_evaluator import TemporalEvaluator
from field_converter.training.config import InputConfig
from field_converter.training.filters import filter_valid_mask_bbox_geometry, filter_valid_mask_in_image
from field_converter.training.root_init import load_root_init_sequence
from field_converter.training.tcn.window_dataset import NormalizedWindowDataset
from field_converter.training.transformer.config import TransformerRunConfig, load_transformer_run_config
from field_converter.utils.io import write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device


BASELINE_NAME = "baseline_ground_intersection_pelvis_centered"

# All these metrics are errors or losses: lower is better.
COMPARISON_METRICS = (
    "root_error_mean_m",
    "root_error_median_m",
    "root_error_p90_m",
    "root_error_x_m",
    "root_error_y_m",
    "root_error_z_m",
    "MPJPE_cam_m",
    "MPJPE_world_m",
    "MPJPE_local_m",
    "challenge_points",
    "reprojection_error_mean_px",
    "reprojection_error_median_px",
    "root_velocity_error_mean_m",
    "root_acceleration_error_mean_m",
)


class ZeroDeltaBaseline:
    """Return a zero correction, making the final prediction ``root_init``."""

    supports_valid_mask = True

    def __call__(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        del valid_mask
        return x.new_zeros((*x.shape[:2], 3))


def _minimal_input_config() -> InputConfig:
    """Only load the input required to give the baseline a window shape."""
    return InputConfig(
        use_x3d_sam_rel=True,
        use_x2d_img=False,
        use_x2d_box=False,
        use_pitch_points_2d=False,
        use_bbox_feat=False,
        use_cam_feat=False,
        use_ground_intersection=False,
        use_valid_joints_as_input=False,
    )


def _make_loader(
    *,
    cfg: TransformerRunConfig,
    split: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    max_sequences: Optional[int],
    root_init_dir: Path,
) -> torch.utils.data.DataLoader:
    # prediction_mode="delta" makes each batch expose root_init_norm. The
    # baseline outputs a zero delta, so TemporalEvaluator evaluates root_init.
    dataset = NormalizedWindowDataset(
        data_dir=cfg.data_dir,
        split=split,  # type: ignore[arg-type]
        input_config=_minimal_input_config(),
        prediction_mode="delta",
        root_init_dir=root_init_dir,
        seed=cfg.seed,
        max_sequences=max_sequences,
        max_windows_per_sequence=None,
        window_size=cfg.dataset.window_size,
        stride=cfg.dataset.stride,
        min_valid_ratio=0.0,
        pad_mode=cfg.dataset.pad_mode,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        min_bbox_width_px=cfg.dataset.min_bbox_width_px,
        min_bbox_height_px=cfg.dataset.min_bbox_height_px,
        min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
        filter_by_min_valid_ratio=False,
    )

    loader_kwargs: dict[str, object] = {}
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 1})

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


def _effective_valid_mask(npz: Any, cfg: TransformerRunConfig) -> np.ndarray:
    valid_mask = np.asarray(npz["valid_mask"], dtype=bool)

    if (
        cfg.dataset.min_bbox_width_px is not None
        or cfg.dataset.min_bbox_height_px is not None
        or cfg.dataset.min_bbox_margin_px is not None
    ):
        valid_mask = filter_valid_mask_bbox_geometry(
            valid_mask=valid_mask,
            boxes_xyxy=np.asarray(npz["boxes_xyxy"], dtype=np.float32),
            image_size=np.asarray(npz["image_size"], dtype=np.float32),
            min_bbox_width_px=cfg.dataset.min_bbox_width_px,
            min_bbox_height_px=cfg.dataset.min_bbox_height_px,
            min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
        )

    if cfg.dataset.min_in_image_joints_ratio is not None:
        valid_mask = filter_valid_mask_in_image(
            valid_mask=valid_mask,
            valid_joints=np.asarray(npz["valid_joints"], dtype=bool),
            Y_2d_gt=np.asarray(npz["Y_2d_gt"], dtype=np.float32),
            image_size=np.asarray(npz["image_size"], dtype=np.float32),
            min_in_image_joints_ratio=float(cfg.dataset.min_in_image_joints_ratio),
        )

    return valid_mask


def _check_intersection_coverage(
    *,
    cfg: TransformerRunConfig,
    split: str,
    sequences: Iterable[str],
    root_init_dir: Path,
) -> int:
    """Ensure the baseline exists on every frame used for model evaluation."""
    num_evaluated = 0
    num_missing = 0

    for sequence in sequences:
        feature_path = cfg.data_dir / split / f"{sequence}.npz"
        with np.load(feature_path, allow_pickle=True) as npz:
            valid_mask = _effective_valid_mask(npz, cfg)

        root_init = load_root_init_sequence(root_init_dir, split, sequence)
        if root_init.shape[:2] != valid_mask.shape:
            raise ValueError(
                f"root init/mask shape mismatch for {sequence}: "
                f"root_init={root_init.shape}, valid_mask={valid_mask.shape}"
            )

        finite = np.isfinite(root_init).all(axis=-1)
        num_evaluated += int(valid_mask.sum())
        num_missing += int((valid_mask & ~finite).sum())

    if num_missing:
        raise RuntimeError(
            f"{split}: {num_missing} evaluated frames have no finite ground intersection. "
            "Refusing to replace them with a zero root because that would bias the baseline."
        )

    return num_evaluated


def _read_split_sequences(cfg: TransformerRunConfig, split: str, max_sequences: Optional[int]) -> list[str]:
    payload = json.loads(cfg.split_json_path.read_text(encoding="utf-8"))
    sequences = payload.get(split)
    if not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences):
        raise ValueError(f"Invalid sequence list for split={split!r}: {cfg.split_json_path}")
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    return sequences


def _generate_corrected_root_init(
    *,
    split: str,
    sequences: Iterable[str],
    raw_features_dir: Path,
    out_dir: Path,
    stats: TorchNormalizationStats,
) -> None:
    """Recompute the corrected baseline without trusting existing root-init files."""
    creator = FeatureCreator(data_dir=raw_features_dir.parent)
    split_out_dir = out_dir / split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    mean_root = stats.mean_root.detach().cpu().numpy().astype(np.float64)
    std_root = stats.std_root.detach().cpu().numpy().astype(np.float64)

    for sequence in sequences:
        feature_path = raw_features_dir / f"{sequence}.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing raw feature file: {feature_path}")
        with np.load(feature_path, allow_pickle=True) as npz:
            required = (
                "skel_2d_sam3dbody_from_bbox_gt",
                "skel_3d_sam3dbody_from_bbox_gt",
                "K",
                "R",
                "t",
                "k",
            )
            missing = [key for key in required if key not in npz.files]
            if missing:
                raise KeyError(f"{feature_path}: missing root-init inputs: {missing}")
            payload = {key: npz[key] for key in required}

        root_init_cam = compute_root_init_cam(payload, creator, sequence=sequence)
        root_init_norm = normalize_one(root_init_cam, mean_root=mean_root, std_root=std_root)
        np.save(split_out_dir / f"{sequence}.npy", root_init_norm)


def _load_model_splits(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError(f"Invalid model metrics file (missing 'splits' object): {path}")
    return splits


def _build_comparison(
    *,
    baseline_splits: Dict[str, Dict[str, float]],
    model_splits: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split, baseline_metrics in baseline_splits.items():
        if split not in model_splits:
            continue
        rows: Dict[str, Dict[str, float]] = {}
        for metric in COMPARISON_METRICS:
            if metric not in baseline_metrics or metric not in model_splits[split]:
                continue
            baseline_value = float(baseline_metrics[metric])
            model_value = float(model_splits[split][metric])
            gain_abs = baseline_value - model_value
            gain_rel = (
                gain_abs / baseline_value
                if np.isfinite(baseline_value) and baseline_value != 0.0
                else float("nan")
            )
            rows[metric] = {
                "baseline": baseline_value,
                "model": model_value,
                "model_gain_abs": gain_abs,
                "model_gain_rel": gain_rel,
            }
        comparison[split] = rows
    return comparison


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return str(value)
    return f"{value:.6g}"


def _print_report(
    *,
    baseline_splits: Dict[str, Dict[str, float]],
    comparison: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    for split, metrics in baseline_splits.items():
        print(f"\n=== Baseline intersection sol — {split} ===")
        for metric, value in metrics.items():
            print(f"{metric:<40} {_format_value(float(value)):>12}")

        rows = comparison.get(split, {})
        if rows:
            print(f"\n--- Comparaison modèle — {split} (gain positif = modèle meilleur) ---")
            print(f"{'métrique':<40} {'baseline':>12} {'modèle':>12} {'gain':>12}")
            for metric, values in rows.items():
                gain_rel = values["model_gain_rel"]
                gain_text = f"{100.0 * gain_rel:.2f}%" if np.isfinite(gain_rel) else "nan"
                print(
                    f"{metric:<40} "
                    f"{_format_value(values['baseline']):>12} "
                    f"{_format_value(values['model']):>12} "
                    f"{gain_text:>12}"
                )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Affiche les métriques du baseline naïf rayon caméra / point SAM le plus bas / sol."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/transformer/root_transformer_v1.yaml"),
        help="Configuration Transformer dont les splits et filtres doivent être reproduits.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "valid", "test"),
        default=None,
        help="Splits à évaluer (par défaut: eval.splits du config).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Surcharge eval.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Surcharge eval.num_workers.")
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Limite de debug. La comparaison au modèle est désactivée si cette option est utilisée.",
    )
    parser.add_argument(
        "--model-metrics",
        type=Path,
        default=None,
        help="metrics.json du modèle (par défaut: rapport du run indiqué par le config).",
    )
    parser.add_argument(
        "--raw-features-dir",
        type=Path,
        default=None,
        help="Features brutes (par défaut: <data_dir parent>/features).",
    )
    parser.add_argument(
        "--no-model-comparison",
        action="store_true",
        help="N'affiche pas la comparaison avec le modèle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON de sortie (par défaut: <eval_reports_dir>/naive_baseline_metrics.json).",
    )
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cfg = load_transformer_run_config(args.config)
    device = get_device(cfg.device)

    splits = list(args.splits) if args.splits is not None else list(cfg.eval.splits)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(cfg.eval.batch_size)
    num_workers = int(args.num_workers) if args.num_workers is not None else int(cfg.eval.num_workers)
    max_sequences = int(args.max_sequences) if args.max_sequences is not None else cfg.dataset.max_sequences
    raw_features_dir = args.raw_features_dir or (cfg.data_dir.parent / "features")
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("--max-sequences must be > 0")

    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")
    evaluator = TemporalEvaluator(
        stats=stats,
        device=device,
        save_predictions_npz=False,
        save_predictions_csv=False,
        prediction_mode="delta",
    )

    baseline_splits: Dict[str, Dict[str, float]] = {}
    evaluated_frames: Dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="field_converter_corrected_root_init_") as temp_dir:
        corrected_root_init_dir = Path(temp_dir)
        for split in splits:
            sequences = _read_split_sequences(cfg, split, max_sequences)
            _generate_corrected_root_init(
                split=split,
                sequences=sequences,
                raw_features_dir=raw_features_dir,
                out_dir=corrected_root_init_dir,
                stats=stats,
            )
            loader = _make_loader(
                cfg=cfg,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                max_sequences=max_sequences,
                root_init_dir=corrected_root_init_dir,
            )
            evaluated_frames[split] = _check_intersection_coverage(
                cfg=cfg,
                split=split,
                sequences=loader.dataset.sequences,
                root_init_dir=corrected_root_init_dir,
            )
            outputs, extras = evaluator.evaluate_split(
                model=ZeroDeltaBaseline(),
                dataloader=loader,
                out_dir=cfg.predictions_dir / BASELINE_NAME,
                split_name=split,
                min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
                min_bbox_width_px=cfg.dataset.min_bbox_width_px,
                min_bbox_height_px=cfg.dataset.min_bbox_height_px,
                min_bbox_margin_px=cfg.dataset.min_bbox_margin_px,
                root_axis_weights=cfg.loss_weights.root_axis_weights,
            )
            metrics = dict(outputs.metrics)
            metrics["num_frames_total"] = float(extras.num_frames_total)
            metrics["num_frames_covered"] = float(extras.num_frames_covered)
            metrics["num_frames_uncovered"] = float(extras.num_frames_uncovered)
            metrics["num_frames_evaluated"] = float(evaluated_frames[split])
            baseline_splits[split] = metrics

    model_metrics_path = args.model_metrics or (cfg.eval_reports_dir / "metrics.json")
    compare_to_model = not args.no_model_comparison and args.max_sequences is None
    model_splits = _load_model_splits(model_metrics_path) if compare_to_model else {}
    comparison = _build_comparison(baseline_splits=baseline_splits, model_splits=model_splits)

    output_path = args.output or (cfg.eval_reports_dir / "naive_baseline_metrics.json")
    report: Dict[str, Any] = {
        "run_name": BASELINE_NAME,
        "baseline": "lowest_sam_joint_camera_ray_ground_intersection_pelvis_centered",
        "root_init_formula": "ground_hit_cam - (lowest_sam_joint_cam - sam_pelvis_cam)",
        "config": str(args.config),
        "raw_features_dir": str(raw_features_dir),
        "legacy_root_init_files_used": False,
        "splits": baseline_splits,
        "model_metrics": str(model_metrics_path) if compare_to_model and model_metrics_path.exists() else None,
        "comparisons": comparison,
    }
    write_json(output_path, report)

    _print_report(baseline_splits=baseline_splits, comparison=comparison)
    if compare_to_model and not model_metrics_path.exists():
        print(f"\nRapport modèle introuvable, comparaison ignorée: {model_metrics_path}")
    elif args.max_sequences is not None and not args.no_model_comparison:
        print("\nComparaison au modèle ignorée car --max-sequences évalue seulement un sous-ensemble.")
    print(f"\nRapport JSON enregistré: {output_path}")


if __name__ == "__main__":
    main()
