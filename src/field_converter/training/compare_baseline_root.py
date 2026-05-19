from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from field_converter.evaluation.baseline import MeanRootBaseline, compute_mean_root_norm
from field_converter.evaluation.evaluator import Evaluator
from field_converter.evaluation.visualization import plot_model_vs_baseline
from field_converter.training.config import load_run_config
from field_converter.training.dataset import NormalizedFrameDataset
from field_converter.utils.io import ensure_dir, write_json
from field_converter.utils.normalization import TorchNormalizationStats
from field_converter.utils.torch_utils import get_device


def _load_metrics(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gain(baseline: float, model: float) -> Tuple[float, float]:
    abs_gain = baseline - model
    rel_gain = abs_gain / baseline if baseline != 0 and baseline == baseline else float("nan")
    return abs_gain, rel_gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Baseline A (mean-root) and compare to model")
    parser.add_argument("--config", type=str, default="configs/root_mlp_v1.yaml")
    parser.add_argument("--baseline_run_name", type=str, default="baseline_mean_root")
    parser.add_argument(
        "--model_metrics",
        type=str,
        default=None,
        help="Path to model metrics.json (defaults to outputs/eval_reports/<run_name>/metrics.json)",
    )
    args = parser.parse_args()

    cfg = load_run_config(Path(args.config))
    device = get_device(cfg.device)

    # ---- Load model metrics
    model_metrics_path = Path(args.model_metrics) if args.model_metrics else (cfg.eval_reports_dir / "metrics.json")
    if not model_metrics_path.exists():
        raise FileNotFoundError(
            f"Model metrics not found: {model_metrics_path}. Run evaluation first."
        )

    model_report = _load_metrics(model_metrics_path)
    model_splits = model_report.get("splits", {})

    # ---- Baseline evaluation
    baseline_run_name = str(args.baseline_run_name)
    baseline_output_dir = cfg.output_dir
    baseline_eval_reports_dir = baseline_output_dir / "eval_reports" / baseline_run_name
    baseline_predictions_dir = baseline_output_dir / "predictions" / baseline_run_name

    ensure_dir(baseline_eval_reports_dir)
    ensure_dir(baseline_predictions_dir)

    stats = TorchNormalizationStats.load(cfg.normalization_stats_path, device="cpu")

    pin_memory = device.type == "cuda"

    dl_kwargs: dict[str, object] = {}
    if cfg.eval.num_workers > 0:
        dl_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 1,
            }
        )

    # Baseline mean in normalized space: compute from train split.
    train_ds = NormalizedFrameDataset(
        data_dir=cfg.data_dir,
        split="train",
        input_config=cfg.input_config,
        seed=cfg.seed,
        max_sequences=cfg.dataset.max_sequences,
        max_samples_per_sequence=cfg.dataset.max_samples_per_sequence,
        subsample_stride=cfg.dataset.subsample_stride,
        min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
    )
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.num_workers,
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    mean_root_norm = compute_mean_root_norm(train_dl, device=device).to("cpu")
    baseline = MeanRootBaseline(mean_root_norm=mean_root_norm)

    evaluator = Evaluator(
        stats=stats,
        device=device,
        save_predictions_npz=True,
        save_predictions_csv=False,
    )

    baseline_report: Dict[str, Dict[str, float]] = {}
    for split in cfg.eval.splits:
        ds = NormalizedFrameDataset(
            data_dir=cfg.data_dir,
            split=split,
            input_config=cfg.input_config,
            seed=cfg.seed,
            max_sequences=cfg.dataset.max_sequences,
            max_samples_per_sequence=cfg.dataset.max_samples_per_sequence,
            subsample_stride=cfg.dataset.subsample_stride,
            min_in_image_joints_ratio=cfg.dataset.min_in_image_joints_ratio,
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            num_workers=cfg.eval.num_workers,
            pin_memory=pin_memory,
            **dl_kwargs,
        )

        out = evaluator.evaluate_split(model=baseline, dataloader=dl, out_dir=baseline_predictions_dir, split_name=split)
        baseline_report[split] = out.metrics

    write_json(
        baseline_eval_reports_dir / "metrics.json",
        {
            "run_name": baseline_run_name,
            "baseline": "mean_root_norm",
            "mean_root_norm": mean_root_norm.tolist(),
            "splits": baseline_report,
        },
    )

    # ---- Compare
    compare_splits = [s for s in cfg.eval.splits if s in model_splits and s in baseline_report]
    if not compare_splits:
        raise ValueError("No common splits found between model and baseline")

    comparisons: Dict[str, Dict[str, float]] = {}
    for split in compare_splits:
        m = model_splits[split]
        b = baseline_report[split]

        comp: Dict[str, float] = {}
        for key in ["root_error_mean_m", "MPJPE_cam_m", "MPJPE_world_m"]:
            bval = float(b.get(key, float("nan")))
            mval = float(m.get(key, float("nan")))
            abs_gain, rel_gain = _gain(bval, mval)
            comp[f"{key}_baseline"] = bval
            comp[f"{key}_model"] = mval
            comp[f"{key}_gain_abs"] = abs_gain
            comp[f"{key}_gain_rel"] = rel_gain
        comparisons[split] = comp

    ensure_dir(cfg.eval_reports_dir / "plots")
    write_json(cfg.eval_reports_dir / "baseline_comparison.json", {"baseline_run": baseline_run_name, "comparisons": comparisons})

    split_for_plot = "valid" if "valid" in comparisons else compare_splits[0]
    plot_model_vs_baseline(
        model_metrics=model_splits[split_for_plot],
        baseline_metrics=baseline_report[split_for_plot],
        out_path=cfg.eval_reports_dir / "plots" / f"model_vs_baseline_{split_for_plot}.png",
        title=f"Model vs Baseline (mean-root) — {split_for_plot}",
    )

    print("Baseline evaluated and comparison saved")


if __name__ == "__main__":
    main()
