from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib

# Non-interactive backend for cluster runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from field_converter.training.config import load_run_config
from field_converter.training.tcn.config import load_tcn_run_config
from field_converter.utils.io import ensure_dir, write_json


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_split_metrics(report: Dict[str, Any], split: str) -> Dict[str, float]:
    splits = report.get("splits", {})
    if not isinstance(splits, dict):
        return {}
    m = splits.get(split, {})
    if not isinstance(m, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in m.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _gain(baseline: float, model: float) -> Tuple[float, float]:
    abs_gain = baseline - model
    rel_gain = abs_gain / baseline if baseline == baseline and baseline != 0.0 else float("nan")
    return abs_gain, rel_gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Baseline vs MLP vs TCN metrics")
    parser.add_argument("--mlp_config", type=str, default="configs/mlp/root_mlp_v1_train.yaml")
    parser.add_argument("--tcn_config", type=str, default="configs/tcn/root_tcn_v1.yaml")
    parser.add_argument("--baseline_run_name", type=str, default="baseline_mean_root")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/eval_reports/compare_root_models",
        help="Directory for comparison JSON + plot",
    )
    args = parser.parse_args()

    mlp_cfg = load_run_config(Path(args.mlp_config))
    tcn_cfg = load_tcn_run_config(Path(args.tcn_config))

    split = str(args.split)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    mlp_metrics_path = mlp_cfg.eval_reports_dir / "metrics.json"
    tcn_metrics_path = tcn_cfg.eval_reports_dir / "metrics.json"

    baseline_metrics_path = tcn_cfg.output_dir / "eval_reports" / str(args.baseline_run_name) / "metrics.json"
    if not baseline_metrics_path.exists():
        alt = mlp_cfg.output_dir / "eval_reports" / str(args.baseline_run_name) / "metrics.json"
        if alt.exists():
            baseline_metrics_path = alt

    missing = [p for p in [mlp_metrics_path, tcn_metrics_path, baseline_metrics_path] if not p.exists()]
    if missing:
        msg = "Missing metrics files:\n" + "\n".join(f"- {p}" for p in missing)
        msg += (
            "\n\nTip: run evaluations first:\n"
            "- PYTHONPATH=src python -m field_converter.training.evaluate_root_mlp --config <mlp_config> --checkpoint best\n"
            "- PYTHONPATH=src python -m field_converter.training.evaluate_root_tcn --config <tcn_config> --checkpoint best\n"
            "- PYTHONPATH=src python -m field_converter.training.compare_baseline_root --config <mlp_config>"
        )
        raise FileNotFoundError(msg)

    baseline_report = _load_metrics_json(baseline_metrics_path)
    mlp_report = _load_metrics_json(mlp_metrics_path)
    tcn_report = _load_metrics_json(tcn_metrics_path)

    base_m = _get_split_metrics(baseline_report, split)
    mlp_m = _get_split_metrics(mlp_report, split)
    tcn_m = _get_split_metrics(tcn_report, split)

    keys = [
        ("root_error_mean_m", "Root mean (m)"),
        ("MPJPE_cam_m", "MPJPE cam (m)"),
        ("MPJPE_world_m", "MPJPE world (m)"),
        ("MPJPE_local_m", "MPJPE local (m)"),
        ("reprojection_error_mean_px", "Reproj mean (px)"),
    ]

    comparisons: Dict[str, Dict[str, float]] = {}
    for k, _lab in keys:
        b = float(base_m.get(k, float("nan")))
        m1 = float(mlp_m.get(k, float("nan")))
        m2 = float(tcn_m.get(k, float("nan")))
        abs_gain_mlp, rel_gain_mlp = _gain(b, m1)
        abs_gain_tcn, rel_gain_tcn = _gain(b, m2)
        comparisons[k] = {
            "baseline": b,
            "mlp": m1,
            "tcn": m2,
            "gain_abs_mlp": abs_gain_mlp,
            "gain_rel_mlp": rel_gain_mlp,
            "gain_abs_tcn": abs_gain_tcn,
            "gain_rel_tcn": rel_gain_tcn,
        }

    out_json = out_dir / f"compare_{split}.json"
    write_json(
        out_json,
        {
            "split": split,
            "baseline_metrics": str(baseline_metrics_path),
            "mlp_metrics": str(mlp_metrics_path),
            "tcn_metrics": str(tcn_metrics_path),
            "comparisons": comparisons,
        },
    )

    # Plot
    labels = [lab for _k, lab in keys]
    base_vals = np.array([float(base_m.get(k, float("nan"))) for k, _ in keys], dtype=np.float64)
    mlp_vals = np.array([float(mlp_m.get(k, float("nan"))) for k, _ in keys], dtype=np.float64)
    tcn_vals = np.array([float(tcn_m.get(k, float("nan"))) for k, _ in keys], dtype=np.float64)

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    ax.bar(x - width, base_vals, width, label="baseline")
    ax.bar(x, mlp_vals, width, label="mlp")
    ax.bar(x + width, tcn_vals, width, label="tcn")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(f"Root refinement comparison — {split}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_png = out_dir / f"compare_{split}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
