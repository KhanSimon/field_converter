from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-field-converter")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/field-converter-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_NAME = "root_tcn_random_search"


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(results_csv: Path) -> list[dict[str, Any]]:
    if not results_csv.exists():
        raise FileNotFoundError(f"Random search results not found: {results_csv}")
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_metric_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [row for row in rows if math.isfinite(to_float(row.get(metric)))]


def sort_rows(rows: list[dict[str, Any]], metric: str, lower_is_better: bool) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: to_float(row.get(metric)), reverse=not lower_is_better)


def numeric_columns(rows: list[dict[str, Any]], metric: str) -> list[str]:
    blocked = {
        "trial",
        "returncode",
        "elapsed_s",
        "best_epoch",
        "last_epoch",
        metric,
    }
    cols: list[str] = []
    for key in rows[0].keys():
        if key in blocked or key.endswith("_path") or key in {"run_name", "status", "device"}:
            continue
        values = [to_float(row.get(key)) for row in rows]
        finite = [v for v in values if math.isfinite(v)]
        if len(finite) >= 2 and len(set(finite)) >= 2:
            cols.append(key)
    preferred = [
        "optimizer.lr",
        "optimizer.weight_decay",
        "model.dropout",
        "dataset.window_size",
        "dataset.stride",
        "model.temporal_hidden_dim",
        "training.batch_size",
        "loss_weights.root_vel",
        "loss_weights.root_acc",
        "loss_weights.cam3d",
        "loss_weights.proj",
        "loss_weights.root_axis_z",
        "dataset.min_valid_ratio",
        "dataset.min_in_image_joints_ratio",
        "dataset.min_bbox_margin_px",
    ]
    ordered = [c for c in preferred if c in cols]
    ordered.extend(c for c in cols if c not in ordered)
    return ordered


def plot_score_by_trial(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: Path,
    *,
    lower_is_better: bool,
) -> None:
    rows = sorted(rows, key=lambda row: int(float(row["trial"])))
    trials = [int(float(row["trial"])) for row in rows]
    scores = [to_float(row.get(metric)) for row in rows]
    chooser = min if lower_is_better else max
    best_idx = chooser(range(len(scores)), key=lambda i: scores[i])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trials, scores, marker="o", linewidth=1.5)
    ax.scatter([trials[best_idx]], [scores[best_idx]], s=90, color="tab:red", label="best")
    ax.set_xlabel("trial")
    ax.set_ylabel(metric)
    ax.set_title("Random search validation score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_trials(rows: list[dict[str, Any]], metric: str, out_path: Path, top_k: int) -> None:
    top = rows[: min(top_k, len(rows))]
    labels = [str(row["run_name"]).replace("root_tcn_random_search_", "") for row in top]
    scores = [to_float(row.get(metric)) for row in top]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(4.0, 0.45 * len(top))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y = list(range(len(top)))
    ax.barh(y, scores, color="tab:blue", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(top)} TCN random-search trials")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hparam_scatters(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: Path,
    *,
    lower_is_better: bool,
) -> None:
    cols = numeric_columns(rows, metric)[:12]
    if not cols:
        return
    ncols = 3
    nrows = math.ceil(len(cols) / ncols)
    scores = [to_float(row.get(metric)) for row in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4, 3.7 * nrows)))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, col in zip(axes_flat, cols):
        xs = [to_float(row.get(col)) for row in rows]
        ax.scatter(xs, scores, s=38, alpha=0.8)
        ax.set_xlabel(col)
        ax.set_ylabel(metric)
        if col in {"optimizer.lr", "optimizer.weight_decay", "loss_weights.root_vel", "loss_weights.root_acc", "loss_weights.cam3d", "loss_weights.proj"}:
            positive = [x for x in xs if math.isfinite(x) and x > 0]
            if len(positive) >= 2 and min(positive) > 0:
                ax.set_xscale("symlog", linthresh=min(positive))
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(cols) :]:
        ax.axis("off")
    direction = "lower is better" if lower_is_better else "higher is better"
    fig.suptitle(f"Hyperparameter sensitivity, {direction}", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_best_evaluation(config_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "field_converter.training.evaluate_root_tcn",
        "--config",
        str(config_path),
        "--checkpoint",
        "best",
    ]
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize a TCN random search")
    parser.add_argument("--search-name", type=str, default=DEFAULT_SEARCH_NAME)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--metric", type=str, default="best_root_error_mean_m")
    parser.add_argument("--higher-is-better", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--no-evaluate-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    search_dir = output_dir / "random_search" / args.search_name
    results_csv = args.results_csv or (search_dir / "random_search_results.csv")
    plots_dir = search_dir / "plots"

    rows = read_rows(results_csv)
    rows = finite_metric_rows(rows, args.metric)
    if not rows:
        raise ValueError(f"No completed trials with finite metric {args.metric!r} in {results_csv}")

    ranked = sort_rows(rows, args.metric, lower_is_better=not args.higher_is_better)
    best = ranked[0]
    config_path = Path(str(best["config_path"]))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Best config not found: {config_path}")

    search_dir.mkdir(parents=True, exist_ok=True)
    write_csv(search_dir / "random_search_results_ranked.csv", ranked)
    shutil.copyfile(config_path, search_dir / "best_config.yaml")

    best_payload = {
        "metric": args.metric,
        "metric_value": to_float(best.get(args.metric)),
        "run_name": best.get("run_name"),
        "trial": best.get("trial"),
        "config_path": str(config_path),
        "eval_reports_dir": str(output_dir / "eval_reports" / str(best.get("run_name"))),
        "predictions_dir": str(output_dir / "predictions" / str(best.get("run_name"))),
        "checkpoints_dir": str(output_dir / "checkpoints" / str(best.get("run_name"))),
        "hyperparameters": {
            key: value
            for key, value in best.items()
            if "." in key and value not in {"", None}
        },
    }
    (search_dir / "best_hyperparameters.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lower_is_better = not args.higher_is_better
    plot_score_by_trial(ranked, args.metric, plots_dir / "score_by_trial.png", lower_is_better=lower_is_better)
    plot_top_trials(ranked, args.metric, plots_dir / "top_trials.png", top_k=args.top_k)
    plot_hparam_scatters(
        ranked,
        args.metric,
        plots_dir / "hyperparameter_scatters.png",
        lower_is_better=lower_is_better,
    )

    if not args.no_evaluate_best:
        run_best_evaluation(config_path)

    best_plots_dir = output_dir / "eval_reports" / str(best.get("run_name")) / "plots"
    print(f"Best run: {best.get('run_name')} ({args.metric}={to_float(best.get(args.metric)):.6f})")
    print(f"Best config: {search_dir / 'best_config.yaml'}")
    print(f"Random-search plots: {plots_dir}")
    print(f"Best-run plots: {best_plots_dir}")


if __name__ == "__main__":
    main()
