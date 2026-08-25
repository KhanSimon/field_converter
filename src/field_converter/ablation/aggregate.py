from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from field_converter.ablation.common import (
    PROJECT_ROOT,
    campaign_output_dir,
    load_manifest,
    load_plan,
    metrics_path,
    predictions_path,
    write_json_atomic,
)

_CACHE_DIR = PROJECT_ROOT / "outputs" / "ablation" / ".plot_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
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
    "num_frames_covered",
    "num_frames_uncovered",
)

ARCH_COLORS = {
    "tcn": "#007C83",
    "transformer": "#D1495B",
    "mlp": "#E5A823",
    "geometry": "#5B6573",
    "mean_root": "#9AA0A6",
}


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    finite = np.isfinite(values)
    values = values[finite]
    clusters = clusters[finite]
    unique, inverse = np.unique(clusters, return_inverse=True)
    if values.size == 0 or unique.size == 0:
        return float("nan"), float("nan")
    sums = np.bincount(inverse, weights=values, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, unique.size, size=(samples, unique.size))
    boot = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1.0)
    alpha = (1.0 - confidence) / 2.0
    quantiles = np.asarray(np.quantile(boot, [alpha, 1.0 - alpha]), dtype=np.float64)
    return float(quantiles[0]), float(quantiles[1])


def _prediction_errors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return (
            np.asarray(npz["root_error_m"], dtype=np.float64),
            np.asarray(npz["seq_id"], dtype=np.int64),
        )


def _canonical_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=True) as npz:
        seq_names = [str(value) for value in npz["seq_names"].tolist()]
        seq_id = np.asarray(npz["seq_id"], dtype=np.int64)
        person = np.asarray(npz["person_idx"], dtype=np.int64)
        frame = np.asarray(npz["frame_idx"], dtype=np.int64)
        errors = np.asarray(npz["root_error_m"], dtype=np.float64)
    name_rank = {name: rank for rank, name in enumerate(sorted(seq_names))}
    canonical_seq = np.asarray([name_rank[seq_names[int(value)]] for value in seq_id], dtype=np.int64)
    order = np.lexsort((frame, person, canonical_seq))
    identifiers = np.column_stack((canonical_seq[order], person[order], frame[order]))
    return identifiers, errors[order], canonical_seq[order], tuple(sorted(seq_names))


def _paired_effect_ci(
    variant_path: Path,
    reference_path: Path,
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float, float] | None:
    variant_keys, variant_errors, variant_clusters, variant_names = _canonical_predictions(variant_path)
    reference_keys, reference_errors, _, reference_names = _canonical_predictions(reference_path)
    if (
        variant_names != reference_names
        or variant_keys.shape != reference_keys.shape
        or not np.array_equal(variant_keys, reference_keys)
    ):
        return None
    differences = variant_errors - reference_errors
    low, high = _cluster_bootstrap_ci(
        differences,
        variant_clusters,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
    return float(np.nanmean(differences)), low, high


def _save_figure(fig: plt.Figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _row_lookup(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    return {
        (str(row["fold"]), str(row["architecture"]), int(row["seed"]), str(row["variant"]), str(row["split"])): row
        for row in rows
        if row.get("source") == "model"
    }


def _plot_overview(
    rows: list[dict[str, Any]], figures_dir: Path, *, primary_fold: str, base_seed: int
) -> None:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row["fold"] != primary_fold or row["split"] != "test":
            continue
        if row["source"] == "model" and int(row["seed"]) == base_seed and row["variant"] == "full":
            selected.append(row)
        elif row["source"] in {"geometry_baseline", "mean_root_baseline"}:
            selected.append(row)
    order = {"geometry": 0, "mean_root": 1, "mlp": 2, "tcn": 3, "transformer": 4}
    selected.sort(key=lambda row: order.get(str(row["architecture"]), 99))
    if not selected:
        return
    labels = [str(row["label"]) for row in selected]
    values = np.asarray([100.0 * _finite_float(row["root_error_mean_m"]) for row in selected])
    lows = np.asarray([100.0 * _finite_float(row.get("root_error_ci_low_m")) for row in selected])
    highs = np.asarray([100.0 * _finite_float(row.get("root_error_ci_high_m")) for row in selected])
    valid_ci = np.isfinite(lows) & np.isfinite(highs)
    errors = np.zeros((2, len(selected)), dtype=float)
    errors[0, valid_ci] = values[valid_ci] - lows[valid_ci]
    errors[1, valid_ci] = highs[valid_ci] - values[valid_ci]

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    y = np.arange(len(selected))
    colors = [ARCH_COLORS.get(str(row["architecture"]), "#777777") for row in selected]
    ax.barh(y, values, color=colors)
    if np.any(valid_ci):
        ax.errorbar(
            values[valid_ci],
            y[valid_ci],
            xerr=errors[:, valid_ci],
            fmt="none",
            ecolor="#111111",
            capsize=3,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean root error (cm), unseen test match")
    ax.grid(axis="x", alpha=0.25)
    _save_figure(fig, figures_dir, "01_primary_test_overview")


def _plot_input_ablation(effects: list[dict[str, Any]], figures_dir: Path) -> None:
    rows = [row for row in effects if row["family"] == "input_ablation" and row["split"] == "test"]
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    if not variants:
        return
    x = np.arange(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for offset, architecture in ((-0.5, "tcn"), (0.5, "transformer")):
        values = []
        lower_errors = []
        upper_errors = []
        for variant in variants:
            match = next(
                (row for row in rows if row["variant"] == variant and row["architecture"] == architecture), None
            )
            value = _finite_float(match.get("paired_delta_root_error_m")) if match else float("nan")
            if not np.isfinite(value) and match:
                value = _finite_float(match["delta_root_error_m"])
            low = _finite_float(match.get("paired_delta_ci_low_m")) if match else float("nan")
            high = _finite_float(match.get("paired_delta_ci_high_m")) if match else float("nan")
            values.append(100.0 * value)
            lower_errors.append(100.0 * max(0.0, value - low) if np.isfinite(low) else 0.0)
            upper_errors.append(100.0 * max(0.0, high - value) if np.isfinite(high) else 0.0)
        ax.bar(
            x + offset * width,
            values,
            width,
            yerr=np.asarray([lower_errors, upper_errors]),
            capsize=3,
            label=architecture.upper(),
            color=ARCH_COLORS[architecture],
        )
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_xticks(x, [variant.replace("_", "\n") for variant in variants])
    ax.set_ylabel("Change in mean root error (cm; positive is worse)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, "02_input_ablation_effects")


def _plot_formulation(
    rows: list[dict[str, Any]], figures_dir: Path, *, primary_fold: str, base_seed: int
) -> None:
    lookup = _row_lookup(rows)
    variants = ("full", "absolute_root_init_input", "absolute")
    labels = ("Delta", "Absolute + root init", "Absolute")
    x = np.arange(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    found = False
    for offset, architecture in ((-0.5, "tcn"), (0.5, "transformer")):
        values = []
        for variant in variants:
            row = lookup.get((primary_fold, architecture, base_seed, variant, "test"))
            values.append(100.0 * _finite_float(row["root_error_mean_m"]) if row else np.nan)
            found |= row is not None
        ax.bar(x + offset * width, values, width, label=architecture.upper(), color=ARCH_COLORS[architecture])
    if not found:
        plt.close(fig)
        return
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean root error (cm)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, "03_delta_vs_absolute")


def _plot_match_robustness(rows: list[dict[str, Any]], figures_dir: Path, *, base_seed: int) -> None:
    folds = list(dict.fromkeys(str(row["fold"]) for row in rows if row["source"] == "model"))
    if not folds:
        return
    x = np.arange(len(folds))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for offset, architecture in ((-0.5, "tcn"), (0.5, "transformer")):
        values = []
        for fold in folds:
            row = next(
                (
                    item
                    for item in rows
                    if item["source"] == "model"
                    and item["fold"] == fold
                    and item["split"] == "test"
                    and item["architecture"] == architecture
                    and item["variant"] == "full"
                    and int(item["seed"]) == base_seed
                ),
                None,
            )
            values.append(100.0 * _finite_float(row["root_error_mean_m"]) if row else np.nan)
        ax.bar(x + offset * width, values, width, label=architecture.upper(), color=ARCH_COLORS[architecture])
    ax.set_xticks(x, [fold.replace("_", " ") for fold in folds])
    ax.set_ylabel("Mean root error (cm)")
    ax.set_xlabel("Held-out match fold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, figures_dir, "04_unseen_match_robustness")


def _plot_error_cdf(
    manifest: dict[str, Any], rows: list[dict[str, Any]], figures_dir: Path, *, primary_fold: str, base_seed: int
) -> None:
    candidates = [
        row
        for row in rows
        if row["source"] == "model"
        and row["fold"] == primary_fold
        and row["split"] == "test"
        and row["variant"] == "full"
        and int(row["seed"]) == base_seed
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    plotted = False
    for row in candidates:
        path = predictions_path(manifest, str(row["run_name"]), "test")
        if not path.exists():
            continue
        errors, _ = _prediction_errors(path)
        errors = np.sort(errors[np.isfinite(errors)]) * 100.0
        if errors.size == 0:
            continue
        cdf = np.arange(1, errors.size + 1) / errors.size
        architecture = str(row["architecture"])
        ax.plot(errors, cdf, label=architecture.upper(), color=ARCH_COLORS[architecture], linewidth=2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Root error (cm)")
    ax.set_ylabel("Empirical CDF")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir, "05_primary_test_error_cdf")


def _plot_metric_heatmap(
    rows: list[dict[str, Any]], figures_dir: Path, *, primary_fold: str, base_seed: int
) -> None:
    lookup = _row_lookup(rows)
    variants = list(
        dict.fromkeys(
            str(row["variant"])
            for row in rows
            if row["source"] == "model"
            and row["fold"] == primary_fold
            and row["family"] == "input_ablation"
            and int(row["seed"]) == base_seed
        )
    )
    metrics = (
        ("root_error_mean_m", "Root"),
        ("MPJPE_cam_m", "MPJPE cam"),
        ("reprojection_error_mean_px", "Reprojection"),
        ("root_velocity_error_mean_m", "Velocity"),
    )
    if not variants:
        return
    row_labels: list[str] = []
    matrix: list[list[float]] = []
    for architecture in ("tcn", "transformer"):
        reference = lookup.get((primary_fold, architecture, base_seed, "full", "test"))
        if reference is None:
            continue
        for variant in variants:
            item = lookup.get((primary_fold, architecture, base_seed, variant, "test"))
            if item is None:
                continue
            values = []
            for metric, _ in metrics:
                ref_value = _finite_float(reference.get(metric))
                value = _finite_float(item.get(metric))
                values.append(100.0 * (value - ref_value) / ref_value if ref_value and np.isfinite(value) else np.nan)
            matrix.append(values)
            row_labels.append(f"{architecture.upper()} {variant.replace('_', ' ')}")
    if not matrix:
        return
    data = np.asarray(matrix, dtype=float)
    finite_abs = np.abs(data[np.isfinite(data)])
    limit = max(float(np.quantile(finite_abs, 0.95)) if finite_abs.size else 1.0, 1.0)
    fig_height = max(4.5, 0.42 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_height))
    image = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for row_index in range(data.shape[0]):
        for column_index in range(data.shape[1]):
            value = data[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                "n/a" if not np.isfinite(value) else f"{value:+.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if np.isfinite(value) and abs(value) > 0.55 * limit else "black",
            )
    fig.colorbar(image, ax=ax, label="Relative error change (%)")
    _save_figure(fig, figures_dir, "06_input_ablation_metric_heatmap")


def _plot_seed_stability(
    rows: list[dict[str, Any]], figures_dir: Path, *, primary_fold: str
) -> None:
    selected = [
        row
        for row in rows
        if row["source"] == "model"
        and row["fold"] == primary_fold
        and row["split"] == "test"
        and row["variant"] == "full"
        and row["architecture"] in {"tcn", "transformer"}
    ]
    seeds = sorted({int(row["seed"]) for row in selected})
    if len(seeds) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for architecture in ("tcn", "transformer"):
        arch_rows = sorted(
            [row for row in selected if row["architecture"] == architecture], key=lambda row: int(row["seed"])
        )
        ax.plot(
            [str(row["seed"]) for row in arch_rows],
            [100.0 * _finite_float(row["root_error_mean_m"]) for row in arch_rows],
            marker="o",
            linewidth=2,
            label=architecture.upper(),
            color=ARCH_COLORS[architecture],
        )
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean root error (cm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir, "07_seed_stability")


def _markdown_table(rows: list[list[str]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def aggregate_campaign(manifest_path: str | Path) -> dict[str, Any]:
    _, manifest = load_manifest(manifest_path)
    plan = load_plan(manifest)
    output_dir = campaign_output_dir(manifest)
    summary_dir = output_dir / "summary"
    figures_dir = summary_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)

    statistics = manifest.get("statistics", {}) or {}
    bootstrap_samples = int(statistics.get("bootstrap_samples", 2000))
    bootstrap_seed = int(statistics.get("bootstrap_seed", 314159))
    confidence = float(statistics.get("confidence_level", 0.95))
    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid campaign bootstrap settings")

    result_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    run_by_name = {str(run["run_name"]): run for run in plan["runs"]}
    for run in plan["runs"]:
        run_name = str(run["run_name"])
        path = metrics_path(manifest, run_name)
        state_path = output_dir / "states" / f"{run_name}.json"
        state_status = "not_started"
        if state_path.exists():
            try:
                state_status = str(json.loads(state_path.read_text(encoding="utf-8")).get("status", "unknown"))
            except (OSError, json.JSONDecodeError):
                state_status = "invalid_state"
        status_rows.append(
            {
                "index": run["index"],
                "run_name": run_name,
                "architecture": run["architecture"],
                "fold": run["fold"],
                "seed": run["seed"],
                "family": run["family"],
                "variant": run["variant"],
                "state": state_status,
                "metrics_present": path.exists(),
            }
        )
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            status_rows[-1]["state"] = f"invalid_metrics: {exc}"
            status_rows[-1]["metrics_present"] = False
            continue
        splits_payload = payload.get("splits", {})
        metrics_complete = isinstance(splits_payload, dict)
        for required_split in ("valid", "test"):
            split_payload = splits_payload.get(required_split) if isinstance(splits_payload, dict) else None
            root_error = _finite_float(split_payload.get("root_error_mean_m")) if isinstance(split_payload, dict) else float("nan")
            metrics_complete = metrics_complete and np.isfinite(root_error)
        status_rows[-1]["metrics_present"] = bool(metrics_complete)
        if not metrics_complete and state_status == "complete":
            status_rows[-1]["state"] = "partial_metrics"
        split_items = splits_payload.items() if isinstance(splits_payload, dict) else ()
        for split, split_metrics in split_items:
            if not isinstance(split_metrics, dict):
                continue
            row: dict[str, Any] = {
                "source": "model",
                "run_name": run_name,
                "run_id": run["run_id"],
                "architecture": run["architecture"],
                "fold": run["fold"],
                "seed": run["seed"],
                "family": run["family"],
                "variant": run["variant"],
                "label": run["label"],
                "prediction_mode": run["prediction_mode"],
                "window_size": run.get("window_size"),
                "split": split,
            }
            row.update({metric: _finite_float(split_metrics.get(metric)) for metric in METRICS})
            prediction_file = predictions_path(manifest, run_name, str(split))
            if prediction_file.exists():
                errors, clusters = _prediction_errors(prediction_file)
                low, high = _cluster_bootstrap_ci(
                    errors,
                    clusters,
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed=bootstrap_seed + int(run["index"]) * 7 + (0 if split == "valid" else 1),
                )
                row["root_error_ci_low_m"] = low
                row["root_error_ci_high_m"] = high
                row["num_prediction_frames"] = int(errors.size)
            result_rows.append(row)

        baseline_reports = payload.get("comparison_reports", {}).get("baseline_mean_root", {})
        for split, split_metrics in baseline_reports.items():
            row = {
                "source": "mean_root_baseline",
                "run_name": f"{run['fold']}_mean_root_baseline",
                "run_id": f"{run['fold']}_mean_root_baseline",
                "architecture": "mean_root",
                "fold": run["fold"],
                "seed": run["seed"],
                "family": "baseline",
                "variant": "mean_root",
                "label": "Training mean root",
                "prediction_mode": "absolute",
                "window_size": run.get("window_size"),
                "split": split,
            }
            row.update({metric: _finite_float(split_metrics.get(metric)) for metric in METRICS})
            result_rows.append(row)

    for fold in manifest["folds"]:
        path = output_dir / "baselines" / str(fold) / "geometry_metrics.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for split, split_metrics in payload.get("splits", {}).items():
            row = {
                "source": "geometry_baseline",
                "run_name": f"{fold}_geometry_baseline",
                "run_id": f"{fold}_geometry_baseline",
                "architecture": "geometry",
                "fold": fold,
                "seed": plan["base_seed"],
                "family": "baseline",
                "variant": "root_init",
                "label": "Ground-intersection root init",
                "prediction_mode": "zero_delta",
                "window_size": None,
                "split": split,
            }
            row.update({metric: _finite_float(split_metrics.get(metric)) for metric in METRICS})
            result_rows.append(row)

    primary_fold = str(plan["primary_fold"])
    base_seed = int(plan["base_seed"])
    lookup = _row_lookup(result_rows)
    effect_rows: list[dict[str, Any]] = []
    for row in result_rows:
        if (
            row["source"] != "model"
            or row["fold"] != primary_fold
            or int(row["seed"]) != base_seed
            or row["architecture"] not in {"tcn", "transformer"}
            or row["variant"] == "full"
        ):
            continue
        reference = lookup.get((primary_fold, str(row["architecture"]), base_seed, "full", str(row["split"])))
        if reference is None:
            continue
        value = _finite_float(row["root_error_mean_m"])
        ref_value = _finite_float(reference["root_error_mean_m"])
        effect: dict[str, Any] = {
            "architecture": row["architecture"],
            "fold": row["fold"],
            "seed": row["seed"],
            "family": row["family"],
            "variant": row["variant"],
            "label": row["label"],
            "split": row["split"],
            "reference_run": reference["run_name"],
            "variant_run": row["run_name"],
            "reference_root_error_m": ref_value,
            "variant_root_error_m": value,
            "delta_root_error_m": value - ref_value,
            "relative_root_error_change_pct": 100.0 * (value - ref_value) / ref_value if ref_value else float("nan"),
        }
        variant_predictions = predictions_path(manifest, str(row["run_name"]), str(row["split"]))
        reference_predictions = predictions_path(manifest, str(reference["run_name"]), str(row["split"]))
        if variant_predictions.exists() and reference_predictions.exists():
            paired = _paired_effect_ci(
                variant_predictions,
                reference_predictions,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=bootstrap_seed + int(run_by_name[str(row["run_name"])]["index"]) * 13,
            )
            if paired is not None:
                effect["paired_delta_root_error_m"] = paired[0]
                effect["paired_delta_ci_low_m"] = paired[1]
                effect["paired_delta_ci_high_m"] = paired[2]
                effect["paired_significant"] = bool(paired[1] > 0.0 or paired[2] < 0.0)
            else:
                effect["paired_note"] = "prediction identifiers differ; unpaired aggregate effect only"
        effect_rows.append(effect)

    seed_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in result_rows:
        if (
            row["source"] == "model"
            and row["fold"] == primary_fold
            and row["variant"] == "full"
            and row["architecture"] in {"tcn", "transformer"}
        ):
            grouped[(str(row["architecture"]), str(row["split"]))].append(row)
    for (architecture, split), items in grouped.items():
        values = np.asarray([_finite_float(item["root_error_mean_m"]) for item in items], dtype=float)
        values = values[np.isfinite(values)]
        seed_rows.append(
            {
                "architecture": architecture,
                "fold": primary_fold,
                "split": split,
                "num_seeds": int(values.size),
                "root_error_mean_over_seeds_m": float(values.mean()) if values.size else float("nan"),
                "root_error_std_over_seeds_m": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda value: int(value["seed"]))),
            }
        )

    _write_csv(summary_dir / "all_metrics.csv", result_rows)
    _write_csv(summary_dir / "ablation_effects.csv", effect_rows)
    _write_csv(summary_dir / "seed_summary.csv", seed_rows)
    _write_csv(summary_dir / "run_status.csv", status_rows)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    _plot_overview(result_rows, figures_dir, primary_fold=primary_fold, base_seed=base_seed)
    _plot_input_ablation(effect_rows, figures_dir)
    _plot_formulation(result_rows, figures_dir, primary_fold=primary_fold, base_seed=base_seed)
    _plot_match_robustness(result_rows, figures_dir, base_seed=base_seed)
    _plot_error_cdf(manifest, result_rows, figures_dir, primary_fold=primary_fold, base_seed=base_seed)
    _plot_metric_heatmap(result_rows, figures_dir, primary_fold=primary_fold, base_seed=base_seed)
    _plot_seed_stability(result_rows, figures_dir, primary_fold=primary_fold)

    complete_runs = sum(bool(row["metrics_present"]) for row in status_rows)
    overview_rows = [
        row
        for row in result_rows
        if row["fold"] == primary_fold
        and row["split"] == "test"
        and (
            row["source"] in {"geometry_baseline", "mean_root_baseline"}
            or (row["source"] == "model" and row["variant"] == "full" and int(row["seed"]) == base_seed)
        )
    ]
    overview_table = []
    for row in overview_rows:
        value = 100.0 * _finite_float(row["root_error_mean_m"])
        low = 100.0 * _finite_float(row.get("root_error_ci_low_m"))
        high = 100.0 * _finite_float(row.get("root_error_ci_high_m"))
        interval = f"[{low:.2f}, {high:.2f}]" if np.isfinite(low) and np.isfinite(high) else "n/a"
        overview_table.append([str(row["label"]), f"{value:.2f}", interval])

    effect_table = []
    for row in effect_rows:
        if row["split"] != "test" or row["family"] != "input_ablation":
            continue
        effect_table.append(
            [
                str(row["architecture"]).upper(),
                str(row["variant"]),
                f"{100.0 * _finite_float(row['delta_root_error_m']):+.2f}",
                str(row.get("paired_significant", "n/a")),
            ]
        )

    report_lines = [
        f"# Ablation campaign: {manifest['campaign_name']}",
        "",
        f"Completed learned runs: **{complete_runs}/{len(status_rows)}**.",
        f"Primary fold: **{primary_fold}**. Base seed: **{base_seed}**.",
        f"Cluster bootstrap: **{bootstrap_samples}** resamples, confidence level **{confidence:.0%}**.",
        "",
        "## Primary unseen-match test",
        "",
        _markdown_table(overview_table, ["Method", "Root error (cm)", "Cluster CI (cm)"])
        if overview_table
        else "No completed primary results yet.",
        "",
        "## Input ablations",
        "",
        _markdown_table(effect_table, ["Model", "Ablation", "Delta root error (cm)", "Paired CI excludes 0"])
        if effect_table
        else "No completed input ablations yet.",
        "",
        "Positive deltas indicate worse performance than the full model. Intervals are clustered by sequence.",
    ]
    missing = [str(row["run_name"]) for row in status_rows if not row["metrics_present"]]
    if missing:
        report_lines.extend(["", "## Missing runs", "", *[f"- `{name}`" for name in missing]])
    (summary_dir / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "campaign_name": manifest["campaign_name"],
        "num_planned_runs": len(status_rows),
        "num_completed_runs": complete_runs,
        "num_missing_runs": len(status_rows) - complete_runs,
        "num_metric_rows": len(result_rows),
        "num_effect_rows": len(effect_rows),
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": confidence,
        "summary_dir": str(summary_dir),
    }
    write_json_atomic(summary_dir / "summary.json", summary)
    print(f"Aggregated {complete_runs}/{len(status_rows)} learned runs into {summary_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate campaign metrics, statistics, and publication figures")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    aggregate_campaign(args.manifest)


if __name__ == "__main__":
    main()
