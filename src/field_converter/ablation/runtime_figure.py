"""Generate publication runtime statistics from completed L40S experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


ARCHITECTURES = ("mlp", "tcn", "transformer")
ARCHITECTURE_LABELS = {
    "mlp": "MLP",
    "tcn": "TCN",
    "transformer": "Transformer",
}
ARCHITECTURE_COLORS = {
    "mlp": "#007C91",
    "tcn": "#D55E00",
    "transformer": "#3B5BA9",
}
COMPLETED_EPOCH_RE = re.compile(
    r"train epoch (\d+): 100%\|[^\n]*?\|\s*(\d+)/(\d+) "
    r"\[([0-9:]+)<"
)


@dataclass(frozen=True)
class RuntimeMeasurement:
    run_name: str
    architecture: str
    fold: str
    batch_size: int
    window_size: int
    completed_epochs: int
    training_player_frames_per_s: float
    evaluation_duration_s: float
    evaluation_video_frames: int
    evaluation_player_frames: int
    inference_video_fps: float
    inference_player_frames_per_s: float
    training_log: str


def _duration_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return 60 * minutes + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return 3600 * hours + 60 * minutes + seconds
    raise ValueError(f"Unsupported tqdm duration: {value}")


def _architecture_from_run_name(run_name: str) -> str:
    for architecture in ARCHITECTURES:
        if architecture in run_name:
            return architecture
    raise ValueError(f"Cannot infer architecture from run name: {run_name}")


def _index_successful_l40s_slurms(slurm_dir: Path) -> dict[str, Path]:
    """Map evaluated run names to their training stderr logs."""
    indexed: dict[str, tuple[float, Path]] = {}
    metrics_re = re.compile(r"/eval_reports/([^/\s]+)/metrics\.json")
    for stdout_path in slurm_dir.glob("slurm_*.out"):
        text = stdout_path.read_text(encoding="utf-8", errors="ignore")
        if "NVIDIA L40S" not in text or "Training done" not in text:
            continue
        run_names = metrics_re.findall(text)
        if not run_names:
            continue
        stderr_path = stdout_path.with_suffix(".err")
        if not stderr_path.exists():
            continue
        mtime = stdout_path.stat().st_mtime
        for run_name in run_names:
            previous = indexed.get(run_name)
            if previous is None or mtime > previous[0]:
                indexed[run_name] = (mtime, stderr_path)
    return {run_name: item[1] for run_name, item in indexed.items()}


def _random_search_training_log(campaign_dir: Path, run_name: str) -> Path | None:
    candidates = list(
        (campaign_dir / "random_search").glob(f"*/logs/{run_name}_train.log")
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _training_throughput(
    log_path: Path,
    *,
    batch_size: int,
    window_size: int,
) -> tuple[float, int]:
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    completed_epochs: dict[int, tuple[int, int]] = {}
    for match in COMPLETED_EPOCH_RE.finditer(text):
        epoch, current, total, elapsed = match.groups()
        if current != total:
            continue
        elapsed_s = _duration_seconds(elapsed)
        if elapsed_s > 0:
            completed_epochs[int(epoch)] = (int(total), elapsed_s)

    if not completed_epochs:
        raise ValueError(f"No completed training epoch found in {log_path}")

    epoch_rates = [
        total_batches * batch_size * window_size / elapsed_s
        for total_batches, elapsed_s in completed_epochs.values()
    ]
    return statistics.mean(epoch_rates), len(completed_epochs)


def _evaluation_counts(prediction_dir: Path) -> tuple[int, int]:
    """Count unique video frames and valid player-frame predictions."""
    player_frames = 0
    video_frames = 0
    for split in ("valid", "test"):
        prediction_path = prediction_dir / f"{split}_predictions.npz"
        with np.load(prediction_path, allow_pickle=True) as predictions:
            sequence_ids = predictions["seq_id"].astype(np.int64, copy=False)
            frame_ids = predictions["frame_idx"].astype(np.int64, copy=False)
        player_frames += int(frame_ids.size)
        packed_frame_ids = (sequence_ids << np.int64(32)) | frame_ids
        video_frames += int(np.unique(packed_frame_ids).size)
    return video_frames, player_frames


def _discover_measurements(
    campaign_dir: Path,
    slurm_dir: Path,
) -> list[RuntimeMeasurement]:
    reports_dir = campaign_dir / "eval_reports"
    predictions_dir = campaign_dir / "predictions"
    slurm_logs = _index_successful_l40s_slurms(slurm_dir)
    counts_by_data_dir: dict[str, tuple[int, int]] = {}
    measurements: list[RuntimeMeasurement] = []

    for report_dir in sorted(reports_dir.iterdir()):
        if not report_dir.is_dir():
            continue
        run_name = report_dir.name
        prediction_dir = predictions_dir / run_name
        required_paths = (
            report_dir / "train_summary.json",
            report_dir / "metrics.json",
            report_dir / "config_used.yaml",
            prediction_dir / "valid_predictions.npz",
            prediction_dir / "test_predictions.npz",
        )
        if not all(path.exists() for path in required_paths):
            continue

        log_path = slurm_logs.get(run_name)
        if log_path is None:
            log_path = _random_search_training_log(campaign_dir, run_name)
        if log_path is None or not log_path.exists():
            continue

        config = yaml.safe_load((report_dir / "config_used.yaml").read_text())
        architecture = _architecture_from_run_name(run_name)
        batch_size = int(config["training"]["batch_size"])
        window_size = int(config.get("dataset", {}).get("window_size") or 1)
        training_fps, completed_epochs = _training_throughput(
            log_path,
            batch_size=batch_size,
            window_size=window_size,
        )

        data_dir = str(config["data_dir"])
        if data_dir not in counts_by_data_dir:
            counts_by_data_dir[data_dir] = _evaluation_counts(prediction_dir)
        video_frames, player_frames = counts_by_data_dir[data_dir]

        training_end = (report_dir / "train_summary.json").stat().st_mtime
        evaluation_end = (prediction_dir / "test_predictions.npz").stat().st_mtime
        evaluation_duration_s = evaluation_end - training_end
        if evaluation_duration_s <= 0:
            raise ValueError(f"Invalid evaluation duration for {run_name}")

        fold = "secondary" if "/secondary/" in data_dir else "primary"
        measurements.append(
            RuntimeMeasurement(
                run_name=run_name,
                architecture=architecture,
                fold=fold,
                batch_size=batch_size,
                window_size=window_size,
                completed_epochs=completed_epochs,
                training_player_frames_per_s=training_fps,
                evaluation_duration_s=evaluation_duration_s,
                evaluation_video_frames=video_frames,
                evaluation_player_frames=player_frames,
                inference_video_fps=video_frames / evaluation_duration_s,
                inference_player_frames_per_s=player_frames / evaluation_duration_s,
                training_log=str(log_path),
            )
        )

    missing_architectures = set(ARCHITECTURES) - {
        measurement.architecture for measurement in measurements
    }
    if missing_architectures:
        names = ", ".join(sorted(missing_architectures))
        raise RuntimeError(f"No successful measurements found for: {names}")
    return measurements


def _summary(measurements: list[RuntimeMeasurement]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[RuntimeMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.architecture].append(measurement)

    summary: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURES:
        rows = grouped[architecture]
        architecture_summary: dict[str, Any] = {
            "n_runs": len(rows),
            "run_names": [row.run_name for row in rows],
        }
        for field in (
            "training_player_frames_per_s",
            "inference_video_fps",
            "inference_player_frames_per_s",
        ):
            values = [float(getattr(row, field)) for row in rows]
            architecture_summary[field] = {
                "mean": statistics.mean(values),
                "standard_deviation": statistics.stdev(values)
                if len(values) > 1
                else 0.0,
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        summary[architecture] = architecture_summary
    return summary


def _write_raw_csv(path: Path, measurements: list[RuntimeMeasurement]) -> None:
    rows = [asdict(measurement) for measurement in measurements]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(path: Path, summary: dict[str, dict[str, Any]]) -> None:
    fields = (
        "training_player_frames_per_s",
        "inference_video_fps",
        "inference_player_frames_per_s",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "architecture",
                "n_runs",
                "metric",
                "mean",
                "standard_deviation",
                "median",
                "minimum",
                "maximum",
            ]
        )
        for architecture in ARCHITECTURES:
            for field in fields:
                values = summary[architecture][field]
                writer.writerow(
                    [
                        architecture,
                        summary[architecture]["n_runs"],
                        field,
                        values["mean"],
                        values["standard_deviation"],
                        values["median"],
                        values["minimum"],
                        values["maximum"],
                    ]
                )


def _plot_runtime(
    output_path: Path,
    measurements: list[RuntimeMeasurement],
    summary: dict[str, dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)
    panels = (
        (
            "training_player_frames_per_s",
            r"Input player-frames s$^{-1}$",
            "Training throughput",
            True,
        ),
        (
            "inference_video_fps",
            r"Video frames s$^{-1}$",
            "End-to-end inference",
            False,
        ),
    )

    for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
        field, ylabel, title, log_scale = panel
        for x, architecture in enumerate(ARCHITECTURES):
            rows = [
                row for row in measurements if row.architecture == architecture
            ]
            values = np.asarray([getattr(row, field) for row in rows], dtype=float)
            offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else [0]
            color = ARCHITECTURE_COLORS[architecture]
            axis.scatter(
                x + np.asarray(offsets),
                values,
                s=20,
                facecolors="white",
                edgecolors=color,
                linewidths=0.9,
                alpha=0.9,
                zorder=3,
                label="Successful run" if x == 0 else None,
            )
            stats = summary[architecture][field]
            axis.errorbar(
                x,
                stats["mean"],
                yerr=stats["standard_deviation"],
                fmt="o",
                markersize=5.8,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.6,
                ecolor="black",
                elinewidth=1.2,
                capsize=3.5,
                capthick=1.2,
                zorder=5,
                label="Mean $\pm$ SD" if x == 0 else None,
            )

        labels = [
            f"{ARCHITECTURE_LABELS[architecture]}\n"
            f"($n={summary[architecture]['n_runs']}$)"
            for architecture in ARCHITECTURES
        ]
        axis.set_xticks(range(len(ARCHITECTURES)), labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title, pad=7)
        axis.text(
            -0.14,
            1.04,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        axis.grid(axis="y", color="#D6D6D6", linewidth=0.65, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.3,
    )
    figure.savefig(output_path, bbox_inches="tight")


def _publication_paragraph(summary: dict[str, dict[str, Any]]) -> str:
    def train(architecture: str) -> tuple[float, float]:
        values = summary[architecture]["training_player_frames_per_s"]
        return values["mean"] / 1000.0, values["standard_deviation"] / 1000.0

    def infer(architecture: str) -> tuple[float, float]:
        values = summary[architecture]["inference_video_fps"]
        return values["mean"], values["standard_deviation"]

    mlp_train, mlp_train_sd = train("mlp")
    tcn_train, tcn_train_sd = train("tcn")
    transformer_train, transformer_train_sd = train("transformer")
    mlp_infer, mlp_infer_sd = infer("mlp")
    tcn_infer, tcn_infer_sd = infer("tcn")
    transformer_infer, transformer_infer_sd = infer("transformer")
    n_runs = sum(int(summary[architecture]["n_runs"]) for architecture in ARCHITECTURES)

    return (
        "\\paragraph{Computational efficiency.} "
        "Runtime was measured on a single NVIDIA L40S GPU (46,068~MiB visible "
        "memory) with PyTorch~2.6 and CUDA~12.4, using all "
        f"{n_runs} successfully completed full-data runs from the unseen-match "
        "protocol. Mean training throughput was "
        f"$({mlp_train:.2f}\\pm{mlp_train_sd:.2f})\\times10^3$, "
        f"$({tcn_train:.1f}\\pm{tcn_train_sd:.1f})\\times10^3$, and "
        f"$({transformer_train:.1f}\\pm{transformer_train_sd:.1f})\\times10^3$ "
        "input player-frames~s$^{-1}$ for the MLP, TCN, and Transformer, "
        "respectively. "
        "End-to-end batched evaluation, including data loading, temporal-window "
        "aggregation where applicable, metric computation, and prediction "
        "serialization, reached "
        f"${mlp_infer:.0f}\\pm{mlp_infer_sd:.0f}$, "
        f"${tcn_infer:.0f}\\pm{tcn_infer_sd:.0f}$, and "
        f"${transformer_infer:.0f}\\pm{transformer_infer_sd:.0f}$ video "
        "frames~s$^{-1}$. "
        "Values are means $\\pm$ standard deviations across successful "
        "configurations; the dispersion therefore captures changes in inputs and "
        "temporal context rather than repeated-run confidence intervals. Upstream "
        "tracking, camera calibration, and SAM3DBody inference are excluded.\n"
    )


def generate_runtime_package(
    campaign_dir: Path,
    slurm_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = _discover_measurements(campaign_dir, slurm_dir)
    summary = _summary(measurements)

    raw_csv = output_dir / "runtime_measurements.csv"
    summary_csv = output_dir / "runtime_summary.csv"
    summary_json = output_dir / "runtime_summary.json"
    figure_pdf = output_dir / "runtime_throughput_by_architecture.pdf"
    figure_png = output_dir / "runtime_throughput_by_architecture.png"
    paragraph_tex = output_dir / "runtime_paragraph.tex"

    _write_raw_csv(raw_csv, measurements)
    _write_summary_csv(summary_csv, summary)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_runtime(figure_pdf, measurements, summary)
    _plot_runtime(figure_png, measurements, summary)
    paragraph_tex.write_text(_publication_paragraph(summary), encoding="utf-8")

    return {
        "n_runs": len(measurements),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "figure_pdf": str(figure_pdf),
        "figure_png": str(figure_png),
        "paragraph_tex": str(paragraph_tex),
        "summary": summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign_dir",
        type=Path,
        default=Path("outputs/ablation/unseen_match_v1"),
    )
    parser.add_argument("--slurm_dir", type=Path, default=Path("slurms"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/ablation/unseen_match_v1/publication/runtime"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = generate_runtime_package(
        campaign_dir=args.campaign_dir,
        slurm_dir=args.slurm_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
