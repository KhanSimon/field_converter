from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_METRIC = "best_root_error_mean_m"


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    elapsed_s: float
    timed_out: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _absolute(path: Path | str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "screening_rank",
        "source_trial",
        "run_name",
        "status",
        "screening_validation_root_error_m",
        "full_validation_root_error_m",
        "best_epoch",
        "last_epoch",
        "elapsed_s",
        "returncode",
        "config_path",
        "log_path",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _upsert_result(path: Path, row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = _read_csv(path) if path.exists() else []
    trial = str(row["source_trial"])
    rows = [existing for existing in rows if str(existing.get("source_trial")) != trial]
    rows.append(row)
    rows.sort(key=lambda item: int(item["screening_rank"]))
    _write_csv(path, rows)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_paths(output_dir: Path, run_name: str) -> tuple[Path, Path, Path]:
    return (
        output_dir / "checkpoints" / run_name / "best.pt",
        output_dir / "eval_reports" / run_name / "train_summary.json",
        output_dir / "eval_reports" / run_name / "metrics.json",
    )


def _training_complete(checkpoint: Path, summary_path: Path) -> bool:
    summary = _read_json(summary_path)
    return checkpoint.exists() and math.isfinite(_to_float(summary.get(SELECTION_METRIC)))


def _python_environment() -> dict[str, str]:
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _run_logged(cmd: list[str], *, log_path: Path, timeout_s: float) -> ProcessResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"command: {' '.join(cmd)}\nstarted_utc: {_utc_now()}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=_python_environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=max(1.0, timeout_s))
            timed_out = False
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            handle.write(f"\nTimed out after {timeout_s:.1f} seconds.\n")
            handle.flush()
            returncode = 124
            timed_out = True
        except BaseException:
            _terminate_process_group(proc)
            raise
    return ProcessResult(
        returncode=int(returncode),
        elapsed_s=time.monotonic() - started,
        timed_out=timed_out,
    )


def _train_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "field_converter.training.train_root_transformer",
        "--config",
        str(config_path),
    ]


def _evaluate_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "field_converter.training.evaluate_root_transformer",
        "--config",
        str(config_path),
        "--checkpoint",
        "best",
        "--no_baseline",
    ]


def _full_config(
    screening_config: Path,
    *,
    run_name: str,
    output_dir: str,
    seed: int,
    epochs: int,
    patience: int,
    train_num_workers: int,
    eval_num_workers: int,
    search_name: str,
    source_trial: int,
    screening_rank: int,
) -> dict[str, Any]:
    cfg = _load_yaml(screening_config)
    cfg["run_name"] = run_name
    cfg["seed"] = int(seed)
    cfg["output_dir"] = output_dir
    cfg.setdefault("dataset", {})["max_sequences"] = None
    cfg["dataset"]["max_windows_per_sequence"] = None
    cfg.setdefault("training", {})["epochs"] = int(epochs)
    cfg["training"]["early_stopping_patience"] = int(patience)
    cfg["training"]["num_workers"] = int(train_num_workers)
    cfg.setdefault("eval", {})["num_workers"] = int(eval_num_workers)
    cfg["eval"]["splits"] = ["valid", "test"]
    cfg["random_search_metadata"] = {
        "search_name": search_name,
        "source_trial": int(source_trial),
        "source_screening_rank": int(screening_rank),
        "fidelity": "full_retrain_top3",
        "selection_split": "valid",
        "test_used_for_selection": False,
    }
    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the top-k Transformer screening trials at full budget")
    parser.add_argument(
        "--search-name", default="root_transformer_random_search_v2_competitive"
    )
    parser.add_argument("--output-dir", default="outputs/ablation/unseen_match_v1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--train-num-workers", type=int, default=2)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=31.0)
    parser.add_argument("--per-run-timeout-hours", type=float, default=9.0)
    parser.add_argument("--evaluation-reserve-hours", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("top_k", "epochs", "early_stopping_patience"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.train_num_workers < 0 or args.eval_num_workers < 0:
        raise ValueError("Worker counts must be >= 0")
    if args.max_hours <= args.evaluation_reserve_hours:
        raise ValueError("Total budget must exceed the evaluation reserve")
    if args.per_run_timeout_hours <= 0.0:
        raise ValueError("Per-run timeout must be positive")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    started = time.monotonic()
    hard_deadline = started + args.max_hours * 3600.0
    output_dir = _absolute(args.output_dir)
    campaign_dir = output_dir / "random_search" / args.search_name
    ranked_screening_path = campaign_dir / "random_search_results_ranked.csv"
    stage_dir = campaign_dir / f"full_budget_top{args.top_k}"
    configs_dir = stage_dir / "configs"
    logs_dir = stage_dir / "logs"
    results_path = stage_dir / "full_budget_results.csv"

    screening_rows = [
        row
        for row in _read_csv(ranked_screening_path)
        if math.isfinite(_to_float(row.get(SELECTION_METRIC)))
    ][: args.top_k]
    if len(screening_rows) < args.top_k:
        raise RuntimeError(f"Expected {args.top_k} completed screening trials, found {len(screening_rows)}")

    generated: list[dict[str, Any]] = []
    for rank, row in enumerate(screening_rows, start=1):
        source_trial = int(row["trial"])
        screening_config = _absolute(row["config_path"])
        run_name = f"{args.search_name}_top{rank:02d}_trial_{source_trial:03d}_full_budget"
        config_path = configs_dir / f"{run_name}.yaml"
        log_path = logs_dir / f"{run_name}_train.log"
        cfg = _full_config(
            screening_config,
            run_name=run_name,
            output_dir=args.output_dir,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.early_stopping_patience,
            train_num_workers=args.train_num_workers,
            eval_num_workers=args.eval_num_workers,
            search_name=args.search_name,
            source_trial=source_trial,
            screening_rank=rank,
        )
        _write_yaml(config_path, cfg)
        generated.append(
            {
                "screening_rank": rank,
                "source_trial": source_trial,
                "screening_run_name": row["run_name"],
                "screening_validation_root_error_m": _to_float(row[SELECTION_METRIC]),
                "run_name": run_name,
                "config_path": config_path,
                "log_path": log_path,
            }
        )

    _write_json(
        stage_dir / "protocol.json",
        {
            "search_name": args.search_name,
            "created_utc": _utc_now(),
            "top_k": args.top_k,
            "source_ranking": str(ranked_screening_path),
            "selection_split": "valid",
            "test_used_for_selection": False,
            "full_epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "training_seed_shared_by_candidates": args.seed,
            "max_hours": args.max_hours,
            "per_run_timeout_hours": args.per_run_timeout_hours,
            "candidates": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in candidate.items()
                    if key != "log_path"
                }
                for candidate in generated
            ],
        },
    )

    if args.dry_run:
        for candidate in generated:
            _upsert_result(
                results_path,
                {
                    **candidate,
                    "config_path": str(candidate["config_path"]),
                    "log_path": str(candidate["log_path"]),
                    "status": "dry_run",
                    "full_validation_root_error_m": None,
                    "best_epoch": None,
                    "last_epoch": None,
                    "elapsed_s": None,
                    "returncode": None,
                },
            )
        print(f"Dry run complete: generated {len(generated)} full-budget configs in {configs_dir}")
        return

    for index, candidate in enumerate(generated):
        run_name = str(candidate["run_name"])
        config_path = Path(candidate["config_path"])
        log_path = Path(candidate["log_path"])
        checkpoint, summary_path, _ = _run_paths(output_dir, run_name)
        result: ProcessResult | None = None

        if _training_complete(checkpoint, summary_path):
            status = "skipped_existing"
            print(f"[{index + 1}/{args.top_k}] reuse {run_name}")
        else:
            pending = args.top_k - index
            available_s = hard_deadline - time.monotonic() - args.evaluation_reserve_hours * 3600.0
            fair_share_s = available_s / max(1, pending)
            timeout_s = min(args.per_run_timeout_hours * 3600.0, fair_share_s)
            if timeout_s < 300.0:
                status = "not_run_budget"
                print(f"[{index + 1}/{args.top_k}] insufficient remaining budget for {run_name}")
            else:
                print(f"[{index + 1}/{args.top_k}] train {run_name} (timeout {timeout_s / 3600.0:.2f} h)")
                result = _run_logged(_train_command(config_path), log_path=log_path, timeout_s=timeout_s)
                if result.timed_out:
                    status = "timed_out"
                elif result.returncode != 0:
                    status = "failed"
                elif _training_complete(checkpoint, summary_path):
                    status = "success"
                else:
                    status = "missing_summary"

        summary = _read_json(summary_path)
        metric = _to_float(summary.get(SELECTION_METRIC))
        metric_text = f"{metric:.6f} m" if math.isfinite(metric) else "unavailable"
        print(f"[{index + 1}/{args.top_k}] {status}: full validation root error {metric_text}")
        _upsert_result(
            results_path,
            {
                **candidate,
                "config_path": str(config_path),
                "log_path": str(log_path),
                "status": status,
                "full_validation_root_error_m": metric if math.isfinite(metric) else None,
                "best_epoch": summary.get("best_epoch"),
                "last_epoch": summary.get("last_epoch"),
                "elapsed_s": None if result is None else round(result.elapsed_s, 3),
                "returncode": None if result is None else result.returncode,
                "updated_utc": _utc_now(),
            },
        )

    expected_trials = {int(candidate["source_trial"]) for candidate in generated}
    completed = [
        row
        for row in _read_csv(results_path)
        if int(row["source_trial"]) in expected_trials
        and math.isfinite(_to_float(row.get("full_validation_root_error_m")))
    ]
    completed.sort(key=lambda row: _to_float(row["full_validation_root_error_m"]))
    if not completed:
        raise RuntimeError(f"No full-budget candidate completed successfully; see {results_path}")
    _write_csv(stage_dir / "full_budget_results_ranked.csv", completed)

    best = completed[0]
    best_config = Path(best["config_path"])
    shutil.copyfile(best_config, stage_dir / "best_config.yaml")
    best_run_name = best["run_name"]
    best_checkpoint, best_summary_path, best_metrics_path = _run_paths(output_dir, best_run_name)

    evaluation_status = "skipped_existing"
    evaluation_result: ProcessResult | None = None
    if not best_metrics_path.exists():
        timeout_s = hard_deadline - time.monotonic() - 180.0
        if timeout_s <= 0.0:
            raise RuntimeError("No time remains for the selected candidate evaluation")
        evaluation_log = logs_dir / f"{best_run_name}_evaluation.log"
        print(f"Evaluate selected run on validation and test: {best_run_name}")
        evaluation_result = _run_logged(
            _evaluate_command(best_config),
            log_path=evaluation_log,
            timeout_s=timeout_s,
        )
        if evaluation_result.timed_out or evaluation_result.returncode != 0 or not best_metrics_path.exists():
            raise RuntimeError(f"Selected candidate evaluation failed; see {evaluation_log}")
        evaluation_status = "success"

    metrics = _read_json(best_metrics_path)
    selection = {
        "completed_utc": _utc_now(),
        "elapsed_hours": (time.monotonic() - started) / 3600.0,
        "selection_split": "valid",
        "selection_metric": "root_error_mean_m",
        "test_used_for_selection": False,
        "selected_screening_rank": int(best["screening_rank"]),
        "selected_source_trial": int(best["source_trial"]),
        "selected_run_name": best_run_name,
        "full_validation_root_error_m": _to_float(best["full_validation_root_error_m"]),
        "config_path": str(best_config),
        "checkpoint_path": str(best_checkpoint),
        "train_summary_path": str(best_summary_path),
        "metrics_path": str(best_metrics_path),
        "evaluation_status": evaluation_status,
        "evaluation_elapsed_s": None if evaluation_result is None else evaluation_result.elapsed_s,
        "metrics": metrics.get("splits", {}),
    }
    _write_json(stage_dir / "selection_summary.json", selection)

    print("Top-3 full-budget retraining complete")
    print(f"- selected run: {best_run_name}")
    print(f"- ranked results: {stage_dir / 'full_budget_results_ranked.csv'}")
    print(f"- final metrics: {best_metrics_path}")


if __name__ == "__main__":
    main()
