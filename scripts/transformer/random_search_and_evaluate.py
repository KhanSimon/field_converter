from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT / "configs" / "transformer" / "random_search_competitive_base.yaml"
)
DEFAULT_SEARCH_NAME = "root_transformer_random_search_v2_competitive"
SELECTION_METRIC = "best_root_error_mean_m"


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    elapsed_s: float
    timed_out: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part!r} is not a mapping")
        node = child
    node[parts[-1]] = value


def _get_nested(payload: dict[str, Any], dotted_key: str) -> Any:
    node: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing {dotted_key!r} in base config")
        node = node[part]
    return deepcopy(node)


def _round_float(value: float, digits: int = 8) -> float:
    return float(f"{value:.{digits}g}")


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0.0 or high <= 0.0 or low >= high:
        raise ValueError("Invalid log-uniform bounds")
    return 10.0 ** rng.uniform(math.log10(low), math.log10(high))


def _base_trial_hparams(base_cfg: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "dataset.window_size",
        "dataset.stride",
        "model.encoder_hidden_dims",
        "model.d_model",
        "model.num_layers",
        "model.num_heads",
        "model.dim_feedforward",
        "model.dropout",
        "model.positional_encoding",
        "model.max_window_size",
        "model.head_hidden_dims",
        "optimizer.lr",
        "optimizer.weight_decay",
    ]
    return {key: _get_nested(base_cfg, key) for key in keys}


def _sample_trial(rng: random.Random) -> dict[str, Any]:
    # The pairs keep overlap and compute in a comparable range. Larger windows
    # are intentionally capped at 81 frames so the final full-data run remains
    # compatible with the 32-hour campaign budget.
    window_size, stride = rng.choices(
        [(21, 8), (41, 8), (81, 20)],
        weights=[0.20, 0.55, 0.25],
        k=1,
    )[0]
    d_model = rng.choices([128, 192, 256], weights=[0.20, 0.30, 0.50], k=1)[0]
    num_layers = rng.choice([1, 2] if window_size == 81 else [1, 2, 3])
    num_heads = rng.choice([4, 8])
    feedforward_multiplier = rng.choice([2, 3, 4])

    encoder_options = [[d_model], [256]]
    if d_model != 256:
        encoder_options.append([256, d_model])

    weight_decay = 0.0 if rng.random() < 0.08 else _log_uniform(rng, 1e-5, 2e-3)
    return {
        "dataset.window_size": window_size,
        "dataset.stride": stride,
        "model.encoder_hidden_dims": rng.choice(encoder_options),
        "model.d_model": d_model,
        "model.num_layers": num_layers,
        "model.num_heads": num_heads,
        "model.dim_feedforward": d_model * feedforward_multiplier,
        "model.dropout": _round_float(rng.uniform(0.03, 0.25), 6),
        "model.positional_encoding": rng.choice(["learned", "sinusoidal"]),
        "model.max_window_size": window_size,
        "model.head_hidden_dims": [rng.choice([64, 128, 256])],
        "optimizer.lr": _round_float(_log_uniform(rng, 4e-5, 5e-4), 8),
        "optimizer.weight_decay": _round_float(weight_decay, 8),
    }


def _sample_campaign(
    *,
    base_cfg: dict[str, Any],
    n_trials: int,
    search_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(search_seed)
    trials = [_base_trial_hparams(base_cfg)]
    trials.extend(_sample_trial(rng) for _ in range(1, n_trials))
    return trials


def _build_trial_config(
    base_cfg: dict[str, Any],
    *,
    run_name: str,
    hparams: dict[str, Any],
    training_seed: int,
    output_dir: str,
    epochs: int,
    max_sequences: int | None,
    max_windows_per_sequence: int | None,
    train_num_workers: int,
    eval_num_workers: int,
    screening: bool,
    search_name: str,
    source_trial: int,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["run_name"] = run_name
    cfg["seed"] = int(training_seed)
    cfg["output_dir"] = output_dir

    for dotted_key, value in hparams.items():
        _set_nested(cfg, dotted_key, value)

    _set_nested(cfg, "training.epochs", int(epochs))
    _set_nested(cfg, "training.num_workers", int(train_num_workers))
    _set_nested(cfg, "eval.num_workers", int(eval_num_workers))
    if screening:
        _set_nested(cfg, "training.early_stopping_patience", int(epochs))
        _set_nested(cfg, "dataset.max_sequences", max_sequences)
        _set_nested(cfg, "dataset.max_windows_per_sequence", max_windows_per_sequence)

    cfg["random_search_metadata"] = {
        "search_name": search_name,
        "source_trial": int(source_trial),
        "fidelity": "screening" if screening else "full_retrain",
        "selection_split": "valid",
        "selection_metric": SELECTION_METRIC,
    }
    return cfg


def _flatten_hparams(hparams: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in hparams.items():
        row[key] = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
    return row


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "trial",
        "run_name",
        "status",
        SELECTION_METRIC,
        "best_epoch",
        "last_epoch",
        "elapsed_s",
        "returncode",
        "config_path",
        "log_path",
    ]
    fields = [key for key in preferred if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _upsert_trial_result(path: Path, row: dict[str, Any]) -> None:
    rows = _read_csv(path)
    trial = str(row["trial"])
    rows = [existing for existing in rows if str(existing.get("trial")) != trial]
    rows.append(row)
    rows.sort(key=lambda item: int(item["trial"]))
    _write_csv(path, rows)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _output_dir_absolute(output_dir: str) -> Path:
    path = Path(output_dir)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run_paths(output_dir: Path, run_name: str) -> tuple[Path, Path, Path]:
    checkpoint = output_dir / "checkpoints" / run_name / "best.pt"
    train_summary = output_dir / "eval_reports" / run_name / "train_summary.json"
    metrics = output_dir / "eval_reports" / run_name / "metrics.json"
    return checkpoint, train_summary, metrics


def _read_train_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _completed_training(checkpoint: Path, train_summary: Path) -> bool:
    summary = _read_train_summary(train_summary)
    metric = _to_float(summary.get(SELECTION_METRIC))
    return checkpoint.exists() and math.isfinite(metric)


def _python_environment() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
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


def _run_logged_process(cmd: list[str], *, log_path: Path, timeout_s: float) -> ProcessResult:
    if timeout_s <= 0.0:
        return ProcessResult(returncode=124, elapsed_s=0.0, timed_out=True)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"command: {' '.join(cmd)}\n")
        log_handle.write(f"started_utc: {_utc_now()}\n\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=_python_environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(proc)
            returncode = 124
            log_handle.write(f"\nTimed out after {timeout_s:.1f} seconds.\n")
            log_handle.flush()
        except BaseException:
            _terminate_process_group(proc)
            raise
    return ProcessResult(
        returncode=int(returncode),
        elapsed_s=time.monotonic() - started,
        timed_out=timed_out,
    )


def _training_command(config_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "field_converter.training.train_root_transformer",
        "--config",
        str(config_path),
    ]


def _evaluation_command(config_path: Path) -> list[str]:
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


def _trial_row(
    *,
    trial_idx: int,
    run_name: str,
    status: str,
    config_path: Path,
    log_path: Path,
    hparams: dict[str, Any],
    summary: dict[str, Any] | None = None,
    process_result: ProcessResult | None = None,
) -> dict[str, Any]:
    summary = summary or {}
    row: dict[str, Any] = {
        "trial": trial_idx,
        "run_name": run_name,
        "status": status,
        SELECTION_METRIC: summary.get(SELECTION_METRIC),
        "best_epoch": summary.get("best_epoch"),
        "last_epoch": summary.get("last_epoch"),
        "elapsed_s": None if process_result is None else round(process_result.elapsed_s, 3),
        "returncode": None if process_result is None else process_result.returncode,
        "config_path": str(config_path),
        "log_path": str(log_path),
        "updated_utc": _utc_now(),
    }
    row.update(_flatten_hparams(hparams))
    return row


def _rank_completed_trials(results_csv: Path) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for row in _read_csv(results_csv):
        metric = _to_float(row.get(SELECTION_METRIC))
        if math.isfinite(metric):
            completed.append(row)
    return sorted(completed, key=lambda row: _to_float(row.get(SELECTION_METRIC)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budgeted Transformer random search followed by full retraining and evaluation"
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--search-name", default=DEFAULT_SEARCH_NAME)
    parser.add_argument("--output-dir", default="outputs/ablation/unseen_match_v1")
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=20260830)
    parser.add_argument("--training-seed", type=int, default=1235)
    parser.add_argument("--search-epochs", type=int, default=8)
    parser.add_argument("--search-max-sequences", type=int, default=0, help="0 keeps all train sequences")
    parser.add_argument("--search-max-windows-per-sequence", type=int, default=300)
    parser.add_argument("--full-epochs", type=int, default=60)
    parser.add_argument("--train-num-workers", type=int, default=2)
    parser.add_argument("--eval-num-workers", type=int, default=2)
    parser.add_argument("--max-hours", type=float, default=31.0)
    parser.add_argument("--final-reserve-hours", type=float, default=10.0)
    parser.add_argument("--evaluation-reserve-hours", type=float, default=1.0)
    parser.add_argument("--per-trial-timeout-hours", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = [
        "n_trials",
        "search_epochs",
        "search_max_windows_per_sequence",
        "full_epochs",
    ]
    for field in positive_int_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be > 0")
    if args.search_max_sequences < 0:
        raise ValueError("--search-max-sequences must be >= 0")
    if args.train_num_workers < 0 or args.eval_num_workers < 0:
        raise ValueError("Worker counts must be >= 0")
    if args.max_hours <= 0.0 or args.per_trial_timeout_hours <= 0.0:
        raise ValueError("Time budgets must be positive")
    if args.final_reserve_hours <= args.evaluation_reserve_hours:
        raise ValueError("Final reserve must be larger than evaluation reserve")
    if args.final_reserve_hours >= args.max_hours:
        raise ValueError("Final reserve must be smaller than total budget")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    started = time.monotonic()
    hard_deadline = started + args.max_hours * 3600.0
    search_deadline = hard_deadline - args.final_reserve_hours * 3600.0

    base_config_path = args.base_config if args.base_config.is_absolute() else PROJECT_ROOT / args.base_config
    base_cfg = _load_yaml(base_config_path)
    output_dir = _output_dir_absolute(args.output_dir)
    search_dir = output_dir / "random_search" / args.search_name
    configs_dir = search_dir / "configs"
    logs_dir = search_dir / "logs"
    results_csv = search_dir / "random_search_results.csv"
    search_dir.mkdir(parents=True, exist_ok=True)

    hparams_by_trial = _sample_campaign(
        base_cfg=base_cfg,
        n_trials=args.n_trials,
        search_seed=args.search_seed,
    )
    max_sequences = None if args.search_max_sequences == 0 else int(args.search_max_sequences)

    _write_json(
        search_dir / "search_manifest.json",
        {
            "search_name": args.search_name,
            "created_utc": _utc_now(),
            "base_config": str(base_config_path),
            "selection_split": "valid",
            "held_out_test_split_used_during_search": False,
            "selection_metric": SELECTION_METRIC,
            "lower_is_better": True,
            "n_trials_requested": args.n_trials,
            "trial_zero_is_base_configuration": True,
            "search_seed": args.search_seed,
            "training_seed_shared_by_trials": args.training_seed,
            "fixed_input_config": deepcopy(base_cfg.get("input_config", {})),
            "screening_fidelity": {
                "epochs": args.search_epochs,
                "max_sequences": max_sequences,
                "max_windows_per_sequence": args.search_max_windows_per_sequence,
            },
            "full_retrain_epochs": args.full_epochs,
            "time_budget_hours": {
                "internal_total": args.max_hours,
                "reserved_for_full_retrain_and_evaluation": args.final_reserve_hours,
                "reserved_for_evaluation": args.evaluation_reserve_hours,
                "per_screening_trial_timeout": args.per_trial_timeout_hours,
            },
            "search_space": {
                "dataset.window_size_and_stride": [[21, 8], [41, 8], [81, 20]],
                "model.d_model": [128, 192, 256],
                "model.num_layers": [1, 2, 3],
                "model.num_heads": [4, 8],
                "model.dim_feedforward_multiplier": [2, 3, 4],
                "model.dropout": [0.03, 0.25],
                "model.positional_encoding": ["learned", "sinusoidal"],
                "model.head_hidden_dim": [64, 128, 256],
                "optimizer.lr_log_uniform": [4e-5, 5e-4],
                "optimizer.weight_decay_log_uniform": [1e-5, 2e-3],
            },
        },
    )

    generated: list[tuple[int, str, Path, Path, dict[str, Any]]] = []
    for trial_idx, hparams in enumerate(hparams_by_trial):
        run_name = f"{args.search_name}_screen_trial_{trial_idx:03d}"
        config_path = configs_dir / f"{run_name}.yaml"
        log_path = logs_dir / f"{run_name}.log"
        cfg = _build_trial_config(
            base_cfg,
            run_name=run_name,
            hparams=hparams,
            training_seed=args.training_seed,
            output_dir=args.output_dir,
            epochs=args.search_epochs,
            max_sequences=max_sequences,
            max_windows_per_sequence=args.search_max_windows_per_sequence,
            train_num_workers=args.train_num_workers,
            eval_num_workers=args.eval_num_workers,
            screening=True,
            search_name=args.search_name,
            source_trial=trial_idx,
        )
        _write_yaml(config_path, cfg)
        generated.append((trial_idx, run_name, config_path, log_path, hparams))

    if args.dry_run:
        for trial_idx, run_name, config_path, log_path, hparams in generated:
            _upsert_trial_result(
                results_csv,
                _trial_row(
                    trial_idx=trial_idx,
                    run_name=run_name,
                    status="dry_run",
                    config_path=config_path,
                    log_path=log_path,
                    hparams=hparams,
                ),
            )
        print(f"Dry run complete: generated {len(generated)} configs in {configs_dir}")
        return

    print(f"Random search: {args.search_name}")
    print(f"Screening candidates: {args.n_trials}; metric: validation root error")
    print(f"Search deadline in {(search_deadline - time.monotonic()) / 3600.0:.2f} h")

    for trial_idx, run_name, config_path, log_path, hparams in generated:
        checkpoint, train_summary_path, _ = _run_paths(output_dir, run_name)
        if _completed_training(checkpoint, train_summary_path):
            summary = _read_train_summary(train_summary_path)
            status = "skipped_existing"
            result = None
            print(f"[{trial_idx:03d}] reuse {run_name}: {summary[SELECTION_METRIC]:.6f} m")
        else:
            remaining_search_s = search_deadline - time.monotonic()
            if remaining_search_s < 300.0:
                _upsert_trial_result(
                    results_csv,
                    _trial_row(
                        trial_idx=trial_idx,
                        run_name=run_name,
                        status="not_run_budget",
                        config_path=config_path,
                        log_path=log_path,
                        hparams=hparams,
                    ),
                )
                print(f"[{trial_idx:03d}] search budget exhausted; keeping time for the full retrain")
                continue

            timeout_s = min(args.per_trial_timeout_hours * 3600.0, remaining_search_s)
            print(f"[{trial_idx:03d}] train {run_name} (timeout {timeout_s / 3600.0:.2f} h)")
            result = _run_logged_process(
                _training_command(config_path),
                log_path=log_path,
                timeout_s=timeout_s,
            )
            summary = _read_train_summary(train_summary_path)
            if result.timed_out:
                status = "timed_out"
            elif result.returncode != 0:
                status = "failed"
            elif not _completed_training(checkpoint, train_summary_path):
                status = "missing_summary"
            else:
                status = "success"

            metric = _to_float(summary.get(SELECTION_METRIC))
            metric_text = f"{metric:.6f} m" if math.isfinite(metric) else "unavailable"
            print(f"[{trial_idx:03d}] {status}: validation root error {metric_text}")

        _upsert_trial_result(
            results_csv,
            _trial_row(
                trial_idx=trial_idx,
                run_name=run_name,
                status=status,
                config_path=config_path,
                log_path=log_path,
                hparams=hparams,
                summary=summary,
                process_result=result,
            ),
        )

    ranked = _rank_completed_trials(results_csv)
    if not ranked:
        raise RuntimeError(f"No screening trial completed with a finite {SELECTION_METRIC}: {results_csv}")
    _write_csv(search_dir / "random_search_results_ranked.csv", ranked)

    best_row = ranked[0]
    best_trial_idx = int(best_row["trial"])
    best_hparams = hparams_by_trial[best_trial_idx]
    best_screen_config = Path(best_row["config_path"])
    shutil.copyfile(best_screen_config, search_dir / "best_screening_config.yaml")
    _write_json(
        search_dir / "best_screening_trial.json",
        {
            "trial": best_trial_idx,
            "run_name": best_row["run_name"],
            "validation_root_error_mean_m": _to_float(best_row[SELECTION_METRIC]),
            "config_path": str(best_screen_config),
            "hyperparameters": best_hparams,
        },
    )
    print(
        f"Best screening trial: {best_trial_idx:03d} "
        f"({SELECTION_METRIC}={_to_float(best_row[SELECTION_METRIC]):.6f} m)"
    )

    full_run_name = f"{args.search_name}_best_trial_{best_trial_idx:03d}_full"
    full_config_path = search_dir / "best_config.yaml"
    full_cfg = _build_trial_config(
        base_cfg,
        run_name=full_run_name,
        hparams=best_hparams,
        training_seed=args.training_seed,
        output_dir=args.output_dir,
        epochs=args.full_epochs,
        max_sequences=None,
        max_windows_per_sequence=None,
        train_num_workers=args.train_num_workers,
        eval_num_workers=args.eval_num_workers,
        screening=False,
        search_name=args.search_name,
        source_trial=best_trial_idx,
    )
    _write_yaml(full_config_path, full_cfg)

    full_checkpoint, full_train_summary_path, full_metrics_path = _run_paths(output_dir, full_run_name)
    full_train_status = "skipped_existing"
    full_train_result: ProcessResult | None = None
    if not _completed_training(full_checkpoint, full_train_summary_path):
        full_train_timeout_s = hard_deadline - time.monotonic() - args.evaluation_reserve_hours * 3600.0
        if full_train_timeout_s < 300.0:
            raise RuntimeError("Insufficient reserved time to start the full-data retraining")
        print(f"Full retrain: {full_run_name} (timeout {full_train_timeout_s / 3600.0:.2f} h)")
        full_train_result = _run_logged_process(
            _training_command(full_config_path),
            log_path=logs_dir / f"{full_run_name}_train.log",
            timeout_s=full_train_timeout_s,
        )
        if full_train_result.timed_out and full_checkpoint.exists():
            full_train_status = "timed_out_using_best_checkpoint"
        elif full_train_result.timed_out:
            raise RuntimeError("Full retraining timed out before producing a best checkpoint")
        elif full_train_result.returncode != 0:
            raise RuntimeError(
                f"Full retraining failed with code {full_train_result.returncode}; "
                f"see {logs_dir / f'{full_run_name}_train.log'}"
            )
        elif not full_checkpoint.exists():
            raise RuntimeError(f"Full retraining did not produce {full_checkpoint}")
        else:
            full_train_status = "success"
    else:
        print(f"Full retrain already complete: {full_checkpoint}")

    evaluation_status = "skipped_existing"
    evaluation_result: ProcessResult | None = None
    if not full_metrics_path.exists():
        evaluation_timeout_s = hard_deadline - time.monotonic() - 180.0
        if evaluation_timeout_s <= 0.0:
            raise RuntimeError("No time remains for final evaluation")
        print(f"Final validation/test evaluation (timeout {evaluation_timeout_s / 3600.0:.2f} h)")
        evaluation_result = _run_logged_process(
            _evaluation_command(full_config_path),
            log_path=logs_dir / f"{full_run_name}_evaluation.log",
            timeout_s=evaluation_timeout_s,
        )
        if evaluation_result.timed_out:
            raise RuntimeError("Final evaluation timed out")
        if evaluation_result.returncode != 0 or not full_metrics_path.exists():
            raise RuntimeError(
                f"Final evaluation failed with code {evaluation_result.returncode}; "
                f"see {logs_dir / f'{full_run_name}_evaluation.log'}"
            )
        evaluation_status = "success"
    else:
        print(f"Final metrics already exist: {full_metrics_path}")

    final_metrics = json.loads(full_metrics_path.read_text(encoding="utf-8"))
    final_summary = {
        "search_name": args.search_name,
        "completed_utc": _utc_now(),
        "elapsed_hours": (time.monotonic() - started) / 3600.0,
        "selection": {
            "split": "valid",
            "metric": SELECTION_METRIC,
            "trial": best_trial_idx,
            "screening_run_name": best_row["run_name"],
            "screening_metric_value": _to_float(best_row[SELECTION_METRIC]),
        },
        "full_run": {
            "run_name": full_run_name,
            "config_path": str(full_config_path),
            "checkpoint_path": str(full_checkpoint),
            "train_summary_path": str(full_train_summary_path),
            "metrics_path": str(full_metrics_path),
            "training_status": full_train_status,
            "evaluation_status": evaluation_status,
            "training_elapsed_s": None if full_train_result is None else full_train_result.elapsed_s,
            "evaluation_elapsed_s": None if evaluation_result is None else evaluation_result.elapsed_s,
        },
        "hyperparameters": best_hparams,
        "metrics": final_metrics.get("splits", {}),
    }
    _write_json(search_dir / "random_search_summary.json", final_summary)

    print("Random search and final evaluation complete")
    print(f"- best full config: {full_config_path}")
    print(f"- final metrics: {full_metrics_path}")
    print(f"- summary: {search_dir / 'random_search_summary.json'}")


if __name__ == "__main__":
    main()
