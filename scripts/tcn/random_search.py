from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "tcn" / "root_tcn_v1.yaml"
DEFAULT_SEARCH_NAME = "root_tcn_random_search"


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError("log_uniform bounds must be positive")
    return 10.0 ** rng.uniform(math.log10(low), math.log10(high))


def maybe_zero_log_uniform(
    rng: random.Random,
    *,
    zero_probability: float,
    low: float,
    high: float,
) -> float:
    if rng.random() < zero_probability:
        return 0.0
    return log_uniform(rng, low, high)


def round_float(value: float, digits: int = 8) -> float:
    return float(f"{value:.{digits}g}")


def sample_trial(rng: random.Random) -> dict[str, Any]:
    window_size = rng.choices([41, 81, 101, 201], weights=[0.15, 0.2, 0.35, 0.3], k=1)[0]
    stride_candidates = [s for s in [8, 20, 40, 60] if s <= window_size]
    temporal_hidden_dim = rng.choices([128, 192, 256], weights=[0.35, 0.35, 0.3], k=1)[0]
    head_hidden_dim = rng.choices([64, 128, 192], weights=[0.25, 0.55, 0.2], k=1)[0]

    if rng.random() < 0.75:
        encoder_hidden_dims = [temporal_hidden_dim, temporal_hidden_dim]
    else:
        encoder_hidden_dims = [max(128, temporal_hidden_dim // 2), temporal_hidden_dim]

    return {

        "dataset.window_size": window_size,
        "dataset.stride": rng.choice(stride_candidates),
        "model.encoder_hidden_dims": encoder_hidden_dims,
        "model.temporal_hidden_dim": temporal_hidden_dim,
        "model.temporal_dilations": rng.choice(
            [
                [1, 2, 4, 8],
                [1, 2, 4, 8, 16],
                [1, 2, 3, 4, 6, 8],
            ]
        ),
        "model.temporal_kernel_size": rng.choices([3, 5], weights=[0.85, 0.15], k=1)[0],
        "model.head_hidden_dims": [head_hidden_dim],
        "optimizer.lr": round_float(log_uniform(rng, 1e-5, 1e-3), 8),
        "optimizer.weight_decay": round_float(
            maybe_zero_log_uniform(rng, zero_probability=0.05, low=5e-5, high=3e-3),
            8,
        ),
        "loss_weights.root": 1.0,
        "loss_weights.root_axis_weights": [
            round_float(rng.uniform(0.9, 1.1), 4),
            round_float(rng.uniform(0.9, 1.1), 4),
            round_float(rng.uniform(1.2, 2.2), 4),
        ],
        "loss_weights.root_vel": round_float(log_uniform(rng, 0.05, 0.4), 8),
        "loss_weights.root_acc": round_float(
            maybe_zero_log_uniform(rng, zero_probability=0.35, low=0.003, high=0.05),
            8,
        ),
        "loss_weights.cam3d": round_float(rng.uniform(0, 1), 4),
        "loss_weights.proj": 0.0,
    }


def set_nested(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key}: {part} is not a mapping")
        node = child
    node[parts[-1]] = value


def apply_trial_to_config(
    base_cfg: dict[str, Any],
    *,
    run_name: str,
    trial_hparams: dict[str, Any],
    seed: int,
    output_dir: str | None,
    epochs: int | None,
    max_sequences: int | None,
    max_windows_per_sequence: int | None,
    eval_num_workers: int | None,
    train_num_workers: int | None,
) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg["run_name"] = run_name
    cfg["seed"] = seed
    if output_dir is not None:
        cfg["output_dir"] = output_dir

    for dotted_key, value in trial_hparams.items():
        set_nested(cfg, dotted_key, value)

    if epochs is not None:
        set_nested(cfg, "training.epochs", int(epochs))
    if max_sequences is not None:
        set_nested(cfg, "dataset.max_sequences", int(max_sequences))
    if max_windows_per_sequence is not None:
        set_nested(cfg, "dataset.max_windows_per_sequence", int(max_windows_per_sequence))
    if eval_num_workers is not None:
        set_nested(cfg, "eval.num_workers", int(eval_num_workers))
    if train_num_workers is not None:
        set_nested(cfg, "training.num_workers", int(train_num_workers))

    return cfg


def flatten_trial(trial_hparams: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in trial_hparams.items():
        if isinstance(value, (list, dict)):
            row[key] = json.dumps(value, separators=(",", ":"))
        else:
            row[key] = value
    root_axis = trial_hparams.get("loss_weights.root_axis_weights")
    if isinstance(root_axis, list) and len(root_axis) == 3:
        row["loss_weights.root_axis_x"] = root_axis[0]
        row["loss_weights.root_axis_y"] = root_axis[1]
        row["loss_weights.root_axis_z"] = root_axis[2]
    return row


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_fields: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_fields = next(reader, [])

    fields = list(existing_fields)
    for key in row:
        if key not in fields:
            fields.append(key)

    rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    rows.append({k: "" if v is None else v for k, v in row.items()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_train_summary(output_dir: Path, run_name: str) -> dict[str, Any]:
    summary_path = output_dir / "eval_reports" / run_name / "train_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_training(config_path: Path, logs_dir: Path) -> tuple[int, float]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "field_converter.training.train_root_tcn",
        "--config",
        str(config_path),
    ]
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"

    started = time.time()
    with (logs_dir / f"{config_path.stem}.out").open("w", encoding="utf-8") as stdout_f, (
        logs_dir / f"{config_path.stem}.err"
    ).open("w", encoding="utf-8") as stderr_f:
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            check=False,
        )
    return proc.returncode, time.time() - started


def positive_int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    out = int(value)
    if out <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search for root TCN hyperparameters")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--search-name", type=str, default=DEFAULT_SEARCH_NAME)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=positive_int_or_none, default=None)
    parser.add_argument("--max-sequences", type=positive_int_or_none, default=None)
    parser.add_argument("--max-windows-per-sequence", type=positive_int_or_none, default=None)
    parser.add_argument("--train-num-workers", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write sampled configs without launching training")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a trial when outputs already contain best.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be > 0")

    base_config = args.base_config if args.base_config.is_absolute() else PROJECT_ROOT / args.base_config
    base_cfg = load_yaml(base_config)
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir_abs = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    search_dir = output_dir_abs / "random_search" / args.search_name
    configs_dir = search_dir / "configs"
    logs_dir = search_dir / "logs"
    results_csv = search_dir / "random_search_results.csv"
    results_jsonl = search_dir / "random_search_results.jsonl"
    search_dir.mkdir(parents=True, exist_ok=True)

    (search_dir / "search_space.json").write_text(
        json.dumps(
            {
                "base_config": str(base_config),
                "seed": args.seed,
                "n_trials": args.n_trials,
                "start_index": args.start_index,
                "notes": "Generated by scripts/tcn/random_search.py",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for _ in range(args.start_index):
        sample_trial(rng)

    for offset in range(args.n_trials):
        trial_idx = args.start_index + offset
        run_name = f"{args.search_name}_trial_{trial_idx:03d}"
        trial_seed = args.seed + trial_idx
        trial_hparams = sample_trial(rng)
        trial_cfg = apply_trial_to_config(
            base_cfg,
            run_name=run_name,
            trial_hparams=trial_hparams,
            seed=trial_seed,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_sequences=args.max_sequences,
            max_windows_per_sequence=args.max_windows_per_sequence,
            eval_num_workers=args.eval_num_workers,
            train_num_workers=args.train_num_workers,
        )
        config_path = configs_dir / f"{run_name}.yaml"
        write_yaml(config_path, trial_cfg)

        best_ckpt = output_dir_abs / "checkpoints" / run_name / "best.pt"
        status = "dry_run" if args.dry_run else "success"
        elapsed_s = 0.0
        returncode = 0

        if args.dry_run:
            print(f"[dry-run] wrote {config_path}")
        elif args.skip_existing and best_ckpt.exists():
            status = "skipped_existing"
            print(f"[skip] {run_name}: best checkpoint already exists")
        else:
            print(f"[trial {trial_idx:03d}] training {run_name}")
            returncode, elapsed_s = run_training(config_path, logs_dir)
            if returncode != 0:
                status = "failed"
                print(f"[trial {trial_idx:03d}] failed with return code {returncode}")

        summary = read_train_summary(output_dir_abs, run_name)
        best_metric = summary.get("best_root_error_mean_m")
        row = {
            "trial": trial_idx,
            "run_name": run_name,
            "status": status,
            "returncode": returncode,
            "elapsed_s": round_float(elapsed_s, 6),
            "config_path": str(config_path),
            "best_root_error_mean_m": best_metric,
            "best_epoch": summary.get("best_epoch"),
            "last_epoch": summary.get("last_epoch"),
            "device": summary.get("device"),
        }
        row.update(flatten_trial(trial_hparams))
        append_csv(results_csv, row)
        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Random search records: {results_csv}")
    print(f"Generated configs: {configs_dir}")


if __name__ == "__main__":
    main()
