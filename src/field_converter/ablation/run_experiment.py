from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from field_converter.ablation.common import campaign_output_dir, load_manifest, load_plan, metrics_path, write_json_atomic
from field_converter.ablation.generate import generate_campaign


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    splits = payload.get("splits", {})
    if not isinstance(splits, dict):
        return False
    for split in ("valid", "test"):
        metrics = splits.get(split)
        if not isinstance(metrics, dict):
            return False
        try:
            root_error = float(metrics["root_error_mean_m"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(root_error):
            return False
    return True


def _run(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def run_experiment(
    manifest_path: str | Path,
    index: int,
    *,
    force_train: bool = False,
    force_eval: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    _, manifest = load_manifest(manifest_path)
    try:
        plan = load_plan(manifest)
    except FileNotFoundError:
        plan = generate_campaign(manifest_path)
    runs = plan["runs"]
    if index < 0 or index >= len(runs):
        raise IndexError(f"Run index {index} is outside [0, {len(runs) - 1}]")
    run = runs[index]
    run_name = str(run["run_name"])
    output_dir = campaign_output_dir(manifest)
    state_path = output_dir / "states" / f"{run_name}.json"
    metrics = metrics_path(manifest, run_name)
    config_path = Path(str(run["config_path"]))
    checkpoint = output_dir / "checkpoints" / run_name / "best.pt"
    report_dir = output_dir / "eval_reports" / run_name
    used_config = report_dir / "config_used.yaml"
    train_summary = report_dir / "train_summary.json"

    has_artifacts = checkpoint.exists() or train_summary.exists() or metrics.exists()
    if has_artifacts and not force_train:
        if not used_config.exists():
            raise RuntimeError(
                f"Existing artifacts for {run_name} have no config_used.yaml. "
                "Use --force-train or a new campaign_name."
            )
        if used_config.read_bytes() != config_path.read_bytes():
            raise RuntimeError(
                f"Existing artifacts use a different config for {run_name}. "
                "Use --force-train or a new campaign_name."
            )

    if _metrics_complete(metrics) and not force_train and not force_eval:
        skipped_state = {
            "status": "complete",
            "skipped": True,
            "reason": "metrics already complete",
            "run": run,
            "metrics": str(metrics),
            "updated_at": _utc_now(),
        }
        write_json_atomic(state_path, skipped_state)
        print(f"[skip] {run_name}: {metrics}")
        return skipped_state

    env = dict(os.environ)
    project_root = Path(__file__).resolve().parents[3]
    src = str(project_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    python = sys.executable
    train_command = [python, "-m", str(run["train_module"]), "--config", str(config_path)]
    eval_command = [
        python,
        "-m",
        str(run["eval_module"]),
        "--config",
        str(config_path),
        "--checkpoint",
        "best",
    ]
    if run["architecture"] == "transformer" and not bool(run.get("run_mean_baseline", False)):
        eval_command.append("--no_baseline")

    state: dict[str, Any] = {
        "status": "running",
        "run": run,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": _utc_now(),
        "commands": {"train": train_command, "evaluate": eval_command},
    }
    write_json_atomic(state_path, state)
    start = time.monotonic()
    try:
        training_complete = checkpoint.exists() and train_summary.exists()
        if force_train or not training_complete:
            _run(train_command, env=env, dry_run=dry_run)
        else:
            print(f"[resume] training is complete, skipping to evaluation: {checkpoint}")
        if force_train or force_eval or not _metrics_complete(metrics):
            _run(eval_command, env=env, dry_run=dry_run)
        state.update(
            {
                "status": "dry_run" if dry_run else "complete",
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - start,
                "checkpoint": str(checkpoint),
                "metrics": str(metrics),
            }
        )
        write_json_atomic(state_path, state)
        return state
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "finished_at": _utc_now(),
                "duration_seconds": time.monotonic() - start,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json_atomic(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one frozen campaign run")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("index", type=int)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_experiment(
        args.manifest,
        args.index,
        force_train=bool(args.force_train),
        force_eval=bool(args.force_eval),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
