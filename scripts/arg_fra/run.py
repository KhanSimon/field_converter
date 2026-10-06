"""Retrain the two paper references with ARG_FRA held out as a complete match."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from field_converter.ablation.common import (
    PROJECT_ROOT,
    build_match_split,
    campaign_data_dir,
    campaign_output_dir,
    canonical_hash,
    fold_data_dirs,
    load_manifest,
    load_plan,
    read_sequences,
    resolve_project_path,
    sequence_match,
    write_json_atomic,
)
from field_converter.ablation.generate import ARCH_MODULES
from field_converter.ablation.run_experiment import run_experiment
from field_converter.ablation.submit import _read_array_concurrency


SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPTS_DIR / "experiment.yaml"


def build_plan(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[Path, str]]:
    resolved_manifest, manifest = load_manifest(manifest_path)
    if list(manifest["folds"]) != ["arg_fra"]:
        raise ValueError("This workflow requires exactly one fold named arg_fra")
    fold = manifest["folds"]["arg_fra"]
    if fold["test_match"] != "ARG_FRA":
        raise ValueError("ARG_FRA must remain the held-out test match")
    sequences = read_sequences(manifest)
    split = build_match_split(sequences, **fold)
    raw_features = resolve_project_path(manifest["data_dir"]) / manifest["preprocessing"]["raw_features_dirname"]
    missing = [seq for seq in sequences if not (raw_features / f"{seq}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} raw feature files in {raw_features}: {missing[:5]}")

    output_dir = campaign_output_dir(manifest)
    features_dir, root_init_dir = fold_data_dirs(manifest, "arg_fra")
    sources = {}
    configs = {}
    runs = []
    for architecture in ("tcn", "transformer"):
        source = resolve_project_path(manifest["base_configs"][architecture])
        base = yaml.safe_load(source.read_text(encoding="utf-8"))
        sources[architecture] = base
        cfg = copy.deepcopy(base)
        run_name = f"arg_fra_{architecture}_s{cfg['seed']}"
        # Only redirect artifacts and data; keep all paper hyperparameters.
        cfg.update(
            run_name=run_name,
            data_dir=str(features_dir),
            root_init_dir=str(root_init_dir),
            output_dir=str(output_dir),
        )
        config_path = output_dir / "configs" / f"{run_name}.yaml"
        configs[config_path] = yaml.safe_dump(cfg, sort_keys=False, width=110)
        train_module, eval_module = ARCH_MODULES[architecture]
        runs.append({
            "index": len(runs),
            "run_name": run_name,
            "architecture": architecture,
            "fold": "arg_fra",
            "seed": cfg["seed"],
            "config_path": str(config_path),
            "train_module": train_module,
            "eval_module": eval_module,
            "run_mean_baseline": False,
        })

    plan = {
        "campaign_name": manifest["campaign_name"],
        "manifest": str(resolved_manifest),
        "manifest_hash": canonical_hash(manifest),
        "plan_hash": canonical_hash({"manifest": manifest, "base_configs": sources, "split": split}),
        "source_runs": manifest.get("source_runs", {}),
        "num_runs": len(runs),
        "folds": {"arg_fra": {
            "valid_match": fold["valid_match"],
            "test_match": fold["test_match"],
            "train_matches": sorted({sequence_match(seq) for seq in split["train"]}),
            "counts": {name: len(values) for name, values in split.items()},
            "sequences": split,
        }},
        "runs": runs,
    }
    return manifest, plan, configs


def generate_plan(manifest_path: Path) -> dict[str, Any]:
    manifest, plan, configs = build_plan(manifest_path)
    output_dir = campaign_output_dir(manifest)
    if (output_dir / "plan.json").exists():
        if load_plan(manifest)["plan_hash"] != plan["plan_hash"]:
            raise RuntimeError("The split or source configs changed. Use a new campaign_name in the manifest.")
    for path, content in configs.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Frozen config changed: {path}. Use a new campaign_name.")
    for path, content in configs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    write_json_atomic(output_dir / "plan.json", plan)
    (output_dir / "manifest_used.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(f"Split: {plan['folds']['arg_fra']['counts']}")
    for run in plan["runs"]:
        print(f"Task {run['index']}: {run['run_name']} -> {run['config_path']}")
    return plan


def check_prepared(manifest_path: Path) -> None:
    from field_converter.ablation.prepare import _fold_is_complete

    manifest, expected, configs = build_plan(manifest_path)
    if load_plan(manifest)["plan_hash"] != expected["plan_hash"]:
        raise RuntimeError("The frozen plan differs from the current settings. Use a new campaign_name.")
    for path, content in configs.items():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Frozen config changed: {path}")
    marker_path = campaign_data_dir(manifest) / "preprocessing_complete.json"
    if not marker_path.exists():
        raise RuntimeError("Preprocessing is missing. Run prepare.sh before training.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete" or marker.get("manifest_hash") != expected["manifest_hash"]:
        raise RuntimeError("Preprocessing does not match the manifest. Run prepare.sh first.")
    features_dir, root_init_dir = fold_data_dirs(manifest, "arg_fra")
    split = expected["folds"]["arg_fra"]["sequences"]
    if not _fold_is_complete(features_dir, root_init_dir, split):
        raise RuntimeError("Normalized data are incomplete or use a different split. Run prepare.sh first.")
    for metadata in (features_dir / "normalization_stats.json", root_init_dir / "root_init_normalization_meta.json"):
        stats = json.loads(metadata.read_text(encoding="utf-8"))
        if stats.get("train_sequences") != split["train"]:
            raise RuntimeError(f"Normalization must use only this fold's training sequences: {metadata}")


def submit(manifest_path: Path, *, dry_run: bool) -> dict[str, Any]:
    plan = generate_plan(manifest_path)
    _, manifest = load_manifest(manifest_path)
    max_parallel = _read_array_concurrency(SCRIPTS_DIR / "train.sh", num_runs=plan["num_runs"])
    export = f"--export=ARG_FRA_MANIFEST={plan['manifest']},FIELD_CONVERTER_ROOT={PROJECT_ROOT}"
    jobs = {}
    for stage, script in (("preprocessing", "prepare.sh"), ("training_array", "train.sh")):
        command = ["sbatch", "--parsable", export]
        if stage == "training_array":
            command.append(f"--dependency=afterok:{jobs['preprocessing']}")
        command.append(str(SCRIPTS_DIR / script))
        print("+ " + shlex.join(command), flush=True)
        if dry_run:
            jobs[stage] = "PREP_JOB" if stage == "preprocessing" else "TRAIN_ARRAY"
        else:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"sbatch returned an invalid job ID: {result.stdout!r}")
            jobs[stage] = job_id
            print(f"Submitted {stage}: {job_id}")
    submission = {
        "manifest": plan["manifest"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "num_runs": plan["num_runs"],
        "max_parallel": max_parallel,
        "jobs": jobs,
    }
    name = "submission_dry_run.json" if dry_run else "submission.json"
    write_json_atomic(campaign_output_dir(manifest) / name, submission)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(os.environ.get("ARG_FRA_MANIFEST", DEFAULT_MANIFEST)))
    commands = parser.add_subparsers(dest="stage", required=True)
    commands.add_parser("plan", help="Generate the two configs and audit the split; no preprocessing or training")
    prepare = commands.add_parser("prepare", help="Recompute normalization from this fold's train matches")
    prepare.add_argument("--overwrite", action="store_true")
    train = commands.add_parser("train", help="Train and evaluate one prepared array task")
    train.add_argument("index", type=int, choices=(0, 1))
    train.add_argument("--force-train", action="store_true")
    train.add_argument("--force-eval", action="store_true")
    submission = commands.add_parser("submit", help="Submit preprocessing followed by the two-task GPU array")
    submission.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    if args.stage == "plan":
        generate_plan(args.manifest)
    elif args.stage == "prepare":
        from field_converter.ablation.prepare import prepare_campaign

        generate_plan(args.manifest)
        prepare_campaign(args.manifest, overwrite=args.overwrite)
        check_prepared(args.manifest)
    elif args.stage == "train":
        check_prepared(args.manifest)
        run_experiment(args.manifest, args.index, force_train=args.force_train, force_eval=args.force_eval)
    else:
        (PROJECT_ROOT / "slurms").mkdir(exist_ok=True)
        submit(args.manifest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
