from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from field_converter.ablation.common import (
    build_match_split,
    campaign_output_dir,
    canonical_hash,
    fold_data_dirs,
    load_manifest,
    read_sequences,
    resolve_project_path,
    sequence_match,
    write_json_atomic,
)


ARCH_MODULES = {
    "tcn": (
        "field_converter.training.train_root_tcn",
        "field_converter.training.evaluate_root_tcn",
    ),
    "transformer": (
        "field_converter.training.train_root_transformer",
        "field_converter.training.evaluate_root_transformer",
    ),
    "mlp": (
        "field_converter.training.train_root_mlp",
        "field_converter.training.evaluate_root_mlp",
    ),
}


def _load_base_configs(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    base_configs: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    raw_sources = manifest.get("base_configs", {})
    for architecture in ("tcn", "transformer", "mlp"):
        if architecture not in raw_sources:
            raise ValueError(f"base_configs.{architecture} is required")
        path = resolve_project_path(str(raw_sources[architecture]))
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Base config must be a YAML mapping: {path}")
        base_configs[architecture] = payload
        sources[architecture] = str(path)
    return base_configs, sources


def _configure_run(
    *,
    base: dict[str, Any],
    architecture: str,
    run_name: str,
    seed: int,
    prediction_mode: str,
    input_updates: dict[str, bool],
    data_dir: Path,
    root_init_dir: Path,
    output_dir: Path,
    window_size: int | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["run_name"] = run_name
    cfg["seed"] = int(seed)
    cfg["device"] = "auto"
    cfg["prediction_mode"] = prediction_mode
    cfg["data_dir"] = str(data_dir)
    cfg["root_init_dir"] = str(root_init_dir)
    cfg["output_dir"] = str(output_dir)

    input_config = cfg.setdefault("input_config", {})
    input_config.setdefault("use_root_init_as_input", False)
    input_config.update(input_updates)

    if window_size is not None:
        if architecture == "mlp":
            raise ValueError("MLP runs do not accept a temporal window size")
        cfg.setdefault("dataset", {})["window_size"] = int(window_size)
        cfg["dataset"]["stride"] = int(stride if stride is not None else max(1, window_size // 5))
        if architecture == "transformer":
            cfg.setdefault("model", {})["max_window_size"] = int(window_size)

    eval_config = cfg.setdefault("eval", {})
    eval_config["splits"] = ["valid", "test"]
    eval_config["save_predictions_npz"] = True
    eval_config["save_predictions_csv"] = False
    cfg.setdefault("plots", {})["enabled"] = False
    if architecture == "tcn":
        cfg.setdefault("diagnostic_plots", {})["enabled"] = False
    return cfg


def generate_campaign(manifest_path: str | Path) -> dict[str, Any]:
    resolved_manifest_path, manifest = load_manifest(manifest_path)
    output_dir = campaign_output_dir(manifest)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "states").mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm").mkdir(parents=True, exist_ok=True)

    base_configs, base_sources = _load_base_configs(manifest)
    sequences = read_sequences(manifest)
    preprocessing = manifest.get("preprocessing", {}) or {}
    raw_features_dir = resolve_project_path(str(manifest["data_dir"])) / str(
        preprocessing.get("raw_features_dirname", "features")
    )
    missing_features = [sequence for sequence in sequences if not (raw_features_dir / f"{sequence}.npz").exists()]
    if missing_features:
        raise FileNotFoundError(
            f"Missing {len(missing_features)} raw feature files; first missing: {missing_features[:5]}"
        )

    fold_audit: dict[str, Any] = {}
    for fold_name, fold_spec in manifest["folds"].items():
        if not isinstance(fold_spec, dict):
            raise ValueError(f"Fold {fold_name!r} must be a mapping")
        split = build_match_split(
            sequences,
            valid_match=str(fold_spec["valid_match"]),
            test_match=str(fold_spec["test_match"]),
            train_matches=list(fold_spec["train_matches"]) if fold_spec.get("train_matches") is not None else None,
        )
        fold_audit[str(fold_name)] = {
            "valid_match": str(fold_spec["valid_match"]),
            "test_match": str(fold_spec["test_match"]),
            "train_matches": sorted({sequence_match(sequence) for sequence in split["train"]}),
            "counts": {name: len(values) for name, values in split.items()},
            "sequences": split,
        }
    design = manifest.get("design", {}) or {}
    primary_fold = str(design.get("primary_fold", "primary"))
    if primary_fold not in manifest["folds"]:
        raise ValueError(f"design.primary_fold does not exist: {primary_fold}")
    confirmation_folds = [str(value) for value in design.get("confirmation_folds", [])]
    for fold in confirmation_folds:
        if fold not in manifest["folds"]:
            raise ValueError(f"Unknown confirmation fold: {fold}")

    base_seed = int(design.get("base_seed", 1235))
    repeat_seeds = [int(value) for value in design.get("repeat_seeds", [2027])]
    temporal_windows = design.get("temporal_windows", {}) or {}
    input_ablations = design.get("input_ablations", {}) or {}
    expected_hours_per_run = float(design.get("expected_hours_per_run", 4.5))

    runs: list[dict[str, Any]] = []

    def add_run(
        architecture: str,
        fold: str,
        seed: int,
        family: str,
        variant: str,
        label: str,
        *,
        prediction_mode: str = "delta",
        input_updates: dict[str, bool] | None = None,
        window_size: int | None = None,
        stride: int | None = None,
        run_mean_baseline: bool = False,
    ) -> None:
        if architecture not in ARCH_MODULES:
            raise ValueError(f"Unsupported architecture: {architecture}")
        run_name = f"{fold}_{architecture}_{variant}_s{seed}"
        features_dir, root_init_dir = fold_data_dirs(manifest, fold)
        cfg = _configure_run(
            base=base_configs[architecture],
            architecture=architecture,
            run_name=run_name,
            seed=seed,
            prediction_mode=prediction_mode,
            input_updates=input_updates or {},
            data_dir=features_dir,
            root_init_dir=root_init_dir,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride,
        )
        config_path = configs_dir / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False, width=110), encoding="utf-8")
        train_module, eval_module = ARCH_MODULES[architecture]
        runs.append(
            {
                "index": len(runs),
                "run_id": run_name,
                "run_name": run_name,
                "architecture": architecture,
                "fold": fold,
                "seed": seed,
                "family": family,
                "variant": variant,
                "label": label,
                "prediction_mode": prediction_mode,
                "input_updates": input_updates or {},
                "window_size": cfg.get("dataset", {}).get("window_size"),
                "config_path": str(config_path),
                "train_module": train_module,
                "eval_module": eval_module,
                "run_mean_baseline": bool(run_mean_baseline),
            }
        )

    for architecture in ("tcn", "transformer"):
        add_run(
            architecture,
            primary_fold,
            base_seed,
            "reference",
            "full",
            f"{architecture.upper()} full delta",
            run_mean_baseline=architecture == "transformer",
        )

    for ablation_name, ablation_raw in input_ablations.items():
        if not isinstance(ablation_raw, dict):
            raise ValueError(f"Input ablation {ablation_name!r} must be a mapping")
        label = str(ablation_raw.get("label", ablation_name))
        per_architecture = ablation_raw.get("updates_by_architecture")
        if per_architecture is not None:
            if not isinstance(per_architecture, dict) or not per_architecture:
                raise ValueError(f"Input ablation {ablation_name!r} has invalid updates_by_architecture")
            architecture_updates = {
                str(architecture): {str(key): bool(value) for key, value in updates.items()}
                for architecture, updates in per_architecture.items()
            }
        else:
            fields = ablation_raw.get("disable", [])
            if not isinstance(fields, list) or not fields:
                raise ValueError(
                    f"Input ablation {ablation_name!r} must define disable or updates_by_architecture"
                )
            updates = {str(field): False for field in fields}
            architecture_updates = {"tcn": updates, "transformer": updates}
        for architecture, updates in architecture_updates.items():
            if architecture not in {"tcn", "transformer"}:
                raise ValueError(f"Input ablation {ablation_name!r} targets unsupported model {architecture!r}")
            add_run(
                architecture,
                primary_fold,
                base_seed,
                "input_ablation",
                str(ablation_name),
                f"{architecture.upper()} {label}",
                input_updates=updates,
            )

    if bool(design.get("include_absolute", True)):
        for architecture in ("tcn", "transformer"):
            add_run(
                architecture,
                primary_fold,
                base_seed,
                "formulation",
                "absolute",
                f"{architecture.upper()} absolute",
                prediction_mode="absolute",
                input_updates={"use_root_init_as_input": False},
            )

    if bool(design.get("include_absolute_with_root_init", True)):
        for architecture in ("tcn", "transformer"):
            add_run(
                architecture,
                primary_fold,
                base_seed,
                "formulation",
                "absolute_root_init_input",
                f"{architecture.upper()} absolute + root init input",
                prediction_mode="absolute",
                input_updates={"use_root_init_as_input": True},
            )

    if bool(design.get("include_temporal_controls", True)):
        for architecture in ("tcn", "transformer"):
            window = int(temporal_windows[architecture])
            stride = int(temporal_windows.get(f"{architecture}_stride", max(1, window // 5)))
            add_run(
                architecture,
                primary_fold,
                base_seed,
                "temporal_context",
                f"window_{window}",
                f"{architecture.upper()} context {window}",
                window_size=window,
                stride=stride,
            )

    if bool(design.get("include_mlp", True)):
        add_run("mlp", primary_fold, base_seed, "framewise", "full", "MLP framewise full delta")

    for seed in repeat_seeds:
        if seed == base_seed:
            continue
        for architecture in ("tcn", "transformer"):
            add_run(
                architecture,
                primary_fold,
                seed,
                "seed_repeat",
                "full",
                f"{architecture.upper()} full delta seed {seed}",
            )

    for fold in confirmation_folds:
        for architecture in ("tcn", "transformer"):
            add_run(
                architecture,
                fold,
                base_seed,
                "match_confirmation",
                "full",
                f"{architecture.upper()} full delta on {fold}",
                run_mean_baseline=architecture == "transformer",
            )

    # Append optional follow-up runs so enabling one does not renumber the
    # original 25-task campaign or invalidate historical Slurm array indices.
    if bool(design.get("include_mlp_absolute", False)):
        if not bool(design.get("include_mlp", True)):
            raise ValueError("design.include_mlp_absolute requires design.include_mlp")
        add_run(
            "mlp",
            primary_fold,
            base_seed,
            "formulation",
            "absolute",
            "MLP absolute",
            prediction_mode="absolute",
            input_updates={"use_root_init_as_input": False},
        )

    expected_gpu_hours = len(runs) * expected_hours_per_run
    max_runs = int(design.get("max_runs", 26))
    max_gpu_hours = float(design.get("max_gpu_hours", 120.0))
    if len(runs) > max_runs:
        raise ValueError(f"Campaign has {len(runs)} runs, exceeding design.max_runs={max_runs}")
    if expected_gpu_hours > max_gpu_hours:
        raise ValueError(
            f"Estimated campaign cost is {expected_gpu_hours:.1f} GPU hours, "
            f"exceeding design.max_gpu_hours={max_gpu_hours:.1f}"
        )

    plan = {
        "campaign_name": manifest["campaign_name"],
        "manifest": str(resolved_manifest_path),
        "manifest_hash": canonical_hash(manifest),
        "base_config_sources": base_sources,
        "base_config_hashes": {name: canonical_hash(config) for name, config in base_configs.items()},
        "source_runs": manifest.get("source_runs", {}),
        "plan_hash": canonical_hash({"manifest": manifest, "base_configs": base_configs}),
        "primary_fold": primary_fold,
        "base_seed": base_seed,
        "num_runs": len(runs),
        "expected_hours_per_run": expected_hours_per_run,
        "expected_gpu_hours": expected_gpu_hours,
        "max_runs": max_runs,
        "max_gpu_hours": max_gpu_hours,
        "source_data": {
            "raw_features_dir": str(raw_features_dir),
            "num_sequences": len(sequences),
            "num_matches": len({sequence_match(sequence) for sequence in sequences}),
        },
        "folds": fold_audit,
        "runs": runs,
    }
    write_json_atomic(output_dir / "plan.json", plan)
    (output_dir / "manifest_used.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, width=110), encoding="utf-8"
    )
    print(
        f"Generated {len(runs)} runs in {output_dir} "
        f"(~{plan['expected_gpu_hours']:.1f} sequential GPU hours)."
    )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen YAML configs for an ablation campaign")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    generate_campaign(args.manifest)


if __name__ == "__main__":
    main()
