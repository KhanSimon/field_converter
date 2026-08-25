from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml  # type: ignore[import-untyped]


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = resolve_project_path(path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Campaign manifest not found: {manifest_path}")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign manifest must be a YAML mapping: {manifest_path}")

    required = ("campaign_name", "data_dir", "output_root", "data_output_root", "base_configs", "folds")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Campaign manifest is missing keys: {missing}")
    if not isinstance(payload["folds"], dict) or not payload["folds"]:
        raise ValueError("Campaign manifest must define at least one fold")
    return manifest_path, payload


def campaign_output_dir(manifest: Mapping[str, Any]) -> Path:
    return resolve_project_path(str(manifest["output_root"])) / str(manifest["campaign_name"])


def campaign_data_dir(manifest: Mapping[str, Any]) -> Path:
    return resolve_project_path(str(manifest["data_output_root"])) / str(manifest["campaign_name"])


def fold_data_dirs(manifest: Mapping[str, Any], fold: str) -> tuple[Path, Path]:
    root = campaign_data_dir(manifest) / fold
    return root / "features_normalized", root / "root_init_cam_normalized"


def sequence_match(sequence: str) -> str:
    try:
        match, suffix = sequence.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Cannot infer match from sequence name: {sequence!r}") from exc
    if not match or not suffix:
        raise ValueError(f"Cannot infer match from sequence name: {sequence!r}")
    return match


def read_sequences(manifest: Mapping[str, Any]) -> list[str]:
    data_dir = resolve_project_path(str(manifest["data_dir"]))
    preprocessing = manifest.get("preprocessing", {}) or {}
    sequences_file = data_dir / str(preprocessing.get("sequences_file", "sequences_gt.txt"))
    if not sequences_file.exists():
        raise FileNotFoundError(f"Sequence list not found: {sequences_file}")
    sequences = [line.strip() for line in sequences_file.read_text(encoding="utf-8").splitlines()]
    sequences = [sequence for sequence in sequences if sequence]
    if not sequences:
        raise ValueError(f"Sequence list is empty: {sequences_file}")
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"Sequence list contains duplicates: {sequences_file}")
    return sequences


def build_match_split(
    sequences: list[str],
    *,
    valid_match: str,
    test_match: str,
    train_matches: list[str] | None = None,
) -> dict[str, list[str]]:
    all_matches = sorted({sequence_match(sequence) for sequence in sequences})
    if valid_match == test_match:
        raise ValueError("Validation and test matches must differ")
    unknown = sorted({valid_match, test_match} - set(all_matches))
    if unknown:
        raise ValueError(f"Unknown held-out matches {unknown}; available matches: {all_matches}")

    selected_train_matches = (
        sorted(set(train_matches))
        if train_matches is not None
        else [match for match in all_matches if match not in {valid_match, test_match}]
    )
    overlap = set(selected_train_matches) & {valid_match, test_match}
    if overlap:
        raise ValueError(f"Held-out matches also appear in train_matches: {sorted(overlap)}")
    missing_matches = sorted(set(all_matches) - set(selected_train_matches) - {valid_match, test_match})
    unknown_train = sorted(set(selected_train_matches) - set(all_matches))
    if unknown_train:
        raise ValueError(f"Unknown train_matches: {unknown_train}")
    if missing_matches:
        raise ValueError(f"Matches are not assigned to any split: {missing_matches}")

    train_match_set = set(selected_train_matches)
    split = {
        "train": [sequence for sequence in sequences if sequence_match(sequence) in train_match_set],
        "valid": [sequence for sequence in sequences if sequence_match(sequence) == valid_match],
        "test": [sequence for sequence in sequences if sequence_match(sequence) == test_match],
    }
    if any(not values for values in split.values()):
        raise ValueError(f"Every split must be non-empty, got counts: { {k: len(v) for k, v in split.items()} }")
    return split


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    temporary.replace(path)


def load_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    plan_path = campaign_output_dir(manifest) / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Campaign plan not found: {plan_path}. Run the generator first.")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError(f"Invalid campaign plan: {plan_path}")
    return payload


def metrics_path(manifest: Mapping[str, Any], run_name: str) -> Path:
    return campaign_output_dir(manifest) / "eval_reports" / run_name / "metrics.json"


def predictions_path(manifest: Mapping[str, Any], run_name: str, split: str) -> Path:
    return campaign_output_dir(manifest) / "predictions" / run_name / f"{split}_predictions.npz"
