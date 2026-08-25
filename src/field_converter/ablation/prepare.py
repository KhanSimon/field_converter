from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from field_converter.ablation.common import (
    build_match_split,
    campaign_data_dir,
    canonical_hash,
    fold_data_dirs,
    load_manifest,
    read_sequences,
    resolve_project_path,
    sequence_match,
    write_json_atomic,
)
from field_converter.data_preparation.generate_root_init import ROOT_INIT_GENERATION_VERSION, generate_all
from field_converter.data_preparation.normalize import Normalizer
from field_converter.data_preparation.normalize_root_init import normalize_all


def _fold_is_complete(features_dir: Path, root_init_dir: Path, split: dict[str, list[str]]) -> bool:
    required_meta = (
        features_dir / "normalization_stats.npz",
        features_dir / "normalization_stats.json",
        features_dir / "split.json",
        root_init_dir / "split.json",
        root_init_dir / "root_init_normalization_meta.json",
    )
    if not all(path.exists() for path in required_meta):
        return False
    try:
        feature_split = json.loads((features_dir / "split.json").read_text(encoding="utf-8"))
        root_split = json.loads((root_init_dir / "split.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for split_name, sequences in split.items():
        if feature_split.get(split_name) != sequences or root_split.get(split_name) != sequences:
            return False
    for split_name, sequences in split.items():
        if any(not (features_dir / split_name / f"{sequence}.npz").exists() for sequence in sequences):
            return False
        if any(not (root_init_dir / split_name / f"{sequence}.npy").exists() for sequence in sequences):
            return False
    return True


def prepare_campaign(
    manifest_path: str | Path,
    *,
    overwrite: bool = False,
    regenerate_root_init: bool = False,
) -> dict[str, Any]:
    resolved_manifest_path, manifest = load_manifest(manifest_path)
    sequences = read_sequences(manifest)
    data_dir = resolve_project_path(str(manifest["data_dir"]))
    preprocessing = manifest.get("preprocessing", {}) or {}
    raw_features_dirname = str(preprocessing.get("raw_features_dirname", "features"))
    raw_root_init_dirname = str(preprocessing.get("raw_root_init_dirname", "root_init_cam"))
    raw_features_dir = data_dir / raw_features_dirname
    raw_root_init_dir = data_dir / raw_root_init_dirname

    missing_features = [sequence for sequence in sequences if not (raw_features_dir / f"{sequence}.npz").exists()]
    if missing_features:
        raise FileNotFoundError(
            f"Missing {len(missing_features)} raw feature files under {raw_features_dir}; "
            f"first missing: {missing_features[:5]}"
        )

    split_seed = int(preprocessing.get("seed", 12345))
    pelvis_mode = str(preprocessing.get("pelvis_mode", "hips_mean"))
    if pelvis_mode not in {"hips_mean", "joint8"}:
        raise ValueError(f"Unsupported preprocessing.pelvis_mode: {pelvis_mode}")
    min_bbox_size_px = float(preprocessing.get("min_bbox_size_px", 10.0))

    root_meta_path = raw_root_init_dir / "root_init_generation_meta.json"
    recorded_pelvis_mode = None
    recorded_generation_version = None
    if root_meta_path.exists():
        try:
            root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
            recorded_pelvis_mode = root_meta.get("pelvis_mode")
            recorded_generation_version = root_meta.get("generation_version")
        except (OSError, json.JSONDecodeError):
            recorded_pelvis_mode = None
            recorded_generation_version = None
    rebuild_all_root_init = (
        regenerate_root_init
        or recorded_pelvis_mode != pelvis_mode
        or recorded_generation_version != ROOT_INIT_GENERATION_VERSION
    )
    sequences_to_generate = sequences if rebuild_all_root_init else [
        sequence for sequence in sequences if not (raw_root_init_dir / f"{sequence}.npy").exists()
    ]
    if sequences_to_generate:
        reason = "forced/provenance mismatch" if rebuild_all_root_init else "missing files"
        print(f"Generating {len(sequences_to_generate)} raw root-init files ({reason})...")
        generate_all(
            data_dir=data_dir,
            features_dirname=raw_features_dirname,
            out_dirname=raw_root_init_dirname,
            sequences=sequences_to_generate,
            overwrite=rebuild_all_root_init,
            pelvis_mode=pelvis_mode,
        )

    fold_reports: dict[str, Any] = {}
    for fold_name, fold_spec_raw in manifest["folds"].items():
        if not isinstance(fold_spec_raw, dict):
            raise ValueError(f"Fold {fold_name!r} must be a mapping")
        valid_match = str(fold_spec_raw["valid_match"])
        test_match = str(fold_spec_raw["test_match"])
        train_matches_raw = fold_spec_raw.get("train_matches")
        train_matches = list(train_matches_raw) if train_matches_raw is not None else None
        split = build_match_split(
            sequences,
            valid_match=valid_match,
            test_match=test_match,
            train_matches=train_matches,
        )
        split_iterables: dict[str, Iterable[str]] = {
            split_name: split_sequences for split_name, split_sequences in split.items()
        }
        features_dir, root_init_dir = fold_data_dirs(manifest, str(fold_name))

        if not overwrite and not rebuild_all_root_init and _fold_is_complete(features_dir, root_init_dir, split):
            print(f"[skip] fold={fold_name}: normalized data are complete")
        else:
            print(
                f"[prepare] fold={fold_name} train={len(split['train'])} "
                f"valid={len(split['valid'])} test={len(split['test'])}"
            )
            normalizer = Normalizer(
                data_dir=data_dir,
                in_features_dirname=raw_features_dirname,
                out_features_dirname=str(features_dir),
                pelvis_mode=pelvis_mode,  # type: ignore[arg-type]
                min_bbox_size_px=min_bbox_size_px,
            )
            stats = normalizer.run_with_split(split_iterables, seed=split_seed, overwrite=overwrite)
            normalize_all(
                data_dir=data_dir,
                in_dirname=raw_root_init_dirname,
                out_dirname=str(root_init_dir),
                split_to_sequences=split_iterables,
                stats=stats,
                overwrite=overwrite or rebuild_all_root_init,
            )

        fold_reports[str(fold_name)] = {
            "valid_match": valid_match,
            "test_match": test_match,
            "train_matches": sorted({sequence_match(sequence) for sequence in split["train"]}),
            "counts": {name: len(values) for name, values in split.items()},
            "sequences": split,
            "features_dir": str(features_dir),
            "root_init_dir": str(root_init_dir),
        }

    marker_payload = {
        "status": "complete",
        "manifest": str(resolved_manifest_path),
        "manifest_hash": canonical_hash(manifest),
        "num_sequences": len(sequences),
        "num_matches": len({sequence_match(sequence) for sequence in sequences}),
        "folds": fold_reports,
    }
    marker = campaign_data_dir(manifest) / "preprocessing_complete.json"
    write_json_atomic(marker, marker_payload)
    print(f"Preprocessing complete: {marker}")
    return marker_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare grouped unseen-match splits for an ablation campaign")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--regenerate-root-init", action="store_true")
    args = parser.parse_args()
    prepare_campaign(
        args.manifest,
        overwrite=bool(args.overwrite),
        regenerate_root_init=bool(args.regenerate_root_init),
    )


if __name__ == "__main__":
    main()
