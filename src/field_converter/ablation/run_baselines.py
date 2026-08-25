from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from field_converter.ablation.common import campaign_output_dir, load_manifest, load_plan, resolve_project_path, write_json_atomic
from field_converter.ablation.generate import generate_campaign


def _report_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for split in ("valid", "test"):
            value = float(payload["splits"][split]["root_error_mean_m"])
            if not math.isfinite(value):
                return False
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return True


def run_baselines(manifest_path: str | Path, *, force: bool = False, dry_run: bool = False) -> dict[str, Any]:
    _, manifest = load_manifest(manifest_path)
    try:
        plan = load_plan(manifest)
    except FileNotFoundError:
        plan = generate_campaign(manifest_path)
    output_dir = campaign_output_dir(manifest)
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    raw_features_dirname = str((manifest.get("preprocessing", {}) or {}).get("raw_features_dirname", "features"))
    raw_features_dir = resolve_project_path(str(manifest["data_dir"])) / raw_features_dirname
    base_seed = int(plan["base_seed"])

    env = dict(os.environ)
    project_root = Path(__file__).resolve().parents[3]
    src = str(project_root / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    reports: dict[str, Any] = {}
    for fold in manifest["folds"]:
        candidates = [
            run
            for run in plan["runs"]
            if run["fold"] == fold
            and run["architecture"] == "transformer"
            and run["variant"] == "full"
            and int(run["seed"]) == base_seed
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one reference Transformer config for fold={fold}, got {len(candidates)}")
        output_path = baseline_dir / str(fold) / "geometry_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "field_converter.utils.naive_baseline_metrics",
            "--config",
            str(candidates[0]["config_path"]),
            "--splits",
            "valid",
            "test",
            "--raw-features-dir",
            str(raw_features_dir),
            "--no-model-comparison",
            "--output",
            str(output_path),
        ]
        if _report_complete(output_path) and not force:
            print(f"[skip] geometry baseline fold={fold}: {output_path}")
            reports[str(fold)] = {"status": "complete", "skipped": True, "output": str(output_path)}
            continue
        print("+ " + " ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, check=True, env=env)
        reports[str(fold)] = {
            "status": "dry_run" if dry_run else "complete",
            "output": str(output_path),
            "command": command,
        }

    state = {
        "status": "dry_run" if dry_run else "complete",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "folds": reports,
    }
    write_json_atomic(baseline_dir / "state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-learned baselines for every campaign fold")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_baselines(args.manifest, force=bool(args.force), dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
