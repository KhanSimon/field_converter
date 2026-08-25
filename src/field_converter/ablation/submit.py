from __future__ import annotations

import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from field_converter.ablation.common import PROJECT_ROOT, campaign_output_dir, load_manifest, write_json_atomic
from field_converter.ablation.generate import generate_campaign


DEFAULT_MANIFEST = Path("configs/ablation/unseen_match_v1.yaml")
ARRAY_DIRECTIVE_PATTERN = re.compile(r"^#SBATCH\s+--array=0-(\d+)%(\d+)\s*$", re.MULTILINE)


def _submit(command: list[str], *, dry_run: bool, dry_id: str, env: dict[str, str]) -> str:
    print("+ " + shlex.join(command), flush=True)
    if dry_run:
        return dry_id
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"sbatch failed: {detail}") from exc
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id:
        raise RuntimeError(f"sbatch returned no job id: {result.stdout!r} {result.stderr!r}")
    print(f"  submitted job {job_id}")
    return job_id


def _read_array_concurrency(script: Path, *, num_runs: int) -> int:
    match = ARRAY_DIRECTIVE_PATTERN.search(script.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Missing '#SBATCH --array=0-N%P' directive in {script}")

    last_index, max_parallel = (int(value) for value in match.groups())
    if last_index + 1 != num_runs:
        raise ValueError(
            f"The array in {script} contains {last_index + 1} tasks, "
            f"but the generated campaign contains {num_runs} runs"
        )
    if max_parallel <= 0:
        raise ValueError(f"Array concurrency must be positive in {script}")
    return max_parallel


def submit_campaign(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    resolved_manifest_path, manifest = load_manifest(manifest_path)
    plan = generate_campaign(resolved_manifest_path)
    output_dir = campaign_output_dir(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = PROJECT_ROOT / "scripts" / "ablation"
    prepare_script = scripts_dir / "prepare_campaign.sh"
    train_script = scripts_dir / "run_campaign_task.sh"
    baseline_script = scripts_dir / "run_baselines.sh"
    aggregate_script = scripts_dir / "aggregate_campaign.sh"
    max_parallel = _read_array_concurrency(train_script, num_runs=int(plan["num_runs"]))

    child_env = dict(os.environ)
    child_env["ABLATION_MANIFEST"] = str(resolved_manifest_path)
    for variable in ("TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
        child_env.pop(variable, None)
    exported_job_environment = (
        f"--export=ABLATION_MANIFEST={resolved_manifest_path},FIELD_CONVERTER_ROOT={PROJECT_ROOT}"
    )

    prep_id = _submit(
        ["sbatch", "--parsable", exported_job_environment, str(prepare_script)],
        dry_run=dry_run,
        dry_id="PREP_JOB",
        env=child_env,
    )
    array_id = _submit(
        [
            "sbatch",
            "--parsable",
            exported_job_environment,
            f"--dependency=afterok:{prep_id}",
            str(train_script),
        ],
        dry_run=dry_run,
        dry_id="ARRAY_JOB",
        env=child_env,
    )
    baseline_id = _submit(
        [
            "sbatch",
            "--parsable",
            exported_job_environment,
            f"--dependency=afterok:{prep_id}",
            str(baseline_script),
        ],
        dry_run=dry_run,
        dry_id="BASELINE_JOB",
        env=child_env,
    )
    aggregate_id = _submit(
        [
            "sbatch",
            "--parsable",
            exported_job_environment,
            f"--dependency=afterany:{array_id}:{baseline_id}",
            str(aggregate_script),
        ],
        dry_run=dry_run,
        dry_id="AGGREGATE_JOB",
        env=child_env,
    )

    submission = {
        "campaign_name": manifest["campaign_name"],
        "manifest": str(resolved_manifest_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "num_runs": plan["num_runs"],
        "max_parallel": max_parallel,
        "jobs": {
            "preprocessing": prep_id,
            "training_array": array_id,
            "baselines": baseline_id,
            "aggregation": aggregate_id,
        },
    }
    submission_name = "submission_dry_run.json" if dry_run else "submission.json"
    write_json_atomic(output_dir / submission_name, submission)
    print(f"Campaign output: {output_dir}")
    return submission


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    manifest_path = Path(os.environ.get("ABLATION_MANIFEST", DEFAULT_MANIFEST))
    submit_campaign(manifest_path, dry_run=_env_flag("ABLATION_DRY_RUN"))


if __name__ == "__main__":
    main()
