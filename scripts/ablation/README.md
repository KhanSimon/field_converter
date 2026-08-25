# Unseen-match ablation campaign

This campaign retrains every learned reference and evaluates complete matches
that never appear in training. The frozen manifest is:

```text
configs/ablation/unseen_match_v1.yaml
```

## Experimental design

The default campaign contains 25 learned runs (about 112.5 sequential GPU
hours at 4.5 hours per run):

| Family | Runs | Purpose |
|---|---:|---|
| Full delta references | 2 | Retrain TCN and Transformer on the new split |
| Input ablations | 12 | SAM3D, player 2D, camera, pitch, ground intersection, valid-joint mask |
| Root formulation | 4 | Absolute, absolute conditioned on root-init, and delta references |
| Temporal context | 2 | TCN with 201 frames and Transformer with 41 frames |
| Framewise model | 1 | Full-feature delta MLP |
| Seed repeat | 2 | Full TCN and Transformer with seed 2027 |
| Match confirmation | 2 | Full TCN and Transformer on a second unseen test match |

No loss ablation is included. Each architecture keeps the loss and optimized
hyperparameters of its source run.

The grouped splits are:

| Fold | Train | Validation | Test |
|---|---|---|---|
| `primary` | 6 matches / 62 clips | `BRA_KOR` / 12 clips | `ENG_FRA` / 15 clips |
| `secondary` | 6 matches / 64 clips | `BRA_KOR` / 12 clips | `CRO_MOR` / 13 clips |

Normalization statistics are recomputed from the training matches of each fold
only. No clip from a validation or test match contributes to normalization.

## Launch everything

Le lancement ne prend aucun argument. Le manifeste et le mode sont définis en
tête des scripts shell.

Soumettre un dry-run qui génère les configurations et affiche le DAG sans
soumettre les jobs de campagne :

```bash
sbatch scripts/ablation/dry_run_campaign.sh
```

Submit preprocessing, the sequential GPU array, baselines, and aggregation:

```bash
sbatch scripts/ablation/submit_campaign.sh
```

The GPU array is `0-24%1`, so at most one run uses a GPU at a time. To change
the manifest, edit `ABLATION_MANIFEST` in `submit_campaign.sh` and
`dry_run_campaign.sh`. All Slurm resources are configured in the `#SBATCH`
headers of the stage scripts. In particular, edit this line in
`run_campaign_task.sh` to change the number of runs or GPU concurrency:

```text
#SBATCH --array=0-24%1
```

The submitter checks that the array contains exactly as many tasks as the
generated campaign and stops before submission if these values disagree.

The submission IDs are saved in:

```text
outputs/ablation/unseen_match_v1/submission.json
```

A dry-run writes `submission_dry_run.json` separately and never replaces the
IDs of the latest real submission.

Monitor the campaign with:

```bash
squeue -u "$USER"
```

## Resume after a failure

Submit the same command again:

```bash
sbatch scripts/ablation/submit_campaign.sh
```

Completed runs are skipped. If a run has both `best.pt` and
`train_summary.json` but no complete `metrics.json`, its array task skips
training and performs evaluation. An interrupted training without
`train_summary.json` restarts from the beginning because the current trainers
do not restore optimizer state. Per-run states live under
`outputs/ablation/unseen_match_v1/states/`.

The Slurm jobs use `--export=NONE`, so no broken temporary setting is inherited
from the submission shell. Each job then creates a short private folder such as
`/dev/shm/fc_<uid>/j<job>_<task>`, with `/var/tmp` as fallback. This avoids
`/tmp`, the failing `/tmp/skhan` directory, and Python multiprocessing's
108-byte Unix socket path limit. Set `FIELD_CONVERTER_TMP_ROOT` in the shell
scripts only when the replacement path remains shorter than 60 characters.

## Run stages manually

Every stage can also be submitted independently without arguments. Its
manifest and optional force settings are editable near the top of its shell
script.

Generate the frozen plan and configs without submitting downstream jobs:

```bash
sbatch scripts/ablation/dry_run_campaign.sh
```

Run preprocessing:

```bash
sbatch scripts/ablation/prepare_campaign.sh
```

Run the complete training array configured by `#SBATCH --array`:

```bash
sbatch scripts/ablation/run_campaign_task.sh
```

Recompute the geometry baselines:

```bash
sbatch scripts/ablation/run_baselines.sh
```

Aggregate all currently available results, including a partial campaign:

```bash
sbatch scripts/ablation/aggregate_campaign.sh
```

Set `OVERWRITE`, `REGENERATE_ROOT_INIT`, `FORCE_TRAIN`, `FORCE_EVAL`, or
`FORCE` directly in the corresponding stage script when needed.

## Preprocessing provenance

On the first preprocessing run, `data/root_init_cam/` is regenerated if
`root_init_generation_meta.json` is absent or does not declare the same pelvis
mode as the campaign (`hips_mean`). This fixes the former implicit `joint8`
default and makes root-init consistent with the normalized SAM3D features.

Derived fold data are written to:

```text
data/ablation/unseen_match_v1/<fold>/features_normalized/
data/ablation/unseen_match_v1/<fold>/root_init_cam_normalized/
```

Set `OVERWRITE=true` in `prepare_campaign.sh` only when normalized fold data
must be rebuilt. Set `REGENERATE_ROOT_INIT=true` there to explicitly rebuild
every raw root-init file.

## Results

All learned artifacts stay isolated under:

```text
outputs/ablation/unseen_match_v1/
```

The final analysis is written to `summary/`:

- `all_metrics.csv`: every split and metric;
- `ablation_effects.csv`: differences against each architecture's full model;
- `seed_summary.csv`: mean and standard deviation over training seeds;
- `run_status.csv`: completion audit;
- `summary.md`: concise paper-facing report;
- `figures/*.png` and `figures/*.pdf`: seven publication-ready plots.

Root-error confidence intervals use a sequence-cluster bootstrap. Ablation
effects additionally use paired bootstrap differences whenever sequence,
player, and frame identifiers match. Positive effect values mean that the
ablation is worse than the full reference.
