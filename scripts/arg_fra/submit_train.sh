#!/bin/bash -l
#SBATCH --job-name=arg_fra_submit
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err
#SBATCH --export=NONE

set -euo pipefail
export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

# Edit settings here; the normal sbatch command takes no arguments.
MANIFEST="scripts/arg_fra/experiment.yaml"
DRY_RUN=false

COMMAND=(python scripts/arg_fra/run.py --manifest "$MANIFEST" submit)
if [[ "$DRY_RUN" == true ]]; then
    COMMAND+=(--dry-run)
fi
"${COMMAND[@]}"
