#!/bin/bash -l
#SBATCH --job-name=arg_fra_prepare
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err
#SBATCH --export=NONE

set -euo pipefail
export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

MANIFEST="${ARG_FRA_MANIFEST:-scripts/arg_fra/experiment.yaml}"
OVERWRITE=false

COMMAND=(python scripts/arg_fra/run.py --manifest "$MANIFEST" prepare)
if [[ "$OVERWRITE" == true ]]; then
    COMMAND+=(--overwrite)
fi
"${COMMAND[@]}"
