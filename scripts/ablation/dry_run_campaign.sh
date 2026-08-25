#!/bin/bash -l
#SBATCH --job-name=dry_run_ablation
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

# Campaign settings. Edit them here; this script takes no terminal arguments.
export ABLATION_MANIFEST="configs/ablation/unseen_match_v1.yaml"
export ABLATION_DRY_RUN=1

mkdir -p slurms
python -m field_converter.ablation.submit
echo "Dry run done: no campaign job was submitted"
