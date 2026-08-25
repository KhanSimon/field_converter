#!/bin/bash -l
#SBATCH --job-name=ablation_aggregate
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err
#SBATCH --export=NONE

set -euo pipefail
export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

# Stage settings. Values exported by submit_campaign.sh take precedence.
MANIFEST="${ABLATION_MANIFEST:-configs/ablation/unseen_match_v1.yaml}"

mkdir -p slurms
python -m field_converter.ablation.aggregate "$MANIFEST"
echo "Done"
