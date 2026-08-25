#!/bin/bash -l
#SBATCH --job-name=ablation_baselines
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

# Stage settings. Values exported by submit_campaign.sh take precedence.
MANIFEST="${ABLATION_MANIFEST:-configs/ablation/unseen_match_v1.yaml}"
FORCE=false

COMMAND=(python -m field_converter.ablation.run_baselines "$MANIFEST")
if [[ "$FORCE" == true ]]; then
    COMMAND+=(--force)
fi

mkdir -p slurms
"${COMMAND[@]}"
echo "Done"
