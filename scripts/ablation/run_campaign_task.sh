#!/bin/bash -l
#SBATCH --job-name=ablation_train
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=12:00:00
#SBATCH --array=0-24%1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%A_%a.out
#SBATCH --error=slurms/slurm_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

# Stage settings. Keep the array size aligned with the generated campaign plan.
MANIFEST="${ABLATION_MANIFEST:-configs/ablation/unseen_match_v1.yaml}"
FORCE_TRAIN=false
FORCE_EVAL=false
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This script must be launched as a Slurm array}"

COMMAND=(python -m field_converter.ablation.run_experiment "$MANIFEST" "$TASK_INDEX")
if [[ "$FORCE_TRAIN" == true ]]; then
    COMMAND+=(--force-train)
fi
if [[ "$FORCE_EVAL" == true ]]; then
    COMMAND+=(--force-eval)
fi

mkdir -p slurms
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
"${COMMAND[@]}"
echo "Done"
