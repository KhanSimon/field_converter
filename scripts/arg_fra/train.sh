#!/bin/bash -l
#SBATCH --job-name=arg_fra_train
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=14:00:00
#SBATCH --array=0-1%1
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%A_%a.out
#SBATCH --error=slurms/slurm_%A_%a.err
#SBATCH --export=NONE

set -euo pipefail
export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

MANIFEST="${ARG_FRA_MANIFEST:-scripts/arg_fra/experiment.yaml}"
FORCE_TRAIN=false
FORCE_EVAL=false
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

COMMAND=(python scripts/arg_fra/run.py --manifest "$MANIFEST" train "$TASK_INDEX")
if [[ "$FORCE_TRAIN" == true ]]; then
    COMMAND+=(--force-train)
fi
if [[ "$FORCE_EVAL" == true ]]; then
    COMMAND+=(--force-eval)
fi
"${COMMAND[@]}"
