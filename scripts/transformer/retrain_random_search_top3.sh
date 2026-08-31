#!/bin/bash -l
#SBATCH --job-name=transformer_top3_comp
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=32:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err
#SBATCH --export=NONE

set -euo pipefail

export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

# Edit this block only. The sbatch command takes no additional argument.
DRY_RUN=false

SEARCH_NAME="root_transformer_random_search_v2_competitive"
OUTPUT_DIR="outputs/ablation/unseen_match_v1"
TOP_K=3

TRAINING_SEED=1235
FULL_EPOCHS=60
EARLY_STOPPING_PATIENCE=10
TRAIN_NUM_WORKERS=2
EVAL_NUM_WORKERS=2

# Slurm grants 32 h. The internal deadline leaves one hour of scheduler margin.
MAX_HOURS=31.0
PER_RUN_TIMEOUT_HOURS=9.0
EVALUATION_RESERVE_HOURS=1.0

mkdir -p slurms
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

COMMAND=(
    python scripts/transformer/retrain_random_search_top3.py
    --search-name "$SEARCH_NAME"
    --output-dir "$OUTPUT_DIR"
    --top-k "$TOP_K"
    --seed "$TRAINING_SEED"
    --epochs "$FULL_EPOCHS"
    --early-stopping-patience "$EARLY_STOPPING_PATIENCE"
    --train-num-workers "$TRAIN_NUM_WORKERS"
    --eval-num-workers "$EVAL_NUM_WORKERS"
    --max-hours "$MAX_HOURS"
    --per-run-timeout-hours "$PER_RUN_TIMEOUT_HOURS"
    --evaluation-reserve-hours "$EVALUATION_RESERVE_HOURS"
)

if [[ "$DRY_RUN" == true ]]; then
    COMMAND+=(--dry-run)
fi

"${COMMAND[@]}"

echo "Done: full-budget top-$TOP_K for $SEARCH_NAME"
