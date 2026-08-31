#!/bin/bash -l
#SBATCH --job-name=transformer_rs_comp
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

# ---------------------------------------------------------------------------
# Campaign parameters. Edit this block only; sbatch takes no command argument.
# ---------------------------------------------------------------------------
DRY_RUN=false

BASE_CONFIG="configs/transformer/random_search_competitive_base.yaml"
SEARCH_NAME="root_transformer_random_search_v2_competitive"
OUTPUT_DIR="outputs/ablation/unseen_match_v1"

N_TRIALS=16
SEARCH_SEED=20260830
TRAINING_SEED=1235

# Screening uses every training match but samples at most this many windows
# from each sequence. The selected candidate is then trained on all windows.
SEARCH_EPOCHS=8
SEARCH_MAX_SEQUENCES=0
SEARCH_MAX_WINDOWS_PER_SEQUENCE=300
FULL_EPOCHS=60

TRAIN_NUM_WORKERS=2
EVAL_NUM_WORKERS=2

# Slurm grants 32 h. The internal 31 h budget leaves shutdown margin, while
# 10 h are protected for the full retrain and final validation/test evaluation.
MAX_HOURS=31.0
FINAL_RESERVE_HOURS=10.0
EVALUATION_RESERVE_HOURS=1.0
PER_TRIAL_TIMEOUT_HOURS=2.0

mkdir -p slurms
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

COMMAND=(
    python scripts/transformer/random_search_and_evaluate.py
    --base-config "$BASE_CONFIG"
    --search-name "$SEARCH_NAME"
    --output-dir "$OUTPUT_DIR"
    --n-trials "$N_TRIALS"
    --search-seed "$SEARCH_SEED"
    --training-seed "$TRAINING_SEED"
    --search-epochs "$SEARCH_EPOCHS"
    --search-max-sequences "$SEARCH_MAX_SEQUENCES"
    --search-max-windows-per-sequence "$SEARCH_MAX_WINDOWS_PER_SEQUENCE"
    --full-epochs "$FULL_EPOCHS"
    --train-num-workers "$TRAIN_NUM_WORKERS"
    --eval-num-workers "$EVAL_NUM_WORKERS"
    --max-hours "$MAX_HOURS"
    --final-reserve-hours "$FINAL_RESERVE_HOURS"
    --evaluation-reserve-hours "$EVALUATION_RESERVE_HOURS"
    --per-trial-timeout-hours "$PER_TRIAL_TIMEOUT_HOURS"
)

if [[ "$DRY_RUN" == true ]]; then
    COMMAND+=(--dry-run)
fi

"${COMMAND[@]}"

echo "Done: $SEARCH_NAME"
