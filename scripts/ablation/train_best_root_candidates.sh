#!/bin/bash -l
#SBATCH --job-name=best_root_candidates
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

# Edit these values in this file; the sbatch command takes no arguments.
FORCE_TRAIN=false
FORCE_EVAL=false
OUTPUT_ROOT="outputs/ablation/unseen_match_v1"

CONFIGS=(
    "configs/ablation/best_candidate_tcn.yaml"
    "configs/ablation/best_candidate_transformer.yaml"
)
RUN_NAMES=(
    "primary_tcn_best_candidate_no_pitch_ground_mask_s1235"
    "primary_transformer_best_candidate_no_pitch_ground_mask_s1235"
)
TRAIN_MODULES=(
    "field_converter.training.train_root_tcn"
    "field_converter.training.train_root_transformer"
)
EVAL_MODULES=(
    "field_converter.training.evaluate_root_tcn"
    "field_converter.training.evaluate_root_transformer"
)

TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This script must be submitted with sbatch}"
if (( TASK_INDEX < 0 || TASK_INDEX >= ${#CONFIGS[@]} )); then
    echo "Invalid array task $TASK_INDEX; expected 0..$((${#CONFIGS[@]} - 1))" >&2
    exit 2
fi

CONFIG="${CONFIGS[$TASK_INDEX]}"
RUN_NAME="${RUN_NAMES[$TASK_INDEX]}"
TRAIN_MODULE="${TRAIN_MODULES[$TASK_INDEX]}"
EVAL_MODULE="${EVAL_MODULES[$TASK_INDEX]}"
CHECKPOINT="$OUTPUT_ROOT/checkpoints/$RUN_NAME/best.pt"
TRAIN_SUMMARY="$OUTPUT_ROOT/eval_reports/$RUN_NAME/train_summary.json"
METRICS="$OUTPUT_ROOT/eval_reports/$RUN_NAME/metrics.json"

mkdir -p slurms
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

if [[ "$FORCE_TRAIN" == true || ! -f "$CHECKPOINT" || ! -f "$TRAIN_SUMMARY" ]]; then
    echo "Training $RUN_NAME with $CONFIG"
    python -m "$TRAIN_MODULE" --config "$CONFIG"
else
    echo "Training already complete; resuming at evaluation: $CHECKPOINT"
fi

if [[ "$FORCE_TRAIN" == true || "$FORCE_EVAL" == true || ! -f "$METRICS" ]]; then
    EVAL_COMMAND=(python -m "$EVAL_MODULE" --config "$CONFIG" --checkpoint best)
    if [[ "$TASK_INDEX" == "1" ]]; then
        EVAL_COMMAND+=(--no_baseline)
    fi
    "${EVAL_COMMAND[@]}"
else
    echo "Evaluation already complete: $METRICS"
fi

echo "Done: $RUN_NAME"
