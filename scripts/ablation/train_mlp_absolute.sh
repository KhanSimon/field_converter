#!/bin/bash -l
#SBATCH --job-name=mlp_absolute
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err
#SBATCH --export=NONE

set -euo pipefail
export FIELD_CONVERTER_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
source "$FIELD_CONVERTER_ROOT/scripts/ablation/common_env.sh"

# Everything is configured here; submit this file without terminal arguments.
MANIFEST="configs/ablation/unseen_match_v1.yaml"
RUN_INDEX="25"
EXPECTED_RUN_NAME="primary_mlp_absolute_s1235"
FORCE_TRAIN=false
FORCE_EVAL=false

mkdir -p slurms

# Materialize the appended run while preserving indices 0-24.
python -m field_converter.ablation.generate "$MANIFEST"
PLAN_PATH="outputs/ablation/unseen_match_v1/plan.json"
ACTUAL_RUN_NAME="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["runs"][int(sys.argv[2])]["run_name"])' "$PLAN_PATH" "$RUN_INDEX")"
if [[ "$ACTUAL_RUN_NAME" != "$EXPECTED_RUN_NAME" ]]; then
    echo "Run-index mismatch: expected $EXPECTED_RUN_NAME at $RUN_INDEX, found $ACTUAL_RUN_NAME" >&2
    exit 2
fi

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"

COMMAND=(python -m field_converter.ablation.run_experiment "$MANIFEST" "$RUN_INDEX")
if [[ "$FORCE_TRAIN" == true ]]; then
    COMMAND+=(--force-train)
fi
if [[ "$FORCE_EVAL" == true ]]; then
    COMMAND+=(--force-eval)
fi
"${COMMAND[@]}"

echo "Done: $EXPECTED_RUN_NAME"
