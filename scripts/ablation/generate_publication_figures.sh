#!/bin/bash -l
#SBATCH --job-name=ablation_figures
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

# Reuse the font cache across jobs. The task-local cache configured by
# common_env.sh is intentionally ephemeral and would rebuild on every run.
export MPLCONFIGDIR="$FIELD_CONVERTER_ROOT/outputs/ablation/.publication_cache/matplotlib"
export XDG_CACHE_HOME="$FIELD_CONVERTER_ROOT/outputs/ablation/.publication_cache/xdg"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# Publication snapshot settings. Edit these values, then submit this file with
# `sbatch scripts/ablation/generate_publication_figures.sh` (no CLI arguments).
MANIFEST="configs/ablation/unseen_match_v1.yaml"
OUTPUT_DIR="outputs/ablation/unseen_match_v1/publication"
FOLD="primary"
SPLIT="test"

FPS="25"
SPEED_WINDOW_FRAMES="5"
AIRBORNE_THRESHOLD_M="0.05"
GROUND_REFERENCE_PERCENTILE="20"
GROUND_REFERENCE_WINDOW_S="5"
MIN_AIRBORNE_FRAMES="2"
MAX_GROUND_GAP_FRAMES="1"

FOOT_BIN_WIDTH_M="0.025"
QUANTILE_BINS="10"
MIN_FRAMES_PER_BIN="100"
BOOTSTRAP_SAMPLES="2000"
CONFIDENCE_LEVEL="0.95"
BOOTSTRAP_SEED="314159"

mkdir -p slurms
python -m field_converter.ablation.publication_figures \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --fold "$FOLD" \
    --split "$SPLIT" \
    --fps "$FPS" \
    --speed-window-frames "$SPEED_WINDOW_FRAMES" \
    --airborne-threshold-m "$AIRBORNE_THRESHOLD_M" \
    --ground-reference-percentile "$GROUND_REFERENCE_PERCENTILE" \
    --ground-reference-window-s "$GROUND_REFERENCE_WINDOW_S" \
    --min-airborne-frames "$MIN_AIRBORNE_FRAMES" \
    --max-ground-gap-frames "$MAX_GROUND_GAP_FRAMES" \
    --foot-bin-width-m "$FOOT_BIN_WIDTH_M" \
    --quantile-bins "$QUANTILE_BINS" \
    --min-frames-per-bin "$MIN_FRAMES_PER_BIN" \
    --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
    --confidence-level "$CONFIDENCE_LEVEL" \
    --bootstrap-seed "$BOOTSTRAP_SEED"

echo "Publication figures written to $OUTPUT_DIR"
