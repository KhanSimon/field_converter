#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${FIELD_CONVERTER_ROOT:-/home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter}"
TASK_TOKEN="${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-main}"
umask 077

if [[ -n "${FIELD_CONVERTER_TMP_ROOT:-}" ]]; then
    TMP_ROOT_CANDIDATES=("$FIELD_CONVERTER_TMP_ROOT")
else
    # /tmp is unreliable on some GPU nodes. /dev/shm is local, short and
    # suitable for Python multiprocessing sockets; /var/tmp is the fallback.
    TMP_ROOT_CANDIDATES=("/dev/shm/fc_${UID}" "/var/tmp/fc_${UID}")
fi

TASK_TMP=""
for TMP_ROOT in "${TMP_ROOT_CANDIDATES[@]}"; do
    CANDIDATE="$TMP_ROOT/j$TASK_TOKEN"
    # Leave room below Linux's 108-byte AF_UNIX path limit for Python's
    # generated pymp-*/listener-* suffix.
    if (( ${#CANDIDATE} <= 60 )) && mkdir -p "$CANDIDATE" 2>/dev/null && [[ -w "$CANDIDATE" ]]; then
        TASK_TMP="$CANDIDATE"
        break
    fi
done

if [[ -z "$TASK_TMP" ]]; then
    echo "Unable to create a short writable temporary directory outside /tmp" >&2
    exit 2
fi

export TMPDIR="$TASK_TMP"
export TMP="$TASK_TMP"
export TEMP="$TASK_TMP"
export MPLCONFIGDIR="$TASK_TMP/matplotlib"
export XDG_CACHE_HOME="$TASK_TMP/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "Job temporary directory: $TASK_TMP"

if type module >/dev/null 2>&1; then
    module purge
    module load EasyBuild Anaconda3
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate cv_train_clean
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
