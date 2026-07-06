#!/bin/bash -l
#SBATCH --job-name=eval_tcn_grid_search
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err

set -euo pipefail
module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv_train
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export MPLCONFIGDIR="${SLURM_TMPDIR:-$PWD/.cache}/matplotlib"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-$PWD/.cache}/field-converter"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms

nvidia-smi || true
python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('available:', torch.cuda.is_available())"

PYTHONPATH=src python scripts/tcn/evaluate_random_search.py \
  --search-kind grid_search \
  --search-name root_tcn_grid_search_rs1 \
  --output-dir outputs \
  --metric best_root_error_mean_m \
  --top-k 8

echo "Done"
