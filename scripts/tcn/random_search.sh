#!/bin/bash -l
#SBATCH --job-name=tcn_random_search
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err

module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv_train
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms

nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('available:', torch.cuda.is_available())"

PYTHONPATH=src python scripts/tcn/random_search.py \
  --base-config configs/tcn/root_tcn_v1.yaml \
  --search-name root_tcn_random_search \
  --n-trials 12 \
  --seed 1234 \
  --output-dir outputs \
  --skip-existing

echo "Done"
