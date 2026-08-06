#!/bin/bash -l
#SBATCH --job-name=train_root_transformer
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=14:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err



module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv_train_clean
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms

nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('available:', torch.cuda.is_available())"

PYTHONPATH=src python -m field_converter.training.train_root_transformer --config configs/transformer/root_transformer_v1.yaml
echo "Done"
