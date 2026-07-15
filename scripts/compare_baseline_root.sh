#!/bin/bash -l
#SBATCH --job-name=compare_baseline_root
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=4:00:00
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

PYTHONPATH=src python -m field_converter.training.compare_baseline_root --config configs/mlp/root_mlp_v1_train.yaml

echo "Done"
