#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --gres=gpu:T4:1




module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv_train


cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms

PYTHONPATH=src python -m field_converter.utils.test