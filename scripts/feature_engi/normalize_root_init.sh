#!/bin/bash -l
#SBATCH --job-name=feature_creation
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err



module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv_train_clean


cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms

PYTHONPATH=src python -m field_converter.data_preparation.normalize_root_init --features-normalized-dirname features_normalized --overwrite