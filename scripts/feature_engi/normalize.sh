#!/bin/bash -l
#SBATCH --job-name=normalize
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

PYTHONPATH=src python -c "from field_converter.data_preparation.normalize import Normalizer; Normalizer(in_features_dirname='features_wo_k', out_features_dirname='features_normalized_wo_k', pelvis_mode='hips_mean', min_bbox_size_px=10).run(train_n=65, valid_n=12, test_n=12, seed=12345, overwrite=True)"
