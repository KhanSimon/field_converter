#!/bin/bash -l
#SBATCH --job-name=feature_creation
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:T4:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simonkhan160@gmail.com
#SBATCH --output=slurms/slurm_%j.out
#SBATCH --error=slurms/slurm_%j.err



module purge
module load EasyBuild Anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv_train


cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms

PYTHONPATH=src python -m field_converter.training.check_rotation_conventions \
    --check-sam \
    --image-size 1920 1080
##    --max-sequences 5 \
##    --save-overlays \
##    --image-root images_gt \
##    --num-frames 20 \
##    --overlay-count 20 \
##    --sequence NET_ARG_231908 \
    

## commenter les dernières lignes pour ne pas save les overlays. 