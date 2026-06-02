#!/bin/bash -l
#SBATCH --job-name=compare_baselines
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:V100S:1
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

## calcul la moyenne (baseline naif) sur le split train et évalue sur les splits valid et test, produit metrics.json
##PYTHONPATH=src python -m field_converter.training.compare_baseline_root --config configs/mlp/root_mlp_v1_train.yaml

## lit les fichiers metrics.json pour le : baseline naif, le mlp et le tcn. 
PYTHONPATH=src python -m field_converter.training.compare_root_models --mlp_config configs/mlp/root_mlp_v1_train.yaml --tcn_config configs/tcn/root_tcn_v1.yaml --split valid
echo "Done"


