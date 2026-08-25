#!/bin/bash -l
#SBATCH --job-name=root_inference
#SBATCH --partition=GPU_Compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd /home/BeeGFS/Laboratories/IBHGC/skhan/Documents/field_converter

mkdir -p slurms
nvidia-smi || true
python -c "import torch; print('torch:', torch.__version__); print('cuda build:', torch.version.cuda); print('available:', torch.cuda.is_available())"
PYTHONPATH=src python -m field_converter.inference \
    --model-type transformer \
    --config outputs/eval_reports/root_transformer_v1_delta_new_root_init_wo_vj_gi_pp_k/config_used.yaml \
    --checkpoint best \
    --sequence ekstraklasa_001999 \
    --source-fps 25\
    --target-fps 50

    echo "Done"