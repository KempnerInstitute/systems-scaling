#!/bin/bash
#SBATCH --account=kempner_grads
#SBATCH --job-name=color-filter
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/systems-scaling/olmo/logs/%A_%a.log
#SBATCH --nodes=1         
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=24
#SBATCH --time=36:00:00
#SBATCH --mem=100GB		
#SBATCH --partition=kempner_h100
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclude=holygpu8a15401
#SBATCH --exclusive

# sleep $((RANDOM % 120))

# module load cuda/12.4.1-fasrc01
# export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
source ~/.bashrc
mamba deactivate
mamba activate sys

# module load cudnn
# export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/lib64:$LD_LIBRARY_PATH
# module load gcc/10.2.0-fasrc01
# Custom environment
# source ~/.bashrc
# conda deactivate

module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01



export TORCH_FSDP_DEBUG=1
export HF_DATASETS_OFFLINE=1 # Only use cached data

export CONFIG=$1

# Accept sweep config as argument
export SWEEP_CONFIG=$2

# Accept job index as argument if there is a second argument
if [ -z "$3" ]
then
    echo $SLURM_ARRAY_TASK_ID
else
    export SLURM_ARRAY_TASK_ID=$3
fi

# Set default path for checkpoints if not set
export CHECKPOINTS_PATH="/n/netscratch/sham_lab/Lab/chloe00/ckpts"

# TODO: does this help?
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set ntasks if not set
if [ -z "$SLURM_NTASKS_PER_NODE" ]
then
    export SLURM_NTASKS_PER_NODE=1
fi

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python -u scripts/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}