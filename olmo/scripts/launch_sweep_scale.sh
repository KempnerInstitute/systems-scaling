#!/bin/bash
#SBATCH --job-name=test-olmo-run
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling/olmo/logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=16
#SBATCH --time=71:00:00
#SBATCH --mem=100GB		
#SBATCH --account=kempner_dev
#SBATCH --partition=kempner

#SBATCH --array=1-6

# sleep $((RANDOM % 120))


# Custom environment
source ~/.bashrc
mamba deactivate
mamba activate olmo_test

module load cudnn
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/lib64:$LD_LIBRARY_PATH
module load gcc/10.2.0-fasrc01



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
export CHECKPOINTS_PATH="/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling-ckpts/"

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

python scripts/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}