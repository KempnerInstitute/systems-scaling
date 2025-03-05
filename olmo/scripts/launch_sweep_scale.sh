#!/bin/bash
#SBATCH --job-name=test-olmo-run
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/systems-scaling/olmo/logs/%A_%a.log
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=24
#SBATCH --time=71:30:00
#SBATCH --mem=1000GB		
#SBATCH --account=kempner_grads
#SBATCH --partition=kempner_h100
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --array=9-96%1

sleep $((RANDOM % 240))


# Custom environment
source ~/.bashrc
mamba deactivate
mamba activate sys

# sleep $(( SLURM_ARRAY_TASK_ID * 60 ))
module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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