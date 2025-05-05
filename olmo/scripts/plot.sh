#!/bin/bash
#SBATCH --job-name=test-olmo-run
#SBATCH --output=/n/holylfs06/LABS/sham_lab/Users/chloe00/systems-scaling/olmo/logs/%A_%a.log
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=16
#SBATCH --time=71:30:00
#SBATCH --mem=0		

#SBATCH --account=kempner_grads
#SBATCH --partition=kempner
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --exclusive

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

# Accept job index as argument if there is a second argument
if [ -z "$3" ]
then
    echo $SLURM_ARRAY_TASK_ID
else
    export SLURM_ARRAY_TASK_ID=$3
fi

# TODO: does this help?
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/analyze_activations.py