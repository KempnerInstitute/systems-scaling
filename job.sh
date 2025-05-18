#!/bin/bash
#SBATCH --job-name=distributed_test
#SBATCH --account={}
#SBATCH --partition=kempner_h100_priority
#SBATCH --output %x_%j/output_%j.out
#SBATCH --error %x_%j/error_%j.out
#SBATCH --time=01:00:00
#SBATCH --nodes=2                   # 2 nodes
#SBATCH --ntasks-per-node=4         # 4 GPUs per node -> 4 tasks per node
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --exclusive

source ~/.bashrc
mamba deactivate
mamba activate sys

# sleep $(( SLURM_ARRAY_TASK_ID * 60 ))
module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01

# Set MASTER_ADDR and MASTER_PORT for PyTorch communication
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Total number of tasks across all nodes
export WORLD_SIZE=$SLURM_NTASKS

# Run script with srun, assigning RANK and LOCAL_RANK manually
srun --export=ALL bash -c '\
    export RANK=$SLURM_PROCID; \
    export LOCAL_RANK=$SLURM_LOCALID; \
    python test_nodes.py \
'