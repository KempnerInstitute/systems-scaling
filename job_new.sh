#!/bin/bash
#SBATCH --job-name=distributed_matmul
#SBATCH --account=kempner_grads
#SBATCH -p kempner
#SBATCH --output=log/output_%j.txt
#SBATCH --error=log/error_%j.txt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --gpus-per-node=2        # GPUs per node
#SBATCH --time=02:00:00
#SBATCH --mem=375G
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

# Load modules and activate environment
eval "$(conda shell.bash hook)"
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
conda activate /n/home13/chloe00/miniconda3/envs/mmc

# Get head node IP
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Ensure the IP is correctly extracted
echo "Head node IP: $head_node_ip"

# Run torchrun with srun
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    gpu_2_nodes.py
