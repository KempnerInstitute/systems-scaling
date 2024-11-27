#!/bin/bash
#SBATCH --job-name=distributed_matmul
#SBATCH --account=kempner_grads
#SBATCH -p kempner                    # Partition name
#SBATCH --output=log/output_%j.txt
#SBATCH --error=log/error_%j.txt
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --gres=gpu:2                  # Number of GPUs per node
#SBATCH --time=02:00:00               # Maximum wall time
#SBATCH --mem=375G                    # Memory per node
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# export FI_PROVIDER="efa"
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export NCCL_SOCKET_IFNAME="ens"
export FI_EFA_USE_DEVICE_RDMA=1

eval "$(conda shell.bash hook)"
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
conda activate /n/home13/chloe00/miniconda3/envs/mmc

export WORLD_SIZE=4
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID
export MASTER_PORT=$(shuf -i 10000-65535 -n 1)  # Random port in range
export RDZV_ENDPOINT="$head_node_ip:$MASTER_PORT"

# srun torchrun --nproc_per_node=2 --nnodes=2 --node_rank=$NODE_RANK --master_addr=$head_node_ip --master_port=12355 gpu_2_nodes.py
# srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node 2 \
#             --rdzv_id $RANDOM --rdzv_backend c10d \
#          --rdzv_endpoint $head_node_ip:29500 \
#          ./gpu_2_nodes.py
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=2 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    gpu_2_nodes.py