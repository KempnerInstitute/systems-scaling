#!/bin/bash
#SBATCH --account=kempner_grads
#SBATCH --job-name=distributed_matmul
#SBATCH --output=log/output_%j.txt        # Output file
#SBATCH --error=log/error_%j.txt          # Error file
#SBATCH --ntasks=2                    # Total number of tasks (GPUs)
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --gres=gpu:2                  # Number of GPUs per node
#SBATCH --time=02:00:00               # Maximum wall time
#SBATCH -p kempner_h100               # Partition name (if applicable)
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ens"
export FI_EFA_USE_DEVICE_RDMA=1

eval "$(conda shell.bash hook)"
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01
conda activate /n/home13/chloe00/miniconda3/envs/mmc


# Set the world size (total number of processes across all nodes)
export WORLD_SIZE=4

# Set the rank for this particular node and task
export RANK=$SLURM_PROCID             # SLURM_PROCID gives the global rank of the task
export LOCAL_RANK=$SLURM_LOCALID      # Local rank of the task (0, 1, ... per node)

# Activate your Python environment (optional)
# source /path/to/your/env/bin/activate

# Run the distributed PyTorch script
srun torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=$(hostname -s) --master_port=12355 gpu_2_nodes.py