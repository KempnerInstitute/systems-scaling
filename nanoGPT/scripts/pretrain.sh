#!/bin/bash
#SBATCH --account=kempner_grads
#SBATCH --nodes=1
#SBATCH -n 1 # Number of cores
#SBATCH -p kempner_h100
#SBATCH -t 2-23:00
#SBATCH --job-name=evaluate_llm_rollout
#SBATCH --output=log/output-%j.out
#SBATCH --error=log/error-%j.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mail-user=csu@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --mem=250G # Memory per cpu in MB


module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01s

# Training on single node
torchrun --standalone --nproc_per_node=4 train.py