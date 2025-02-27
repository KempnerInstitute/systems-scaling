#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 1 # Number of cores
#SBATCH -p gpu
#SBATCH -t 2-23:00
#SBATCH --job-name=pretrain_gpt
#SBATCH --output=log/output-%j.out
#SBATCH --error=log/error-%j.err
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mail-user=weifanjiang@g.harvard.edu
#SBATCH --mail-type=END
#SBATCH --mem=250G # Memory per cpu in MB

# default args
precision="bfloat16"
round="None"

while getopts "p:r:" opt; do
  case $opt in
    p) precision=$OPTARG ;;
    r) round=$OPTARG ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done


module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01

echo "Command-line inputs: precision $precision round $round"
# Training on single node
torchrun --standalone --nproc_per_node=4 train.py -p $precision -r $round

# Training on multi nodes
