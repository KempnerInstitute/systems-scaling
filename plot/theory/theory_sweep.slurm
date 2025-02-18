#!/bin/bash
#SBATCH --job-name=theory_sweep_loss_scaling
#SBATCH --output=theory_sweep_loss_scaling_%A_%a.out
#SBATCH --error=theory_sweep_loss_scaling_%A_%a.err
#SBATCH --array=1-6  # Adjust this according to the number of values of N
#SBATCH --cpus-per-task=64  # Adjust based on your computational needs
#SBATCH --time=24:00:00    # Adjust based on your computational needs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --mem=128G          # Adjust based on your memory needs

# Load required modules (e.g., JAX if available as a module, otherwise ensure it's in your environment)

mamba init # use conda if mamba is not available
mamba activate <environment name here> # adjust based on your environment name

# List of N values to iterate over, must match the SLURM array index
N_values=(100 200 300 400 500 600)
N=${N_values[$SLURM_ARRAY_TASK_ID - 1]}

echo "Job Starting"

python theory_sweep_gpu.py $N >> task_output.txt