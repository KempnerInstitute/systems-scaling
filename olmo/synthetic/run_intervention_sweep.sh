#!/bin/bash
#SBATCH --job-name=intervention_sweep
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling/intervention_sweep/logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=71:30:00
#SBATCH --mem=256G
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-15

export PYTHONPATH=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling-anon:$PYTHONPATH
export CUBLAS_WORKSPACE_CONFIG=:4096:8


module load cuda/12.4.1-fasrc01 && module load gcc/12.2.0-fasrc01

mamba activate olmo_test

TABLE=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling-anon/systems-scaling/olmo/synthetic/intervention_sweep_4.tsv
CLI=$(awk -F'\t' -v id="$SLURM_ARRAY_TASK_ID" 'NR==id+1 { $1=""; sub(/^\t/, ""); gsub(/\r/, ""); print }' "$TABLE")

echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $CLI"
eval "$CLI"