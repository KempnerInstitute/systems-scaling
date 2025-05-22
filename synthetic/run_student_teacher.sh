#!/bin/bash
#SBATCH --job-name=theory-sweeps-eps
#SBATCH --output=<LOG_DIR>/theory_sweeps_eps_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=71:30:00
#SBATCH --mem=24G
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --array=<START>-<END>

mamba init
mamba activate <ENV_NAME>

module load cuda/12.4.1-fasrc01
module load gcc/12.2.0-fasrc01

export PYTHONPATH=<PATH_TO_SYSTEMS_SCALING_DIR>:$PYTHONPATH

# change table below to a sweep of your choice
TABLE=<DIR>/theory_sweeps_eps.tsv
CLI=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $TABLE | tr -d '\r')

echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $CLI"

python3 <DIR>/student_teacher.py $CLI