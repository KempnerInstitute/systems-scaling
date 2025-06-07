#!/bin/bash
#SBATCH --job-name=test_ln_fix
#SBATCH --output=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling/test_ln_fix/logs/%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=71:30:00
#SBATCH --mem=48G
#SBATCH --partition=kempner_eng
#SBATCH --account=kempner_dev
#SBATCH --array=0-1

export PYTHONPATH=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling:$PYTHONPATH

module load cuda/12.4.1-fasrc01 && module load gcc/12.2.0-fasrc01

mamba activate olmo_test

python synthetic/student_teacher_v2.py --depth 3 --width 512 --batch 2048 --lr_max 5e-4 --wandb_project mx_measure_clip --store_full_gradients --log_weight_clipping --val_every 100 --steps 11000 --dont_quantize_layernorm
python synthetic/student_teacher_v2.py --depth 3 --width 512 --batch 2048 --lr_max 5e-4 --wandb_project mx_measure_clip --store_full_gradients --log_weight_clipping --val_every 100 --steps 11000 --bump_up_overflow_exponent