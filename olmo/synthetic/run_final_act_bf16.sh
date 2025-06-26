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
#SBATCH --array=0-0

export PYTHONPATH=/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling:$PYTHONPATH

module load cuda/12.4.1-fasrc01 && module load gcc/12.2.0-fasrc01

mamba activate olmo_test

python /n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling-anon/systems-scaling/olmo/synthetic/student_teacher_v3.py --depth 4 --width 512 --batch 2048 --lr_max 6e-4 --wandb_project mx_intervention --store_full_gradients --log_weight_clipping --val_every 100 --steps_total 9000 --save_checkpoints --checkpoint_window_center 14100 --checkpoint_window_size 200 --checkpoint_every 20 --a_elem_format bfloat16 --a_elem_format_bp_ex bfloat16 --a_elem_format_bp_os bfloat16 --wandb_name full_bf16_act
