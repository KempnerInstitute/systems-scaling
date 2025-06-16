#!/usr/bin/env python
"""
Create sweep.tsv: one line = <index>\t<command string>.
Edit PATHS or COMMON_FLAGS if you change directories or hyper-params.
"""

import itertools, pathlib, uuid, csv

REPO_ROOT = pathlib.Path("/n/holylfs06/LABS/kempner_dev/Lab/nikhilanand/systems-scaling-anon/systems-scaling/olmo")
CHECKPOINT_DIR = pathlib.Path(
    "/n/netscratch/kempner_dev/Lab/nikhilanand/eps_sweeps_depth_width/"
    "a_fmtfp8_e4m3_a_bp_exfp8_e4m3_a_bp_osfp8_e4m3_actgelu_bs2048_L4_lr6e-04_s1337_fmtfp8_e4m3_bpfp8_e4m3_D512_b5b6/"
    "checkpoints"
)

CKPTS = [
    "ckpt_mx_step13500.pt",
    "ckpt_mx_step13700.pt",
    #"ckpt_mx_step14100.pt",
    #"ckpt_mx_step14200.pt",
]

COMMON_FLAGS = (
    "--run_intervention "
    "--depth 4 --width 512 --batch 2048 "
    "--lr_max 6e-4 "
    "--wandb_project mx_intervention "
    "--store_full_gradients --log_weight_clipping "
    "--val_every 100 --steps_total 9000 "
)

INTERVENTIONS = {
    # tag                  flags
    "no_mx_inject":        "--dont_inject_mx_ops",
    "bump_exp":           "--bump_up_overflow_exponent",
    "no_ln_q":            "--dont_quantize_layernorm",
    "bf16_act":           "--a_elem_format bfloat16 "
                          "--a_elem_format_bp_ex fp8_e4m3 "
                          "--a_elem_format_bp_os fp8_e4m3 "
                          "--use_custom_cuda",
    "bf16_act_all":       "--a_elem_format bfloat16 "
                          "--a_elem_format_bp_ex bfloat16 "
                          "--a_elem_format_bp_os bfloat16 "
                          "--use_custom_cuda",
    "no_bp_quant":        "--dont_quantize_backprop",
    "no_gelu_quant":      "--dont_quantize_gelu",
    "bf16_weight":        "--w_elem_format bfloat16",
}

rows = []
for ckpt, (tag, flags) in itertools.product(CKPTS, INTERVENTIONS.items()):
    ckpt_path = CHECKPOINT_DIR / ckpt
    wandb_tag = f"{tag}_{ckpt.split('_')[-1].split('.')[0]}"
    cmd = (
        f"python {REPO_ROOT}/synthetic/student_teacher_v3.py "
        f"{COMMON_FLAGS}"
        f"--intervention_checkpoint {ckpt_path} "
        f"{flags} "
        f"--wandb_name {wandb_tag}"
    )
    rows.append(cmd)

# full-length BF16 activation run (no --run_intervention)
FULL_RUN = (
    f"python {REPO_ROOT}/synthetic/student_teacher_v3.py "
    "--depth 4 --width 512 --batch 2048 "
    "--lr_max 6e-4 "
    "--wandb_project mx_intervention "
    "--store_full_gradients --log_weight_clipping "
    "--val_every 100 --steps_total 9000 "
    "--save_checkpoints --checkpoint_window_center 14100 "
    "--checkpoint_window_size 200 --checkpoint_every 20 "
    "--a_elem_format bfloat16 "
    "--a_elem_format_bp_ex bfloat16 "
    "--a_elem_format_bp_os bfloat16 "
    "--wandb_name full_bf16_act"
)
rows.append(FULL_RUN)

with open("intervention_sweep_4.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    for idx, cmd in enumerate(rows):
        writer.writerow([idx, cmd])

print(f"Wrote sweep.tsv with {len(rows)} commands.")
