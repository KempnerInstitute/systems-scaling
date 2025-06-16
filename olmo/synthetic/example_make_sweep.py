#!/usr/bin/env python
"""
Generate an example TSV for the *learning‑rate* sweep.

Grid
----
depth     : [3, 4, 5,6,7,8]
width     : [256, 384, 512]
seed      : 1337

precision configs
    • mx‑mix  : fwd fp8_e4m3  | bwd fp8_e5m2
    • mx‑fp6  : fp6_e2m3 everywhere

learning‑rate variants
    • constant LR   : 1e‑5, 5e‑5, 1e‑4, 5e‑4, 1e‑3
    • cosine schedule: start 1e‑3  → 1e‑5  (adds --lr_schedule)

Fixed hyper‑parameters
----------------------
batch        2048
steps_total   7000
act          gelu
val_every     200
val_batch    2048
noise_std   0.005
layernorm    ON
teacher_*    default (scales with student)
project      test_student_teacher

Each TSV row is a *single* column that can be queued with:

    python student_teacher.py <CMD‑LINE>

Edit BASE_ARGS if you need extra flags 
"""

import itertools, csv, argparse, pathlib, random, textwrap

DEPTHS  = [3, 4, 5, 6, 7, 8]
WIDTHS  = [256, 384, 512]
SEEDS   = [1337]

# constant LRs (no --lr_schedule flag)m
CONST_LRS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

PRECISION_CFGS = {
    "mx-mix": ("fp8_e4m3", "fp8_e5m2"),
    "mx-fp6": ("fp6_e2m3", "fp6_e2m3"),
}

BASE_ARGS = [
    "--batch 2048",
    "--steps_total 7000",
    "--act gelu",
    "--val_batch 2048",
    "--val_every 200",
    "--noise_std 0.005",
    "--wandb_project test_student_teacher",
]

def cli(depth, width, seed, fmt_fwd, fmt_bwd, lr_max,
        schedule: bool=False, lr_min: float|None=None):
    args = BASE_ARGS + [
        f"--depth {depth}",
        f"--width {width}",
        f"--seed {seed}",
        f"--elem_format {fmt_fwd}",
        f"--elem_format_bp_w {fmt_bwd}",
        f"--elem_format_bp_ex {fmt_bwd}",
        f"--elem_format_bp_os {fmt_bwd}",
        f"--lr_max {lr_max:g}",
    ]
    if schedule:
        args += ["--lr_schedule", f"--lr_min {lr_min:g}"]
    return " ".join(args)

def main(out_tsv: str):
    rows = []

    # constant‑LR runs
    for depth, width, (fmt_fwd, fmt_bwd), lr, seed in itertools.product(
            DEPTHS, WIDTHS, PRECISION_CFGS.values(), CONST_LRS, SEEDS):
        rows.append(cli(depth, width, seed, fmt_fwd, fmt_bwd, lr))

    # cosine‑schedule run (one per (depth,width,prec,seed))
    for depth, width, (fmt_fwd, fmt_bwd), seed in itertools.product(
            DEPTHS, WIDTHS, PRECISION_CFGS.values(), SEEDS):
        rows.append(cli(depth, width, seed, fmt_fwd, fmt_bwd,
                        lr_max=1e-3, schedule=True, lr_min=1e-5))

    random.shuffle(rows)

    outfile = pathlib.Path(out_tsv)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with outfile.open("w", newline="") as fh:
        csv.writer(fh, delimiter="\t").writerows([[r] for r in rows])

    print(textwrap.dedent(f"""\
        Wrote {len(rows)} rows → {outfile}
        • depths     : {DEPTHS}
        • widths     : {WIDTHS}
        • precisions : {', '.join(PRECISION_CFGS)}
        • seeds      : {SEEDS}
        • LRs (const): {CONST_LRS}
        • + one cosine schedule per (depth,width,precision,seed)
    """))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outfile", nargs="?", default="lr_sweep.tsv",
                    help="destination TSV (default lr_sweep.tsv)")
    main(ap.parse_args().outfile)