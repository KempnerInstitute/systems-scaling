# Training code for Loss-to-loss prediction

## Description

Code to train models that was used for the experiments on loss-to-loss prediction in the paper [Loss-to-loss prediction: scaling laws for all datasets](https://arxiv.org/abs/2411.12925).

For data analysis and results see https://github.com//loss-to-loss-notebooks.

For model checkpoints see https://huggingface.co//loss-to-loss.

## Installation

Build a conda environment and install the required packages:

```bash
conda create -n loss-to-loss python=3.10
conda activate loss-to-loss
pip install -e .[all]
```

## Usage

Training models requires preparing all of the tokenized data and then running the following command to launch the training sweep on a slurm cluster:

```bash
sbatch launch_sweep_scale.sh configs/base.yaml configs/sweeps/scale.yaml
```

Note: you will need to update paths to the data in `olmo/registry.py` and define the array job as well as paths for logs and checkpoints in `launch_sweep_scale.sh`.

## Citation

If you use this code in your research, please cite the following papers:


