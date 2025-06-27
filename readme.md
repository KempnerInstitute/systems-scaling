# [Characterization and Mitigation of Training Instabilities in Microscaling Formats](https://arxiv.org/abs/2506.20752)

[Chloe Su](https://x.com/Huangyu58589918)*, [Mujin Kwun](https://x.com/MJK12341234), Stephanie Gil, [Sham Kakade](https://x.com/ShamKakade6), [Nikhil Anand](https://x.com/nikhil_anand91)\*

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2506.20752-B31B1B.svg)](https://arxiv.org/abs/2506.20752)



### Setup
```
module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01
```
Create a Conda environment and Install dependencies
```
git clone git@github.com:Hither1/systems-scaling.git
cd systems-scaling
conda create -n scaling python=3.10
conda activate scaling
pip install -e .[all]
```
### Usage
#### Language model training
In `olmo/` folder, run
```
sbatch scripts/launch_sweep_scale.sh configs/base.yaml configs/sweeps/scale.yaml
```

#### Synthetic training

Code for the student-teacher model experiments on synthetic data can be found under `olmo/synthetic`.  For example, you can run 

```
python synthetic/student_teacher.py --depth 4 --width 512 --batch 2048 --lr_max 6e-4 --wandb_project <YOUR PROJECT NAME> --store_full_gradients --log_weight_clipping --val_every 100 --steps 9000 --save_checkpoints --checkpoint_window_center 14100 --checkpoint_window_size 200 --checkpoint_every 20
```

which will run the synthetic experiment and save checkpoints at starting at global step 13900 every 20 steps.  To run an intervention experiment, for example returning to full precision at the intervention point, you can run

```
python synthetic/student_teacher.py --run_intervention --intervention_checkpoint <PATH TO YOUR CHECKPOINT> --depth 4 --width 512 --batch 2048 --lr_max 6e-4 --steps_total 500 --store_full_gradients --log_weight_clipping --val_every 100 --wandb_project <YOUR PROJECT NAME> --dont_inject_mx_ops
```

#### Small edits in microxcaling. 
* Minor changes were made to the MX Pytorch Simulation Library in order to experiment with selectively quantizing different parts of the network, e.g., turning off quantization for LayerNorm affine parameters.  By default, all of these modifications are turned off and so the library will behave as expected.


### Contents
```
systems-scaling/             
│── olmo/
│   │── mx/             # microxcaling library (with our modificaitons)
│   │    ├── activations.py                      
│   │    ├── layernorm.py
│   │    ├── mx_mapping.py
│   │    ├── mx_ops.py
│   │    ├── norm_utils.py
│   │    ├── specs.py              
│   │    └── vector_ops.py
│   │            
│   │── synthetic
│   │   └── student_teacher.py
│   │
│   └── olmo/               # OLMo (model) code
│        ├── main.py               # Configuration files
│        ├── configs/              #  functions
│        ├── scripts/              # 
│        └── __init__.py           
│── plot/                # Jupyter notebooks/scripts for plots and analysis
│    ├── accuracy_relationships.ipynb
│    └── curve_fitting_and_instability.ipynb    # Scaling law plots
│   
│── nanoGPT/                  # nanoGPT code (not used in our paper finally)
│   ├── main.py               # train script
│   ├── models/               
│   └── __init__.py
│        
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
│── .gitignore            
```


### Citation

If you use this code in your research, please cite the following papers and repositories:

Our paper:
```
@misc{su2025characterizationmitigationtraininginstabilities,
      title={Characterization and Mitigation of Training Instabilities in Microscaling Formats}, 
      author={Huangyuan Su and Mujin Kwun and Stephanie Gil and Sham Kakade and Nikhil Anand},
      year={2025},
      eprint={2506.20752},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.20752}, 
}
```

OLMo and the Pytorch MX Simulator
```
@article{groeneveld2024olmo,
  title={Olmo: Accelerating the science of language models},
  author={Groeneveld, Dirk and Beltagy, Iz and Walsh, Pete and Bhagia, Akshita and Kinney, Rodney and Tafjord, Oyvind and Jha, Ananya Harsh and Ivison, Hamish and Magnusson, Ian and Wang, Yizhong and others},
  journal={arXiv preprint arXiv:2402.00838},
  year={2024}
}

@misc{mx_library,
  author = {{Microsoft}},
  title = {MX Pytorch Emulation Library},
  year = {2024},
  url = {https://github.com/microsoft/microxcaling/tree/main},
  urldate = {2025-05-15}
}
```

(Optional) Downstream evaluation
```bash
    from olmo.eval.downstream import *
    tokenizer = Tokenizer.from_file("olmo/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json")
    for x in label_to_task_map.values():
        print(x)
        kwargs = {}
        if isinstance(x, tuple):
            x, kwargs = x
        x(tokenizer=tokenizer, **kwargs)
    ```