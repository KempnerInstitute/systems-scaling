# Characterization and Mitigation of Training Instabilities in Microscaling Formats

[Chloe Su](https://x.com/Huangyu58589918)*, [Mujin Kwun](https://x.com/MJK12341234), Stephanie Gil, [Sham Kakade](https://x.com/ShamKakade6), [Nikhil Anand](https://x.com/nikhil_anand91)\*

### Setup
```
module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01
```
Create a Conda environment
```
git clone systems-scaling
cd systems-scaling
conda create 
```
Install dependencies


Small edits in microxcaling. 
* microxcaling: 
* Tatm: data/datasets.py


### Contents
```
systems-scaling/
│── microxcaling/             # Folder containing datasets
│   ├──                       # 
│   ├──                       
│   └── README.md                
│── olmo/                  
│   │── synthetic
│   │   └── student_teacher.py
│   │
│   └── olmo/               # OLMo (model) code
│        ├── main.py               # Configuration files
│        ├── configs/              #  functions
│        ├── scripts/               # 
│        └── __init__.py           
│── plot/                # Jupyter notebooks/scripts for plots and analysis
│    ├── accuracy_relationships.ipynb
│    └── curve_fitting_and_instability.ipynb    # Scaling law plots
│── nanoGPT/                  # nanoGPT code (not used in our paper finally)
│   ├── main.py               # train script
│   ├── models/               
│   └── __init__.py        
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
│── .gitignore            
```



If you use this work, please cite:
```
bibtex
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
