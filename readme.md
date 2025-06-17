## Characterization and Mitigation of Training Instabilities in Microscaling Formats

Chloe Su*, Mujin Kwun, Stephanie Gil, Sham Kakade, Nikhil Anand*

### Setup
```
module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01
```
Create a Conda environment
```

```

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
│        ├── main.py               # train script
│        ├── utils.py              # Utility functions
│        ├── models/               # 
│        └── __init__.py           
│── notebooks/                # Jupyter notebooks for analysis
│    └──
│── nanoGPT/                  # nanoGPT code (not used in our paper finally)
│   ├── main.py               # train script
│   ├── utils.py              # Utility functions
│   ├── models/               
│   └── __init__.py        
│── configs/                  # Configuration files
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
