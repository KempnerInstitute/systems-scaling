```module load cuda/12.4.1-fasrc01
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${HOME}/cuda-12.0/targets/x86_64-linux/include
module load gcc/12.2.0-fasrc01```


I made small edits in microxcaling.


systems-scaling/
│── microxcaling/                     # Folder containing datasets
│   ├──             # 
│   ├──             # Processed/cleaned data
│   └── README.md             # Documentation for data
│── nanoGPT/                  # nanoGPT (model) code
│   ├── main.py               #  script
│   ├── utils.py              # Utility functions
│   ├── models/               # 
│   └── __init__.py           
│── olmo/                  # OLMo (model) code
│   ├── main.py               #  script
│   ├── utils.py              # Utility functions
│   ├── models/               # 
│   └── __init__.py           
│── tmrc/                # Kempner transformer training lib
│   ├── main.py               #  script
│── notebooks/                # Jupyter notebooks for analysis
│── configs/                  # Configuration files
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
│── .gitignore                