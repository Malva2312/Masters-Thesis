# Python project for preprocessed data generation and visualization
A Python project to load [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) preprocessed data.

## Table of Contents
- [1. Project structure](#1-project-structure)
- [2. Environment Setup](#2-environment-setup)
- [3. Input files](#3-input-files)
- [4. Usage](#4-usage)
- [5. Features](#5-features)
- [6. Authors](#6-authors)
- [7. References](#7-references)

## 1. Project structure
```
├── conda_environment_files/
│   └── conda_environment_for_preprocessed_data_loading.yml  
│
├── config/
│   ├── data/
│   │   ├── preprocessed/
│   │   │   └── loader/
│   │   │       ├── lidc_idri_preprocessed_data_loader.yaml
│   │   │       └── lidc_idri_preprocessed_data_loader_jn_demo.yaml
│   │   └── raw/
│   │       └── loader/
│   │           └── lidc_idri_raw_data_loader.yaml
│   ├── metadata/   
│   │   └── preprocessed/
│   │       ├── lidc_idri_preprocessed_metadata.yaml
│   │       └── lidc_idri_preprocessed_metadata_jn_demo.yaml 
│   └── main.yaml 
│ 
├── docs/  
│   ├── BRAINSTORMING.md  
│   ├── CHANGELOG.md  
│   └── TODO.md    
│
├── jupyter_notebooks/  
│   └── preprocessed_data_loader_demo.ipynb  
│
├── src/  
│   └── modules/    
│       ├── data/
│       │   ├── data_loader/  
│       │   │   ├── preprocessed_data_loader.py
│       │   │   └── raw_data_loader.py
│       │   ├── data_augmentation.py  
│       │   └── metadata.py  
│       └── utils/   
│           └── paths.py
│ 
└── README.md  
```

## 2. Environment Setup

### Ubuntu + Conda

Steps to set up the remote Conda environment `preprocessed_data_loading_jn_venv` for preprocessed data loading:

1. Open the Ubuntu remote terminal at `conda_environment_files/`, then write and execute the following command to create the `preprocessed_data_loading_jn_venv` environment at `<PATH TO CONDA PARENT DIRECTORY>/conda/envs/` using the `conda_environment_for_preprocessed_data_loading.yml` file:

```commandline
conda env create -f conda_environment_for_preprocessed_data_loading.yml
```

2. Write and execute the following command in the Ubuntu terminal to activate the Conda environment:

```commandline
conda activate preprocessed_data_loading_jn_venv
```

[OPTIONAL] Additional steps to set up the Conda environment in Jupyter Notebook to run the `preprocessed_data_loader_demo.ipynb` demo script:

1. Write and execute the following command in the Ubuntu terminal to add the `preprocessed_data_loading_jn_venv` Conda environment to Jupyter Notebook as a kernel:

```commandline
python -m ipykernel install --user --name preprocessed_data_loading_jn_venv
```

2. Open the `preprocessed_data_loader_demo.ipynb` file located at `jupyter_notebook/` folder, go to Kernel -> Change kernel -> `preprocessed_data_loading_jn_venv`

[OPTIONAL] To remove the created Conda virtual environment, open the Ubuntu remote terminal, then write and execute the following commands:

```commandline
conda deactivate
conda env remove --name preprocessed_data_loading_jn_venv
```

## 3. Usage
A usage demonstration is provided in the Jupyter Notebook script preprocessed_data_loader_demo.ipynb. To run this script, follow the instructions in item 2 to set up the environment, then follow the instructions within the notebook.

## 4. Features
 - LIDC-IDRI data loader supporting:
   - Data augmentation based on the BigAug method proposed by Ling Zhang et al. (2020) in "Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation."
   - Different data preprocessing protocols 
   - Stratified K-fold cross-validation


## 5. Authors
 - Eduardo de Matos Rodrigues

## 6. References
 - Python frameworks:
   - [Albumentations](https://albumentations.ai/)
   - [Hydra](https://hydra.cc/)
   - [PyTorch Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
 - Python project development best practices
   - [Keep a changelog](https://keepachangelog.com/en/1.1.0/)
   - [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
   - [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
