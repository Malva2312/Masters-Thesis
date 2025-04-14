# Python project for a PyTorch Lightning deep learning pipeline template
A Python project providing a structured template for deep learning pipelines using PyTorch Lightning, designed to streamline model development and experimentation.
A Python project to generate and visualize data from the raw [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/) dataset.

## Table of Contents
- [1. Project structure](#1-project-structure)
- [2. Environment Setup](#2-installation)
- [3. Input and output files](#3-input-and-output-files)
- [4. Usage](#4-usage)
- [5. Features](#5-features)
- [6. Examples](#6-examples)
- [7. Authors](#7-authors)
- [8. References](#8-references)

## 1. Project structure
```
├── conda_environment_files/  
│   └── luna25_conda_environment.yaml
├── config_files/  
│   ├── data/  
│   │   ├── dataloader/  
│   │   │   ├── lidc_idri_preprocessed_dataloader.yaml     
│   │   │   └── luna25_preprocessed_dataloader.yaml   
│   │   └── metadataframe/
│   │       ├── lidc_idri_preprocessed_metadataframe.yaml     
│   │       └── luna25_preprocessed_metadataframe.yaml  
│   ├── experiment_execution/  
│   │   ├── ids/  
│   │   │   └── ids.yaml   
│   │   ├── info/
│   │   │   └── info.yaml  
│   │   ├── paths/
│   │   │   └── paths.yaml 
│   │   └── data.yaml  
│   ├── hyperparameter_grid_based_execution/  
│   │   └── luna25_hyperparameter_grid_based_execution.yaml  
│   ├── model/  
│   │   ├── model_pipeline/  
│   │   │   └── luna25_model_pipeline.yaml   
│   │   ├── pytorch_lightning_model/
│   │   │   └── luna25_pytorch_lightning_efficient_net.yaml  
│   │   └── model.yaml  
│   ├── results_analysis/  
│   │   ├── model_test_performance_metrics_dataframe/  
│   │   │   └── luna25_model_test_performance_metrics_dataframe.yaml   
│   │   └── model_training_performance_metrics_figure/
│   │       └── luna25_model_training_performance_metrics_figure.yaml  
│   └── main.yaml  
├── documents/  
│   ├── BRAINSTORMING.md  
│   ├── CHANGELOG.md  
│   ├── EXPERIMENT_LOGS.md
│   └── TODO.md  
├── experiment_results/   
├── jupyter_notebooks/  
├── slurm_files/  
│   ├── output_files/  
│   └── shell_script_files/  
│       └── run_job.sh  
├── src/  
│   ├── modules/  
│   │   ├── config/  
│   │   │   └── configuration_settings.py  
│   │   ├── data/  
│   │   │   ├── data_transformations.py  
│   │   │   ├── dataloader.py  
│   │   │   └── metadataframes.py  
│   │   ├── jupyter_notebook/  
│   │   │   └── figure_plotting.py  
│   │   ├── lung/  
│   │   │   └── lung_nodule.py
│   │   ├── preprocessed_data_generation_protocol/  
│   │   │   ├── __init__.py  
│   │   │   ├── info.py  
│   │   │   └── utils.py
│   │   └── utils/   
│   │       ├── execution_datetimes.py  
│   │       └── paths.py
│   ├── pylidc_files_to_replace/  
│   │   ├── Annotation.py  
│   │   ├── Contour.py  
│   │   └── Scan.py  
│   └── scripts/  
│       ├── generate_lung_nodule_dataset.py  
│       └── visualize_preprocessed_data.py  
└── README.md  
```

## 2. Environment Setup

### Ubuntu + Conda

Steps to set up the remote Conda environment `<dataset_name>_pl_dl_pipeline_template_venv`:

1. Open the Ubuntu remote terminal at `conda_environment_files/`, then write and execute the following command to create the `<dataset_name>_pl_dl_pipeline_template_venv` environment at `/nas-ctm01/homes/<slurm_username>/.conda/envs/` using the `<dataset_name>_conda_environment.yaml` file:

```commandline
conda env create -f conda_environment.yaml
```

2. Replace the file `csv_logs.py` in folder `/nas-ctm01/homes/<slurm_username>/.conda/envs/<dataset_name>_pl_dl_pipeline_template/lib/python3.11/site-packages/pytorch_lightning/loggers/` with files of the same name located inside folder `/src/files_to_replace/`  

3. (Optional) To remove the created Conda virtual environment, open the Ubuntu remote terminal, then write and execute the following commands:

```commandline
conda env remove --name <dataset_name>_pl_dl_pipeline_template_venv
```

## 3. Input and output files
### 3.1. Input files
#### Config file
The config file (`config.yaml`) located at `/config/` contains the parameters used in the preprocessed data generation protocol. Below is its structure, with the corresponding explanations:

**Structure and explanations:**
```
data:
  raw:
    transforms:
      min_max_clip_hu_values: [ NUMBER1, NUMBER2 ] -> Specifies the range of Hounsfield Unit (HU) values to retain during preprocessing. Any values below -NUMBER1 or above NUMBER2 will be clipped
      min_max_value_range: [ NUMBER1, NUMBER2 ] -> Specifies the normalization range for pixel values. Pixel intensities are scaled linearly to fit within the range NUMBER1 to NUMBER2
      slice_thickness: NUMBER -> Sets the uniform thickness for CT slices in millimeters. This ensures consistency across scans from different machines
      pixel_spacing: NUMBER -> Standardizes the spacing between pixels in millimeters to ensure uniform resolution across images
      axis_transposition_protocol: [ NUMBER1, NUMBER2, NUMBER3 ] -> Indicates the order of axes to transpose during preprocessing. . For example, [2, 0, 1] changes the axis order from (depth, height, width) to (width, depth, height)
      flip_axis: NUMBER -> Specifies the axis along which the image should be flipped
  preprocessed:
    bounding_box:
      type:
        lung_nodule_image: BOOL -> Indicates that bounding boxes will be generated or not for lung nodule images
        lung_nodule_mask: BOOL -> Indicates that bounding boxes will be generated or not for lung nodule masks
      dimension: 2D | 2.5D | 3D -> Specifies the dimensionality of the bounding box. 2D for planar data, 3d for volumetric data, and 2.5D for planar data, but considering the central plane of each anatomical plane
      size: NUMBER -> Specifies the size of the bounding box in pixels or voxels. For 3D data, this represents a cubic box with dimensions NUMBERxNUMBERxNUMBER
```

**Example:**
```yaml
data:
  raw:
    transforms:
      min_max_clip_hu_values: [ -1000, 400 ]
      min_max_value_range: [ 0, 1 ]
      slice_thickness: 1
      pixel_spacing: 1
      axis_transposition_protocol: [ 2, 0, 1 ]
      flip_axis: 1
  preprocessed:
    bounding_box:
      type:
        lung_nodule_image: True
        lung_nodule_mask: True
      dimension: 3D
      size: 32
```

### Shell script file
The shell script file (`run_job.sh`) located at `/slurm_files/shell_script_files/` contains the script that will be used to submit jobs on Slurm

**Structure and explanations:**
```
#!/bin/bash

# Slurm job execution flags. REQUIRED TO SET: --job-name, --partition and --qos
#SBATCH --job-name=job -> [REQUIRED TO SET] Job name. Choose the shortest possible name that conveys what the work is about, using underscore ("_") as separator
#SBATCH --output=../output_files/JOBNAME=%x_ID=%j.out -> [DO NOT CHANGE] Job STDOUT file path
#SBATCH --error=../output_files/JOBNAME=%x_ID=%j.out -> [DO NOT CHANGE] Job STDERR file path
#SBATCH --partition=cpu_8cores -> [REQUIRED TO SET] Specify the Slurm partition which contains the servers under which your script will be run
#SBATCH --qos=cpu_8cores -> [REQUIRED TO SET] Specify the Quality of Service (QoS) for the job

# Preprocessed data generation protocol info: Fill in the details below
WHO=VALUE -> [REQUIRED TO SET] Who is generating the data (full name)?
CENTRE=VALUE -> [REQUIRED TO SET] Which centre is the creator associated with?
GROUP=VALUE -> [REQUIRED TO SET] Which group is the creator associated with?
WHAT=VALUE -> [REQUIRED TO SET] What is being generated?
WHY=VALUE -> [REQUIRED TO SET] Why is this data being generated?

# Run the script for dataset generation
python3 ../../src/scripts/generate_lung_nodule_dataset.py --who "$WHO" --centre "$CENTRE" --group "$GROUP" --what "$WHAT" --why "$WHY" -> A command-line execution where arguments (flags) are provided, and the values for these flags are derived from environment variables
```

**Example:**
```sh
#!/bin/bash

# Set the Slurm job execution flags
#SBATCH --job-name=job
#SBATCH --output=../output_files/JOBNAME=%x_ID=%j.out
#SBATCH --error=../output_files/JOBNAME=%x_ID=%j.out
#SBATCH --partition=cpu_8cores
#SBATCH --qos=cpu_8cores

# Preprocessed data generation protocol info: Fill in the details below
WHO="Quim Barreiros"
CENTRE="CTM"
GROUP="VCMI"
WHAT="2D Lung nodule bounding boxes (images and nodule masks)"
WHY="To be used in the paper 'Deep learning model for lung cancer cure'"

# Run the script for dataset generation
python3 ../../src/scripts/generate_lung_nodule_dataset.py --who "$WHO" --centre "$CENTRE" --group "$GROUP" --what "$WHAT" --why "$WHY"
```
### 3.2. Output files

## 4. Usage
To generate a new preprocessed data protocol:
1. Set the protocol settings in the configuration file `config.yaml` located at `/config/`
2. Set the Slurm job execution flags (that are required to set) and the environment variables in the shell script file `run_job.sh` located at `/slurm_files/shell_script_files/`
3. Open the Ubuntu terminal in the same folder as the shell script file, and run the following commands:
```commandline
conda activate dgv_venv
sbatch run_job.sh
```

## 5. Features
 - Generation of lung nodule bounding boxes (images and masks) with different dimensions, sizes and transformations
 - Visualization of lung nodule bounding boxes

## 7. Authors
 - Eduardo de Matos Rodrigues

## 8. References
 - [Keep a changelog](https://keepachangelog.com/en/1.1.0/)
 - [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
 - [Semantic Versioning](https://semver.org/spec/v2.0.0.html)