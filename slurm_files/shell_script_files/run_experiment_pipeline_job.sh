#!/bin/bash

# Slurm job execution flags. REQUIRED TO SET: --job-name, --partition and --qos.
#SBATCH --job-name=test_net
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb_ext
#SBATCH --output=../output_files/JOBNAME=%x_ID=%j.out
#SBATCH --error=../output_files/JOBNAME=%x_ID=%j.out

# Command-line to run the script for dataset generation.
srun python3 ../../src/scripts/run_experiment_pipeline.py 

