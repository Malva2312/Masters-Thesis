#!/bin/bash

# Slurm job execution flags. REQUIRED TO SET: --job-name, --partition and --qos.
#SBATCH --job-name=test
#SBATCH --partition=debug_8gb
#SBATCH --qos=debug_8gb
#SBATCH --output=../output_files/JOBNAME=%x_ID=%j.out
#SBATCH --error=../output_files/JOBNAME=%x_ID=%j.out

# Command-line to run the script for dataset generation.
srun python3 ../../src/scripts/run_experiment_pipeline.py 

