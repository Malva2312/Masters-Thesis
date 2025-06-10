# Masters-Thesis
Information Fusion-Based Model for Lung Nodule Characterization
# Documentation

## How to Run the Project

1. **Set Up the Conda Environment**:
    - Open the terminal at the root directory and execute the following command to create the `lung_fusion_env` environment at `<PATH TO CONDA PARENT DIRECTORY>/conda/envs/` using the `conda_env.yaml` file:

    ```commandline
    conda env create -f ./conda_env/conda_env.yaml
    ```

    - Activate the Conda environment by running:

    ```commandline
    conda activate lung_fusion_env
    ```
    
    [OPTIONAL] To update the created Conda virtual environment, execute:
   ```commandline
   conda env update --name lung_fusion_env --file ./conda_env/conda_env.yaml --prune
   ```
   
    [OPTIONAL] To remove the created Conda virtual environment, execute:

    ```commandline
    conda deactivate
    conda env remove --name lung_fusion_env
    ```

3. **Replace the CSV Logger File**:
    - Navigate to the Conda environment's site-packages directory:
      `<template_venv_name>/lib/python3.11/site-packages/pytorch_lightning/loggers/`.
    - Replace the `csv_logs.py` file with the provided version located at `src/files_to_replace/csv_logs.py`.

4. **Prepare the Data**:
    - If running locally:
      1. Navigate to the `data` directory in your project.
      2. Unzip the `data.zip` file located in the `data` directory.
      3. Update the logic in the `set_paths` method in `src/modules/experiment_execution/config.py` to set the local data paths dynamically. Replace the `dataset_dir_path` assignment for the `LIDC-IDRI` and `LUNA25` datasets with the following:

      ```python
      if config.data.dataset_name == "LIDC-IDRI":
        dataset_dir_path = "./data/preprocessed/lidc_idri"
      elif config.data.dataset_name == "LUNA25":
        dataset_dir_path = "./data/preprocessed/luna25"
      ```

      4. Ensure your working directory in the terminal or IDE (e.g., Visual Studio Code) is set to the project root directory before running the application. You can change the directory in the terminal using:
      ```bash
      cd /path/to/project/root
      ```
    - If running remotely:
        - Sync the local project directory with the remote server using a tool like WinSCP (for example):
        - Ensure the data is available on the remote server.

5. **Run the Experiment Pipeline**:
    - For remote execution, use the provided Slurm shell script:
      ```commandline
      sbatch ./slurm_files/shell_script_files/run_experiment_pipeline_job.sh
      ```
    - For local execution, navigate to the `src` directory and run the experiment script:
      ```commandline
      cd ./src
      python scripts/run_experiment_pipeline.py
      ```
