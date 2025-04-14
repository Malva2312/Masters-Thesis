# Masters-Thesis
Information Fusion-Based Model for Lung Nodule Characterization
# Documentation

## How to Run the Project

1. **Extract Data**:
    - Use the `set_up` script or command to copy the necessary data into the appropriate directories.

2. **The Conda Environment**:
1. Open the terminal at root, then write and execute the following command to create the `lung_char_fusion_env` environment at `<PATH TO CONDA PARENT DIRECTORY>/conda/envs/` using the `./src/conda_env/conda_env.yml` file:

```commandline
conda env create -f ./src/conda_env/conda_env.yml
```

2. Write and execute the following command in the terminal to activate the Conda environment:

```commandline
conda activate lung_char_fusion_env
```

[OPTIONAL] To remove the created Conda virtual environment, open the terminal, then write and execute the following commands:

```commandline
conda deactivate
conda env remove --name lung_char_fusion_env
```

3. **Navigate to the Source Directory**:
    - Change the working directory to the `src` folder where the main project code resides.
```commandline
cd ./src
```

4. **Launch TensorBoard**:
    - If TensorBoard is not installed, you can install it using pip. Run the following command in the terminal:
        ```commandline
        pip install tensorboard
        ```
    - To visualize and analyze the training logs, use TensorBoard. Run the following command in the terminal:
    ```commandline
    tensorboard --logdir=lightning_logs/
    ```
    - Open a web browser and navigate to the URL provided by TensorBoard (usually `http://localhost:6006`) to view the logs.

5. **Run Experiments**:
    - Execute the main experiment script to start the desired experiments.
    - Use the following command as an example:
    ```commandline
    python main.py --config_file <path_to_experiment_config_file>
    ```
    - Replace `<path_to_experiment_config_file>` with the path to the YAML configuration file for the experiment (e.g., `./config/experiments/exp1.yml`).
    - For detailed information on the configuration file structure, refer to the `experiments` folder or the documentation.
