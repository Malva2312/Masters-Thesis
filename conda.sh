#!/bin/bash
# Check if conda is installed and update if necessary

# Check if the conda environment exists
if conda env list | grep -q "lung_char_fusion_env"; then
    echo "Conda environment already exists. Skipping creation."
else
    echo "Creating conda environment"
    conda env create -f ./src/conda_env/conda_env.yml
fi

echo "Conda environment setup completed."
