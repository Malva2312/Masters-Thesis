@echo off
REM Check if conda is installed and update if necessary


REM Create the conda environment if it doesn't already exist
conda env list | find "lung_char_fusion_env" >nul 2>&1
if %errorlevel% == 0 (
    echo Conda environment already exists. Skipping creation.
) else (
    echo Creating conda environment
    conda env create -f .\src\conda_env\conda_env.yml
)

echo Conda environment setup completed.