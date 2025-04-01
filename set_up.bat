@echo off
REM Batch script to unzip a file to a specific location

REM Set the path to the zip file
set ZIP_FILE_PATH=.\src\data_loading.zip

REM Set the destination directory
set DEST_DIR=.\src

REM Check if the destination directory exists and remove it if it does
if exist %DEST_DIR%\data_loading (
    echo Removing existing directory %DEST_DIR%\data_loading
    rmdir /s /q %DEST_DIR%\data_loading
)

REM Unzip the file
tar -xf %ZIP_FILE_PATH% -C %DEST_DIR%

echo Unzipping completed.

REM Check if the destination directory already exists and remove it if it does
if exist .\data (
    echo Removing existing directory .\data
    rmdir /s /q .\data
)

REM Copy data from .\src\data_loading\data to .\
xcopy .\src\data_loading\data .\data\ /s /e /i


REM Create the conda environment if it doesn't already exist
conda env list | find "lung_char_fusion_env" >nul 2>&1
if %errorlevel% == 0 (
    echo Conda environment already exists. Skipping creation.
) else (
    echo Creating conda environment
    conda env create -f .\src\conda_env\conda_env.yml
)

echo Conda environment setup completed.