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

echo Data copied successfully..
