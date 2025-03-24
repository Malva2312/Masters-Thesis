@echo off
REM Batch script to unzip a file to a specific location

REM Set the path to the zip file
set ZIP_FILE_PATH=.\src\data_loading.zip

REM Set the destination directory
set DEST_DIR=.\src

REM Unzip the file
tar -xf %ZIP_FILE_PATH% -C %DEST_DIR%

echo Unzipping completed.
pause