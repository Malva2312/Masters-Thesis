#!/bin/bash
# Shell script to unzip a file to a specific location

# Set the path to the zip file
ZIP_FILE_PATH=./src/data_loading.zip

# Set the destination directory
DEST_DIR=./src

# Check if the destination directory exists and remove it if it does
if [ -d "$DEST_DIR/data_loading" ]; then
    echo "Removing existing directory $DEST_DIR/data_loading"
    rm -rf "$DEST_DIR/data_loading"
fi

# Unzip the file
tar -xf "$ZIP_FILE_PATH" -C "$DEST_DIR"

echo "Unzipping completed."

# Check if the data directory exists and remove it if it does
if [ -d ./data ]; then
    echo "Removing existing directory ./data"
    rm -rf ./data
fi

# Copy data from ./src/data_loading/data to ./
cp -r ./src/data_loading/data ./data

# Call conda.sh to set up the environment
bash ./conda.sh

echo "Setup completed."
