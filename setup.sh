#!/bin/bash

# Telling the user that the startup process has started
echo "[INFO] Setting up the python environment for the OpenCV Server..."

# Create the virtual enviroment, if the envirement has not been created before.
if [ ! -d "./venv" ]; then
    # Telling the user that we are creating the virtual environment 
    echo "[INFO] Creating virtual environment..."
    mkdir venv
    python -m venv ./venv
fi

# Activating the virtual environment with the generated script
echo "[INFO] Activating virtual environment..."
source ./venv/bin/activate

# Installing dependencies
echo "[INFO] Installing dependencies from requirements.txt.."
pip install -r requirements.txt
echo "Setup complete!"
