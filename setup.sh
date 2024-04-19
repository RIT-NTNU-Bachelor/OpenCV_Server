#!/bin/bash

# Check Python version
REQUIRED_PYTHON="3.10"
PYTHON_VERSION=$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)

if [ "$PYTHON_VERSION" == "$REQUIRED_PYTHON" ]; then
    echo "[INFO] Correct Python version ($PYTHON_VERSION).X is selected."
else
    echo "[ERROR] This script requires Python $REQUIRED_PYTHON.x. Current version is Python $PYTHON_VERSION."
    exit 1
fi

# Installing dependencies
echo "[INFO] Installing dependencies from requirements.txt.."
pip install -r requirements.txt
echo "Setup complete!"
