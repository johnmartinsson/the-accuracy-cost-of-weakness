#!/bin/bash

# Define the virtual environment directory and requirements file
VENV_DIR="./weak-labeling"
REQUIREMENTS_FILE="./requirements.txt"

# Create the virtual environment
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the dependencies from requirements.txt
pip install -r $REQUIREMENTS_FILE

# Add the virtual environment as a Jupyter kernel: weak-labeling
python -m ipykernel install --user --name weak-labeling --display-name "Python (weak-labeling)"

# Deactivate the environment
deactivate

echo "Virtual environment setup and dependencies installed successfully."
echo "To deactivate the virtual environment, run: deactivate"
echo "To activate the virtual environment, run: source $VENV_DIR/bin/activate"
