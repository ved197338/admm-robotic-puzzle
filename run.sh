#!/bin/bash
cd file_directory

# Create a virtual environment if it doesn't already exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install all the required modules
echo "Installing required modules (scikit-learn, matplotlib, numpy)..."
pip install scikit-learn matplotlib numpy

# Run the python script
echo -e "\nRunning main.py..."
python main.py
