#!/bin/bash

# Define environment name
ENV_NAME="smote_env"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install it first."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment: $ENV_NAME..."
python3.10 -m venv $ENV_NAME

# Activate the virtual environment
echo "Activating the environment..."
source $ENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install essential packages
echo "Installing core ML and data processing libraries..."
pip install numpy pandas scikit-learn matplotlib seaborn jupyter ipykernel

# Install imbalanced data handling libraries
echo "Installing imbalanced data processing libraries..."
pip install imbalanced-learn smote-variants

# Install additional useful libraries
echo "Installing additional useful libraries..."
pip install tqdm joblib scipy networkx

# Deactivate environment
echo "Setup complete. To activate the environment, run:"
echo "source $ENV_NAME/bin/activate"

# Verify installation
echo "Verifying installations..."
python -c "import imblearn; import smote_variants; print('All packages installed successfully!')"

# Deactivate environment
deactivate