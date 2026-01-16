#!/bin/bash -l
#SBATCH --job-name=david-scoring
#SBATCH --partition=gpu_short
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-80:1

ENV_NAME="beer"
PYTHON_VERSION="3.12"

# Initialize pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Check if pyenv environment exists
if pyenv versions --bare | grep -q "^${ENV_NAME}$"; then
    echo "Activating existing pyenv environment '${ENV_NAME}'"
    pyenv activate ${ENV_NAME}
else
    echo "Environment '${ENV_NAME}' not found. Creating new pyenv virtualenv..."
    
    # Check if Python 3.12 is installed via pyenv
    if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
        echo "Python ${PYTHON_VERSION} not found in pyenv. Installing..."
        pyenv install ${PYTHON_VERSION}
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install Python ${PYTHON_VERSION}"
        fi
    fi
    
    # Create pyenv virtualenv
    echo "Creating pyenv virtualenv '${ENV_NAME}' with Python ${PYTHON_VERSION}"
    pyenv virtualenv ${PYTHON_VERSION} ${ENV_NAME}
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create pyenv virtualenv"
    fi
    
    # Activate the new environment
    pyenv activate ${ENV_NAME}
    
    # Install dependencies
    echo "Installing dependencies from requirements.txt..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
    fi
    
    echo "Environment '${ENV_NAME}' created and dependencies installed successfully"
fi

# Run the training script
cd ../src/mbeer
python3 scoring.py >& scoring.out.log