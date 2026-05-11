#!/bin/bash -l
#SBATCH --job-name=david-scoring
#SBATCH --partition=gpu_long
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:1
#SBATCH --array=0-2%2
#SBATCH --output=scoring-%A_%a.slurm.log
#SBATCH --error=scoring-%A_%a.slurm.log

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

# Move to the directory hosting the scoring script.
cd ../src/mbeer

models=(
    "numind/NuNER-v2.0"
    "SIRIS-Lab/citation-parser-ENTITY"
    "rv2307/electra-small-ner"
    "peoplek/Llama-3.2-1B-NER"
)

# Resolve which model this array task is responsible for. Outside SLURM we
# default to task 0 so the script can still be exercised manually.
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$task_id" -ge "${#models[@]}" ]; then
    echo "SLURM_ARRAY_TASK_ID=$task_id is out of range (have ${#models[@]} models)." >&2
    exit 1
fi
model="${models[$task_id]}"
model_name=$(echo "$model" | tr '/' '-')

# SLURM allocates exactly one GPU to this task and exposes it via
# CUDA_VISIBLE_DEVICES, so we don't have to manage device assignment ourselves.
echo "Array task $task_id: scoring $model on GPU(s)=${CUDA_VISIBLE_DEVICES:-<unset>}"
python3 scoring.py --mbrd --model-name "$model" \
    >& "scoring-${model_name}.out.log"
