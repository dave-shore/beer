#!/bin/bash -l
ENV_NAME="beer"
PYTHON_VERSION="3.12.1"
DEVICES="cuda:0"

# Initialize pyenv
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
    echo "Installing dependencies from poetry.lock..."
    python3 -m pip install --upgrade pip
    python3 -m poetry install

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
    fi

    echo "Environment '${ENV_NAME}' created and dependencies installed successfully"
fi

# Move to the directory hosting the scoring script.
cd ../src/mbeer

models=(
    "numind/NuNER-v2.0"
    "KomeijiForce/roberta-large-metaie"
)

for model in "${models[@]}"; do
    model_name=$(echo "$model" | tr '/' '-')
    echo "Scoring $model on GPU(s)=${DEVICES}"
    python3 scoring.py --mbrd --test --model-name "$model" --device "$DEVICES" --dataset-name "xiaobendanyn/tacred"\
        >& "scoring-${model_name}.out.log"
done
