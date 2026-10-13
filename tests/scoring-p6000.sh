#!/bin/bash -l

ENV_NAME="beer"
PYTHON_VERSION="3.12.1"
DEVICES="cuda:0"

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
    "tanaos/tanaos-NER-v1"
    "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
    "Babelscape/wikineural-multilingual-ner"
)

datasets=(
    "xiaobendanyn/tacred"
    "DFKI-SLT/nyt-multi"
)

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        model_name=$(echo "$model" | tr '/' '-')
        dataset_name=$(echo "$dataset" | tr '/' '-')
        echo "Scoring $model on dataset $dataset"
        python3 scoring.py --mbrd --baselines --model-name "$model" --device "$DEVICES" --dataset-name "$dataset" \
                >& "scoring-${model_name}-${dataset_name}.out.log"
    done
done
