#!/bin/bash -l
#SBATCH --job-name=david-scoring
#SBATCH --partition=gpu_long
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100-80:4

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
models=("tanaos/tanaos-NER-v1" "dslim/bert-base-NER" "Babelscape/wikineural-multilingual-ner" "numind/NuNER-v2.0" "Mozilla/distilbert-uncased-NER-LoRA" "rv2307/electra-small-ner")
# Determine the GPUs we can dispatch to. SLURM exports CUDA_VISIBLE_DEVICES
# with the allocated devices; outside SLURM we fall back to nvidia-smi.
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t gpu_ids < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
num_gpus=${#gpu_ids[@]}
if [ "$num_gpus" -eq 0 ]; then
    echo "No GPUs detected; aborting." >&2
    exit 1
fi
echo "Dispatching across ${num_gpus} GPU slot(s): ${gpu_ids[*]}"

# Run at most one job per GPU concurrently and queue further models as
# slots free up. pid_to_gpu maps a running child PID to the GPU it owns.
declare -A pid_to_gpu
for model in "${models[@]}"; do
    # Block until a GPU is free.
    while [ "${#pid_to_gpu[@]}" -ge "$num_gpus" ]; do
        wait -n
        for pid in "${!pid_to_gpu[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                unset "pid_to_gpu[$pid]"
                break
            fi
        done
    done

    # Pick the first GPU that isn't currently in use.
    busy=" ${pid_to_gpu[*]} "
    chosen_gpu=""
    for g in "${gpu_ids[@]}"; do
        if [[ "$busy" != *" $g "* ]]; then
            chosen_gpu="$g"
            break
        fi
    done

    model_name=$(echo "$model" | tr '/' '-')
    echo "Launching $model on GPU $chosen_gpu"
    CUDA_VISIBLE_DEVICES="$chosen_gpu" \
        python3 scoring.py --mbrd --model-name "$model" \
        >& "scoring-${model_name}.out.log" &
    pid_to_gpu[$!]="$chosen_gpu"
done

# Drain any remaining jobs.
wait