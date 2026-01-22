#!/bin/bash
set -e

# Configuration
WORKSPACE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_DIR="${WORKSPACE_DIR}/.venv"
HF_CACHE_DIR="/mnt/shared/boileo/hf_cache"

echo "Using Workspace: ${WORKSPACE_DIR}"

# 1. Setup Virtual Environment
if [ ! -d "${ENV_DIR}" ]; then
    echo "Creating virtual environment at ${ENV_DIR}..."
    python3 -m venv "${ENV_DIR}"
else
    echo "Virtual environment exists."
fi

# 2. Activate Virtual Environment
source "${ENV_DIR}/bin/activate"

# 3. Install Dependencies
echo "Installing dependencies..."
pip install --upgrade pip
if [ -f "${WORKSPACE_DIR}/requirements.txt" ]; then
    pip install -r "${WORKSPACE_DIR}/requirements.txt"
else
    echo "Warning: requirements.txt not found!"
fi
# Install other needed packages not in requirements explicitly if needed, 
# or ensure requirements.txt is complete. 
# Based on usage in train_rlvr.py:
pip install flash-attn --no-build-isolation || echo "Flash attention install failed, continuing..."

# 4. Set Environment Variables
export HF_HOME="${HF_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ensure the directory exists
mkdir -p "${HF_HOME}"

echo "HF_HOME set to: ${HF_HOME}"

# 5. Run Training Script
echo "Starting training..."
python "${WORKSPACE_DIR}/train_grpo.py"
