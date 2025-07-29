#!/bin/bash

# VERL GSM8K Data Preparation Script
# Downloads dataset and model for PPO training

set -e

# Environment should be sourced before running this script:
# source /home/yeopjin/orcd/pool/init_NTL.sh

# Add VERL to Python path
export PYTHONPATH=/home/yeopjin/orcd/pool/workspace/RL_NTL:$PYTHONPATH

# Configuration
DATA_DIR="/home/yeopjin/orcd/pool/workspace/RL_NTL/data"
export BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

echo "=== VERL GSM8K Data Preparation ==="
echo "Dataset: GSM8K"
echo "Model: ${BASE_MODEL} (will use HuggingFace cache)"
echo "Data directory: ${DATA_DIR}"
echo ""

# Prepare Dataset
echo "Preparing GSM8K dataset..."
mkdir -p $DATA_DIR

# Try to use the packaged version first, fallback to local file
if python3 -c "import verl.examples.data_preprocess.gsm8k" 2>/dev/null; then
    echo "Using packaged verl data preprocessing..."
    python3 -m verl.examples.data_preprocess.gsm8k --local_dir $DATA_DIR
else
    echo "Using local data preprocessing file..."
    GSM8K_SCRIPT="/home/yeopjin/orcd/pool/workspace/RL_NTL/verl/examples/data_preprocess/gsm8k.py"
    if [ -f "$GSM8K_SCRIPT" ]; then
        cd /home/yeopjin/orcd/pool/workspace/RL_NTL/verl/examples/data_preprocess
        python3 gsm8k.py --local_dir $DATA_DIR
    else
        echo "ERROR: GSM8K preprocessing script not found!"
        echo "Data may already be prepared in $DATA_DIR"
        exit 1
    fi
fi

echo ""
echo "Data preparation complete!"
echo "Dataset saved to: ${DATA_DIR}"
echo "Model will be automatically downloaded from HuggingFace cache during training"
echo ""
echo "Next step: Run ./train_gsm8k_ppo.sh to start training"