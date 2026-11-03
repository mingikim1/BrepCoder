#!/bin/bash

# CoCa Training Script
# Usage: bash train.sh [options]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COCA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$COCA_DIR/.." && pwd)"

# Convert relative paths to absolute paths
CODE_DIR="${PROJECT_ROOT}/Dataset/code"
BREP_DIR="${PROJECT_ROOT}/Dataset/graph"
TOKENIZER_PATH="${PROJECT_ROOT}/Checkpoint/Qwen2.5-1.5B"
SPLIT_JSON="${PROJECT_ROOT}/Dataset/train_val_test_split.json"
VOCAB_MAP_PATH="${COCA_DIR}/cad_vocab_map.json"
OUTPUT_DIR="${COCA_DIR}/output"
DS_CONFIG="${COCA_DIR}/ds_config/zero1.json"

# Gradient checkpointing (saves memory, slightly slower)
# Set to "true" to enable, "false" to disable
USE_GRADIENT_CHECKPOINTING="false"

# Build gradient checkpointing argument
GRAD_CKPT_ARG=""
if [ "$USE_GRADIENT_CHECKPOINTING" = "true" ]; then
    GRAD_CKPT_ARG="--use_gradient_checkpointing"
    echo "✅ Gradient checkpointing: enabled"
else
    echo "❌ Gradient checkpointing: disabled"
fi

# Change to CoCa directory to run train.py
cd "$COCA_DIR"

# Find available port (using Python for cross-platform compatibility)
MASTER_PORT=29500
while ! python3 -c "import socket; s = socket.socket(); s.bind(('', ${MASTER_PORT})); s.close()" 2>/dev/null; do
    MASTER_PORT=$((MASTER_PORT + 1))
done
export MASTER_PORT
echo "Using master port: ${MASTER_PORT}"

deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT train.py \
    --deepspeed_config $DS_CONFIG \
    --code_dir $CODE_DIR \
    --brep_dir $BREP_DIR \
    --tokenizer_path $TOKENIZER_PATH \
    --split_json $SPLIT_JSON \
    --vocab_map_path $VOCAB_MAP_PATH \
    --output_dir $OUTPUT_DIR \
    $GRAD_CKPT_ARG \
    --num_epochs 150 \
    --num_layers 6 \
    --embed_dim 512 \
    --dropout 0.1 \
    --max_seq_len 2048 \
    --save_every 1

