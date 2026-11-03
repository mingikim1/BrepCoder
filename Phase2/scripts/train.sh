#!/bin/bash
# ============================================================================
# BrepCoder Phase 2 - Training: Reverse Engineering (B-rep graph -> CAD code)
#
#   B-rep graph -> B-rep encoder (Phase 1) -> Projector -> Qwen2.5-1.5B -> CAD code
#
# Usage:
#   bash scripts/train.sh [--num_gpus N] [--epochs N] [--batch_size N] [--output_dir DIR] [--master_port PORT]
# ============================================================================

set -e

export TOKENIZERS_PARALLELISM=false
# Make the pip-installed CUDA libraries visible to DGL.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cusparse/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH}"

# ============================================================================
# Paths
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE2_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$PHASE2_DIR/.." && pwd)"
TRAIN_SCRIPT="${PHASE2_DIR}/train.py"

# Data
CODE_DIR="${ROOT_DIR}/Dataset/code"
GRAPH_DIR="${ROOT_DIR}/Dataset/graph"
SPLIT_JSON="${ROOT_DIR}/Dataset/train_val_test_split.json"

# Model
LLM_MODEL_PATH="${ROOT_DIR}/Checkpoint/Qwen2.5-1.5B"
# Phase 1 B-rep encoder. If the file does not exist, the encoder is randomly initialized.
BREP_ENCODER_CHECKPOINT="${ROOT_DIR}/Phase1/output/best_model/brep_encoder.pth"

# Output
OUTPUT_DIR="${PHASE2_DIR}/output/reverse_engineering"

# ============================================================================
# Training configuration
# ============================================================================
EPOCHS=3
BATCH_SIZE=16                     # per GPU
GRADIENT_ACCUMULATION_STEPS=1
MAX_LENGTH=2048
NUM_WORKERS=8

# Model
CODE_HIDDEN_SIZE=1536             # Qwen2.5-1.5B hidden size
GRAPH_HIDDEN_DIM=128
GRAPH_NUM_LAYERS=3
GRAPH_NUM_MLP_LAYERS=2

# Optimizer
ENCODER_LR=2e-5
PROJECTOR_LR=1e-4
LLM_LR=1e-4
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
SCHEDULER_TYPE="cosine"
MAX_GRAD_NORM=1.0

# Freeze (set FREEZE_ENCODER=false to train the B-rep encoder as well)
FREEZE_ENCODER=true
FREEZE_PROJECTION=false
FREEZE_LLM=false

# Flags
BF16=true
DO_EVAL=true
SAVE_EVERY_EPOCH=true
USE_FLASH_ATTN=true
USE_GRADIENT_CHECKPOINTING=true

# MLflow (optional)
USE_MLFLOW=false
MLFLOW_TRACKING_URI=""
MLFLOW_USERNAME=""
MLFLOW_PASSWORD=""
MLFLOW_EXPERIMENT_NAME="BrepCoder"
MLFLOW_RUN_NAME="Reverse Engineering"

# Launcher
USE_DEEPSPEED=true
DS_CONFIG="${PHASE2_DIR}/ds_config/zero2.json"
NUM_GPUS=4
MASTER_PORT=29502

# ============================================================================
# Command line arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --master_port) MASTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Build arguments
# ============================================================================
mkdir -p "$OUTPUT_DIR"

ARGS=(
    --code_dir "$CODE_DIR"
    --graph_dir "$GRAPH_DIR"
    --split_json "$SPLIT_JSON"
    --max_length "$MAX_LENGTH"
    --llm_model_path "$LLM_MODEL_PATH"
    --brep_encoder_checkpoint "$BREP_ENCODER_CHECKPOINT"
    --code_hidden_size "$CODE_HIDDEN_SIZE"
    --graph_hidden_dim "$GRAPH_HIDDEN_DIM"
    --graph_num_layers "$GRAPH_NUM_LAYERS"
    --graph_num_mlp_layers "$GRAPH_NUM_MLP_LAYERS"
    --output_dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --num_workers "$NUM_WORKERS"
    --encoder_lr "$ENCODER_LR"
    --projector_lr "$PROJECTOR_LR"
    --llm_lr "$LLM_LR"
    --weight_decay "$WEIGHT_DECAY"
    --warmup_ratio "$WARMUP_RATIO"
    --scheduler_type "$SCHEDULER_TYPE"
    --max_grad_norm "$MAX_GRAD_NORM"
)
[ "$FREEZE_ENCODER" = true ] && ARGS+=(--freeze_encoder)
[ "$FREEZE_PROJECTION" = true ] && ARGS+=(--freeze_projection)
[ "$FREEZE_LLM" = true ] && ARGS+=(--freeze_llm)
[ "$BF16" = true ] && ARGS+=(--bf16)
[ "$DO_EVAL" = true ] && ARGS+=(--do_eval)
[ "$SAVE_EVERY_EPOCH" = true ] && ARGS+=(--save_every_epoch)
[ "$USE_FLASH_ATTN" = true ] && ARGS+=(--use_flash_attn)
[ "$USE_GRADIENT_CHECKPOINTING" = true ] && ARGS+=(--use_gradient_checkpointing)

if [ "$USE_MLFLOW" = true ]; then
    export MLFLOW_TRACKING_URI MLFLOW_USERNAME MLFLOW_PASSWORD
    ARGS+=(--use_mlflow --mlflow_experiment_name "$MLFLOW_EXPERIMENT_NAME" --mlflow_run_name "$MLFLOW_RUN_NAME")
    [ -n "$MLFLOW_TRACKING_URI" ] && ARGS+=(--mlflow_tracking_uri "$MLFLOW_TRACKING_URI")
fi

# ============================================================================
# Print configuration
# ============================================================================
echo "============================================================================"
echo "BrepCoder Phase 2 Training - Reverse Engineering"
echo "============================================================================"
echo "LLM: $LLM_MODEL_PATH"
echo "B-rep encoder: $BREP_ENCODER_CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS | Batch: $BATCH_SIZE x $NUM_GPUS GPUs (x$GRADIENT_ACCUMULATION_STEPS accumulation)"
echo "Freeze: encoder=$FREEZE_ENCODER projection=$FREEZE_PROJECTION llm=$FREEZE_LLM"
echo "LR: encoder=$ENCODER_LR projector=$PROJECTOR_LR llm=$LLM_LR"
echo "DeepSpeed: $USE_DEEPSPEED | FlashAttn: $USE_FLASH_ATTN | BF16: $BF16"
echo "============================================================================"

# ============================================================================
# Run training
# ============================================================================
find_free_port() {
    local port=$1
    while command -v lsof >/dev/null 2>&1 && lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; do
        port=$((port + 1))
    done
    echo "$port"
}

if [ "$USE_DEEPSPEED" = true ]; then
    MASTER_PORT=$(find_free_port "$MASTER_PORT")
    echo "Running with DeepSpeed (${NUM_GPUS} GPUs, port ${MASTER_PORT})..."
    deepspeed --num_gpus="$NUM_GPUS" --master_port="$MASTER_PORT" "$TRAIN_SCRIPT" "${ARGS[@]}" --deepspeed "$DS_CONFIG"
elif [ "$NUM_GPUS" -gt 1 ]; then
    MASTER_PORT=$(find_free_port "$MASTER_PORT")
    echo "Running with torchrun (${NUM_GPUS} GPUs, port ${MASTER_PORT})..."
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "$TRAIN_SCRIPT" "${ARGS[@]}"
else
    echo "Running on a single GPU..."
    python "$TRAIN_SCRIPT" "${ARGS[@]}"
fi

echo "============================================================================"
echo "Training completed! Output: $OUTPUT_DIR"
echo "============================================================================"
