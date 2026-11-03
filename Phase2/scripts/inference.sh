#!/bin/bash
# ============================================================================
# BrepCoder Phase 2 - Inference: Reverse Engineering (B-rep graph -> CAD code)
#
# Usage:
#   bash scripts/inference.sh                              # test split, 1 GPU
#   bash scripts/inference.sh --num_gpus 4                 # test split, sharded over 4 GPUs
#   bash scripts/inference.sh --input_graph path/to/00000007.bin
#
# Options: --model_path DIR, --split {train,val,test}, --max_samples N, --output_file FILE,
#          --max_new_tokens N, --temperature T, --device DEVICE
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
INFERENCE_SCRIPT="${PHASE2_DIR}/inference.py"

# Model
MODEL_PATH="${PHASE2_DIR}/output/reverse_engineering/best_model"
LLM_MODEL_PATH="${ROOT_DIR}/Checkpoint/Qwen2.5-1.5B"

# Data
CODE_DIR="${ROOT_DIR}/Dataset/code"
GRAPH_DIR="${ROOT_DIR}/Dataset/graph"
SPLIT_JSON="${ROOT_DIR}/Dataset/train_val_test_split.json"

# ============================================================================
# Model configuration (must match training)
# ============================================================================
CODE_HIDDEN_SIZE=1536
GRAPH_HIDDEN_DIM=128
GRAPH_NUM_LAYERS=3
GRAPH_NUM_MLP_LAYERS=2

# ============================================================================
# Inference configuration
# ============================================================================
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0                   # 0 = greedy decoding
TOP_P=0.95
REPETITION_PENALTY=1.0
NUM_BEAMS=1
DO_SAMPLE="auto"

SPLIT="test"
MAX_SAMPLES=""
OUTPUT_FILE=""
INPUT_GRAPH=""

BF16=true
USE_FLASH_ATTN=true
DEVICE=""
NUM_GPUS=1

# ============================================================================
# Command line arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --input_graph) INPUT_GRAPH="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
        --output_file) OUTPUT_FILE="$2"; shift 2 ;;
        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Build arguments
# ============================================================================
ARGS=(
    --model_path "$MODEL_PATH"
    --llm_model_path "$LLM_MODEL_PATH"
    --code_hidden_size "$CODE_HIDDEN_SIZE"
    --graph_hidden_dim "$GRAPH_HIDDEN_DIM"
    --graph_num_layers "$GRAPH_NUM_LAYERS"
    --graph_num_mlp_layers "$GRAPH_NUM_MLP_LAYERS"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --repetition_penalty "$REPETITION_PENALTY"
    --num_beams "$NUM_BEAMS"
    --do_sample "$DO_SAMPLE"
)
[ "$BF16" = true ] && ARGS+=(--bf16)
[ "$USE_FLASH_ATTN" = true ] && ARGS+=(--use_flash_attn)

DATA_ARGS=(--code_dir "$CODE_DIR" --graph_dir "$GRAPH_DIR" --split_json "$SPLIT_JSON" --split "$SPLIT")
[ -n "$MAX_SAMPLES" ] && DATA_ARGS+=(--max_samples "$MAX_SAMPLES")

[ -n "$OUTPUT_FILE" ] || OUTPUT_FILE="${MODEL_PATH}/inference_results_${SPLIT}.json"

echo "=================================================================================="
echo "BrepCoder Phase 2 Inference - Reverse Engineering"
echo "=================================================================================="
echo "Model: $MODEL_PATH"

# ============================================================================
# Single graph
# ============================================================================
if [ -n "$INPUT_GRAPH" ]; then
    echo "Input graph: $INPUT_GRAPH"
    [ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
    python "$INFERENCE_SCRIPT" "${ARGS[@]}" --input_graph "$INPUT_GRAPH"
    exit 0
fi

echo "Split: $SPLIT | GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_FILE"
echo "=================================================================================="

# ============================================================================
# Whole split on one device
# ============================================================================
if [ "$NUM_GPUS" -le 1 ]; then
    [ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")
    python "$INFERENCE_SCRIPT" "${ARGS[@]}" "${DATA_ARGS[@]}" --output_file "$OUTPUT_FILE"
    exit 0
fi

# ============================================================================
# Whole split sharded over multiple GPUs (one process per GPU), then merged
# ============================================================================
OUTPUT_BASE="${OUTPUT_FILE%.json}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$GPU_ID python "$INFERENCE_SCRIPT" "${ARGS[@]}" "${DATA_ARGS[@]}" \
        --device cuda:0 --num_shards "$NUM_GPUS" --shard_id "$GPU_ID" \
        --output_file "${OUTPUT_BASE}_shard${GPU_ID}.json" \
        > "${OUTPUT_BASE}_shard${GPU_ID}.log" 2>&1 &
    PIDS+=($!)
    echo "  GPU $GPU_ID: shard $GPU_ID/$NUM_GPUS (log: ${OUTPUT_BASE}_shard${GPU_ID}.log)"
done

FAILED=0
for GPU_ID in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$GPU_ID]}"; then
        echo "GPU $GPU_ID failed. See ${OUTPUT_BASE}_shard${GPU_ID}.log"
        FAILED=$((FAILED + 1))
    fi
done
[ $FAILED -gt 0 ] && echo "Error: $FAILED shard(s) failed." && exit 1

python - "$OUTPUT_BASE" "$NUM_GPUS" << 'EOF'
import json, os, sys

base, num_shards = sys.argv[1], int(sys.argv[2])
results = []
for i in range(num_shards):
    with open(f"{base}_shard{i}.json", encoding="utf-8") as f:
        results.extend(json.load(f)["results"])
results.sort(key=lambda r: r["id"])

with open(f"{base}.json", "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, indent=2, ensure_ascii=False)
for i in range(num_shards):
    os.remove(f"{base}_shard{i}.json")
    os.remove(f"{base}_shard{i}.log")
print(f"Merged {len(results)} results -> {base}.json")
EOF
