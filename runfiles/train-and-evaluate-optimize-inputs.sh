#!/bin/bash

set -e  # exit if any command fails

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --name) NAME="$2"; shift ;;
    --channel) CHANNEL="$2"; shift ;;
    --N) N="$2"; shift ;;
    --batch) BATCH="$2"; shift ;;
    --model_size) MODEL_SIZE="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --steps_per_epoch) STEPS_PER_EPOCH="$2"; shift ;;
    --mc_length) MC_LENGTH="$2"; shift ;;
    --save_name) SAVE_NAME="$2"; shift ;;
    --verbose) VERBOSE="$2"; shift ;;
    --code_rate) CODE_RATE="$2"; shift ;;
    --list_num) LIST_NUM="$2"; shift ;;
    --load_path) LOAD_PATH="$2"; shift ;;
    --threshold) THRESHOLD="$2"; shift ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Define paths
NAME="${NAME:-debug}"
CHANNEL="${CHANNEL:-ising}"
N="${N:-1024}"
BATCH="${BATCH:-64}"
MODEL_SIZE="${MODEL_SIZE:-small}"
EPOCHS="${EPOCHS:-100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
MC_LENGTH="${MC_LENGTH:-1000}"
SAVE_NAME="${SAVE_NAME:-model}"
VERBOSE="${VERBOSE:-2}"
CODE_RATE="${CODE_RATE:-0.4}"
LIST_NUM="${LIST_NUM:-8}"
LOAD_PATH="${LOAD_PATH:-}"
THRESHOLD="${THRESHOLD:-}"

RUN_NAME="${NAME}-optimize-${MODEL_SIZE}-${CHANNEL}-batch-${BATCH}-N-${N}"
OPTIMIZER_ESTIMATION_CONFIG_PATH="configs/optimizer_config.json"
OPTIMIZER_IMPROVEMENT_CONFIG_PATH="configs/optimizer_improvement_config.json"
NPD_CONFIG_PATH="configs/npd_${MODEL_SIZE}_config.json"
SAVE_DIR_PATH="results/${RUN_NAME}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${SAVE_DIR_PATH}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/train.log"

# Training
echo "Starting training..."
python src/npd_optimize_inputs.py \
    --channel "$CHANNEL" \
    --batch "$BATCH" \
    --N "$N" \
    --load_path "$LOAD_PATH" \
    --epochs "$EPOCHS" \
    --steps_per_epoch "$STEPS_PER_EPOCH" \
    --mc_length "$MC_LENGTH" \
    --save_name "$SAVE_NAME" \
    --save_dir_path "$OUTPUT_DIR" \
    --npd_config_path "$NPD_CONFIG_PATH" \
    --optimizer_estimation_config_path "$OPTIMIZER_ESTIMATION_CONFIG_PATH" \
    --optimizer_improvement_config_path "$OPTIMIZER_IMPROVEMENT_CONFIG_PATH" \
    --verbose "$VERBOSE" 

# Evaluation
echo "Starting evaluation..."

# Log file
LOG_FILE="${OUTPUT_DIR}/evaluate.log"

# Evaluation
python src/npd_decode_hy.py \
    --channel "$CHANNEL" \
    --batch "$BATCH" \
    --N "$N" \
    --mc_length "$MC_LENGTH" \
    --threshold "$THRESHOLD" \
    --save_dir_path "$OUTPUT_DIR" \
    --code_rate "$CODE_RATE" \
    --list_num "$LIST_NUM" \
    --npd_config_path "$NPD_CONFIG_PATH" \
    --load_path "$OUTPUT_DIR/model/model.weights.h5" \
    --verbose "$VERBOSE" \
    2>&1 | tee "$LOG_FILE"

echo "Done."