#!/bin/bash

set -e

# Configuration
DATASETS=(
    "instruct-humaneval"
    "instruct-humaneval-nocontext"
    "multiple-js"
)

METHODS=("unsteered" "caa")
DEVICE=0
BASE_PROJECT="qwen25_coder_new_datasets"
LOG_DIR="evaluation_logs"

print_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create log directory
mkdir -p "$LOG_DIR"

# Track progress
TOTAL_RUNS=$((${#DATASETS[@]} * ${#METHODS[@]}))
CURRENT_RUN=0
START_TIME=$(date +%s)

print_status "Starting evaluation of $TOTAL_RUNS runs on full datasets"

# Main evaluation loop
for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        LOG_FILE="${LOG_DIR}/${method}_${dataset}_$(date +%Y%m%d_%H%M%S).log"
        
        print_status "[$CURRENT_RUN/$TOTAL_RUNS] Running: $method $dataset (full dataset)"
        
        python evaluate_bigcode_wandb.py \
            --method "$method" \
            --dataset "$dataset" \
            --device "$DEVICE" \
            --project "${BASE_PROJECT}_${dataset}" \
            2>&1 | tee "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_status "Completed: $method $dataset"
        else
            print_status "Failed: $method $dataset (check $LOG_FILE)"
            continue
        fi
        
        sleep 2
    done
done

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

print_status "All evaluations completed in ${HOURS}h ${MINUTES}m"