#!/bin/bash

# ==============================================================================
# Recode Dataset Evaluation Script for Qwen2.5-Coder
# ==============================================================================
#
# Purpose:
# This script evaluates Qwen2.5-Coder models on Recode (perturbed HumanEval) 
# datasets to assess robustness to code perturbations. These datasets test 
# whether models can maintain performance when code is reformatted or modified
# in semantically equivalent ways.
#
# Special Requirements:
# Recode datasets REQUIRE greedy generation settings to ensure deterministic,
# reproducible results:
#   - batch_size=1: Required for greedy generation (avoids num_return_sequences error)
#   - temperature=0.0: Ensures deterministic output
#   - do_sample=False: Disables sampling for greedy decoding
#   - n_samples=1: Single completion per problem
#
# Datasets Evaluated:
#   - perturbed-humaneval-format-num_seeds_5: Format/whitespace perturbations
#   - perturbed-humaneval-func_name-num_seeds_5: Function name changes
#   - perturbed-humaneval-natgen-num_seeds_5: Natural language perturbations
#   - perturbed-humaneval-nlaugmenter-num_seeds_5: NL augmentation perturbations
#
# References:
#   - Recode paper: https://arxiv.org/abs/2212.10264
#   - BigCode evaluation harness: https://github.com/bigcode-project/bigcode-evaluation-harness
#
# Usage:
#   ./run_bigcode_recode_datasets.sh
#
# ==============================================================================

set -e

# Configuration
RECODE_DATASETS=(
    "perturbed-humaneval-format-num_seeds_5"
    "perturbed-humaneval-func_name-num_seeds_5"
    "perturbed-humaneval-natgen-num_seeds_5"
    "perturbed-humaneval-nlaugmenter-num_seeds_5"
)

METHODS=("unsteered" "caa")
DEVICE=0
BASE_PROJECT="qwen25_coder_recode"
LOG_DIR="evaluation_logs"

# Greedy generation parameters for Recode datasets
BATCH_SIZE=1        # Must be 1 for greedy generation
TEMPERATURE=0.0     # Deterministic output
DO_SAMPLE="False"   # Greedy decoding
N_SAMPLES=1         # Single completion per problem

print_status() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

print_separator() {
    echo "=============================================================="
}

# Create log directory
mkdir -p "$LOG_DIR"

# Track progress
TOTAL_RUNS=$((${#RECODE_DATASETS[@]} * ${#METHODS[@]}))
CURRENT_RUN=0
START_TIME=$(date +%s)

print_separator
print_status "Starting Recode dataset evaluation"
print_status "Total evaluations to run: $TOTAL_RUNS"
print_status "Using greedy generation settings:"
print_status "  - batch_size: $BATCH_SIZE"
print_status "  - temperature: $TEMPERATURE"
print_status "  - do_sample: $DO_SAMPLE"
print_status "  - n_samples: $N_SAMPLES"
print_separator

# Main evaluation loop
for dataset in "${RECODE_DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        LOG_FILE="${LOG_DIR}/${method}_${dataset}_$(date +%Y%m%d_%H%M%S).log"
        
        print_status "[$CURRENT_RUN/$TOTAL_RUNS] Running: $method on $dataset"
        print_status "Log file: $LOG_FILE"
        
        # Run evaluation with greedy generation parameters
        python evaluate_bigcode_wandb.py \
            --method "$method" \
            --dataset "$dataset" \
            --batch_size "$BATCH_SIZE" \
            --temperature "$TEMPERATURE" \
            --do_sample "$DO_SAMPLE" \
            --n_samples "$N_SAMPLES" \
            --device "$DEVICE" \
            --project "${BASE_PROJECT}" \
            2>&1 | tee "$LOG_FILE"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_status "✓ Successfully completed: $method on $dataset"
        else
            print_status "✗ Failed: $method on $dataset (check $LOG_FILE for details)"
            print_status "Continuing with next evaluation..."
            continue
        fi
        
        # Brief pause between evaluations
        sleep 2
    done
    
    print_separator
done

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_separator
print_status "Recode evaluation completed!"
print_status "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
print_status "Logs saved in: $LOG_DIR"
print_separator