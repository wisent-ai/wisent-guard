#!/bin/bash

# Comprehensive Math Evaluation Script for Qwen2.5-Math Models
# Evaluates both steered (CAA) and unsteered models on multiple math tasks
# with consistent settings for reproducibility and comparison

set -e  # Exit on error

# ==================== Configuration ====================

# Math tasks to evaluate
TASKS=(
    "asdiv"
    "gsm8k"
    "hendrycks_math"
    "mathqa"
    "minerva_math"
)

# Methods to test
# METHODS=("unsteered" "caa")
METHODS=("caa")

# Few-shot settings
SHOTS=(0 5)

# Consistent evaluation parameters
SEED=42
DEVICE=0
BASE_PROJECT="qwen25_math_comprehensive_high_steering"

# Output directories
OUTPUT_BASE_DIR="evaluation_results"
LOG_DIR="evaluation_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==================== Functions ====================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# ==================== Main Execution ====================

print_header "Qwen2.5-Math Comprehensive Evaluation"
echo "Tasks: ${TASKS[@]}"
echo "Methods: ${METHODS[@]}"
echo "Few-shot settings: ${SHOTS[@]}"
echo "Seed: $SEED"
echo "Max Batch Size: $MAX_BATCH_SIZE"
echo "Device: cuda:$DEVICE"
echo ""

# Create directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOG_DIR"

# Track overall progress
TOTAL_RUNS=$((${#TASKS[@]} * ${#METHODS[@]} * ${#SHOTS[@]}))
CURRENT_RUN=0

# Start time tracking
START_TIME=$(date +%s)

# Main evaluation loop
for task in "${TASKS[@]}"; do
    print_header "Task: $task"
    
    for method in "${METHODS[@]}"; do
        for shots in "${SHOTS[@]}"; do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            
            # Construct WandB project name
            WANDB_PROJECT="${BASE_PROJECT}_${task}"
            
            # Construct run name for logging
            RUN_NAME="${method}_${task}_${shots}shot"
            LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
            
            print_status "[$CURRENT_RUN/$TOTAL_RUNS] Running: $RUN_NAME"
            echo "  Method: $method"
            echo "  Task: $task"
            echo "  Few-shot: $shots"
            echo "  Log: $LOG_FILE"
            
            # Run the evaluation
            python examples/evaluation/evaluate_lm_eval_wandb.py \
                --method "$method" \
                --tasks "$task" \
                --num_fewshot "$shots" \
                --seed "$SEED" \
                --max_batch_size 16 \
                --device "$DEVICE" \
                --wandb_project "$WANDB_PROJECT" \
                --output_dir "$OUTPUT_BASE_DIR" \
                2>&1 | tee "$LOG_FILE"
            
            # Check if the evaluation was successful
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                print_status "✅ Completed: $RUN_NAME"
            else
                print_error "Failed: $RUN_NAME (check $LOG_FILE for details)"
                # Continue with next evaluation instead of exiting
                continue
            fi
            
            # Add a small delay between runs to avoid overwhelming the system
            sleep 2
        done
    done
    
    print_status "Completed all evaluations for task: $task"
    echo ""
done

# ==================== Summary ====================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

print_header "Evaluation Complete!"
echo "Total runs: $TOTAL_RUNS"
echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved to: $OUTPUT_BASE_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
print_status "Check WandB for detailed results and visualizations"

# Optional: Generate summary report
if command -v jq &> /dev/null; then
    print_status "Generating summary report..."
    
    SUMMARY_FILE="${OUTPUT_BASE_DIR}/evaluation_summary_$(date +%Y%m%d_%H%M%S).txt"
    echo "Evaluation Summary" > "$SUMMARY_FILE"
    echo "==================" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    for task in "${TASKS[@]}"; do
        echo "Task: $task" >> "$SUMMARY_FILE"
        echo "----------" >> "$SUMMARY_FILE"
        
        # Find and parse result files for this task
        for method in "${METHODS[@]}"; do
            for shots in "${SHOTS[@]}"; do
                echo "  $method (${shots}-shot):" >> "$SUMMARY_FILE"
                # Look for result files (implementation depends on output format)
                RESULT_PATTERN="${OUTPUT_BASE_DIR}/${method}_${task}_*"
                if ls $RESULT_PATTERN 1> /dev/null 2>&1; then
                    echo "    Results available" >> "$SUMMARY_FILE"
                else
                    echo "    No results found" >> "$SUMMARY_FILE"
                fi
            done
        done
        echo "" >> "$SUMMARY_FILE"
    done
    
    print_status "Summary saved to: $SUMMARY_FILE"
else
    print_warning "jq not installed, skipping summary generation"
fi

print_header "All evaluations completed successfully!"