#!/bin/bash

# Quick Test Script for Math Evaluation Pipeline
# Tests the evaluation setup with limited samples before running full evaluation

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Math Evaluation Quick Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Test configuration
SEED=42
MAX_BATCH_SIZE=8
DEVICE=0
CACHE_PATH="/workspace/wisent-guard/lm_eval_cache_test"
LIMIT=5  # Only evaluate 5 samples for testing

# Create cache directory
mkdir -p "$CACHE_PATH"

echo -e "${GREEN}Test 1: Unsteered model on GSM8K (0-shot, 5 samples)${NC}"
python examples/evaluation/evaluate_lm_eval_wandb.py \
    --method unsteered \
    --tasks gsm8k \
    --num_fewshot 0 \
    --seed "$SEED" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --device "$DEVICE" \
    --use_cache "$CACHE_PATH" \
    --limit "$LIMIT" \
    --wandb_project "test_math_eval" \
    --output_dir "test_evaluation_results"

echo ""
echo -e "${GREEN}Test 2: CAA model on MathQA (5-shot, 5 samples)${NC}"
python examples/evaluation/evaluate_lm_eval_wandb.py \
    --method caa \
    --tasks mathqa \
    --num_fewshot 5 \
    --seed "$SEED" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --device "$DEVICE" \
    --use_cache "$CACHE_PATH" \
    --limit "$LIMIT" \
    --wandb_project "test_math_eval" \
    --output_dir "test_evaluation_results"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Quick tests completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "If both tests passed, you can run the full evaluation with:"
echo -e "${YELLOW}./examples/evaluation/run_math_evaluations.sh${NC}"
echo ""
echo "Note: The full evaluation will take several hours to complete."