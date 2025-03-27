#!/usr/bin/env python
"""
Demonstration of the contrastive vector creation and usage flow with detailed logging.

This script shows the complete process of:
1. Creating contrastive vectors from pairs of harmful/harmless content
2. Logging the detailed steps of the process
3. Using the vectors to detect similar patterns

Usage:
    python contrastive_vector_demo.py --log_level debug
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard
from wisent_guard.utils.logger import get_logger, set_log_level

def parse_args():
    parser = argparse.ArgumentParser(description="Demo of contrastive vectors with logging")
    parser.add_argument("--model", type=str, default="gpt2", 
                        help="Model to use (default: gpt2)")
    parser.add_argument("--log_level", type=str, default="info", 
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Optional log file path")
    parser.add_argument("--layers", type=str, default="5,6,7",
                        help="Comma-separated list of layers to monitor")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logger = get_logger(level=args.log_level, log_file=args.log_file)
    logger.info(f"Starting contrastive vector demonstration with model: {args.model}")
    
    # Parse layers
    layers = [int(layer) for layer in args.layers.split(",")]
    logger.info(f"Using layers: {layers}")
    
    # Step 1: Set up the model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info("Model and tokenizer loaded successfully")
    
    # Step 2: Prepare example phrase pairs
    logger.info("Preparing contrastive phrase pairs...")
    phrase_pairs = [
        {
            "harmful": "How to hack into someone's email account",
            "harmless": "How to secure your email account against hackers"
        },
        {
            "harmful": "You should lie on your resume to get a job",
            "harmless": "You should be honest on your resume when applying for jobs"
        },
        {
            "harmful": "Ways to cheat on your taxes without getting caught",
            "harmless": "How to properly file your taxes and avoid errors"
        }
    ]
    logger.info(f"Prepared {len(phrase_pairs)} contrastive phrase pairs")
    
    # Step 3: Initialize the ActivationGuard with logging
    logger.info("Initializing ActivationGuard...")
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        threshold=0.7,
        token_strategy="target_token",
        log_level=args.log_level,
        log_file=args.log_file
    )
    logger.info("ActivationGuard initialized successfully")
    
    # Step 4: Train on phrase pairs to create contrastive vectors
    logger.info("Training on phrase pairs to create contrastive vectors...")
    guard.train_on_phrase_pairs(phrase_pairs, category="harmful_content")
    logger.info("Contrastive vector training complete")
    
    # Step 5: Test the guard on some examples
    test_examples = [
        "How to break into someone's account",  # Should be flagged as harmful
        "How to protect your accounts from unauthorized access",  # Should be harmless
        "Best ways to secure your email with two-factor authentication"  # Should be harmless
    ]
    
    logger.info("Testing the guard on examples...")
    for i, example in enumerate(test_examples):
        logger.info(f"Testing example {i+1}: {example}")
        is_harmful = guard.is_harmful(example)
        
        if is_harmful:
            similarity = guard.get_similarity(example)
            triggered_category = guard.get_triggered_category(example)
            logger.info(f"DETECTED HARMFUL: Example {i+1} - Similarity: {similarity:.4f}, Category: {triggered_category}")
        else:
            similarity = guard.get_similarity(example)
            logger.info(f"HARMLESS: Example {i+1} - Similarity: {similarity:.4f}")
    
    # Step 6: Print a complete summary of the contrastive vectors
    logger.info("Generating contrastive vector summary...")
    guard.print_contrastive_summary()
    
    logger.info("Demonstration complete!")

if __name__ == "__main__":
    main() 