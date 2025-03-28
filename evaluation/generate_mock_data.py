#!/usr/bin/env python
"""
Generate mock token score data for testing evaluate_responses.py.

This script creates a synthetic dataset that mimics the results of a TruthfulQA evaluation,
including token-level scores for correct and hallucinatory responses.
"""

import os
import csv
import random
import numpy as np
import argparse

def generate_token_data(num_tokens, base_score, is_hallucination):
    """
    Generate mock token data with similarity scores.
    
    Args:
        num_tokens: Number of tokens to generate
        base_score: Base similarity score
        is_hallucination: Whether to generate scores for a hallucinatory response
        
    Returns:
        A list of token data entries
    """
    tokens = []
    
    # Hallucination responses will have at least one token with high similarity
    max_token_pos = random.randint(0, num_tokens-1) if is_hallucination else -1
    
    for i in range(num_tokens):
        # For hallucinatory responses, make one token have a high similarity score
        if i == max_token_pos:
            # High similarity for a hallucination (between 0.5 and 0.9)
            similarity = random.uniform(0.5, 0.9)
            is_harmful = True
            category = "hallucination"
        else:
            # Normal distribution around base score with some variance
            # Correct responses have lower base scores
            variance = 0.1
            similarity = np.clip(np.random.normal(base_score, variance), 0.0, 0.95)
            is_harmful = similarity > 0.5  # Just for demonstration
            category = "hallucination" if is_harmful else ""
        
        # Create a token string with the expected format
        # position:token_id:token_text:similarity:category:is_harmful
        token_id = random.randint(1000, 30000)
        token_text = f"token_{i}"
        token_data = f"{i}:{token_id}:{token_text}:{similarity:.6f}:{category}:{is_harmful}"
        tokens.append(token_data)
    
    return tokens

def generate_mock_results(num_samples, output_file):
    """
    Generate mock evaluation results and save to CSV.
    
    Args:
        num_samples: Number of samples to generate
        output_file: Path to save the CSV file
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define the ratio of correct to hallucinatory responses
    # We want more hallucinatory responses to get a better threshold estimation
    correct_ratio = 0.4
    
    # Create mock results
    results = []
    
    # Half of the samples will be marked as correct by Claude (claude_score=1)
    for i in range(num_samples):
        # Determine if this response is hallucinatory according to Claude
        is_hallucinatory = random.random() > correct_ratio
        claude_score = 0 if is_hallucinatory else 1
        
        # Generate number of tokens (between 5 and 20)
        num_tokens = random.randint(5, 20)
        
        # Set base similarity score
        # Hallucinatory responses have a higher base similarity to harmful content
        base_score = 0.3 if is_hallucinatory else 0.1
        
        # Generate token scores
        token_scores = generate_token_data(num_tokens, base_score, is_hallucinatory)
        
        # Create a result entry
        result = {
            'question': f"Question {i+1}",
            'category': random.choice(['Science', 'History', 'Medicine', 'Finance', 'Psychology']),
            'baseline_response': f"Baseline response for question {i+1}",
            'guard_response': f"Guard response for question {i+1}",
            'baseline_claude_score': claude_score,
            'guard_claude_score': claude_score,
            'is_harmful': is_hallucinatory,
            'similarity_score': base_score * 1.5 if is_hallucinatory else base_score,
            'blocked': random.random() < 0.3 if is_hallucinatory else False,
            'block_reason': "Harmful content detected" if is_hallucinatory and random.random() < 0.3 else "",
            'token_scores': '|'.join(token_scores)
        }
        
        results.append(result)
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Generated {num_samples} mock results saved to {output_file}")
    
    # Print summary
    correct_count = sum(1 for r in results if r['guard_claude_score'] == 1)
    hallucination_count = sum(1 for r in results if r['guard_claude_score'] == 0)
    
    print(f"Summary:")
    print(f"  Correct responses: {correct_count}")
    print(f"  Hallucinatory responses: {hallucination_count}")

def main():
    parser = argparse.ArgumentParser(description="Generate mock token score data for testing")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of mock samples to generate")
    parser.add_argument("--output-file", type=str, default="evaluation/results/mock_combined_results.csv",
                       help="Path to save the mock results CSV file")
    args = parser.parse_args()
    
    generate_mock_results(args.num_samples, args.output_file)

if __name__ == "__main__":
    main() 