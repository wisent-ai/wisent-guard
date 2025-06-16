#!/usr/bin/env python
"""
Example script demonstrating inference with wisent-guard CLI.

This script shows how to run inference with a trained detector using the CLI.

Usage:
python just_inference.py
"""

import os
import argparse
import csv
import subprocess
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with wisent-guard CLI")
    parser.add_argument("--output-dir", type=str, default="./examples/data",
                        help="Directory containing training data files")
    parser.add_argument("--detector-type", type=str, default="hallucination",
                        choices=["hallucination", "bias", "harmful", "personal", "scheming", "bad_code"],
                        help="Type of detector to use")
    return parser.parse_args()

def create_inference_prompts():
    """Create test prompts for inference."""
    prompts = [
        "What is the capital of France?",
        "How do I learn programming?", 
        "What's the weather like today?",
        "Tell me about artificial intelligence.",
        "What are some good study techniques?"
    ]
    return prompts

def save_inference_prompts(prompts, output_dir):
    """Save inference prompts to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "inference_prompts.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'correct_answer', 'incorrect_answer'])
        writer.writeheader()
            
        # Create dummy entries for inference testing
        for prompt in prompts:
            writer.writerow({
                'question': prompt,
                'correct_answer': "This is a test for inference.",
                'incorrect_answer': "This is also a test for inference."
            })
    
    print(f"✅ Inference prompts saved to: {csv_path}")
    return csv_path

def main():
    """Main function."""
    args = parse_args()
    
    print("==== Inference with Wisent-Guard CLI ====")
    
    # Create test prompts
    prompts = create_inference_prompts()
    data_file = save_inference_prompts(prompts, args.output_dir)
    
    # Map detector types to data files
    detector_files = {
        "hallucination": "hallucination_detection_data.csv",
        "bias": "bias_detection_data.csv",
        "harmful": "harmful_content_detection_data.csv",
        "personal": "personal_info_detection_data.csv",
        "scheming": "scheming_detection_data.csv",
        "bad_code": "bad_code_detection_data.csv"
    }
    
    detector_file = detector_files.get(args.detector_type, "hallucination_detection_data.csv")
    detector_path = os.path.join(args.output_dir, detector_file)
    
    if not os.path.exists(detector_path):
        print(f"❌ Detector data file not found: {detector_path}")
        print(f"Run the corresponding detect_{args.detector_type}.py script first to create training data.")
        return
    
    print(f"\n🚀 Running inference with {args.detector_type} detector...")
    
    # Run inference with the trained detector
    cmd = [
        sys.executable, "-m", "wisent_guard.cli",
        "tasks", data_file,
        "--from-csv",
        "--model", "meta-llama/Llama-3.1-8B-Instruct", 
        "--layer", "15",
        "--limit", str(len(prompts)),
        "--ground-truth-method", "interactive",
        "--verbose"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()