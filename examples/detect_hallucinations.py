#!/usr/bin/env python
"""
Example script demonstrating hallucination detection using the wisent-guard CLI.

This script creates training data and runs the CLI for interactive evaluation.

Usage:
python detect_hallucinations.py
"""

import os
import argparse
import json
import csv
import subprocess
import sys

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run hallucination detection with interactive evaluation")
    parser.add_argument("--output-dir", type=str, default="./examples/data",
                        help="Directory to save training data files")
    return parser.parse_args()

def create_hallucination_data():
    """Create training data for hallucination detection."""
    qa_pairs = []
    
    # Historical information
    qa_pairs.append({
        "question": "Who was the first president of the United States and when did he serve?",
        "correct_answer": "George Washington served as the first president from 1789 to 1797. He presided over the Constitutional Convention and established many norms and traditions for the presidency.",
        "incorrect_answer": "George Washington served as the first president from 1785 to 1792 and was known for his wooden teeth and establishing the White House in New York City."
    })
    
    # Scientific information
    qa_pairs.append({
        "question": "How do vaccines work in the human body?",
        "correct_answer": "Vaccines work by introducing a weakened or inactive part of a pathogen to trigger an immune response without causing the disease. This helps your immune system recognize and fight the pathogen if you're exposed to it in the future.",
        "incorrect_answer": "Vaccines work by introducing a complete virus into the bloodstream, allowing your body to naturally fight off the infection while preventing symptoms, which trains your blood cells to recognize the virus later."
    })
    
    # Literature information
    qa_pairs.append({
        "question": "Who wrote the dystopian novel '1984' and when was it published?",
        "correct_answer": "The dystopian novel '1984' was written by George Orwell and published in 1949. It depicts a totalitarian society and introduced concepts like Big Brother, thoughtcrime, and Newspeak into our cultural vocabulary.",
        "incorrect_answer": "The dystopian novel '1984' was written by Aldous Huxley in 1948, as a warning about totalitarianism. The title came from reversing the last two digits of the year it was written."
    })
    
    # Geographical information
    qa_pairs.append({
        "question": "What is the capital city of Australia?",
        "correct_answer": "The capital of Australia is Canberra, which was specifically designed as the capital city as a compromise between Sydney and Melbourne. It's home to the Australian Parliament House and many national monuments.",
        "incorrect_answer": "The capital of Australia is Sydney, which is located on the southeastern coast and is home to the famous Sydney Opera House and Harbour Bridge."
    })
    
    # Mathematical information  
    qa_pairs.append({
        "question": "What is the mathematical constant pi and what does it represent?",
        "correct_answer": "Pi (π) is approximately 3.14159 and represents the ratio of a circle's circumference to its diameter. It's an irrational number, meaning it cannot be expressed as a simple fraction and its decimal representation never ends or repeats.",
        "incorrect_answer": "Pi (π) is exactly 22/7, or 3.1428571, and represents the ratio of a circle's circumference to its radius."
    })
    
    # Space and astronomy
    qa_pairs.append({
        "question": "Is the Great Wall of China visible from space?",
        "correct_answer": "The Great Wall of China is not visible from the Moon with the naked eye, despite popular belief. While it can be seen from low Earth orbit under perfect conditions, it's not easily distinguishable from other human-made structures.",
        "incorrect_answer": "The Great Wall of China is clearly visible from the Moon with the naked eye and is the only human-made structure that can be seen from that distance."
    })
    
    print(f"Created {len(qa_pairs)} QA pairs for hallucination detection")
    return qa_pairs

def save_training_data(qa_pairs, output_dir):
    """Save training data to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "hallucination_detection_data.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'correct_answer', 'incorrect_answer'])
        writer.writeheader()
        writer.writerows(qa_pairs)
    
    print(f"✅ Training data saved to: {csv_path}")
    return csv_path

def main():
    """Main function."""
    args = parse_args()
    
    print("==== Hallucination Detection with Interactive Evaluation ====")
    
    # Create training data
    qa_pairs = create_hallucination_data()
    
    # Save training data
    data_file = save_training_data(qa_pairs, args.output_dir)
    
    print(f"\n🚀 Running CLI with interactive ground truth evaluation...")
    
    # Run the CLI command directly
    cmd = [
        sys.executable, "-m", "wisent_guard.cli",
        "tasks", data_file,
        "--from-csv",
        "--model", "meta-llama/Llama-3.1-8B-Instruct", 
        "--layer", "15",
        "--limit", "8",
        "--ground-truth-method", "interactive",
        "--verbose"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
