#!/usr/bin/env python3
"""
TruthfulQA Classifier Script for Llama 3.1 8B
Run hallucination detection on layer 15 for the entire dataset and save results as CSV for manual labeling.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_truthfulqa_classification():
    """
    Run the TruthfulQA classification pipeline using Llama 3.1 8B on layer 15.
    Saves results as CSV for manual labeling.
    """
    
    # Create output directory
    output_dir = "truthfulqa_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*80)
    print("🎯 TRUTHFULQA HALLUCINATION DETECTION - LAYER 15")
    print("="*80)
    print(f"📋 Configuration:")
    print(f"   • Model: meta-llama/Llama-3.1-8B-Instruct")
    print(f"   • Layer: 15")
    print(f"   • Dataset: TruthfulQA (entire dataset)")
    print(f"   • Task: Hallucination detection")
    print(f"   • Split: 80% train, 20% test")
    print(f"   • Classifier: Logistic regression")
    print(f"   • Output directory: {output_dir}")
    print(f"   • CSV will be auto-generated with timestamp")
    print("="*80)
    
    # Build the command
    cmd = [
        sys.executable, "-m", "wisent_guard.cli",
        "tasks", "truthful_qa",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--layer", "15",
        "--split-ratio", "0.8",
        "--classifier-type", "logistic",
        "--max-new-tokens", "300",
        "--token-aggregation", "average",
        "--verbose",
        "--seed", "42",
        # Save results to output directory
        "--output", output_dir,
        # Performance settings
        "--enable-memory-tracking",
        "--show-memory-usage",
        "--show-timing-summary"
    ]
    
    print(f"🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        print("\n" + "="*80)
        print("✅ CLASSIFICATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Results saved to: {output_dir}")
        print("   • CSV file (for labeling): classification_results_[timestamp].csv")
        print("   • JSON file (full results): results_[timestamp].json")
        print()
        print("📝 Next steps:")
        print("   1. Open the CSV file to review predictions")
        print("   2. Fill in the 'ground_truth' column with:")
        print("      - 'TRUTHFUL' for truthful responses")
        print("      - 'HALLUCINATION' for hallucinated responses")
        print("   3. Use the labeled data for model evaluation")
        print("="*80)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Classification failed with exit code {e.returncode}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False
    
    except KeyboardInterrupt:
        print("\n🛑 Classification interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def run_quick_test():
    """
    Run a quick test with limited samples to verify the setup.
    """
    output_dir = "truthfulqa_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧪 Running quick test with 10 samples...")
    
    cmd = [
        sys.executable, "-m", "wisent_guard.cli",
        "tasks", "truthful_qa",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--layer", "15",
        "--limit", "10",
        "--split-ratio", "0.8",
        "--classifier-type", "logistic",
        "--verbose",
        "--output", output_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ Test completed! Results in {output_dir}")
        return True
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="TruthfulQA Classifier for Llama 3.1 8B")
    parser.add_argument("--test", action="store_true", 
                       help="Run quick test with 10 samples instead of full dataset")
    parser.add_argument("--output-dir", default="truthfulqa_results",
                       help="Output directory for results (default: truthfulqa_results)")
    
    args = parser.parse_args()
    
    if args.test:
        success = run_quick_test()
    else:
        success = run_truthfulqa_classification()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
