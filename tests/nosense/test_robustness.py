#!/usr/bin/env python3
"""
Test script for comparing real vs nonsense benchmark performance.

This script runs the same evaluation on both real and nonsense data
to check if the evaluation system can distinguish between them.
"""

import argparse
import json
import subprocess
import tempfile
import time
import sys
from pathlib import Path

# Add wisent-guard to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from math500_nosense import Math500NosenseGenerator


def run_wisent_evaluation(data_source, model, limit, is_nonsense=False):
    """
    Run wisent-guard evaluation on data.

    Args:
        data_source: Task name (str) or file path for nonsense data
        model: Model name
        limit: Sample limit
        is_nonsense: Whether this is nonsense data

    Returns:
        Dictionary with results
    """
    if is_nonsense:
        # For nonsense data, use file input
        cmd = [
            "python", "-m", "wisent_guard", "tasks", data_source,
            "--from-json",
            "--model", model,
            "--limit", str(limit),
            "--layer", "15",
            "--verbose"
        ]
    else:
        # For real data, use task name
        cmd = [
            "python", "-m", "wisent_guard", "tasks", data_source,
            "--model", model,
            "--limit", str(limit),
            "--layer", "15",
            "--verbose"
        ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"DEBUG: Command failed with return code: {result.returncode}")
            print(f"DEBUG: stderr: '{result.stderr}'")
            print(f"DEBUG: stdout: '{result.stdout}'")
            return {
                "success": False,
                "error": f"Command failed: {result.stderr}",
                "accuracy": 0.0
            }

        # Parse output for accuracy
        output = result.stdout
        accuracies = extract_accuracy(output)

        return {
            "success": True,
            "training_accuracy": accuracies["training_accuracy"],
            "evaluation_accuracy": accuracies["evaluation_accuracy"],
            "lm_eval_accuracy": accuracies["lm_eval_accuracy"],
            "accuracy": accuracies["lm_eval_accuracy"],  # For backward compatibility - use lm-eval as primary
            "output": output
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Evaluation timed out",
            "accuracy": 0.0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "accuracy": 0.0
        }


def extract_accuracy(output: str) -> dict:
    """Extract training, evaluation, and lm-eval-harness accuracy from wisent-guard output."""
    lines = output.split('\n')

    training_accuracy = None
    evaluation_accuracy = None
    lm_eval_accuracy = None

    for line in lines:
        # Look for training accuracy
        if "Training accuracy:" in line and "%" in line:
            try:
                acc_part = line.split("Training accuracy:")[1].split("%")[0].strip()
                training_accuracy = float(acc_part) 
            except:
                continue

        # Look for classifier evaluation accuracy
        elif "Classifier evaluation accuracy:" in line and "%" in line:
            try:
                acc_part = line.split("Classifier evaluation accuracy:")[1].split("%")[0].strip()
                evaluation_accuracy = float(acc_part) 
            except:
                continue

        # Look for lm-eval-harness accuracy (appears after "LM-eval-harness evaluation completed")
        elif "📊 Accuracy:" in line and "%" in line:
            try:
                acc_part = line.split("📊 Accuracy:")[1].split("%")[0].strip()
                lm_eval_accuracy = float(acc_part) 
            except:
                continue

        # Look for general accuracy patterns as fallback
        elif "Accuracy:" in line and "%" in line and lm_eval_accuracy is None:
            try:
                acc_part = line.split("Accuracy:")[1].split("%")[0].strip()
                # If we haven't found specific accuracies, use this as lm-eval
                if lm_eval_accuracy is None:
                    lm_eval_accuracy = float(acc_part) 
            except:
                continue

    return {
        "training_accuracy": training_accuracy if training_accuracy is not None else 0.0,
        "evaluation_accuracy": evaluation_accuracy if evaluation_accuracy is not None else 0.0,
        "lm_eval_accuracy": lm_eval_accuracy if lm_eval_accuracy is not None else 0.0
    }


def test_math500_robustness(model="EleutherAI/gpt-neo-1.3B", limit=10, verbose=False):
    """Test Math500 robustness with real vs nonsense data."""

    print("🧪 MATH500 ROBUSTNESS TEST")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Sample limit: {limit}")
    print("=" * 50)

    # Step 1: Test with real data
    print("\n📊 Testing with REAL data...")
    real_results = run_wisent_evaluation("math500", model, limit, is_nonsense=False)

    if verbose and real_results.get("output"):
        print("Real data output:")
        print(real_results["output"])

    print(f"Real data training accuracy: {real_results['training_accuracy']}%")
    print(f"Real data evaluation accuracy: {real_results['evaluation_accuracy']}%")
    print(f"Real data lm-eval accuracy: {real_results['lm_eval_accuracy']}%")

    if not real_results["success"]:
        print(f"❌ Real data evaluation failed: {real_results.get('error', 'Unknown error')}")
        return

    # Step 2: Generate nonsense data
    print("\n🎭 Generating NONSENSE data...")
    generator = Math500NosenseGenerator()
    nonsense_data = generator.generate_nonsense_data(limit=limit)

    # Save nonsense data to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(nonsense_data, f, indent=2)
        temp_file = f.name

    print(f"Generated {len(nonsense_data)} nonsense items")
    if verbose:
        print("Sample nonsense item:")
        sample = nonsense_data[0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Correct answer: {sample['correct_answer']}")
        print(f"  Incorrect answer: {sample['incorrect_answer']}")

    # Step 3: Test with nonsense data
    print("\n🎭 Testing with NONSENSE data...")
    try:
        nonsense_results = run_wisent_evaluation(temp_file, model, limit, is_nonsense=True)

        print(f"DEBUG: nonsense_results keys: {list(nonsense_results.keys())}")
        print(f"DEBUG: nonsense_results: {nonsense_results}")

        if verbose and nonsense_results.get("output"):
            print("Nonsense data output:")
            print(nonsense_results["output"])

        print(f"Nonsense data training accuracy: {nonsense_results['training_accuracy']}%")
        print(f"Nonsense data evaluation accuracy: {nonsense_results['evaluation_accuracy']}%")
        print(f"Nonsense data lm-eval accuracy: {nonsense_results['lm_eval_accuracy']}%")

        if not nonsense_results["success"]:
            print(f"❌ Nonsense data evaluation failed: {nonsense_results.get('error', 'Unknown error')}")
            return

    finally:
        # Clean up temp file
        import os
        try:
            os.unlink(temp_file)
        except:
            pass

    # Step 4: Compare results
    print("\n📈 COMPARISON RESULTS")
    print("=" * 60)
    print("TRAINING ACCURACY:")
    print(f"  Real data:     {real_results['training_accuracy']:.1%}")
    print(f"  Nonsense data: {nonsense_results['training_accuracy']:.1%}")
    training_difference = real_results['training_accuracy'] - nonsense_results['training_accuracy']
    print(f"  Difference:    {training_difference:.1%}")

    print("\nCLASSIFIER EVALUATION ACCURACY:")
    print(f"  Real data:     {real_results['evaluation_accuracy']:.1%}")
    print(f"  Nonsense data: {nonsense_results['evaluation_accuracy']:.1%}")
    eval_difference = real_results['evaluation_accuracy'] - nonsense_results['evaluation_accuracy']
    print(f"  Difference:    {eval_difference:.1%}")

    print("\nLM-EVAL-HARNESS ACCURACY:")
    print(f"  Real data:     {real_results['lm_eval_accuracy']:.1%}")
    print(f"  Nonsense data: {nonsense_results['lm_eval_accuracy']:.1%}")
    lm_eval_difference = real_results['lm_eval_accuracy'] - nonsense_results['lm_eval_accuracy']
    print(f"  Difference:    {lm_eval_difference:.1%}")

    # Analysis
    print("\n🔍 ROBUSTNESS ANALYSIS")
    print("-" * 60)

    print("📊 LM-EVAL-HARNESS ROBUSTNESS:")
    if nonsense_results['lm_eval_accuracy'] > 0.2:  # > 20%
        print("🚨 WARNING: Nonsense data has high lm-eval accuracy (>20%)")
        print("   This suggests the lm-eval evaluation may be broken")
    elif nonsense_results['lm_eval_accuracy'] > 0.1:  # > 10%
        print("⚠️  CONCERN: Nonsense data has moderate lm-eval accuracy (>10%)")
        print("   The lm-eval evaluation may have issues")
    else:
        print("✅ GOOD: Nonsense data has low lm-eval accuracy (<10%)")

    if lm_eval_difference < 0.3:  # < 30% difference
        print("🚨 WARNING: Small lm-eval difference between real and nonsense (<30%)")
        print("   The lm-eval system may not be learning meaningful patterns")
    else:
        print("✅ GOOD: Large lm-eval difference between real and nonsense (≥30%)")

    print("\n🎯 CLASSIFIER ROBUSTNESS:")
    if nonsense_results['evaluation_accuracy'] > 0.2:  # > 20%
        print("🚨 WARNING: Nonsense data has high classifier accuracy (>20%)")
        print("   This suggests the classifier evaluation may be broken")
    elif nonsense_results['evaluation_accuracy'] > 0.1:  # > 10%
        print("⚠️  CONCERN: Nonsense data has moderate classifier accuracy (>10%)")
        print("   The classifier evaluation may have issues")
    else:
        print("✅ GOOD: Nonsense data has low classifier accuracy (<10%)")

    if eval_difference < 0.3:  # < 30% difference
        print("🚨 WARNING: Small classifier difference between real and nonsense (<30%)")
        print("   The classifier may not be learning meaningful patterns")
    else:
        print("✅ GOOD: Large classifier difference between real and nonsense (≥30%)")

    # Overall assessment
    lm_eval_robust = (nonsense_results['lm_eval_accuracy'] < 0.1 and
                      lm_eval_difference >= 0.3 and
                      real_results['lm_eval_accuracy'] > 0.3)

    classifier_robust = (nonsense_results['evaluation_accuracy'] < 0.1 and
                        eval_difference >= 0.3 and
                        real_results['evaluation_accuracy'] > 0.3)

    if lm_eval_robust and classifier_robust:
        print("\n🎉 ROBUSTNESS TEST: PASSED")
        print("   Both lm-eval and classifier systems appear robust")
    elif lm_eval_robust:
        print("\n🎉 LM-EVAL ROBUSTNESS: PASSED")
        print("⚠️  CLASSIFIER ROBUSTNESS: ISSUES DETECTED")
    elif classifier_robust:
        print("\n🎉 CLASSIFIER ROBUSTNESS: PASSED")
        print("⚠️  LM-EVAL ROBUSTNESS: ISSUES DETECTED")
    else:
        print("\n⚠️  ROBUSTNESS TEST: ISSUES DETECTED")
        print("   Both evaluation systems may have problems")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Math500 robustness with nonsense data")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-1.3B",
                       help="Model to test (default: EleutherAI/gpt-neo-1.3B)")
    parser.add_argument("--limit", type=int, default=10,
                       help="Number of samples to test (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output")

    args = parser.parse_args()

    try:
        test_math500_robustness(
            model=args.model,
            limit=args.limit,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())