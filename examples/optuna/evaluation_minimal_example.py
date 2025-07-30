#!/usr/bin/env python3
"""
Evaluation Minimal Example - Testing Best Parameters

This example demonstrates how to evaluate steering parameters without running 
a full Optuna optimization. It shows:

1. Manual parameter evaluation with pre-configured settings
2. Cross-dataset evaluation (test on multiple datasets)  
3. Parameter comparison (test multiple parameter sets)
4. Loading parameters from previous optimization runs

EXPECTED BEHAVIOR:
- Fine-tuned model (realtreetune/rho-1b-sft-GSM8K) baseline: ~14-27% on GSM8K
- Small steering α (0.1-0.2) should maintain or slightly hurt performance
- Zero steering should match baseline performance
- Cross-dataset results may vary significantly

USAGE:
    # Basic evaluation with optimal parameters
    python evaluation_minimal_example.py
    
    # Test multiple parameter sets
    python evaluation_minimal_example.py --compare-params
    
    # Cross-dataset evaluation
    python evaluation_minimal_example.py --cross-dataset
    
    # Load from existing study
    python evaluation_minimal_example.py --study-db path/to/study.db
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add wisent-guard to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs import EXAMPLE_BEST_PARAMS, create_evaluation_config

from wisent_guard.core.optuna.optuna_pipeline import OptimizationPipeline


def evaluate_single_params(params: dict[str, Any], config_name: str = "default") -> dict[str, Any]:
    """Evaluate a single set of parameters."""
    print(f"\n🔬 Evaluating {config_name} parameters:")
    print(f"   Layer: {params['layer_id']}")
    print(f"   Method: {params['steering_method']}")
    print(f"   Alpha: {params['steering_alpha']}")
    
    # Create evaluation config
    config = create_evaluation_config(
        test_limit=50,  # Quick evaluation
        layer_id=params["layer_id"]
    )
    
    # Run evaluation
    pipeline = OptimizationPipeline(config)
    
    try:
        results = pipeline.evaluate_only(params)
        
        # Extract key metrics
        baseline_acc = results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = results["steered_benchmark_metrics"]["accuracy"] 
        improvement = results["accuracy_improvement"]
        probe_auc = results["test_probe_metrics"]["auc"]
        
        print("   📊 Results:")
        print(f"      Baseline:    {baseline_acc:.4f}")
        print(f"      Steered:     {steered_acc:.4f}")
        print(f"      Improvement: {improvement:+.4f}")
        print(f"      Probe AUC:   {probe_auc:.4f}")
        
        return {
            "config_name": config_name,
            "params": params,
            "baseline_accuracy": baseline_acc,
            "steered_accuracy": steered_acc,
            "improvement": improvement,
            "probe_auc": probe_auc
        }
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "config_name": config_name,
            "params": params,
            "error": str(e)
        }
    finally:
        pipeline.cleanup_memory()


def compare_multiple_params() -> list[dict[str, Any]]:
    """Compare multiple parameter sets side by side."""
    print("\n" + "="*80)
    print("🔬 PARAMETER COMPARISON")
    print("="*80)
    print("Testing multiple parameter sets to compare performance...")
    
    # Select interesting parameter sets to compare
    param_sets = {
        "Zero Steering": EXAMPLE_BEST_PARAMS["zero_steering"],
        "Minimal CAA": EXAMPLE_BEST_PARAMS["minimal_caa"], 
        "Minimal DAC": EXAMPLE_BEST_PARAMS["minimal_dac"],
    }
    
    results = []
    
    for name, params in param_sets.items():
        result = evaluate_single_params(params, name)
        results.append(result)
    
    # Display comparison table
    print("\n" + "="*80)
    print("📊 COMPARISON RESULTS")
    print("="*80)
    print(f"{'Configuration':<15} | {'Method':<6} | {'Alpha':<6} | {'Baseline':<8} | {'Steered':<8} | {'Δ':<8}")
    print("-" * 80)
    
    for result in results:
        if "error" in result:
            print(f"{result['config_name']:<15} | ERROR: {result['error']}")
        else:
            params = result["params"]
            print(f"{result['config_name']:<15} | "
                  f"{params['steering_method']:<6} | "
                  f"{params['steering_alpha']:<6.2f} | "
                  f"{result['baseline_accuracy']:<8.4f} | "
                  f"{result['steered_accuracy']:<8.4f} | "
                  f"{result['improvement']:<+8.4f}")
    
    print("="*80)
    
    return results


def cross_dataset_evaluation(params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the same parameters across multiple datasets."""
    print("\n" + "="*80)
    print("🌍 CROSS-DATASET EVALUATION") 
    print("="*80)
    print(f"Testing {params['steering_method']} (α={params['steering_alpha']}) across datasets...")
    
    datasets = ["gsm8k", "hendrycks_math", "aime"]
    results = {}
    
    for dataset in datasets:
        print(f"\n📊 Evaluating on {dataset}...")
        
        # Create dataset-specific config
        config = create_evaluation_config(
            test_dataset=dataset,
            test_limit=30,  # Smaller for speed
            layer_id=params["layer_id"]
        )
        
        pipeline = OptimizationPipeline(config)
        
        try:
            result = pipeline.evaluate_on_dataset(params, dataset, 30)
            
            baseline_acc = result["baseline_benchmark_metrics"]["accuracy"]
            steered_acc = result["steered_benchmark_metrics"]["accuracy"]
            improvement = result["accuracy_improvement"]
            
            print(f"   Baseline:    {baseline_acc:.4f}")
            print(f"   Steered:     {steered_acc:.4f}")
            print(f"   Improvement: {improvement:+.4f}")
            
            results[dataset] = {
                "baseline_accuracy": baseline_acc,
                "steered_accuracy": steered_acc, 
                "improvement": improvement
            }
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[dataset] = {"error": str(e)}
            
        finally:
            pipeline.cleanup_memory()
    
    # Display cross-dataset summary
    print("\n" + "="*60)
    print("🌍 CROSS-DATASET SUMMARY")
    print("="*60)
    print(f"{'Dataset':<15} | {'Baseline':<8} | {'Steered':<8} | {'Δ':<8}")
    print("-" * 60)
    
    for dataset, result in results.items():
        if "error" in result:
            print(f"{dataset:<15} | ERROR: {result['error']}")
        else:
            print(f"{dataset:<15} | "
                  f"{result['baseline_accuracy']:<8.4f} | "
                  f"{result['steered_accuracy']:<8.4f} | "
                  f"{result['improvement']:<+8.4f}")
    
    print("="*60)
    
    return results


def load_from_study(study_db: str) -> dict[str, Any]:
    """Load best parameters from existing Optuna study."""
    print(f"\n📂 Loading from study: {study_db}")
    
    try:
        pipeline, study = OptimizationPipeline.from_saved_study(study_db)
        
        print(f"   Study name: {study.study_name}")
        print(f"   Total trials: {len(study.trials)}")
        print(f"   Best value: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        
        # Evaluate the best parameters
        results = pipeline.evaluate_only(study.best_params)
        
        baseline_acc = results["baseline_benchmark_metrics"]["accuracy"]
        steered_acc = results["steered_benchmark_metrics"]["accuracy"]
        improvement = results["accuracy_improvement"]
        
        print("\n📊 Study Results:")
        print(f"   Baseline:    {baseline_acc:.4f}")
        print(f"   Steered:     {steered_acc:.4f}")
        print(f"   Improvement: {improvement:+.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to load study: {e}")
        return {"error": str(e)}
    finally:
        if "pipeline" in locals():
            pipeline.cleanup_memory()


def main():
    """Main evaluation example."""
    parser = argparse.ArgumentParser(
        description="Evaluation minimal example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--compare-params", action="store_true",
                       help="Compare multiple parameter sets")
    parser.add_argument("--cross-dataset", action="store_true", 
                       help="Evaluate across multiple datasets")
    parser.add_argument("--study-db", type=str,
                       help="Load parameters from existing study database")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 EVALUATION MINIMAL EXAMPLE")
    logger.info("="*80)
    logger.info("Model: realtreetune/rho-1b-sft-GSM8K (fine-tuned)")
    logger.info("Purpose: Test steering parameters without full optimization")
    logger.info("="*80)
    
    try:
        if args.study_db:
            # Load from existing study
            results = load_from_study(args.study_db)
            
        elif args.compare_params:
            # Compare multiple parameter sets
            results = compare_multiple_params()
            
        elif args.cross_dataset:
            # Cross-dataset evaluation with optimal parameters
            optimal_params = EXAMPLE_BEST_PARAMS["minimal_caa"]
            results = cross_dataset_evaluation(optimal_params)
            
        else:
            # Default: Single evaluation with optimal parameters
            print("\n🎯 Default evaluation with optimal CAA parameters")
            optimal_params = EXAMPLE_BEST_PARAMS["minimal_caa"]
            results = evaluate_single_params(optimal_params, "Optimal CAA")
        
        print("\n✅ Evaluation completed successfully!")
        
        # Analysis and insights
        print("\n" + "="*60)
        print("💡 INSIGHTS")
        print("="*60)
        print("• Fine-tuned models typically perform best with minimal steering")
        print("• Zero steering (α=0.0) should match baseline performance")
        print("• Small positive steering may hurt performance on fine-tuned models")
        print("• Cross-dataset results show generalization capabilities")
        print("• DAC method can handle higher α values than CAA")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None


if __name__ == "__main__":
    main()