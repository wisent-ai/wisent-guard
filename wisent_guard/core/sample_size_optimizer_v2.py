"""
Simplified Sample Size Optimizer using training-limit and testing-limit flags.
Supports both classification and steering methods.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from ..cli import run_task_pipeline
from .model_config_manager import ModelConfigManager

logger = logging.getLogger(__name__)


class SimplifiedSampleSizeOptimizer:
    """Simplified sample size optimizer that leverages CLI training/testing limits."""
    
    def __init__(
        self,
        model_name: str,
        task_name: str,
        layer: int,
        method_type: str = "classification",  # "classification" or "steering"
        sample_sizes: Optional[List[int]] = None,
        test_size: int = 200,
        seed: int = 42,
        verbose: bool = False,
        **method_kwargs
    ):
        """
        Initialize the optimizer.
        
        Args:
            model_name: Model to optimize
            task_name: Task to optimize for
            layer: Layer to use
            method_type: "classification" or "steering"
            sample_sizes: List of training sample sizes to test
            test_size: Fixed test set size
            seed: Random seed for reproducibility
            verbose: Verbose output
            **method_kwargs: Additional arguments for the method
                For classification: token_aggregation, threshold, classifier_type
                For steering: steering_method, steering_strength, token_targeting_strategy
        """
        self.model_name = model_name
        self.task_name = task_name
        self.layer = layer
        self.method_type = method_type
        self.sample_sizes = sample_sizes or [5, 10, 20, 50, 100, 200, 500]
        self.test_size = test_size
        self.seed = seed
        self.verbose = verbose
        self.method_kwargs = method_kwargs
        
        # Results storage
        self.results = {
            "sample_sizes": [],
            "accuracies": [],
            "f1_scores": [],
            "training_times": [],
            "evaluation_times": []
        }
        
    def run_single_experiment(self, training_size: int) -> Dict[str, Any]:
        """
        Run a single experiment with a specific training size.
        
        Args:
            training_size: Number of training samples
            
        Returns:
            Dictionary with results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Testing {self.method_type} with {training_size} training samples")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build arguments for run_task_pipeline
        pipeline_args = {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "layer": str(self.layer),
            "training_limit": training_size,
            "testing_limit": self.test_size,
            "seed": self.seed,
            "verbose": self.verbose,
            "split_ratio": 0.8,  # Standard split
            "limit": training_size + self.test_size + 100,  # Ensure enough data
        }
        
        # Add method-specific arguments
        if self.method_type == "classification":
            pipeline_args.update({
                "token_aggregation": self.method_kwargs.get("token_aggregation", "average"),
                "detection_threshold": self.method_kwargs.get("threshold", 0.5),
                "classifier_type": self.method_kwargs.get("classifier_type", "logistic"),
                "steering_mode": False
            })
        else:  # steering
            pipeline_args.update({
                "steering_mode": True,
                "steering_method": self.method_kwargs.get("steering_method", "CAA"),
                "steering_strength": self.method_kwargs.get("steering_strength", 1.0),
                "token_targeting_strategy": self.method_kwargs.get("token_targeting_strategy", "LAST_TOKEN"),
                "token_aggregation": self.method_kwargs.get("token_aggregation", "average"),
            })
        
        try:
            # Run the pipeline
            result = run_task_pipeline(**pipeline_args)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Extract metrics based on method type
            if self.method_type == "classification":
                accuracy = result.get("test_accuracy", 0.0)
                f1_score = result.get("test_f1_score", 0.0)
            else:  # steering
                # For steering, we look at the evaluation results
                eval_results = result.get("evaluation_results", {})
                accuracy = eval_results.get("accuracy", 0.0)
                # Convert to float if it's a string percentage
                if isinstance(accuracy, str) and accuracy.endswith('%'):
                    accuracy = float(accuracy.rstrip('%')) / 100.0
                f1_score = accuracy  # Use accuracy as proxy for F1 in steering
            
            return {
                "accuracy": accuracy,
                "f1_score": f1_score,
                "training_time": result.get("training_time", total_time * 0.8),
                "evaluation_time": total_time * 0.2,
                "total_time": total_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to run experiment with {training_size} samples: {e}")
            return {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "training_time": 0.0,
                "evaluation_time": 0.0,
                "total_time": 0.0,
                "success": False,
                "error": str(e)
            }
    
    def run_optimization(self) -> Dict[str, Any]:
        """
        Run the complete optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting {self.method_type} sample size optimization...")
        logger.info(f"Model: {self.model_name}, Task: {self.task_name}, Layer: {self.layer}")
        logger.info(f"Testing sample sizes: {self.sample_sizes}")
        logger.info(f"Fixed test size: {self.test_size}")
        
        # Run experiments for each sample size
        for sample_size in self.sample_sizes:
            result = self.run_single_experiment(sample_size)
            
            if result["success"]:
                self.results["sample_sizes"].append(sample_size)
                self.results["accuracies"].append(result["accuracy"])
                self.results["f1_scores"].append(result["f1_score"])
                self.results["training_times"].append(result["training_time"])
                self.results["evaluation_times"].append(result["evaluation_time"])
                
                if self.verbose:
                    print(f"\n✓ Tested {sample_size} samples: accuracy={result['accuracy']:.3f}, f1={result['f1_score']:.3f}")
        
        # Find optimal sample size
        optimal_idx, optimal_size = self.find_optimal_sample_size()
        
        return {
            "optimal_sample_size": optimal_size,
            "optimal_accuracy": self.results["accuracies"][optimal_idx] if optimal_idx >= 0 else None,
            "optimal_f1_score": self.results["f1_scores"][optimal_idx] if optimal_idx >= 0 else None,
            "all_results": self.results,
            "method_type": self.method_type,
            "method_kwargs": self.method_kwargs
        }
    
    def find_optimal_sample_size(self) -> Tuple[int, int]:
        """
        Find the optimal sample size based on accuracy and efficiency.
        
        Returns:
            Tuple of (optimal_index, optimal_sample_size)
        """
        if not self.results["accuracies"]:
            return -1, 0
        
        accuracies = np.array(self.results["accuracies"])
        sample_sizes = np.array(self.results["sample_sizes"])
        
        # Find the point of diminishing returns
        # We want the smallest sample size that achieves near-optimal accuracy
        max_accuracy = np.max(accuracies)
        threshold = max_accuracy * 0.95  # Within 95% of best accuracy
        
        # Find indices where accuracy is above threshold
        good_indices = np.where(accuracies >= threshold)[0]
        
        if len(good_indices) > 0:
            # Choose the smallest sample size among good ones
            optimal_idx = good_indices[0]
        else:
            # If no good indices, choose the best accuracy
            optimal_idx = np.argmax(accuracies)
        
        return optimal_idx, sample_sizes[optimal_idx]
    
    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimization results.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results["sample_sizes"]:
            logger.warning("No results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy and F1 score
        ax1.plot(self.results["sample_sizes"], self.results["accuracies"], 
                 'b-o', label='Accuracy', markersize=8)
        ax1.plot(self.results["sample_sizes"], self.results["f1_scores"], 
                 'r--s', label='F1 Score', markersize=8)
        
        # Mark optimal point
        optimal_idx, optimal_size = self.find_optimal_sample_size()
        if optimal_idx >= 0:
            ax1.axvline(x=optimal_size, color='g', linestyle=':', alpha=0.7, 
                       label=f'Optimal: {optimal_size}')
        
        ax1.set_xlabel('Training Sample Size')
        ax1.set_ylabel('Score')
        ax1.set_title(f'{self.method_type.capitalize()} Performance vs Sample Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot training time
        ax2.plot(self.results["sample_sizes"], self.results["training_times"], 
                 'g-^', label='Training Time', markersize=8)
        ax2.set_xlabel('Training Sample Size')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Training Time vs Sample Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.suptitle(f'Sample Size Optimization: {self.model_name} on {self.task_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def optimize_sample_size(
    model_name: str,
    task_name: str,
    layer: int,
    method_type: str = "classification",
    sample_sizes: Optional[List[int]] = None,
    test_size: int = 200,
    seed: int = 42,
    verbose: bool = False,
    save_plot: bool = False,
    save_to_config: bool = True,
    **method_kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run sample size optimization.
    
    Args:
        model_name: Model to optimize
        task_name: Task to optimize for
        layer: Layer to use
        method_type: "classification" or "steering"
        sample_sizes: Sample sizes to test
        test_size: Fixed test set size
        seed: Random seed
        verbose: Verbose output
        save_plot: Whether to save the plot
        save_to_config: Whether to save results to model config
        **method_kwargs: Method-specific arguments
        
    Returns:
        Optimization results
    """
    optimizer = SimplifiedSampleSizeOptimizer(
        model_name=model_name,
        task_name=task_name,
        layer=layer,
        method_type=method_type,
        sample_sizes=sample_sizes,
        test_size=test_size,
        seed=seed,
        verbose=verbose,
        **method_kwargs
    )
    
    results = optimizer.run_optimization()
    
    # Create plot if requested
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = f"sample_size_optimization/{model_name}"
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(
            plot_dir, 
            f"{task_name}_{method_type}_layer{layer}_{timestamp}.png"
        )
        optimizer.plot_results(plot_path)
    
    # Save to config if requested
    if save_to_config and results["optimal_sample_size"] > 0:
        try:
            config_manager = ModelConfigManager()
            
            # For now, just log the optimal sample size
            # TODO: Implement save_optimal_sample_size in ModelConfigManager
            logger.info(
                f"Optimal {method_type} sample size for {model_name} on {task_name}: "
                f"{results['optimal_sample_size']} (accuracy: {results.get('optimal_accuracy', 'N/A')})"
            )
            
            if verbose:
                print(f"\n💡 Note: To use this optimal sample size, add --limit {results['optimal_sample_size']} to your commands")
        except Exception as e:
            logger.warning(f"Could not save to config: {e}")
    
    return results