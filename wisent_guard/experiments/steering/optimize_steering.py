#!/usr/bin/env python3
"""
Steering Parameter Optimization

This script systematically searches through all steering parameters to find
the optimal combination for maximizing steering accuracy on a given task.

Parameters optimized:
- Layer selection (which transformer layer to apply steering)
- Steering method (CAA, HPR, KSteering, etc.)
- Method-specific parameters (strengths, betas, alphas, etc.)
- Training data split sizes
- Other hyperparameters

Usage:
    python optimize_steering.py --task truthfulqa_mc1 --limit 100 --optimization-budget 50
"""

import argparse
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from itertools import product
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SteeringConfig:
    """Configuration for a steering experiment."""
    layer: int
    method: str
    strength: float
    
    # Method-specific parameters
    hpr_beta: Optional[float] = None
    ksteering_alpha: Optional[float] = None
    ksteering_target_labels: Optional[str] = None
    ksteering_avoid_labels: Optional[str] = None
    
    # Training parameters
    split_ratio: float = 0.8
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class OptimizationResult:
    """Result of a steering optimization run."""
    config: SteeringConfig
    accuracy: float
    likelihood_change: float
    runtime_seconds: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'config': self.config.to_dict(),
            'accuracy': self.accuracy,
            'likelihood_change': self.likelihood_change,
            'runtime_seconds': self.runtime_seconds
        }
        if self.error:
            result['error'] = self.error
        return result

class SteeringOptimizer:
    """Optimizes steering parameters for maximum accuracy."""
    
    def __init__(self, task_name: str, model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 limit: int = 100, device: str = "cuda", verbose: bool = True, 
                 custom_layer_range: Optional[str] = None):
        self.task_name = task_name
        self.model_name = model_name
        self.limit = limit
        self.device = device
        self.verbose = verbose
        self.custom_layer_range = custom_layer_range
        self.results: List[OptimizationResult] = []
        
        # Get model info to determine layer count (only if no custom range specified)
        if not custom_layer_range:
            self.num_layers = self._get_model_layer_count()
        else:
            self.num_layers = None  # Will be set based on custom range
        
        # Define search spaces for different parameters
        self.search_spaces = self._define_search_spaces()
    
    def _get_model_layer_count(self) -> int:
        """Get the number of layers in the model using existing detection logic."""
        try:
            from wisent_guard.core.model import Model
            from wisent_guard.core.hyperparameter_optimizer import detect_model_layers
            
            if self.verbose:
                logger.info(f"Loading model {self.model_name} to determine layer count...")
            
            model = Model(self.model_name, device=self.device)
            num_layers = detect_model_layers(model)
            
            # Clean up model to save memory  
            del model
            
            if self.verbose:
                logger.info(f"Model has {num_layers} layers")
            
            return num_layers
            
        except Exception as e:
            logger.error(f"Failed to detect layer count for {self.model_name}: {e}")
            raise RuntimeError(f"Could not determine layer count for model {self.model_name}. "
                             f"Use --layer-range to specify layers manually.") from e
        
    def _define_search_spaces(self) -> Dict[str, Any]:
        """Define search spaces for all parameters."""
        
        # Handle custom layer range
        if self.custom_layer_range:
            layers = self._parse_layer_range(self.custom_layer_range)
            if self.verbose:
                logger.info(f"Using custom layer range: {layers}")
        else:
            # Search all layers by default - let optimization discover what works best
            layers = list(range(1, self.num_layers + 1))
            
            if self.verbose:
                logger.info(f"Searching all layers 1-{self.num_layers} ({self.num_layers} total layers)")
        
        return {
            'layers': layers,  # Dynamic based on model architecture
            'methods': ['CAA', 'HPR', 'KSteering'],
            'strengths': {
                'CAA': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                'HPR': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                'KSteering': [0.1, 0.2, 0.5, 1.0, 2.0]
            },
            'method_params': {
                'HPR': {
                    'hpr_beta': [0.1, 0.5, 1.0, 1.5, 2.0]
                },
                'KSteering': {
                    'ksteering_alpha': [0.5, 1.0, 2.0, 5.0, 10.0],
                    'ksteering_target_labels': ['0', '1'],
                    'ksteering_avoid_labels': ['', '0', '1']
                }
            },
            'split_ratios': [0.7, 0.8, 0.9],
            'seeds': [42, 123, 456]  # Multiple seeds for robustness
        }
    
    def _parse_layer_range(self, layer_range: str) -> List[int]:
        """Parse custom layer range string."""
        try:
            if '-' in layer_range:
                # Range format: "8-24"
                start, end = map(int, layer_range.split('-'))
                return list(range(start, end + 1))
            elif ',' in layer_range:
                # Comma-separated format: "10,15,20"
                return [int(x.strip()) for x in layer_range.split(',')]
            else:
                # Single layer: "15"
                return [int(layer_range)]
        except ValueError as e:
            logger.error(f"Invalid layer range format: {layer_range}. Use format like '8-24' or '10,15,20'")
            raise e
    
    def generate_grid_search_configs(self) -> List[SteeringConfig]:
        """Generate all configurations for grid search."""
        configs = []
        
        for layer in self.search_spaces['layers']:
            for method in self.search_spaces['methods']:
                for strength in self.search_spaces['strengths'][method]:
                    for split_ratio in self.search_spaces['split_ratios']:
                        for seed in self.search_spaces['seeds']:
                            
                            base_config = SteeringConfig(
                                layer=layer,
                                method=method,
                                strength=strength,
                                split_ratio=split_ratio,
                                seed=seed
                            )
                            
                            # Add method-specific parameters
                            if method in self.search_spaces['method_params']:
                                method_params = self.search_spaces['method_params'][method]
                                
                                if method == 'HPR':
                                    for beta in method_params['hpr_beta']:
                                        config = SteeringConfig(
                                            layer=layer, method=method, strength=strength,
                                            split_ratio=split_ratio, seed=seed, hpr_beta=beta
                                        )
                                        configs.append(config)
                                
                                elif method == 'KSteering':
                                    param_combinations = product(
                                        method_params['ksteering_alpha'],
                                        method_params['ksteering_target_labels'],
                                        method_params['ksteering_avoid_labels']
                                    )
                                    for alpha, target, avoid in param_combinations:
                                        config = SteeringConfig(
                                            layer=layer, method=method, strength=strength,
                                            split_ratio=split_ratio, seed=seed,
                                            ksteering_alpha=alpha,
                                            ksteering_target_labels=target,
                                            ksteering_avoid_labels=avoid
                                        )
                                        configs.append(config)
                            else:
                                configs.append(base_config)
        
        return configs
    
    def generate_random_search_configs(self, num_configs: int) -> List[SteeringConfig]:
        """Generate random configurations for random search."""
        configs = []
        
        for _ in range(num_configs):
            layer = random.choice(self.search_spaces['layers'])
            method = random.choice(self.search_spaces['methods'])
            strength = random.choice(self.search_spaces['strengths'][method])
            split_ratio = random.choice(self.search_spaces['split_ratios'])
            seed = random.choice(self.search_spaces['seeds'])
            
            config = SteeringConfig(
                layer=layer,
                method=method,
                strength=strength,
                split_ratio=split_ratio,
                seed=seed
            )
            
            # Add method-specific parameters
            if method in self.search_spaces['method_params']:
                if method == 'HPR':
                    config.hpr_beta = random.choice(self.search_spaces['method_params']['HPR']['hpr_beta'])
                elif method == 'KSteering':
                    params = self.search_spaces['method_params']['KSteering']
                    config.ksteering_alpha = random.choice(params['ksteering_alpha'])
                    config.ksteering_target_labels = random.choice(params['ksteering_target_labels'])
                    config.ksteering_avoid_labels = random.choice(params['ksteering_avoid_labels'])
            
            configs.append(config)
        
        return configs
    
    def generate_smart_search_configs(self, num_configs: int) -> List[SteeringConfig]:
        """Generate configurations using smart heuristics."""
        configs = []
        
        # For smart search, we don't assume which layers are "promising" - sample uniformly
        all_layers = self.search_spaces['layers']
        
        # Prioritize certain method-strength combinations
        promising_combinations = [
            ('CAA', [0.5, 1.0, 2.0]),
            ('HPR', [1.0, 1.5, 2.0]),
            ('KSteering', [0.5, 1.0])
        ]
        
        # Generate configurations 
        for _ in range(num_configs):
            # Sample layers uniformly - let optimization discover what works
            layer = random.choice(all_layers)
            
            # Choose method and strength with bias toward promising combinations
            if random.random() < 0.8:
                method, strengths = random.choice(promising_combinations)
                strength = random.choice(strengths)
            else:
                method = random.choice(self.search_spaces['methods'])
                strength = random.choice(self.search_spaces['strengths'][method])
            
            config = SteeringConfig(
                layer=layer,
                method=method,
                strength=strength,
                split_ratio=random.choice([0.8, 0.9]),  # Prefer higher split ratios
                seed=random.choice(self.search_spaces['seeds'])
            )
            
            # Add method-specific parameters with smart defaults
            if method == 'HPR':
                config.hpr_beta = random.choice([0.5, 1.0, 1.5])  # Most promising betas
            elif method == 'KSteering':
                config.ksteering_alpha = random.choice([1.0, 2.0, 5.0])
                config.ksteering_target_labels = random.choice(['0', '1'])
                config.ksteering_avoid_labels = random.choice(['', '0'])
            
            configs.append(config)
        
        return configs
    
    def evaluate_config(self, config: SteeringConfig) -> OptimizationResult:
        """Evaluate a single configuration."""
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from wisent_guard.cli import run_task_pipeline
            
            # Convert config to CLI arguments
            cli_args = {
                'task_name': self.task_name,
                'model_name': self.model_name,
                'layer': str(config.layer),
                'limit': self.limit,
                'steering_mode': True,
                'steering_method': config.method,
                'steering_strength': config.strength,
                'split_ratio': config.split_ratio,
                'seed': config.seed,
                'device': self.device,
                'verbose': False,
                'allow_small_dataset': True,
                'output_mode': 'likelihoods'  # Faster evaluation
            }
            
            # Add method-specific parameters
            if config.hpr_beta is not None:
                cli_args['hpr_beta'] = config.hpr_beta
            if config.ksteering_alpha is not None:
                cli_args['ksteering_alpha'] = config.ksteering_alpha
            if config.ksteering_target_labels is not None:
                cli_args['ksteering_target_labels'] = config.ksteering_target_labels
            if config.ksteering_avoid_labels is not None:
                cli_args['ksteering_avoid_labels'] = config.ksteering_avoid_labels
            
            # Run the evaluation
            result = run_task_pipeline(**cli_args)
            
            # Extract metrics
            accuracy = result.get('accuracy', 0.0)
            
            # Calculate likelihood change from baseline vs steered likelihoods
            likelihood_change = 0.0
            if 'evaluation_results' in result and 'baseline_likelihoods' in result['evaluation_results'] and 'steered_likelihoods' in result['evaluation_results']:
                baseline = result['evaluation_results']['baseline_likelihoods']
                steered = result['evaluation_results']['steered_likelihoods']
                if baseline and steered and len(baseline) == len(steered):
                    likelihood_change = sum(s - b for s, b in zip(steered, baseline)) / len(baseline)
            
            runtime = time.time() - start_time
            
            return OptimizationResult(
                config=config,
                accuracy=accuracy,
                likelihood_change=likelihood_change,
                runtime_seconds=runtime
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            error_msg = str(e)
            
            if self.verbose:
                logger.error(f"Error evaluating config {config.to_dict()}: {error_msg}")
            
            return OptimizationResult(
                config=config,
                accuracy=0.0,
                likelihood_change=0.0,
                runtime_seconds=runtime,
                error=error_msg
            )
    
    def _estimate_runtime(self, budget: int) -> float:
        """Estimate total runtime by running a quick timing test."""
        if self.verbose:
            print("🔍 Running timing estimation...")
        
        # Run a quick test with minimal examples
        test_config = SteeringConfig(
            layer=15,  # Use a middle layer
            method='CAA',  # Use simplest method
            strength=1.0,
            split_ratio=0.8,
            seed=42
        )
        
        start_time = time.time()
        try:
            # Import here to avoid circular imports
            from wisent_guard.cli import run_task_pipeline
            
            # Run with minimal examples for timing
            cli_args = {
                'task_name': self.task_name,
                'model_name': self.model_name,
                'layer': str(test_config.layer),
                'limit': 5,  # Minimal for timing
                'steering_mode': True,
                'steering_method': test_config.method,
                'steering_strength': test_config.strength,
                'split_ratio': test_config.split_ratio,
                'seed': test_config.seed,
                'device': self.device,
                'verbose': False,
                'allow_small_dataset': True,
                'output_mode': 'likelihoods'
            }
            
            run_task_pipeline(**cli_args)
            test_time = time.time() - start_time
            
            # Scale up: 5 examples -> self.limit examples, 1 config -> budget configs
            scaling_factor = (self.limit / 5.0) * budget
            estimated_total = test_time * scaling_factor
            
            if self.verbose:
                print(f"   Test run (5 examples): {test_time:.1f}s")
                print(f"   Scaling factor: {scaling_factor:.0f}x")
                print(f"   Estimated total: {estimated_total/3600:.1f} hours")
            
            return estimated_total
            
        except Exception as e:
            if self.verbose:
                print(f"   Warning: Could not estimate runtime: {e}")
            return 3600.0  # Default 1 hour estimate
    
    def _confirm_runtime(self, estimated_seconds: float, budget: int) -> bool:
        """Ask user to confirm if they want to proceed with estimated runtime."""
        hours = estimated_seconds / 3600
        
        print(f"\n⏱️  RUNTIME ESTIMATION:")
        print(f"   • Task: {self.task_name}")
        print(f"   • Examples: {self.limit}")
        print(f"   • Configurations to test: {budget}")
        
        if hours < 0.1:
            print(f"   • Estimated time: {estimated_seconds/60:.1f} minutes")
        elif hours < 2:
            print(f"   • Estimated time: {hours:.1f} hours")
        else:
            print(f"   • Estimated time: {hours:.1f} hours ({hours/24:.1f} days)")
        
        if hours > 12:
            print(f"   ⚠️  WARNING: This will take a very long time!")
        
        while True:
            response = input(f"\nProceed with optimization? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                print("Optimization cancelled.")
                return False
            else:
                print("Please enter 'y' or 'n'")

    def optimize(self, search_strategy: str = 'smart', budget: int = 50, confirm_runtime: bool = True) -> List[OptimizationResult]:
        """Run optimization with specified strategy and budget."""
        
        if self.verbose:
            logger.info(f"Starting steering optimization for {self.task_name}")
            logger.info(f"Strategy: {search_strategy}, Budget: {budget} evaluations")
        
        # Estimate runtime and ask for confirmation
        if confirm_runtime and (self.limit > 100 or budget > 5):
            estimated_time = self._estimate_runtime(budget)
            if not self._confirm_runtime(estimated_time, budget):
                return []
        
        # Generate configurations based on strategy
        if search_strategy == 'grid':
            configs = self.generate_grid_search_configs()
            if len(configs) > budget:
                configs = random.sample(configs, budget)
                if self.verbose:
                    logger.info(f"Sampled {budget} configurations from {len(configs)} total grid configs")
        elif search_strategy == 'random':
            configs = self.generate_random_search_configs(budget)
        elif search_strategy == 'smart':
            configs = self.generate_smart_search_configs(budget)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")
        
        if self.verbose:
            logger.info(f"Generated {len(configs)} configurations to evaluate")
        
        # Evaluate all configurations
        results = []
        for i, config in enumerate(configs):
            if self.verbose and (i + 1) % 5 == 0:
                logger.info(f"Evaluating configuration {i + 1}/{len(configs)}")
            
            result = self.evaluate_config(config)
            results.append(result)
            
            if self.verbose and result.error is None:
                logger.info(f"Config {i + 1}: Layer {config.layer}, {config.method}, "
                          f"Strength {config.strength} -> Accuracy: {result.accuracy:.3f}")
        
        # Sort by accuracy (descending)
        results.sort(key=lambda x: x.accuracy, reverse=True)
        
        self.results = results
        return results
    
    def get_best_config(self) -> Optional[OptimizationResult]:
        """Get the best performing configuration."""
        if not self.results:
            return None
        return self.results[0]
    
    def get_top_configs(self, n: int = 10) -> List[OptimizationResult]:
        """Get the top N performing configurations."""
        return self.results[:n]
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results and return insights."""
        if not self.results:
            return {}
        
        # Filter out failed runs
        successful_results = [r for r in self.results if r.error is None and r.accuracy > 0]
        
        if not successful_results:
            return {'error': 'No successful optimization runs'}
        
        analysis = {
            'total_runs': len(self.results),
            'successful_runs': len(successful_results),
            'best_accuracy': successful_results[0].accuracy,
            'mean_accuracy': np.mean([r.accuracy for r in successful_results]),
            'std_accuracy': np.std([r.accuracy for r in successful_results])
        }
        
        # Analyze by method
        method_performance = {}
        for method in self.search_spaces['methods']:
            method_results = [r for r in successful_results if r.config.method == method]
            if method_results:
                method_performance[method] = {
                    'count': len(method_results),
                    'best_accuracy': max(r.accuracy for r in method_results),
                    'mean_accuracy': np.mean([r.accuracy for r in method_results]),
                    'std_accuracy': np.std([r.accuracy for r in method_results])
                }
        
        analysis['method_performance'] = method_performance
        
        # Analyze by layer
        layer_performance = {}
        for layer in self.search_spaces['layers']:
            layer_results = [r for r in successful_results if r.config.layer == layer]
            if layer_results:
                layer_performance[layer] = {
                    'count': len(layer_results),
                    'best_accuracy': max(r.accuracy for r in layer_results),
                    'mean_accuracy': np.mean([r.accuracy for r in layer_results])
                }
        
        # Find best performing layer
        if layer_performance:
            best_layer = max(layer_performance.keys(), 
                           key=lambda l: layer_performance[l]['best_accuracy'])
            analysis['best_layer'] = best_layer
        
        return analysis
    
    def save_results(self, output_dir: str = None):
        """Save optimization results to files."""
        if output_dir is None:
            output_dir = "optimization_results"
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"{self.task_name}_{timestamp}"
        
        # Save full results
        full_results_path = Path(output_dir) / f"{base_name}_full_results.json"
        output_data = {
            'task_name': self.task_name,
            'model_name': self.model_name,
            'limit': self.limit,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'analysis': self.analyze_results(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(full_results_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save best configuration separately for easy reuse
        best_result = self.get_best_config()
        if best_result:
            best_config_path = Path(output_dir) / f"{base_name}_best_config.json"
            best_config_data = {
                'task_name': self.task_name,
                'model_name': self.model_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'best_configuration': best_result.config.to_dict(),
                'performance': {
                    'accuracy': best_result.accuracy,
                    'likelihood_change': best_result.likelihood_change,
                    'runtime_seconds': best_result.runtime_seconds
                },
                'cli_command': self._generate_cli_command(best_result.config)
            }
            
            with open(best_config_path, 'w') as f:
                json.dump(best_config_data, f, indent=2)
            
            if self.verbose:
                logger.info(f"Best config saved to {best_config_path}")
        
        if self.verbose:
            logger.info(f"Full results saved to {full_results_path}")
            
        return str(full_results_path), str(best_config_path) if best_result else None
    
    def _generate_cli_command(self, config: SteeringConfig) -> str:
        """Generate CLI command to reproduce the best configuration."""
        cmd_parts = [
            "python -m wisent_guard.cli tasks",
            self.task_name,
            f"--limit {self.limit}",
            "--steering-mode",
            f"--steering-method {config.method}",
            f"--steering-strength {config.strength}",
            f"--layer {config.layer}",
            f"--split-ratio {config.split_ratio}",
            f"--seed {config.seed}"
        ]
        
        # Add method-specific parameters
        if config.hpr_beta is not None:
            cmd_parts.append(f"--hpr-beta {config.hpr_beta}")
        if config.ksteering_alpha is not None:
            cmd_parts.append(f"--ksteering-alpha {config.ksteering_alpha}")
        if config.ksteering_target_labels is not None:
            cmd_parts.append(f"--ksteering-target-labels {config.ksteering_target_labels}")
        if config.ksteering_avoid_labels is not None:
            cmd_parts.append(f"--ksteering-avoid-labels {config.ksteering_avoid_labels}")
            
        cmd_parts.append("--verbose")
        
        return " ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description='Optimize steering parameters for maximum accuracy')
    parser.add_argument('--task', required=True, help='Task name (e.g., truthfulqa_mc1)')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct', help='Model name')
    parser.add_argument('--limit', type=int, default=100, help='Number of samples per evaluation')
    parser.add_argument('--budget', type=int, default=50, help='Number of configurations to evaluate')
    parser.add_argument('--strategy', choices=['grid', 'random', 'smart'], default='smart',
                       help='Search strategy')
    parser.add_argument('--layer-range', help='Custom layer range (e.g., "8-24" or "10,15,20")')
    parser.add_argument('--output-dir', help='Output directory path (default: optimization_results)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip-confirmation', action='store_true', help='Skip runtime confirmation prompt')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = SteeringOptimizer(
        task_name=args.task,
        model_name=args.model,
        limit=args.limit,
        device=args.device,
        verbose=args.verbose,
        custom_layer_range=getattr(args, 'layer_range')
    )
    
    # Run optimization
    skip_confirmation = getattr(args, 'skip_confirmation', False)
    results = optimizer.optimize(search_strategy=args.strategy, budget=args.budget, confirm_runtime=not skip_confirmation)
    
    # Check if optimization was cancelled
    if not results:
        return
    
    # Print results
    best_result = optimizer.get_best_config()
    if best_result:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Layer: {best_result.config.layer}")
        print(f"   Method: {best_result.config.method}")
        print(f"   Strength: {best_result.config.strength}")
        if best_result.config.hpr_beta:
            print(f"   HPR Beta: {best_result.config.hpr_beta}")
        if best_result.config.ksteering_alpha:
            print(f"   KSteering Alpha: {best_result.config.ksteering_alpha}")
        print(f"   Accuracy: {best_result.accuracy:.3f}")
        print(f"   Likelihood Change: {best_result.likelihood_change:.3f}")
        print(f"   Runtime: {best_result.runtime_seconds:.1f}s")
        
        # Show top 5 configurations
        print(f"\n📊 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(optimizer.get_top_configs(5)):
            if result.error is None:
                print(f"   {i+1}. Layer {result.config.layer}, {result.config.method}, "
                      f"Strength {result.config.strength} -> {result.accuracy:.3f}")
    
    # Print analysis
    analysis = optimizer.analyze_results()
    if 'method_performance' in analysis:
        print(f"\n🔬 METHOD ANALYSIS:")
        for method, perf in analysis['method_performance'].items():
            print(f"   {method}: Best {perf['best_accuracy']:.3f}, "
                  f"Mean {perf['mean_accuracy']:.3f} ± {perf['std_accuracy']:.3f}")
    
    # Save results
    output_dir = getattr(args, 'output_dir', None) or "optimization_results"
    full_results_path, best_config_path = optimizer.save_results(output_dir)
    
    print(f"\n💾 Results saved:")
    print(f"   Full results: {full_results_path}")
    if best_config_path:
        print(f"   Best config: {best_config_path}")
        
        # Show the CLI command to reproduce best result
        best_result = optimizer.get_best_config()
        if best_result:
            print(f"\n🔄 To reproduce best result:")
            print(f"   {optimizer._generate_cli_command(best_result.config)}")

if __name__ == '__main__':
    main()
