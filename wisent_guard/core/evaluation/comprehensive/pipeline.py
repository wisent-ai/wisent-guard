"""
Main comprehensive evaluation pipeline.
"""

import json
import logging
import gc
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
import wandb

from .config import ComprehensiveEvaluationConfig
from . import data_utils
from . import metrics


class ComprehensiveEvaluationPipeline:
    """Comprehensive evaluation pipeline separating benchmark and probe performance."""
    
    def __init__(self, config: ComprehensiveEvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Set up output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        
        # Results storage
        self.results = {
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Wandb setup
        self.wandb_run = None
        if config.enable_wandb:
            self._setup_wandb()
    
    def _setup_wandb(self):
        """Initialize wandb experiment."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict(),
                tags=self.config.wandb_tags,
                entity=self.config.wandb_entity
            )
            self.logger.info("Wandb experiment initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.enable_wandb = False
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline."""
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING COMPREHENSIVE EVALUATION PIPELINE")
        self.logger.info("="*80)
        self.logger.info(f"Train: {self.config.train_dataset}")
        self.logger.info(f"Validation: {self.config.val_dataset}")
        self.logger.info(f"Test: {self.config.test_dataset}")
        self.logger.info(f"Model: {self.config.model_name}")
        
        # Load model once
        self.model, self.tokenizer = data_utils.load_model_and_tokenizer(
            self.config.model_name, self.device
        )
        
        # Load all datasets
        self.logger.info("\n📊 Loading datasets...")
        train_samples = data_utils.load_dataset_samples(self.config.train_dataset, self.config.train_limit)
        val_samples = data_utils.load_dataset_samples(self.config.val_dataset, self.config.val_limit)
        test_samples = data_utils.load_dataset_samples(self.config.test_dataset, self.config.test_limit)
        
        # Phase 1: Train Probes
        self.logger.info("\n🎯 Phase 1: Training Probes...")
        probe_training_results = self._train_probes(train_samples)
        self.results["probe_training_results"] = probe_training_results
        
        # Phase 2: Validation - Optimize Steering Based on Benchmark Performance
        self.logger.info("\n🔍 Phase 2: Steering Optimization (Grid Search)...")
        steering_optimization_results = self._optimize_steering_grid_search(val_samples, probe_training_results)
        self.results["steering_optimization_results"] = steering_optimization_results
        
        # Phase 3: Final Test Evaluation
        self.logger.info("\n🏆 Phase 3: Final Test Evaluation...")
        test_results = self._final_test_evaluation(test_samples, steering_optimization_results, probe_training_results)
        self.results["test_results"] = test_results
        
        # Free model memory
        data_utils.free_model_memory(self.model, self.tokenizer)
        
        # Save results
        self._save_results()
        
        if self.config.enable_wandb and self.wandb_run:
            # Create wandb-safe version without probe objects
            wandb_results = self._create_wandb_safe_results()
            self.wandb_run.log(wandb_results)
            self.wandb_run.finish()
        
        return self.results
    
    def _train_probes(self, train_samples: List[Dict]) -> Dict[str, Any]:
        """Train probes on training set to classify correctness."""
        probe_results = {}
        
        for layer in self.config.probe_layers:
            self.logger.info(f"Training probes for layer {layer}...")
            
            # Create training data
            X_train, y_train = data_utils.create_probe_training_data(
                self.model, self.tokenizer, train_samples, layer,
                self.config.batch_size, self.config.max_length, self.device
            )
            
            layer_results = {}
            for C in self.config.probe_c_values:
                # Train probe
                probe = LogisticRegression(C=C, random_state=self.config.seed, max_iter=1000)
                probe.fit(X_train, y_train)
                
                # Evaluate on training set
                y_pred = probe.predict(X_train)
                y_pred_proba = probe.predict_proba(X_train)[:, 1]
                
                probe_metrics = metrics.evaluate_probe_performance(y_train, y_pred, y_pred_proba)
                probe_metrics["probe"] = probe  # Store for later use
                
                layer_results[f"C_{C}"] = probe_metrics
                
                self.logger.info(f"  Layer {layer}, C={C}: Acc={probe_metrics['accuracy']:.3f}, AUC={probe_metrics['auc']:.3f}")
            
            probe_results[f"layer_{layer}"] = layer_results
        
        return probe_results
    
    def _optimize_steering_grid_search(self, val_samples: List[Dict], probe_training_results: Dict) -> Dict[str, Any]:
        """Optimize both steering AND probe hyperparameters using grid search."""
        self.logger.info("Performing comprehensive grid search optimization...")
        self.logger.info("Optimizing: steering configs + probe layer/C selection")
        
        optimization_results = []
        best_config = None
        best_combined_score = -1
        
        # Grid search over all steering combinations
        for method in self.config.steering_methods:
            for steering_layer in self.config.steering_layers:
                for strength in self.config.steering_strengths:
                    steering_config = {
                        "method": method,
                        "layer": steering_layer, 
                        "strength": strength
                    }
                    
                    self.logger.info(f"Testing steering config: {steering_config}")
                    
                    # 1. Evaluate benchmark performance with this steering config
                    predictions, ground_truths = self._generate_benchmark_predictions_with_steering(
                        val_samples, method, steering_layer, strength
                    )
                    benchmark_metrics = metrics.evaluate_benchmark_performance(predictions, ground_truths)
                    
                    # 2. For this steering config, find best probe layer + C combination
                    best_probe_config, best_probe_metrics = self._optimize_probe_hyperparams_for_steering(
                        val_samples, steering_config, probe_training_results
                    )
                    
                    # 3. Calculate combined score
                    combined_score = metrics.calculate_combined_score(
                        benchmark_metrics, best_probe_metrics,
                        self.config.benchmark_weight, self.config.probe_weight
                    )
                    
                    config_result = {
                        "steering_config": steering_config,
                        "best_probe_config": best_probe_config,
                        "benchmark_metrics": benchmark_metrics,
                        "probe_metrics": best_probe_metrics,
                        "combined_score": combined_score
                    }
                    optimization_results.append(config_result)
                    
                    # Track best configuration based on combined score
                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_config = config_result
                    
                    self.logger.info(f"  Benchmark accuracy: {benchmark_metrics['accuracy']:.3f}")
                    self.logger.info(f"  Best probe (layer {best_probe_config['layer']}, C={best_probe_config['C']}): AUC={best_probe_metrics['auc']:.3f}")
                    self.logger.info(f"  Combined score: {combined_score:.3f}")
        
        self.logger.info(f"\n🏆 Best overall config:")
        self.logger.info(f"  Steering: {best_config['steering_config']}")
        self.logger.info(f"  Probe: layer {best_config['best_probe_config']['layer']}, C={best_config['best_probe_config']['C']}")
        self.logger.info(f"  Combined score: {best_combined_score:.3f}")
        
        return {
            "all_configs": optimization_results,
            "best_config": best_config,
            "best_combined_score": best_combined_score
        }
    
    def _optimize_probe_hyperparams_for_steering(self, val_samples: List[Dict], steering_config: Dict, 
                                               probe_training_results: Dict) -> Tuple[Dict, Dict]:
        """Find best probe layer + C combination for a given steering configuration."""
        best_probe_config = None
        best_probe_metrics = None
        best_probe_score = -1
        
        # Test all probe layer + C combinations
        for layer in self.config.probe_layers:
            layer_key = f"layer_{layer}"
            if layer_key not in probe_training_results:
                continue
                
            for C in self.config.probe_c_values:
                c_key = f"C_{C}"
                if c_key not in probe_training_results[layer_key]:
                    continue
                
                # Get the trained probe
                probe_info = probe_training_results[layer_key][c_key]
                if 'probe' not in probe_info:
                    continue
                
                probe = probe_info['probe']
                
                # Evaluate this probe on validation data with current steering config
                probe_metrics = self._evaluate_single_probe_on_validation(
                    val_samples, layer, probe, steering_config
                )
                
                # Use AUC as primary probe performance metric
                probe_score = probe_metrics["auc"]
                
                if probe_score > best_probe_score:
                    best_probe_score = probe_score
                    best_probe_config = {
                        "layer": layer,
                        "C": C
                    }
                    best_probe_metrics = probe_metrics
        
        # Fallback if no valid probe found
        if best_probe_config is None:
            self.logger.warning("No valid probe configuration found, using default")
            best_probe_config = {
                "layer": self.config.probe_layers[0],
                "C": self.config.probe_c_values[0]
            }
            best_probe_metrics = {
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "auc": 0.5
            }
        
        return best_probe_config, best_probe_metrics
    
    def _evaluate_single_probe_on_validation(self, val_samples: List[Dict], layer: int, probe, 
                                           steering_config: Dict) -> Dict[str, float]:
        """Evaluate a single probe on validation data with steering applied."""
        # Create validation probe data with steering applied
        X_val, y_val = self._create_probe_validation_data_with_steering(
            val_samples, layer, steering_config
        )
        
        if len(X_val) == 0:
            # Return default metrics if no data
            return {
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "auc": 0.5
            }
        
        # Evaluate probe
        y_pred = probe.predict(X_val)
        y_pred_proba = probe.predict_proba(X_val)[:, 1]
        
        return metrics.evaluate_probe_performance(y_val, y_pred, y_pred_proba)
    
    def _create_probe_validation_data_with_steering(self, val_samples: List[Dict], layer: int, 
                                                  steering_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create probe validation data with steering applied."""
        # For now, use the same logic as training data creation
        # TODO: In future, apply actual steering to the activations
        
        if steering_config["method"] == "baseline":
            # No steering applied, use regular probe data creation
            return data_utils.create_probe_training_data(
                self.model, self.tokenizer, val_samples, layer,
                self.config.batch_size, self.config.max_length, self.device
            )
        else:
            # TODO: Implement actual steering in activation extraction
            self.logger.warning(f"Steering method {steering_config['method']} not implemented, using baseline")
            return data_utils.create_probe_training_data(
                self.model, self.tokenizer, val_samples, layer,
                self.config.batch_size, self.config.max_length, self.device
            )
    
    def _generate_benchmark_predictions_with_steering(self, samples: List[Dict], method: str, 
                                                    layer: int, strength: float) -> Tuple[List[str], List[str]]:
        """Generate predictions with steering (placeholder for now)."""
        # For now, just use baseline predictions
        # TODO: Implement actual steering methods
        if method == "baseline":
            return data_utils.generate_benchmark_predictions(
                self.model, self.tokenizer, samples,
                self.config.batch_size, self.config.max_length, self.device
            )
        else:
            self.logger.warning(f"Steering method {method} not implemented yet, using baseline")
            return data_utils.generate_benchmark_predictions(
                self.model, self.tokenizer, samples,
                self.config.batch_size, self.config.max_length, self.device
            )
    
    def _final_test_evaluation(self, test_samples: List[Dict], steering_results: Dict, 
                             probe_results: Dict) -> Dict[str, Any]:
        """Final comprehensive evaluation on test set."""
        best_config = steering_results["best_config"]
        best_steering_config = best_config["steering_config"]
        best_probe_config = best_config["best_probe_config"]
        
        self.logger.info(f"Using optimized configurations:")
        self.logger.info(f"  Steering: {best_steering_config}")
        self.logger.info(f"  Probe: layer {best_probe_config['layer']}, C={best_probe_config['C']}")
        
        # 1. Benchmark Performance Evaluation
        self.logger.info("Evaluating benchmark performance...")
        
        # Base model performance
        base_predictions, ground_truths = data_utils.generate_benchmark_predictions(
            self.model, self.tokenizer, test_samples,
            self.config.batch_size, self.config.max_length, self.device
        )
        base_benchmark_metrics = metrics.evaluate_benchmark_performance(base_predictions, ground_truths)
        
        # Steered model performance (using best config from validation)
        steered_predictions, _ = self._generate_benchmark_predictions_with_steering(
            test_samples, 
            best_steering_config["method"],
            best_steering_config["layer"], 
            best_steering_config["strength"]
        )
        steered_benchmark_metrics = metrics.evaluate_benchmark_performance(steered_predictions, ground_truths)
        
        # 2. Probe Performance Evaluation (using optimized probe layer/C)
        self.logger.info("Evaluating probe performance...")
        
        # Get the optimized probe
        layer_key = f"layer_{best_probe_config['layer']}"
        c_key = f"C_{best_probe_config['C']}"
        optimized_probe = probe_results[layer_key][c_key]['probe']
        
        # Evaluate optimized probe on base model activations
        base_probe_metrics = self._evaluate_optimized_probe_on_test(
            test_samples, best_probe_config['layer'], optimized_probe, steered=False
        )
        
        # Evaluate optimized probe on steered model activations
        steered_probe_metrics = self._evaluate_optimized_probe_on_test(
            test_samples, best_probe_config['layer'], optimized_probe, 
            steered=True, steering_config=best_steering_config
        )
        
        return {
            "base_model_benchmark_results": base_benchmark_metrics,
            "steered_model_benchmark_results": steered_benchmark_metrics,
            "base_model_probe_results": base_probe_metrics,
            "steered_model_probe_results": steered_probe_metrics,
            "optimized_steering_config": best_steering_config,
            "optimized_probe_config": best_probe_config,
            "validation_combined_score": best_config["combined_score"]
        }
    
    def _evaluate_optimized_probe_on_test(self, test_samples: List[Dict], layer: int, probe, 
                                        steered: bool = False, steering_config: Dict = None) -> Dict[str, float]:
        """Evaluate the optimized probe on test data."""
        # Create test probe data
        if steered and steering_config:
            X_test, y_test = self._create_probe_validation_data_with_steering(
                test_samples, layer, steering_config
            )
        else:
            X_test, y_test = data_utils.create_probe_training_data(
                self.model, self.tokenizer, test_samples, layer,
                self.config.batch_size, self.config.max_length, self.device
            )
        
        if len(X_test) == 0:
            # Return default metrics if no data
            return {
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "auc": 0.5,
                "total_samples": 0
            }
        
        # Evaluate probe
        y_pred = probe.predict(X_test)
        y_pred_proba = probe.predict_proba(X_test)[:, 1]
        
        probe_metrics = metrics.evaluate_probe_performance(y_test, y_pred, y_pred_proba)
        probe_metrics["total_samples"] = len(X_test)
        
        return probe_metrics
    
    def _create_wandb_safe_results(self) -> Dict[str, Any]:
        """Create wandb-safe version of results without non-serializable objects."""
        import copy
        
        def remove_probes(obj):
            if isinstance(obj, dict):
                return {k: remove_probes(v) for k, v in obj.items() if k != 'probe'}
            elif isinstance(obj, list):
                return [remove_probes(item) for item in obj]
            else:
                return obj
        
        return remove_probes(copy.deepcopy(self.results))
    
    def _save_results(self):
        """Save results to JSON file."""
        # Remove probe objects before saving (not JSON serializable)
        results_to_save = json.loads(json.dumps(self.results, default=str))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"comprehensive_evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        self.logger.info(f"📁 Results saved to: {filename}")
        
        return filename