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
from tqdm import tqdm

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
        self.model_wrapper = None  # For wisent-guard Model wrapper
        
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
        
        # Create wisent-guard Model wrapper for DAC
        try:
            from wisent_guard.core.model import Model
            self.model_wrapper = Model(name=self.config.model_name, hf_model=self.model)
        except (ValueError, EOFError) as e:
            # If model format detection fails, create a minimal wrapper
            self.logger.warning(f"Could not create Model wrapper: {e}")
            self.logger.info("Creating minimal model wrapper for DAC")
            # Create a simple object with the attributes DAC needs
            class SimpleModelWrapper:
                def __init__(self, hf_model, tokenizer):
                    self.hf_model = hf_model
                    self.model = hf_model
                    self.tokenizer = tokenizer
                    self.device = hf_model.device
                
                def format_prompt(self, prompt):
                    """Simple prompt formatting - just return the prompt as-is."""
                    return prompt
                
                def generate(self, prompt: str, layer_index: int, max_new_tokens: int = 50, **kwargs):
                    """Simple generate method compatible with SyntheticContrastivePairGenerator."""
                    # Format prompt (simple pass-through)
                    formatted_prompt = self.format_prompt(prompt)
                    
                    # Tokenize
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                    
                    # Generate
                    with torch.no_grad():
                        outputs = self.hf_model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            **kwargs
                        )
                    
                    # Decode response
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_text = response[len(formatted_prompt):].strip()
                    
                    # Return format expected by SyntheticContrastivePairGenerator
                    # (generated_text, activations) - we return None for activations since we don't extract them here
                    return generated_text, None
            
            self.model_wrapper = SimpleModelWrapper(self.model, self.tokenizer)
        
        # Load all datasets
        self.logger.info("\n📊 Loading datasets...")
        train_samples = data_utils.load_dataset_samples(self.config.train_dataset, self.config.train_limit)
        val_samples = data_utils.load_dataset_samples(self.config.val_dataset, self.config.val_limit)
        test_samples = data_utils.load_dataset_samples(self.config.test_dataset, self.config.test_limit)
        
        # Phase 1: Train Probes
        self.logger.info("\n🎯 Phase 1: Training Probes...")
        self.logger.info(f"📊 Configured probe layers: {self.config.probe_layers}")
        self.logger.info(f"⚙️ Configured steering layers: {self.config.steering_layers}")
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
        
        self.logger.info(f"Training probes for {len(self.config.probe_layers)} configured layers: {self.config.probe_layers}")
        
        # Add progress bar for probe training
        probe_progress = tqdm(self.config.probe_layers, desc="🎯 Training probes", unit="layer")
        
        for layer in probe_progress:
            self.logger.info(f"Training probes for layer {layer}...")
            
            # Create training data
            X_train, y_train = data_utils.create_probe_training_data(
                self.model, self.tokenizer, train_samples, layer,
                self.config.batch_size, self.config.max_length, self.device
            )
            
            layer_results = {}
            c_progress = tqdm(self.config.probe_c_values, desc=f"Layer {layer}", leave=False, unit="C")
            for C in c_progress:
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
        
        # Calculate total combinations for progress bar
        total_combinations = (len(self.config.steering_layers) * 
                            len(self.config.steering_strengths) * 
                            len(self.config.dac_entropy_thresholds) * 
                            len(self.config.dac_ptop_values) * 
                            len(self.config.dac_max_alpha_values))
        
        self.logger.info(f"Testing {total_combinations} DAC hyperparameter combinations")
        
        # Create progress bar for grid search
        grid_progress = tqdm(total=total_combinations, desc="🔍 Grid search", unit="config")
        
        # Grid search over all DAC hyperparameter combinations
        for steering_layer in self.config.steering_layers:
            for strength in self.config.steering_strengths:
                for entropy_threshold in self.config.dac_entropy_thresholds:
                    for ptop in self.config.dac_ptop_values:
                        for max_alpha in self.config.dac_max_alpha_values:
                            steering_config = {
                                "method": "dac",
                                "layer": steering_layer, 
                                "strength": strength,
                                "entropy_threshold": entropy_threshold,
                                "ptop": ptop,
                                "max_alpha": max_alpha
                            }
                            
                            self.logger.info(f"Testing DAC config: {steering_config}")
                            
                            # 1. Evaluate benchmark performance with this steering config
                            predictions, ground_truths = self._generate_benchmark_predictions_with_steering(
                                val_samples, steering_config
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
                            
                            # Update progress bar with current best score
                            grid_progress.set_postfix({
                                'best_score': f"{best_combined_score:.3f}",
                                'current': f"{combined_score:.3f}"
                            })
                            grid_progress.update(1)
                            
                            self.logger.info(f"  Benchmark accuracy: {benchmark_metrics['accuracy']:.3f}")
                            self.logger.info(f"  Best probe (layer {best_probe_config['layer']}, C={best_probe_config['C']}): AUC={best_probe_metrics['auc']:.3f}")
                            self.logger.info(f"  Combined score: {combined_score:.3f}")
        
        # Close progress bar
        grid_progress.close()
        
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
        total_probe_combinations = len(self.config.probe_layers) * len(self.config.probe_c_values)
        probe_combinations_tested = 0
        
        for layer in self.config.probe_layers:
            layer_key = f"layer_{layer}"
            if layer_key not in probe_training_results:
                continue
                
            for C in self.config.probe_c_values:
                probe_combinations_tested += 1
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
    
    def _generate_benchmark_predictions_with_steering(self, samples: List[Dict], 
                                                    steering_config: Dict) -> Tuple[List[str], List[str]]:
        """Generate predictions with DAC steering using configured hyperparameters."""
        method = steering_config.get("method", "baseline")
        
        if method == "baseline":
            return data_utils.generate_benchmark_predictions(
                self.model, self.tokenizer, samples,
                self.config.batch_size, self.config.max_length, self.device
            )
        elif method == "dac":
            # Import DAC and related classes
            from wisent_guard.core.steering_methods.dac import DAC
            from wisent_guard.core.benchmark_extractors import GSM8KExtractor
            from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
            from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
            from wisent_guard.core.response import Response
            
            # Create DAC instance with configured hyperparameters
            dac = DAC(
                device=self.device,
                dynamic_control=True,
                entropy_threshold=steering_config.get("entropy_threshold", 1.0),
                ptop=steering_config.get("ptop", 0.4),
                max_alpha=steering_config.get("max_alpha", 2.0)
            )
            
            # Set model reference for KL computation
            dac.set_model_reference(self.model)
            
            try:
                # Use real mathematical training data instead of synthetic generation
                extractor = GSM8KExtractor()
                
                # Create contrastive pairs from actual training samples
                contrastive_pairs = []
                for sample in samples[:10]:  # Use first 10 training samples
                    pair_data = extractor.extract_contrastive_pair(sample)
                    if pair_data:
                        contrastive_pairs.append(pair_data)
                
                self.logger.info(f"Extracted {len(contrastive_pairs)} contrastive pairs from training data")
                
                if not contrastive_pairs:
                    self.logger.warning("No contrastive pairs extracted, falling back to baseline")
                    return data_utils.generate_benchmark_predictions(
                        self.model, self.tokenizer, samples,
                        self.config.batch_size, self.config.max_length, self.device
                    )
                
                # Convert to ContrastivePairSet format
                layer_index = steering_config.get("layer", self.config.steering_layers[0])
                pair_set = self._create_pair_set_from_extracted_pairs(contrastive_pairs, layer_index)
                
                # Train DAC on the real mathematical pairs
                training_result = dac.train(pair_set, layer_index)
                
                if training_result.get("success", False):
                    self.logger.info(f"DAC trained successfully with hyperparameters: "
                                   f"entropy_threshold={steering_config.get('entropy_threshold')}, "
                                   f"ptop={steering_config.get('ptop')}, max_alpha={steering_config.get('max_alpha')}")
                    
                    # Generate predictions with DAC steering
                    return self._generate_dac_steered_predictions(
                        samples, dac, layer_index, steering_config.get("strength", 1.0)
                    )
                else:
                    self.logger.warning("DAC training failed, using baseline predictions")
                    return data_utils.generate_benchmark_predictions(
                        self.model, self.tokenizer, samples,
                        self.config.batch_size, self.config.max_length, self.device
                    )
                    
            except Exception as e:
                self.logger.warning(f"DAC steering failed: {e}. Using baseline predictions.")
                return data_utils.generate_benchmark_predictions(
                    self.model, self.tokenizer, samples,
                    self.config.batch_size, self.config.max_length, self.device
                )
        else:
            self.logger.warning(f"Steering method {method} not implemented, using baseline")
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
            best_steering_config
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
    
    def _create_layer_specific_pair_set(self, pair_set, layer_index):
        """Create a pair set with activations extracted from a specific layer."""
        from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
        
        # Create new pair set for this layer
        layer_pair_set = ContrastivePairSet(
            name=pair_set.name,
            task_type=pair_set.task_type
        )
        
        # Extract activations for this layer and create new pairs
        for pair in pair_set.pairs:
            # Create new pair with layer-specific activations
            new_pair = type(pair)(
                prompt=pair.prompt,
                positive_response=type(pair.positive_response)(
                    text=pair.positive_response.text,
                    activations=self._extract_activations_for_text(pair.positive_response.text, layer_index)
                ),
                negative_response=type(pair.negative_response)(
                    text=pair.negative_response.text,
                    activations=self._extract_activations_for_text(pair.negative_response.text, layer_index)
                )
            )
            layer_pair_set.pairs.append(new_pair)
        
        return layer_pair_set
    
    def _create_pair_set_from_extracted_pairs(self, contrastive_pairs: List[Dict], layer_index: int) -> 'ContrastivePairSet':
        """Convert extracted contrastive pairs to ContrastivePairSet format."""
        from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
        from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
        from wisent_guard.core.response import Response
        
        pair_set = ContrastivePairSet(
            name="math_training_pairs",
            task_type="mathematical_reasoning"
        )
        
        self.logger.info(f"Creating {len(contrastive_pairs)} contrastive pairs for layer {layer_index}")
        
        for pair_data in contrastive_pairs:
            question = pair_data['question']
            correct_answer = pair_data['correct_answer']
            incorrect_answer = pair_data['incorrect_answer']
            
            # Extract activations for correct and incorrect responses
            correct_activations = self._extract_activations_for_text(
                f"{question} {correct_answer}", layer_index
            )
            incorrect_activations = self._extract_activations_for_text(
                f"{question} {incorrect_answer}", layer_index
            )
            
            # Create Response objects
            positive_response = Response(
                text=correct_answer,
                activations=correct_activations
            )
            negative_response = Response(
                text=incorrect_answer,
                activations=incorrect_activations
            )
            
            # Create ContrastivePair
            pair = ContrastivePair(
                prompt=question,
                positive_response=positive_response,
                negative_response=negative_response
            )
            
            pair_set.pairs.append(pair)
        
        self.logger.info(f"Successfully created ContrastivePairSet with {len(pair_set.pairs)} pairs")
        return pair_set
    
    def _extract_activations_for_text(self, text: str, layer_index: int) -> torch.Tensor:
        """Extract activations from a specific layer for given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        activations = []
        def hook(module, input, output):
            activations.append(output[0][:, -1, :].clone())
        
        handle = self.model.model.layers[layer_index].register_forward_hook(hook)
        
        with torch.no_grad():
            self.model(**inputs)
        
        handle.remove()
        return activations[0].squeeze(0)
    
    def _generate_dac_steered_predictions(self, samples: List[Dict], dac, layer_index: int, strength: float) -> Tuple[List[str], List[str]]:
        """Generate predictions using DAC steering."""
        predictions = []
        ground_truths = []
        
        # Add progress bar for prediction generation
        sample_progress = tqdm(samples, desc="📝 Generating predictions", leave=False, unit="sample")
        
        for sample in sample_progress:
            question = sample.get('question', sample.get('prompt', ''))
            answer = sample.get('answer', sample.get('correct_answer', ''))
            ground_truths.append(answer)
            
            # Generate prediction with DAC steering
            def steering_hook(module, input, output):
                hidden_states = output[0]
                last_token = hidden_states[:, -1:, :]
                # DAC uses alpha parameter instead of strength
                steered = dac.apply_steering(last_token, alpha=strength)
                hidden_states[:, -1:, :] = steered
                return (hidden_states,) + output[1:]
            
            handle = self.model.model.layers[layer_index].register_forward_hook(steering_hook)
            
            # Format prompt and generate
            formatted_prompt = question
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs["input_ids"].shape[1] + 30,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            handle.remove()
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = response[len(formatted_prompt):].strip()
            predictions.append(prediction)
        
        return predictions, ground_truths