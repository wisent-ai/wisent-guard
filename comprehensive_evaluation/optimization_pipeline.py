"""
Dataset-Agnostic Optimization Pipeline with Optuna

This script builds a reproducible pipeline that:
1. Trains probes and learns steering vectors on the training split
2. Selects the best layer, probe type, steering method, and hyperparameters on validation split via Optuna
3. Evaluates once on the test split with the single best configuration determined on validation

Key features:
- Optuna-based hyperparameter optimization with pruners
- Activation caching for efficiency
- Configurable datasets for train/val/test splits
- Steering evaluation with model re-forwarding
- Reproducibility bundle generation
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np

import torch
import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
import joblib

# Wisent Guard imports
from wisent_guard.core.evaluation.comprehensive import (
    ComprehensiveEvaluationConfig,
    data_utils,
    metrics
)
from wisent_guard.core.task_interface import get_task
from wisent_guard.core.steering_methods.dac import DAC
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent_guard.core.response import Response

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for dataset-agnostic optimization pipeline."""
    # Model and data
    model_name: str = "PradhyumnaPoralla/gpt2_gsm8k_CLM"  # 5% accuracy on GSM8K
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset configuration (fully configurable)
    train_dataset: str = "hendrycks_math"  # Use hendrycks_math for training
    val_dataset: str = "gsm8k"             # Use GSM8K for validation
    test_dataset: str = "gsm8k"            # Use GSM8K for testing
    
    # Sample limits
    train_limit: int = 500
    val_limit: int = 200
    test_limit: int = 300
    
    # Search space
    layer_search_range: Tuple[int, int] = (6, 12)  # Last 6 blocks for GPT2-like model
    probe_types: List[str] = field(default_factory=lambda: ["logistic_regression"])
    steering_methods: List[str] = field(default_factory=lambda: ["dac", "caa"])
    
    # Optuna configuration
    study_name: str = "optimization_pipeline"
    n_trials: int = 50
    sampler: str = "TPE"  # TPE, Random, CmaEs
    pruner: str = "MedianPruner"  # MedianPruner, SuccessiveHalvingPruner
    
    # Technical parameters
    batch_size: int = 8
    max_length: int = 512
    max_new_tokens: int = 256
    seed: int = 42
    
    # Generation parameters
    temperature: float = 0.0  # 0.0 for deterministic, >0.0 for random sampling
    do_sample: bool = False   # True for sampling, False for greedy/deterministic
    
    # Output configuration
    output_dir: str = "outputs/optimization_pipeline"
    cache_dir: str = "cache/optimization_pipeline"
    
    # Efficiency controls
    max_layers_to_search: int = 6
    early_stopping_patience: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ActivationCache:
    """Efficient activation caching system with proper cache keys."""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _generate_cache_key(self, split: str, layer_id: int, 
                          tokenization_config: Dict[str, Any], 
                          prompt_variant: str = "default") -> str:
        """Generate unique cache key for activations."""
        config_str = json.dumps(tokenization_config, sort_keys=True)
        key_data = f"{split}_{layer_id}_{config_str}_{prompt_variant}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"activations_{cache_key}.pkl"
    
    def has_cached_activations(self, split: str, layer_id: int, 
                             tokenization_config: Dict[str, Any],
                             prompt_variant: str = "default") -> bool:
        """Check if activations are cached."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        return self._get_cache_path(cache_key).exists()
    
    def save_activations(self, activations: np.ndarray, labels: np.ndarray,
                        split: str, layer_id: int, 
                        tokenization_config: Dict[str, Any],
                        prompt_variant: str = "default"):
        """Save activations to cache."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'activations': activations,
            'labels': labels,
            'metadata': {
                'split': split,
                'layer_id': layer_id,
                'tokenization_config': tokenization_config,
                'prompt_variant': prompt_variant,
                'timestamp': datetime.now().isoformat(),
                'shape': activations.shape
            }
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        self.logger.info(f"Cached activations for {split} layer {layer_id}: {activations.shape}")
    
    def load_activations(self, split: str, layer_id: int,
                        tokenization_config: Dict[str, Any],
                        prompt_variant: str = "default") -> Tuple[np.ndarray, np.ndarray]:
        """Load activations from cache."""
        cache_key = self._generate_cache_key(split, layer_id, tokenization_config, prompt_variant)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            raise FileNotFoundError(f"No cached activations found for key: {cache_key}")
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.logger.info(f"Loaded cached activations for {split} layer {layer_id}: {cache_data['activations'].shape}")
        return cache_data['activations'], cache_data['labels']


class OptimizationPipeline:
    """Main optimization pipeline using Optuna for hyperparameter search."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Setup output directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache
        self.cache = ActivationCache(config.cache_dir)
        
        # Model and data storage
        self.model = None
        self.tokenizer = None
        self.train_samples = None
        self.val_samples = None
        self.test_samples = None
        
        # Tokenization config for caching
        self.tokenization_config = {
            'max_length': config.max_length,
            'padding': True,
            'truncation': True,
            'return_tensors': 'pt'
        }
        
    def run_optimization(self) -> Dict[str, Any]:
        """Run the complete optimization pipeline."""
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING OPTIMIZATION PIPELINE WITH OPTUNA")
        self.logger.info("="*80)
        
        # Phase 0: Setup
        self._setup_experiment()
        
        # Phase 1: Create Optuna study
        study = self._create_optuna_study()
        
        # Phase 2: Run optimization
        study.optimize(self._objective_function, n_trials=self.config.n_trials)
        
        # Phase 3: Final evaluation with best configuration
        best_trial = study.best_trial
        final_results = self._final_evaluation(best_trial)
        
        # Phase 4: Save reproducibility bundle
        self._save_reproducibility_bundle(study, final_results)
        
        self.logger.info("✅ Optimization completed successfully!")
        return final_results
    
    def _setup_experiment(self):
        """Setup model, tokenizer, and load datasets."""
        self.logger.info("📊 Setting up experiment...")
        
        # Load model and tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load datasets
        self.train_samples = data_utils.load_dataset_samples(
            self.config.train_dataset, self.config.train_limit
        )
        self.val_samples = data_utils.load_dataset_samples(
            self.config.val_dataset, self.config.val_limit
        )
        self.test_samples = data_utils.load_dataset_samples(
            self.config.test_dataset, self.config.test_limit
        )
        
        self.logger.info(f"Loaded {len(self.train_samples)} train, {len(self.val_samples)} val, {len(self.test_samples)} test samples")
        
        # Pre-cache activations for all layers on all splits
        self._precache_activations()
    
    def _precache_activations(self):
        """Pre-cache activations for all layers and splits to improve efficiency."""
        self.logger.info("🔄 Pre-caching activations for efficiency...")
        
        layer_range = range(self.config.layer_search_range[0], 
                           self.config.layer_search_range[1] + 1)
        
        splits_data = [
            ("train", self.train_samples),
            ("val", self.val_samples),
            ("test", self.test_samples)
        ]
        
        for split_name, samples in splits_data:
            for layer_id in layer_range:
                if not self.cache.has_cached_activations(split_name, layer_id, self.tokenization_config):
                    self.logger.info(f"Caching activations for {split_name} split, layer {layer_id}")
                    
                    # Create probe training data (positive and negative examples)
                    # Use appropriate dataset for each split
                    dataset_name = {
                        "train": self.config.train_dataset,
                        "val": self.config.val_dataset,
                        "test": self.config.test_dataset
                    }[split_name]
                    
                    activations, labels = self._create_probe_data(samples, layer_id, dataset_name)
                    
                    # Cache the activations
                    self.cache.save_activations(
                        activations, labels, split_name, layer_id, self.tokenization_config
                    )
                else:
                    self.logger.info(f"Activations already cached for {split_name} split, layer {layer_id}")
    
    def _create_probe_data(self, samples: List[Dict], layer_id: int, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create contrastive probe training data for a specific layer."""
        # Get task for the specified dataset
        task = get_task(dataset_name)
        extractor = task.get_extractor()
        
        texts = []
        labels = []
        
        for sample in samples:
            # Extract QA pair
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create positive example (correct)
            correct_text = f"{question} {correct_answer}"
            texts.append(correct_text)
            labels.append(1)
            
            # Create negative example (generate incorrect answer)
            # For now, use a simple incorrect answer - in practice you'd want more sophisticated negatives
            incorrect_answer = "42"  # Simple incorrect answer
            if incorrect_answer != correct_answer:
                incorrect_text = f"{question} {incorrect_answer}"
                texts.append(incorrect_text)
                labels.append(0)
        
        # Extract activations
        activations = data_utils.extract_activations_with_hook(
            self.model, self.tokenizer, texts, layer_id,
            self.config.batch_size, self.config.max_length, self.device
        )
        
        return activations, np.array(labels)
    
    def _create_optuna_study(self) -> optuna.Study:
        """Create Optuna study with specified sampler and pruner."""
        self.logger.info("📋 Creating Optuna study...")
        
        # Setup sampler
        if self.config.sampler == "TPE":
            sampler = TPESampler(seed=self.config.seed)
        elif self.config.sampler == "Random":
            sampler = optuna.samplers.RandomSampler(seed=self.config.seed)
        else:
            sampler = TPESampler(seed=self.config.seed)
        
        # Setup pruner
        if self.config.pruner == "MedianPruner":
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == "SuccessiveHalvingPruner":
            pruner = SuccessiveHalvingPruner()
        else:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction="maximize",  # Maximize validation accuracy
            sampler=sampler,
            pruner=pruner
        )
        
        return study
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        try:
            # Sample hyperparameters
            layer_id = trial.suggest_int(
                "layer_id", 
                self.config.layer_search_range[0], 
                self.config.layer_search_range[1]
            )
            
            probe_type = trial.suggest_categorical("probe_type", self.config.probe_types)
            probe_c = trial.suggest_float("probe_c", 0.01, 100.0, log=True)
            
            steering_method = trial.suggest_categorical("steering_method", self.config.steering_methods)
            
            # Method-specific hyperparameters
            # TODO this should be moved to the config. 
            if steering_method == "dac":
                steering_alpha = trial.suggest_float("steering_alpha", 0.1, 5.0)
                entropy_threshold = trial.suggest_float("entropy_threshold", 0.5, 2.0)
                ptop = trial.suggest_float("ptop", 0.2, 0.8)
                max_alpha = trial.suggest_float("max_alpha", 1.0, 5.0)
            elif steering_method == "caa":
                steering_alpha = trial.suggest_float("steering_alpha", 0.1, 5.0)
            
            # Step 1: Train probe on training data using cached activations
            probe_score = self._train_and_evaluate_probe(trial, layer_id, probe_type, probe_c)
            
            # Report intermediate result for pruning
            trial.report(probe_score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Step 2: Train steering method on training data  
            steering_method_instance = self._train_steering_method(
                trial, steering_method, layer_id, locals()
            )
            
            # Step 3: Evaluate steering on validation data (must re-run forward passes)
            validation_accuracy = self._evaluate_steering_on_validation(
                steering_method_instance, steering_method, layer_id, locals()
            )
            
            # Report final result
            trial.report(validation_accuracy, step=1)
            
            return validation_accuracy
            
        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return 0.0
    
    def _train_and_evaluate_probe(self, trial: optuna.Trial, layer_id: int, 
                                 probe_type: str, probe_c: float) -> float:
        """Train probe on training data and evaluate on validation data using cached activations."""
        # Load cached training activations
        X_train, y_train = self.cache.load_activations(
            "train", layer_id, self.tokenization_config
        )
        
        # Train probe
        if probe_type == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            probe = LogisticRegression(C=probe_c, random_state=self.config.seed, max_iter=1000)
            probe.fit(X_train, y_train)
        else:
            raise ValueError(f"Unsupported probe type: {probe_type}")
        
        # Evaluate on validation data using cached activations
        X_val, y_val = self.cache.load_activations(
            "val", layer_id, self.tokenization_config
        )
        
        from sklearn.metrics import roc_auc_score
        y_pred_proba = probe.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        
        # Store probe for later use
        trial.set_user_attr("trained_probe", probe)
        
        return auc_score
    
    def _train_steering_method(self, trial: optuna.Trial, method_name: str, 
                              layer_id: int, hyperparams: Dict[str, Any]) -> Any:
        """Train steering method on training data."""
        # Create contrastive pairs from training data
        contrastive_pairs = self._create_contrastive_pairs(self.train_samples, layer_id, self.config.train_dataset, limit=50)
        
        if method_name == "dac":
            # Create DAC instance
            dac = DAC(
                entropy_threshold=hyperparams["entropy_threshold"],
                ptop=hyperparams["ptop"],
                max_alpha=hyperparams["max_alpha"]
            )
            
            # Train DAC
            dac.train(contrastive_pairs, layer_id)
            return dac
            
        elif method_name == "caa":
            # Create CAA instance
            from wisent_guard.core.steering_methods.caa import CAA
            caa = CAA(device=self.device)
            
            # Train CAA
            caa.train(contrastive_pairs, layer_id)
            return caa
            
        else:
            raise ValueError(f"Unsupported steering method: {method_name}")
    
    def _create_contrastive_pairs(self, samples: List[Dict], layer_id: int, dataset_name: str, limit: int = None) -> ContrastivePairSet:
        """Create contrastive pairs with activations for steering training."""
        contrastive_pairs = []
        task = get_task(dataset_name)
        extractor = task.get_extractor()
        
        samples_to_use = samples[:limit] if limit else samples
        
        for sample in samples_to_use:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if qa_pair:
                positive_response = Response(
                    text=qa_pair['correct_answer'],
                    label=1  # Positive label
                )
                negative_response = Response(
                    text="Wrong answer",  # Simple incorrect answer
                    label=0  # Negative label
                )
                
                pair = ContrastivePair(
                    prompt=qa_pair['formatted_question'],
                    positive_response=positive_response,
                    negative_response=negative_response
                )
                contrastive_pairs.append(pair)
        
        # Create pair set and extract activations
        pair_set = ContrastivePairSet(name=f"{dataset_name}_training", pairs=contrastive_pairs)
        
        # Extract activations for the contrastive pairs
        for pair in pair_set.pairs:
            pos_text = f"{pair.prompt} {pair.positive_response.text}"
            neg_text = f"{pair.prompt} {pair.negative_response.text}"
            
            # Extract activations
            pos_activations = self._extract_single_activation(pos_text, layer_id)
            neg_activations = self._extract_single_activation(neg_text, layer_id)
            
            # Store activations in the responses (as expected by get_activation_pairs)
            pair.positive_response.activations = pos_activations
            pair.negative_response.activations = neg_activations
        
        return pair_set
    
    def _extract_single_activation(self, text: str, layer_id: int) -> torch.Tensor:
        """Extract activation for a single text."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True, 
                               max_length=self.config.max_length).to(self.device)
        
        activations = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Get last token activation
            activations.append(hidden_states[:, -1, :].detach())
        
        # Register hook
        if hasattr(self.model, 'transformer'):
            target_layer = self.model.transformer.h[layer_id]
        elif hasattr(self.model, 'model'):
            target_layer = self.model.model.layers[layer_id]
        else:
            raise ValueError("Unknown model architecture")
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.model(**inputs)
        finally:
            handle.remove()
        
        return activations[0] if activations else torch.zeros(1, self.model.config.hidden_size)
    
    def _evaluate_steering_on_validation(self, steering_instance: Any, method_name: str,
                                       layer_id: int, hyperparams: Dict[str, Any]) -> float:
        """Evaluate steering method on validation data by re-running forward passes."""
        if steering_instance is None:
            return 0.0
        
        # Generate predictions with steering applied
        predictions = []
        ground_truths = []
        
        task = get_task(self.config.val_dataset)
        extractor = task.get_extractor()
        
        for sample in self.val_samples[:20]:  # Limit for efficiency during optimization
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue
                
            question = qa_pair['formatted_question']
            ground_truth = qa_pair['correct_answer']
            
            # Generate with steering (this must re-run forward passes)
            try:
                if method_name == "dac":
                    prediction = self._generate_with_dac_steering(
                        steering_instance, question, hyperparams["steering_alpha"], layer_id
                    )
                elif method_name == "caa":
                    prediction = self._generate_with_caa_steering(
                        steering_instance, question, hyperparams["steering_alpha"], layer_id
                    )
                else:
                    prediction = self._generate_baseline(question)
            except Exception as e:
                self.logger.warning(f"Generation failed: {e}")
                prediction = "Error"
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        
        if not predictions:
            return 0.0
        
        # Evaluate benchmark performance
        benchmark_metrics = metrics.evaluate_benchmark_performance(
            predictions, ground_truths, self.config.val_dataset
        )
        
        return benchmark_metrics.get("accuracy", 0.0)
    
    def _generate_with_dac_steering(self, dac: DAC, question: str, alpha: float, layer_id: int) -> str:
        """Generate response with DAC steering applied."""
        # DAC steering is dynamic and complex - for now use simplified approach
        return self._generate_with_steering_hook(question, dac.steering_vector, layer_id, alpha)
    
    def _generate_with_caa_steering(self, caa, question: str, alpha: float, layer_id: int) -> str:
        """Generate response with CAA steering applied."""
        if not hasattr(caa, 'steering_vector') or caa.steering_vector is None:
            return self._generate_baseline(question)
        
        return self._generate_with_steering_hook(question, caa.steering_vector, layer_id, alpha)
    
    def _generate_with_steering_hook(self, question: str, steering_vector: torch.Tensor, 
                                   layer_id: int, alpha: float) -> str:
        """Generate response with steering vector applied via hook (re-runs forward pass)."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        def steering_hook(module, input, output):
            """Hook that applies steering vector during forward pass."""
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Apply steering to the last token
                hidden_states[:, -1, :] += alpha * steering_vector.to(hidden_states.device)
                return (hidden_states,) + output[1:]
            else:
                hidden_states = output
                hidden_states[:, -1, :] += alpha * steering_vector.to(hidden_states.device)
                return hidden_states
        
        # Register hook on target layer
        if hasattr(self.model, 'transformer'):
            target_layer = self.model.transformer.h[layer_id]
        elif hasattr(self.model, 'model'):
            target_layer = self.model.model.layers[layer_id]
        else:
            raise ValueError("Unknown model architecture")
        
        handle = target_layer.register_forward_hook(steering_hook)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.do_sample else 1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        finally:
            handle.remove()
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _generate_baseline(self, question: str) -> str:
        """Generate baseline response without steering."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _final_evaluation(self, best_trial: optuna.Trial) -> Dict[str, Any]:
        """Run final evaluation on test split with best configuration."""
        self.logger.info("🏆 Running final evaluation with best configuration...")
        
        # Extract best hyperparameters
        best_params = best_trial.params
        layer_id = best_params["layer_id"]
        
        self.logger.info(f"Best configuration: {best_params}")
        
        # Re-train best probe and steering method on training data
        from sklearn.linear_model import LogisticRegression
        
        # Train best probe
        X_train, y_train = self.cache.load_activations("train", layer_id, self.tokenization_config)
        probe = LogisticRegression(C=best_params["probe_c"], random_state=self.config.seed, max_iter=1000)
        probe.fit(X_train, y_train)
        
        # Train best steering method
        steering_instance = self._train_steering_method(best_trial, best_params["steering_method"], 
                                                       layer_id, best_params)
        
        # Generate baseline predictions (no steering)
        self.logger.info("Generating baseline predictions...")
        baseline_predictions, test_ground_truths = self._generate_test_predictions(
            None, None, layer_id, 0.0
        )
        
        # Generate steered predictions
        self.logger.info("Generating steered predictions...")
        steered_predictions, _ = self._generate_test_predictions(
            steering_instance, best_params["steering_method"], layer_id, best_params["steering_alpha"]
        )
        
        # Calculate benchmark metrics
        baseline_benchmark_metrics = metrics.evaluate_benchmark_performance(
            baseline_predictions, test_ground_truths, self.config.test_dataset
        )
        
        steered_benchmark_metrics = metrics.evaluate_benchmark_performance(
            steered_predictions, test_ground_truths, self.config.test_dataset
        )
        
        # Evaluate probe on test data
        X_test, y_test = self.cache.load_activations("test", layer_id, self.tokenization_config)
        test_probe_metrics = self._evaluate_probe_metrics(probe, X_test, y_test)
        
        # Calculate improvement
        accuracy_improvement = (steered_benchmark_metrics.get("accuracy", 0.0) - 
                              baseline_benchmark_metrics.get("accuracy", 0.0))
        
        final_results = {
            "best_trial_params": best_params,
            "best_validation_score": best_trial.value,
            "baseline_benchmark_metrics": baseline_benchmark_metrics,
            "steered_benchmark_metrics": steered_benchmark_metrics,
            "accuracy_improvement": accuracy_improvement,
            "test_probe_metrics": test_probe_metrics,
            "config": self.config.to_dict(),
            "num_test_samples": len(test_ground_truths)
        }
        
        # Log final results
        self.logger.info("="*60)
        self.logger.info("🏆 FINAL TEST RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Baseline accuracy: {baseline_benchmark_metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"Steered accuracy: {steered_benchmark_metrics.get('accuracy', 0.0):.4f}")
        self.logger.info(f"Improvement: {accuracy_improvement:+.4f}")
        self.logger.info(f"Probe AUC: {test_probe_metrics.get('auc', 0.5):.4f}")
        self.logger.info(f"Test samples: {len(test_ground_truths)}")
        self.logger.info("="*60)
        
        return final_results
    
    def _generate_test_predictions(self, steering_instance: Any, method_name: str, 
                                 layer_id: int, alpha: float) -> Tuple[List[str], List[str]]:
        """Generate predictions on test data with or without steering."""
        predictions = []
        ground_truths = []
        
        task = get_task(self.config.test_dataset)
        extractor = task.get_extractor()
        
        for sample in self.test_samples:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                continue
                
            question = qa_pair['formatted_question']
            ground_truth = qa_pair['correct_answer']
            
            # Generate prediction
            try:
                if steering_instance is None:
                    # Baseline generation
                    prediction = self._generate_baseline(question)
                elif method_name == "dac":
                    prediction = self._generate_with_dac_steering(
                        steering_instance, question, alpha, layer_id
                    )
                elif method_name == "caa":
                    prediction = self._generate_with_caa_steering(
                        steering_instance, question, alpha, layer_id
                    )
                else:
                    prediction = self._generate_baseline(question)
            except Exception as e:
                self.logger.warning(f"Generation failed for sample: {e}")
                prediction = "Error"
            
            predictions.append(prediction)
            ground_truths.append(ground_truth)
        
        return predictions, ground_truths
    
    def _evaluate_probe_metrics(self, probe, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate probe metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        y_pred = probe.predict(X_test)
        y_pred_proba = probe.predict_proba(X_test)[:, 1]
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }
    
    def _save_reproducibility_bundle(self, study: optuna.Study, final_results: Dict[str, Any]):
        """Save complete reproducibility bundle."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save Optuna study
        study_path = self.output_dir / f"optuna_study_{timestamp}.db"
        study.study_name = str(study_path)
        
        # Save configuration
        config_path = self.output_dir / f"config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save final results
        results_path = self.output_dir / f"final_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Save best configuration
        best_config = {
            "best_params": study.best_trial.params,
            "best_value": study.best_trial.value,
            "model_name": self.config.model_name,
            "random_seed": self.config.seed,
            "commit_hash": self._get_git_commit_hash(),
            "timestamp": timestamp
        }
        
        best_config_path = self.output_dir / f"best_configuration_{timestamp}.json"
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        # Save study trials summary
        trials_df = study.trials_dataframe()
        trials_path = self.output_dir / f"study_trials_{timestamp}.csv"
        trials_df.to_csv(trials_path, index=False)
        
        self.logger.info(f"💾 Reproducibility bundle saved to: {self.output_dir}")
        self.logger.info(f"📊 Study database: {study_path}")
        self.logger.info(f"⚙️ Configuration: {config_path}")
        self.logger.info(f"🏆 Results: {results_path}")
        self.logger.info(f"🎯 Best config: {best_config_path}")
        
    def _get_git_commit_hash(self) -> Optional[str]:
        """Get current git commit hash for reproducibility."""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def cleanup_memory(self):
        """Clean up GPU/MPS memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clean up device memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        import gc
        gc.collect()


def main():
    """Main entry point for optimization pipeline."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = OptimizationConfig(
        model_name="PradhyumnaPoralla/gpt2_gsm8k_CLM",  # 5% accuracy on GSM8K
        train_limit=100,  # Start small for testing
        val_limit=50,
        test_limit=50,
        n_trials=20,
        layer_search_range=(4, 8)  # Smaller range for testing
    )
    
    # Run optimization
    pipeline = OptimizationPipeline(config)
    try:
        results = pipeline.run_optimization()
        
        print("🎉 Optimization completed!")
        print(f"Best validation score: {results['best_validation_score']:.4f}")
        print(f"Test accuracy: {results['steered_benchmark_metrics']['accuracy']:.4f}")
        print(f"Accuracy improvement: {results['accuracy_improvement']:+.4f}")
        
    finally:
        # Clean up memory
        pipeline.cleanup_memory()
    

if __name__ == "__main__":
    main()