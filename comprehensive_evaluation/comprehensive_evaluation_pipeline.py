#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for Wisent Guard

This pipeline properly separates:
1. Benchmark Performance: Model's ability to solve problems  
2. Probe Performance: Probe's ability to detect correctness from activations
3. Steering Optimization: Grid search based on benchmark results (not probe performance)

Key phases:
- Training: Train probes on training set
- Validation: Optimize steering hyperparameters using benchmark performance on validation set
- Testing: Final evaluation of both benchmark and probe performance on test set
"""

import argparse
import json
import logging
import gc
import os
import sys
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

# Set HuggingFace cache to permanent directory
os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/.cache/huggingface/transformers'
os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'

# Create cache directories if they don't exist
os.makedirs('/workspace/.cache/huggingface/transformers', exist_ok=True)
os.makedirs('/workspace/.cache/huggingface/datasets', exist_ok=True)

# Add project root to path for imports
sys.path.append('/workspace/wisent-guard')

# Import math tasks configuration
from wisent_guard.parameters.task_config import MATH_TASKS

# Import task classes for dataset loading
from wisent_guard.core.tasks.math500_task import Math500Task
from wisent_guard.core.tasks.aime_task import AIMETask


@dataclass
class ComprehensiveEvaluationConfig:
    """Configuration for comprehensive evaluation pipeline."""
    
    # Model configuration
    model_name: str = "distilbert/distilgpt2"  # Start with lightweight model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset configuration (fully configurable) - defaults to 100 samples for math tasks
    train_dataset: str = "math500"
    val_dataset: str = "aime2024" 
    test_dataset: str = "aime2025"
    train_limit: int = 100  # Default 100 for math tasks
    val_limit: int = 30     # Adjusted for AIME datasets
    test_limit: int = 30    # Adjusted for AIME datasets
    
    # Probe training configuration
    probe_layers: List[int] = None  # Will default based on model
    probe_c_values: List[float] = None  # Will default to [0.1, 1.0, 10.0]
    
    # Steering configuration for grid search
    steering_methods: List[str] = None  # Will default to ["baseline"]
    steering_layers: List[int] = None   # Will default based on model
    steering_strengths: List[float] = None  # Will default to [1.0]
    
    # Output configuration - using outputs/ directory
    output_dir: str = "outputs/comprehensive_evaluation_results"
    experiment_name: str = "math_comprehensive_evaluation"
    
    # Wandb configuration
    wandb_project: str = "wisent-guard-comprehensive-evaluation"
    wandb_tags: List[str] = None
    wandb_entity: Optional[str] = None
    enable_wandb: bool = True
    
    # Technical configuration
    batch_size: int = 4  # Smaller for distilgpt2
    max_length: int = 128  # Shorter for distilgpt2
    seed: int = 42
    verbose: bool = False
    
    def __post_init__(self):
        """Set defaults based on model choice."""
        # Set layer defaults based on model
        if "distilgpt2" in self.model_name.lower():
            if self.probe_layers is None:
                self.probe_layers = [2, 3, 4, 5]  # distilgpt2 has 6 layers (0-5)
            if self.steering_layers is None:
                self.steering_layers = [3, 4, 5]
        elif "llama" in self.model_name.lower():
            if self.probe_layers is None:
                self.probe_layers = [8, 16, 24, 32]  # Assuming 32-layer model
            if self.steering_layers is None:
                self.steering_layers = [16, 24, 32]
        elif "qwen" in self.model_name.lower():
            if self.probe_layers is None:
                self.probe_layers = [8, 16, 24, 32]  # Assuming similar to other large models
            if self.steering_layers is None:
                self.steering_layers = [16, 24, 32]
        elif "gpt2" in self.model_name.lower():
            if self.probe_layers is None:
                self.probe_layers = [4, 6, 8, 10]  # GPT2 has 12 layers
            if self.steering_layers is None:
                self.steering_layers = [6, 8, 10]
        else:
            # Generic defaults
            if self.probe_layers is None:
                self.probe_layers = [4, 6, 8]
            if self.steering_layers is None:
                self.steering_layers = [6, 8]
        
        # Set other defaults
        if self.probe_c_values is None:
            self.probe_c_values = [0.1, 1.0, 10.0]
        if self.steering_methods is None:
            self.steering_methods = ["baseline"]  # Start with baseline (no steering)
        if self.steering_strengths is None:
            self.steering_strengths = [1.0]  # Start with strength 1.0
        if self.wandb_tags is None:
            self.wandb_tags = ["comprehensive_evaluation", self.train_dataset, self.val_dataset, self.test_dataset]


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
            "config": config.__dict__,
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
                config=self.config.__dict__,
                tags=self.config.wandb_tags
            )
            self.logger.info("Wandb experiment initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.config.enable_wandb = False
    
    def _load_model_once(self):
        """Load model and tokenizer once for all experiments."""
        self.logger.info(f"Loading model {self.config.model_name} (ONCE)...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map={"": 0} if self.device.type == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"✓ Model loaded on {self.device}, GPU memory: {memory_gb:.2f} GB")
    
    def _load_dataset_samples(self, dataset_name: str, limit: int) -> List[Dict]:
        """Load samples from a dataset using proper task implementations."""
        self.logger.info(f"Loading {limit} samples from {dataset_name}...")
        
        try:
            # Check if this is a math task from our configuration
            if dataset_name not in MATH_TASKS:
                raise ValueError(f"Dataset {dataset_name} not found in MATH_TASKS. Available: {list(MATH_TASKS)}")
            
            # Handle AIME tasks
            if dataset_name.lower() in ['aime2024', 'aime2025', 'aime']:
                year_mapping = {
                    'aime2024': '2024',
                    'aime2025': '2025', 
                    'aime': '2025'
                }
                
                year = year_mapping.get(dataset_name.lower(), '2025')
                task = AIMETask(year=year, limit=limit)
                samples = task.load_data(limit=limit)
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_name} via AIMETask")
                return samples
            
            # Handle MATH-500 tasks
            elif dataset_name.lower() in ['math500', 'math', 'hendrycks_math']:
                task = Math500Task(limit=limit)
                samples = task.load_data(limit=limit)
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_name} via Math500Task")
                return samples
            
            # For now, other math tasks fall back to Math500Task
            # TODO: Implement specific task classes for GSM8K, ASDiv, etc.
            else:
                self.logger.warning(f"Using Math500Task as fallback for {dataset_name}")
                task = Math500Task(limit=limit)
                samples = task.load_data(limit=limit)
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_name} via Math500Task (fallback)")
                return samples
                
        except Exception as e:
            self.logger.error(f"Failed to load {dataset_name}: {e}")
            raise
    
    def _extract_activations_with_hook(self, texts: List[str], layer: int) -> np.ndarray:
        """Extract activations from a specific layer using hooks."""
        activations = []
        
        def hook_fn(module, input, output):
            # Handle different output formats (some layers return tuples)
            if isinstance(output, tuple):
                hidden_states = output[0]  # First element is usually hidden states
            else:
                hidden_states = output
            
            # Extract last token activations (typical for causal LM)
            if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
                last_token_acts = hidden_states[:, -1, :].detach().cpu().numpy()
                activations.extend(last_token_acts)
        
        # Register hook
        if hasattr(self.model, 'transformer'):  # GPT-style models
            target_layer = self.model.transformer.h[layer]
        elif hasattr(self.model, 'model'):  # Some other architectures
            target_layer = self.model.model.layers[layer]
        else:
            raise ValueError(f"Unknown model architecture for {self.config.model_name}")
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            # Process texts in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                with torch.no_grad():
                    _ = self.model(**inputs)
        
        finally:
            handle.remove()
        
        return np.array(activations)
    
    def _generate_benchmark_predictions(self, samples: List[Dict]) -> Tuple[List[str], List[str]]:
        """Generate model predictions for benchmark evaluation."""
        predictions = []
        ground_truths = []
        
        for sample in samples:
            # Extract question and answer
            if 'problem' in sample and 'answer' in sample:
                question = sample['problem']
                correct_answer = str(sample['answer'])
            elif 'Problem' in sample and 'Answer' in sample:
                question = sample['Problem'] 
                correct_answer = str(sample['Answer'])
            elif 'question' in sample and 'answer' in sample:
                question = sample['question']
                correct_answer = str(sample['answer'])
            else:
                self.logger.warning(f"Skipping sample with unknown format: {sample.keys()}")
                continue
            
            # Create prompt
            prompt = f"Question: {question}\nAnswer:"
            
            # Generate prediction
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract generated text
            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            generated = generated.strip()
            
            predictions.append(generated)
            ground_truths.append(correct_answer)
        
        return predictions, ground_truths
    
    def _evaluate_benchmark_performance(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate benchmark performance with various metrics."""
        # Simple exact match for now (can be improved with mathematical equivalence)
        exact_matches = [pred.strip().lower() == gt.strip().lower() for pred, gt in zip(predictions, ground_truths)]
        accuracy = np.mean(exact_matches)
        
        return {
            "accuracy": accuracy,
            "total_samples": len(predictions),
            "correct": sum(exact_matches)
        }
    
    def _create_probe_training_data(self, samples: List[Dict], layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create training data for probes: activations -> correctness labels."""
        texts = []
        labels = []
        
        for sample in samples:
            # Extract question and answer
            if 'problem' in sample and 'answer' in sample:
                question = sample['problem']
                correct_answer = str(sample['answer'])
            elif 'Problem' in sample and 'Answer' in sample:
                question = sample['Problem'] 
                correct_answer = str(sample['Answer'])
            elif 'question' in sample and 'answer' in sample:
                question = sample['question']
                correct_answer = str(sample['answer'])
            else:
                continue
            
            # Generate model prediction to create positive/negative examples
            prompt = f"Question: {question}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Create examples with model's actual prediction
            correct_text = f"Question: {question}\nAnswer: {correct_answer}"
            incorrect_text = f"Question: {question}\nAnswer: {generated}"
            
            texts.extend([correct_text, incorrect_text])
            # 1 for correct, 0 for model's prediction (which might be wrong)
            is_correct = generated.strip().lower() == correct_answer.strip().lower()
            labels.extend([1, 1 if is_correct else 0])
        
        # Extract activations
        activations = self._extract_activations_with_hook(texts, layer)
        
        return activations, np.array(labels)
    
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
        self._load_model_once()
        
        # Load all datasets
        self.logger.info("\n📊 Loading datasets...")
        train_samples = self._load_dataset_samples(self.config.train_dataset, self.config.train_limit)
        val_samples = self._load_dataset_samples(self.config.val_dataset, self.config.val_limit)
        test_samples = self._load_dataset_samples(self.config.test_dataset, self.config.test_limit)
        
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
            X_train, y_train = self._create_probe_training_data(train_samples, layer)
            
            layer_results = {}
            for C in self.config.probe_c_values:
                # Train probe
                probe = LogisticRegression(C=C, random_state=self.config.seed, max_iter=1000)
                probe.fit(X_train, y_train)
                
                # Evaluate on training set
                y_pred = probe.predict(X_train)
                y_pred_proba = probe.predict_proba(X_train)[:, 1]
                
                accuracy = accuracy_score(y_train, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_train, y_pred, average='binary')
                
                try:
                    auc = roc_auc_score(y_train, y_pred_proba)
                except:
                    auc = 0.5  # Default for cases where AUC can't be computed
                
                layer_results[f"C_{C}"] = {
                    "accuracy": accuracy,
                    "precision": precision,  
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    "probe": probe  # Store for later use
                }
                
                self.logger.info(f"  Layer {layer}, C={C}: Acc={accuracy:.3f}, AUC={auc:.3f}")
            
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
                    benchmark_metrics = self._evaluate_benchmark_performance(predictions, ground_truths)
                    
                    # 2. For this steering config, find best probe layer + C combination
                    best_probe_config, best_probe_metrics = self._optimize_probe_hyperparams_for_steering(
                        val_samples, steering_config, probe_training_results
                    )
                    
                    # 3. Combine scores (for now, use weighted sum - can be made configurable)
                    # Higher weight on benchmark performance as it's the primary objective
                    benchmark_weight = 0.7
                    probe_weight = 0.3
                    combined_score = (benchmark_weight * benchmark_metrics["accuracy"] + 
                                    probe_weight * best_probe_metrics["auc"])
                    
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
    
    def _optimize_probe_hyperparams_for_steering(self, val_samples: List[Dict], steering_config: Dict, probe_training_results: Dict) -> Tuple[Dict, Dict]:
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
    
    def _evaluate_single_probe_on_validation(self, val_samples: List[Dict], layer: int, probe, steering_config: Dict) -> Dict[str, float]:
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
        
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
        except:
            auc = 0.5  # Default for cases where AUC can't be computed
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }
    
    def _create_probe_validation_data_with_steering(self, val_samples: List[Dict], layer: int, steering_config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Create probe validation data with steering applied."""
        # For now, use the same logic as training data creation
        # TODO: In future, apply actual steering to the activations
        
        if steering_config["method"] == "baseline":
            # No steering applied, use regular probe data creation
            return self._create_probe_training_data(val_samples, layer)
        else:
            # TODO: Implement actual steering in activation extraction
            self.logger.warning(f"Steering method {steering_config['method']} not implemented, using baseline")
            return self._create_probe_training_data(val_samples, layer)
    
    def _generate_benchmark_predictions_with_steering(self, samples: List[Dict], method: str, layer: int, strength: float) -> Tuple[List[str], List[str]]:
        """Generate predictions with steering (placeholder for now)."""
        # For now, just use baseline predictions
        # TODO: Implement actual steering methods
        if method == "baseline":
            return self._generate_benchmark_predictions(samples)
        else:
            self.logger.warning(f"Steering method {method} not implemented yet, using baseline")
            return self._generate_benchmark_predictions(samples)
    
    def _final_test_evaluation(self, test_samples: List[Dict], steering_results: Dict, probe_results: Dict) -> Dict[str, Any]:
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
        base_predictions, ground_truths = self._generate_benchmark_predictions(test_samples)
        base_benchmark_metrics = self._evaluate_benchmark_performance(base_predictions, ground_truths)
        
        # Steered model performance (using best config from validation)
        steered_predictions, _ = self._generate_benchmark_predictions_with_steering(
            test_samples, 
            best_steering_config["method"],
            best_steering_config["layer"], 
            best_steering_config["strength"]
        )
        steered_benchmark_metrics = self._evaluate_benchmark_performance(steered_predictions, ground_truths)
        
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
    
    def _evaluate_optimized_probe_on_test(self, test_samples: List[Dict], layer: int, probe, steered: bool = False, steering_config: Dict = None) -> Dict[str, float]:
        """Evaluate the optimized probe on test data."""
        # Create test probe data
        if steered and steering_config:
            X_test, y_test = self._create_probe_validation_data_with_steering(
                test_samples, layer, steering_config
            )
        else:
            X_test, y_test = self._create_probe_training_data(test_samples, layer)
        
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
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.5  # Default for cases where AUC can't be computed
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "total_samples": len(X_test)
        }
    
    def _evaluate_probe_performance(self, test_samples: List[Dict], probe_results: Dict, steered: bool = False, steering_config: Dict = None) -> Dict[str, Any]:
        """Evaluate probe performance on test set."""
        probe_evaluation = {}
        
        for layer_key, layer_probes in probe_results.items():
            layer = int(layer_key.split('_')[1])
            
            # Create test data for this layer
            if steered and steering_config:
                # TODO: Create test data with steering
                X_test, y_test = self._create_probe_training_data(test_samples, layer)  # Placeholder
            else:
                X_test, y_test = self._create_probe_training_data(test_samples, layer)
            
            layer_evaluation = {}
            for c_key, probe_info in layer_probes.items():
                if 'probe' not in probe_info:
                    continue
                    
                probe = probe_info['probe']
                
                # Evaluate probe
                y_pred = probe.predict(X_test)
                y_pred_proba = probe.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
                
                try:
                    auc = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc = 0.5
                
                layer_evaluation[c_key] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall, 
                    "f1": f1,
                    "auc": auc
                }
            
            probe_evaluation[layer_key] = layer_evaluation
        
        return probe_evaluation
    
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


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Evaluation Pipeline for Math Tasks")
    
    # Model configuration
    parser.add_argument("--model-name", default="distilbert/distilgpt2", 
                       choices=["distilbert/distilgpt2", "gpt2", "/workspace/models/llama31-8b-instruct-hf", "Qwen/Qwen3-8B"],
                       help="Model name")
    
    # Dataset configuration - using math tasks
    math_choices = list(MATH_TASKS)
    parser.add_argument("--train-dataset", default="math500", choices=math_choices, help="Training dataset")
    parser.add_argument("--val-dataset", default="aime2024", choices=math_choices, help="Validation dataset") 
    parser.add_argument("--test-dataset", default="aime2025", choices=math_choices, help="Test dataset")
    parser.add_argument("--train-limit", type=int, default=100, help="Training samples limit (default: 100)")
    parser.add_argument("--val-limit", type=int, default=30, help="Validation samples limit (default: 30)")
    parser.add_argument("--test-limit", type=int, default=30, help="Test samples limit (default: 30)")
    
    # Probe configuration
    parser.add_argument("--probe-layers", nargs="+", type=int, help="Layers for probe training")
    parser.add_argument("--probe-c-values", nargs="+", type=float, help="C values for probe training")
    
    # Steering configuration  
    parser.add_argument("--steering-methods", nargs="+", default=["baseline"], help="Steering methods")
    parser.add_argument("--steering-layers", nargs="+", type=int, help="Layers for steering")
    parser.add_argument("--steering-strengths", nargs="+", type=float, help="Steering strengths")
    
    # Other configuration
    parser.add_argument("--output-dir", default="outputs/comprehensive_evaluation_results", help="Output directory")
    parser.add_argument("--enable-wandb", action="store_true", default=True, help="Enable wandb logging (default: True)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ComprehensiveEvaluationConfig(
        model_name=args.model_name,
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        test_dataset=args.test_dataset,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        probe_layers=args.probe_layers,
        probe_c_values=args.probe_c_values,
        steering_methods=args.steering_methods,
        steering_layers=args.steering_layers, 
        steering_strengths=args.steering_strengths,
        output_dir=args.output_dir,
        enable_wandb=args.enable_wandb,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    pipeline = ComprehensiveEvaluationPipeline(config)
    results = pipeline.run_comprehensive_evaluation()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*80)
    if "test_results" in results:
        test_results = results["test_results"]
        print(f"Base Model Benchmark Accuracy: {test_results['base_model_benchmark_results']['accuracy']:.3f}")
        print(f"Steered Model Benchmark Accuracy: {test_results['steered_model_benchmark_results']['accuracy']:.3f}")
        print(f"")
        print(f"Base Model Probe Performance (AUC): {test_results['base_model_probe_results']['auc']:.3f}")
        print(f"Steered Model Probe Performance (AUC): {test_results['steered_model_probe_results']['auc']:.3f}")
        print(f"")
        print(f"Optimized Steering Config: {test_results['optimized_steering_config']}")
        print(f"Optimized Probe Config: Layer {test_results['optimized_probe_config']['layer']}, C={test_results['optimized_probe_config']['C']}")
        print(f"Validation Combined Score: {test_results['validation_combined_score']:.3f}")


if __name__ == "__main__":
    main()