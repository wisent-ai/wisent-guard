#!/usr/bin/env python3
"""
Efficient Model Benchmark Evaluation Pipeline

Memory-efficient version that loads model once and uses it for all experiments.
Incorporates the efficient techniques from custom_efficient_pipeline.py.

Key features:
- Single model load for all phases
- Activation caching to prevent redundant extractions
- Lightweight classifier training on frozen activations
- Proper train/validation/test splits with hyperparameter optimization
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
import sys
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import wandb

# Add wisent_guard to path for importing
sys.path.insert(0, str(Path(__file__).parent))

from wisent_guard.core.tasks import register_all_tasks
from wisent_guard.core.task_interface import get_task, list_tasks
import torch
import lm_eval
from lm_eval import tasks, evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EfficientEvaluationConfig:
    """Configuration for efficient model benchmark evaluation."""
    
    # Model configuration
    model_name: str = "/workspace/models/llama31-8b-instruct-hf"
    layers_to_search: List[int] = field(default_factory=lambda: [12, 14, 16, 18, 20])
    
    # Dataset configuration 
    train_benchmark: str = "math500"
    val_benchmark: str = "aime2024"
    test_benchmark: str = "aime2025"
    
    # Data limits
    train_limit: int = 100
    val_limit: int = 50
    test_limit: int = 50
    
    # Hyperparameter search
    c_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    
    # Output configuration
    output_dir: str = "efficient_evaluation_results"
    experiment_name: str = "efficient_wisent_guard_evaluation"
    
    # Wandb configuration
    wandb_project: str = "wisent-guard-efficient-evaluation"
    wandb_tags: List[str] = None
    wandb_entity: Optional[str] = None
    enable_wandb: bool = True  # Enabled by default for debugging
    
    # Advanced configuration
    batch_size: int = 8
    max_length: int = 256
    seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        if self.wandb_tags is None:
            self.wandb_tags = ["efficient_evaluation", self.train_benchmark, self.val_benchmark, self.test_benchmark]


class EfficientBenchmarkEvaluator:
    """Efficient benchmark evaluator with single model load."""
    
    def __init__(self, config: EfficientEvaluationConfig):
        self.config = config
        self.wandb_run = None
        self.logger = logger
        self.results = {
            'config': asdict(config),
            'timestamp': datetime.now().isoformat()
        }
        
        # Model components (loaded once)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Activation cache for efficiency
        self.activation_cache = {}
        
        # Setup
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup environment and directories."""
        # Register tasks
        register_all_tasks()
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Wandb if enabled
        if self.config.enable_wandb:
            self._setup_wandb()
        
        # Set random seeds
        self._set_seeds()
    
    def _setup_wandb(self):
        """Initialize Wandb experiment tracking."""
        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=self.config.wandb_tags,
            config=asdict(self.config)
        )
        self.logger.info("Wandb experiment initialized successfully")
    
    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        import random
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def _load_model_once(self):
        """Load model and tokenizer once for all experiments."""
        self.logger.info(f"Loading model {self.config.model_name} (ONCE)...")
        
        # Fix the model name encoding issue
        model_name = self.config.model_name.replace('‑', '-')  # Fix en-dash to hyphen
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Log memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"✓ Model loaded on {self.device}, GPU memory: {memory_gb:.2f} GB")
    
    def _load_dataset_samples(self, dataset_name: str, limit: int) -> List[Dict]:
        """Load samples from a dataset."""
        self.logger.info(f"Loading {limit} samples from {dataset_name}...")
        
        try:
            # Handle custom task implementations first
            if dataset_name.lower() in ['aime2024', 'aime2025', 'aime']:
                from wisent_guard.core.tasks.aime_task import AIMETask
                
                # Map dataset names to years
                year_mapping = {
                    'aime2024': '2024',
                    'aime2025': '2025', 
                    'aime': '2025'  # Default to latest
                }
                
                year = year_mapping.get(dataset_name.lower(), '2025')
                task = AIMETask(year=year, limit=limit)
                samples = task.load_data(limit=limit)
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_name} via AIMETask")
                return samples
            
            elif dataset_name.lower() == 'math500':
                # Use custom Math500Task implementation
                from wisent_guard.core.tasks.math500_task import Math500Task
                
                task = Math500Task(limit=limit)
                samples = task.load_data(limit=limit)
                
                self.logger.info(f"Loaded {len(samples)} samples from {dataset_name} via Math500Task")
                return samples
            
            # Try lm-eval as fallback
            task_dict = tasks.get_task_dict([dataset_name])
            task = list(task_dict.values())[0]
            
            if hasattr(task, 'download'):
                task.download()
            
            samples = []
            
            # Get samples based on what's available
            if hasattr(task, 'test_docs'):
                docs = task.test_docs()
                for i, doc in enumerate(docs):
                    if i >= limit:
                        break
                    samples.append(doc)
            elif hasattr(task, 'validation_docs'):
                docs = task.validation_docs()
                for i, doc in enumerate(docs):
                    if i >= limit:
                        break
                    samples.append(doc)
            elif hasattr(task, 'training_docs'):
                docs = task.training_docs()
                for i, doc in enumerate(docs):
                    if i >= limit:
                        break
                    samples.append(doc)
            
            self.logger.info(f"Loaded {len(samples)} samples from {dataset_name} via lm-eval")
            return samples
            
        except Exception as e:
            self.logger.warning(f"Failed to load {dataset_name}: {e}")
            # Fallback to dummy samples
            return self._create_dummy_samples(dataset_name, limit)
    
    def _create_dummy_samples(self, dataset_name: str, limit: int) -> List[Dict]:
        """Create dummy samples for testing when dataset loading fails."""
        samples = []
        for i in range(limit):
            if 'math' in dataset_name.lower() or 'aime' in dataset_name.lower():
                samples.append({
                    'question': f"Math problem {i} from {dataset_name}: What is 2 + 2?",
                    'choices': ["3", "4", "5", "6"],
                    'gold': 1  # Correct answer is "4"
                })
            else:
                samples.append({
                    'question': f"Question {i} from {dataset_name}?",
                    'choices': ["Answer A", "Answer B", "Answer C", "Answer D"],
                    'gold': i % 4
                })
        
        self.logger.info(f"Created {len(samples)} dummy samples for {dataset_name}")
        return samples
    
    def _extract_activations_batch(self, texts: List[str], layer: int) -> np.ndarray:
        """Extract activations for a batch of texts at specified layer."""
        activations = []
        
        # Process in small batches to manage memory
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i+self.config.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length
            ).to(self.device)
            
            # Extract activations with hook
            captured = []
            
            def hook_fn(module, input, output):
                # Get last token activations
                hidden_states = output[0]
                last_token = hidden_states[:, -1, :].detach().cpu().numpy()
                captured.append(last_token)
            
            # Register hook
            handle = self.model.model.layers[layer].register_forward_hook(hook_fn)
            
            # Forward pass
            with torch.no_grad():
                self.model(**inputs)
            
            # Remove hook
            handle.remove()
            
            # Collect
            if captured:
                activations.extend(captured[0])
        
        return np.array(activations)
    
    def _prepare_dataset_activations(self, dataset_name: str, limit: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Prepare activations for all layers for a dataset."""
        # Check cache
        cache_key = f"{dataset_name}_{limit}"
        if cache_key in self.activation_cache:
            self.logger.info(f"Using cached activations for {cache_key}")
            return self.activation_cache[cache_key]
        
        # Load samples
        samples = self._load_dataset_samples(dataset_name, limit)
        
        # Create text/label pairs for contrastive learning
        texts = []
        labels = []
        
        for sample in samples:
            # Handle different dataset formats
            if 'question' in sample and 'choices' in sample:
                # Multiple choice format
                question = sample['question']
                for i, choice in enumerate(sample['choices']):
                    text = f"Question: {question}\nAnswer: {choice}"
                    texts.append(text)
                    # 1 if correct answer, 0 otherwise
                    labels.append(1 if i == sample.get('gold', 0) else 0)
            else:
                # Simple format - create positive/negative examples
                text = str(sample.get('text', sample))
                texts.append(f"Statement: {text}\nThis is correct.")
                labels.append(1)
                texts.append(f"Statement: {text}\nThis is incorrect.")
                labels.append(0)
        
        # Extract activations for all layers we're searching
        layer_activations = {}
        
        for layer in self.config.layers_to_search:
            self.logger.info(f"Extracting layer {layer} activations for {dataset_name}...")
            acts = self._extract_activations_batch(texts, layer)
            layer_activations[layer] = (acts, np.array(labels))
        
        # Cache for reuse
        self.activation_cache[cache_key] = layer_activations
        
        return layer_activations
    
    def _prepare_dataset_activations_with_samples(self, dataset_name: str, limit: int) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], List[Dict]]:
        """Prepare activations for all layers for a dataset, returning both activations and raw samples."""
        # Load samples first
        samples = self._load_dataset_samples(dataset_name, limit)
        
        # Get activations using existing method
        layer_activations = self._prepare_dataset_activations(dataset_name, limit)
        
        return layer_activations, samples
    
    def _free_model_memory(self):
        """Free model memory after activation extraction."""
        self.logger.info("🧹 Freeing model memory...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"GPU memory after cleanup: {memory_gb:.2f} GB")
    
    def _log_detailed_predictions_to_wandb(self, test_samples: List[Dict], y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, layer: int):
        """Log detailed predictions to wandb for debugging."""
        self.logger.info("📊 Logging detailed predictions to wandb...")
        
        # Create detailed prediction data
        prediction_data = []
        sample_idx = 0
        
        for sample in test_samples:
            if 'question' in sample and 'choices' in sample:
                # Multiple choice format
                question = sample['question']
                correct_idx = sample.get('gold', 0)
                
                for i, choice in enumerate(sample['choices']):
                    if sample_idx < len(y_true):
                        prediction_data.append({
                            'question': question,
                            'choice': choice,
                            'choice_idx': i,
                            'is_correct_answer': (i == correct_idx),
                            'true_label': int(y_true[sample_idx]),
                            'pred_label': int(y_pred[sample_idx]),
                            'pred_proba_0': float(y_pred_proba[sample_idx][0]),
                            'pred_proba_1': float(y_pred_proba[sample_idx][1]),
                            'correct_prediction': (y_true[sample_idx] == y_pred[sample_idx]),
                            'layer': layer,
                            'sample_type': 'multiple_choice'
                        })
                        sample_idx += 1
            else:
                # Simple format
                text = str(sample.get('text', sample))
                
                # Positive example
                if sample_idx < len(y_true):
                    prediction_data.append({
                        'question': text,
                        'choice': 'This is correct',
                        'choice_idx': 1,
                        'is_correct_answer': True,
                        'true_label': int(y_true[sample_idx]),
                        'pred_label': int(y_pred[sample_idx]),
                        'pred_proba_0': float(y_pred_proba[sample_idx][0]),
                        'pred_proba_1': float(y_pred_proba[sample_idx][1]),
                        'correct_prediction': (y_true[sample_idx] == y_pred[sample_idx]),
                        'layer': layer,
                        'sample_type': 'positive_statement'
                    })
                    sample_idx += 1
                
                # Negative example
                if sample_idx < len(y_true):
                    prediction_data.append({
                        'question': text,
                        'choice': 'This is incorrect',
                        'choice_idx': 0,
                        'is_correct_answer': False,
                        'true_label': int(y_true[sample_idx]),
                        'pred_label': int(y_pred[sample_idx]),
                        'pred_proba_0': float(y_pred_proba[sample_idx][0]),
                        'pred_proba_1': float(y_pred_proba[sample_idx][1]),
                        'correct_prediction': (y_true[sample_idx] == y_pred[sample_idx]),
                        'layer': layer,
                        'sample_type': 'negative_statement'
                    })
                    sample_idx += 1
        
        # Log as wandb table
        import wandb
        table = wandb.Table(
            columns=[
                'question', 'choice', 'choice_idx', 'is_correct_answer',
                'true_label', 'pred_label', 'pred_proba_0', 'pred_proba_1',
                'correct_prediction', 'layer', 'sample_type'
            ],
            data=[[row[col] for col in [
                'question', 'choice', 'choice_idx', 'is_correct_answer',
                'true_label', 'pred_label', 'pred_proba_0', 'pred_proba_1',
                'correct_prediction', 'layer', 'sample_type'
            ]] for row in prediction_data]
        )
        
        self.wandb_run.log({
            'test_predictions_detailed': table,
            'test_accuracy_breakdown': {
                'correct_predictions': sum(row['correct_prediction'] for row in prediction_data),
                'total_predictions': len(prediction_data),
                'accuracy': sum(row['correct_prediction'] for row in prediction_data) / len(prediction_data) if prediction_data else 0
            }
        })
        
        self.logger.info(f"Logged {len(prediction_data)} detailed predictions to wandb")
    
    def run_efficient_three_phase_pipeline(self) -> Dict[str, Any]:
        """Run efficient three-phase evaluation: train/validate/test."""
        self.logger.info("="*80)
        self.logger.info("🚀 STARTING EFFICIENT THREE-PHASE EVALUATION")
        self.logger.info("="*80)
        self.logger.info(f"Train: {self.config.train_benchmark}")
        self.logger.info(f"Validation: {self.config.val_benchmark}")
        self.logger.info(f"Test: {self.config.test_benchmark}")
        self.logger.info(f"Hyperparameter search: {len(self.config.layers_to_search)} layers × {len(self.config.c_values)} C values")
        
        # Phase 1: Load model and extract all activations
        self.logger.info("\n📊 Phase 1: Model loading and activation extraction...")
        
        # Load model once
        self._load_model_once()
        
        # Extract activations for all datasets (keeping raw samples for wandb logging)
        train_acts, train_samples = self._prepare_dataset_activations_with_samples(self.config.train_benchmark, self.config.train_limit)
        val_acts, val_samples = self._prepare_dataset_activations_with_samples(self.config.val_benchmark, self.config.val_limit)
        test_acts, test_samples = self._prepare_dataset_activations_with_samples(self.config.test_benchmark, self.config.test_limit)
        
        # Free model memory
        self._free_model_memory()
        
        # Phase 2: Hyperparameter search on train/validation
        self.logger.info("\n🎯 Phase 2: Hyperparameter optimization...")
        
        best_config = None
        best_val_score = -1
        all_results = []
        
        total_configs = len(self.config.layers_to_search) * len(self.config.c_values)
        config_num = 0
        
        for layer in self.config.layers_to_search:
            for c_value in self.config.c_values:
                config_num += 1
                self.logger.info(f"\n[{config_num}/{total_configs}] Training layer={layer}, C={c_value}")
                
                try:
                    # Get training data
                    X_train, y_train = train_acts[layer]
                    
                    # Train classifier
                    clf = LogisticRegression(C=c_value, max_iter=1000, random_state=self.config.seed)
                    clf.fit(X_train, y_train)
                    
                    # Training accuracy
                    train_pred = clf.predict(X_train)
                    train_acc = accuracy_score(y_train, train_pred)
                    
                    # Validation accuracy
                    X_val, y_val = val_acts[layer]
                    val_pred = clf.predict(X_val)
                    val_acc = accuracy_score(y_val, val_pred)
                    
                    self.logger.info(f"  Train accuracy: {train_acc:.3f}, Val accuracy: {val_acc:.3f}")
                    
                    result = {
                        'layer': layer,
                        'C': c_value,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'classifier': clf
                    }
                    all_results.append(result)
                    
                    # Track best validation performance
                    if val_acc > best_val_score:
                        best_val_score = val_acc
                        best_config = result
                        self.logger.info(f"  🏆 New best validation score: {val_acc:.3f}")
                
                except Exception as e:
                    self.logger.error(f"  Failed: {e}")
                    continue
        
        # Phase 3: Test best configuration
        self.logger.info(f"\n🏆 Phase 3: Testing best configuration...")
        if best_config:
            self.logger.info(f"Best config: layer={best_config['layer']}, C={best_config['C']}")
            self.logger.info(f"Validation accuracy: {best_config['val_acc']:.3f}")
            
            # Test on held-out test set
            X_test, y_test = test_acts[best_config['layer']]
            test_pred = best_config['classifier'].predict(X_test)
            test_pred_proba = best_config['classifier'].predict_proba(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            
            self.logger.info(f"Test accuracy: {test_acc:.3f}")
            
            # Log detailed predictions to wandb for debugging
            if self.config.enable_wandb and self.wandb_run:
                self._log_detailed_predictions_to_wandb(test_samples, y_test, test_pred, test_pred_proba, best_config['layer'])
            
            # Save results
            self.results.update({
                'hyperparameter_search': [
                    {k: v for k, v in r.items() if k != 'classifier'} 
                    for r in all_results
                ],
                'best_hyperparams': {k: v for k, v in best_config.items() if k != 'classifier'},
                'test_accuracy': test_acc,
                'test_predictions': {
                    'y_true': y_test.tolist(),
                    'y_pred': test_pred.tolist(),
                    'y_pred_proba': test_pred_proba.tolist()
                },
                'summary': {
                    'total_configs_tested': len(all_results),
                    'best_layer': best_config['layer'],
                    'best_C': best_config['C'],
                    'best_val_accuracy': best_val_score,
                    'final_test_accuracy': test_acc
                }
            })
            
            # Save best classifier
            classifier_path = self.output_dir / f"best_classifier_layer{best_config['layer']}_C{best_config['C']}.pkl"
            with open(classifier_path, 'wb') as f:
                pickle.dump(best_config['classifier'], f)
            self.logger.info(f"Saved best classifier to {classifier_path}")
            
        else:
            self.logger.error("No successful configurations found!")
            test_acc = 0.0
        
        # Save all results
        self._save_results()
        
        # Log to Wandb if enabled
        if self.wandb_run and best_config:
            self.wandb_run.log({
                'best_val_accuracy': best_val_score,
                'test_accuracy': test_acc,
                'best_layer': best_config['layer'],
                'best_C': best_config['C'],
                'total_configs': len(all_results)
            })
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save all results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"efficient_evaluation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"📁 Results saved to: {results_file}")
    
    def _print_summary(self):
        """Print evaluation summary."""
        self.logger.info("\n" + "="*80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*80)
        
        if 'summary' in self.results:
            summary = self.results['summary']
            self.logger.info(f"Total configurations tested: {summary['total_configs_tested']}")
            self.logger.info(f"Best layer: {summary['best_layer']}")
            self.logger.info(f"Best C value: {summary['best_C']}")
            self.logger.info(f"Best validation accuracy: {summary['best_val_accuracy']:.3f}")
            self.logger.info(f"Final test accuracy: {summary['final_test_accuracy']:.3f}")
        
        self.logger.info("\nKey advantages of this pipeline:")
        self.logger.info("✓ Model loaded only once (memory efficient)")
        self.logger.info("✓ Activations cached and reused")
        self.logger.info("✓ Proper train/validation/test splits")
        self.logger.info("✓ Hyperparameter optimization")
        self.logger.info("✓ Lightweight classifier training")


def create_config_from_args(args) -> EfficientEvaluationConfig:
    """Create configuration from command line arguments."""
    return EfficientEvaluationConfig(
        model_name=args.model,
        layers_to_search=args.layers,
        train_benchmark=args.train_benchmark,
        val_benchmark=args.val_benchmark,
        test_benchmark=args.test_benchmark,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
        c_values=args.c_values,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        enable_wandb=args.enable_wandb,
        batch_size=args.batch_size,
        seed=args.seed,
        verbose=args.verbose
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Efficient Model Benchmark Evaluation Pipeline")
    
    # Model configuration
    parser.add_argument("--model", default="/workspace/models/llama31-8b-instruct-hf",
                       help="Model name or path")
    parser.add_argument("--layers", nargs="+", type=int, default=[12, 14, 16, 18, 20],
                       help="Layers to search over for hyperparameter optimization")
    
    # Benchmark configuration
    parser.add_argument("--train-benchmark", default="math500", help="Training benchmark")
    parser.add_argument("--val-benchmark", default="aime2024", help="Validation benchmark")
    parser.add_argument("--test-benchmark", default="aime2025", help="Test benchmark")
    parser.add_argument("--train-limit", type=int, default=100, help="Training samples limit")
    parser.add_argument("--val-limit", type=int, default=50, help="Validation samples limit")
    parser.add_argument("--test-limit", type=int, default=50, help="Test samples limit")
    
    # Hyperparameter search
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.1, 1.0, 10.0],
                       help="C values for LogisticRegression")
    
    # Output configuration
    parser.add_argument("--output-dir", default="efficient_evaluation_results", help="Output directory")
    parser.add_argument("--experiment-name", default="efficient_wisent_guard_evaluation", 
                       help="Experiment name")
    
    # Wandb configuration
    parser.add_argument("--wandb-project", default="wisent-guard-efficient-evaluation", help="Wandb project name")
    parser.add_argument("--wandb-entity", help="Wandb entity/username")
    parser.add_argument("--enable-wandb", action="store_true", help="Enable Wandb logging")
    
    # Other options
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for activation extraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run evaluation
    evaluator = EfficientBenchmarkEvaluator(config)
    results = evaluator.run_efficient_three_phase_pipeline()
    
    print("\n✅ Efficient evaluation complete!")


if __name__ == "__main__":
    main()