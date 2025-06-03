"""
Benchmark runner for Wisent-Guard that handles evaluation of hallucination detection.
"""

import logging
import json
from pathlib import Path
import torch
from typing import List, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from .benchmark_loader import BenchmarkLoader
from .monitor import ActivationMonitor, ContrastiveVectors

class BenchmarkRunner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_benchmark(self, benchmark_names: List[str], model_name: str, layer: int, device: str = "mps") -> Dict[str, Dict[str, Any]]:
        """
        Run benchmark evaluation with Wisent-Guard hallucination detection.
        
        Args:
            benchmark_names: List of benchmark names to run
            model_name: Model name to use
            layer: Layer number to monitor
            device: Device to run on (mps, cpu)
            
        Returns:
            Dictionary of benchmark results with metrics for each benchmark
        """
        results = {}
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set device
        if device == "mps":
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                print("MPS not available, falling back to CPU")
                device = "cpu"
                model = model.to(device)
            
            # Check device
            print(f"Using device: {device}")
            print(f"Model device: {next(model.parameters()).device}")
        
        # Initialize activation monitor with contrastive vectors
        vectors = ContrastiveVectors()
        # Create a list of layers to monitor
        layers = [layer]  # Convert single layer to list
        monitor = ActivationMonitor(
            model=model,
            layers=layers,
            vectors=vectors,
            token_strategy="last"
        )
        
        for benchmark_name in benchmark_names:
            self.logger.info(f"\nStarting benchmark: {benchmark_name}")
            
            # Load benchmark
            loader = BenchmarkLoader(benchmark_name)
            train_docs, test_docs = loader.get_train_test_split()
            
            # Prepare training data
            train_pairs = loader.get_prompt_pairs(train_docs)
            
            # Initialize classifier
            classifier = LogisticRegression()
            
            # Extract activations for training
            train_activations = []
            train_labels = []
            
            for prompt, target in train_pairs:
                # Get model response
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get activations from specified layer
                activations = monitor.get_activations(outputs.hidden_states[layer])
                
                # Store activations and labels
                train_activations.append(activations)
                train_labels.append(1)  # 1 for good responses
                
                # Generate a bad response for negative example
                # This is a simplified version - in practice we might want more sophisticated methods
                bad_prompt = f"{prompt}\nBad response:"  # Add some indicator for bad responses
                bad_inputs = tokenizer(bad_prompt, return_tensors="pt").to(device)
                bad_outputs = model(**bad_inputs, output_hidden_states=True)
                
                bad_activations = monitor.get_activations(bad_outputs.hidden_states[layer])
                train_activations.append(bad_activations)
                train_labels.append(0)  # 0 for bad responses
            
            # Train classifier
            classifier.fit(train_activations, train_labels)
            
            # Evaluate on test set
            test_pairs = loader.get_prompt_pairs(test_docs)
            test_activations = []
            test_labels = []
            
            for prompt, target in test_pairs:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs, output_hidden_states=True)
                
                activations = monitor.get_activations(outputs.hidden_states[layer])
                test_activations.append(activations)
                test_labels.append(1)  # 1 for good responses
                
                # Generate bad response for evaluation
                bad_prompt = f"{prompt}\nBad response:"
                bad_inputs = tokenizer(bad_prompt, return_tensors="pt").to(device)
                bad_outputs = model(**bad_inputs, output_hidden_states=True)
                
                bad_activations = monitor.get_activations(bad_outputs.hidden_states[layer])
                test_activations.append(bad_activations)
                test_labels.append(0)  # 0 for bad responses
            
            # Make predictions
            predictions = classifier.predict(test_activations)
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            
            # Count hallucinations caught
            hallucinations_caught = sum(1 for pred, label in zip(predictions, test_labels) 
                                      if label == 0 and pred == 1)
            
            # Save results
            results[benchmark_name] = {
                'accuracy': float(accuracy),
                'hallucinations_caught': hallucinations_caught,
                'total_hallucinations': test_labels.count(0),
                'detection_rate': float(hallucinations_caught / test_labels.count(0))
            }
            
            self.logger.info(f"Benchmark {benchmark_name} completed")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Detection rate: {results[benchmark_name]['detection_rate']:.4f}")
        
        return results
