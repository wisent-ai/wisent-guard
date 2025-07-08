#!/usr/bin/env python3
"""
Minimal pipeline test: 4 training + 1 testing contrastive pairs per benchmark.
Fails hard on any error. Tests one layer only.
"""

import os
import sys
import json
import pickle
import torch
import argparse
import time
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parents[2]
sys.path.insert(0, str(project_root))

from create_testing_responses import TestingResponseGenerator
from wisent_guard.core.classifiers.pipeline_steps.download_full_benchmarks import FullBenchmarkDownloader


class MinimalPipelineTester:
    def __init__(self, model_name: str, layer: int):
        self.model_name = model_name
        self.layer = layer
        self.script_dir = Path(__file__).parent
        
        # Output directory
        self.output_dir = self.script_dir / "minimal_pipeline_test"
        self.output_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            'model_name': model_name,
            'layer': layer,
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {}
        }
        
        print(f"🧪 Minimal Pipeline Tester")
        print(f"   Model: {model_name}")
        print(f"   Layer: {layer}")
        print(f"   Output: {self.output_dir}")
    
    def ensure_benchmarks_downloaded(self):
        """Download benchmarks if needed."""
        benchmarks_dir = self.script_dir / "full_benchmarks" / "data"
        if not benchmarks_dir.exists() or not list(benchmarks_dir.glob("*.pkl")):
            print(f"📥 Downloading benchmarks...")
            downloader = FullBenchmarkDownloader(str(benchmarks_dir.parent))
            downloader.download_all_benchmarks()
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmark files."""
        benchmarks_dir = self.script_dir / "full_benchmarks" / "data"
        benchmark_files = list(benchmarks_dir.glob("*.pkl"))
        return [f.stem for f in benchmark_files]
    
    def extract_single_layer_activations(self, response_generator, texts: List[str]) -> torch.Tensor:
        """Extract activations from only the specified layer for a batch of texts."""
        from wisent_guard.core.layer import Layer
        
        print(f"   🧠 Extracting activations for layer {self.layer} from {len(texts)} texts...")
        
        layer = Layer(index=self.layer, type="transformer")
        activations_list = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Empty text provided for activation extraction at index {i}")
            
            layer_activations = response_generator.model.extract_activations(text, layer)
            
            if layer_activations is None or not isinstance(layer_activations, torch.Tensor):
                raise ValueError(f"Failed to extract activations for layer {self.layer} at text index {i}")
            
            # Ensure 2D tensor [batch_size, hidden_dim]
            if layer_activations.dim() == 1:
                layer_activations = layer_activations.unsqueeze(0)
            
            activations_list.append(layer_activations)
        
        # Stack all activations
        all_activations = torch.cat(activations_list, dim=0)
        print(f"   ✅ Extracted activations: shape {all_activations.shape}")
        
        return all_activations
    
    def test_benchmark(self, benchmark: str) -> Dict:
        """Test complete pipeline for one benchmark."""
        print(f"\n🎯 Testing {benchmark}...")
        
        benchmark_result = {
            'benchmark': benchmark,
            'training_pairs': [],
            'testing_pairs': [],
            'training_activations_shape': None,
            'testing_activations_shape': None, 
            'token_position': None,
            'classifier_trained': False,
            'prediction': None,
            'ground_truth': None,
            'accuracy': None
        }
        
        # Step 1: Generate 5 contrastive pairs (4 train + 1 test)
        print(f"   🔄 Generating 5 contrastive pairs...")
        response_generator = TestingResponseGenerator(self.model_name, str(self.output_dir))
        
        # Generate test responses with limit of 5
        test_data = response_generator.generate_responses_for_benchmark(
            benchmark, limit=5
        )
        
        if len(test_data['responses']) != 5:
            raise ValueError(f"Expected 5 responses, got {len(test_data['responses'])}")
        
        # Split into train/test
        train_responses = test_data['responses'][:4]
        test_responses = test_data['responses'][4:5]
        
        # Save contrastive pairs
        for i, resp in enumerate(train_responses):
            benchmark_result['training_pairs'].append({
                'question': resp['question'],
                'good_response': resp['evaluation_result']['good_response'],
                'bad_response': resp['evaluation_result']['bad_response'],
                'good_activation_text': resp['evaluation_result']['good_activation_text'],
                'bad_activation_text': resp['evaluation_result']['bad_activation_text']
            })
        
        for resp in test_responses:
            benchmark_result['testing_pairs'].append({
                'question': resp['question'],
                'good_response': resp['evaluation_result']['good_response'],
                'bad_response': resp['evaluation_result']['bad_response'],
                'good_activation_text': resp['evaluation_result']['good_activation_text'],
                'bad_activation_text': resp['evaluation_result']['bad_activation_text']
            })
        
        # Step 2: Extract activations for specified layer
        print(f"   🧠 Extracting activations for layer {self.layer}...")
        
        # Prepare all texts and labels
        train_texts = []
        train_labels = []
        for pair in benchmark_result['training_pairs']:
            train_texts.append(pair['good_activation_text'])
            train_labels.append('GOOD')
            train_texts.append(pair['bad_activation_text'])
            train_labels.append('BAD')
        
        test_texts = []
        test_labels = []
        for pair in benchmark_result['testing_pairs']:
            test_texts.append(pair['good_activation_text'])
            test_labels.append('GOOD')
            test_texts.append(pair['bad_activation_text'])
            test_labels.append('BAD')
        
        # Extract activations for ONLY the specified layer
        train_activations = self.extract_single_layer_activations(response_generator, train_texts)
        test_activations = self.extract_single_layer_activations(response_generator, test_texts)
        
        benchmark_result['training_activations_shape'] = list(train_activations.shape)
        benchmark_result['testing_activations_shape'] = list(test_activations.shape)
        benchmark_result['token_position'] = "last_token"  # Default assumption
        
        # Step 3: Train classifier
        print(f"   🏗️ Training classifier...")
        
        # Convert to numpy
        X_train = train_activations.detach().cpu().numpy()
        y_train = [1 if label == 'GOOD' else 0 for label in train_labels]
        
        # Train classifier
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train)
        benchmark_result['classifier_trained'] = True
        
        # Step 4: Test classifier
        print(f"   🧪 Testing classifier...")
        
        # Convert to numpy
        X_test = test_activations.detach().cpu().numpy()
        y_true = [1 if label == 'GOOD' else 0 for label in test_labels]
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_true, y_pred)
        
        benchmark_result['prediction'] = y_pred.tolist()
        benchmark_result['ground_truth'] = y_true
        benchmark_result['accuracy'] = accuracy
        
        print(f"   ✅ Success! Accuracy: {accuracy:.3f}")
        
        return benchmark_result
    
    def run_all_benchmarks(self):
        """Run pipeline test on all benchmarks."""
        print(f"🚀 Starting minimal pipeline test...")
        
        # Ensure benchmarks are downloaded
        self.ensure_benchmarks_downloaded()
        
        # Get available benchmarks
        benchmarks = self.get_available_benchmarks()
        print(f"📊 Found {len(benchmarks)} benchmarks")
        
        # Test each benchmark
        for i, benchmark in enumerate(benchmarks, 1):
            print(f"\n[{i}/{len(benchmarks)}] Processing {benchmark}...")
            
            result = self.test_benchmark(benchmark)
            self.results['benchmarks'][benchmark] = result
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"pipeline_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def print_summary(self):
        """Print test summary."""
        total = len(self.results['benchmarks'])
        completed = sum(1 for r in self.results['benchmarks'].values() if 'accuracy' in r and r['accuracy'] is not None)
        failed = total - completed
        
        print(f"\n{'='*50}")
        print(f"📊 PIPELINE TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total benchmarks: {total}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        
        if completed > 0:
            accuracies = [r['accuracy'] for r in self.results['benchmarks'].values() 
                         if 'accuracy' in r and r['accuracy'] is not None]
            avg_accuracy = np.mean(accuracies)
            print(f"Average accuracy: {avg_accuracy:.3f}")
        
        print(f"Model: {self.model_name}")
        print(f"Layer: {self.layer}")


def main():
    parser = argparse.ArgumentParser(description='Minimal pipeline test: 4 train + 1 test per benchmark')
    parser.add_argument('model', help='Model name')
    parser.add_argument('layer', type=int, help='Layer to test')
    
    args = parser.parse_args()
    
    tester = MinimalPipelineTester(args.model, args.layer)
    tester.run_all_benchmarks()


if __name__ == "__main__":
    main() 