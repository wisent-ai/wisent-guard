#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Across All Benchmarks and Layers

This script provides a complete evaluation pipeline that:
1. Checks which classifiers exist for a given model
2. Trains missing classifiers using existing infrastructure
3. Generates test responses if needed
4. Evaluates all classifiers and provides comprehensive results

Uses existing infrastructure:
- download_full_benchmarks.py for benchmark data
- generate_classifiers_all_layers_benchmarks.py for training
- create_testing_responses.py for test data generation
"""

import os
import sys
import json
import pickle
import torch
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parents[2]  # Go up to wisent-activation-guardrails root
sys.path.insert(0, str(project_root))

# Import existing infrastructure
from generate_classifiers_all_layers_benchmarks import CLIBatchClassifierGenerator
from create_testing_responses import TestingResponseGenerator


class ComprehensiveModelEvaluator:
    """Comprehensive evaluator that trains missing classifiers and evaluates all."""
    
    def __init__(self, model_name: str, base_dir: Optional[str] = None):
        """
        Initialize the comprehensive evaluator.
        
        Args:
            model_name: Name of the model to evaluate
            base_dir: Base directory for classifiers and test data
        """
        self.model_name = model_name
        self.model_safe = model_name.replace('/', '_').replace('-', '_')
        self.script_dir = Path(__file__).parent
        
        # Set up directories
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = self.script_dir
        
        self.classifiers_dir = self.base_dir / self.model_safe
        self.test_responses_dir = self.base_dir / "testing_responses"
        self.results_dir = self.base_dir / "evaluation_results"
        
        # Create directories
        self.classifiers_dir.mkdir(parents=True, exist_ok=True)
        self.test_responses_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize infrastructure components
        self.classifier_generator = CLIBatchClassifierGenerator(model_name, str(self.base_dir))
        self.response_generator = TestingResponseGenerator(model_name, str(self.test_responses_dir))
        
        print(f"🔬 Comprehensive Model Evaluator")
        print(f"   Model: {model_name}")
        print(f"   Classifiers: {self.classifiers_dir}")
        print(f"   Test responses: {self.test_responses_dir}")
        print(f"   Results: {self.results_dir}")
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmarks."""
        return self.classifier_generator.get_available_benchmarks()
    
    def get_model_layers(self) -> List[int]:
        """Get list of model layers."""
        return self.classifier_generator.detect_model_layers()
    
    def check_classifier_coverage(self, benchmarks: List[str], layers: List[int]) -> Dict[str, Any]:
        """
        Check which classifiers exist and which are missing.
        
        Args:
            benchmarks: List of benchmark names
            layers: List of layer indices
            
        Returns:
            Dictionary with coverage information
        """
        print(f"\n📊 Checking classifier coverage...")
        print(f"   Benchmarks: {len(benchmarks)}")
        print(f"   Layers: {len(layers)}")
        
        coverage = {
            'existing_classifiers': [],
            'missing_classifiers': [],
            'total_expected': len(benchmarks) * len(layers),
            'total_existing': 0,
            'total_missing': 0,
            'coverage_by_benchmark': {},
            'coverage_by_layer': {}
        }
        
        for benchmark in benchmarks:
            benchmark_dir = self.classifiers_dir / benchmark
            benchmark_coverage = {'existing': [], 'missing': []}
            
            for layer in layers:
                # Check for classifier file (CLI creates files with _pkl_layer_{layer} suffix)
                classifier_file = benchmark_dir / f"layer_{layer}_pkl_layer_{layer}.pkl"
                
                if classifier_file.exists():
                    coverage['existing_classifiers'].append((benchmark, layer))
                    benchmark_coverage['existing'].append(layer)
                else:
                    coverage['missing_classifiers'].append((benchmark, layer))
                    benchmark_coverage['missing'].append(layer)
            
            coverage['coverage_by_benchmark'][benchmark] = benchmark_coverage
        
        # Coverage by layer
        for layer in layers:
            layer_coverage = {'existing': [], 'missing': []}
            for benchmark in benchmarks:
                benchmark_dir = self.classifiers_dir / benchmark
                classifier_file = benchmark_dir / f"layer_{layer}_pkl_layer_{layer}.pkl"
                
                if classifier_file.exists():
                    layer_coverage['existing'].append(benchmark)
                else:
                    layer_coverage['missing'].append(benchmark)
            
            coverage['coverage_by_layer'][layer] = layer_coverage
        
        coverage['total_existing'] = len(coverage['existing_classifiers'])
        coverage['total_missing'] = len(coverage['missing_classifiers'])
        
        print(f"   ✅ Existing: {coverage['total_existing']}/{coverage['total_expected']}")
        print(f"   ❌ Missing: {coverage['total_missing']}/{coverage['total_expected']}")
        print(f"   📊 Coverage: {coverage['total_existing']/coverage['total_expected']*100:.1f}%")
        
        return coverage
    
    def train_missing_classifiers(self, missing_classifiers: List[Tuple[str, int]], 
                                 split_ratio: float = 0.8, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Train missing classifiers using existing infrastructure.
        
        Args:
            missing_classifiers: List of (benchmark, layer) tuples that need training
            split_ratio: Train/validation split ratio
            limit: Limit samples per benchmark
            
        Returns:
            Training results
        """
        if not missing_classifiers:
            print(f"✅ No missing classifiers to train!")
            return {'trained': 0, 'failed': 0, 'details': []}
        
        print(f"\n🏗️ Training {len(missing_classifiers)} missing classifiers...")
        
        # Group by benchmark
        benchmarks_to_train = {}
        for benchmark, layer in missing_classifiers:
            if benchmark not in benchmarks_to_train:
                benchmarks_to_train[benchmark] = []
            benchmarks_to_train[benchmark].append(layer)
        
        print(f"   📊 Benchmarks to train: {len(benchmarks_to_train)}")
        for benchmark, layers in benchmarks_to_train.items():
            print(f"      • {benchmark}: layers {layers}")
        
        # Use existing classifier generator
        training_results = {
            'trained': 0,
            'failed': 0,
            'details': []
        }
        
        for benchmark, layers in benchmarks_to_train.items():
            print(f"\n🎯 Training classifiers for {benchmark}...")
            
            benchmark_results = self.classifier_generator.generate_classifiers_for_benchmark(
                benchmark, layers, split_ratio, limit
            )
            
            training_results['trained'] += len(benchmark_results['successful'])
            training_results['failed'] += len(benchmark_results['failed']) + len(benchmark_results['errors'])
            training_results['details'].append(benchmark_results)
        
        print(f"\n📊 Training Summary:")
        print(f"   ✅ Successfully trained: {training_results['trained']}")
        print(f"   ❌ Failed to train: {training_results['failed']}")
        
        return training_results
    
    def ensure_test_responses_exist(self, benchmarks: List[str], 
                                   limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Ensure test responses exist for all benchmarks.
        
        Args:
            benchmarks: List of benchmark names
            limit: Limit test samples per benchmark
            
        Returns:
            Test response generation results
        """
        print(f"\n🧪 Checking test response coverage...")
        
        missing_responses = []
        for benchmark in benchmarks:
            response_dir = self.test_responses_dir / benchmark
            responses_file = response_dir / "responses.json"
            
            if not responses_file.exists():
                missing_responses.append(benchmark)
        
        if not missing_responses:
            print(f"   ✅ All test responses exist!")
            return {'generated': 0, 'already_existed': len(benchmarks)}
        
        print(f"   ❌ Missing responses for {len(missing_responses)} benchmarks")
        print(f"   🔧 Generating test responses...")
        
        # Generate missing test responses
        results = self.response_generator.generate_all_testing_responses(
            benchmarks=missing_responses,
            limit_per_benchmark=limit
        )
        
        return {
            'generated': len(results.get('benchmarks_processed', [])),
            'already_existed': len(benchmarks) - len(missing_responses),
            'details': results
        }
    
    def load_classifier(self, benchmark: str, layer: int) -> Optional[Any]:
        """Load a trained classifier."""
        classifier_file = self.classifiers_dir / benchmark / f"layer_{layer}_pkl_layer_{layer}.pkl"
        
        if not classifier_file.exists():
            return None
        
        try:
            with open(classifier_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"      ⚠️ Failed to load classifier {benchmark}/layer_{layer}: {e}")
            return None
    
    def load_test_data(self, benchmark: str) -> Optional[Dict[str, Any]]:
        """Load test responses and activations for a benchmark."""
        test_dir = self.test_responses_dir / benchmark
        responses_file = test_dir / "responses.json"
        activations_dir = test_dir / "activations"
        
        if not responses_file.exists() or not activations_dir.exists():
            return None
        
        try:
            # Load responses
            with open(responses_file, 'r') as f:
                responses_data = json.load(f)
            
            # Load activations for all layers
            activations = {}
            for layer_file in activations_dir.glob("layer_*.pt"):
                layer_num = int(layer_file.stem.split('_')[1])
                activations[layer_num] = torch.load(layer_file)
            
            return {
                'responses': responses_data,
                'activations': activations
            }
        except Exception as e:
            print(f"      ⚠️ Failed to load test data for {benchmark}: {e}")
            return None
    
    def evaluate_classifier(self, classifier: Any, activations: torch.Tensor, 
                          labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate a single classifier.
        
        Args:
            classifier: Trained classifier
            activations: Test activations
            labels: True labels ('GOOD' or 'BAD')
            
        Returns:
            Evaluation metrics
        """
        try:
            # Convert activations to numpy for sklearn
            if isinstance(activations, torch.Tensor):
                X_test = activations.detach().cpu().numpy()
            else:
                X_test = activations
            
            # Convert labels to binary (1 for GOOD, 0 for BAD)
            y_true = [1 if label == 'GOOD' else 0 for label in labels]
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            y_pred_proba = None
            
            # Get prediction probabilities if available
            if hasattr(classifier, 'predict_proba'):
                y_pred_proba = classifier.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist(),
                'num_samples': len(y_true),
                'num_positive': sum(y_true),
                'num_negative': len(y_true) - sum(y_true),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        except Exception as e:
            return {
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def evaluate_all_classifiers(self, benchmarks: List[str], layers: List[int], 
                               test_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate all existing classifiers.
        
        Args:
            benchmarks: List of benchmark names
            layers: List of layer indices
            test_limit: Limit test samples per benchmark
            
        Returns:
            Comprehensive evaluation results
        """
        print(f"\n🔬 Evaluating all classifiers...")
        
        evaluation_results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'benchmarks_evaluated': [],
            'layers_evaluated': layers,
            'benchmark_results': {},
            'layer_results': {},
            'overall_metrics': {},
            'summary': {
                'total_classifiers': 0,
                'successful_evaluations': 0,
                'failed_evaluations': 0,
                'avg_accuracy': 0.0,
                'avg_f1_score': 0.0
            }
        }
        
        all_accuracies = []
        all_f1_scores = []
        
        for benchmark in benchmarks:
            print(f"\n📊 Evaluating benchmark: {benchmark}")
            
            # Load test data
            test_data = self.load_test_data(benchmark)
            if not test_data:
                print(f"   ❌ No test data available for {benchmark}")
                continue
            
            responses = test_data['responses']['responses']
            activations = test_data['activations']
            
            # Limit test samples if specified
            if test_limit and test_limit < len(responses):
                responses = responses[:test_limit]
                for layer_idx in activations:
                    if activations[layer_idx] is not None:
                        activations[layer_idx] = activations[layer_idx][:test_limit]
            
            # Extract labels
            labels = [resp['evaluation_result']['label'] for resp in responses]
            
            benchmark_results = {'layers': {}, 'summary': {}}
            
            for layer in layers:
                # Load classifier
                classifier = self.load_classifier(benchmark, layer)
                if not classifier:
                    print(f"      ❌ No classifier for layer {layer}")
                    continue
                
                # Get activations for this layer
                layer_activations = activations.get(layer)
                if layer_activations is None:
                    print(f"      ❌ No activations for layer {layer}")
                    continue
                
                # Evaluate classifier
                metrics = self.evaluate_classifier(classifier, layer_activations, labels)
                
                if 'error' not in metrics:
                    print(f"      ✅ Layer {layer}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
                    all_accuracies.append(metrics['accuracy'])
                    all_f1_scores.append(metrics['f1_score'])
                    evaluation_results['summary']['successful_evaluations'] += 1
                else:
                    print(f"      ❌ Layer {layer}: {metrics['error']}")
                    evaluation_results['summary']['failed_evaluations'] += 1
                
                benchmark_results['layers'][layer] = metrics
                evaluation_results['summary']['total_classifiers'] += 1
            
            # Calculate benchmark summary
            layer_metrics = [m for m in benchmark_results['layers'].values() if 'error' not in m]
            if layer_metrics:
                benchmark_results['summary'] = {
                    'avg_accuracy': np.mean([m['accuracy'] for m in layer_metrics]),
                    'avg_f1_score': np.mean([m['f1_score'] for m in layer_metrics]),
                    'best_layer': max(layer_metrics, key=lambda x: x['f1_score']),
                    'worst_layer': min(layer_metrics, key=lambda x: x['f1_score']),
                    'num_layers_evaluated': len(layer_metrics)
                }
            
            evaluation_results['benchmark_results'][benchmark] = benchmark_results
            evaluation_results['benchmarks_evaluated'].append(benchmark)
        
        # Calculate overall metrics
        if all_accuracies:
            evaluation_results['summary']['avg_accuracy'] = np.mean(all_accuracies)
            evaluation_results['summary']['avg_f1_score'] = np.mean(all_f1_scores)
        
        # Calculate layer-wise summary
        for layer in layers:
            layer_metrics = []
            for benchmark in benchmarks:
                if benchmark in evaluation_results['benchmark_results']:
                    layer_result = evaluation_results['benchmark_results'][benchmark]['layers'].get(layer)
                    if layer_result and 'error' not in layer_result:
                        layer_metrics.append(layer_result)
            
            if layer_metrics:
                evaluation_results['layer_results'][layer] = {
                    'avg_accuracy': np.mean([m['accuracy'] for m in layer_metrics]),
                    'avg_f1_score': np.mean([m['f1_score'] for m in layer_metrics]),
                    'num_benchmarks': len(layer_metrics),
                    'best_benchmark': max(layer_metrics, key=lambda x: x['f1_score']),
                    'worst_benchmark': min(layer_metrics, key=lambda x: x['f1_score'])
                }
        
        return evaluation_results
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save evaluation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{self.model_safe}_{timestamp}.json"
        
        results_file = self.results_dir / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved evaluation results to: {results_file}")
        return str(results_file)
    
    def print_evaluation_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results."""
        print(f"\n{'='*80}")
        print(f"🔬 COMPREHENSIVE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        summary = results['summary']
        print(f"📊 Overall Performance:")
        print(f"   Total classifiers: {summary['total_classifiers']}")
        print(f"   Successful evaluations: {summary['successful_evaluations']}")
        print(f"   Failed evaluations: {summary['failed_evaluations']}")
        print(f"   Average accuracy: {summary['avg_accuracy']:.3f}")
        print(f"   Average F1 score: {summary['avg_f1_score']:.3f}")
        
        print(f"\n📋 Benchmark Performance:")
        for benchmark, result in results['benchmark_results'].items():
            if 'summary' in result and result['summary']:
                summary = result['summary']
                print(f"   • {benchmark}: Acc={summary['avg_accuracy']:.3f}, F1={summary['avg_f1_score']:.3f} ({summary['num_layers_evaluated']} layers)")
        
        print(f"\n🧠 Layer Performance:")
        for layer, result in results['layer_results'].items():
            print(f"   • Layer {layer}: Acc={result['avg_accuracy']:.3f}, F1={result['avg_f1_score']:.3f} ({result['num_benchmarks']} benchmarks)")
        
        # Find best overall performance
        if results['benchmark_results']:
            best_performances = []
            for benchmark, result in results['benchmark_results'].items():
                for layer, layer_result in result['layers'].items():
                    if 'error' not in layer_result:
                        best_performances.append((benchmark, layer, layer_result['f1_score']))
            
            if best_performances:
                best_performances.sort(key=lambda x: x[2], reverse=True)
                best_benchmark, best_layer, best_f1 = best_performances[0]
                print(f"\n🏆 Best Performance:")
                print(f"   {best_benchmark} at layer {best_layer}: F1={best_f1:.3f}")
    
    def run_comprehensive_evaluation(self, benchmarks: Optional[List[str]] = None,
                                   layers: Optional[List[int]] = None,
                                   train_missing: bool = True,
                                   test_limit: Optional[int] = None,
                                   train_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation pipeline.
        
        Args:
            benchmarks: List of benchmarks to evaluate (None for all)
            layers: List of layers to evaluate (None for all)
            train_missing: Whether to train missing classifiers
            test_limit: Limit test samples per benchmark
            train_limit: Limit training samples per benchmark
            
        Returns:
            Complete evaluation results
        """
        print(f"🚀 Starting Comprehensive Model Evaluation")
        print(f"{'='*80}")
        
        # Get available benchmarks and layers
        available_benchmarks = self.get_available_benchmarks()
        available_layers = self.get_model_layers()
        
        benchmarks = benchmarks or available_benchmarks
        layers = layers or available_layers
        
        # Filter to available benchmarks
        benchmarks = [b for b in benchmarks if b in available_benchmarks]
        
        if not benchmarks:
            print(f"❌ No valid benchmarks found!")
            return {}
        
        print(f"📊 Evaluation Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Benchmarks: {len(benchmarks)}")
        print(f"   Layers: {len(layers)}")
        print(f"   Train missing: {train_missing}")
        if test_limit:
            print(f"   Test limit: {test_limit} samples per benchmark")
        if train_limit:
            print(f"   Training limit: {train_limit} samples per benchmark")
        
        # Step 1: Check classifier coverage
        coverage = self.check_classifier_coverage(benchmarks, layers)
        
        # Step 2: Train missing classifiers if requested
        training_results = {}
        if train_missing and coverage['missing_classifiers']:
            training_results = self.train_missing_classifiers(
                coverage['missing_classifiers'], limit=train_limit
            )
        
        # Step 3: Ensure test responses exist
        test_response_results = self.ensure_test_responses_exist(benchmarks, test_limit)
        
        # Step 4: Evaluate all classifiers
        evaluation_results = self.evaluate_all_classifiers(benchmarks, layers, test_limit)
        
        # Step 5: Compile complete results
        complete_results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'benchmarks': benchmarks,
                'layers': layers,
                'train_missing': train_missing,
                'test_limit': test_limit,
                'train_limit': train_limit
            },
            'coverage_check': coverage,
            'training_results': training_results,
            'test_response_results': test_response_results,
            'evaluation_results': evaluation_results
        }
        
        # Step 6: Save and display results
        results_file = self.save_evaluation_results(complete_results)
        self.print_evaluation_summary(evaluation_results)
        
        print(f"\n🎉 Comprehensive evaluation completed!")
        print(f"📁 Results saved to: {results_file}")
        
        return complete_results


def main():
    """Main function for comprehensive model evaluation."""
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation across all benchmarks and layers')
    parser.add_argument('model', help='Model name to evaluate')
    parser.add_argument('--train-limit', type=int, default=1000,
                       help='Limit training samples per benchmark (default: 1000)')
    parser.add_argument('--test-limit', type=int, default=None,
                       help='Limit test samples per benchmark (default: no limit)')
    
    args = parser.parse_args()
    
    print(f"🔬 Comprehensive Model Evaluation")
    print(f"{'='*80}")
    
    # Initialize evaluator with sensible defaults
    evaluator = ComprehensiveModelEvaluator(args.model)
    
    # Run comprehensive evaluation with all defaults:
    # - All available benchmarks
    # - All model layers  
    # - Train missing classifiers
    # - No limits on test/train samples
    # - Use default base directory
    try:
        results = evaluator.run_comprehensive_evaluation(
            benchmarks=None,          # All available benchmarks
            layers=None,              # All model layers
            train_missing=True,       # Train missing classifiers
            test_limit=args.test_limit,    # User-specified test limit
            train_limit=args.train_limit   # User-specified training limit (default: 1000)
        )
        
        if results:
            print(f"\n✅ Evaluation completed successfully!")
        else:
            print(f"\n❌ Evaluation failed!")
            
    except KeyboardInterrupt:
        print(f"\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()

