#!/usr/bin/env python3
"""
Evaluate Classifier Performance vs Sample Size

This script trains classifiers for a given model, benchmark, and layer using
different sample sizes (1, 5, 10, 50, 100, 250, 500, 1000) and evaluates
their accuracy to generate an accuracy vs sample size curve.

This helps understand:
- How much training data is needed for good classifier performance
- The point of diminishing returns for additional training samples
- Whether certain benchmarks or layers need more data than others
"""

import os
import sys
import json
import pickle
import torch
import time
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parents[2]  # Go up to wisent-activation-guardrails root
sys.path.insert(0, str(project_root))

# Import existing infrastructure
from download_full_benchmarks import FullBenchmarkDownloader
from create_testing_responses import TestingResponseGenerator


class SampleSizeEvaluator:
    """Evaluate classifier performance across different sample sizes."""
    
    def __init__(self, model_name: str, benchmark_name: str, layer: int, 
                 output_dir: Optional[str] = None, test_limit: Optional[int] = None):
        """
        Initialize the sample size evaluator.
        
        Args:
            model_name: Name of the model
            benchmark_name: Name of the benchmark
            layer: Layer index to evaluate
            output_dir: Directory to save results and plots
            test_limit: Limit test samples (default: None for all)
        """
        self.model_name = model_name
        self.benchmark_name = benchmark_name
        self.layer = layer
        self.test_limit = test_limit
        self.model_safe = model_name.replace('/', '_').replace('-', '_')
        self.script_dir = Path(__file__).parent
        self.project_root = project_root
        
        # Set up directories
        if output_dir is None:
            self.output_dir = self.script_dir / "sample_size_results"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks_dir = self.script_dir / "full_benchmarks" / "data"
        self.classifiers_dir = self.script_dir / "sample_size_classifiers"
        self.test_responses_dir = self.script_dir / "testing_responses"
        
        # Create directories
        self.classifiers_dir.mkdir(parents=True, exist_ok=True)
        self.test_responses_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample sizes to evaluate
        self.sample_sizes = [1, 5, 10, 50, 100, 250, 500, 1000]
        
        print(f"🔬 Sample Size Evaluator")
        print(f"   Model: {model_name}")
        print(f"   Benchmark: {benchmark_name}")
        print(f"   Layer: {layer}")
        print(f"   Sample sizes: {self.sample_sizes}")
        print(f"   Test limit: {test_limit if test_limit else 'No limit'}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Classifiers directory: {self.classifiers_dir}")
    
    def ensure_benchmark_downloaded(self) -> bool:
        """Ensure benchmark data is downloaded."""
        benchmark_file = self.benchmarks_dir / f"{self.benchmark_name}.pkl"
        
        if benchmark_file.exists():
            print(f"✅ Benchmark data already exists: {benchmark_file}")
            return True
        
        print(f"📥 Downloading benchmark: {self.benchmark_name}")
        try:
            downloader = FullBenchmarkDownloader(download_dir=str(self.script_dir / "full_benchmarks"))
            results = downloader.download_all_benchmarks(
                benchmarks=[self.benchmark_name],
                force=False
            )
            
            if self.benchmark_name in results['successful']:
                print(f"✅ Successfully downloaded benchmark: {self.benchmark_name}")
                return True
            else:
                print(f"❌ Failed to download benchmark: {self.benchmark_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading benchmark: {e}")
            return False
    
    def ensure_test_responses_exist(self) -> bool:
        """Ensure test responses exist for evaluation."""
        # Check if test responses exist in the new format
        test_responses_file = self.test_responses_dir / f"{self.model_safe}_{self.benchmark_name}_responses.pkl"
        
        if test_responses_file.exists():
            print(f"✅ Test responses already exist: {test_responses_file}")
            return True
        
        print(f"🧪 Generating test responses for: {self.benchmark_name}")
        try:
            response_generator = TestingResponseGenerator(
                self.model_name, 
                str(self.test_responses_dir)
            )
            
            results = response_generator.generate_responses_for_benchmark(
                self.benchmark_name,
                limit=self.test_limit
            )
            
            # Check if we have results and successful responses
            if results and results.get('num_successful_responses', 0) > 0:
                print(f"✅ Successfully generated {results['num_successful_responses']} test responses")
                
                # Save in the format expected by load_test_data
                responses = results['responses']
                activations = results['stacked_activations']
                labels = [resp['evaluation_result']['label'] for resp in responses]
                
                # Save to pickle file
                test_data = {
                    'responses': responses,
                    'activations': activations,
                    'labels': labels
                }
                
                with open(test_responses_file, 'wb') as f:
                    pickle.dump(test_data, f)
                
                print(f"💾 Saved test responses to: {test_responses_file}")
                return True
            else:
                print(f"❌ Failed to generate test responses: {results}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating test responses: {e}")
            return False
    
    def train_classifier_with_sample_size(self, sample_size: int) -> Dict[str, Any]:
        """
        Train a classifier with a specific sample size.
        
        Args:
            sample_size: Number of training samples to use
            
        Returns:
            Training results
        """
        print(f"🔧 Training classifier with {sample_size} samples...")
        
        # Create classifier save path
        classifier_subdir = self.classifiers_dir / self.model_safe / self.benchmark_name
        classifier_subdir.mkdir(parents=True, exist_ok=True)
        classifier_path = classifier_subdir / f"layer_{self.layer}_samples_{sample_size}.pkl"
        
        # Skip if classifier already exists
        if classifier_path.exists():
            print(f"   ✅ Classifier already exists: {classifier_path}")
            return {
                'status': 'already_exists',
                'classifier_path': str(classifier_path),
                'sample_size': sample_size
            }
        
        # Construct CLI command
        cmd = [
            sys.executable, "-m", "wisent_guard", "tasks", self.benchmark_name,
            "--model", self.model_name,
            "--layer", str(self.layer),
            "--train-only",
            "--save-classifier", str(classifier_path),
            "--split-ratio", "0.8",
            "--limit", str(sample_size),
            "--verbose"
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                # Check if classifier was created (CLI appends suffix)
                actual_path = classifier_subdir / f"layer_{self.layer}_samples_{sample_size}_pkl_layer_{self.layer}.pkl"
                if actual_path.exists():
                    print(f"   ✅ Success ({execution_time:.1f}s): {actual_path}")
                    return {
                        'status': 'success',
                        'execution_time': execution_time,
                        'classifier_path': str(actual_path),
                        'sample_size': sample_size
                    }
                else:
                    print(f"   ❌ CLI completed but no classifier file found")
                    return {
                        'status': 'no_output',
                        'execution_time': execution_time,
                        'sample_size': sample_size
                    }
            else:
                print(f"   ❌ CLI failed (code {result.returncode})")
                print(f"   Error: {result.stderr}")
                return {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'return_code': result.returncode,
                    'error': result.stderr,
                    'sample_size': sample_size
                }
                
        except subprocess.TimeoutExpired:
            print(f"   ❌ Training timed out after 30 minutes")
            return {
                'status': 'timeout',
                'execution_time': 1800,
                'sample_size': sample_size
            }
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'sample_size': sample_size
            }
    
    def load_classifier(self, sample_size: int) -> Optional[Any]:
        """Load a trained classifier."""
        classifier_subdir = self.classifiers_dir / self.model_safe / self.benchmark_name
        classifier_path = classifier_subdir / f"layer_{self.layer}_samples_{sample_size}_pkl_layer_{self.layer}.pkl"
        
        if not classifier_path.exists():
            return None
        
        try:
            with open(classifier_path, 'rb') as f:
                saved_data = pickle.load(f)
                
            # Extract the actual classifier from the saved data structure
            if isinstance(saved_data, dict) and 'classifier' in saved_data:
                return saved_data['classifier']
            else:
                # Fallback in case it's saved directly as a classifier
                return saved_data
        except Exception as e:
            print(f"⚠️ Failed to load classifier for {sample_size} samples: {e}")
            return None
    
    def load_test_data(self) -> Tuple[List[str], Dict[int, torch.Tensor], List[str]]:
        """
        Load test data for evaluation.
        
        Returns:
            Tuple of (test_responses, test_activations, test_labels)
        """
        # Load test responses
        test_responses_file = self.test_responses_dir / f"{self.model_safe}_{self.benchmark_name}_responses.pkl"
        
        if not test_responses_file.exists():
            print(f"   ❌ Test responses file not found: {test_responses_file}")
            return [], {}, []
            
        with open(test_responses_file, 'rb') as f:
            test_data = pickle.load(f)
            
        test_responses = test_data['responses']
        test_activations_all = test_data['activations']
        test_labels = test_data['labels']
        
        # Apply test limit if specified
        if self.test_limit and self.test_limit < len(test_responses):
            print(f"   📏 Limiting test data to {self.test_limit} samples (from {len(test_responses)})")
            test_responses = test_responses[:self.test_limit]
            test_labels = test_labels[:self.test_limit]
            
            # Apply limit to activations
            test_activations = {}
            for layer_idx, activations in test_activations_all.items():
                if activations is not None:
                    test_activations[layer_idx] = activations[:self.test_limit]
                else:
                    test_activations[layer_idx] = None
        else:
            test_activations = test_activations_all
        
        print(f"   📊 Test data loaded: {len(test_responses)} samples")
        return test_responses, test_activations, test_labels
    
    def evaluate_classifier(self, classifier: Any, activations: torch.Tensor, 
                          labels: List[str]) -> Dict[str, Any]:
        """Evaluate a classifier and return metrics."""
        try:
            # Convert activations to numpy
            if isinstance(activations, torch.Tensor):
                X_test = activations.detach().cpu().numpy()
            else:
                X_test = activations
            
            # Convert labels to binary
            y_true = [1 if label == 'GOOD' else 0 for label in labels]
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            
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
                'num_negative': len(y_true) - sum(y_true)
            }
        except Exception as e:
            return {
                'error': str(e),
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
    
    def run_sample_size_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete sample size evaluation.
        
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n🚀 Starting sample size evaluation")
        print(f"   Model: {self.model_name}")
        print(f"   Benchmark: {self.benchmark_name}")
        print(f"   Layer: {self.layer}")
        print(f"   Sample sizes: {self.sample_sizes}")
        print(f"   Test limit: {self.test_limit if self.test_limit else 'No limit'}")
        
        # Step 1: Ensure benchmark data is downloaded
        if not self.ensure_benchmark_downloaded():
            return {'error': 'Failed to download benchmark data'}
        
        # Step 2: Ensure test responses exist
        if not self.ensure_test_responses_exist():
            return {'error': 'Failed to generate test responses'}
        
        # Step 3: Load test data
        test_responses, test_activations_all, test_labels = self.load_test_data()
        if not test_responses:
            return {'error': 'Failed to load test data'}
        
        # Extract activations for the specific layer
        if self.layer not in test_activations_all:
            return {'error': f'No activations found for layer {self.layer}'}
        
        test_activations = test_activations_all[self.layer]
        if test_activations is None:
            return {'error': f'Activations for layer {self.layer} are None'}
        
        print(f"📊 Test data loaded:")
        print(f"   Test samples: {len(test_responses)}")
        print(f"   Activations shape: {test_activations.shape}")
        print(f"   Labels: {len(test_labels)} ({sum(1 for l in test_labels if l == 'GOOD')} GOOD, {sum(1 for l in test_labels if l == 'BAD')} BAD)")
        
        # Step 4: Train classifiers with different sample sizes
        results = {
            'model_name': self.model_name,
            'benchmark_name': self.benchmark_name,
            'layer': self.layer,
            'sample_sizes': self.sample_sizes,
            'test_limit': self.test_limit,
            'timestamp': datetime.now().isoformat(),
            'training_results': {},
            'evaluation_results': {},
            'summary': {
                'total_trained': 0,
                'total_evaluated': 0,
                'successful_evaluations': 0,
                'failed_evaluations': 0
            }
        }
        
        print(f"\n🏗️ Training classifiers with different sample sizes...")
        for sample_size in self.sample_sizes:
            print(f"\n[{sample_size:4d} samples] Training classifier...")
            
            # Train classifier
            training_result = self.train_classifier_with_sample_size(sample_size)
            results['training_results'][sample_size] = training_result
            
            if training_result['status'] in ['success', 'already_exists']:
                results['summary']['total_trained'] += 1
                
                # Load and evaluate classifier
                classifier = self.load_classifier(sample_size)
                if classifier:
                    evaluation_result = self.evaluate_classifier(
                        classifier, test_activations, test_labels
                    )
                    results['evaluation_results'][sample_size] = evaluation_result
                    results['summary']['total_evaluated'] += 1
                    
                    if 'error' not in evaluation_result:
                        results['summary']['successful_evaluations'] += 1
                        accuracy = evaluation_result['accuracy']
                        f1 = evaluation_result['f1_score']
                        print(f"   ✅ Evaluation: Accuracy={accuracy:.3f}, F1={f1:.3f}")
                    else:
                        results['summary']['failed_evaluations'] += 1
                        print(f"   ❌ Evaluation failed: {evaluation_result['error']}")
                else:
                    results['summary']['failed_evaluations'] += 1
                    print(f"   ❌ Failed to load classifier")
        
        # Step 5: Save results
        self.save_results(results)
        
        # Step 6: Create visualization
        self.create_visualization(results)
        
        # Step 7: Print summary
        self.print_summary(results)
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_size_results_{self.model_safe}_{self.benchmark_name}_layer_{self.layer}_{timestamp}.json"
        results_file = self.output_dir / filename
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    def create_visualization(self, results: Dict[str, Any]):
        """Create accuracy vs sample size visualization."""
        # Extract data for plotting
        sample_sizes = []
        accuracies = []
        f1_scores = []
        
        for sample_size in self.sample_sizes:
            if sample_size in results['evaluation_results']:
                eval_result = results['evaluation_results'][sample_size]
                if 'error' not in eval_result:
                    sample_sizes.append(sample_size)
                    accuracies.append(eval_result['accuracy'])
                    f1_scores.append(eval_result['f1_score'])
        
        if not sample_sizes:
            print("⚠️ No successful evaluations to plot")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy and F1 score
        plt.subplot(2, 2, 1)
        plt.semilogx(sample_sizes, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=8)
        plt.semilogx(sample_sizes, f1_scores, 's-', label='F1 Score', linewidth=2, markersize=8)
        plt.xlabel('Training Sample Size')
        plt.ylabel('Performance')
        plt.title(f'Classifier Performance vs Sample Size\n{self.model_name} - {self.benchmark_name} - Layer {self.layer}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot accuracy only with error bars if we have multiple runs
        plt.subplot(2, 2, 2)
        plt.semilogx(sample_sizes, accuracies, 'o-', color='blue', linewidth=2, markersize=8)
        plt.xlabel('Training Sample Size')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Sample Size')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot F1 score only
        plt.subplot(2, 2, 3)
        plt.semilogx(sample_sizes, f1_scores, 's-', color='red', linewidth=2, markersize=8)
        plt.xlabel('Training Sample Size')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Sample Size')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Plot improvement rate
        plt.subplot(2, 2, 4)
        if len(accuracies) > 1:
            improvement_rates = []
            for i in range(1, len(accuracies)):
                rate = (accuracies[i] - accuracies[i-1]) / (sample_sizes[i] - sample_sizes[i-1])
                improvement_rates.append(rate)
            
            plt.loglog(sample_sizes[1:], improvement_rates, 'o-', color='green', linewidth=2, markersize=8)
            plt.xlabel('Training Sample Size')
            plt.ylabel('Accuracy Improvement Rate')
            plt.title('Marginal Improvement Rate')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"sample_size_curve_{self.model_safe}_{self.benchmark_name}_layer_{self.layer}_{timestamp}.png"
        plot_path = self.output_dir / plot_filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved to: {plot_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"🔬 SAMPLE SIZE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        print(f"📊 Configuration:")
        print(f"   Model: {self.model_name}")
        print(f"   Benchmark: {self.benchmark_name}")
        print(f"   Layer: {self.layer}")
        print(f"   Sample sizes tested: {self.sample_sizes}")
        
        summary = results['summary']
        print(f"\n📈 Results:")
        print(f"   Total trained: {summary['total_trained']}/{len(self.sample_sizes)}")
        print(f"   Total evaluated: {summary['total_evaluated']}/{len(self.sample_sizes)}")
        print(f"   Successful evaluations: {summary['successful_evaluations']}")
        print(f"   Failed evaluations: {summary['failed_evaluations']}")
        
        # Show performance by sample size
        if results['evaluation_results']:
            print(f"\n📊 Performance by Sample Size:")
            print(f"   {'Size':>6} {'Accuracy':>9} {'F1 Score':>9} {'Precision':>9} {'Recall':>9}")
            print(f"   {'-'*50}")
            
            for sample_size in self.sample_sizes:
                if sample_size in results['evaluation_results']:
                    eval_result = results['evaluation_results'][sample_size]
                    if 'error' not in eval_result:
                        accuracy = eval_result['accuracy']
                        f1 = eval_result['f1_score']
                        precision = eval_result['precision']
                        recall = eval_result['recall']
                        print(f"   {sample_size:>6} {accuracy:>9.3f} {f1:>9.3f} {precision:>9.3f} {recall:>9.3f}")
                    else:
                        print(f"   {sample_size:>6} {'ERROR':>9} {'ERROR':>9} {'ERROR':>9} {'ERROR':>9}")
        
        # Find optimal sample size
        best_sample_size = None
        best_accuracy = 0.0
        diminishing_returns_point = None
        
        accuracies = []
        for sample_size in self.sample_sizes:
            if sample_size in results['evaluation_results']:
                eval_result = results['evaluation_results'][sample_size]
                if 'error' not in eval_result:
                    accuracy = eval_result['accuracy']
                    accuracies.append((sample_size, accuracy))
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_sample_size = sample_size
        
        if best_sample_size:
            print(f"\n🏆 Best Performance:")
            print(f"   Sample size: {best_sample_size}")
            print(f"   Accuracy: {best_accuracy:.3f}")
            
            # Find diminishing returns point (where improvement < 0.01 per 100 samples)
            for i in range(1, len(accuracies)):
                prev_size, prev_acc = accuracies[i-1]
                curr_size, curr_acc = accuracies[i]
                improvement_rate = (curr_acc - prev_acc) / (curr_size - prev_size)
                if improvement_rate < 0.0001:  # Less than 0.01 improvement per 100 samples
                    diminishing_returns_point = prev_size
                    break
            
            if diminishing_returns_point:
                print(f"   Diminishing returns after: {diminishing_returns_point} samples")
        
        print(f"\n📁 Results saved to: {self.output_dir}")


def main():
    """Main function for sample size evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate classifier performance vs sample size')
    parser.add_argument('model', help='Model name')
    parser.add_argument('benchmark', help='Benchmark name')
    parser.add_argument('layer', type=int, help='Layer index')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results and plots')
    parser.add_argument('--sample-sizes', type=int, nargs='+', 
                       default=[1, 5, 10, 50, 100, 250, 500, 1000],
                       help='Sample sizes to evaluate')
    parser.add_argument('--test-limit', type=int, default=200,
                       help='Limit the number of test samples to use (default: 200)')
    
    args = parser.parse_args()
    
    print(f"🔬 Sample Size Evaluation")
    print(f"{'='*80}")
    
    # Initialize evaluator
    evaluator = SampleSizeEvaluator(
        model_name=args.model,
        benchmark_name=args.benchmark,
        layer=args.layer,
        output_dir=args.output_dir,
        test_limit=args.test_limit
    )
    
    # Override default sample sizes if provided
    if args.sample_sizes:
        evaluator.sample_sizes = sorted(args.sample_sizes)
    
    try:
        results = evaluator.run_sample_size_evaluation()
        
        if 'error' in results:
            print(f"\n❌ Evaluation failed: {results['error']}")
        else:
            print(f"\n✅ Evaluation completed successfully!")
            
    except KeyboardInterrupt:
        print(f"\n❌ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
