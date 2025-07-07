#!/usr/bin/env python3
"""
Generate Classifiers for All Downloaded Benchmarks and Layers

This script uses the CLI tasks command to generate classifiers for all 
downloaded benchmarks across multiple layers with precise control over
datasets, layers, and training parameters.

Directory structure created:
- Uses CLI's existing classifier storage system
"""

import os
import sys
import time
import argparse
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parents[3]  # Go up to wisent-activation-guardrails root
sys.path.insert(0, str(project_root))


class CLIBatchClassifierGenerator:
    """Generate classifiers using CLI tasks command for all downloaded benchmarks."""
    
    def __init__(self, model_name: str, classifier_dir: Optional[str] = None):
        """
        Initialize the CLI-based classifier generator.
        
        Args:
            model_name: Name of the model to generate classifiers for
            classifier_dir: Directory to save classifiers (None for default structure)
        """
        self.model_name = model_name
        self.script_dir = Path(__file__).parent
        self.benchmarks_dir = self.script_dir / "full_benchmarks" / "data"
        self.project_root = project_root
        
        # Default to wisent_guard/core/classifiers/ structure
        if classifier_dir is None:
            self.classifier_dir = self.script_dir
        else:
            self.classifier_dir = Path(classifier_dir)
        
        print(f"🚀 CLI-Based Batch Classifier Generator")
        print(f"   Model: {model_name}")
        print(f"   Benchmarks source: {self.benchmarks_dir}")
        if classifier_dir is None:
            print(f"   Classifier output: {self.classifier_dir} (hierarchical structure)")
        else:
            print(f"   Classifier output: {self.classifier_dir}")
        print(f"   Using CLI tasks command")
        
        # Create classifier directory
        self.classifier_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-download benchmarks if they don't exist
        if not self.benchmarks_dir.exists():
            print(f"   📥 Benchmarks directory not found: {self.benchmarks_dir}")
            print(f"   🚀 Auto-downloading all benchmarks (this may take a few minutes)...")
            self._auto_download_benchmarks()
    
    def _auto_download_benchmarks(self):
        """Automatically download benchmarks if they don't exist."""
        try:
            # Import the downloader
            from download_full_benchmarks import FullBenchmarkDownloader
            
            print(f"   📦 Initializing benchmark downloader...")
            downloader = FullBenchmarkDownloader(download_dir="full_benchmarks")
            
            print(f"   📥 Downloading all available benchmarks...")
            results = downloader.download_all_benchmarks(force=False)
            
            # Print summary
            successful = len(results.get('successful', []))
            failed = len(results.get('failed', []))
            total_time = results.get('total_time', 0)
            
            print(f"   ✅ Benchmark download complete!")
            print(f"      🎯 Successfully downloaded: {successful} benchmarks")
            if failed > 0:
                print(f"      ❌ Failed: {failed} benchmarks")
            print(f"      ⏱️  Total time: {total_time/60:.1f} minutes")
            
            # Verify download worked
            if not self.benchmarks_dir.exists():
                raise FileNotFoundError(f"Benchmark download failed - directory still doesn't exist: {self.benchmarks_dir}")
            
            print(f"   🎉 Benchmarks ready for classifier training!")
            
        except ImportError as e:
            raise FileNotFoundError(
                f"❌ Could not import benchmark downloader: {e}\n"
                f"\n🔧 To fix this, please install the required dependencies and download benchmarks:\n"
                f"   pip install lm-eval[api]\n"
                f"   python download_full_benchmarks.py --all\n"
                f"\n🎯 Then re-run this script to continue classifier training."
            )
        except Exception as e:
            if "No module named 'lm_eval'" in str(e):
                raise FileNotFoundError(
                    f"❌ Automatic benchmark download failed - missing lm_eval dependency\n"
                    f"\n🔧 To fix this, please install the required dependencies and download benchmarks:\n"
                    f"   pip install lm-eval[api]\n"
                    f"   python download_full_benchmarks.py --all\n"
                    f"\n🎯 Then re-run this script to continue classifier training."
                )
            else:
                raise FileNotFoundError(
                    f"❌ Automatic benchmark download failed: {e}\n"
                    f"\n🔧 To fix this, please manually download benchmarks:\n"
                    f"   python download_full_benchmarks.py --all\n"
                    f"\n🎯 Then re-run this script to continue classifier training."
                )
    
    def detect_model_layers(self) -> List[int]:
        """Detect all available model layers."""
        print(f"\n📦 Detecting model layers for: {self.model_name}")
        
        # Import model to get layer count
        try:
            from wisent_guard.core.model import Model
            model = Model(self.model_name)
            num_layers = model.model.config.num_hidden_layers
            layers = list(range(num_layers))
            print(f"   🧠 Detected {num_layers} layers: {layers}")
            return layers
        except Exception as e:
            print(f"   ⚠️ Failed to detect layers: {e}")
            print(f"   🔄 Using default range 0-31")
            return list(range(32))  # Default fallback
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available downloaded benchmarks that work with CLI."""
        if not self.benchmarks_dir.exists():
            return []
        
        pkl_files = list(self.benchmarks_dir.glob("*.pkl"))
        all_benchmarks = [f.stem for f in pkl_files]
        
        # Filter out benchmarks that are not available in CLI
        cli_unavailable = {'math_qa', 'mmmlu', 'naturalqs', 'squad2'}
        available_benchmarks = [b for b in all_benchmarks if b not in cli_unavailable]
        
        # Log filtered benchmarks
        filtered_out = [b for b in all_benchmarks if b in cli_unavailable]
        if filtered_out:
            print(f"   🚫 Filtered out CLI-unavailable benchmarks: {', '.join(filtered_out)}")
        
        return available_benchmarks
    
    def create_classifier_path(self, benchmark_name: str, layer: int) -> str:
        """Create classifier save path in hierarchical structure."""
        # Create directory structure: wisent_guard/core/classifiers/{model}/{benchmark}/layer_{layer}.pkl
        model_safe = self.model_name.replace('/', '_').replace('-', '_')
        classifier_subdir = self.classifier_dir / model_safe / benchmark_name
        classifier_subdir.mkdir(parents=True, exist_ok=True)
        
        filename = f"layer_{layer}.pkl"
        return str(classifier_subdir / filename)
    
    def run_cli_command(self, benchmark_name: str, layer: int, 
                       split_ratio: float = 0.8, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Run CLI tasks command to train classifier for specific benchmark and layer.
        
        Args:
            benchmark_name: Name of the benchmark
            layer: Layer to train classifier for
            split_ratio: Train/validation split ratio
            limit: Limit number of samples (None for all)
            
        Returns:
            Dictionary with result information
        """
        print(f"      🔧 Training classifier: {benchmark_name} at layer {layer}")
        
        # Create classifier save path
        classifier_path = self.create_classifier_path(benchmark_name, layer)
        
        # Construct CLI command using tasks
        cmd = [
            sys.executable, "-m", "wisent_guard", "tasks", benchmark_name,
            "--model", self.model_name,
            "--layer", str(layer),
            "--train-only",  # Only train, don't run inference
            "--save-classifier", classifier_path,
            "--split-ratio", str(split_ratio),
            "--verbose"
        ]
        
        # Add limit if specified
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        print(f"         Command: {' '.join(cmd)}")
        
        try:
            # Run CLI command with real-time output
            start_time = time.time()
            print(f"         📺 Starting CLI command with real-time output...")
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                # Remove capture_output=True to show real-time output
                stdout=None,  # Output goes to console
                stderr=None,  # Errors go to console
                # No timeout - let it run as long as needed
            )
            execution_time = time.time() - start_time
            
            # Check if successful
            if result.returncode == 0:
                # Verify classifier file was created
                # CLI appends _pkl_layer_{layer} to the filename, so check for that pattern
                model_safe = self.model_name.replace('/', '_').replace('-', '_')
                classifier_subdir = self.classifier_dir / model_safe / benchmark_name
                expected_path = classifier_subdir / f"layer_{layer}_pkl_layer_{layer}.pkl"
                
                if expected_path.exists():
                    actual_path = str(expected_path)
                    print(f"         ✅ Success ({execution_time:.1f}s): {actual_path}")
                    return {
                        'status': 'success',
                        'execution_time': execution_time,
                        'classifier_path': actual_path
                    }
                else:
                    print(f"         ⚠️ CLI completed but no classifier file found")
                    print(f"         Expected: {expected_path}")
                    # List files in directory to help debug
                    if classifier_subdir.exists():
                        files = list(classifier_subdir.glob("*.pkl"))
                        print(f"         Found files: {[f.name for f in files]}")
                    return {
                        'status': 'no_output',
                        'execution_time': execution_time
                    }
            else:
                print(f"         ❌ CLI failed (code {result.returncode})")
                print(f"         Error: See output above for details")
                return {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'return_code': result.returncode
                }
                
        except Exception as e:
            print(f"         ❌ Execution failed: {e}")
            return {
                'status': 'error',
                'execution_time': 0,
                'error': str(e)
            }
    
    def generate_classifiers_for_benchmark(self, benchmark_name: str, layers: List[int], 
                                         split_ratio: float = 0.8, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate classifiers for all layers of a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            layers: List of layers to process
            split_ratio: Train/validation split ratio
            limit: Limit number of samples (None for all)
            
        Returns:
            Dictionary with results for this benchmark
        """
        print(f"\n🎯 Processing benchmark: {benchmark_name}")
        print(f"   Layers: {layers}")
        print(f"   Split ratio: {split_ratio} (train/validation)")
        if limit:
            print(f"   Sample limit: {limit}")
        else:
            print(f"   Sample limit: All available")
        
        results = {
            'benchmark_name': benchmark_name,
            'layers_processed': [],
            'successful': [],
            'failed': [],
            'errors': [],
            'no_output': [],
            'classifier_paths': [],
            'total_time': 0
        }
        
        start_time = time.time()
        
        for i, layer in enumerate(layers, 1):
            print(f"   [{i}/{len(layers)}] Layer {layer}:")
            
            # Run CLI command for this layer
            cli_result = self.run_cli_command(benchmark_name, layer, split_ratio, limit)
            
            results['layers_processed'].append(layer)
            
            if cli_result['status'] == 'success':
                results['successful'].append((layer, cli_result['execution_time']))
                results['classifier_paths'].append(cli_result['classifier_path'])
            elif cli_result['status'] == 'failed':
                results['failed'].append((layer, cli_result.get('return_code', -1)))
            elif cli_result['status'] == 'no_output':
                results['no_output'].append(layer)
            else:
                results['errors'].append((layer, cli_result.get('error', 'Unknown error')))
        
        results['total_time'] = time.time() - start_time
        
        # Print benchmark summary
        print(f"\n   📊 Benchmark {benchmark_name} Summary:")
        print(f"      ✅ Successful: {len(results['successful'])}")
        print(f"      ❌ Failed: {len(results['failed'])}")
        print(f"      ⚠️ No output: {len(results['no_output'])}")
        print(f"      🔥 Errors: {len(results['errors'])}")
        print(f"      ⏱️ Total time: {results['total_time']/60:.1f} minutes")
        
        return results
    
    def generate_all_classifiers(self, 
                               benchmarks: Optional[List[str]] = None,
                               layers: Optional[List[int]] = None,
                               split_ratio: float = 0.8,
                               limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate classifiers for all benchmarks and layers using CLI tasks command.
        
        Args:
            benchmarks: List of benchmark names to process (None for all available)
            layers: List of layers to process (None for default)
            split_ratio: Train/validation split ratio
            limit: Limit number of samples per benchmark (None for all)
            
        Returns:
            Dictionary with complete results
        """
        # Get available benchmarks
        available_benchmarks = self.get_available_benchmarks()
        
        if not available_benchmarks:
            print(f"❌ No downloaded benchmarks found in {self.benchmarks_dir}")
            print(f"   Please run download_full_benchmarks.py first!")
            return {}
        
        benchmarks = benchmarks or available_benchmarks
        
        # Filter to only use benchmarks that are actually downloaded
        benchmarks = [b for b in benchmarks if b in available_benchmarks]
        
        if not benchmarks:
            print(f"❌ None of the specified benchmarks are available in {self.benchmarks_dir}")
            print(f"   Available benchmarks: {available_benchmarks}")
            return {}
        
        # Detect model layers if not provided
        if layers is None:
            layers = self.detect_model_layers()
        
        print(f"\n🏗️ Generating classifiers using CLI tasks command")
        print(f"   Available benchmarks: {len(available_benchmarks)}")
        print(f"   Processing benchmarks: {len(benchmarks)}")
        print(f"   Layers: {layers}")
        print(f"   Split ratio: {split_ratio}")
        if limit:
            print(f"   Sample limit per benchmark: {limit}")
        print(f"   Total operations: {len(benchmarks) * len(layers)}")
        
        # Track overall results
        overall_results = {
            'benchmarks_processed': [],
            'total_successful': 0,
            'total_failed': 0,
            'total_errors': 0,
            'total_no_output': 0,
            'total_time': 0,
            'all_classifier_paths': [],
            'benchmark_details': []
        }
        
        overall_start_time = time.time()
        
        for i, benchmark_name in enumerate(benchmarks, 1):
            print(f"\n[{i:2d}/{len(benchmarks)}] 📊 Processing benchmark: {benchmark_name}")
            
            try:
                # Generate classifiers for this benchmark
                benchmark_results = self.generate_classifiers_for_benchmark(
                    benchmark_name, layers, split_ratio, limit
                )
                
                # Update overall results
                overall_results['benchmarks_processed'].append(benchmark_name)
                overall_results['total_successful'] += len(benchmark_results['successful'])
                overall_results['total_failed'] += len(benchmark_results['failed'])
                overall_results['total_errors'] += len(benchmark_results['errors'])
                overall_results['total_no_output'] += len(benchmark_results['no_output'])
                overall_results['all_classifier_paths'].extend(benchmark_results['classifier_paths'])
                overall_results['benchmark_details'].append(benchmark_results)
                
            except KeyboardInterrupt:
                print(f"\n❌ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Error processing {benchmark_name}: {e}")
                overall_results['total_errors'] += len(layers)
                continue
            
            # Progress update
            completed = i
            remaining = len(benchmarks) - i
            elapsed = time.time() - overall_start_time
            if completed > 0:
                eta = (elapsed / completed) * remaining
                print(f"\n📊 Progress: {completed}/{len(benchmarks)} benchmarks completed")
                print(f"   ⏱️ Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
        
        overall_results['total_time'] = time.time() - overall_start_time
        
        # Print final summary
        self._print_final_summary(overall_results)
        
        return overall_results
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final generation summary."""
        print(f"\n{'='*80}")
        print(f"📊 CLI TASKS CLASSIFIER GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Total Successful: {results['total_successful']}")
        print(f"❌ Total Failed: {results['total_failed']}")
        print(f"⚠️ Total No Output: {results['total_no_output']}")
        print(f"🔥 Total Errors: {results['total_errors']}")
        print(f"📊 Benchmarks Processed: {len(results['benchmarks_processed'])}")
        print(f"⏱️ Total Time: {results['total_time']/60:.1f} minutes")
        
        print(f"\n🎯 Method: CLI tasks command with precise control")
        print(f"   ✅ Exact benchmark selection")
        print(f"   ✅ Exact layer specification")
        print(f"   ✅ Complete dataset usage")
        print(f"   ✅ Configurable train/validation splits")
        
        if results['all_classifier_paths']:
            print(f"\n💾 Generated {len(results['all_classifier_paths'])} classifier files:")
            print(f"   📁 Directory: {self.classifier_dir}")
            print(f"   🔧 Files: {len(results['all_classifier_paths'])} classifiers saved")
        
        if results['benchmark_details']:
            print(f"\n📋 Per-Benchmark Results:")
            for detail in results['benchmark_details']:
                name = detail['benchmark_name']
                successful = len(detail['successful'])
                total_layers = len(detail['layers_processed'])
                print(f"   • {name}: {successful}/{total_layers} layers successful")
        
        print(f"\n🤖 All classifiers saved to: {self.classifier_dir}")
        print(f"   📁 Structure: {self.classifier_dir}/{{model}}/{{benchmark}}/layer_{{layer}}.pkl")


def main():
    """Main function to run CLI-based classifier generation."""
    parser = argparse.ArgumentParser(description='Generate classifiers using CLI tasks for all downloaded benchmarks')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name to generate classifiers for')
    parser.add_argument('--benchmarks', nargs='+',
                       help='Specific benchmarks to generate (default: all downloaded)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                       help='Layers to generate classifiers for (default: all model layers)')
    parser.add_argument('--split-ratio', type=float, default=0.8,
                       help='Train/validation split ratio (default: 0.8)')
    parser.add_argument('--limit', type=int, default=1000,
                       help='Limit number of samples per benchmark (default: 1000)')
    parser.add_argument('--classifier-dir', type=str, default=None,
                       help='Directory to save classifiers (default: wisent_guard/core/classifiers/)')
    
    args = parser.parse_args()
    
    print(f"🚀 CLI Tasks Classifier Generation")
    print(f"{'='*60}")
    
    # Initialize generator
    generator = CLIBatchClassifierGenerator(args.model, args.classifier_dir)
    
    # Show available benchmarks
    available = generator.get_available_benchmarks()
    print(f"📊 Available downloaded benchmarks: {len(available)}")
    for benchmark in available[:10]:  # Show first 10
        print(f"   • {benchmark}")
    if len(available) > 10:
        print(f"   ... and {len(available) - 10} more")
    
    # Generate classifiers
    try:
        results = generator.generate_all_classifiers(
            benchmarks=args.benchmarks,
            layers=args.layers,
            split_ratio=args.split_ratio,
            limit=args.limit
        )
        
        if results and results['total_successful'] > 0:
            print(f"\n🎉 SUCCESS! Generated {results['total_successful']} classifiers!")
            if args.classifier_dir:
                print(f"   📁 Saved to: {args.classifier_dir}")
            else:
                print(f"   📁 Saved to: wisent_guard/core/classifiers/")
            print(f"   📊 Using complete downloaded benchmark data")
            print(f"   🎯 With precise layer and dataset control")
        else:
            print(f"\n⚠️ No classifiers were generated successfully")
        
    except KeyboardInterrupt:
        print(f"\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
