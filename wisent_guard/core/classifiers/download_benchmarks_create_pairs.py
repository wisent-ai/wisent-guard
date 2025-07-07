#!/usr/bin/env python3
"""
Download Benchmarks and Create Contrastive Pairs

This script downloads all available benchmarks and creates contrastive pairs
from them, saving the results in a 'benchmarks' folder in the same directory.

This should be run before generating classifiers to ensure all benchmark data
is available locally.
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import pickle

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parents[3]  # Go up to wisent-activation-guardrails root
sys.path.insert(0, str(project_root))

from wisent_guard.core.model import Model
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet

# Import benchmarks from the correct path
sys.path.insert(0, str(current_dir.parent / 'lm-harness-integration'))
from only_benchmarks import CORE_BENCHMARKS


class BenchmarkDownloader:
    """Download benchmarks and create contrastive pairs."""
    
    def __init__(self, model_name: str):
        """
        Initialize the benchmark downloader.
        
        Args:
            model_name: Name of the model to use for generating contrastive pairs
        """
        self.model_name = model_name
        self.script_dir = Path(__file__).parent
        self.benchmarks_dir = self.script_dir / "benchmarks"
        self.model = None
        
        # Create benchmarks directory
        self.benchmarks_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Benchmark Downloader and Contrastive Pair Generator")
        print(f"   Model: {model_name}")
        print(f"   Benchmarks directory: {self.benchmarks_dir}")
    
    def setup_model(self):
        """Initialize the model."""
        print(f"\n📦 Loading model: {self.model_name}")
        self.model = Model(self.model_name)
        print(f"   ✅ Model loaded successfully")
    
    def download_and_process_benchmark(
        self, 
        benchmark_name: str, 
        benchmark_config: Dict[str, Any],
        max_samples: int = 1000,
        force: bool = False
    ) -> Optional[str]:
        """
        Download a benchmark and create contrastive pairs from it.
        
        Args:
            benchmark_name: Name of the benchmark
            benchmark_config: Benchmark configuration
            max_samples: Maximum number of samples to process
            force: Force reprocessing even if data exists
            
        Returns:
            Path to saved benchmark data or None if failed
        """
        benchmark_file = self.benchmarks_dir / f"{benchmark_name}.pkl"
        
        # Check if benchmark already processed
        if benchmark_file.exists() and not force:
            print(f"   ⏩ Benchmark already processed: {benchmark_name}")
            return str(benchmark_file)
        
        print(f"   📥 Downloading and processing benchmark: {benchmark_name}")
        
        try:
            start_time = time.time()
            
            # Get the task name from config
            task_name = benchmark_config.get('task', benchmark_name)
            
            print(f"      🔄 Loading benchmark data for task: {task_name}")
            
            # Load benchmark data using existing infrastructure
            sys.path.insert(0, str(current_dir.parent / 'lm-harness-integration'))
            from populate_tasks import get_task_samples_for_analysis
            
            samples_result = get_task_samples_for_analysis(task_name, num_samples=max_samples)
            
            if "error" in samples_result:
                print(f"      ❌ Failed to load benchmark data: {samples_result['error']}")
                return None
            
            benchmark_data = samples_result.get("samples", [])
            print(f"      📊 Loaded {len(benchmark_data)} samples from {benchmark_name}")
            
            # Create contrastive pairs using existing infrastructure
            print(f"      🔗 Creating contrastive pairs...")
            
            try:
                # Use the existing benchmark loading infrastructure
                if benchmark_data:
                    # Create contrastive pairs from the loaded samples
                    contrastive_pairs = []
                    
                    for sample in benchmark_data[:min(500, len(benchmark_data))]:
                        # Extract prompt and target from sample
                        prompt = sample.get('input', sample.get('question', sample.get('prompt', str(sample))))
                        target = sample.get('target', sample.get('answer', sample.get('correct', '')))
                        
                        # Create a simple contrastive pair
                        contrastive_pairs.append({
                            'prompt': prompt,
                            'positive_response': target,
                            'negative_response': '[INCORRECT]',
                            'metadata': sample
                        })
                else:
                    contrastive_pairs = []
                    
            except Exception as e:
                print(f"      ⚠️  Error creating contrastive pairs: {e}")
                contrastive_pairs = []
            
            processing_time = time.time() - start_time
            
            # Prepare data to save
            processed_data = {
                'benchmark_name': benchmark_name,
                'task_name': task_name,
                'config': benchmark_config,
                'raw_data': benchmark_data,
                'contrastive_pairs': contrastive_pairs,
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'model_name': self.model_name,
                    'processing_time_seconds': processing_time,
                    'num_raw_samples': len(benchmark_data),
                    'num_contrastive_pairs': len(contrastive_pairs)
                }
            }
            
            # Save processed data
            with open(benchmark_file, 'wb') as f:
                pickle.dump(processed_data, f)
            
            print(f"      ✅ Saved: {benchmark_file.name}")
            print(f"      📊 Raw samples: {len(benchmark_data)}, "
                  f"Contrastive pairs: {len(contrastive_pairs)}, "
                  f"Time: {processing_time:.1f}s")
            
            return str(benchmark_file)
            
        except Exception as e:
            print(f"      ❌ Failed to process benchmark {benchmark_name}: {e}")
            return None
    
    def download_all_benchmarks(
        self, 
        benchmarks: Optional[List[str]] = None,
        max_samples: int = 1000,
        force: bool = False
    ):
        """
        Download and process all benchmarks.
        
        Args:
            benchmarks: List of benchmark names to process (None for all)
            max_samples: Maximum number of samples per benchmark
            force: Force reprocessing even if data exists
        """
        benchmarks = benchmarks or list(CORE_BENCHMARKS.keys())
        
        print(f"\n🏗️ Downloading and processing {len(benchmarks)} benchmarks")
        print(f"   Max samples per benchmark: {max_samples}")
        print(f"   Force reprocessing: {force}")
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Track results
        results = {
            'successful': [],
            'failed': [],
            'skipped': [],
            'total_time': 0
        }
        
        start_time = time.time()
        
        for i, benchmark_name in enumerate(benchmarks, 1):
            if benchmark_name not in CORE_BENCHMARKS:
                print(f"⚠️  Benchmark {benchmark_name} not found in CORE_BENCHMARKS")
                continue
                
            benchmark_config = CORE_BENCHMARKS[benchmark_name]
            print(f"\n[{i:2d}/{len(benchmarks)}] 🎯 Processing benchmark: {benchmark_name}")
            print(f"   Task: {benchmark_config.get('task', benchmark_name)}")
            print(f"   Tags: {benchmark_config.get('tags', [])}")
            
            try:
                result_path = self.download_and_process_benchmark(
                    benchmark_name, benchmark_config, max_samples, force
                )
                
                if result_path:
                    if "already processed" in str(result_path):
                        results['skipped'].append(benchmark_name)
                    else:
                        results['successful'].append(benchmark_name)
                else:
                    results['failed'].append(benchmark_name)
                    
            except KeyboardInterrupt:
                print(f"\n❌ Interrupted by user")
                return results
            except Exception as e:
                print(f"      ❌ Unexpected error: {e}")
                results['failed'].append(benchmark_name)
            
            # Progress update
            completed = i
            remaining = len(benchmarks) - i
            elapsed = time.time() - start_time
            if completed > 0:
                eta = (elapsed / completed) * remaining
                print(f"\n📊 Progress: {completed}/{len(benchmarks)} benchmarks completed")
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        # Print final summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print processing summary."""
        successful = len(results['successful'])
        failed = len(results['failed'])
        skipped = len(results['skipped'])
        total_time = results['total_time']
        
        print(f"\n{'='*80}")
        print(f"📊 BENCHMARK DOWNLOAD AND PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Successful: {successful}")
        print(f"⏩ Skipped (already exist): {skipped}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Benchmarks directory: {self.benchmarks_dir}")
        
        if results['successful']:
            print(f"\n🎯 Processed benchmarks saved as:")
            print(f"   {self.benchmarks_dir}/{{benchmark_name}}.pkl")
            print(f"\n📊 Each file contains:")
            print(f"   - Raw benchmark data")
            print(f"   - Generated contrastive pairs")
            print(f"   - Processing metadata")
        
        if results['failed']:
            print(f"\n❌ Failed benchmarks:")
            for benchmark in results['failed']:
                print(f"   - {benchmark}")
    
    def validate_downloaded_benchmarks(self) -> Dict[str, Any]:
        """
        Validate that downloaded benchmarks can be loaded.
        
        Returns:
            Validation results
        """
        print(f"\n🔍 Validating downloaded benchmarks...")
        
        validation_results = {
            'valid': True,
            'total_files': 0,
            'loadable': 0,
            'failed': [],
            'benchmarks': {}
        }
        
        benchmark_files = list(self.benchmarks_dir.glob("*.pkl"))
        validation_results['total_files'] = len(benchmark_files)
        
        for benchmark_file in benchmark_files:
            benchmark_name = benchmark_file.stem
            
            try:
                with open(benchmark_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Check if it has expected structure
                required_keys = ['benchmark_name', 'raw_data', 'contrastive_pairs', 'metadata']
                if all(key in data for key in required_keys):
                    validation_results['loadable'] += 1
                    validation_results['benchmarks'][benchmark_name] = {
                        'raw_samples': len(data['raw_data']),
                        'contrastive_pairs': len(data['contrastive_pairs']),
                        'task_name': data.get('task_name', benchmark_name)
                    }
                else:
                    validation_results['failed'].append(f"{benchmark_name}: Missing required keys")
                    
            except Exception as e:
                validation_results['failed'].append(f"{benchmark_name}: {e}")
        
        # Print validation summary
        print(f"   📊 Total benchmark files: {validation_results['total_files']}")
        print(f"   ✅ Loadable: {validation_results['loadable']}")
        print(f"   ❌ Failed: {len(validation_results['failed'])}")
        
        if validation_results['benchmarks']:
            print(f"\n📈 Benchmark statistics:")
            for name, stats in list(validation_results['benchmarks'].items())[:5]:
                print(f"   {name}: {stats['raw_samples']} samples, {stats['contrastive_pairs']} pairs")
            if len(validation_results['benchmarks']) > 5:
                print(f"   ... and {len(validation_results['benchmarks']) - 5} more")
        
        if validation_results['failed']:
            print(f"\n❌ Failed to load:")
            for failed in validation_results['failed'][:5]:
                print(f"      {failed}")
            if len(validation_results['failed']) > 5:
                print(f"      ... and {len(validation_results['failed']) - 5} more")
        
        validation_results['valid'] = len(validation_results['failed']) == 0
        return validation_results





def main():
    """Main function to run benchmark downloading."""
    parser = argparse.ArgumentParser(description='Download benchmarks and create contrastive pairs')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name to use for generating contrastive pairs')
    parser.add_argument('--benchmarks', nargs='+',
                       help='Specific benchmarks to download (default: all)')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum samples per benchmark')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing even if data exists')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing data without downloading')
    
    args = parser.parse_args()
    
    print(f"🚀 Benchmark Download and Contrastive Pair Generation")
    print(f"{'='*60}")
    
    # Initialize downloader
    downloader = BenchmarkDownloader(args.model)
    
    # Validate existing data
    if args.validate_only:
        validation_results = downloader.validate_downloaded_benchmarks()
        if validation_results['valid']:
            print(f"\n✅ All benchmark data is valid!")
        else:
            print(f"\n❌ Some benchmark data failed validation")
        return
    
    # Download and process benchmarks
    try:
        results = downloader.download_all_benchmarks(
            benchmarks=args.benchmarks,
            max_samples=args.max_samples,
            force=args.force
        )
        
        # Validate downloaded data
        print(f"\n🔍 Validating downloaded benchmarks...")
        validation_results = downloader.validate_downloaded_benchmarks()
        
        if validation_results['valid'] and len(results['successful']) > 0:
            print(f"\n🎉 SUCCESS! Downloaded and processed {len(results['successful'])} benchmarks!")
            print(f"   📁 Location: {downloader.benchmarks_dir}")
            print(f"   🤖 Ready for classifier generation!")
        else:
            print(f"\n⚠️  Some benchmarks may have issues. Check validation results above.")
        
    except KeyboardInterrupt:
        print(f"\n❌ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main() 