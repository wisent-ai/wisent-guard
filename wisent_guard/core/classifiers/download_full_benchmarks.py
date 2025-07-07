#!/usr/bin/env python3
"""
Full Benchmark Downloader

Downloads complete benchmarks from lm-eval-harness and saves them in a structured format.
Downloads the ENTIRE benchmark datasets, not just samples.

Usage:
    python download_full_benchmarks.py --benchmarks glue mmlu --force
    python download_full_benchmarks.py --all  # Download all benchmarks
"""

import os
import sys
import time
import argparse
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add current directory to path to import local modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent / 'lm-harness-integration'))

# Import the benchmark list
from only_benchmarks import CORE_BENCHMARKS

class FullBenchmarkDownloader:
    """Downloads complete benchmarks and saves them to disk."""
    
    def __init__(self, download_dir: str = "full_benchmarks"):
        """
        Initialize the benchmark downloader.
        
        Args:
            download_dir: Directory to save downloaded benchmarks
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.download_dir / "data"
        self.metadata_dir = self.download_dir / "metadata"
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Full Benchmark Downloader")
        print(f"📁 Download directory: {self.download_dir.absolute()}")
    
    def download_complete_benchmark(self, benchmark_name: str, benchmark_config: dict, force: bool = False) -> Optional[str]:
        """
        Download a complete benchmark dataset.
        
        Args:
            benchmark_name: Display name of the benchmark
            benchmark_config: Config dict with 'task' and 'tags' keys
            force: Force redownload even if exists
            
        Returns:
            Path to saved benchmark file, or None if failed
        """
        task_name = benchmark_config["task"]
        tags = benchmark_config.get("tags", [])
        
        # Check if already exists
        data_file = self.data_dir / f"{benchmark_name}.pkl"
        metadata_file = self.metadata_dir / f"{benchmark_name}_metadata.json"
        
        if data_file.exists() and metadata_file.exists() and not force:
            print(f"   ⏩ Skipping {benchmark_name} (already exists)")
            return str(data_file)
        
        print(f"   📥 Downloading complete benchmark: {benchmark_name}")
        print(f"      🔄 Loading full dataset for task: {task_name}")
        
        start_time = time.time()
        
        try:
            # Import lm_eval to download complete datasets
            import lm_eval
            from lm_eval import tasks
            
            # Get the task
            task_dict = tasks.get_task_dict([task_name])
            if task_name not in task_dict:
                print(f"      ❌ Task {task_name} not found in lm_eval")
                return None
            
            task = task_dict[task_name]
            
            # Download complete dataset - combine all splits into one unified dataset
            complete_data = {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "config": benchmark_config,
                "all_samples": [],
                "total_samples": 0,
                "splits_found": []
            }
            
            # Get all available document splits
            splits_to_try = ["test", "validation", "train", "dev"]
            
            for split in splits_to_try:
                try:
                    if hasattr(task, f"{split}_docs"):
                        docs_method = getattr(task, f"{split}_docs")
                        docs = list(docs_method())
                        
                        if docs:
                            print(f"      📊 Found {len(docs)} samples in {split} split")
                            complete_data["splits_found"].append(split)
                            
                            # Convert documents to serializable format and add to unified list
                            for i, doc in enumerate(docs):
                                if i % 1000 == 0 and i > 0:
                                    print(f"         Processing {split} {i}/{len(docs)}...")
                                
                                # Convert doc to dict, handling different doc types
                                if hasattr(doc, '__dict__'):
                                    doc_dict = doc.__dict__.copy()
                                elif isinstance(doc, dict):
                                    doc_dict = doc.copy()
                                else:
                                    doc_dict = {"content": str(doc)}
                                
                                # Add split origin info
                                doc_dict["_split_origin"] = split
                                
                                # Ensure all values are serializable
                                serializable_doc = {}
                                for key, value in doc_dict.items():
                                    try:
                                        json.dumps(value)  # Test if serializable
                                        serializable_doc[key] = value
                                    except (TypeError, ValueError):
                                        serializable_doc[key] = str(value)
                                
                                complete_data["all_samples"].append(serializable_doc)
                            
                            complete_data["total_samples"] += len(docs)
                
                except Exception as e:
                    print(f"      ⚠️  Could not load {split} split: {e}")
                    continue
            
            if complete_data["total_samples"] == 0:
                print(f"      ❌ No data found for {benchmark_name}")
                return None
            
            processing_time = time.time() - start_time
            
            # Add metadata
            metadata = {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "config": benchmark_config,
                "download_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "total_samples": complete_data["total_samples"],
                "splits_found": complete_data["splits_found"],
                "task_info": {
                    "description": getattr(task, "DESCRIPTION", "No description available"),
                    "citation": getattr(task, "CITATION", "No citation available"),
                    "homepage": getattr(task, "HOMEPAGE", "No homepage available"),
                }
            }
            
            # Save the complete benchmark data
            with open(data_file, 'wb') as f:
                pickle.dump(complete_data, f)
            
            # Save metadata as JSON
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"      ✅ Saved complete benchmark: {benchmark_name}")
            print(f"         📊 Total samples: {complete_data['total_samples']}")
            print(f"         📋 Splits found: {complete_data['splits_found']}")
            print(f"         ⏱️  Time: {processing_time:.1f}s")
            
            return str(data_file)
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"      ❌ Failed to download {benchmark_name}: {e}")
            print(f"         ⏱️  Time: {processing_time:.1f}s")
            return None
    
    def download_all_benchmarks(self, benchmarks: Optional[List[str]] = None, force: bool = False) -> Dict[str, Any]:
        """
        Download multiple complete benchmarks.
        
        Args:
            benchmarks: List of benchmark names to download, or None for all
            force: Force redownload even if exists
            
        Returns:
            Dictionary with download results
        """
        if benchmarks is None:
            benchmarks_to_download = CORE_BENCHMARKS
        else:
            benchmarks_to_download = {name: CORE_BENCHMARKS[name] for name in benchmarks if name in CORE_BENCHMARKS}
            
            # Check for invalid benchmarks
            invalid = [name for name in benchmarks if name not in CORE_BENCHMARKS]
            if invalid:
                print(f"⚠️  Invalid benchmarks (skipping): {invalid}")
        
        print(f"\n🏗️ Downloading {len(benchmarks_to_download)} complete benchmarks")
        print(f"   Force redownload: {force}")
        
        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "total_time": 0
        }
        
        total_start_time = time.time()
        
        for i, (benchmark_name, benchmark_config) in enumerate(benchmarks_to_download.items(), 1):
            print(f"\n[{i:2d}/{len(benchmarks_to_download)}] 🎯 Processing benchmark: {benchmark_name}")
            print(f"   Task: {benchmark_config['task']}")
            print(f"   Tags: {benchmark_config.get('tags', [])}")
            
            try:
                result_path = self.download_complete_benchmark(benchmark_name, benchmark_config, force)
                
                if result_path:
                    results["successful"].append(benchmark_name)
                else:
                    results["failed"].append(benchmark_name)
                    
            except Exception as e:
                print(f"   ❌ Exception downloading {benchmark_name}: {e}")
                results["failed"].append(benchmark_name)
            
            # Progress update
            elapsed = time.time() - total_start_time
            if i < len(benchmarks_to_download):
                eta = elapsed * (len(benchmarks_to_download) - i) / i
                print(f"\n📊 Progress: {i}/{len(benchmarks_to_download)} benchmarks completed")
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
        
        results["total_time"] = time.time() - total_start_time
        return results

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Download complete benchmarks from lm-eval-harness")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to download")
    group.add_argument("--all", action="store_true", help="Download all available benchmarks")
    
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists")
    parser.add_argument("--download-dir", default="full_benchmarks", help="Directory to save downloads")
    
    args = parser.parse_args()
    
    print("🚀 Full Benchmark Downloader")
    print("=" * 60)
    
    # Create downloader
    downloader = FullBenchmarkDownloader(download_dir=args.download_dir)
    
    # Download benchmarks
    try:
        if args.all:
            benchmarks_to_download = None
            print(f"📋 Downloading ALL {len(CORE_BENCHMARKS)} available benchmarks")
        else:
            benchmarks_to_download = args.benchmarks
            print(f"📋 Downloading {len(args.benchmarks)} specified benchmarks: {args.benchmarks}")
        
        results = downloader.download_all_benchmarks(
            benchmarks=benchmarks_to_download,
            force=args.force
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 FULL BENCHMARK DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {len(results['successful'])}")
        print(f"⏩ Skipped (already exist): {len(results['skipped'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        print(f"⏱️  Total time: {results['total_time']/60:.1f} minutes")
        print(f"📁 Download directory: {downloader.download_dir.absolute()}")
        
        if results["successful"]:
            print(f"\n🎯 Successfully downloaded:")
            for benchmark in results["successful"]:
                print(f"   ✅ {benchmark}")
        
        if results["failed"]:
            print(f"\n❌ Failed downloads:")
            for benchmark in results["failed"]:
                print(f"   ❌ {benchmark}")
        
        print(f"\n📊 Complete benchmark data saved in:")
        print(f"   📁 Data: {downloader.data_dir}")
        print(f"   📁 Metadata: {downloader.metadata_dir}")
        
        if results["successful"]:
            print(f"\n🎉 SUCCESS! Downloaded {len(results['successful'])} complete benchmarks!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 