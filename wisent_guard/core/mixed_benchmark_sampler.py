"""
Mixed Benchmark Sampler for tag-based random sampling across multiple benchmarks.

This module enables training and evaluation on random samples from multiple benchmarks
that share common tags (e.g., 'coding', 'reasoning', 'math').
"""

import random
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Suppress BigCode debug output
import builtins
_original_print = getattr(builtins, '_original_print', builtins.print)

def _quiet_print(*args, **kwargs):
    """Filter out BigCode debug messages."""
    message = ' '.join(str(arg) for arg in args)
    if any(x in message for x in ['DEBUG', 'Available tasks:', 'ERROR extracting', 'bigcode_eval']):
        return
    _original_print(*args, **kwargs)

# Store original print and patch
builtins._original_print = builtins.print
builtins.print = _quiet_print

try:
    from .lm_harness_integration.only_benchmarks import CORE_BENCHMARKS
except ImportError:
    # Try alternative import path
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, "lm-harness-integration"))
    from only_benchmarks import CORE_BENCHMARKS

from .contrastive_pairs import ContrastivePairSet
from .managed_cached_benchmarks import ManagedCachedBenchmarks, get_managed_cache

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """A single sample from a benchmark."""
    benchmark_name: str
    sample_data: Dict[str, Any]
    tags: List[str]


class MixedBenchmarkSampler:
    """
    Samples randomly from multiple benchmarks based on tags.
    
    This creates more robust classifiers by training on diverse data
    from multiple sources rather than a single benchmark.
    """
    
    def __init__(self, cache_dir: str = "./benchmark_cache"):
        """
        Initialize the mixed benchmark sampler.
        
        Args:
            cache_dir: Directory for cached benchmark data
        """
        self.cache_dir = cache_dir
        self.managed_cache = get_managed_cache(cache_dir)
        self._benchmark_registry = self._build_benchmark_registry()
    
    def _build_benchmark_registry(self) -> Dict[str, List[str]]:
        """Build a registry mapping tags to benchmark names."""
        tag_to_benchmarks = defaultdict(list)
        
        for benchmark_name, config in CORE_BENCHMARKS.items():
            tags = config.get("tags", [])
            for tag in tags:
                tag_to_benchmarks[tag].append(benchmark_name)
        
        return dict(tag_to_benchmarks)
    
    def get_benchmarks_by_tag(self, tag: str) -> List[str]:
        """Get all benchmarks that have a specific tag."""
        return self._benchmark_registry.get(tag, [])
    
    def get_benchmarks_by_tags(self, tags: List[str], mode: str = "any") -> List[str]:
        """
        Get benchmarks that match the given tags.
        
        Args:
            tags: List of tags to match
            mode: "any" (benchmark has at least one tag) or "all" (benchmark has all tags)
            
        Returns:
            List of benchmark names matching the criteria
        """
        if mode == "any":
            # Get benchmarks that have ANY of the specified tags
            matching_benchmarks = set()
            for tag in tags:
                matching_benchmarks.update(self.get_benchmarks_by_tag(tag))
            return list(matching_benchmarks)
        
        elif mode == "all":
            # Get benchmarks that have ALL of the specified tags
            if not tags:
                return []
            
            # Start with benchmarks that have the first tag
            matching_benchmarks = set(self.get_benchmarks_by_tag(tags[0]))
            
            # Intersect with benchmarks for each additional tag
            for tag in tags[1:]:
                matching_benchmarks &= set(self.get_benchmarks_by_tag(tag))
            
            return list(matching_benchmarks)
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'any' or 'all'")
    
    def sample_mixed_dataset(
        self,
        tags: List[str],
        total_samples: int,
        split_ratio: float = 0.8,
        random_seed: Optional[int] = None,
        tag_mode: str = "any",
        benchmark_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[List[BenchmarkSample], List[BenchmarkSample]]:
        """
        Sample a mixed dataset from benchmarks matching the given tags.
        
        Args:
            tags: Tags to filter benchmarks (e.g., ["coding", "python"])
            total_samples: Total number of samples to collect
            split_ratio: Train/test split ratio
            random_seed: Random seed for reproducibility
            tag_mode: "any" or "all" for tag matching
            benchmark_weights: Optional weights for sampling probability per benchmark
            
        Returns:
            Tuple of (train_samples, test_samples)
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Get matching benchmarks
        matching_benchmarks = self.get_benchmarks_by_tags(tags, mode=tag_mode)
        
        if not matching_benchmarks:
            raise ValueError(f"No benchmarks found with tags {tags} (mode={tag_mode})")
        
        logger.info(f"Found {len(matching_benchmarks)} benchmarks matching tags {tags}")
        logger.info(f"Matching benchmarks: {matching_benchmarks[:10]}...")  # Show first 10
        
        # Collect all available samples from matching benchmarks
        all_samples = []
        benchmark_sample_counts = {}
        
        # Skip benchmarks that require code execution permission
        code_execution_benchmarks = {"apps", "ds1000", "mercury"}
        
        for benchmark_name in matching_benchmarks:
            # Skip benchmarks that require code execution for safety
            if benchmark_name in code_execution_benchmarks:
                logger.info(f"Skipping {benchmark_name} (requires code execution permission)")
                continue
                
            try:
                # Get samples from this benchmark
                samples_per_benchmark = max(10, total_samples // len(matching_benchmarks))
                
                cached_samples = self.managed_cache.get_task_samples(
                    task_name=benchmark_name,
                    limit=samples_per_benchmark,
                    force_fresh=False
                )
                
                # Convert to BenchmarkSample objects
                for sample in cached_samples:
                    benchmark_sample = BenchmarkSample(
                        benchmark_name=benchmark_name,
                        sample_data=sample,
                        tags=CORE_BENCHMARKS[benchmark_name].get("tags", [])
                    )
                    all_samples.append(benchmark_sample)
                
                benchmark_sample_counts[benchmark_name] = len(cached_samples)
                
            except Exception as e:
                logger.warning(f"Failed to load samples from {benchmark_name}: {e}")
                continue
        
        if not all_samples:
            raise ValueError(f"No samples could be loaded from any benchmark with tags {tags}")
        
        logger.info(f"Collected {len(all_samples)} total samples from {len(benchmark_sample_counts)} benchmarks")
        for benchmark, count in benchmark_sample_counts.items():
            logger.debug(f"  {benchmark}: {count} samples")
        
        # Apply benchmark weights if provided
        if benchmark_weights:
            weighted_samples = []
            for sample in all_samples:
                weight = benchmark_weights.get(sample.benchmark_name, 1.0)
                # Duplicate samples based on weight (simple approach)
                weighted_samples.extend([sample] * int(weight))
            all_samples = weighted_samples
        
        # Randomly sample and shuffle
        if len(all_samples) > total_samples:
            all_samples = random.sample(all_samples, total_samples)
        else:
            # If we have fewer samples than requested, use all and log warning
            logger.warning(f"Only {len(all_samples)} samples available, requested {total_samples}")
        
        random.shuffle(all_samples)
        
        # Split into train/test
        split_point = int(len(all_samples) * split_ratio)
        train_samples = all_samples[:split_point]
        test_samples = all_samples[split_point:]
        
        # Log distribution
        train_dist = defaultdict(int)
        test_dist = defaultdict(int)
        
        for sample in train_samples:
            train_dist[sample.benchmark_name] += 1
        
        for sample in test_samples:
            test_dist[sample.benchmark_name] += 1
        
        logger.info(f"Train set: {len(train_samples)} samples from {len(train_dist)} benchmarks")
        logger.info(f"Test set: {len(test_samples)} samples from {len(test_dist)} benchmarks")
        
        return train_samples, test_samples
    
    def extract_contrastive_pairs_from_mixed_samples(
        self,
        samples: List[BenchmarkSample]
    ) -> List[Dict[str, Any]]:
        """
        Extract contrastive pairs from mixed benchmark samples.
        
        Args:
            samples: List of BenchmarkSample objects
            
        Returns:
            List of contrastive pairs with question, correct_answer, incorrect_answer
        """
        contrastive_pairs = []
        
        for sample in samples:
            try:
                # Each sample already has normalized QA pair from managed cache
                qa_pair = sample.sample_data.get("normalized", {})
                
                if qa_pair and all(k in qa_pair for k in ["question", "correct_answer", "incorrect_answer"]):
                    # Add benchmark source info
                    qa_pair["source_benchmark"] = sample.benchmark_name
                    qa_pair["tags"] = sample.tags
                    contrastive_pairs.append(qa_pair)
                else:
                    logger.warning(f"Invalid QA pair from {sample.benchmark_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract pair from {sample.benchmark_name}: {e}")
                continue
        
        logger.info(f"Extracted {len(contrastive_pairs)} contrastive pairs from mixed samples")
        
        return contrastive_pairs
    
    def create_mixed_contrastive_pair_set(
        self,
        tags: List[str],
        total_samples: int,
        name: Optional[str] = None,
        **kwargs
    ) -> ContrastivePairSet:
        """
        Create a ContrastivePairSet from mixed benchmark samples.
        
        Args:
            tags: Tags to filter benchmarks
            total_samples: Number of samples to include
            name: Name for the pair set (auto-generated if not provided)
            **kwargs: Additional arguments for sample_mixed_dataset
            
        Returns:
            ContrastivePairSet ready for training
        """
        # Sample mixed dataset
        train_samples, test_samples = self.sample_mixed_dataset(
            tags=tags,
            total_samples=total_samples,
            **kwargs
        )
        
        # Extract contrastive pairs
        all_samples = train_samples + test_samples
        contrastive_pairs = self.extract_contrastive_pairs_from_mixed_samples(all_samples)
        
        # Create name if not provided
        if name is None:
            name = f"mixed_{'_'.join(tags)}_{total_samples}_samples"
        
        # Create ContrastivePairSet
        return ContrastivePairSet.from_contrastive_pairs(
            name=name,
            contrastive_pairs=contrastive_pairs,
            task_type="mixed_benchmark"
        )


def sample_benchmarks_by_tag(
    tag: str,
    samples_per_benchmark: int = 10,
    max_benchmarks: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to sample from all benchmarks with a specific tag.
    
    Args:
        tag: Tag to filter benchmarks (e.g., "coding")
        samples_per_benchmark: Number of samples from each benchmark
        max_benchmarks: Maximum number of benchmarks to sample from
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping benchmark names to their samples
    """
    sampler = MixedBenchmarkSampler()
    
    # Get all benchmarks with the tag
    benchmarks = sampler.get_benchmarks_by_tag(tag)
    
    if max_benchmarks and len(benchmarks) > max_benchmarks:
        if random_seed is not None:
            random.seed(random_seed)
        benchmarks = random.sample(benchmarks, max_benchmarks)
    
    # Sample from each benchmark
    results = {}
    cache = get_managed_cache()
    
    for benchmark_name in benchmarks:
        try:
            samples = cache.get_task_samples(
                task_name=benchmark_name,
                limit=samples_per_benchmark,
                force_fresh=False
            )
            results[benchmark_name] = samples
            logger.info(f"Sampled {len(samples)} from {benchmark_name}")
            
        except Exception as e:
            logger.warning(f"Failed to sample from {benchmark_name}: {e}")
            continue
    
    return results