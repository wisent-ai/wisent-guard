"""
Managed Cached Benchmarks service for intelligent dataset downloading and caching.

This service controls how much of each benchmark is downloaded based on the limit parameter:
- If limit=5, download only 5 samples
- If limit=3 and we have 5 cached, reuse cached samples
- If limit=10 and we have 5 cached, download 5 more
- Hard errors for unsupported benchmarks, no fallbacks
"""

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark_extractors import EXTRACTORS, get_extractor

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Base exception for benchmark-related errors."""


class UnsupportedBenchmarkError(BenchmarkError):
    """Raised when benchmark has no adapter."""


class SampleNormalizationError(BenchmarkError):
    """Raised when sample normalization fails."""


class InsufficientSamplesError(BenchmarkError):
    """Raised when benchmark doesn't have enough samples."""


class CacheCorruptionError(BenchmarkError):
    """Raised when cache data is corrupted."""


@dataclass
class CacheInfo:
    """Information about cached benchmark data."""

    task_name: str
    samples_count: int
    last_updated: datetime
    cache_version: str
    chunks: List[str]  # List of chunk filenames


@dataclass
class CacheMetadata:
    """Global cache metadata."""

    version: str
    created_at: datetime
    last_cleanup: datetime
    tasks: Dict[str, CacheInfo]


class ManagedCachedBenchmarks:
    """
    Service for intelligent benchmark downloading and caching.

    Features:
    - Downloads only what's needed based on limit parameter
    - Reuses cached data when possible
    - Incremental downloads for growing limits
    - Hard errors for unsupported benchmarks
    - Chunk-based storage for efficiency
    """

    CACHE_VERSION = "1.0"
    CHUNK_SIZE = 25  # Samples per chunk
    MAX_CACHE_AGE_DAYS = 30
    SUPPORTED_BENCHMARKS = None  # Will be initialized in __init__

    def __init__(self, cache_dir: str = "./benchmark_cache"):
        """
        Initialize the managed cache service.

        Args:
            cache_dir: Directory to store cached benchmark data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()

        # Initialize supported benchmarks including BigCode tasks
        if ManagedCachedBenchmarks.SUPPORTED_BENCHMARKS is None:
            supported = set(EXTRACTORS.keys())
            try:
                from .bigcode_integration import BigCodeTaskLoader

                loader = BigCodeTaskLoader()
                supported.update(loader.TASK_MAPPING.keys())
            except ImportError:
                pass
            ManagedCachedBenchmarks.SUPPORTED_BENCHMARKS = supported

        # Validate all supported benchmarks have extractors
        self._validate_extractor_registry()

    def _validate_extractor_registry(self):
        """Ensure every supported benchmark has a working extractor."""
        for benchmark in self.SUPPORTED_BENCHMARKS:
            try:
                extractor = get_extractor(benchmark)
                if not hasattr(extractor, "extract_qa_pair"):
                    raise AttributeError(f"Extractor for {benchmark} missing extract_qa_pair method")
            except Exception as e:
                raise BenchmarkError(f"Invalid extractor for supported benchmark '{benchmark}': {e}")

    def get_task_samples(self, task_name: str, limit: int, force_fresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get samples for a task, using intelligent caching.

        Args:
            task_name: Name of the benchmark task
            limit: Number of samples needed
            force_fresh: Force fresh download even if cached

        Returns:
            List of normalized sample dictionaries

        Raises:
            UnsupportedBenchmarkError: If task has no extractor
            InsufficientSamplesError: If benchmark doesn't have enough samples
            SampleNormalizationError: If sample extraction fails
        """
        # Hard error for unsupported benchmarks
        if task_name not in self.SUPPORTED_BENCHMARKS:
            raise UnsupportedBenchmarkError(
                f"No extractor found for benchmark '{task_name}'. "
                f"Supported benchmarks: {sorted(self.SUPPORTED_BENCHMARKS)}"
            )

        if limit <= 0:
            return []

        logger.info(f"Getting {limit} samples for task '{task_name}'")

        # Check cache status
        cached_count = self._get_cached_sample_count(task_name)
        logger.info(f"Found {cached_count} cached samples for '{task_name}'")

        if force_fresh:
            logger.info(f"Force fresh download requested for '{task_name}'")
            self._clear_task_cache(task_name)
            cached_count = 0

        # Decision logic
        if cached_count >= limit:
            # Case 1: We have enough - load from cache
            logger.info(f"Loading {limit} samples from cache for '{task_name}'")
            return self._load_cached_samples(task_name, limit)

        if cached_count > 0 and limit <= cached_count * 2:
            # Case 2: We have some, need a bit more - incremental download
            needed = limit - cached_count
            logger.info(f"Incremental download: need {needed} more samples for '{task_name}'")

            new_samples = self._download_samples(task_name, needed, start_offset=cached_count)
            self._append_to_cache(task_name, new_samples)

            return self._load_cached_samples(task_name, limit)

        # Case 3: Major mismatch - fresh download
        logger.info(f"Fresh download: getting {limit} samples for '{task_name}'")
        self._clear_task_cache(task_name)

        new_samples = self._download_samples(task_name, limit, start_offset=0)
        self._save_to_cache(task_name, new_samples)

        return new_samples

    def _get_cached_sample_count(self, task_name: str) -> int:
        """Get number of cached samples for a task."""
        if task_name not in self._metadata.tasks:
            return 0
        return self._metadata.tasks[task_name].samples_count

    def _download_samples(self, task_name: str, limit: int, start_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Download samples from a benchmark task.

        Args:
            task_name: Name of the benchmark
            limit: Number of samples to download
            start_offset: Offset to start downloading from

        Returns:
            List of normalized samples
        """
        logger.info(f"Downloading {limit} samples for '{task_name}' (offset: {start_offset})")

        # Get extractor (hard error if not found)
        extractor = get_extractor(task_name)

        # Load raw task from lm-eval, BigCode, or TaskInterface
        try:
            task = self._load_lm_eval_task(task_name)
        except Exception as e:
            # Check if it's a BigCode task
            from .bigcode_integration import BigCodeTaskLoader

            loader = BigCodeTaskLoader()
            if loader.is_bigcode_task(task_name):
                task = self._load_bigcode_task(task_name)
            # Check if it's a TaskInterface task (like AIME, HLE, etc.)
            elif self._is_taskinterface_task(task_name):
                task = self._load_taskinterface_task(task_name, limit=start_offset + limit)
            else:
                raise BenchmarkError(f"Failed to load task '{task_name}' from lm-eval: {e}")

        # Get sample iterator
        try:
            sample_iterator = self._get_task_sample_iterator(task, start_offset + limit)
        except Exception as e:
            raise BenchmarkError(f"Failed to get samples from task '{task_name}': {e}")

        # Skip to start offset
        for _ in range(start_offset):
            try:
                next(sample_iterator)
            except StopIteration:
                raise InsufficientSamplesError(
                    f"Task '{task_name}' only has {start_offset} samples, cannot skip to offset {start_offset}"
                )

        # Extract samples
        samples = []
        for i in range(limit):
            try:
                raw_sample = next(sample_iterator)
            except StopIteration:
                raise InsufficientSamplesError(
                    f"Task '{task_name}' only has {start_offset + i} samples, but {start_offset + limit} were requested"
                )

            # Extract contrastive pair using extractor (includes both correct and incorrect answers)
            try:
                qa_pair = extractor.extract_contrastive_pair(raw_sample, task)
                if qa_pair is None:
                    raise ValueError("Extractor returned None")
            except Exception as e:
                raise SampleNormalizationError(f"Failed to normalize sample {start_offset + i} from '{task_name}': {e}")

            samples.append(
                {
                    "id": f"sample_{start_offset + i:03d}",
                    "raw_data": raw_sample,
                    "normalized": qa_pair,
                    "extracted_at": datetime.now().isoformat(),
                }
            )

        logger.info(f"Successfully downloaded {len(samples)} samples for '{task_name}'")
        return samples

    def _load_bigcode_task(self, task_name: str):
        """Load task from bigcode-evaluation-harness."""
        from .bigcode_integration import BigCodeTaskLoader

        loader = BigCodeTaskLoader()

        # For APPS, we need to check if HF_ALLOW_CODE_EVAL is set
        if task_name == "apps" and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
            print(f"\n⚠️  Task '{task_name}' requires code evaluation permission.")
            print("This task will execute model-generated code which could be potentially harmful.")
            print("Please review the safety information at: https://arxiv.org/abs/2107.03374")
            response = input("\nDo you want to enable code evaluation? (yes/no): ").strip().lower()

            if response == "yes":
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                print("✅ Code evaluation enabled for this session.")
            else:
                raise BenchmarkError(f"Code evaluation permission denied for task '{task_name}'")

        return loader.load_task(task_name)

    def _load_lm_eval_task(self, task_name: str):
        """Load task from lm-eval-harness."""
        try:
            from lm_eval.tasks import get_task_dict

            # First check if it's a BigCode task before trying lm-eval
            from .bigcode_integration import BigCodeTaskLoader

            loader = BigCodeTaskLoader()
            if loader.is_bigcode_task(task_name):
                raise ValueError(f"Task '{task_name}' is a BigCode task. Use --bigcode flag or BigCodeTaskLoader")

            # Check if we need HF_ALLOW_CODE_EVAL for code evaluation tasks
            code_eval_tasks = ["mbpp", "mbpp_plus", "humaneval", "humaneval_plus"]
            if task_name in code_eval_tasks and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
                print(f"\n⚠️  Task '{task_name}' requires code evaluation permission.")
                print("This task will execute model-generated code which could be potentially harmful.")
                print("Please review the safety information at: https://arxiv.org/abs/2107.03374")
                response = input("\nDo you want to enable code evaluation? (yes/no): ").strip().lower()

                if response == "yes":
                    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                    print("✅ Code evaluation enabled for this session.")
                else:
                    raise BenchmarkError(f"Code evaluation permission denied for task '{task_name}'")

            task_dict = get_task_dict([task_name])
            if task_name not in task_dict:
                raise ValueError(f"Task '{task_name}' not found in lm-eval")

            return task_dict[task_name]
        except ImportError as e:
            raise BenchmarkError("lm-evaluation-harness not available") from e

    def _is_taskinterface_task(self, task_name: str) -> bool:
        """Check if task is a TaskInterface-based task by checking the task registry."""
        from .task_interface import list_tasks

        return task_name in list_tasks()

    def _load_taskinterface_task(self, task_name: str, limit: Optional[int] = None):
        """Load TaskInterface task using the central task registry."""
        from .task_interface import get_task

        try:
            return get_task(task_name, limit=limit)
        except Exception as e:
            raise BenchmarkError(f"Failed to load TaskInterface task '{task_name}': {e}")

    def _get_task_sample_iterator(self, task, limit: int) -> Iterator[Dict[str, Any]]:
        """Get iterator over task samples."""
        # Try different document sources in order of preference
        if hasattr(task, "validation_docs") and task.has_validation_docs():
            docs = task.validation_docs()
        elif hasattr(task, "test_docs") and task.has_test_docs():
            docs = task.test_docs()
        elif hasattr(task, "training_docs") and task.has_training_docs():
            docs = task.training_docs()
        else:
            raise BenchmarkError("No document source available for task")

        # Convert to iterator and limit
        doc_iter = iter(docs)
        for i, doc in enumerate(doc_iter):
            if i >= limit:
                break
            yield doc

    def _save_to_cache(self, task_name: str, samples: List[Dict[str, Any]]):
        """Save samples to cache in chunks."""
        task_dir = self.cache_dir / task_name
        task_dir.mkdir(exist_ok=True)

        # Save in chunks
        chunks = []
        for i in range(0, len(samples), self.CHUNK_SIZE):
            chunk_samples = samples[i : i + self.CHUNK_SIZE]
            chunk_filename = f"samples_{i + 1}_to_{i + len(chunk_samples)}.json"
            chunk_path = task_dir / chunk_filename

            with open(chunk_path, "w") as f:
                json.dump(chunk_samples, f, indent=2)

            chunks.append(chunk_filename)

        # Update metadata
        cache_info = CacheInfo(
            task_name=task_name,
            samples_count=len(samples),
            last_updated=datetime.now(),
            cache_version=self.CACHE_VERSION,
            chunks=chunks,
        )

        self._metadata.tasks[task_name] = cache_info
        self._save_metadata()

        logger.info(f"Saved {len(samples)} samples to cache for '{task_name}' in {len(chunks)} chunks")

    def _append_to_cache(self, task_name: str, new_samples: List[Dict[str, Any]]):
        """Append new samples to existing cache."""
        if task_name not in self._metadata.tasks:
            return self._save_to_cache(task_name, new_samples)

        task_dir = self.cache_dir / task_name
        cache_info = self._metadata.tasks[task_name]

        # Load existing samples
        existing_samples = self._load_all_cached_samples(task_name)

        # Combine and re-save in chunks
        all_samples = existing_samples + new_samples
        self._save_to_cache(task_name, all_samples)

    def _load_cached_samples(self, task_name: str, limit: int) -> List[Dict[str, Any]]:
        """Load cached samples up to limit."""
        if task_name not in self._metadata.tasks:
            return []

        cache_info = self._metadata.tasks[task_name]
        task_dir = self.cache_dir / task_name

        samples = []
        samples_loaded = 0

        for chunk_filename in cache_info.chunks:
            if samples_loaded >= limit:
                break

            chunk_path = task_dir / chunk_filename
            if not chunk_path.exists():
                raise CacheCorruptionError(f"Missing chunk file: {chunk_path}")

            try:
                with open(chunk_path) as f:
                    chunk_samples = json.load(f)
            except Exception as e:
                raise CacheCorruptionError(f"Corrupted chunk file {chunk_path}: {e}")

            # Add samples until we reach the limit
            for sample in chunk_samples:
                if samples_loaded >= limit:
                    break
                samples.append(sample)
                samples_loaded += 1

        logger.info(f"Loaded {len(samples)} cached samples for '{task_name}'")
        return samples

    def _load_all_cached_samples(self, task_name: str) -> List[Dict[str, Any]]:
        """Load all cached samples for a task."""
        if task_name not in self._metadata.tasks:
            return []

        cache_info = self._metadata.tasks[task_name]
        return self._load_cached_samples(task_name, cache_info.samples_count)

    def _clear_task_cache(self, task_name: str):
        """Clear cache for a specific task."""
        task_dir = self.cache_dir / task_name

        if task_dir.exists():
            import shutil

            shutil.rmtree(task_dir)

        if task_name in self._metadata.tasks:
            del self._metadata.tasks[task_name]
            self._save_metadata()

        logger.info(f"Cleared cache for task '{task_name}'")

    def _load_metadata(self) -> CacheMetadata:
        """Load cache metadata."""
        if not self.metadata_file.exists():
            return CacheMetadata(
                version=self.CACHE_VERSION, created_at=datetime.now(), last_cleanup=datetime.now(), tasks={}
            )

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            tasks = {}
            for task_name, task_data in data.get("tasks", {}).items():
                tasks[task_name] = CacheInfo(
                    task_name=task_data["task_name"],
                    samples_count=task_data["samples_count"],
                    last_updated=datetime.fromisoformat(task_data["last_updated"]),
                    cache_version=task_data["cache_version"],
                    chunks=task_data["chunks"],
                )

            return CacheMetadata(
                version=data.get("version", self.CACHE_VERSION),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_cleanup=datetime.fromisoformat(data["last_cleanup"]),
                tasks=tasks,
            )

        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return CacheMetadata(
                version=self.CACHE_VERSION, created_at=datetime.now(), last_cleanup=datetime.now(), tasks={}
            )

    def _save_metadata(self):
        """Save cache metadata."""
        # Convert to serializable format
        tasks_data = {}
        for task_name, cache_info in self._metadata.tasks.items():
            tasks_data[task_name] = {
                "task_name": cache_info.task_name,
                "samples_count": cache_info.samples_count,
                "last_updated": cache_info.last_updated.isoformat(),
                "cache_version": cache_info.cache_version,
                "chunks": cache_info.chunks,
            }

        data = {
            "version": self._metadata.version,
            "created_at": self._metadata.created_at.isoformat(),
            "last_cleanup": self._metadata.last_cleanup.isoformat(),
            "tasks": tasks_data,
        }

        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status."""
        total_samples = sum(info.samples_count for info in self._metadata.tasks.values())
        total_size = sum(
            sum(
                (self.cache_dir / task_name / chunk).stat().st_size
                for chunk in info.chunks
                if (self.cache_dir / task_name / chunk).exists()
            )
            for task_name, info in self._metadata.tasks.items()
        )

        return {
            "cache_version": self._metadata.version,
            "total_tasks": len(self._metadata.tasks),
            "total_samples": total_samples,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "created_at": self._metadata.created_at.isoformat(),
            "last_cleanup": self._metadata.last_cleanup.isoformat(),
            "tasks": {
                task_name: {
                    "samples_count": info.samples_count,
                    "last_updated": info.last_updated.isoformat(),
                    "chunks": len(info.chunks),
                }
                for task_name, info in self._metadata.tasks.items()
            },
        }

    def cleanup_cache(self, max_age_days: int = None):
        """Clean up old cache entries."""
        if max_age_days is None:
            max_age_days = self.MAX_CACHE_AGE_DAYS

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        tasks_to_remove = []

        for task_name, cache_info in self._metadata.tasks.items():
            if cache_info.last_updated < cutoff_date:
                tasks_to_remove.append(task_name)

        for task_name in tasks_to_remove:
            self._clear_task_cache(task_name)

        self._metadata.last_cleanup = datetime.now()
        self._save_metadata()

        logger.info(f"Cleaned up {len(tasks_to_remove)} old cache entries")
        return len(tasks_to_remove)

    def preload_tasks(self, task_limits: Dict[str, int]):
        """Preload multiple tasks with specified limits."""
        results = {}

        for task_name, limit in task_limits.items():
            try:
                samples = self.get_task_samples(task_name, limit)
                results[task_name] = {"status": "success", "samples_loaded": len(samples), "requested_limit": limit}
                logger.info(f"Preloaded {len(samples)} samples for '{task_name}'")
            except Exception as e:
                results[task_name] = {"status": "error", "error": str(e), "requested_limit": limit}
                logger.error(f"Failed to preload '{task_name}': {e}")

        return results


# Global instance
_managed_cache = None


def get_managed_cache(cache_dir: str = "./benchmark_cache") -> ManagedCachedBenchmarks:
    """Get the global managed cache instance."""
    global _managed_cache
    if _managed_cache is None:
        _managed_cache = ManagedCachedBenchmarks(cache_dir)
    return _managed_cache
