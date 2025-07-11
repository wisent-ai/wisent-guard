"""
Managed Cached Benchmarks Service

This service intelligently manages benchmark dataset downloading and caching based on
the requested limit parameter. It ensures we only download what we need and reuse
existing cached data when possible.

Key Features:
- Downloads only the requested number of samples (limit parameter)
- Reuses cached data when limit <= cached samples
- Incrementally downloads more data when limit > cached samples
- Maintains metadata about cached sample counts
- Supports per-task caching with different limits
"""

import os
import json
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ManagedCachedBenchmarks:
    """
    Manages intelligent caching of benchmark datasets based on sample limits.
    """
    
    def __init__(self, cache_dir: str = "./managed_benchmark_cache"):
        """
        Initialize the managed cached benchmarks service.
        
        Args:
            cache_dir: Directory to store cached benchmark data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track cached sample counts
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {"tasks": {}, "version": "1.0"}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def _get_task_cache_path(self, task_name: str) -> Path:
        """Get the cache file path for a specific task."""
        # Use hash to handle special characters in task names
        task_hash = hashlib.md5(task_name.encode()).hexdigest()[:8]
        safe_name = "".join(c for c in task_name if c.isalnum() or c in "_-")[:50]
        return self.cache_dir / f"{safe_name}_{task_hash}.pkl"
    
    def _get_cached_sample_count(self, task_name: str) -> int:
        """Get the number of cached samples for a task."""
        return self.metadata.get("tasks", {}).get(task_name, {}).get("sample_count", 0)
    
    def _update_task_metadata(self, task_name: str, sample_count: int, task_info: Dict[str, Any]):
        """Update metadata for a cached task."""
        if "tasks" not in self.metadata:
            self.metadata["tasks"] = {}
        
        self.metadata["tasks"][task_name] = {
            "sample_count": sample_count,
            "task_info": task_info,
            "last_updated": self._get_current_timestamp()
        }
        self._save_metadata()
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _download_task_samples(self, task_name: str, limit: int) -> List[Dict[str, Any]]:
        """
        Download samples from a task using lm-eval-harness.
        
        Args:
            task_name: Name of the task to download
            limit: Maximum number of samples to download
            
        Returns:
            List of sample documents
        """
        try:
            # Import lm-eval components
            from lm_eval import tasks
            from lm_eval.tasks import get_task_dict
            
            logger.info(f"📥 Downloading {limit} samples from {task_name}...")
            
            # Get task
            task_dict = get_task_dict([task_name])
            if task_name not in task_dict:
                raise ValueError(f"Task {task_name} not found in lm-eval registry")
            
            task = task_dict[task_name]
            
            # Get documents from the most appropriate split
            docs = []
            
            # Try different splits in order of preference
            if hasattr(task, 'has_validation_docs') and task.has_validation_docs():
                docs = list(task.validation_docs())
            elif hasattr(task, 'has_test_docs') and task.has_test_docs():
                docs = list(task.test_docs())
            elif hasattr(task, 'has_training_docs') and task.has_training_docs():
                docs = list(task.training_docs())
            else:
                raise RuntimeError(f"No documents available for task {task_name}")
            
            # Limit the number of documents
            if limit > 0:
                docs = docs[:limit]
            
            logger.info(f"✅ Downloaded {len(docs)} samples from {task_name}")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to download samples from {task_name}: {e}")
            raise
    
    def _load_cached_samples(self, task_name: str, limit: int) -> Optional[List[Dict[str, Any]]]:
        """
        Load cached samples for a task.
        
        Args:
            task_name: Name of the task
            limit: Number of samples needed
            
        Returns:
            List of cached samples or None if not available
        """
        cache_path = self._get_task_cache_path(task_name)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            samples = cached_data.get("samples", [])
            
            # Return only the requested number of samples
            return samples[:limit] if limit > 0 else samples
            
        except Exception as e:
            logger.error(f"Failed to load cached samples for {task_name}: {e}")
            return None
    
    def _save_cached_samples(self, task_name: str, samples: List[Dict[str, Any]], task_info: Dict[str, Any]):
        """
        Save samples to cache.
        
        Args:
            task_name: Name of the task
            samples: List of sample documents
            task_info: Metadata about the task
        """
        cache_path = self._get_task_cache_path(task_name)
        
        try:
            cached_data = {
                "task_name": task_name,
                "samples": samples,
                "task_info": task_info,
                "cached_at": self._get_current_timestamp()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Update metadata
            self._update_task_metadata(task_name, len(samples), task_info)
            
            logger.info(f"💾 Cached {len(samples)} samples for {task_name}")
            
        except Exception as e:
            logger.error(f"Failed to cache samples for {task_name}: {e}")
            raise
    
    def get_task_samples(self, task_name: str, limit: int, force_download: bool = False) -> List[Dict[str, Any]]:
        """
        Get samples for a task with intelligent caching.
        
        Args:
            task_name: Name of the task
            limit: Number of samples needed
            force_download: Force fresh download even if cached
            
        Returns:
            List of sample documents
        """
        if limit <= 0:
            return []
        
        cached_count = self._get_cached_sample_count(task_name)
        
        logger.info(f"🔍 Task {task_name}: need {limit} samples, have {cached_count} cached")
        
        if not force_download and cached_count >= limit:
            # We have enough cached samples, use them
            logger.info(f"✅ Using cached samples for {task_name} (have {cached_count}, need {limit})")
            cached_samples = self._load_cached_samples(task_name, limit)
            if cached_samples is not None:
                return cached_samples
            else:
                logger.warning(f"⚠️ Failed to load cached samples for {task_name}, downloading fresh")
        
        # Need to download more samples
        if cached_count < limit:
            logger.info(f"📥 Downloading {limit} samples for {task_name} (cached: {cached_count})")
        else:
            logger.info(f"🔄 Force downloading {limit} samples for {task_name}")
        
        # Download the requested number of samples
        samples = self._download_task_samples(task_name, limit)
        
        # Cache the downloaded samples
        task_info = {
            "original_limit": limit,
            "download_method": "lm_eval_harness"
        }
        self._save_cached_samples(task_name, samples, task_info)
        
        return samples
    
    def get_cache_status(self, task_name: str) -> Dict[str, Any]:
        """
        Get cache status for a task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Dictionary with cache status information
        """
        cached_count = self._get_cached_sample_count(task_name)
        cache_path = self._get_task_cache_path(task_name)
        
        status = {
            "task_name": task_name,
            "cached_samples": cached_count,
            "cache_exists": cache_path.exists(),
            "cache_path": str(cache_path)
        }
        
        if task_name in self.metadata.get("tasks", {}):
            task_meta = self.metadata["tasks"][task_name]
            status.update({
                "last_updated": task_meta.get("last_updated"),
                "task_info": task_meta.get("task_info", {})
            })
        
        return status
    
    def clear_task_cache(self, task_name: str):
        """
        Clear cache for a specific task.
        
        Args:
            task_name: Name of the task to clear
        """
        cache_path = self._get_task_cache_path(task_name)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"🗑️ Cleared cache for {task_name}")
        
        # Remove from metadata
        if task_name in self.metadata.get("tasks", {}):
            del self.metadata["tasks"][task_name]
            self._save_metadata()
    
    def clear_all_cache(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        self.metadata = {"tasks": {}, "version": "1.0"}
        self._save_metadata()
        
        logger.info("🗑️ Cleared all cached benchmark data")
    
    def get_cache_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all cached tasks.
        
        Returns:
            Dictionary with cache summary
        """
        summary = {
            "total_tasks": len(self.metadata.get("tasks", {})),
            "total_samples": sum(
                task_info.get("sample_count", 0) 
                for task_info in self.metadata.get("tasks", {}).values()
            ),
            "cache_dir": str(self.cache_dir),
            "tasks": {}
        }
        
        for task_name, task_info in self.metadata.get("tasks", {}).items():
            summary["tasks"][task_name] = {
                "sample_count": task_info.get("sample_count", 0),
                "last_updated": task_info.get("last_updated"),
            }
        
        return summary


# Global instance for easy access
_managed_cache = None


def get_managed_cache(cache_dir: str = "./managed_benchmark_cache") -> ManagedCachedBenchmarks:
    """Get the global managed cache instance."""
    global _managed_cache
    if _managed_cache is None:
        _managed_cache = ManagedCachedBenchmarks(cache_dir)
    return _managed_cache


def clear_managed_cache():
    """Clear the global managed cache instance."""
    global _managed_cache
    _managed_cache = None 