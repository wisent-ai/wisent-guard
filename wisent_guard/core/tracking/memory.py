"""
Memory usage tracking for wisent-guard operations.

This module provides comprehensive memory monitoring capabilities including
GPU and CPU memory tracking, peak usage detection, and memory profiling.
"""

import gc
import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import torch

from wisent_guard.core.utils.device import resolve_default_device

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    timestamp: float
    cpu_memory_mb: float
    cpu_memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    allocated_tensors: Optional[int] = None
    cached_memory_mb: Optional[float] = None
    operation: Optional[str] = None


@dataclass
class MemoryStats:
    """Aggregated memory statistics over a period."""
    peak_cpu_mb: float
    peak_gpu_mb: Optional[float]
    avg_cpu_mb: float
    avg_gpu_mb: Optional[float]
    min_cpu_mb: float
    min_gpu_mb: Optional[float]
    duration_seconds: float
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)


class MemoryTracker:
    """
    Comprehensive memory usage tracker for wisent-guard operations.
    
    Tracks both CPU and GPU memory usage with optional continuous monitoring.
    """
    
    def __init__(
        self,
        track_gpu: bool = True,
        sampling_interval: float = 0.1,
        auto_cleanup: bool = True
    ):
        """
        Initialize memory tracker.
        
        Args:
            track_gpu: Whether to track GPU memory (requires CUDA)
            sampling_interval: How often to sample memory (seconds)
            auto_cleanup: Whether to automatically run garbage collection
        """
        self.device_kind = resolve_default_device()
        self.track_gpu = track_gpu and self.device_kind in {"cuda", "mps"}
        self.sampling_interval = sampling_interval
        self.auto_cleanup = auto_cleanup
        
        self.snapshots: List[MemorySnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        
        # Initialize GPU monitoring if available
        if self.track_gpu and self.device_kind == "cuda" and NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
            except Exception:
                self.gpu_available = False
                self.gpu_handle = None
        else:
            self.gpu_handle = None
            self.gpu_available = False
    
    def take_snapshot(self, operation: Optional[str] = None) -> MemorySnapshot:
        """Take a single memory snapshot."""
        timestamp = time.time()
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_memory_mb = memory_info.rss / 1024 / 1024
        cpu_memory_percent = process.memory_percent()
        
        # GPU memory
        gpu_memory_mb = None
        gpu_memory_percent = None
        allocated_tensors = None
        cached_memory_mb = None
        
        if self.track_gpu:
            if self.device_kind == "cuda" and torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                cached_memory_mb = torch.cuda.memory_reserved() / 1024 / 1024
                allocated_tensors = len(
                    [
                        obj
                        for obj in gc.get_objects()
                        if torch.is_tensor(obj) and getattr(obj, "is_cuda", False)
                    ]
                )

                if self.gpu_available and self.gpu_handle is not None:
                    try:
                        gpu_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        total_gpu_mb = gpu_info.total / 1024 / 1024
                        gpu_memory_percent = (gpu_memory_mb / total_gpu_mb) * 100
                    except Exception:
                        pass
            elif self.device_kind == "mps" and hasattr(torch, "mps"):
                try:
                    allocated_bytes = torch.mps.current_allocated_memory()
                except AttributeError:
                    allocated_bytes = 0

                try:
                    cached_bytes = torch.mps.driver_allocated_memory()
                except AttributeError:
                    cached_bytes = allocated_bytes

                gpu_memory_mb = allocated_bytes / 1024 / 1024
                cached_memory_mb = cached_bytes / 1024 / 1024
                allocated_tensors = len(
                    [
                        obj
                        for obj in gc.get_objects()
                        if torch.is_tensor(obj) and getattr(getattr(obj, "device", None), "type", None) == "mps"
                    ]
                )
        
        snapshot = MemorySnapshot(
            timestamp=timestamp,
            cpu_memory_mb=cpu_memory_mb,
            cpu_memory_percent=cpu_memory_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_memory_percent=gpu_memory_percent,
            allocated_tensors=allocated_tensors,
            cached_memory_mb=cached_memory_mb,
            operation=operation
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_monitoring(self) -> None:
        """Start continuous memory monitoring in a background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        self.snapshots.clear()
        
        def monitor_loop():
            while self.is_monitoring:
                self.take_snapshot("continuous_monitoring")
                time.sleep(self.sampling_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> MemoryStats:
        """Stop continuous monitoring and return aggregated statistics."""
        if not self.is_monitoring:
            raise ValueError("Monitoring is not active")
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return self.get_stats()
    
    def get_stats(self) -> MemoryStats:
        """Get aggregated memory statistics from collected snapshots."""
        if not self.snapshots:
            raise ValueError("No snapshots available")
        
        cpu_values = [s.cpu_memory_mb for s in self.snapshots]
        gpu_values = [s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None]
        
        duration = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        operations = list(set(s.operation for s in self.snapshots if s.operation))
        
        return MemoryStats(
            peak_cpu_mb=max(cpu_values),
            peak_gpu_mb=max(gpu_values) if gpu_values else None,
            avg_cpu_mb=sum(cpu_values) / len(cpu_values),
            avg_gpu_mb=sum(gpu_values) / len(gpu_values) if gpu_values else None,
            min_cpu_mb=min(cpu_values),
            min_gpu_mb=min(gpu_values) if gpu_values else None,
            duration_seconds=duration,
            snapshots=self.snapshots.copy(),
            operations=operations
        )
    
    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if self.auto_cleanup:
            gc.collect()
            if self.device_kind == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif self.device_kind == "mps" and hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass
    
    def reset(self) -> None:
        """Reset the tracker, clearing all snapshots."""
        if self.is_monitoring:
            self.stop_monitoring()
        self.snapshots.clear()
        self.start_time = None
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track memory usage during a specific operation."""
        self.take_snapshot(f"{operation_name}_start")
        start_time = time.time()
        
        try:
            yield self
        finally:
            end_time = time.time()
            self.take_snapshot(f"{operation_name}_end")
            
            if self.auto_cleanup:
                self.clear_cache()
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current memory usage without storing a snapshot."""
        snapshot = self.take_snapshot("current_check")
        self.snapshots.pop()  # Remove the snapshot we just added
        
        usage = {
            "cpu_memory_mb": snapshot.cpu_memory_mb,
            "cpu_memory_percent": snapshot.cpu_memory_percent,
        }
        
        if snapshot.gpu_memory_mb is not None:
            usage.update({
                "gpu_memory_mb": snapshot.gpu_memory_mb,
                "gpu_memory_percent": snapshot.gpu_memory_percent,
                "allocated_tensors": snapshot.allocated_tensors,
                "cached_memory_mb": snapshot.cached_memory_mb,
            })
        
        return usage
    
    def format_stats(self, stats: MemoryStats, detailed: bool = False) -> str:
        """Format memory statistics as a readable string."""
        lines = [
            "Memory Usage Statistics:",
            f"  Duration: {stats.duration_seconds:.2f} seconds",
            f"  CPU Memory:",
            f"    Peak: {stats.peak_cpu_mb:.1f} MB",
            f"    Average: {stats.avg_cpu_mb:.1f} MB", 
            f"    Minimum: {stats.min_cpu_mb:.1f} MB",
        ]
        
        if stats.peak_gpu_mb is not None:
            lines.extend([
                f"  GPU Memory:",
                f"    Peak: {stats.peak_gpu_mb:.1f} MB",
                f"    Average: {stats.avg_gpu_mb:.1f} MB",
                f"    Minimum: {stats.min_gpu_mb:.1f} MB",
            ])
        
        if stats.operations:
            lines.append(f"  Operations: {', '.join(stats.operations)}")
        
        if detailed and stats.snapshots:
            lines.append(f"  Snapshots: {len(stats.snapshots)} collected")
            
            # Show peak usage snapshot
            peak_snapshot = max(stats.snapshots, key=lambda s: s.cpu_memory_mb)
            lines.extend([
                f"  Peak Usage Snapshot:",
                f"    Time: {peak_snapshot.timestamp:.2f}",
                f"    CPU: {peak_snapshot.cpu_memory_mb:.1f} MB ({peak_snapshot.cpu_memory_percent:.1f}%)",
            ])
            
            if peak_snapshot.gpu_memory_mb is not None:
                lines.append(f"    GPU: {peak_snapshot.gpu_memory_mb:.1f} MB")
                if peak_snapshot.allocated_tensors is not None:
                    lines.append(f"    Tensors: {peak_snapshot.allocated_tensors}")
        
        return "\n".join(lines)


# Global memory tracker instance
_global_tracker: Optional[MemoryTracker] = None


def get_global_tracker() -> MemoryTracker:
    """Get or create the global memory tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker()
    return _global_tracker


def track_memory(operation_name: str):
    """Decorator to track memory usage of a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tracker = get_global_tracker()
            with tracker.track_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information without tracking."""
    tracker = MemoryTracker(auto_cleanup=False)
    return tracker.get_current_usage()


def format_memory_usage(usage: Dict[str, Any]) -> str:
    """Format memory usage dictionary as a readable string."""
    lines = [
        f"CPU Memory: {usage['cpu_memory_mb']:.1f} MB ({usage['cpu_memory_percent']:.1f}%)"
    ]
    
    if 'gpu_memory_mb' in usage and usage['gpu_memory_mb'] is not None:
        lines.append(f"GPU Memory: {usage['gpu_memory_mb']:.1f} MB")
        if 'gpu_memory_percent' in usage and usage['gpu_memory_percent'] is not None:
            lines[-1] += f" ({usage['gpu_memory_percent']:.1f}%)"
        
        if 'cached_memory_mb' in usage:
            lines.append(f"GPU Cached: {usage['cached_memory_mb']:.1f} MB")
        
        if 'allocated_tensors' in usage:
            lines.append(f"GPU Tensors: {usage['allocated_tensors']}")
    
    return " | ".join(lines)
