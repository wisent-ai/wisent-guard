"""
Latency tracking for wisent-guard operations.

This module provides comprehensive timing and performance monitoring capabilities
for all aspects of the wisent-guard pipeline including model operations,
steering computations, and text generation.
"""

import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
import functools


@dataclass
class TimingEvent:
    """Single timing event measurement."""
    name: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


@dataclass
class LatencyStats:
    """Aggregated latency statistics for an operation."""
    operation: str
    count: int
    total_time: float
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    events: List[TimingEvent] = field(default_factory=list)
    
    @property
    def mean_time_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean_time * 1000
    
    @property
    def total_time_ms(self) -> float:
        """Total time in milliseconds."""
        return self.total_time * 1000


@dataclass
class GenerationMetrics:
    """User-facing generation performance metrics."""
    time_to_first_token: float  # seconds
    total_generation_time: float  # seconds
    token_count: int
    tokens_per_second: float
    prompt_length: int = 0
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        return self.time_to_first_token * 1000
    
    @property
    def total_time_ms(self) -> float:
        """Total generation time in milliseconds."""
        return self.total_generation_time * 1000


@dataclass
class TrainingMetrics:
    """User-facing training performance metrics."""
    total_training_time: float  # seconds
    training_samples: int
    method: str
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def training_time_ms(self) -> float:
        """Training time in milliseconds."""
        return self.total_training_time * 1000
    
    @property
    def samples_per_second(self) -> float:
        """Training samples processed per second."""
        return self.training_samples / self.total_training_time if self.total_training_time > 0 else 0


class LatencyTracker:
    """
    Comprehensive latency tracker for wisent-guard operations.
    
    Tracks timing for individual operations and provides aggregated statistics.
    Supports nested operation tracking and hierarchical timing analysis.
    """
    
    def __init__(self, auto_start: bool = True):
        """
        Initialize latency tracker.
        
        Args:
            auto_start: Whether to automatically start tracking
        """
        self.events: List[TimingEvent] = []
        self.active_operations: Dict[str, float] = {}
        self.operation_stack: List[str] = []
        self.is_tracking = auto_start
        self.start_time = time.time() if auto_start else None
    
    def start_tracking(self) -> None:
        """Start or resume latency tracking."""
        self.is_tracking = True
        if self.start_time is None:
            self.start_time = time.time()
    
    def stop_tracking(self) -> None:
        """Stop latency tracking."""
        self.is_tracking = False
    
    def start_operation(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start timing an operation.
        
        Args:
            name: Name of the operation
            metadata: Optional metadata to store with the event
            
        Returns:
            Operation ID for later reference
        """
        if not self.is_tracking:
            return name
        
        current_time = time.time()
        operation_id = f"{name}_{len(self.events)}"
        
        self.active_operations[operation_id] = current_time
        self.operation_stack.append(operation_id)
        
        return operation_id
    
    def end_operation(
        self, 
        operation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[TimingEvent]:
        """
        End timing an operation.
        
        Args:
            operation_id: ID returned from start_operation
            metadata: Additional metadata to store
            
        Returns:
            TimingEvent if operation was found, None otherwise
        """
        if not self.is_tracking or operation_id not in self.active_operations:
            return None
        
        end_time = time.time()
        start_time = self.active_operations.pop(operation_id)
        duration = end_time - start_time
        
        # Extract operation name from ID
        name = operation_id.rsplit('_', 1)[0]
        
        # Determine parent operation
        parent = None
        if operation_id in self.operation_stack:
            stack_index = self.operation_stack.index(operation_id)
            if stack_index > 0:
                parent_id = self.operation_stack[stack_index - 1]
                parent = parent_id.rsplit('_', 1)[0]
            self.operation_stack.remove(operation_id)
        
        # Merge metadata
        combined_metadata = metadata or {}
        
        event = TimingEvent(
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            metadata=combined_metadata,
            parent=parent
        )
        
        self.events.append(event)
        return event
    
    @contextmanager
    def time_operation(
        self, 
        name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for timing operations.
        
        Args:
            name: Name of the operation
            metadata: Optional metadata to store
            
        Yields:
            TimingEvent that will be populated when context exits
        """
        operation_id = self.start_operation(name, metadata)
        event_placeholder = {"event": None}
        
        try:
            yield event_placeholder
        finally:
            event = self.end_operation(operation_id, metadata)
            event_placeholder["event"] = event
    
    @contextmanager
    def time_generation(self, name: str = "response_generation", prompt_length: int = 0):
        """
        Context manager for timing text generation with TTFT tracking.
        
        Args:
            name: Name of the generation operation
            prompt_length: Length of the input prompt in tokens
            
        Yields:
            Dict with methods to mark first token and update token count
        """
        start_time = time.time()
        operation_id = self.start_operation(name, {"prompt_length": prompt_length})
        
        generation_state = {
            "first_token_time": None,
            "token_count": 0
        }
        
        # Add methods that modify the dict
        generation_state["mark_first_token"] = lambda: generation_state.update({"first_token_time": time.time()})
        generation_state["update_tokens"] = lambda count: generation_state.update({"token_count": count})
        
        try:
            yield generation_state
        finally:
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Calculate TTFT
            ttft = generation_state["first_token_time"] - start_time if generation_state["first_token_time"] else 0.0
            
            # Calculate tokens per second
            tokens_per_sec = generation_state["token_count"] / total_duration if total_duration > 0 else 0.0
            
            metadata = {
                "prompt_length": prompt_length,
                "time_to_first_token": ttft,
                "token_count": generation_state["token_count"],
                "tokens_per_second": tokens_per_sec
            }
            
            self.end_operation(operation_id, metadata)
    
    def get_stats(self, operation_name: Optional[str] = None) -> Union[LatencyStats, Dict[str, LatencyStats]]:
        """
        Get latency statistics.
        
        Args:
            operation_name: Specific operation to get stats for, or None for all
            
        Returns:
            LatencyStats for specific operation or dict of all operation stats
        """
        if operation_name:
            events = [e for e in self.events if e.name == operation_name]
            if not events:
                raise ValueError(f"No events found for operation: {operation_name}")
            return self._calculate_stats(operation_name, events)
        else:
            # Group events by operation name
            operation_events = defaultdict(list)
            for event in self.events:
                operation_events[event.name].append(event)
            
            return {
                name: self._calculate_stats(name, events)
                for name, events in operation_events.items()
            }
    
    def _calculate_stats(self, operation: str, events: List[TimingEvent]) -> LatencyStats:
        """Calculate statistics for a list of timing events."""
        durations = [e.duration for e in events]
        
        if not durations:
            return LatencyStats(
                operation=operation,
                count=0,
                total_time=0,
                mean_time=0,
                median_time=0,
                min_time=0,
                max_time=0,
                std_dev=0,
                percentile_95=0,
                percentile_99=0,
                events=[]
            )
        
        durations.sort()
        
        return LatencyStats(
            operation=operation,
            count=len(durations),
            total_time=sum(durations),
            mean_time=statistics.mean(durations),
            median_time=statistics.median(durations),
            min_time=min(durations),
            max_time=max(durations),
            std_dev=statistics.stdev(durations) if len(durations) > 1 else 0,
            percentile_95=self._percentile(durations, 95),
            percentile_99=self._percentile(durations, 99),
            events=events.copy()
        )
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0
        
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_timeline(self) -> List[TimingEvent]:
        """Get chronological timeline of all events."""
        return sorted(self.events, key=lambda e: e.start_time)
    
    def get_hierarchy(self) -> Dict[str, List[TimingEvent]]:
        """Get hierarchical view of operations (parent -> children)."""
        hierarchy = defaultdict(list)
        
        for event in self.events:
            parent = event.parent or "root"
            hierarchy[parent].append(event)
        
        return dict(hierarchy)
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.events.clear()
        self.active_operations.clear()
        self.operation_stack.clear()
        self.start_time = time.time() if self.is_tracking else None
    
    def get_generation_metrics(self, operation_name: str = "response_generation") -> Optional[GenerationMetrics]:
        """Get user-facing generation metrics."""
        events = [e for e in self.events if e.name == operation_name]
        if not events:
            return None
        
        # Use the most recent event
        latest_event = events[-1]
        metadata = latest_event.metadata
        
        return GenerationMetrics(
            time_to_first_token=metadata.get('time_to_first_token', 0.0),
            total_generation_time=latest_event.duration,
            token_count=metadata.get('token_count', 0),
            tokens_per_second=metadata.get('tokens_per_second', 0.0),
            prompt_length=metadata.get('prompt_length', 0)
        )
    
    def get_training_metrics(self, operation_name: str = "total_training_time") -> Optional[TrainingMetrics]:
        """Get user-facing training metrics."""
        events = [e for e in self.events if e.name == operation_name]
        if not events:
            return None
        
        latest_event = events[-1]
        metadata = latest_event.metadata
        
        return TrainingMetrics(
            total_training_time=latest_event.duration,
            training_samples=metadata.get('training_samples', 0),
            method=metadata.get('method', 'unknown'),
            success=metadata.get('success', True),
            error_message=metadata.get('error_message')
        )
    
    def format_user_metrics(self) -> str:
        """Format user-facing performance metrics."""
        lines = ["🚀 Performance Summary:"]
        
        # Training metrics
        training_metrics = self.get_training_metrics()
        if training_metrics:
            lines.extend([
                f"\n📚 Training:",
                f"  Method: {training_metrics.method}",
                f"  Total Time: {training_metrics.training_time_ms:.0f} ms",
                f"  Samples: {training_metrics.training_samples}",
                f"  Speed: {training_metrics.samples_per_second:.1f} samples/sec"
            ])
        
        # Generation metrics - check for both response_generation and individual generation events
        generation_metrics = self.get_generation_metrics("response_generation")
        if not generation_metrics:
            # Try to get metrics from steered_generation if response_generation doesn't exist
            generation_metrics = self.get_generation_metrics("steered_generation")
        
        if generation_metrics and generation_metrics.token_count > 0:
            lines.extend([
                f"\n🎭 Generation:",
                f"  Time to First Token: {generation_metrics.ttft_ms:.0f} ms",
                f"  Total Generation: {generation_metrics.total_time_ms:.0f} ms",
                f"  Tokens Generated: {generation_metrics.token_count}",
                f"  Speed: {generation_metrics.tokens_per_second:.1f} tokens/sec"
            ])
        
        # Steering overhead comparison
        steered_events = [e for e in self.events if e.name == "steered_generation"]
        unsteered_events = [e for e in self.events if e.name == "unsteered_generation"]
        
        if steered_events and unsteered_events:
            steered_avg = sum(e.duration for e in steered_events) / len(steered_events)
            unsteered_avg = sum(e.duration for e in unsteered_events) / len(unsteered_events)
            overhead = ((steered_avg - unsteered_avg) / unsteered_avg) * 100
            
            lines.extend([
                f"\n⚡ Steering Overhead:",
                f"  Unsteered Avg: {unsteered_avg * 1000:.0f} ms ({len(unsteered_events)} runs)",
                f"  Steered Avg: {steered_avg * 1000:.0f} ms ({len(steered_events)} runs)",
                f"  Overhead: {overhead:+.1f}%"
            ])
        elif steered_events:
            # Show steered performance even without comparison
            steered_avg = sum(e.duration for e in steered_events) / len(steered_events)
            lines.extend([
                f"\n🎯 Steered Generation:",
                f"  Average Time: {steered_avg * 1000:.0f} ms ({len(steered_events)} runs)"
            ])
        elif unsteered_events:
            # Show unsteered performance even without comparison
            unsteered_avg = sum(e.duration for e in unsteered_events) / len(unsteered_events)
            lines.extend([
                f"\n🔄 Unsteered Generation:",
                f"  Average Time: {unsteered_avg * 1000:.0f} ms ({len(unsteered_events)} runs)"
            ])
        
        # Show warning if no generation metrics found
        if not generation_metrics or generation_metrics.token_count == 0:
            lines.extend([
                f"\n⚠️ No generation metrics available",
                f"  (Responses may be empty or timing failed)"
            ])
        
        return '\n'.join(lines)

    def format_stats(
        self, 
        stats: Union[LatencyStats, Dict[str, LatencyStats]], 
        detailed: bool = False
    ) -> str:
        """Format latency statistics as a readable string."""
        if isinstance(stats, LatencyStats):
            return self._format_single_stats(stats, detailed)
        else:
            lines = ["Latency Statistics Summary:"]
            for operation, op_stats in stats.items():
                lines.append(f"\n{operation}:")
                lines.extend([f"  {line}" for line in self._format_single_stats(op_stats, detailed).split('\n')])
            return '\n'.join(lines)
    
    def _format_single_stats(self, stats: LatencyStats, detailed: bool) -> str:
        """Format statistics for a single operation."""
        lines = [
            f"Operation: {stats.operation}",
            f"Count: {stats.count}",
            f"Total Time: {stats.total_time_ms:.1f} ms",
            f"Mean Time: {stats.mean_time_ms:.1f} ms",
            f"Median Time: {stats.median_time * 1000:.1f} ms",
            f"Min Time: {stats.min_time * 1000:.1f} ms",
            f"Max Time: {stats.max_time * 1000:.1f} ms",
        ]
        
        if stats.count > 1:
            lines.extend([
                f"Std Dev: {stats.std_dev * 1000:.1f} ms",
                f"95th Percentile: {stats.percentile_95 * 1000:.1f} ms",
                f"99th Percentile: {stats.percentile_99 * 1000:.1f} ms",
            ])
        
        if detailed and stats.events:
            lines.append(f"Recent Events:")
            for event in stats.events[-5:]:  # Show last 5 events
                lines.append(f"  {event.duration_ms:.1f} ms")
                if event.metadata:
                    lines.append(f"    Metadata: {event.metadata}")
        
        return '\n'.join(lines)
    
    def export_csv(self, filename: str) -> None:
        """Export timing events to CSV file."""
        import csv
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'operation', 'start_time', 'end_time', 'duration_ms', 
                'parent', 'metadata'
            ])
            
            for event in self.events:
                writer.writerow([
                    event.name,
                    event.start_time,
                    event.end_time,
                    event.duration_ms,
                    event.parent or '',
                    str(event.metadata) if event.metadata else ''
                ])


# Global latency tracker instance
_global_tracker: Optional[LatencyTracker] = None


def get_global_tracker() -> LatencyTracker:
    """Get or create the global latency tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LatencyTracker()
    return _global_tracker


def time_function(operation_name: Optional[str] = None):
    """
    Decorator to automatically time function execution.
    
    Args:
        operation_name: Name for the operation (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_global_tracker()
            with tracker.time_operation(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def time_operation(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Global context manager for timing operations."""
    tracker = get_global_tracker()
    with tracker.time_operation(name, metadata) as event_ref:
        yield event_ref


def get_timing_summary() -> Dict[str, LatencyStats]:
    """Get timing summary from global tracker."""
    tracker = get_global_tracker()
    return tracker.get_stats()


def format_timing_summary(detailed: bool = False) -> str:
    """Format timing summary as a readable string."""
    tracker = get_global_tracker()
    stats = tracker.get_stats()
    return tracker.format_stats(stats, detailed)


def reset_timing() -> None:
    """Reset global timing data."""
    tracker = get_global_tracker()
    tracker.reset()


# Common operation names for user-facing metrics
class Operations:
    """Standard operation names for user-facing performance metrics."""
    # Core user-facing metrics
    TOTAL_TRAINING_TIME = "total_training_time"
    TIME_TO_FIRST_TOKEN = "time_to_first_token"
    RESPONSE_GENERATION = "response_generation"
    UNSTEERED_GENERATION = "unsteered_generation"
    STEERED_GENERATION = "steered_generation"
    
    # Batch processing
    BATCH_INFERENCE = "batch_inference"
    PER_RESPONSE = "per_response"
    
    # Training phases
    STEERING_VECTOR_TRAINING = "steering_vector_training"
    CLASSIFIER_TRAINING = "classifier_training"
    
    # Legacy (for backward compatibility)
    MODEL_LOADING = "model_loading"
    ACTIVATION_EXTRACTION = "activation_extraction"
