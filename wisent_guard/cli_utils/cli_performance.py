"""Helper utilities for CLI performance tracking (memory & latency).
"""
from __future__ import annotations
from typing import Tuple, Optional


def setup_performance_tracking(
    enable_memory_tracking: bool,
    enable_latency_tracking: bool,
    track_gpu_memory: bool,
    memory_sampling_interval: float,
    show_memory_usage: bool,
    verbose: bool = False,
) -> Tuple[Optional[object], Optional[object]]:
    """Initialize and optionally start memory and latency tracking.

    Args:
        enable_memory_tracking: Whether to start periodic memory sampling.
        enable_latency_tracking: Whether to enable latency measurement context manager.
        track_gpu_memory: Whether GPU memory should also be sampled (if available).
        memory_sampling_interval: Sampling interval in seconds for memory tracker.
        show_memory_usage: If True, prints a one-off current memory usage snapshot.
        verbose: If True, prints human-friendly status messages.

    Returns:
        (memory_tracker, latency_tracker) — either may be ``None`` if disabled.
    """
    memory_tracker = None
    latency_tracker = None

    if not (enable_memory_tracking or enable_latency_tracking or show_memory_usage):
        # Fast path: nothing requested
        return memory_tracker, latency_tracker

    # Import lazily so normal CLI usage without tracking stays lightweight
    try:
        from ..core.tracking import (
            format_memory_usage,
            get_global_latency_tracker,
            get_global_memory_tracker,
            get_memory_info,
        )
    except Exception as e:  # pragma: no cover - defensive: tracking is optional
        if verbose:
            print(f"Performance tracking utilities unavailable: {e}")
        return memory_tracker, latency_tracker

    if enable_memory_tracking or enable_latency_tracking:
        if verbose:
            print(
                f"Performance tracking enabled: memory={enable_memory_tracking}, latency={enable_latency_tracking}"
            )

    # Memory tracking setup
    if enable_memory_tracking:
        try:
            memory_tracker = get_global_memory_tracker()
            memory_tracker.track_gpu = track_gpu_memory
            memory_tracker.sampling_interval = memory_sampling_interval
            memory_tracker.start_monitoring()
            if verbose:
                gpu_note = " (GPU)" if track_gpu_memory else ""
                print(
                    f"   Memory tracking started every {memory_sampling_interval}s{gpu_note}"
                )
        except Exception as e:  # pragma: no cover - robustness
            memory_tracker = None
            if verbose:
                print(f"   Failed to start memory tracking: {e}")

    # Latency tracking setup
    if enable_latency_tracking:
        try:
            latency_tracker = get_global_latency_tracker()
            latency_tracker.start_tracking()
            if verbose:
                print("   Latency tracking started")
        except Exception as e:  # pragma: no cover
            latency_tracker = None
            if verbose:
                print(f"   Failed to start latency tracking: {e}")

    # One-off snapshot if requested (independent of enabling persistent tracking)
    if show_memory_usage:
        try:
            current_memory = get_memory_info()
            print(f"\nCurrent Memory Usage: {format_memory_usage(current_memory)}")
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"   Failed to read current memory usage: {e}")

    return memory_tracker, latency_tracker
