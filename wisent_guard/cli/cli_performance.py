# cli_performance.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Trackers:
    memory: Optional[object] = None
    latency: Optional[object] = None

def start_trackers(
    enable_memory_tracking: bool,
    enable_latency_tracking: bool,
    memory_sampling_interval: float,
    track_gpu_memory: bool,
    show_memory_usage: bool,
    verbose: bool,
) -> Trackers:
    trackers = Trackers()
    if enable_memory_tracking or enable_latency_tracking:
        from wisent_guard.core.tracking import (
            get_global_memory_tracker,
            get_global_latency_tracker,
            get_memory_info,
            format_memory_usage,
        )
        if enable_memory_tracking:
            trackers.memory = get_global_memory_tracker()
            trackers.memory.track_gpu = track_gpu_memory
            trackers.memory.sampling_interval = memory_sampling_interval
            trackers.memory.start_monitoring()
            if verbose:
                print(f"   • Memory tracking started ({memory_sampling_interval}s interval)")
        if enable_latency_tracking:
            trackers.latency = get_global_latency_tracker()
            trackers.latency.start_tracking()
            if verbose:
                print("   • Latency tracking started")
        if show_memory_usage:
            current = get_memory_info()
            print(f"\n💾 Current Memory Usage: {format_memory_usage(current)}")
    return trackers

def stop_and_report(trackers: Trackers, export_csv, detailed: bool, show_timing: bool, verbose: bool, result=None, error: str=None):
    try:
        if trackers is None:
            return {"error": error} if error else result
        # Always stop monitoring even if not verbose so resources are released
        mem_stats = None
        if trackers.memory:
            mem_stats = trackers.memory.stop_monitoring()

        # Export CSV regardless of verbosity (side-effect requested by user)
        if export_csv and trackers.latency:
            trackers.latency.export_csv(export_csv)
            if verbose:
                print(f"\n📄 Exported performance CSV: {export_csv}")

        # Skip printing if not verbose
        if not verbose:
            return {"error": error} if error else result

        print("\n📊 PERFORMANCE REPORT:")
        print("=" * 50)
        if mem_stats is not None:
            print("💾 Memory Usage:")
            print(trackers.memory.format_stats(mem_stats, detailed))
        if trackers.latency or show_timing:
            if trackers.latency:
                print("\n⏱️ Performance Metrics:")
                print(trackers.latency.format_user_metrics())
            else:
                from wisent_guard.core.tracking import format_timing_summary
                print("\n⏱️ Timing Summary:")
                print(format_timing_summary(detailed))
        print("=" * 50)
    except Exception:
        pass
    return {"error": error} if error else result

if __name__ == "__main__":
    print("\n[cli_performance] smoke test")
    try:
        trackers = start_trackers(
            enable_memory_tracking=False,
            enable_latency_tracking=True,
            memory_sampling_interval=0.1,
            track_gpu_memory=False,
            show_memory_usage=False,
            verbose=True,
        )
        out = stop_and_report(trackers, export_csv=None, detailed=False, show_timing=True, verbose=True, result={"ok": True})
        print("  stop_and_report ->", out)
    except ImportError as e:
        print("  SKIPPED (missing deps):", e)
