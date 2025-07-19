"""
Task implementations for wisent-guard.

This package contains task-agnostic implementations for various benchmarks.
"""

from ..task_interface import register_task
from .livecodebench_task import LiveCodeBenchTask
from .lm_eval_task import MBPPTask, GSM8KTask, TruthfulQATask, MMLUTask


def register_all_tasks():
    """Register all available tasks."""
    # Register LiveCodeBench task
    register_task("livecodebench", LiveCodeBenchTask)
    
    # Register common lm-eval tasks
    register_task("mbpp", MBPPTask)
    register_task("gsm8k", GSM8KTask)
    register_task("truthfulqa_mc1", TruthfulQATask)
    register_task("mmlu", MMLUTask)


# Auto-register tasks when the module is imported
register_all_tasks()


__all__ = [
    "LiveCodeBenchTask",
    "MBPPTask", 
    "GSM8KTask",
    "TruthfulQATask",
    "MMLUTask",
    "register_all_tasks"
]