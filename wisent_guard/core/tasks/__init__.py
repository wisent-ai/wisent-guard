"""
Task implementations for wisent-guard.

This package contains task-agnostic implementations for various benchmarks.
"""

from ..task_interface import register_task
from .livecodebench_task import LiveCodeBenchTask
from .lm_eval_task import MBPPTask, GSM8KTask, TruthfulQATask, MMLUTask
from .hle_task import HLETask, HLEExactMatchTask, HLEMultipleChoiceTask
from .math500_task import Math500Task
from .aime2024_task import AIME2024Task


def register_all_tasks():
    """Register all available tasks."""
    # Register LiveCodeBench task
    register_task("livecodebench", LiveCodeBenchTask)
    
    # Register common lm-eval tasks
    register_task("mbpp", MBPPTask)
    register_task("gsm8k", GSM8KTask)
    register_task("truthfulqa_mc1", TruthfulQATask)
    register_task("mmlu", MMLUTask)
    
    # Register HLE tasks
    register_task("hle", HLETask)
    register_task("hle_exact_match", HLEExactMatchTask)
    register_task("hle_multiple_choice", HLEMultipleChoiceTask)
    
    # Register MATH-500 tasks
    register_task("math500", Math500Task)
    register_task("math", Math500Task)  
    register_task("hendrycks_math", Math500Task)
    
    # Register AIME 2024 task
    register_task("aime2024", AIME2024Task)


# Auto-register tasks when the module is imported
register_all_tasks()


__all__ = [
    "LiveCodeBenchTask",
    "MBPPTask", 
    "GSM8KTask",
    "TruthfulQATask",
    "MMLUTask",
    "HLETask",
    "HLEExactMatchTask", 
    "HLEMultipleChoiceTask",
    "Math500Task",
    "AIME2024Task",
    "register_all_tasks"
]