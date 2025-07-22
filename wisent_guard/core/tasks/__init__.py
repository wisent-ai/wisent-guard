"""
Task implementations for wisent-guard.

This package contains task-agnostic implementations for various benchmarks.
"""

from ..task_interface import register_task
from .livecodebench_task import LiveCodeBenchTask
from .lm_eval_task import MBPPTask, GSM8KTask, TruthfulQATask, MMLUTask
from .hle_task import HLETask, HLEExactMatchTask, HLEMultipleChoiceTask
from .math500_task import Math500Task
from .aime_task import AIMETask


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
    
    # Register AIME tasks (general + year-specific)
    register_task("aime2025-1", lambda limit=None: AIMETask(year="2025", limit=limit, config_name="AIME2025-I"))
    register_task("aime2025-2", lambda limit=None: AIMETask(year="2025", limit=limit, config_name="AIME2025-II"))
    register_task("aime2024", lambda limit=None: AIMETask(year="2024", limit=limit))


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
    "AIMETask",
    "register_all_tasks"
]