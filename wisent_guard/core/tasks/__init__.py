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
from .hmmt_task import HMMTTask
from .polymath_task import PolyMathTask
from .livemathbench_task import LiveMathBenchTask


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
    register_task("aime", lambda limit=None: AIMETask(year="2025", limit=limit))  # Default: latest year (2025)
    register_task("aime2025", lambda limit=None: AIMETask(year="2025", limit=limit))
    register_task("aime2024", lambda limit=None: AIMETask(year="2024", limit=limit))
    
    # Register HMMT tasks (general + competition-specific)
    register_task("hmmt", lambda limit=None: HMMTTask(competition="feb_2025", limit=limit))  # Default: latest competition
    register_task("hmmt_feb_2025", lambda limit=None: HMMTTask(competition="feb_2025", limit=limit))
    
    # Register PolyMath tasks (Chinese and English, medium difficulty)
    register_task("polymath", lambda limit=None: PolyMathTask(language="en", difficulty="medium", limit=limit))  # Default: English medium
    register_task("polymath_en_medium", lambda limit=None: PolyMathTask(language="en", difficulty="medium", limit=limit))
    register_task("polymath_zh_medium", lambda limit=None: PolyMathTask(language="zh", difficulty="medium", limit=limit))
    register_task("polymath_en_high", lambda limit=None: PolyMathTask(language="en", difficulty="high", limit=limit))
    register_task("polymath_zh_high", lambda limit=None: PolyMathTask(language="zh", difficulty="high", limit=limit))
    
    # Register LiveMathBench tasks (CNMO 2024 Chinese and English)
    register_task("livemathbench", lambda limit=None: LiveMathBenchTask(language="en", limit=limit))  # Default: English
    register_task("livemathbench_cnmo_en", lambda limit=None: LiveMathBenchTask(language="en", limit=limit))
    register_task("livemathbench_cnmo_zh", lambda limit=None: LiveMathBenchTask(language="zh", limit=limit))


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
    "HMMTTask",
    "PolyMathTask",
    "LiveMathBenchTask",
    "register_all_tasks"
]