"""
Task implementations for wisent-guard.

This package contains task-agnostic implementations for various benchmarks.
"""

from ..task_interface import register_task
from .aime_task import AIMETask
from .hle_task import HLEExactMatchTask, HLEMultipleChoiceTask, HLETask
from .hmmt_task import HMMTTask
from .livecodebench_task import LiveCodeBenchTask
from .livemathbench_task import LiveMathBenchTask
from .lm_eval_task import (
    AppsTask,
    CodexglueCodeToTextGoTask,
    CodexglueCodeToTextJavascriptTask,
    CodexglueCodeToTextJavaTask,
    CodexglueCodeToTextPhpTask,
    CodexglueCodeToTextPythonTask,
    CodexglueCodeToTextRubyTask,
    ConalaTask,
    ConcodeTask,
    DS1000Task,
    GSM8KTask,
    HumanEvalPlusTask,
    HumanEvalTask,
    InstructHumanEvalTask,
    MBPPPlusTask,
    MBPPTask,
    MercuryTask,
    MMLUTask,
    MultipleCppTask,
    MultipleGoTask,
    MultipleJavaTask,
    MultipleJsTask,
    MultiplePyTask,
    MultipleRsTask,
    RecodeTask,
    Squad2Task,
    TruthfulQATask,
)
from .math500_task import Math500Task
from .polymath_task import PolyMathTask
from .supergpqa_task import SuperGPQABiologyTask, SuperGPQAChemistryTask, SuperGPQAPhysicsTask, SuperGPQATask


def register_all_tasks():
    """Register all available tasks."""
    # Register LiveCodeBench task
    register_task("livecodebench", lambda limit=None: LiveCodeBenchTask(release_version="release_v1", limit=limit))

    # Register common lm-eval tasks
    register_task("gsm8k", GSM8KTask)
    register_task("truthfulqa_mc1", TruthfulQATask)
    register_task("mmlu", MMLUTask)

    # Register all coding tasks
    register_task("mbpp", MBPPTask)
    register_task("humaneval", HumanEvalTask)
    register_task("mbpp_plus", MBPPPlusTask)
    register_task("instructhumaneval", InstructHumanEvalTask)
    register_task("humaneval_plus", HumanEvalPlusTask)
    register_task("conala", ConalaTask)
    register_task("concode", ConcodeTask)
    register_task("mercury", MercuryTask)
    register_task("apps", AppsTask)
    register_task("ds1000", DS1000Task)
    register_task("multiple_py", MultiplePyTask)
    register_task("multiple_js", MultipleJsTask)
    register_task("multiple_java", MultipleJavaTask)
    register_task("multiple_cpp", MultipleCppTask)
    register_task("multiple_rs", MultipleRsTask)
    register_task("multiple_go", MultipleGoTask)
    register_task("codexglue_code_to_text_python", CodexglueCodeToTextPythonTask)
    register_task("codexglue_code_to_text_go", CodexglueCodeToTextGoTask)
    register_task("codexglue_code_to_text_ruby", CodexglueCodeToTextRubyTask)
    register_task("codexglue_code_to_text_java", CodexglueCodeToTextJavaTask)
    register_task("codexglue_code_to_text_javascript", CodexglueCodeToTextJavascriptTask)
    register_task("codexglue_code_to_text_php", CodexglueCodeToTextPhpTask)
    register_task("recode", RecodeTask)
    register_task("squad2", Squad2Task)

    # Register HLE tasks
    register_task("hle", lambda limit=None: HLETask(limit=limit))
    register_task("hle_exact_match", lambda limit=None: HLEExactMatchTask(limit=limit))
    register_task("hle_multiple_choice", lambda limit=None: HLEMultipleChoiceTask(limit=limit))

    # Register MATH-500 tasks
    register_task("math500", lambda limit=None: Math500Task(limit=limit))
    register_task("math", lambda limit=None: Math500Task(limit=limit))
    register_task("hendrycks_math", lambda limit=None: Math500Task(limit=limit))

    # Register AIME tasks (general + year-specific)
    register_task("aime", lambda limit=None: AIMETask(year="2025", limit=limit))  # Default: latest year (2025)
    register_task("aime2025", lambda limit=None: AIMETask(year="2025", limit=limit))
    register_task("aime2024", lambda limit=None: AIMETask(year="2024", limit=limit))

    # Register HMMT tasks (general + competition-specific)
    register_task(
        "hmmt", lambda limit=None: HMMTTask(competition="feb_2025", limit=limit)
    )  # Default: latest competition
    register_task("hmmt_feb_2025", lambda limit=None: HMMTTask(competition="feb_2025", limit=limit))

    # Register PolyMath tasks (Chinese and English, medium difficulty)
    register_task(
        "polymath", lambda limit=None: PolyMathTask(language="en", difficulty="medium", limit=limit)
    )  # Default: English medium
    register_task(
        "polymath_en_medium", lambda limit=None: PolyMathTask(language="en", difficulty="medium", limit=limit)
    )
    register_task(
        "polymath_zh_medium", lambda limit=None: PolyMathTask(language="zh", difficulty="medium", limit=limit)
    )
    register_task("polymath_en_high", lambda limit=None: PolyMathTask(language="en", difficulty="high", limit=limit))
    register_task("polymath_zh_high", lambda limit=None: PolyMathTask(language="zh", difficulty="high", limit=limit))

    # Register LiveMathBench tasks (CNMO 2024 Chinese and English)
    register_task("livemathbench", lambda limit=None: LiveMathBenchTask(language="en", limit=limit))  # Default: English
    register_task("livemathbench_cnmo_en", lambda limit=None: LiveMathBenchTask(language="en", limit=limit))
    register_task("livemathbench_cnmo_zh", lambda limit=None: LiveMathBenchTask(language="zh", limit=limit))

    # Register SuperGPQA tasks (scientific reasoning)
    register_task("supergpqa", lambda limit=None: SuperGPQATask(limit=limit))  # Default: all subjects
    register_task("supergpqa_physics", lambda limit=None: SuperGPQAPhysicsTask(limit=limit))
    register_task("supergpqa_chemistry", lambda limit=None: SuperGPQAChemistryTask(limit=limit))
    register_task("supergpqa_biology", lambda limit=None: SuperGPQABiologyTask(limit=limit))


# Auto-register tasks when the module is imported
register_all_tasks()


__all__ = [
    "AIMETask",
    "AppsTask",
    "CodexglueCodeToTextGoTask",
    "CodexglueCodeToTextJavaTask",
    "CodexglueCodeToTextJavascriptTask",
    "CodexglueCodeToTextPhpTask",
    "CodexglueCodeToTextPythonTask",
    "CodexglueCodeToTextRubyTask",
    "ConalaTask",
    "ConcodeTask",
    "DS1000Task",
    "GSM8KTask",
    "HLEExactMatchTask",
    "HLEMultipleChoiceTask",
    "HLETask",
    "HMMTTask",
    "HumanEvalPlusTask",
    "HumanEvalTask",
    "InstructHumanEvalTask",
    "LiveCodeBenchTask",
    "LiveMathBenchTask",
    "MBPPPlusTask",
    "MBPPTask",
    "MMLUTask",
    "Math500Task",
    "MercuryTask",
    "MultipleCppTask",
    "MultipleGoTask",
    "MultipleJavaTask",
    "MultipleJsTask",
    "MultiplePyTask",
    "MultipleRsTask",
    "PolyMathTask",
    "RecodeTask",
    "Squad2Task",
    "SuperGPQABiologyTask",
    "SuperGPQAChemistryTask",
    "SuperGPQAPhysicsTask",
    "SuperGPQATask",
    "TruthfulQATask",
    "register_all_tasks",
]
