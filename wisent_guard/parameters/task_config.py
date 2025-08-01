"""
Centralized task configuration for wisent-guard.

This module defines the allowed tasks and task-related configurations
that are used throughout the application.
"""

# === Task Groups ===

# Original tested tasks - VERIFIED WORKING
ORIGINAL_TASKS_WORKING = {
    "drop",
    "record",
    "squad2",
    "wikitext",
    "winogrande",
    "webqs",
}

NEEDS_TRUST_REMOTE_CODE = {
    "math_qa",  # ❌ Requires trust_remote_code (unfixable per user request)
}
# Problematic original tasks (unfixable or missing)
ORIGINAL_TASKS_TO_FIX = {
    "truthfulqa_gen",  # ❌ Task doesn't exist (only truthfulqa_mc1/mc2 available)
}

# Keep original name for backward compatibility but now use all working tasks
ORIGINAL_TASKS = ORIGINAL_TASKS_WORKING

# Multiple choice benchmarks
MULTIPLE_CHOICE_TASKS = {
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "mmlu",
    "mmmlu",
    "piqa",
    "copa",
    "openbookqa",
    "race",
    "boolq",
}

# GPQA scientific reasoning benchmarks
GPQA_TASKS = {
    "gpqa",
    "gpqa_diamond",
    "gpqa_extended",
    # GPQA specific variants (zeroshot only for focused testing)
    "gpqa_main_zeroshot",
    "gpqa_diamond_zeroshot",
    "gpqa_extended_zeroshot",
    # GPQA Chain-of-Thought variants for text generation testing
    "gpqa_main_cot_zeroshot",
    "gpqa_diamond_cot_zeroshot",
    "gpqa_extended_cot_zeroshot",
    # SuperGPQA scientific reasoning benchmarks
    "supergpqa",
    "supergpqa_physics",
    "supergpqa_chemistry",
    "supergpqa_biology",
}

# Mathematical reasoning benchmarks
MATH_TASKS = {
    # Basic math benchmarks
    "gsm8k",
    # MATH-500 mathematical reasoning benchmarks
    "math",
    "math500",
    "hendrycks_math",
    # AIME contest math problems (general + year-specific)
    "aime",  # Latest AIME (2025)
    "aime2025",  # AIME 2025
    "aime2024",  # AIME 2024
    # HMMT contest math problems (general + competition-specific)
    "hmmt",  # Latest HMMT (February 2025)
    "hmmt_feb_2025",  # HMMT February 2025
    # PolyMath multilingual mathematical reasoning
    "polymath",  # Default: English medium
    "polymath_en_medium",  # English medium
    "polymath_zh_medium",  # Chinese medium
    "polymath_en_high",  # English high
    "polymath_zh_high",  # Chinese high
    # LiveMathBench CNMO 2024 (Chinese and English)
    "livemathbench",  # Default: English
    "livemathbench_cnmo_en",  # CNMO 2024 English
    "livemathbench_cnmo_zh",  # CNMO 2024 Chinese
}

# QA benchmarks
QA_TASKS = {
    "coqa",
    "naturalqs",
    "triviaqa",
}

# Coding benchmarks
CODING_TASKS = {
    "mbpp",
    "livecodebench",
    # BigCode benchmarks
    "humaneval",
    "humaneval_plus",
    "instructhumaneval",
    "apps",
    "mbpp_plus",
    "ds1000",
    "humanevalpack",
    "multiple_py",
    "multiple_js",
    "multiple_java",
    "multiple_cpp",
    "multiple_rs",
    "multiple_go",
    "recode",
    "conala",
    "concode",
    "codexglue_code_to_text",
    "codexglue_code_to_text_python",
    "codexglue_code_to_text_go",
    "codexglue_code_to_text_ruby",
    "codexglue_code_to_text_java",
    "codexglue_code_to_text_javascript",
    "codexglue_code_to_text_php",
    "mercury",
}

# HLE (Human-Level Evaluation) benchmarks
HLE_TASKS = {
    "hle",
    "hle_exact_match",
    "hle_multiple_choice",
}

# Additional miscellaneous benchmarks
MISC_TASKS = {
    "cb",
    "logiqa",
    "multirc",
    "mutual",
    "prost",
    "pubmedqa",
    "sciq",
    "swag",
    "toxigen",
    "wic",
    "wsc",
    "wsc273",
    "anli",  # Adversarial benchmarks
}

# Combine all task groups into the final allowed tasks set
# This uses Python's unpacking operator (*) to merge sets - equivalent to [...array] in JavaScript
ALLOWED_TASKS = {
    *ORIGINAL_TASKS,  # Now includes all working original tasks (drop, record, squad2, wikitext, winogrande)
    *MULTIPLE_CHOICE_TASKS,
    *GPQA_TASKS,
    *MATH_TASKS,
    *QA_TASKS,
    *CODING_TASKS,
    *HLE_TASKS,
    *MISC_TASKS,
}

# Tasks that are verified to work in CI tests
# Including working ORIGINAL_TASKS + previously tested task groups
TEST_ALLOWED_TASKS = list(
    {
        *ORIGINAL_TASKS,  # Only verified working original tasks (drop, record)
        *MULTIPLE_CHOICE_TASKS,  # All multiple choice tasks work
        *QA_TASKS,  # All QA tasks work
        *GPQA_TASKS,  # All GPQA tasks work
        *MATH_TASKS,  # All math tasks work
        *HLE_TASKS,  # All HLE tasks work
    }
)


def get_taskinterface_tasks():
    """
    Get all tasks registered in the TaskInterface registry.

    This provides dynamic access to TaskInterface tasks without hardcoding.
    """
    try:
        from ..core.task_interface import list_tasks

        return set(list_tasks())
    except ImportError:
        # Return empty set if task_interface is not available
        return set()


def get_all_available_tasks():
    """
    Get all available tasks by combining ALLOWED_TASKS with registered TaskInterface tasks.

    This ensures we have a complete list of all tasks that can be used.
    """
    taskinterface_tasks = get_taskinterface_tasks()
    return ALLOWED_TASKS | taskinterface_tasks
