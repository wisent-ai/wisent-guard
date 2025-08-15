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
    # "mmlu",  # Removed: Not working in lm-evaluation-harness currently (see https://github.com/EleutherAI/lm-evaluation-harness/issues/3171)
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
    "asdiv",
    "arithmetic",
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
    # === VERIFIED WORKING (✅) ===
    "mbpp",  # ✅ Direct match
    "humaneval",  # ✅ Direct match
    "conala",  # ✅ Direct match
    "concode",  # ✅ Direct match
    "mercury",  # ✅ Direct match
    # === TASKS WITH INTERNAL MAPPING (✅ has BigCode mapping) ===
    "humaneval_plus",  # → maps to 'humanevalplus' in BigCode
    "instructhumaneval",  # → maps to 'instruct-humaneval' in BigCode
    "mbpp_plus",  # → maps to 'mbppplus' in BigCode
    "apps",  # → maps to 'apps-introductory' in BigCode
    "ds1000",  # → maps to 'ds1000-all-completion' in BigCode
    # === MULTI-LANGUAGE TASKS (✅ has BigCode mapping) ===
    "multiple_py",  # → maps to 'multiple-py' in BigCode
    "multiple_js",  # → maps to 'multiple-js' in BigCode
    "multiple_java",  # → maps to 'multiple-java' in BigCode
    "multiple_cpp",  # → maps to 'multiple-cpp' in BigCode (need to fix mapping)
    "multiple_rs",  # → maps to 'multiple-rs' in BigCode
    "multiple_go",  # → maps to 'multiple-go' in BigCode
    # === CODE-TO-TEXT TASKS (✅ has BigCode mapping) ===
    "codexglue_code_to_text_python",  # → maps to 'codexglue_code_to_text-python'
    "codexglue_code_to_text_go",  # → maps to 'codexglue_code_to_text-go'
    "codexglue_code_to_text_ruby",  # → maps to 'codexglue_code_to_text-ruby'
    "codexglue_code_to_text_java",  # → maps to 'codexglue_code_to_text-java'
    "codexglue_code_to_text_javascript",  # → maps to 'codexglue_code_to_text-javascript'
    "codexglue_code_to_text_php",  # → maps to 'codexglue_code_to_text-php'
    # === FIXED MAPPINGS (✅) ===
    "recode",  # → now maps to 'perturbed-humaneval-natgen-num_seeds_1'
    # === REMOVED BROKEN TASKS ===
    # "humanevalpack",  # ❌ REMOVED - no simple BigCode mapping exists
    # === REMOVED NON-EXISTENT TASKS ===
    # "livecodebench",     # ❌ Not in BigCode registry - REMOVED
}

# HLE (Human-Level Evaluation) benchmarks
HLE_TASKS = {
    "hle",
    "hle_exact_match",
    "hle_multiple_choice",
}

# Additional miscellaneous benchmarks - ALL WORKING ✅
MISC_TASKS = {
    "swag",  # ✅ FIXED - now uses updated _convert_multiple_choice_numeric()
}

# Tasks that were in original MISC list but are NOT AVAILABLE in current system:
MISC_TASKS_NOT_AVAILABLE = {
    "cb",  #  TODO Need investigation, due to not creating sufficient contrastive pairs
    "anli",  # ❌ Not available in current wisent-guard system
    "logiqa",  # ❌ Not available in current wisent-guard system
    "multirc",  # ❌ Not available in current wisent-guard system
    "mutual",  # ❌ Not available in current wisent-guard system
    "prost",  # ❌ Not available in current wisent-guard system
    "pubmedqa",  # ❌ Not available in current wisent-guard system
    "sciq",  # ❌ Not available in current wisent-guard system
    "toxigen",  # ❌ Not available in current wisent-guard system
    "wic",  # ❌ Not available in current wisent-guard system
    "wsc",  # ❌ Not available in current wisent-guard system
    "wsc273",  # ❌ Not available in current wisent-guard system
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
        *ORIGINAL_TASKS,  # All verified working original tasks (drop, record, squad2, wikitext, winogrande, webqs)
        *MULTIPLE_CHOICE_TASKS,  # All multiple choice tasks work
        *QA_TASKS,  # All QA tasks work
        *MISC_TASKS,  # All working MISC tasks (cb, swag)
        *GPQA_TASKS,  # All GPQA tasks work
        *MATH_TASKS,  # All math tasks work
        *HLE_TASKS,  # All HLE tasks work
    }
)

# Tasks that can be tested in sandbox environments (includes coding tasks)
# These are tasks that require special handling or are unsafe for CI
SANDBOX_TESTS_ALLOWED_TASKS = list(
    {
        *CODING_TASKS,  # All coding tasks requiring bigcode-evaluation-harness and --trust-code-execution
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
