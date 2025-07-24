"""
Centralized task configuration for wisent-guard.

This module defines the allowed tasks and task-related configurations
that are used throughout the application.
"""

# Define the whitelist of allowed tasks
# This combines both lm-eval tasks and TaskInterface tasks
ALLOWED_TASKS = {
    # === Original tested tasks ===
    "math_qa",
    "webqs",
    "truthfulqa_gen",
    "drop",
    "record",
    "squad2",
    "wikitext",
    "winogrande",
    
    # === Multiple choice benchmarks ===
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
    
    # === GPQA benchmarks ===
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
    
    # === Boolean benchmarks ===
    "boolq",
    
    # === Math benchmarks ===
    "gsm8k",
    "asdiv",
    "arithmetic",
    
    # === MATH-500 mathematical reasoning benchmarks ===
    "math",
    "math500",
    "hendrycks_math",
    
    # === AIME contest math problems (general + year-specific) ===
    "aime",        # Latest AIME (2025)
    "aime2025",    # AIME 2025
    "aime2024",    # AIME 2024
    
    # === HMMT contest math problems (general + competition-specific) ===
    "hmmt",        # Latest HMMT (February 2025)
    "hmmt_feb_2025",  # HMMT February 2025
    
    # === PolyMath multilingual mathematical reasoning ===
    "polymath",    # Default: English medium
    "polymath_en_medium",  # English medium
    "polymath_zh_medium",  # Chinese medium
    "polymath_en_high",    # English high
    "polymath_zh_high",    # Chinese high
    
    # === LiveMathBench CNMO 2024 (Chinese and English) ===
    "livemathbench",       # Default: English
    "livemathbench_cnmo_en",  # CNMO 2024 English
    "livemathbench_cnmo_zh",  # CNMO 2024 Chinese
    
    # === QA benchmarks ===
    "coqa",
    "naturalqs",
    "triviaqa",
    
    # === Coding benchmarks ===
    "mbpp",
    "livecodebench",
    
    # === Additional benchmarks ===
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
    
    # === Adversarial benchmarks ===
    "anli",
    
    # === BigCode benchmarks ===
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
    
    # === HLE (Human-Level Evaluation) benchmarks ===
    "hle",
    "hle_exact_match",
    "hle_multiple_choice",
    
    # === SuperGPQA scientific reasoning benchmarks ===
    "supergpqa",
    "supergpqa_physics",
    "supergpqa_chemistry", 
    "supergpqa_biology",
}

# Tasks for CLI integration testing
# This is a subset of ALLOWED_TASKS that are known to work with our test infrastructure
TEST_ALLOWED_TASKS = [
    # GPQA benchmarks
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
    # Boolean benchmarks
    "boolq",
    # Math benchmarks
    "gsm8k",
    "asdiv",
    "arithmetic",
    # MATH-500 mathematical reasoning benchmarks
    "math",
    "math500",
    "hendrycks_math",
    # AIME contest math problems (general + year-specific)
    "aime",        # Latest AIME (2025)
    "aime2025",    # AIME 2025
    "aime2024",    # AIME 2024
    # HMMT contest math problems (general + competition-specific)
    "hmmt",        # Latest HMMT (February 2025)
    "hmmt_feb_2025",  # HMMT February 2025
    # PolyMath multilingual mathematical reasoning (Chinese and English, medium difficulty)
    "polymath",    # Default: English medium
    "polymath_en_medium",  # English medium
    "polymath_zh_medium",  # Chinese medium
    "polymath_en_high",    # English high
    "polymath_zh_high",    # Chinese high
    # LiveMathBench CNMO 2024 (Chinese and English)
    "livemathbench",       # Default: English
    "livemathbench_cnmo_en",  # CNMO 2024 English
    "livemathbench_cnmo_zh",  # CNMO 2024 Chinese,
    # HLE (Human-Level Evaluation) benchmarks
    "hle",
    "hle_exact_match",
    "hle_multiple_choice",
    # SuperGPQA scientific reasoning benchmarks
    "supergpqa",
    "supergpqa_physics",
    "supergpqa_chemistry", 
    "supergpqa_biology",
]

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