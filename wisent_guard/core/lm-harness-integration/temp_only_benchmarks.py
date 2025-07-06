#!/usr/bin/env python3
"""
Updated benchmark definitions with README-based tags.
"""

CORE_BENCHMARKS = {
    # Benchmark Suites
    "glue": {
        "task": "glue",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "superglue": {
        "task": "superglue",
        "tags": ["reasoning", "general knowledge", "long context"]
    },

    # SuperGLUE individual tasks
    "cb": {
        "task": "cb",
        "tags": ["reasoning", "general knowledge", "adversarial robustness"]
    },
    "copa": {
        "task": "copa",
        "tags": ["reasoning", "general knowledge", "creative writing"]
    },
    "multirc": {
        "task": "multirc",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "record": {
        "task": "record",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "wic": {
        "task": "wic",
        "tags": ["reasoning", "general knowledge", "multilingual"]
    },
    "wsc": {
        "task": "wsc",
        "tags": ["reasoning", "general knowledge", "adversarial robustness"]
    },

    # Hallucination and Truthfulness
    "truthfulqa_mc1": {
        "task": "truthfulqa_mc1",
        "tags": ["hallucination", "deception", "general knowledge"]
    },
    "truthfulqa_mc2": {
        "task": "truthfulqa_mc2",
        "tags": ["hallucination", "deception", "general knowledge"]
    },
    "truthfulqa_gen": {
        "task": "truthfulqa_gen",
        "tags": ["hallucination", "deception", "general knowledge"]
    },

    # Reasoning and Comprehension
    "hellaswag": {
        "task": "hellaswag",
        "tags": ["reasoning", "long context", "multilingual"]
    },
    "piqa": {
        "task": "piqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "winogrande": {
        "task": "winogrande",
        "tags": ["reasoning", "long context", "bias"]
    },
    "openbookqa": {
        "task": "openbookqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "swag": {
        "task": "swag",
        "tags": ["reasoning", "long context", "creative writing"]
    },
    "storycloze": {
        "task": "storycloze",
        "tags": ["reasoning", "long context", "creative writing"]
    },
    "logiqa": {
        "task": "logiqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "wsc273": {
        "task": "wsc273",
        "tags": ["reasoning", "general knowledge", "long context"]
    },

    # Reading Comprehension and QA
    "coqa": {
        "task": "coqa",
        "tags": ["general knowledge", "long context", "reasoning"]
    },
    "drop": {
        "task": "drop",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "boolq": {
        "task": "boolq",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "race": {
        "task": "race",
        "tags": ["reasoning", "long context", "multilingual"]
    },
    "squad2": {
        "task": "squad2",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "triviaqa": {
        "task": "triviaqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "naturalqs": {
        "task": "nq_open",
        "tags": ["general knowledge", "tool use", "reasoning"]
    },
    "webqs": {
        "task": "webqs",
        "tags": ["general knowledge", "long context", "multilingual"]
    },
    "headqa": {
        "task": "headqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "qasper": {
        "task": "qasper",
        "tags": ["general knowledge", "long context", "creative writing"]
    },
    "qa4mre": {
        "task": "qa4mre",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "mutual": {
        "task": "mutual",
        "tags": ["reasoning", "long context", "general knowledge"]
    },

    # Knowledge and Academic
    "mmlu": {
        "task": "mmlu",
        "tags": ["reasoning", "long context", "creative writing"]
    },
    "ai2_arc": {
        "task": "ai2_arc",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "arc_easy": {
        "task": "arc_easy",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "arc_challenge": {
        "task": "arc_challenge",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "sciq": {
        "task": "sciq",
        "tags": ["long context", "science", "reasoning"]
    },
    "social_i_qa": {
        "task": "social_i_qa",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Mathematics
    "gsm8k": {
        "task": "gsm8k",
        "tags": ["reasoning", "creative writing", "mathematics"]
    },
    "math_qa": {
        "task": "math_qa",
        "tags": ["mathematics", "reasoning", "general knowledge"]
    },
    "hendrycks_math": {
        "task": "hendrycks_math",
        "tags": ["long context", "mathematics", "coding"]
    },
    "arithmetic": {
        "task": "arithmetic",
        "tags": ["general knowledge", "long context", "mathematics"]
    },
    "asdiv": {
        "task": "asdiv",
        "tags": ["long context", "creative writing", "mathematics"]
    },

    # Coding
    "humaneval": {
        "task": "humaneval",
        "tags": ["long context", "creative writing", "coding"]
    },
    "mbpp": {
        "task": "mbpp",
        "tags": ["general knowledge", "long context", "creative writing"]
    },

    # Bias and Toxicity
    "toxigen": {
        "task": "toxigen",
        "tags": ["long context", "toxicity", "adversarial robustness"]
    },
    "crows_pairs": {
        "task": "crows_pairs",
        "tags": ["long context", "multilingual", "history"]
    },
    "hendrycks_ethics": {
        "task": "hendrycks_ethics",
        "tags": ["general knowledge", "long context", "law"]
    },

    # Adversarial
    "anli": {
        "task": "anli",
        "tags": ["reasoning", "long context", "multilingual"]
    },

    # Multilinguality
    "xnli": {
        "task": "xnli",
        "tags": ["long context", "multilingual", "reasoning"]
    },
    "xcopa": {
        "task": "xcopa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "xstorycloze": {
        "task": "xstorycloze",
        "tags": ["long context", "creative writing", "science"]
    },
    "xwinograd": {
        "task": "xwinograd",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "paws_x": {
        "task": "paws_x",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "mmmlu": {
        "task": "mmmlu",
        "tags": ["multilingual", "general knowledge", "reasoning"]
    },
    "mgsm": {
        "task": "mgsm",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "belebele": {
        "task": "belebele",
        "tags": ["reasoning", "general knowledge", "long context"]
    },

    # Medical and Law
    "medqa": {
        "task": "medqa",
        "tags": ["medical", "reasoning", "general knowledge"]
    },
    "pubmedqa": {
        "task": "pubmedqa",
        "tags": ["general knowledge", "long context", "medical"]
    },

    # Language Modeling and Generation
    "lambada": {
        "task": "lambada",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "lambada_cloze": {
        "task": "lambada_cloze",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "lambada_multilingual": {
        "task": "lambada_multilingual",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "wikitext": {
        "task": "wikitext",
        "tags": ["long context", "multilingual", "reasoning"]
    },

    # Long Context
    "narrativeqa": {
        "task": "narrativeqa",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "scrolls": {
        "task": "scrolls",
        "tags": ["reasoning", "general knowledge", "long context"]
    },

    # Temporal and Event Understanding
    "mctaco": {
        "task": "mctaco",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "prost": {
        "task": "prost",
        "tags": ["reasoning", "long context", "multilingual"]
    },

    # Linguistic Understanding
    "blimp": {
        "task": "blimp",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "unscramble": {
        "task": "unscramble",
        "tags": ["general knowledge", "long context", "multilingual"]
    },

    # Translation
    "wmt": {
        "task": "wmt",
        "tags": ["multilingual", "creative writing", "general knowledge"]
    },

    # Comprehensive Suites
    "big_bench": {
        "task": "big_bench",
        "tags": ["general knowledge", "reasoning", "creative writing"]
    },

    # Dialogue and Conversation
    "babi": {
        "task": "babi",
        "tags": ["reasoning", "general knowledge", "long context"]
    }
}
