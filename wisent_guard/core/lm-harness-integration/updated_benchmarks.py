#!/usr/bin/env python3
"""
Updated benchmark definitions with README-based tags.
"""

CORE_BENCHMARKS = {
    # Benchmark Suites
    "glue": {
        "task": "glue",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "superglue": {
        "task": "superglue",
        "tags": ["reasoning", "general knowledge", "adversarial robustness"]
    },

    # SuperGLUE individual tasks
    "cb": {
        "task": "cb",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "copa": {
        "task": "copa",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "multirc": {
        "task": "multirc",
        "tags": ["reasoning", "long context", "general knowledge"]
    },
    "record": {
        "task": "record",
        "tags": ["reasoning", "long context", "general knowledge"]
    },
    "wic": {
        "task": "wic",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "wsc": {
        "task": "wsc",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Hallucination and Truthfulness
    "truthfulqa_mc1": {
        "task": "truthfulqa_mc1",
        "tags": ["hallucination", "general knowledge", "reasoning"]
    },
    "truthfulqa_mc2": {
        "task": "truthfulqa_mc2",
        "tags": ["hallucination", "general knowledge", "reasoning"]
    },
    "truthfulqa_gen": {
        "task": "truthfulqa_gen",
        "tags": ["hallucination", "general knowledge", "reasoning"]
    },

    # Reasoning and Comprehension
    "hellaswag": {
        "task": "hellaswag",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "piqa": {
        "task": "piqa",
        "tags": ["reasoning", "science", "general knowledge"]
    },
    "winogrande": {
        "task": "winogrande",
        "tags": ["reasoning", "general knowledge", "adversarial robustness"]
    },
    "openbookqa": {
        "task": "openbookqa",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "swag": {
        "task": "swag",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "storycloze": {
        "task": "storycloze",
        "tags": ["long context", "creative writing", "reasoning"]
    },
    "logiqa": {
        "task": "logiqa",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "wsc273": {
        "task": "wsc273",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Reading Comprehension and QA
    "coqa": {
        "task": "coqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "drop": {
        "task": "drop",
        "tags": ["mathematics", "reasoning", "long context"]
    },
    "boolq": {
        "task": "boolq",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "race": {
        "task": "race",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "squad2": {
        "task": "squad2",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "triviaqa": {
        "task": "triviaqa",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "naturalqs": {
        "task": "nq_open",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "webqs": {
        "task": "webqs",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "headqa": {
        "task": "headqa",
        "tags": ["medical", "multilingual", "adversarial robustness"]
    },
    "qasper": {
        "task": "qasper",
        "tags": ["science", "long context", "reasoning"]
    },
    "qa4mre": {
        "task": "qa4mre",
        "tags": ["medical", "long context", "reasoning"]
    },
    "mutual": {
        "task": "mutual",
        "tags": ["long context", "reasoning", "general knowledge"]
    },

    # Knowledge and Academic
    "mmlu": {
        "task": "mmlu",
        "tags": ["general knowledge", "science", "reasoning"]
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
        "tags": ["mathematics", "reasoning", "science"]
    },
    "math_qa": {
        "task": "math_qa",
        "tags": ["mathematics", "reasoning", "science"]
    },
    "hendrycks_math": {
        "task": "hendrycks_math",
        "tags": ["mathematics", "reasoning", "science"]
    },
    "arithmetic": {
        "task": "arithmetic",
        "tags": ["mathematics", "long context", "reasoning"]
    },
    "asdiv": {
        "task": "asdiv",
        "tags": ["mathematics", "adversarial robustness", "long context"]
    },

    # Coding
    "humaneval": {
        "task": "humaneval",
        "tags": ["coding", "reasoning", "mathematics"]
    },
    "mbpp": {
        "task": "mbpp",
        "tags": ["coding", "reasoning", "mathematics"]
    },

    # Bias and Toxicity
    "toxigen": {
        "task": "toxigen",
        "tags": ["adversarial robustness", "long context", "reasoning"]
    },
    "crows_pairs": {
        "task": "crows_pairs",
        "tags": ["bias", "reasoning", "general knowledge"]
    },
    "hendrycks_ethics": {
        "task": "hendrycks_ethics",
        "tags": ["long context", "reasoning", "general knowledge"]
    },

    # Adversarial
    "anli": {
        "task": "anli",
        "tags": ["adversarial robustness", "reasoning", "general knowledge"]
    },

    # Multilinguality
    "xnli": {
        "task": "xnli",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "xcopa": {
        "task": "xcopa",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "xstorycloze": {
        "task": "xstorycloze",
        "tags": ["multilingual", "long context", "creative writing"]
    },
    "xwinograd": {
        "task": "xwinograd",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "paws_x": {
        "task": "paws_x",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "mmmlu": {
        "task": "mmmlu",
        "tags": ["general knowledge", "science", "reasoning"]
    },
    "mgsm": {
        "task": "mgsm",
        "tags": ["multilingual", "mathematics", "reasoning"]
    },
    "belebele": {
        "task": "belebele",
        "tags": ["multilingual", "adversarial robustness", "long context"]
    },

    # Medical and Law
    "medqa": {
        "task": "medqa",
        "tags": ["medical", "science", "general knowledge"]
    },
    "pubmedqa": {
        "task": "pubmedqa",
        "tags": ["medical", "science", "reasoning"]
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
        "tags": ["long context", "reasoning", "general knowledge"]
    },

    # Long Context
    "narrativeqa": {
        "task": "narrativeqa",
        "tags": ["reasoning", "long context", "general knowledge"]
    },
    "scrolls": {
        "task": "scrolls",
        "tags": ["long context", "reasoning", "general knowledge"]
    },

    # Temporal and Event Understanding
    "mctaco": {
        "task": "mctaco",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "prost": {
        "task": "prost",
        "tags": ["long context", "history", "reasoning"]
    },

    # Linguistic Understanding
    "blimp": {
        "task": "blimp",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "unscramble": {
        "task": "unscramble",
        "tags": ["long context", "reasoning", "general knowledge"]
    },

    # Translation
    "wmt": {
        "task": "wmt",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Comprehensive Suites
    "big_bench": {
        "task": "big_bench",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Dialogue and Conversation
    "babi": {
        "task": "babi",
        "tags": ["reasoning", "general knowledge", "long context"]
    }
}
