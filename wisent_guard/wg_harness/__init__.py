"""
wisent-guard integration with lm-evaluation-harness

This package provides tools to run LM evaluation benchmarks through the wisent-guard
pipeline for detecting harmful content, hallucinations, and other undesirable behaviors.

Usage:
    python -m wisent tasks hellaswag,mmlu --layer 15 --model meta-llama/Llama-3.1-8B
"""

from .data import load_task, split_docs
from .generate import generate_responses
from .labeler import label_responses
from .train_guard import GuardPipeline
from .evaluate import evaluate_guard
from .cli import main as cli_main

__version__ = "0.1.0"

__all__ = [
    "load_task",
    "split_docs", 
    "generate_responses",
    "label_responses",
    "GuardPipeline",
    "evaluate_guard",
    "cli_main"
] 