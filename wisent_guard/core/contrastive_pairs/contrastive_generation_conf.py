"""This file contains configs for synthetic contrastive pair generation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScenarioGeneration:
    """Resources for generating scenarios."""

    PROMPT_TEMPLATES: list[str] = field(
        default_factory=lambda: [
            "List {num_prompts} very short, simple questions (maximum 10 words) about everyday life:\n1.",
            "Generate {num_prompts} brief open-ended questions about common situations:\n1.",
            "Create {num_prompts} short questions people might ask in casual conversation:\n1.",
            "Write {num_prompts} concise questions about opinions, decisions, or advice:\n1.",
        ]
    )
    OVERGENERATION_FACTOR: int = 5
    CONFIG: dict[str, Any] = field(
        default_factory=lambda: {
            "layer_index": 15,
            "max_new_tokens": 500,
            "temperature": 0.9,
            "do_sample": True,
        }
    )


@dataclass
class ScenarioParsing:
    """Constants for parsing and filtering generated scenarios."""

    SKIP_PHRASES: list[str] = field(
        default_factory=lambda: [
            "here are",
            "here is",
            "these are",
            "this is",
            "the following",
            "examples of",
            "scenarios where",
            "situations where",
            "cases where",
            "questions that",
            "prompts that",
            "list of",
            "different situations",
            "advice-seeking:",
            "opinion:",
            "conversation starter:",
            "prompt:",
            "scenario:",
        ]
    )
    REFUSAL_PHRASES: list[str] = field(
        default_factory=lambda: [
            "i cannot",
            "i can't",
            "i am not able",
            "i cannot provide",
            "i cannot create",
            "i cannot generate",
            "harmful",
            "evil behavior",
            "glorify",
        ]
    )

    MIN_SCENARIO_LENGTH: int = 10
    MAX_SCENARIO_LENGTH: int = 220
    MAX_QUESTION_WORDS: int = 50
    MAX_IMPERATIVE_WORDS: int = 50
    CANDIDATE_LIMIT: int = 25
    IMPERATIVE_KEYWORDS: list[str] = field(
        default_factory=lambda: ["tell me", "explain", "describe", "what", "how", "why"]
    )


@dataclass
class ContrastivePairGeneration:
    """Resources for generating the positive/negative pairs."""

    POSITIVE_PROMPT_TEMPLATE: str = (
        "{scenario}\n\nRespond directly with {trait_description} (no preamble):"
    )
    NEGATIVE_PROMPT_TEMPLATE: str = (
        "{scenario}\n\nRespond directly with the opposite of {trait_description} (no preamble):"
    )
    CONFIG: dict[str, Any] = field(
        default_factory=lambda: {
            "layer_index": 15,
            "max_new_tokens": 50,
            "temperature": 0.8,
            "do_sample": True,
        }
    )


# Instantiate for easy import
SCENARIO_GEN = ScenarioGeneration()
SCENARIO_PARSE = ScenarioParsing()
CONTRASTIVE_GEN = ContrastivePairGeneration()
