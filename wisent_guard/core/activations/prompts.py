from enum import Enum


class PromptConstructionStrategy(Enum):
    """Different strategies for constructing prompts from question-answer pairs."""

    MULTIPLE_CHOICE = "multiple_choice"  # Original: "Which is better: Q A. bad B. good" -> "A"/"B"
    ROLE_PLAYING = "role_playing"  # "Behave like person who would answer Q with good_resp" -> "I"
    DIRECT_COMPLETION = "direct_completion"  # "Q" -> "good_resp"/"bad_resp"
    INSTRUCTION_FOLLOWING = "instruction_following"  # "[INST] Q [/INST]" -> "good_resp"/"bad_resp"


class PromptPair:
    """Represents a pair of prompts for positive and negative cases."""

    def __init__(self, positive_prompt: str, negative_prompt: str, target_token: str = None):
        self.positive_prompt = positive_prompt
        self.negative_prompt = negative_prompt
        self.target_token = target_token  # Token to target for extraction
