from enum import Enum


class TokenTargetingStrategy(Enum):
    """Different strategies for targeting tokens in activation extraction."""

    CHOICE_TOKEN = "choice_token"  # Target A/B choice tokens (for multiple choice)
    CONTINUATION_TOKEN = "continuation_token"  # Target first token of continuation ("I", etc.)
    LAST_TOKEN = "last_token"  # Always use last token
    FIRST_TOKEN = "first_token"  # Always use first token
    MEAN_POOLING = "mean_pooling"  # Use mean of all tokens
    MAX_POOLING = "max_pooling"  # Use max pooling across tokens
