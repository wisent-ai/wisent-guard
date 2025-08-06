"""
Essential utility functions for CAA validation tests.

These functions provide CAA-compatible tokenization and probability extraction
without requiring the external CAA repository.
"""

import torch
from typing import List


def tokenize_llama_base_format(tokenizer, user_input: str, model_output: str = None) -> List[int]:
    """Tokenize text using CAA's base format without importing CAA.

    Replicates: CAA's tokenize_llama_base(tokenizer, user_input, model_output)
    Format: "Input: {user_input}\nResponse: {model_output}"
    """
    input_content = ""
    input_content += f"Input: {user_input.strip()}"
    if model_output is not None:
        input_content += f"\nResponse: {model_output.strip()}"
    return tokenizer.encode(input_content)


def add_vector_from_position(matrix, vector, position_ids, from_pos=None):
    """Apply steering vector to positions >= from_pos without importing CAA.

    Replicates: CAA's add_vector_from_position(matrix, vector, position_ids, from_pos)
    """
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    mask = position_ids >= from_id
    mask = mask.unsqueeze(-1)

    matrix += mask.float() * vector
    return matrix


def find_instruction_end_position_fallback(tokens=None):
    """Find instruction end position using fallback method.

    Since CAA's find_instruction_end_postion returns -1 for our tokenization,
    we use -1 directly to trigger the "all positions" fallback behavior.

    Args:
        tokens: Unused, kept for interface compatibility

    Returns:
        -1 to trigger CAA's fallback behavior (steer ALL positions)
    """
    # CAA's function returns -1 when it can't find the pattern
    # [29871, 13, 5103, 29901] (space + newline + "Response" + colon)
    # This triggers steering ALL positions via from_pos=-1
    _ = tokens  # Unused parameter, kept for interface compatibility
    return -1


def get_a_b_probs_from_logits(logits, a_token_id, b_token_id):
    """Extract A/B probabilities from logits without importing CAA.

    Replicates: CAA's get_a_b_probs(logits, a_token_id, b_token_id)
    """
    last_token_logits = logits[0, -1, :]
    last_token_probs = torch.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob
