"""
Token selection strategies for extracting probe-relevant activations.

When probing hidden states, we need to decide which token position(s) to read.
Different strategies give different information about the model's internal state.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch


class TokenStrategy(str, Enum):
    """Available token selection strategies."""

    LAST_PROMPT = "last_prompt"      # Last token of the prompt (before generation)
    LAST_RESPONSE = "last_response"  # Last token of the full response
    FIRST_RESPONSE = "first_response"  # First generated token
    MEAN_RESPONSE = "mean_response"  # Mean across all response tokens
    MAX_RESPONSE = "max_response"    # Token with max absolute projection
    LAST_THREE = "last_three"        # Mean of last 3 tokens (DCT paper default)


def select_tokens(
    hidden_states: torch.Tensor,
    strategy: TokenStrategy | str,
    prompt_length: int,
    direction: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select token(s) from hidden states according to strategy.

    Args:
        hidden_states: (seq_len, d_model) activations for one example
        strategy: which token selection strategy to use
        prompt_length: number of tokens in the prompt (for separating
                       prompt from response)
        direction: (d_model,) probe direction, required for MAX_RESPONSE
                   strategy to find the token with maximum projection

    Returns:
        (d_model,) tensor — the selected activation vector
    """
    strategy = TokenStrategy(strategy)
    seq_len = hidden_states.shape[0]

    # Ensure we have response tokens
    response_start = prompt_length
    if response_start >= seq_len:
        # Fallback: if no response tokens, use last token
        return hidden_states[-1]

    if strategy == TokenStrategy.LAST_PROMPT:
        return hidden_states[prompt_length - 1]

    elif strategy == TokenStrategy.LAST_RESPONSE:
        return hidden_states[-1]

    elif strategy == TokenStrategy.FIRST_RESPONSE:
        return hidden_states[response_start]

    elif strategy == TokenStrategy.MEAN_RESPONSE:
        return hidden_states[response_start:].mean(dim=0)

    elif strategy == TokenStrategy.MAX_RESPONSE:
        if direction is None:
            raise ValueError("MAX_RESPONSE strategy requires a direction vector")
        response_states = hidden_states[response_start:]
        direction = direction.to(response_states.device, response_states.dtype)
        projections = response_states @ direction
        max_idx = projections.abs().argmax()
        return response_states[max_idx]

    elif strategy == TokenStrategy.LAST_THREE:
        return hidden_states[-3:].mean(dim=0)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
