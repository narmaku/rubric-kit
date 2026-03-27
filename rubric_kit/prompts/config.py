"""LLM configuration and system prompts for rubric-kit.

This module contains the LLMConfig dataclass and pre-configured instances
for different LLM personas (evaluator, generator, tool-call evaluator).
"""

from dataclasses import dataclass


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

EVALUATOR_SYSTEM_PROMPT = (
    "You are a precise evaluator. Follow instructions exactly. Be concise and accurate."
)

GENERATOR_SYSTEM_PROMPT = (
    "You are an expert at creating evaluation rubrics. "
    "You always respond with valid JSON only, no additional text."
)


# =============================================================================
# LLM CONFIGURATIONS
# =============================================================================


@dataclass
class LLMConfig:
    """
    Configuration for LLM API calls.

    Bundles together all parameters needed for a specific LLM "persona":
    - System prompt defining the role
    - Temperature controlling randomness/creativity
    - Max tokens limiting response length

    This makes it easy to maintain different configurations for different
    use cases (e.g., deterministic evaluation vs creative generation).

    Attributes:
        system_prompt: The system message defining the LLM's role
        temperature: Controls randomness (0.0=deterministic, 1.0=creative)
        max_tokens: Maximum number of tokens in the response
    """

    system_prompt: str
    temperature: float
    max_tokens: int


# Named configurations for different LLM personas
EVALUATOR_CONFIG = LLMConfig(
    system_prompt=EVALUATOR_SYSTEM_PROMPT,
    temperature=0.0,  # Deterministic for consistent evaluation
    max_tokens=8192,  # Sufficient for detailed evaluations
)

TOOL_CALL_EVALUATOR_CONFIG = LLMConfig(
    system_prompt=EVALUATOR_SYSTEM_PROMPT,
    temperature=0.0,  # Deterministic for consistent evaluation
    max_tokens=16384,  # More tokens needed for structural comparison and reasoning
)

GENERATOR_CONFIG = LLMConfig(
    system_prompt=GENERATOR_SYSTEM_PROMPT,
    temperature=0.7,  # More creative for generation tasks
    max_tokens=16384,  # Longer responses for generating rubrics (increased for complex rubrics)
)
