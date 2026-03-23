"""Prompt templates for LLM-based evaluation and generation.

This package centralizes all prompts and LLM configurations used in rubric-kit for:
- Criterion evaluation (binary, score-based, and tool-call)
- Dimension generation
- Criteria generation
- Rubric refinement

All public names are re-exported here so that existing imports like
``from rubric_kit.prompts import EVALUATOR_CONFIG`` continue to work.
"""

from .config import (
    EVALUATOR_CONFIG,
    EVALUATOR_SYSTEM_PROMPT,
    GENERATOR_CONFIG,
    GENERATOR_SYSTEM_PROMPT,
    TOOL_CALL_EVALUATOR_CONFIG,
    LLMConfig,
)
from .evaluation import (
    build_binary_criterion_prompt,
    build_score_criterion_prompt,
    build_tool_call_evaluation_prompt,
)
from .generation import (
    build_chat_criteria_generation_prompt,
    build_chat_dimension_generation_prompt,
    build_criteria_generation_prompt,
    build_dimension_generation_prompt,
)
from .refinement import (
    build_refine_rubric_prompt,
    build_refine_rubric_with_chat_prompt,
    build_refine_rubric_with_qa_prompt,
)


__all__ = [
    # Config
    "LLMConfig",
    "EVALUATOR_SYSTEM_PROMPT",
    "GENERATOR_SYSTEM_PROMPT",
    "EVALUATOR_CONFIG",
    "TOOL_CALL_EVALUATOR_CONFIG",
    "GENERATOR_CONFIG",
    # Evaluation
    "build_binary_criterion_prompt",
    "build_score_criterion_prompt",
    "build_tool_call_evaluation_prompt",
    # Generation
    "build_dimension_generation_prompt",
    "build_criteria_generation_prompt",
    "build_chat_dimension_generation_prompt",
    "build_chat_criteria_generation_prompt",
    # Refinement
    "build_refine_rubric_prompt",
    "build_refine_rubric_with_qa_prompt",
    "build_refine_rubric_with_chat_prompt",
]
