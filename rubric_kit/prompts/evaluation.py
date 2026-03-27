"""Evaluation prompt builders for rubric-kit.

This module contains functions that build prompts for criterion evaluation:
- Binary (pass/fail) evaluation
- Score-based evaluation
- Tool call evaluation
"""

from typing import Any

from rubric_kit.models.schema import Criterion, Dimension

from .tool_calls import (
    _build_actual_calls_section,
    _build_optional_tools_section,
    _build_order_evaluation_body,
    _build_param_check_instructions,
    _build_presence_evaluation_body,
    _build_prohibited_tools_section,
    _build_required_tool_lists,
    _build_required_tools_section,
)


def build_binary_criterion_prompt(criterion: Criterion, chat_content: str) -> str:
    """
    Build a prompt for binary (pass/fail) criterion evaluation.

    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content to evaluate

    Returns:
        Formatted prompt string for the LLM
    """
    return f"""You are an expert evaluator. Your task is to evaluate whether a chat session meets a specific criterion.

**Criterion Details:**
- Dimension: {criterion.dimension}
- Category: {criterion.category}
- Criterion: {criterion.criterion}

**Chat Session:**
{chat_content}

**Instructions:**

Carefully read the criterion above and determine what it requires. Then evaluate the chat session:

**Step 1 - Understand the requirement:**
- Does the criterion check for CORRECTNESS? (words like "correctly", "accurately", "true", or specifies exact values to match)
- Or does it check for PRESENCE? (words like "includes", "mentions", "contains")

**Step 2A - If checking CORRECTNESS:**
1. Find the authoritative source in the chat (tool outputs, function results, provided data)
2. Locate the specific data point mentioned in the criterion within that source
3. Extract the exact value from the source (this is ground truth)
4. Find what the assistant claimed about this in their final response
5. Compare: Does the assistant's claim match the source exactly?
   - Even small discrepancies = FAIL
   - Wrong numbers, wrong labels, wrong units = FAIL
   - Topic mentioned but value wrong = FAIL
   - Only PASS if values match exactly

**Step 2B - If checking PRESENCE:**
1. Look for the required information in the chat session
2. The information must be EXPLICITLY stated, not implied
3. Do NOT make inferences - only PASS if the information is directly stated
4. Do NOT consider related but different information - only exact matches count
5. PASS if present, FAIL if missing or incomplete

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence. For correctness: state source value and assistant's claim. For presence: quote relevant text or state what's missing.]

**Examples:**

RESULT: PASS
REASON: Source data shows "X=10" and assistant correctly stated "X is 10".

RESULT: FAIL
REASON: Source shows "value A" but assistant incorrectly claimed "value B".

RESULT: PASS
REASON: Response explicitly includes the required information about topic Z.

RESULT: FAIL
REASON: Required information about topic Y is not mentioned in the response.

**Your Response:**"""


def build_tool_call_evaluation_prompt(
    criterion: Criterion,
    chat_content: str,
    tool_call_sequence: list[str] | None = None,
    parsed_tool_calls: list[Any] | None = None,
) -> str:
    """
    Build a prompt for tool call evaluation.

    Tool call evaluation compares extracted tool calls against specifications.
    If tool_call_sequence is provided (pre-parsed), evaluation is deterministic.
    Otherwise, the judge must extract tool calls from raw chat content.

    Args:
        criterion: The criterion with tool_calls specification
        chat_content: The chat session content to evaluate
        tool_call_sequence: Optional pre-parsed list of tool names in order
        parsed_tool_calls: Optional pre-parsed list of ToolCall objects with parameters

    Returns:
        Formatted prompt string for the LLM

    Raises:
        ValueError: If criterion doesn't have tool_calls defined
    """
    if not criterion.tool_calls:
        raise ValueError(
            f"Criterion '{criterion.name}' must have tool_calls defined for tool call evaluation"
        )

    tool_calls = criterion.tool_calls
    has_preparsed_data = tool_call_sequence is not None

    # Build tool specification sections
    required_section = _build_required_tools_section(tool_calls)
    optional_section = _build_optional_tools_section(tool_calls)
    prohibited_section = _build_prohibited_tools_section(tool_calls)

    # Build required tool lists in various formats
    (
        required_tool_list_numbered,
        required_tool_list,
        required_tool_names_list,
        required_tool_names_bullets,
    ) = _build_required_tool_lists(tool_calls)

    # Build parameter checking instructions
    param_check_instructions = _build_param_check_instructions(tool_calls)

    # Build actual calls section if pre-parsed data available
    actual_calls_section = _build_actual_calls_section(tool_call_sequence, parsed_tool_calls)

    # Build evaluation body based on order sensitivity and data availability
    if tool_calls.respect_order:
        evaluation_body = _build_order_evaluation_body(
            tool_calls,
            required_tool_list_numbered,
            required_tool_list,
            required_tool_names_bullets,
            required_tool_names_list,
            param_check_instructions,
            actual_calls_section,
            has_preparsed_data,
        )
    else:
        evaluation_body = _build_presence_evaluation_body(
            tool_calls,
            required_tool_list,
            required_tool_names_bullets,
            required_tool_names_list,
            param_check_instructions,
            actual_calls_section,
            has_preparsed_data,
        )

    return f"""You are an expert at evaluating tool usage in chat sessions.

**Tool Usage Specification:**
{required_section}{optional_section}{prohibited_section}

**Chat Session:**
{chat_content}

{evaluation_body}

**Your Response:**"""


def build_score_criterion_prompt(
    criterion: Criterion, chat_content: str, dimension: Dimension
) -> str:
    """
    Build a prompt for score-based criterion evaluation.

    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content to evaluate
        dimension: The dimension with score scale definitions

    Returns:
        Formatted prompt string for the LLM

    Raises:
        ValueError: If dimension doesn't have scores defined
    """
    if not dimension.scores:
        raise ValueError(
            f"Dimension '{dimension.name}' does not have scores defined. "
            "Score-based evaluation requires a dimension with scores."
        )

    score_descriptions = "\n".join(
        [f"{score}: {desc}" for score, desc in sorted(dimension.scores.items())]
    )

    return f"""You are an expert evaluator. Your task is to score a chat session based on a specific criterion.

**Criterion Details:**
- Dimension: {criterion.dimension}
- Category: {criterion.category}
- Description: {dimension.description}
- Criterion: {criterion.criterion}

**Scoring Scale:**
{score_descriptions}

**Chat Session:**
{chat_content}

**Instructions:**
Read the scoring scale carefully. Evaluate the chat session and assign the most appropriate score.
Your response MUST be in this exact format (2 lines only):
SCORE: [numeric score from {min(dimension.scores.keys())} to {max(dimension.scores.keys())}]
REASON: [One sentence explaining why this score fits. Keep it brief and specific.]

Example response:
SCORE: 3
REASON: Response includes all essential information with no gaps.

**Your Response:**"""
