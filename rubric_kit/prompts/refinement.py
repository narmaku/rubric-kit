"""Refinement prompt builders for rubric-kit.

This module contains functions that build prompts for refining existing
evaluation rubrics, with optional Q&A or chat session context.
"""

from typing import Any

import yaml

from rubric_kit.models.schema import Criterion, Dimension

from .generation import _convert_criterion_to_dict_for_yaml


# =============================================================================
# SHARED TEMPLATE SECTIONS FOR REFINE/GENERATE PROMPTS
# =============================================================================

_WEIGHT_CONSTRAINTS = """**CRITICAL - Weight Constraints:**
- Criterion weight MUST be an integer from 0 to 3 (inclusive), OR the string "from_scores"
- 0 = informational only, 1 = low importance, 2 = medium importance, 3 = high importance
- Use "from_scores" only for score-type dimensions where criterion="from_scores"
- DO NOT use weights outside the 0-3 range (e.g., 10 is INVALID)"""

_DIMENSION_CONSTRAINTS = """**CRITICAL - Dimension Constraints:**
- If grading_type is "score", the dimension MUST have a "scores" dictionary with integer keys (0-3) and string descriptions
- If grading_type is "binary", do NOT include a scores dictionary"""

_TOOL_SCORING_MODEL = """**CRITICAL - Tool Evaluation Scoring (for tool_use dimensions with score type):**
If a tool_use dimension uses grading_type "score", use this scoring model.
The checks depend on tool_calls configuration (respect_order, params, params_strict_mode):

- 3: All applicable checks pass - tool called with correct count, correct order (if respect_order=true), correct parameters (if params specified)
- 2: Tool called with correct order and parameters, but call count outside min/max bounds
- 1: Tool called but with incorrect parameters (if params specified) OR wrong order (if respect_order=true)
- 0: Required tool not called at all

Note: If respect_order=false, order is not checked. If no params specified, params are not checked."""

_VARIABLES_GUIDANCE = """**IMPORTANT - Variables:**
- Extract specific data values (e.g. names, numbers, identifiers, IP addresses, memory amounts, OS names, percentages, etc.) to a "variables" section
- Variables should ONLY contain actual, correct values from the source data - NOT examples of incorrect values or placeholders
- Use {{variable_name}} placeholders in criterion text AND tool_calls params instead of hard-coded values
- This makes the rubric reusable with different data
- If variables already exist, preserve them and add any new ones needed"""

_NO_VARIABLES_GUIDANCE = """**IMPORTANT - No Variables Mode:**
- Do NOT create a variables section
- Use hard-coded values directly in criterion text and tool_calls params
- Write specific, concrete values directly into the criteria (e.g., "IP address is '10.0.187.159'" not "IP address is '{{ip_address}}'")
- This creates a rubric specific to this exact input"""

_ATOMIC_CRITERIA_GUIDANCE = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- Each factual accuracy criterion MUST check exactly ONE atomic value
- NEVER combine multiple values in a single criterion
- BAD: "The response reports RAM (~{{ram_total}}) and disk size ({{disk_size}})" - Mixes two values!
- GOOD: Split into separate criteria:
  1. "The response correctly reports RAM as ~{{ram_total}}"
  2. "The response correctly reports disk size as {{disk_size}}"
- This ensures clear pass/fail evaluation for each individual fact
- If an existing criterion mixes multiple values, SPLIT it into separate atomic criteria"""

_GRANULAR_TOOL_CRITERIA = """**Granular Tool Criteria with Scoring Modes:**
When refining tool usage criteria, use SEPARATE criteria with the `mode` field:
- mode: "required" - Core tools that MUST be called (Pass = weight, Fail = 0)
- mode: "bonus" - Nice-to-have tools (Pass = extra credit, Fail = 0)
- mode: "penalty" - Prohibited tools (Pass = 0, Fail = -weight)"""

_TOOL_CALLS_PRESERVE = """**IMPORTANT - Tool Calls:**
- If a criterion has a "tool_calls" specification in the current rubric, you MUST include it in the refined rubric
- Tool call specifications are critical for evaluating tool usage and must be preserved
- Only add tool_calls to criteria that evaluate tool usage (typically criteria in the "Tools" category)"""

_JSON_OUTPUT_FORMAT = """Return ONLY a JSON object with this format:
{{
  "variables": {{
    "ip_address": "10.0.187.159",
    "host": "server01"
  }},
  "dimensions": [
    {{
      "name": "dimension_name",
      "description": "Clear description",
      "grading_type": "binary"
    }}
  ],
  "criteria": [
    {{
      "name": "core_tools",
      "category": "Tools",
      "weight": 3,
      "dimension": "tool_use",
      "criterion": "Must call essential tools.",
      "tool_calls": {{
        "respect_order": false,
        "required": [{{"name": "get_system_info", "min_calls": 1, "params": {{"host": "{{host}}"}}}}]
      }}
    }},
    {{
      "name": "fact_check",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_accuracy",
      "criterion": "IP address is '{{ip_address}}'."
    }}
  ]
}}

Note: Scoring is inferred from tool lists (required/optional/prohibited). Omit tool_calls for non-tool criteria."""

_JSON_OUTPUT_FORMAT_NO_VARS = """Return ONLY a JSON object with this format (NO variables section):
{{
  "dimensions": [
    {{
      "name": "dimension_name",
      "description": "Clear description",
      "grading_type": "binary"
    }}
  ],
  "criteria": [
    {{
      "name": "core_tools",
      "category": "Tools",
      "weight": 3,
      "dimension": "tool_use",
      "criterion": "Must call essential tools.",
      "tool_calls": {{
        "respect_order": false,
        "required": [{{"name": "get_system_info", "min_calls": 1, "params": {{"host": "server01"}}}}]
      }}
    }},
    {{
      "name": "fact_check",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_accuracy",
      "criterion": "IP address is '10.0.187.159'."
    }}
  ]
}}

Note: Use hard-coded values directly in criteria - do NOT use variable placeholders."""


# =============================================================================
# HELPER FUNCTIONS FOR RUBRIC PROMPTS
# =============================================================================


def _rubric_to_yaml(
    dimensions: list[Dimension], criteria: list[Criterion], variables: dict[str, str] | None = None
) -> str:
    """Convert rubric components to YAML string for prompt inclusion."""
    rubric_dict: dict[str, Any] = {}

    if variables:
        rubric_dict["variables"] = variables

    rubric_dict["dimensions"] = [
        {
            "name": d.name,
            "description": d.description,
            "grading_type": d.grading_type,
            **({"scores": d.scores} if d.scores else {}),
        }
        for d in dimensions
    ]
    rubric_dict["criteria"] = [_convert_criterion_to_dict_for_yaml(c) for c in criteria]

    return yaml.dump(rubric_dict, sort_keys=False)


def _build_tool_calls_instruction(criteria: list[Criterion]) -> str:
    """Build tool calls preservation instruction if any criteria have tool_calls."""
    has_tool_calls = any(c.tool_calls for c in criteria)
    if not has_tool_calls:
        return ""

    return (
        "\n\n**CRITICAL - Tool Calls Specifications:**\n"
        "- If a criterion in the current rubric has a 'tool_calls' specification, you MUST preserve it in the refined rubric\n"
        "- Tool call specifications include: respect_order, required tools (with min_calls/max_calls), optional tools, and prohibited tools\n"
        "- Only modify tool_calls if explicitly improving them, otherwise preserve them exactly as shown"
    )


def _build_default_feedback(context_type: str | None = None) -> str:
    """Build default feedback section based on context type."""
    base_items = [
        "- Improving descriptions for clarity",
        "- Ensuring proper weight distribution (0-3 range)",
        "- Adding detail where criteria are too vague",
        "- Extracting specific values to variables if not already done",
    ]

    if context_type == "qa":
        return (
            "\n\nPlease improve the rubric by:\n"
            "- Making criteria more specific and measurable based on the Q&A pair\n"
            + "\n".join(base_items)
            + "\n"
            "- Ensuring criteria accurately reflect what should be evaluated in the answer"
        )
    elif context_type == "chat":
        return (
            "\n\nPlease improve the rubric by:\n"
            "- Making criteria more specific and measurable based on the chat session\n"
            + "\n".join(base_items)
            + "\n"
            "- Ensuring criteria accurately reflect tool usage, output quality, and other aspects shown in the chat"
        )
    else:
        return (
            "\n\nPlease improve the rubric by:\n"
            "- Making criteria more specific and measurable\n" + "\n".join(base_items)
        )


def _build_refine_prompt_core(
    rubric_yaml: str,
    feedback_section: str,
    tool_calls_instruction: str,
    context_header: str = "",
    analysis_instruction: str = "",
) -> str:
    """Core prompt builder for rubric refinement."""
    intro = "Refine the following evaluation rubric to improve its quality"
    if context_header:
        intro += f", using the {context_header} as context"
    intro += "."

    return f"""{intro}

{context_header and "**Current Rubric:**" or "Current Rubric:"}
{rubric_yaml}{feedback_section}{tool_calls_instruction}

{analysis_instruction}

{_WEIGHT_CONSTRAINTS}

{_DIMENSION_CONSTRAINTS}

{_TOOL_SCORING_MODEL}

{_VARIABLES_GUIDANCE}

{_ATOMIC_CRITERIA_GUIDANCE}

Return the refined rubric as JSON with the same structure. Maintain all dimension names that criteria reference.

{_TOOL_CALLS_PRESERVE}

{_GRANULAR_TOOL_CRITERIA}

{_JSON_OUTPUT_FORMAT}"""


# =============================================================================
# REFINEMENT PROMPT BUILDERS
# =============================================================================


def build_refine_rubric_prompt(
    dimensions: list[Dimension],
    criteria: list[Criterion],
    feedback: str | None = None,
    variables: dict[str, str] | None = None,
    use_variables: bool = True,
) -> str:
    """Build a prompt for refining an existing rubric.

    Args:
        dimensions: List of dimensions to include
        criteria: List of criteria to include
        feedback: Optional specific feedback for refinement
        variables: Optional variables dict to include in rubric
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
    """
    rubric_yaml = _rubric_to_yaml(dimensions, criteria, variables if use_variables else None)
    feedback_section = (
        f"\n\nSpecific Feedback:\n{feedback}" if feedback else _build_default_feedback()
    )
    tool_calls_instruction = _build_tool_calls_instruction(criteria)

    variables_guidance = _VARIABLES_GUIDANCE if use_variables else _NO_VARIABLES_GUIDANCE
    json_format = _JSON_OUTPUT_FORMAT if use_variables else _JSON_OUTPUT_FORMAT_NO_VARS

    return f"""Refine the following evaluation rubric to improve its quality.

Current Rubric:
{rubric_yaml}{feedback_section}{tool_calls_instruction}

{_WEIGHT_CONSTRAINTS}

{_DIMENSION_CONSTRAINTS}

{_TOOL_SCORING_MODEL}

{variables_guidance}

{_ATOMIC_CRITERIA_GUIDANCE}

Return the refined rubric as JSON with the same structure. Maintain all dimension names that criteria reference.

{_TOOL_CALLS_PRESERVE}

{_GRANULAR_TOOL_CRITERIA}

{json_format}"""


def build_refine_rubric_with_qa_prompt(
    dimensions: list[Dimension],
    criteria: list[Criterion],
    question: str,
    answer: str,
    feedback: str | None = None,
    context: str | None = None,
    variables: dict[str, str] | None = None,
    use_variables: bool = True,
) -> str:
    """Build a prompt for refining an existing rubric using Q&A context.

    Args:
        dimensions: List of dimensions to include
        criteria: List of criteria to include
        question: The question from the Q&A pair
        answer: The answer from the Q&A pair
        feedback: Optional specific feedback for refinement
        context: Optional additional context
        variables: Optional variables dict to include in rubric
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
    """
    rubric_yaml = _rubric_to_yaml(dimensions, criteria, variables if use_variables else None)
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    feedback_section = (
        f"\n\nSpecific Feedback:\n{feedback}" if feedback else _build_default_feedback("qa")
    )
    tool_calls_instruction = _build_tool_calls_instruction(criteria)

    variables_guidance = _VARIABLES_GUIDANCE if use_variables else _NO_VARIABLES_GUIDANCE
    json_format = _JSON_OUTPUT_FORMAT if use_variables else _JSON_OUTPUT_FORMAT_NO_VARS

    return f"""Refine the following evaluation rubric to improve its quality, using the Q&A pair as context.

**Q&A Pair:**
Question: {question}
Answer: {answer}{context_info}

**Current Rubric:**
{rubric_yaml}{feedback_section}{tool_calls_instruction}

Analyze the Q&A pair and refine the rubric to better evaluate answers like the one provided. Ensure criteria are specific and measurable based on the actual content.

{_WEIGHT_CONSTRAINTS}

{_DIMENSION_CONSTRAINTS}

{_TOOL_SCORING_MODEL}

{variables_guidance}

{_ATOMIC_CRITERIA_GUIDANCE}

Return the refined rubric as JSON with the same structure. Maintain all dimension names that criteria reference.

{_GRANULAR_TOOL_CRITERIA}

{json_format}"""


def build_refine_rubric_with_chat_prompt(
    dimensions: list[Dimension],
    criteria: list[Criterion],
    chat_content: str,
    feedback: str | None = None,
    context: str | None = None,
    variables: dict[str, str] | None = None,
    use_variables: bool = True,
) -> str:
    """Build a prompt for refining an existing rubric using chat session context.

    Args:
        dimensions: List of dimensions to include
        criteria: List of criteria to include
        chat_content: The chat session content
        feedback: Optional specific feedback for refinement
        context: Optional additional context
        variables: Optional variables dict to include in rubric
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
    """
    rubric_yaml = _rubric_to_yaml(dimensions, criteria, variables if use_variables else None)
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    feedback_section = (
        f"\n\nSpecific Feedback:\n{feedback}" if feedback else _build_default_feedback("chat")
    )
    tool_calls_instruction = _build_tool_calls_instruction(criteria)

    variables_guidance = _VARIABLES_GUIDANCE if use_variables else _NO_VARIABLES_GUIDANCE
    json_format = _JSON_OUTPUT_FORMAT if use_variables else _JSON_OUTPUT_FORMAT_NO_VARS

    return f"""Refine the following evaluation rubric to improve its quality, using the chat session as context.

**Chat Session:**
{chat_content}{context_info}

**Current Rubric:**
{rubric_yaml}{feedback_section}{tool_calls_instruction}

Analyze the chat session and refine the rubric to better evaluate similar interactions. Consider tool usage, output accuracy, completeness, and other relevant aspects shown in the chat.

{_WEIGHT_CONSTRAINTS}

{_DIMENSION_CONSTRAINTS}

{_TOOL_SCORING_MODEL}

{variables_guidance}

{_ATOMIC_CRITERIA_GUIDANCE}

Return the refined rubric as JSON with the same structure. Maintain all dimension names that criteria reference.

{_GRANULAR_TOOL_CRITERIA}

{json_format}"""
