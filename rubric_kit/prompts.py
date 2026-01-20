"""Prompt templates for LLM-based evaluation and generation.

This module centralizes all prompts and LLM configurations used in rubric-kit for:
- Criterion evaluation (binary and score-based)
- Dimension generation
- Criteria generation
- Rubric refinement

All prompts and configurations are designed to be easily identifiable and modifiable.
"""

from dataclasses import dataclass
from typing import Any

import yaml

from rubric_kit.schema import Criterion, Dimension, ToolCalls, ToolSpec


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
# HELPER FUNCTIONS FOR TOOL CALL PROMPTS
# =============================================================================


def _format_tool_constraints(tool: ToolSpec) -> str:
    """Format min/max call constraints for a tool."""
    constraints = []
    if tool.min_calls is not None:
        constraints.append(f"min: {tool.min_calls}")
    if tool.max_calls is not None:
        constraints.append(f"max: {tool.max_calls}")
    return f" ({', '.join(constraints)})" if constraints else ""


def _format_tool_params(tool: ToolSpec) -> str:
    """Format parameter requirements for a tool."""
    if tool.params is None:
        # No validation - don't show params
        return ""
    if tool.params == {}:
        # Explicitly check that no params were used
        return " (must be called with NO parameters)"
    # Show specified params
    params_list = [f"{k}: {v}" for k, v in tool.params.items()]
    return f" with parameters: {', '.join(params_list)}"


def _build_required_tools_section(tool_calls: ToolCalls) -> str:
    """Build the required tools section of the prompt."""
    if not tool_calls.required:
        return ""

    lines = []
    for tool in tool_calls.required:
        constraint = _format_tool_constraints(tool)
        params_info = _format_tool_params(tool)
        lines.append(f"  - {tool.name}{constraint}{params_info}")

    return "**Required Tools:**\n" + "\n".join(lines)


def _build_optional_tools_section(tool_calls: ToolCalls) -> str:
    """Build the optional tools section of the prompt."""
    if not tool_calls.optional:
        return ""

    lines = []
    for tool in tool_calls.optional:
        max_constraint = f" (max: {tool.max_calls})" if tool.max_calls is not None else ""
        lines.append(f"  - {tool.name}{max_constraint}")

    return "\n\n**Optional Tools:**\n" + "\n".join(lines)


def _build_prohibited_tools_section(tool_calls: ToolCalls) -> str:
    """Build the prohibited tools section of the prompt."""
    if not tool_calls.prohibited:
        return ""

    lines = [f"  - {tool.name}" for tool in tool_calls.prohibited]
    return "\n\n**Prohibited Tools:**\n" + "\n".join(lines)


def _build_required_tool_lists(tool_calls: ToolCalls) -> tuple[str, str, str, str]:
    """
    Build various formats of required tool lists.

    Returns:
        Tuple of (numbered_list, labeled_list, comma_separated, bullet_list)
    """
    if not tool_calls.required:
        return "", "", "", ""

    tool_names = [tool.name for tool in tool_calls.required]
    numbered_items = [f"{i}. {name}" for i, name in enumerate(tool_names, 1)]
    labeled_items = [f"REQUIRED TOOL #{i}: {name}" for i, name in enumerate(tool_names, 1)]
    comma_separated = ", ".join(tool_names)
    bullet_list = "\n".join([f"   - {name}" for name in tool_names])

    return ("\n".join(numbered_items), "\n".join(labeled_items), comma_separated, bullet_list)


def _build_param_check_instructions(tool_calls: ToolCalls) -> str:
    """Build parameter checking instructions based on params specification.

    Logic:
    - If params is None (not declared) → no validation, return empty string
    - If params is {} (empty dict) → check that tool was called without params
    - If params has values → check only specified params (ignore extra unless strict mode)
    """
    if not tool_calls.required:
        return ""

    # Check if any tool has params validation requirements
    tools_with_empty_params = [tool for tool in tool_calls.required if tool.params == {}]
    tools_with_specified_params = [
        tool for tool in tool_calls.required if tool.params is not None and tool.params != {}
    ]

    # If no tools have params validation requirements, return empty
    if not tools_with_empty_params and not tools_with_specified_params:
        return ""

    instructions = []
    instructions.append("\n   **Check parameters** (CRITICAL)")

    # Handle tools that must be called with NO parameters
    if tools_with_empty_params:
        tool_names = [tool.name for tool in tools_with_empty_params]
        instructions.append(
            f"   - The following tools MUST be called with NO parameters: {', '.join(tool_names)}"
        )
        instructions.append("   - If any of these tools were called WITH parameters → FAIL")

    # Handle tools with specified parameters
    if tools_with_specified_params:
        instructions.append(
            "   - For each required tool that specifies parameters, verify the actual call used the EXACT parameter values"
        )
        instructions.append(
            "   - Compare expected parameters (from specification above) with actual parameters (from extracted calls)"
        )
        instructions.append("   - Parameter names must match exactly (case-sensitive)")
        instructions.append(
            '   - Parameter values must match exactly (no partial matches, no "close enough")'
        )
        instructions.append("   - Missing parameters = FAIL")
        instructions.append("   - Wrong parameter values = FAIL")

        if tool_calls.params_strict_mode:
            instructions.append(
                "   - STRICT MODE: Extra parameters are NOT allowed - exactly the specified params must match"
            )
            instructions.append("   - If ANY extra parameter is present → FAIL")
        else:
            instructions.append("   - Extra parameters are OK (only required ones must match)")

        instructions.append("   - If ANY required parameter is missing or wrong → FAIL")

    return "\n".join(instructions)


def _find_tool_call_parameters(tool_name: str, parsed_tool_calls: list[Any] | None) -> str:
    """Find and format parameters for a specific tool call."""
    if not parsed_tool_calls:
        return ""

    for tc in parsed_tool_calls:
        if _matches_tool_name(tc, tool_name) and tc.parameters:
            params_list = []
            for k, v in tc.parameters.items():
                param_value = "null" if v is None else str(v)
                params_list.append(f"{k}: {param_value}")
            if params_list:
                return f" (parameters: {', '.join(params_list)})"

    return ""


def _matches_tool_name(tool_call: Any, name: str) -> bool:
    """Check if a tool call matches a given name."""
    return (
        tool_call.full_name == name
        or tool_call.function == name
        or name.endswith(f".{tool_call.function}")
        or tool_call.full_name.endswith(f".{name}")
    )


def _build_actual_calls_section(
    tool_call_sequence: list[str] | None, parsed_tool_calls: list[Any] | None
) -> str:
    """Build the actual tool calls section from pre-parsed data."""
    if tool_call_sequence is None:
        return ""

    call_lines = []
    for i, name in enumerate(tool_call_sequence, 1):
        params_str = _find_tool_call_parameters(name, parsed_tool_calls)
        call_lines.append(f"{i}. {name}{params_str}")

    return f"""
**EXTRACTED TOOL CALLS (in order):**
{chr(10).join(call_lines)}
"""


def _build_order_evaluation_body(
    tool_calls: ToolCalls,
    required_tool_list_numbered: str,
    required_tool_list: str,
    required_tool_names_bullets: str,
    required_tool_names_list: str,
    param_check_instructions: str,
    actual_calls_section: str,
    has_preparsed_data: bool,
) -> str:
    """Build evaluation body for order-sensitive tool call evaluation."""
    if has_preparsed_data:
        return _build_order_evaluation_with_data(
            required_tool_list_numbered,
            required_tool_list,
            required_tool_names_bullets,
            required_tool_names_list,
            param_check_instructions,
            actual_calls_section,
        )

    return _build_order_evaluation_without_data(
        required_tool_list_numbered,
        required_tool_list,
        required_tool_names_bullets,
        required_tool_names_list,
        param_check_instructions,
    )


def _build_order_evaluation_with_data(
    required_tool_list_numbered: str,
    required_tool_list: str,
    required_tool_names_bullets: str,
    required_tool_names_list: str,
    param_check_instructions: str,
    actual_calls_section: str,
) -> str:
    """Build order evaluation body when pre-parsed data is available."""
    first_tool_example = (
        required_tool_names_list.split(",")[0].strip() if required_tool_names_list else ""
    )

    return f"""**Evaluation Instructions:**

Expected order:
{required_tool_list_numbered}

The specification requires these tools IN THIS EXACT ORDER:
{required_tool_list}
{actual_calls_section}

**Your task:**

1. **Compare the extracted calls against required order**
   - Position 1: Does extracted call #1 match REQUIRED TOOL #1?
   - Position 2: Does extracted call #2 match REQUIRED TOOL #2?
   - Continue for all positions
   - If ANY position doesn't match → ORDER IS WRONG → FAIL

2. **Check other requirements**
   - All required tools present? The required tools are:
{required_tool_names_bullets}
   - Call counts within limits (if specified)?
   - Optional tools within limits (if any)?
   - No prohibited tools called (if any)?{param_check_instructions}

3. **Final result**
   - Order wrong → FAIL
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL{"   - Wrong or missing parameters → FAIL" if param_check_instructions else ""}
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence. For order failures: state both the required order and actual order using the exact tool identifiers. For missing tools: you MUST state which specific tool from this list was not called: {required_tool_names_list}.{" For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual." if param_check_instructions else ""} Copy the exact tool identifier from the list above, such as "{first_tool_example}" or another tool from the list.]
"""


def _build_order_evaluation_without_data(
    required_tool_list_numbered: str,
    required_tool_list: str,
    required_tool_names_bullets: str,
    required_tool_names_list: str,
    param_check_instructions: str,
) -> str:
    """Build order evaluation body when data must be extracted from chat."""
    return f"""**Evaluation Instructions:**

Expected order:
{required_tool_list_numbered}

The specification requires these tools IN THIS EXACT ORDER:
{required_tool_list}

**Your task:**

1. **Find the tool calls in the chat session**
   - Scan through the chat session and identify all tool calls
   - Extract the tool names in the order they were called

2. **Write down the actual order you found**
   - List them: "First tool called: <actual_tool_name>, Second tool called: <actual_tool_name>, ..."
   - IMPORTANT: Use the actual tool names you found in the chat session, not placeholders

3. **Compare against the required order**
   - Position 1: Does first tool called = REQUIRED TOOL #1? (MUST match exactly)
   - Position 2: Does second tool called = REQUIRED TOOL #2? (MUST match exactly)
   - Continue for all positions
   - If ANY position doesn't match → ORDER IS WRONG → FAIL

4. **Check other requirements**
   - All required tools present? The required tools are:
{required_tool_names_bullets}
   - Call counts within limits (if specified)?
   - Optional tools within limits (if any)?
   - No prohibited tools called (if any)?{param_check_instructions}

5. **Final result**
   - Order wrong → FAIL
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL{"   - Wrong or missing parameters → FAIL" if param_check_instructions else ""}
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence. For order failures: state both the required order and actual order using the exact tool names from the specification. For missing tools: you MUST state the exact tool name that was not called from this list: {required_tool_names_list}.{" For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual." if param_check_instructions else ""} Use the exact tool name, not a placeholder or the word "name".]
"""


def _build_presence_evaluation_body(
    tool_calls: ToolCalls,
    required_tool_list: str,
    required_tool_names_bullets: str,
    required_tool_names_list: str,
    param_check_instructions: str,
    actual_calls_section: str,
    has_preparsed_data: bool,
) -> str:
    """Build evaluation body for order-insensitive tool call evaluation."""
    if has_preparsed_data:
        return _build_presence_evaluation_with_data(
            required_tool_list,
            required_tool_names_bullets,
            required_tool_names_list,
            param_check_instructions,
            actual_calls_section,
        )

    return _build_presence_evaluation_without_data(
        required_tool_list,
        required_tool_names_bullets,
        required_tool_names_list,
        param_check_instructions,
    )


def _build_presence_evaluation_with_data(
    required_tool_list: str,
    required_tool_names_bullets: str,
    required_tool_names_list: str,
    param_check_instructions: str,
    actual_calls_section: str,
) -> str:
    """Build presence evaluation body when pre-parsed data is available."""
    first_tool_example = (
        required_tool_names_list.split(",")[0].strip()
        if required_tool_names_list
        else "[tool_identifier]"
    )

    return f"""**Evaluation Instructions:**

The specification requires these tools (ORDER DOESN'T MATTER):
{required_tool_list}
{actual_calls_section}

**Your task:**

1. **Check presence**
   - The following required tools MUST be present in the extracted calls:
{required_tool_names_bullets}
   - Check if each of these exact tool identifiers appears in the extracted calls list above
   - Order doesn't matter
   - If reporting a missing tool in your REASON, copy one of these exact identifiers: {required_tool_names_list}

2. **Check counts** (if limits specified)
   - Are call counts within min/max limits?
   - Are optional tools within limits (if any)?{param_check_instructions}

3. **Check prohibitions** (if any)
   - Were any prohibited tools called?

4. **Final result**
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL
   - Called prohibited tool → FAIL{"   - Wrong or missing parameters → FAIL" if param_check_instructions else ""}
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence explaining what passed or what violation occurred. If a required tool is missing, you MUST copy one of these exact tool identifiers that was not called: {required_tool_names_list}.{" For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual." if param_check_instructions else ""} For example, if {first_tool_example} was not called, write: "Required tool {first_tool_example} was not called."]
"""


def _build_presence_evaluation_without_data(
    required_tool_list: str,
    required_tool_names_bullets: str,
    required_tool_names_list: str,
    param_check_instructions: str,
) -> str:
    """Build presence evaluation body when data must be extracted from chat."""
    first_tool_example = (
        required_tool_names_list.split(",")[0].strip()
        if required_tool_names_list
        else "[tool_identifier]"
    )

    return f"""**Evaluation Instructions:**

The specification requires these tools (ORDER DOESN'T MATTER):
{required_tool_list}

**Your task:**

1. **Find all tool calls in the chat session**
   - Scan through and identify all tool calls
   - Order doesn't matter for this evaluation

2. **Check presence**
   - The following required tools MUST be called at least once:
{required_tool_names_bullets}
   - Check if each of these exact tool identifiers appears in the chat session
   - If reporting a missing tool in your REASON, copy one of these exact identifiers: {required_tool_names_list}

3. **Check counts** (if limits specified)
   - Are call counts within min/max limits?
   - Are optional tools within limits (if any)?{param_check_instructions}

4. **Check prohibitions** (if any)
   - Were any prohibited tools called?

5. **Final result**
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL
   - Called prohibited tool → FAIL{"   - Wrong or missing parameters → FAIL" if param_check_instructions else ""}
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence explaining what passed or what violation occurred. If a required tool is missing, you MUST copy one of these exact tool identifiers that was not called: {required_tool_names_list}.{" For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual." if param_check_instructions else ""} For example, if {first_tool_example} was not called, write: "Required tool {first_tool_example} was not called."]
"""


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================


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


# =============================================================================
# GENERATION PROMPTS
# =============================================================================


def build_dimension_generation_prompt(
    question: str,
    answer: str,
    num_dimensions: int | None,
    context: str | None = None,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation dimensions from a Q&A pair.

    Args:
        question: The question being evaluated
        answer: The answer being evaluated
        num_dimensions: Number of dimensions to generate, or None for auto
        context: Optional additional context
        guidelines: Optional specific guidelines/hints to guide dimension generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    count_instruction = (
        f"Generate {num_dimensions} evaluation dimensions"
        if num_dimensions is not None
        else "Generate an appropriate number of evaluation dimensions (between 3 and 10)"
    )

    return f"""Given the following Question and Answer pair, {count_instruction} for assessing answer quality.

Question: {question}

Answer: {answer}{context_info}{guidelines_section}

Each dimension should:
1. Have a unique, descriptive name (lowercase with underscores, e.g., "factual_correctness")
2. Have a **GENERIC** description of what aspect it evaluates
3. **DO NOT** mention specific data values or fields in the dimension description
4. Specify a grading_type: either "binary" (pass/fail) or "score" (numeric scale from 0 to 3)
5. For "score" type, you MUST include a "scores" dictionary with integer keys (0-3) and description values

**CRITICAL - Score Dimensions:**
- If grading_type is "score", the "scores" field is REQUIRED - do NOT set it to null or omit it
- Scores must have keys 0, 1, 2, 3 with string descriptions for each level
- If you don't need nuanced scoring, use grading_type "binary" instead

**CRITICAL - Dimension Design:**
- Dimensions should be GENERIC and reusable (e.g., "factual_correctness" not "cpu_count_correctness")
- Do NOT create separate dimensions for each piece of data
- One "factual_correctness" dimension can be used by MANY criteria checking different facts
- The CRITERIA will specify what specific values to check

IMPORTANT: Prefer "binary" grading type unless a dimension truly requires nuanced scoring.

Common dimensions to consider:
- factual_correctness: Factual accuracy of information
- completeness: Whether all key information is provided
- relevance: How well the answer addresses the question
- clarity: How clear and understandable the answer is

Return ONLY a JSON array of dimension objects. Example format:
[
  {{
    "name": "factual_correctness",
    "description": "Evaluates whether the information provided is factually accurate and correct",
    "grading_type": "binary"
  }},
  {{
    "name": "completeness",
    "description": "Evaluates how complete and comprehensive the answer is",
    "grading_type": "score",
    "scores": {{
      "0": "No relevant information provided",
      "1": "Missing most key information",
      "2": "Partially complete, missing some key details",
      "3": "Complete with all essential information"
    }}
  }}
]"""


def build_criteria_generation_prompt(
    question: str,
    answer: str,
    dimensions: list[Dimension],
    num_criteria: int | None,
    category_hints: list[str] | None = None,
    context: str | None = None,
    use_variables: bool = True,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation criteria from Q&A and dimensions.

    Args:
        question: The question being evaluated
        answer: The answer being evaluated
        dimensions: List of dimensions to create criteria for
        num_criteria: Number of criteria to generate, or None for auto
        category_hints: Optional list of category names to guide generation
        context: Optional additional context
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
        guidelines: Optional specific guidelines/hints to guide criteria generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    # Format dimensions for prompt
    dimensions_str = "\n".join(
        [f"- {d.name} ({d.grading_type}): {d.description}" for d in dimensions]
    )

    category_guidance = (
        f"\n\nPreferred categories to use: {', '.join(category_hints)}"
        if category_hints
        else "\n\nSuggested categories: Output, Reasoning, Completeness, Accuracy, Clarity"
    )

    count_instruction = (
        f"generate {num_criteria} specific evaluation criteria"
        if num_criteria is not None
        else "generate an appropriate number of specific evaluation criteria (between 5 and 10, as many as needed to thoroughly evaluate the answer)"
    )

    if use_variables:
        variables_section = """**IMPORTANT - Variables Section:**
Extract specific data values from the answer (names, numbers, identifiers, etc.) and put them in a "variables" section. Variables should ONLY contain actual, correct values - NOT examples of incorrect values. Then use {{variable_name}} placeholders in your criterion text AND tool_calls params instead of hard-coding the values. This makes the rubric reusable with different data."""

        criteria_item_2 = """2. **Use variables for specific values** - extract specific data values to the variables section and reference them using {{variable_name}} syntax"""

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The answer correctly reports RAM (~{{ram_total}}) and disk size ({{disk_size}})" - This mixes two values!
- RIGHT: Create TWO separate criteria:
  1. "The answer correctly reports RAM as ~{{ram_total}}"
  2. "The answer correctly reports disk size as {{disk_size}}"
- Each factual accuracy criterion should verify ONE atomic value against ground truth"""

        criterion_field = """- criterion: Specific text describing what to check, using {{variable_name}} for specific values (or "from_scores" for score dimensions)"""

        json_example = """Return ONLY a JSON object with "variables" and "criteria" keys. Example format:
{
  "variables": {
    "capital_city": "Paris",
    "country_name": "France"
  },
  "criteria": [
    {
      "name": "capital_accuracy",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_correctness",
      "criterion": "The answer must correctly identify {{capital_city}} as the capital of {{country_name}}"
    },
    {
      "name": "completeness_score",
      "category": "Completeness",
      "weight": "from_scores",
      "dimension": "completeness",
      "criterion": "from_scores"
    }
  ]
}

Note: Extract ALL specific data values (names, numbers, identifiers, etc.) to the variables section."""
    else:
        variables_section = """**IMPORTANT - No Variables Mode:**
Do NOT create a variables section. Use hard-coded values directly in criterion text and tool_calls params. Write specific, concrete values directly into the criteria."""

        criteria_item_2 = """2. **Use hard-coded values** - write specific values directly into criteria (e.g., "IP address is '10.0.187.159'" not "IP address is '{{ip_address}}'")"""

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The answer correctly reports RAM (~1.7GB) and disk size (50GB)" - This mixes two values!
- RIGHT: Create TWO separate criteria:
  1. "The answer correctly reports RAM as ~1.7GB"
  2. "The answer correctly reports disk size as 50GB"
- Each factual accuracy criterion should verify ONE atomic value against ground truth"""

        criterion_field = """- criterion: Specific text describing what to check with hard-coded values (or "from_scores" for score dimensions)"""

        json_example = """Return ONLY a JSON object with "criteria" key (NO variables section). Example format:
{
  "criteria": [
    {
      "name": "capital_accuracy",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_correctness",
      "criterion": "The answer must correctly identify Paris as the capital of France"
    },
    {
      "name": "completeness_score",
      "category": "Completeness",
      "weight": "from_scores",
      "dimension": "completeness",
      "criterion": "from_scores"
    }
  ]
}

Note: Use hard-coded values directly in criteria - do NOT use variable placeholders."""

    return f"""Given the following Question, Answer, and Dimensions, {count_instruction}.

Question: {question}

Answer: {answer}{context_info}{guidelines_section}

Dimensions:
{dimensions_str}{category_guidance}

{variables_section}

Criteria should be:
1. **ATOMIC** - each criterion checks exactly ONE specific thing (one fact, one value, one requirement)
{criteria_item_2}
3. **Never mix multiple values in one factual accuracy criterion** - create separate criteria for each value to check
4. Specific and measurable
5. Distributed across the provided dimensions
6. Assigned appropriate categories (e.g., Output, Reasoning, Completeness)
7. Given weights between 0-3 based on importance (3=most important, 0=informational only)
8. For score-type dimensions, use weight="from_scores" and criterion="from_scores"

{atomic_examples}

Each criterion should have:
- name: Unique identifier (lowercase with underscores)
- category: Category name (will be auto-assigned based on the criterion type)
- weight: Integer 0-3, or "from_scores" for score-type dimensions
- dimension: Must reference one of the dimension names above
{criterion_field}

**CRITICAL - Weight Constraints:**
- Criterion weight MUST be an integer from 0 to 3 (inclusive), OR the string "from_scores"
- DO NOT use weights outside the 0-3 range (e.g., 10 is INVALID)

**CRITICAL - Dimension Reference:**
- If referencing a dimension with grading_type "score", ensure that dimension has a "scores" dictionary defined

{json_example}"""


def _convert_criterion_to_dict_for_yaml(criterion: Criterion) -> dict[str, Any]:
    """Convert a criterion to dict format for YAML display, including tool_calls if present."""
    crit_dict: dict[str, Any] = {
        "name": criterion.name,
        "category": criterion.category,
        "weight": criterion.weight,
        "dimension": criterion.dimension,
        "criterion": criterion.criterion,
    }

    if criterion.tool_calls:
        required_list = [
            {
                tc.name: {
                    "min_calls": tc.min_calls,
                    "max_calls": tc.max_calls,
                    **({"params": tc.params} if tc.params else {}),
                }
            }
            for tc in criterion.tool_calls.required
        ]
        optional_list = [
            {
                tc.name: {
                    "min_calls": tc.min_calls,
                    "max_calls": tc.max_calls,
                    **({"params": tc.params} if tc.params else {}),
                }
            }
            for tc in criterion.tool_calls.optional
        ]
        prohibited_list = [tc.name for tc in criterion.tool_calls.prohibited]

        crit_dict["tool_calls"] = {
            "respect_order": criterion.tool_calls.respect_order,
            "required": required_list,
            "optional": optional_list if optional_list else [],
            "prohibited": prohibited_list if prohibited_list else [],
        }

    return crit_dict


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


def build_chat_dimension_generation_prompt(
    chat_content: str,
    num_dimensions: int | None,
    context: str | None = None,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation dimensions from a chat session.

    Args:
        chat_content: The raw chat session content
        num_dimensions: Number of dimensions to generate, or None for auto
        context: Optional additional context
        guidelines: Optional specific guidelines/hints to guide dimension generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    count_instruction = (
        f"Generate {num_dimensions} evaluation dimensions"
        if num_dimensions is not None
        else "Generate an appropriate number of evaluation dimensions (between 5 and 10, as many as needed)"
    )

    return f"""Given the following chat session, {count_instruction} for assessing the assistant's performance.

**Chat Session:**
{chat_content}{context_info}{guidelines_section}

**Instructions:**
Analyze the chat session above to understand what happened. Consider:
- Tool usage (if tools were used): correct selection, proper ordering, completeness
- **Output accuracy**: factual correctness of information provided
- Output completeness: whether all requested information was provided
- Output quality: clarity, relevance, organization

Each dimension should:
1. Have a unique, descriptive name (lowercase with underscores, e.g., "tool_usage_correctness", "factual_accuracy")
2. Have a **GENERIC** description of what aspect it evaluates (e.g., "checks if stated facts are correct")
3. **DO NOT** mention specific tools, data values, or fields in the dimension description
4. Specify a grading_type: either "binary" (pass/fail) or "score" (numeric scale from 0 to 3)
5. For "score" type, you MUST include a "scores" dictionary with integer keys (0-3) and description values

**CRITICAL - Score Dimensions:**
- If grading_type is "score", the "scores" field is REQUIRED - do NOT set it to null or omit it
- Scores must have keys 0, 1, 2, 3 with string descriptions for each level
- If you don't need nuanced scoring, use grading_type "binary" instead

**CRITICAL - Dimension Design:**
- Dimensions should be GENERIC and reusable (e.g., "factual_accuracy" not "data_field_accuracy")
- Do NOT create separate dimensions for each category or type of data
- One "factual_accuracy" dimension can be used by MANY criteria checking different facts
- The CRITERIA will specify what specific values to check (e.g., "field X equals value Y")

IMPORTANT:
- If tools were used, include one dimension for tool usage evaluation (typically named "tool_use")
- **Prefer "binary" grading type for fact-checking dimensions**
- Typical dimensions needed: tool_use, factual_accuracy, completeness, clarity
- Use "score" type only for dimensions that genuinely need nuanced evaluation (e.g., overall clarity, completeness)

**CRITICAL - Tool Evaluation Scoring (if using score type for tool_use):**
If you use grading_type "score" for tool evaluation, use this scoring model.
The checks depend on tool_calls configuration (respect_order, params, params_strict_mode):

- 3: All applicable checks pass - tool called with correct count, correct order (if respect_order=true), correct parameters (if params specified)
- 2: Tool called with correct order and parameters, but call count outside min/max bounds
- 1: Tool called but with incorrect parameters (if params specified) OR wrong order (if respect_order=true)
- 0: Required tool not called at all

Note: If respect_order=false, order is not checked. If no params specified, params are not checked.

Return ONLY a JSON array of dimension objects. Example format:
[
  {{
    "name": "tool_use",
    "description": "Evaluates whether the assistant correctly used tools to accomplish the task",
    "grading_type": "binary"
  }},
  {{
    "name": "factual_accuracy",
    "description": "Evaluates whether stated facts and data values are correct",
    "grading_type": "binary"
  }},
  {{
    "name": "completeness",
    "description": "Evaluates whether all requested information was provided",
    "grading_type": "score",
    "scores": {{
      "0": "No relevant information provided",
      "1": "Missing most requested information",
      "2": "Some information provided but incomplete",
      "3": "All requested information comprehensively provided"
    }}
  }},
  {{
    "name": "clarity",
    "description": "Evaluates the readability and organization of the response",
    "grading_type": "score",
    "scores": {{
      "0": "Completely unintelligible or no response",
      "1": "Poorly organized or difficult to understand",
      "2": "Generally clear but could be improved",
      "3": "Exceptionally clear and well-organized"
    }}
  }}
]"""


def build_chat_criteria_generation_prompt(
    chat_content: str,
    dimensions: list[Dimension],
    num_criteria: int | None,
    category_hints: list[str] | None = None,
    context: str | None = None,
    use_variables: bool = True,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation criteria from a chat session.

    Args:
        chat_content: The raw chat session content
        dimensions: List of dimensions to create criteria for
        num_criteria: Number of criteria to generate, or None for auto
        category_hints: Optional list of category names to guide generation
        context: Optional additional context
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
        guidelines: Optional specific guidelines/hints to guide criteria generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    # Format dimensions for prompt
    dimensions_str = "\n".join(
        [f"- {d.name} ({d.grading_type}): {d.description}" for d in dimensions]
    )

    category_guidance = (
        f"\n\nPreferred categories to use: {', '.join(category_hints)}"
        if category_hints
        else "\n\nSuggested categories: Tools, Output, Reasoning, Completeness, Accuracy"
    )

    count_instruction = (
        f"generate {num_criteria} specific evaluation criteria"
        if num_criteria is not None
        else "generate an appropriate number of specific evaluation criteria (between 7 and 12, create enough to check all important aspects including tool calls and key facts)"
    )

    if use_variables:
        variables_section = """**IMPORTANT - Variables Section:**
Extract specific data values from the chat session (IP addresses, RAM amounts, percentages, identifiers, etc.) and put them in a "variables" section. Variables should ONLY contain actual, correct values - NOT examples of incorrect values. Then use {{variable_name}} placeholders in your criterion text AND tool_calls params instead of hard-coding the values."""

        criteria_item_3 = (
            """3. **Use variables** - reference values using {{variable_name}} syntax"""
        )

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The response correctly states the RAM (~{{ram_total}}) and IP address ({{ip_address}})" - Mixes two values!
- RIGHT: Create SEPARATE criteria:
  1. "The response correctly states RAM as ~{{ram_total}}"
  2. "The response correctly states IP address as {{ip_address}}"
- ONE value per factual accuracy criterion - no exceptions"""

        json_example = """Return ONLY a JSON object with "variables" and "criteria" keys. Example format:
{
  "variables": {
    "ip_address": "10.0.187.159",
    "ram_total": "1.7GB",
    "host": "server01"
  },
  "criteria": [
    {
      "name": "core_tools_called",
      "category": "Tools",
      "weight": 3,
      "dimension": "tool_use",
      "criterion": "Must call essential diagnostic tools.",
      "tool_calls": {
        "respect_order": false,
        "required": [
          {"name": "get_system_info", "min_calls": 1, "params": {"host": "{{host}}"}},
          {"name": "get_memory_info", "min_calls": 1}
        ]
      }
    },
    {
      "name": "optional_diagnostics",
      "category": "Tools",
      "weight": 1,
      "dimension": "tool_use",
      "criterion": "Extra credit for additional diagnostics.",
      "tool_calls": {
        "optional": [
          {"name": "get_network_interfaces", "min_calls": 1}
        ]
      }
    },
    {
      "name": "no_dangerous_ops",
      "category": "Tools",
      "weight": 2,
      "dimension": "tool_use",
      "criterion": "Must not call destructive operations.",
      "tool_calls": {
        "prohibited": [
          {"name": "reboot_system"}
        ]
      }
    },
    {
      "name": "ip_address_correct",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_accuracy",
      "criterion": "The response correctly states the IP address is '{{ip_address}}'."
    }
  ]
}

Note:
- Extract actual tool names from the chat session
- Scoring inferred from lists: required=pass/fail, optional=bonus, prohibited=penalty
- Extract ALL specific data values to the variables section"""
    else:
        variables_section = """**IMPORTANT - No Variables Mode:**
Do NOT create a variables section. Use hard-coded values directly in criterion text and tool_calls params. Write specific, concrete values directly into the criteria."""

        criteria_item_3 = """3. **Use hard-coded values** - write specific values directly into criteria (e.g., "IP address is '10.0.187.159'" not "IP address is '{{ip_address}}'")"""

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The response correctly states the RAM (~1.7GB) and IP address (10.0.187.159)" - Mixes two values!
- RIGHT: Create SEPARATE criteria:
  1. "The response correctly states RAM as ~1.7GB"
  2. "The response correctly states IP address as 10.0.187.159"
- ONE value per factual accuracy criterion - no exceptions"""

        json_example = """Return ONLY a JSON object with "criteria" key (NO variables section). Example format:
{
  "criteria": [
    {
      "name": "core_tools_called",
      "category": "Tools",
      "weight": 3,
      "dimension": "tool_use",
      "criterion": "Must call essential diagnostic tools.",
      "tool_calls": {
        "respect_order": false,
        "required": [
          {"name": "get_system_info", "min_calls": 1, "params": {"host": "server01"}},
          {"name": "get_memory_info", "min_calls": 1}
        ]
      }
    },
    {
      "name": "ip_address_correct",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_accuracy",
      "criterion": "The response correctly states the IP address is '10.0.187.159'."
    }
  ]
}

Note:
- Extract actual tool names from the chat session
- Use hard-coded values directly in criteria - do NOT use variable placeholders"""

    return f"""Given the following chat session and dimensions, {count_instruction}.

**Chat Session:**
{chat_content}{context_info}{guidelines_section}

**Dimensions:**
{dimensions_str}{category_guidance}

**Instructions:**
Analyze the chat session above. If you detect tool calls in the session, create criteria that evaluate them.

{variables_section}

**CRITICAL - Granular Tool Criteria:**
When evaluating tool usage, create SEPARATE criteria for different tool categories. Scoring is inferred from which lists are populated:

1. **Required tools** (use `required` list) - Core/essential tools that MUST be called
   - Pass = full weight, Fail = 0

2. **Bonus tools** (use `optional` list only) - Nice-to-have tools
   - Pass = extra credit, Fail = 0 (no penalty for not calling)

3. **Penalty tools** (use `prohibited` list only) - Tools that should NOT be called
   - No violation = 0, Violation = negative score

**Strategy for tool criteria:**
- Create ONE criterion per tool category, not one giant criterion
- Each criterion has its own weight reflecting importance
- Granular scoring: required pass/fail, optional give bonus, prohibited deduct points

Criteria should be:
1. **ATOMIC** - each criterion checks exactly ONE specific thing (one fact, one value, one tool requirement)
2. **Fact-based where possible** - create separate criteria for each distinct fact or data point
{criteria_item_3}
4. **Never mix multiple values in one factual accuracy criterion** - this is critical for reliable evaluation
5. Measurable and unambiguous
6. Distributed across the provided dimensions
7. Given weights between 0-3 based on importance (3=most important, 0=informational only)
8. For score-type dimensions, use weight="from_scores" and criterion="from_scores"

{atomic_examples}

Each criterion should have:
- name: Unique identifier (lowercase with underscores)
- category: Category name (Tools, Output, Reasoning, etc.)
- weight: Integer 0-3, or "from_scores" for score-type dimensions
- dimension: Must reference one of the dimension names above
- criterion: Specific text describing what to check
- tool_calls: (ONLY for tool usage criteria) Tool call specification

**CRITICAL - Weight Constraints:**
- Criterion weight MUST be an integer from 0 to 3 (inclusive), OR the string "from_scores"
- DO NOT use weights outside the 0-3 range (e.g., 10 is INVALID)

**CRITICAL - Dimension Reference:**
- If referencing a dimension with grading_type "score", ensure that dimension has a "scores" dictionary defined

{json_example}"""
