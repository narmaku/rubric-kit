"""Tool call formatting helpers for tool-call evaluation prompts.

This module contains all helper functions used to build the structured
prompt sections that describe tool-call specifications, extract actual
calls, and generate evaluation instructions for order-sensitive and
presence-only evaluation modes.
"""

from typing import Any

from rubric_kit.models.schema import ToolCalls, ToolSpec


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
    - If params is None (not declared) -> no validation, return empty string
    - If params is {} (empty dict) -> check that tool was called without params
    - If params has values -> check only specified params (ignore extra unless strict mode)
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
        instructions.append("   - If any of these tools were called WITH parameters -> FAIL")

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
            instructions.append("   - If ANY extra parameter is present -> FAIL")
        else:
            instructions.append("   - Extra parameters are OK (only required ones must match)")

        instructions.append("   - If ANY required parameter is missing or wrong -> FAIL")

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
   - If ANY position doesn't match -> ORDER IS WRONG -> FAIL

2. **Check other requirements**
   - All required tools present? The required tools are:
{required_tool_names_bullets}
   - Call counts within limits (if specified)?
   - Optional tools within limits (if any)?
   - No prohibited tools called (if any)?{param_check_instructions}

3. **Final result**
   - Order wrong -> FAIL
   - If any of the required tools ({required_tool_names_list}) is missing -> FAIL
   - Violated any limit -> FAIL{"   - Wrong or missing parameters -> FAIL" if param_check_instructions else ""}
   - Otherwise -> PASS

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
   - If ANY position doesn't match -> ORDER IS WRONG -> FAIL

4. **Check other requirements**
   - All required tools present? The required tools are:
{required_tool_names_bullets}
   - Call counts within limits (if specified)?
   - Optional tools within limits (if any)?
   - No prohibited tools called (if any)?{param_check_instructions}

5. **Final result**
   - Order wrong -> FAIL
   - If any of the required tools ({required_tool_names_list}) is missing -> FAIL
   - Violated any limit -> FAIL{"   - Wrong or missing parameters -> FAIL" if param_check_instructions else ""}
   - Otherwise -> PASS

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
   - If any of the required tools ({required_tool_names_list}) is missing -> FAIL
   - Violated any limit -> FAIL
   - Called prohibited tool -> FAIL{"   - Wrong or missing parameters -> FAIL" if param_check_instructions else ""}
   - Otherwise -> PASS

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
   - If any of the required tools ({required_tool_names_list}) is missing -> FAIL
   - Violated any limit -> FAIL
   - Called prohibited tool -> FAIL{"   - Wrong or missing parameters -> FAIL" if param_check_instructions else ""}
   - Otherwise -> PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence explaining what passed or what violation occurred. If a required tool is missing, you MUST copy one of these exact tool identifiers that was not called: {required_tool_names_list}.{" For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual." if param_check_instructions else ""} For example, if {first_tool_example} was not called, write: "Required tool {first_tool_example} was not called."]
"""
