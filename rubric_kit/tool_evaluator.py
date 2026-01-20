"""Tool call evaluator with programmatic checks and LLM-based parameter validation.

This module provides granular tool call evaluation that:
1. Programmatically checks: presence, count, order
2. Uses LLM for: semantic parameter validation
3. Generates: per-tool breakdown with scores (0-3 scale) and issues

Scoring Model (0-3 scale):
- 3: All checks pass (presence ✓, count ✓, order ✓, params ✓)
- 2: Called with correct order + params, but wrong count
- 1: Called but wrong params OR wrong order
- 0: Not called at all (for required tools)
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rubric_kit.parser import ChatSession, ToolCall
from rubric_kit.schema import Criterion, ToolSpec


class ToolType(Enum):
    """Type of tool in the specification."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    PROHIBITED = "prohibited"


@dataclass
class ToolResult:
    """Result of evaluating a single tool.

    Uses 0-3 scoring:
    - 3: Perfect (all checks pass)
    - 2: Adequate (called correctly but count issues)
    - 1: Partial (called but params/order wrong)
    - 0: Failed (not called)
    """

    name: str
    tool_type: ToolType
    called: bool
    count: int
    count_ok: bool
    order_ok: bool | None  # None if order doesn't matter for this tool
    params_ok: bool | None  # None if no param validation needed
    score: int  # 0-3 score
    max_score: int  # Maximum possible score (3 for required, varies for optional)
    issues: list[str] = field(default_factory=list)
    actual_params: dict[str, Any] | None = None
    expected_params: dict[str, Any] | None = None


@dataclass
class ToolBreakdown:
    """Complete breakdown of tool call evaluation for a criterion.

    overall_score is the weighted average of all tool scores (0-3 scale).
    """

    criterion_name: str
    mode: str  # required, bonus, penalty
    overall_pass: bool
    overall_score: float  # Weighted average score (0.0-3.0)
    order_ok: bool | None  # None if order doesn't matter
    tool_results: list[ToolResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    summary: str = ""  # LLM-generated summary


def _find_matching_calls(tool_name: str, parsed_calls: list[ToolCall]) -> list[ToolCall]:
    """Find all parsed tool calls matching a tool name."""
    matches = []
    for tc in parsed_calls:
        # Match by full name or function name
        if (
            tc.full_name == tool_name
            or tc.function == tool_name
            or tool_name.endswith(f".{tc.function}")
            or tc.full_name.endswith(f".{tool_name}")
        ):
            matches.append(tc)
    return matches


def _check_count(
    matches: list[ToolCall], spec: ToolSpec, tool_type: ToolType
) -> tuple[bool, list[str]]:
    """Check if call count is within min/max limits."""
    count = len(matches)
    issues = []

    if tool_type == ToolType.PROHIBITED:
        # For prohibited, any call is a failure
        if count > 0:
            issues.append(f"Prohibited tool '{spec.name}' was called {count} time(s)")
            return False, issues
        return True, issues

    # For optional tools, not calling them is perfectly fine (no issues)
    if tool_type == ToolType.OPTIONAL:
        if count == 0:
            return True, []  # Optional tool not called - that's OK, no issues
        # If called, check max_calls constraint
        if spec.max_calls is not None and count > spec.max_calls:
            issues.append(
                f"Tool '{spec.name}' called {count} times, maximum allowed: {spec.max_calls}"
            )
            return False, issues
        return True, issues

    # For required tools, check min_calls
    if spec.min_calls is not None and count < spec.min_calls:
        issues.append(
            f"Tool '{spec.name}' called {count} times, minimum required: {spec.min_calls}"
        )
        return False, issues

    # Check max_calls
    if spec.max_calls is not None and count > spec.max_calls:
        issues.append(f"Tool '{spec.name}' called {count} times, maximum allowed: {spec.max_calls}")
        return False, issues

    return True, issues


def _check_order(
    tool_specs: list[ToolSpec], parsed_calls: list[ToolCall]
) -> tuple[bool, list[str]]:
    """Check if required tools were called in the specified order."""
    if not tool_specs:
        return True, []

    # Get the sequence of tool calls
    call_sequence = [tc.full_name for tc in sorted(parsed_calls, key=lambda x: x.index)]

    # Find the indices of required tools in the call sequence
    required_names = [spec.name for spec in tool_specs]
    found_indices = []

    for req_name in required_names:
        for i, call_name in enumerate(call_sequence):
            # Match by full name or suffix
            if (
                call_name == req_name
                or call_name.endswith(f".{req_name}")
                or req_name.endswith(f".{call_name.split('.')[-1]}")
            ):
                found_indices.append(i)
                break
        else:
            # Tool not found - order check passes (missing tool is a different error)
            found_indices.append(-1)

    # Check if the found indices are in ascending order (ignoring -1s)
    valid_indices = [i for i in found_indices if i >= 0]
    if valid_indices != sorted(valid_indices):
        issues = [f"Tools were not called in the required order. Expected: {required_names}"]
        return False, issues

    return True, []


def _evaluate_tool_spec(
    spec: ToolSpec,
    tool_type: ToolType,
    parsed_calls: list[ToolCall],
    mode: str,
    order_ok: bool | None = None,
) -> ToolResult:
    """Evaluate a single tool specification programmatically.

    Args:
        spec: Tool specification to evaluate
        tool_type: Type of tool (required/optional/prohibited)
        parsed_calls: List of parsed tool calls from the session
        mode: Scoring mode (required/bonus/penalty)
        order_ok: Whether order check passed (None if order doesn't matter)

    Returns:
        ToolResult with 0-3 score
    """
    matches = _find_matching_calls(spec.name, parsed_calls)
    called = len(matches) > 0
    count = len(matches)

    # Check count constraints
    count_ok, count_issues = _check_count(matches, spec, tool_type)

    # Parameter validation - will be enhanced with LLM later
    params_ok = None
    actual_params = None
    expected_params = spec.params if spec.params else None

    if expected_params is not None and matches:
        # For now, just extract actual params - LLM will validate
        actual_params = matches[0].parameters if matches else None
        # Placeholder: will be set by LLM validation
        params_ok = None

    # Calculate score based on the 0-3 scoring model
    score, max_score = _calculate_tool_score_0_3(
        tool_type, mode, called, count_ok, order_ok, params_ok
    )

    issues = count_issues.copy()
    if not called and tool_type == ToolType.REQUIRED:
        issues.append(f"Required tool '{spec.name}' was not called")

    return ToolResult(
        name=spec.name,
        tool_type=tool_type,
        called=called,
        count=count,
        count_ok=count_ok,
        order_ok=order_ok,
        params_ok=params_ok,
        score=score,
        max_score=max_score,
        issues=issues,
        actual_params=actual_params,
        expected_params=expected_params,
    )


def _calculate_tool_score_0_3(
    tool_type: ToolType,
    mode: str,
    called: bool,
    count_ok: bool,
    order_ok: bool | None,
    params_ok: bool | None,
) -> tuple[int, int]:
    """Calculate 0-3 score for a single tool.

    Scoring model for REQUIRED tools:
    - 3: All checks pass (presence ✓, count ✓, order ✓, params ✓)
    - 2: Called with correct order + params, but wrong count
    - 1: Called but wrong params OR wrong order
    - 0: Not called at all

    For OPTIONAL tools:
    - Same scale but 0 if not called (no penalty)

    For PROHIBITED tools:
    - 3: Not called (good)
    - 0: Called (violation)

    Args:
        tool_type: Type of tool
        mode: Scoring mode
        called: Whether tool was called
        count_ok: Whether call count is within bounds
        order_ok: Whether order is correct (None if order doesn't matter)
        params_ok: Whether params are correct (None if not yet validated)

    Returns:
        Tuple of (score, max_score) where both are 0-3
    """
    if tool_type == ToolType.REQUIRED:
        max_score = 3
        if not called:
            return 0, max_score

        # Tool was called - determine score based on checks
        # Assume params and order are OK if not yet validated (None)
        params_passed = params_ok is not False
        order_passed = order_ok is not False

        if not params_passed or not order_passed:
            # Score 1: Called but params or order wrong
            return 1, max_score

        if not count_ok:
            # Score 2: Called correctly but count outside bounds
            return 2, max_score

        # Score 3: All checks pass
        return 3, max_score

    elif tool_type == ToolType.OPTIONAL:
        max_score = 3
        if not called:
            # Optional tool not called - no score contribution, no penalty
            return 0, 0  # max_score=0 means it won't affect weighted average

        # Tool was called - apply same scoring as required
        params_passed = params_ok is not False
        order_passed = order_ok is not False

        if not params_passed or not order_passed:
            return 1, max_score

        if not count_ok:
            return 2, max_score

        return 3, max_score

    elif tool_type == ToolType.PROHIBITED:
        # Prohibited tools use a simpler model
        max_score = 3
        if called:
            # Violation - worst score
            return 0, max_score
        # Not called - perfect score
        return 3, max_score

    return 0, 0


def _infer_mode_from_tool_calls(tool_calls) -> str:
    """Infer the scoring mode from which tool lists are populated."""
    has_required = len(tool_calls.required) > 0
    has_optional = len(tool_calls.optional) > 0
    has_prohibited = len(tool_calls.prohibited) > 0

    if has_required:
        return "required"
    elif has_optional and not has_prohibited:
        return "bonus"
    elif has_prohibited and not has_optional:
        return "penalty"
    else:
        return "required"  # default


def evaluate_tool_calls_programmatic(
    criterion: Criterion, parsed_session: ChatSession | None
) -> ToolBreakdown:
    """
    Evaluate tool calls programmatically (presence, count, order).

    Uses the 0-3 scoring model:
    - 3: All checks pass (presence ✓, count ✓, order ✓, params ✓)
    - 2: Called with correct order + params, but wrong count
    - 1: Called but wrong params OR wrong order
    - 0: Not called at all

    Multiple tools are aggregated using weighted average.

    Does NOT validate parameters - that requires LLM.

    Args:
        criterion: The criterion with tool_calls specification
        parsed_session: Parsed chat session with extracted tool calls

    Returns:
        ToolBreakdown with per-tool results and weighted average score
    """
    if not criterion.tool_calls:
        raise ValueError(f"Criterion '{criterion.name}' has no tool_calls specification")

    tool_calls = criterion.tool_calls
    mode = _infer_mode_from_tool_calls(tool_calls)
    parsed_calls = parsed_session.tool_calls if parsed_session else []

    # Check order first (affects all required tools)
    order_ok = None
    order_issues = []
    if tool_calls.respect_order and tool_calls.required:
        order_ok, order_issues = _check_order(tool_calls.required, parsed_calls)

    tool_results = []
    all_issues = []

    # Evaluate required tools (pass order_ok to each)
    for spec in tool_calls.required:
        result = _evaluate_tool_spec(spec, ToolType.REQUIRED, parsed_calls, mode, order_ok)
        tool_results.append(result)
        all_issues.extend(result.issues)

    # Evaluate optional tools (order doesn't apply to optional)
    for spec in tool_calls.optional:
        result = _evaluate_tool_spec(spec, ToolType.OPTIONAL, parsed_calls, mode, None)
        tool_results.append(result)
        all_issues.extend(result.issues)

    # Evaluate prohibited tools
    for spec in tool_calls.prohibited:
        result = _evaluate_tool_spec(spec, ToolType.PROHIBITED, parsed_calls, mode, None)
        tool_results.append(result)
        all_issues.extend(result.issues)

    # Add order issues to all_issues
    all_issues.extend(order_issues)

    # Calculate weighted average score (0-3 scale)
    overall_score = _calculate_weighted_average_score(tool_results)

    # Pass if no issues and order is OK (or not applicable)
    overall_pass = len(all_issues) == 0 and (order_ok is None or order_ok)

    return ToolBreakdown(
        criterion_name=criterion.name,
        mode=mode,
        overall_pass=overall_pass,
        overall_score=overall_score,
        order_ok=order_ok,
        tool_results=tool_results,
        issues=all_issues,
        summary="",  # Will be filled by LLM
    )


def _calculate_weighted_average_score(tool_results: list[ToolResult]) -> float:
    """Calculate weighted average score from tool results.

    Uses max_score as weight for each tool. Optional tools that weren't
    called have max_score=0 and don't affect the average.

    Args:
        tool_results: List of ToolResult objects

    Returns:
        Weighted average score on 0.0-3.0 scale
    """
    total_weighted_score = 0
    total_weight = 0

    for result in tool_results:
        if result.max_score > 0:
            total_weighted_score += result.score
            total_weight += result.max_score

    if total_weight == 0:
        return 3.0  # No tools to evaluate = perfect score

    return (total_weighted_score / total_weight) * 3.0


def build_param_validation_prompt(breakdown: ToolBreakdown) -> str | None:
    """
    Build a prompt for LLM to validate parameter semantic equivalence.

    Returns None if no parameter validation is needed.
    """
    tools_needing_validation = [
        r for r in breakdown.tool_results if r.expected_params and r.actual_params is not None
    ]

    if not tools_needing_validation:
        return None

    prompt_parts = [
        "You are validating tool call parameters to determine if they are functionally equivalent.",
        "",
        "**IMPORTANT: Focus on the underlying data value, not formatting artifacts.**",
        "",
        "A tool call is SUCCESSFUL (params_ok: true) if:",
        "- The actual raw data matches the expected data semantically",
        "- The tool would execute correctly with the provided parameters",
        "- Any formatting differences are cosmetic and don't affect functionality",
        "",
        "**Formatting artifacts to IGNORE (treat as equivalent):**",
        "- Backticks around values: `10.0.185.247` vs 10.0.185.247",
        "- Single vs double quotes: '10.0.185.247' vs \"10.0.185.247\"",
        '- Leading/trailing whitespace: " value " vs "value"',
        "- Markdown formatting: **value** or _value_ vs value",
        "- Different quote escaping styles",
        "- Extra wrapper characters that are clearly formatting artifacts",
        "",
        "**Only mark params_ok: false when:**",
        "- The actual underlying data is completely different from expected",
        "- The value is missing or null when required",
        "- The data type is fundamentally wrong (e.g., number expected, got unrelated string)",
        "",
        "**Key insight:** If the chat session shows formatting artifacts (backticks, quotes, etc.)",
        "around the parameter values but the raw data underneath is correct, the tool call was",
        "almost certainly successful. The formatting is a display/export issue, not a tool failure.",
        "",
        "Tools to validate:",
    ]

    for r in tools_needing_validation:
        prompt_parts.append(f"\n**{r.name}:**")
        prompt_parts.append(f"  Expected: {r.expected_params}")
        prompt_parts.append(f"  Actual: {r.actual_params}")

    prompt_parts.extend(
        [
            "",
            "For each tool, respond with ONLY a JSON object:",
            "```json",
            "{",
            '  "results": [',
            '    {"tool": "tool_name", "params_ok": true/false, "issue": "explanation if false"}',
            "  ]",
            "}",
            "```",
            "",
            "Remember: If the core data matches (e.g., same IP address), mark params_ok: true,",
            "even if there are formatting characters like backticks around the value.",
        ]
    )

    return "\n".join(prompt_parts)


def build_summary_prompt(breakdown: ToolBreakdown) -> str:
    """Build a prompt for LLM to generate a summary of the tool evaluation."""

    # Format tool results for the prompt
    tool_lines = []
    for r in breakdown.tool_results:
        status = "✓" if r.called else "✗"
        params = "✓" if r.params_ok else ("✗" if r.params_ok is False else "N/A")
        order = "✓" if r.order_ok else ("✗" if r.order_ok is False else "N/A")
        tool_lines.append(
            f"- {r.name} ({r.tool_type.value}): score={r.score}/3, "
            f"called={status}, params={params}, order={order}"
        )

    issues_text = (
        "\n".join(f"- {issue}" for issue in breakdown.issues) if breakdown.issues else "None"
    )

    prompt = f"""Based on this tool call evaluation breakdown, write a brief 1-2 sentence summary.

**Criterion:** {breakdown.criterion_name}
**Mode:** {breakdown.mode}
**Overall Score:** {breakdown.overall_score:.1f}/3.0 ({"PASS" if breakdown.overall_pass else "FAIL"})
**Order Check:** {"✓" if breakdown.order_ok else "✗" if breakdown.order_ok is False else "N/A"}

**Score Scale:**
- 3: Perfect (all checks pass)
- 2: Adequate (called correctly but count issues)
- 1: Partial (called but params/order wrong)
- 0: Failed (not called)

**Tools:**
{chr(10).join(tool_lines)}

**Issues:**
{issues_text}

Write a concise summary (1-2 sentences) explaining the result. Focus on what passed or failed and why.

**Summary:**"""

    return prompt


def breakdown_to_dict(breakdown: ToolBreakdown) -> dict[str, Any]:
    """Convert ToolBreakdown to a dictionary for JSON/YAML serialization."""
    return {
        "criterion_name": breakdown.criterion_name,
        "mode": breakdown.mode,
        "overall_pass": breakdown.overall_pass,
        "overall_score": breakdown.overall_score,  # 0.0-3.0 weighted average
        "order_ok": breakdown.order_ok,
        "tool_results": [
            {
                "name": r.name,
                "type": r.tool_type.value,
                "called": r.called,
                "count": r.count,
                "count_ok": r.count_ok,
                "order_ok": r.order_ok,
                "params_ok": r.params_ok,
                "score": r.score,  # 0-3 integer score
                "max_score": r.max_score,
                "issues": r.issues,
                "actual_params": r.actual_params,
                "expected_params": r.expected_params,
            }
            for r in breakdown.tool_results
        ],
        "issues": breakdown.issues,
        "summary": breakdown.summary,
    }


def apply_param_validation_results(
    breakdown: ToolBreakdown, validation_results: list[dict[str, Any]]
) -> None:
    """
    Apply LLM parameter validation results to the breakdown.

    Updates params_ok, issues, and recalculates scores using the 0-3 model:
    - If params fail, score drops to 1 (from 2 or 3)

    Args:
        breakdown: The breakdown to update (modified in place)
        validation_results: List of {"tool": str, "params_ok": bool, "issue": str}
    """
    # Build lookup by tool name
    validation_map = {r["tool"]: r for r in validation_results}

    for tool_result in breakdown.tool_results:
        if tool_result.name in validation_map:
            result = validation_map[tool_result.name]
            tool_result.params_ok = result.get("params_ok", True)

            if not tool_result.params_ok and result.get("issue"):
                issue_text = f"Tool '{tool_result.name}' parameter mismatch: {result['issue']}"
                tool_result.issues.append(issue_text)
                if issue_text not in breakdown.issues:
                    breakdown.issues.append(issue_text)

            # Recalculate score if params failed (0-3 model)
            if tool_result.params_ok is False and tool_result.called:
                # With failed params, score is 1 (tool called but with issues)
                tool_result.score = 1

    # Recalculate weighted average score
    breakdown.overall_score = _calculate_weighted_average_score(breakdown.tool_results)
    breakdown.overall_pass = len(breakdown.issues) == 0 and (
        breakdown.order_ok is None or breakdown.order_ok
    )


def parse_param_validation_response(response: str) -> list[dict[str, Any]]:
    """Parse LLM response for parameter validation."""
    # Try to extract JSON from response
    json_match = re.search(r"\{[\s\S]*\}", response)
    if not json_match:
        return []

    try:
        data = json.loads(json_match.group())
        return data.get("results", [])
    except json.JSONDecodeError:
        return []


def parse_summary_response(response: str) -> str:
    """Parse LLM response for summary."""
    # Clean up the response
    summary = response.strip()

    # Remove any markdown formatting
    summary = re.sub(r"\*\*Summary:\*\*\s*", "", summary)
    summary = re.sub(r"^Summary:\s*", "", summary, flags=re.IGNORECASE)

    return summary.strip()
