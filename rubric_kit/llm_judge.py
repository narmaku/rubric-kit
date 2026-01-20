"""LLM-based judge panel for automatic criterion evaluation."""

import json
import random
import re
import time
from typing import TYPE_CHECKING, Any, Optional

import litellm

from rubric_kit.schema import Criterion, Dimension, JudgeConfig, JudgePanelConfig, Rubric


if TYPE_CHECKING:
    from rubric_kit.metrics import MetricsAggregator
from rubric_kit.consensus import apply_binary_consensus, apply_score_consensus
from rubric_kit.execution import execute_judges
from rubric_kit.generator import parse_qa_input
from rubric_kit.parser import ChatSession, parse_chat_session
from rubric_kit.prompts import (
    EVALUATOR_CONFIG,
    TOOL_CALL_EVALUATOR_CONFIG,
    build_binary_criterion_prompt,
    build_score_criterion_prompt,
    build_tool_call_evaluation_prompt,
)
from rubric_kit.tool_evaluator import (
    ToolBreakdown,
    apply_param_validation_results,
    breakdown_to_dict,
    build_param_validation_prompt,
    build_summary_prompt,
    evaluate_tool_calls_programmatic,
    parse_param_validation_response,
    parse_summary_response,
)


def read_chat_session(file_path: str) -> str:
    """
    Read chat session from a plain text file.

    Args:
        file_path: Path to the chat session file

    Returns:
        Content of the chat session
    """
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def _extract_reason(response: str) -> str:
    """Extract reason from response if present."""
    if "REASON:" not in response.upper():
        return ""

    reason_parts = response.split("REASON:")
    if len(reason_parts) <= 1:
        return ""

    return reason_parts[1].strip()


def parse_binary_response(response: str) -> dict[str, Any]:
    """
    Parse LLM response for binary criterion.

    Args:
        response: LLM response text

    Returns:
        Dictionary with 'passes' (bool) and 'reason' (str)
    """
    response_upper = response.upper().strip()

    # Extract result - check RESULT: first, then fallback to PASS/FAIL keywords
    passes = False
    if "RESULT:" in response_upper:
        result_line = response_upper.split("RESULT:")[1].split("\n")[0].strip()
        passes = "PASS" in result_line
    elif "PASS" in response_upper:
        passes = True

    return {"passes": passes, "reason": _extract_reason(response)}


def _extract_score(response: str) -> int:
    """Extract score from response."""
    response_upper = response.upper()

    # Try SCORE: field first
    if "SCORE:" in response_upper:
        score_line = response_upper.split("SCORE:")[1].split("\n")[0].strip()
        match = re.search(r"\d+", score_line)
        if match:
            return int(match.group())

    # Fallback: extract first number from response
    match = re.search(r"\d+", response.strip())
    if match:
        return int(match.group())

    raise ValueError(f"Could not parse score from response: {response}")


def parse_score_response(response: str) -> dict[str, Any]:
    """
    Parse LLM response for score criterion.

    Args:
        response: LLM response text

    Returns:
        Dictionary with 'score' (int) and 'reason' (str)
    """
    return {"score": _extract_score(response), "reason": _extract_reason(response)}


def _get_api_key(judge_config: JudgeConfig) -> str | None:
    """Get API key from judge config.

    Returns the API key if explicitly provided in judge config.
    If not provided, returns None and LiteLLM will auto-detect
    from environment variables based on the provider/model.

    For explicit control, users can set the appropriate env var:
    - OPENAI_API_KEY for OpenAI models
    - GEMINI_API_KEY for Google Gemini/Vertex AI
    - WATSONX_APIKEY for IBM WatsonX
    - ANTHROPIC_API_KEY for Anthropic Claude
    - etc. (see LiteLLM docs for full list)
    """
    return judge_config.api_key


def _prepare_tool_call_evaluation(
    criterion: Criterion, chat_content: str, parsed_session: ChatSession | None
) -> tuple[str, str]:
    """Prepare prompt and config for tool call evaluation."""
    tool_sequence = None
    parsed_tool_calls = None

    if parsed_session:
        tool_sequence = parsed_session.get_tool_call_sequence()
        parsed_tool_calls = sorted(parsed_session.tool_calls, key=lambda tc: tc.index)

    prompt = build_tool_call_evaluation_prompt(
        criterion, chat_content, tool_sequence, parsed_tool_calls
    )
    return prompt, "tool_call"


def _evaluate_tool_calls_hybrid(
    criterion: Criterion,
    parsed_session: ChatSession | None,
    judge_config: JudgeConfig,
    metrics: Optional["MetricsAggregator"] = None,
) -> tuple[ToolBreakdown, bool, float]:
    """
    Evaluate tool calls using hybrid approach: programmatic + LLM.

    Uses 0-3 scoring model:
    - 3: All checks pass (presence ✓, count ✓, order ✓, params ✓)
    - 2: Called with correct order + params, but wrong count
    - 1: Called but wrong params OR wrong order
    - 0: Not called at all

    Steps:
    1. Programmatic: presence, count, order checks
    2. LLM: parameter semantic validation
    3. LLM: summary generation

    Returns:
        Tuple of (ToolBreakdown, overall_pass, overall_score)
        - overall_score is 0.0-3.0 weighted average
    """
    # Step 1: Programmatic evaluation
    breakdown = evaluate_tool_calls_programmatic(criterion, parsed_session)

    # Step 2: LLM parameter validation (if needed)
    param_prompt = build_param_validation_prompt(breakdown)
    if param_prompt:
        try:
            response = _call_llm(
                judge_config,
                param_prompt,
                EVALUATOR_CONFIG,
                metrics=metrics,
                call_type="param_validation",
                context_id=criterion.name,
            )
            validation_results = parse_param_validation_response(response)
            apply_param_validation_results(breakdown, validation_results)
        except Exception as e:
            print(f"Warning: Parameter validation failed: {e}")
            # Continue without param validation - programmatic results stand

    # Step 3: LLM summary generation
    summary_prompt = build_summary_prompt(breakdown)
    try:
        response = _call_llm(
            judge_config,
            summary_prompt,
            EVALUATOR_CONFIG,
            metrics=metrics,
            call_type="summary_generation",
            context_id=criterion.name,
        )
        breakdown.summary = parse_summary_response(response)
    except Exception as e:
        print(f"Warning: Summary generation failed: {e}")
        # Generate a basic summary from the data
        status = "passed" if breakdown.overall_pass else "failed"
        breakdown.summary = f"Tool evaluation {status} with score {breakdown.overall_score:.1f}/3."

    return breakdown, breakdown.overall_pass, breakdown.overall_score


def _evaluate_tool_criterion_with_breakdown(
    judge_config: JudgeConfig,
    criterion: Criterion,
    chat_content: str,
    dimension: Dimension | None,
    parsed_session: ChatSession | None,
    metrics: Optional["MetricsAggregator"] = None,
) -> dict[str, Any]:
    """
    Evaluate a tool call criterion using hybrid approach.

    Returns evaluation result with:
    - passes: boolean pass/fail
    - score: 0-3 integer score (rounded from weighted average)
    - tool_breakdown: detailed per-tool results
    """
    breakdown, passes, overall_score = _evaluate_tool_calls_hybrid(
        criterion, parsed_session, judge_config, metrics=metrics
    )

    return {
        "passes": passes,
        "score": round(overall_score),  # 0-3 integer score
        "reason": breakdown.summary,
        "tool_breakdown": breakdown_to_dict(breakdown),
    }


def _prepare_evaluation_prompt(
    criterion: Criterion,
    chat_content: str,
    dimension: Dimension | None,
    parsed_session: ChatSession | None,
) -> tuple[str, str, Any]:
    """Prepare prompt and evaluation type based on criterion type."""
    if criterion.tool_calls:
        prompt, evaluation_type = _prepare_tool_call_evaluation(
            criterion, chat_content, parsed_session
        )
        return prompt, evaluation_type, TOOL_CALL_EVALUATOR_CONFIG

    grading_type = dimension.grading_type if dimension else "binary"

    if grading_type == "binary":
        prompt = build_binary_criterion_prompt(criterion, chat_content)
        return prompt, "binary", EVALUATOR_CONFIG

    if not dimension:
        raise ValueError("Dimension required for score-based criterion")

    prompt = build_score_criterion_prompt(criterion, chat_content, dimension)
    return prompt, "score", EVALUATOR_CONFIG


def _call_llm(
    judge_config: JudgeConfig,
    prompt: str,
    config: Any,
    metrics: Optional["MetricsAggregator"] = None,
    call_type: str = "evaluate_criterion",
    context_id: str | None = None,
) -> str:
    """Call LLM API via LiteLLM and return response content.

    Uses judge-specific LLM parameters if provided, otherwise falls back to
    defaults from the config object (from prompts.py).

    LiteLLM automatically routes to the correct provider based on model name:
    - "gpt-4" -> OpenAI
    - "vertex_ai/gemini-2.5-flash" -> Google Vertex AI
    - "watsonx/meta-llama/llama-3-8b-instruct" -> IBM WatsonX
    - "ollama/llama3" -> Local Ollama

    Args:
        judge_config: Configuration for the judge
        prompt: Prompt to send to LLM
        config: LLM config object (from prompts.py)
        metrics: Optional MetricsAggregator for tracking
        call_type: Type of call for metrics (e.g., 'evaluate_criterion')
        context_id: Optional context identifier (e.g., criterion name)
    """
    # Build API call parameters, using judge-specific values if provided,
    # otherwise falling back to config defaults
    api_params = {
        "model": judge_config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": judge_config.temperature
        if judge_config.temperature is not None
        else config.temperature,
        "max_tokens": judge_config.max_tokens
        if judge_config.max_tokens is not None
        else config.max_tokens,
    }

    # Add optional parameters if provided in judge config
    if judge_config.top_p is not None:
        api_params["top_p"] = judge_config.top_p
    if judge_config.frequency_penalty is not None:
        api_params["frequency_penalty"] = judge_config.frequency_penalty
    if judge_config.presence_penalty is not None:
        api_params["presence_penalty"] = judge_config.presence_penalty

    # Add explicit API key if provided in judge config
    api_key = _get_api_key(judge_config)
    if api_key:
        api_params["api_key"] = api_key

    # Add custom base URL if provided (for OpenAI-compatible endpoints)
    if judge_config.base_url:
        api_params["api_base"] = judge_config.base_url

    # Track timing for metrics
    start_time = time.time()
    try:
        response = litellm.completion(**api_params)
    except Exception as e:
        raise ValueError(
            f"Judge evaluation failed: API call error for {judge_config.model}: {e!s}"
        ) from e
    latency = time.time() - start_time

    # Record metrics if aggregator is configured
    if metrics is not None:
        metrics.record_call(
            call_type=call_type,
            model=judge_config.model,
            usage=response.usage,
            latency=latency,
            context_id=context_id,
            response=response,
        )

    content = response.choices[0].message.content
    if content is None:
        _raise_empty_content_error(judge_config.model, response)

    return content.strip()


def _raise_empty_content_error(model: str, response: Any) -> None:
    """Raise error for empty content response with debug info."""
    try:
        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else response.dict()
        )
        response_json = json.dumps(response_dict, indent=2, default=str)
    except Exception:
        response_json = str(response)

    raise ValueError(
        f"Judge evaluation failed: {model} returned empty content. "
        f"This may be due to API compatibility issues, content filters, or refusal responses.\n"
        f"Response structure:\n{response_json}"
    )


def _parse_evaluation_response(llm_response: str, evaluation_type: str) -> dict[str, Any]:
    """Parse LLM response based on evaluation type."""
    if evaluation_type in ("binary", "tool_call"):
        parsed = parse_binary_response(llm_response)
        return {"passes": parsed["passes"], "reason": parsed["reason"]}

    parsed = parse_score_response(llm_response)
    return {"score": parsed["score"], "reason": parsed["reason"]}


def _single_judge_evaluate(
    judge_config: JudgeConfig,
    criterion: Criterion,
    chat_content: str,
    dimension: Dimension | None,
    parsed_session: ChatSession | None = None,
    metrics: Optional["MetricsAggregator"] = None,
) -> dict[str, Any]:
    """
    Evaluate a criterion using a single judge.

    This is the function passed to execute_judges for each judge.
    For tool call criteria, uses hybrid approach (programmatic + LLM).

    Args:
        judge_config: Configuration for this judge
        criterion: The criterion to evaluate
        chat_content: The chat session content
        dimension: Optional dimension for score-based criteria
        parsed_session: Optional pre-parsed chat session for structured evaluation
        metrics: Optional MetricsAggregator for tracking LLM calls

    Returns:
        Evaluation result dictionary
    """
    # Use hybrid approach for tool call criteria
    if criterion.tool_calls:
        return _evaluate_tool_criterion_with_breakdown(
            judge_config, criterion, chat_content, dimension, parsed_session, metrics=metrics
        )

    # Standard LLM-only evaluation for non-tool criteria
    prompt, evaluation_type, config = _prepare_evaluation_prompt(
        criterion, chat_content, dimension, parsed_session
    )
    llm_response = _call_llm(
        judge_config,
        prompt,
        config,
        metrics=metrics,
        call_type="evaluate_criterion",
        context_id=criterion.name,
    )
    return _parse_evaluation_response(llm_response, evaluation_type)


def _check_judge_errors(judge_results: list[dict[str, Any]]) -> None:
    """Check for errors in judge results and raise if any found."""
    errors = [r for r in judge_results if "error" in r]
    if not errors:
        return

    error_msgs = [f"{r['judge']}: {r['error']}" for r in errors]
    raise Exception(f"Judge evaluation failed: {'; '.join(error_msgs)}")


def _is_binary_evaluation(dimension: Dimension | None, criterion: Criterion) -> bool:
    """Determine if evaluation is binary type."""
    if criterion.tool_calls:
        return True

    grading_type = dimension.grading_type if dimension else "binary"
    return grading_type == "binary"


def _extract_tool_breakdown(
    judge_votes: list[dict[str, Any]], consensus_result: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """
    Extract tool_breakdown from judge votes, preferring agreeing judges.

    When a consensus result is provided, prioritizes breakdowns from judges
    that agreed with the final result. This ensures the breakdown shown
    matches the evaluation outcome.

    Args:
        judge_votes: List of judge vote dictionaries
        consensus_result: Optional consensus result with 'passes' or 'score' key

    Returns:
        Tool breakdown dictionary from an agreeing judge, or any available
        breakdown as fallback, or None if no breakdown exists.
    """
    # First, try to find breakdown from a judge that agrees with the consensus
    if consensus_result is not None:
        agreeing_votes = _get_agreeing_judges(judge_votes, consensus_result)
        for vote in agreeing_votes:
            if "tool_breakdown" in vote:
                return vote["tool_breakdown"]

    # Fallback: return first available breakdown (original behavior)
    for vote in judge_votes:
        if "tool_breakdown" in vote:
            return vote["tool_breakdown"]

    return None


def _build_binary_result(consensus_result: dict[str, Any]) -> dict[str, Any]:
    """Build binary evaluation result from consensus."""
    result = {
        "type": "binary",
        "passes": consensus_result["passes"],
        "consensus_reached": consensus_result["consensus_reached"],
        "consensus_count": consensus_result["consensus_count"],
        "judge_votes": consensus_result["judge_votes"],
        "reason": _build_consensus_reason(consensus_result),
    }

    # Preserve tool_breakdown if present (from hybrid tool evaluation)
    # Pass consensus_result to prefer breakdown from agreeing judges
    tool_breakdown = _extract_tool_breakdown(consensus_result["judge_votes"], consensus_result)
    if tool_breakdown:
        result["tool_breakdown"] = tool_breakdown
        # Include score from tool evaluation (0-3 scale) at top level for easy access
        if "overall_score" in tool_breakdown:
            result["score"] = round(tool_breakdown["overall_score"])

    return result


def _build_score_result(consensus_result: dict[str, Any]) -> dict[str, Any]:
    """Build score evaluation result from consensus."""
    return {
        "type": "score",
        "score": consensus_result["score"],
        "consensus_reached": consensus_result["consensus_reached"],
        "consensus_count": consensus_result["consensus_count"],
        "judge_votes": consensus_result["judge_votes"],
        "reason": _build_consensus_reason(consensus_result),
    }


def evaluate_criterion_with_panel(
    criterion: Criterion,
    chat_content: str,
    dimension: Dimension | None,
    panel_config: JudgePanelConfig,
    parsed_session: ChatSession | None = None,
    metrics: Optional["MetricsAggregator"] = None,
) -> dict[str, Any]:
    """
    Evaluate a single criterion using a judge panel.

    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content
        dimension: Dimension for score-based criteria (optional)
        panel_config: Judge panel configuration
        parsed_session: Optional pre-parsed chat session for structured evaluation
        metrics: Optional MetricsAggregator for tracking LLM calls

    Returns:
        Evaluation result dictionary with consensus information:
        For binary: {"type": "binary", "passes": bool, "consensus_reached": bool,
                    "consensus_count": int, "judge_votes": List[Dict], "reason": str}
        For score: {"type": "score", "score": int, "consensus_reached": bool,
                   "consensus_count": int, "judge_votes": List[Dict], "reason": str}
    """
    judge_results = execute_judges(
        judges=panel_config.judges,
        judge_function=_single_judge_evaluate,
        execution_mode=panel_config.execution.mode,
        criterion=criterion,
        chat_content=chat_content,
        dimension=dimension,
        parsed_session=parsed_session,
        batch_size=panel_config.execution.batch_size,
        timeout=panel_config.execution.timeout,
        metrics=metrics,
    )

    _check_judge_errors(judge_results)

    is_binary = _is_binary_evaluation(dimension, criterion)

    if is_binary:
        consensus_result = apply_binary_consensus(
            votes=judge_results,
            threshold=panel_config.consensus.threshold,
            on_no_consensus=panel_config.consensus.on_no_consensus,
        )
        return _build_binary_result(consensus_result)

    consensus_result = apply_score_consensus(
        votes=judge_results,
        threshold=panel_config.consensus.threshold,
        on_no_consensus=panel_config.consensus.on_no_consensus,
    )
    return _build_score_result(consensus_result)


def _get_agreeing_judges(
    judge_votes: list[dict[str, Any]], consensus_result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Get list of judges that agreed with the final consensus result."""
    if "passes" in consensus_result:
        final_result = consensus_result["passes"]
        return [v for v in judge_votes if v.get("passes") == final_result]

    final_score = consensus_result["score"]
    return [v for v in judge_votes if v.get("score") == final_score]


def _select_judge_reason(agreeing_judges: list[dict[str, Any]]) -> str:
    """Select and format reason from agreeing judges."""
    judges_with_reasons = [v for v in agreeing_judges if v.get("reason")]
    if not judges_with_reasons:
        return ""

    selected = random.choice(judges_with_reasons)
    reason = selected.get("reason", "")
    judge_name = selected.get("judge", "unknown")
    return f"{reason} (from {judge_name})"


def _build_consensus_reason(consensus_result: dict[str, Any]) -> str:
    """
    Build a human-readable reason from consensus result.

    Extracts one reason from judges that agreed on the final result.
    If there's agreement, randomly selects one agreeing judge's reason
    and labels it with the judge name.

    Args:
        consensus_result: Result from apply_binary_consensus or apply_score_consensus

    Returns:
        Reason string from one agreeing judge, labeled with judge name
    """
    judge_votes = consensus_result["judge_votes"]

    if len(judge_votes) == 1:
        return judge_votes[0].get("reason", "")

    agreeing_judges = _get_agreeing_judges(judge_votes, consensus_result)
    if agreeing_judges:
        reason = _select_judge_reason(agreeing_judges)
        if reason:
            return reason

    return judge_votes[0].get("reason", "") if judge_votes else ""


def _parse_chat_session_safe(chat_content: str, use_parser: bool) -> ChatSession | None:
    """Parse chat session safely, returning None on failure."""
    if not use_parser:
        return None

    try:
        parsed_session = parse_chat_session(chat_content)
        print(f"Parsed chat session: found {len(parsed_session.tool_calls)} tool calls")
        return parsed_session
    except Exception as e:
        print(f"Warning: Failed to parse chat session: {e}")
        print("Falling back to raw content evaluation")
        return None


def _evaluate_criterion_safe(
    criterion: Criterion,
    rubric: Rubric,
    chat_content: str,
    panel_config: JudgePanelConfig,
    parsed_session: ChatSession | None,
    metrics: Optional["MetricsAggregator"] = None,
) -> dict[str, Any]:
    """Evaluate a single criterion, validating dimension exists."""
    dimension = rubric.get_dimension(criterion.dimension)
    if not dimension:
        raise ValueError(
            f"Dimension '{criterion.dimension}' not found for criterion '{criterion.name}'"
        )

    return evaluate_criterion_with_panel(
        criterion=criterion,
        chat_content=chat_content,
        dimension=dimension,
        panel_config=panel_config,
        parsed_session=parsed_session,
        metrics=metrics,
    )


def evaluate_rubric_with_panel(
    rubric: Rubric,
    chat_session_file: str,
    panel_config: JudgePanelConfig,
    use_parser: bool = True,
    metrics: Optional["MetricsAggregator"] = None,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all criteria in a rubric using a judge panel.

    Args:
        rubric: The rubric to evaluate
        chat_session_file: Path to the chat session file
        panel_config: Judge panel configuration
        use_parser: Whether to pre-parse the chat session for structured evaluation (default: True)
        metrics: Optional MetricsAggregator for tracking LLM calls

    Returns:
        Dictionary mapping criterion names to evaluation results
    """
    chat_content = read_chat_session(chat_session_file)
    parsed_session = _parse_chat_session_safe(chat_content, use_parser)

    evaluations = {}
    for criterion in rubric.criteria:
        evaluations[criterion.name] = _evaluate_criterion_safe(
            criterion, rubric, chat_content, panel_config, parsed_session, metrics=metrics
        )

    return evaluations


def _format_qa_content(qa_input: Any) -> str:
    """Format Q&A input as text content for evaluation."""
    content_parts = [f"Question: {qa_input.question}", f"\nAnswer:\n{qa_input.answer}"]

    if qa_input.context:
        content_parts.append(f"\n\nContext:\n{qa_input.context}")

    return "\n".join(content_parts)


def evaluate_rubric_with_panel_from_qa(
    rubric: Rubric,
    qna_file: str,
    panel_config: JudgePanelConfig,
    metrics: Optional["MetricsAggregator"] = None,
) -> dict[str, dict[str, Any]]:
    """
    Evaluate all criteria in a rubric using a judge panel, from a Q&A YAML file.

    Args:
        rubric: The rubric to evaluate
        qna_file: Path to Q&A YAML file (must contain question, answer, optional context)
        panel_config: Judge panel configuration
        metrics: Optional MetricsAggregator for tracking LLM calls

    Returns:
        Dictionary mapping criterion names to evaluation results
    """
    qa_input = parse_qa_input(qna_file)
    chat_content = _format_qa_content(qa_input)

    # Q&A format doesn't have tool calls, so don't use parser
    parsed_session = None

    evaluations = {}
    for criterion in rubric.criteria:
        evaluations[criterion.name] = _evaluate_criterion_safe(
            criterion, rubric, chat_content, panel_config, parsed_session, metrics=metrics
        )

    return evaluations
