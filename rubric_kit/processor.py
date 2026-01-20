"""Score processing logic for rubric evaluation."""

from typing import Any

from rubric_kit.schema import Criterion, Dimension, Rubric


def _calculate_tool_criterion_score(
    weight: int, passes: bool, has_required: bool, has_optional: bool, has_prohibited: bool
) -> tuple[int, int]:
    """
    Calculate score and max_score based on tool criterion content.

    Scoring is inferred from which tool lists are present:
    - has_required: Standard scoring (Pass = weight, Fail = 0, max_score = weight)
    - has_optional only: Bonus scoring (Pass = weight, Fail = 0, max_score = 0)
    - has_prohibited only: Penalty scoring (Pass = 0, Fail = -weight, max_score = 0)

    Returns:
        Tuple of (score, max_score)
    """
    # If has required tools, use standard scoring
    if has_required:
        return (weight if passes else 0, weight)
    # If only optional tools, use bonus scoring
    elif has_optional and not has_prohibited:
        return (weight if passes else 0, 0)
    # If only prohibited tools, use penalty scoring
    elif has_prohibited and not has_optional:
        return (0 if passes else -weight, 0)
    else:
        # Mixed optional + prohibited, or empty - use standard
        return (weight if passes else 0, weight)


def evaluate_binary_criterion(
    criterion: Criterion,
    passes: bool,
    reason: str = "",
    consensus_reached: bool = True,
    consensus_count: int = 1,
    judge_votes: list[dict[str, Any]] = None,
    tool_breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a binary (pass/fail) criterion.

    Args:
        criterion: The criterion to evaluate
        passes: Whether the criterion passes
        reason: Optional reasoning for the evaluation
        consensus_reached: Whether consensus was reached
        consensus_count: Number of judges that agreed
        judge_votes: List of individual judge votes
        tool_breakdown: Optional tool breakdown for tool call criteria

    Returns:
        Dictionary with evaluation results
    """
    weight = criterion.weight if isinstance(criterion.weight, int) else 0

    # For tool criteria, infer scoring from tool lists
    if criterion.tool_calls:
        has_required = len(criterion.tool_calls.required) > 0
        has_optional = len(criterion.tool_calls.optional) > 0
        has_prohibited = len(criterion.tool_calls.prohibited) > 0
        score, max_score = _calculate_tool_criterion_score(
            weight, passes, has_required, has_optional, has_prohibited
        )
    else:
        score = weight if passes else 0
        max_score = weight

    result = {
        "criterion_name": criterion.name,
        "criterion_text": criterion.criterion,
        "category": criterion.category,
        "dimension": criterion.dimension,
        "result": "pass" if passes else "fail",
        "score": score,
        "max_score": max_score,
        "reason": reason,
        "consensus_reached": consensus_reached,
        "consensus_count": consensus_count,
    }

    if judge_votes:
        result["judge_votes"] = judge_votes

    if tool_breakdown:
        result["tool_breakdown"] = tool_breakdown

    return result


def evaluate_score_criterion(
    criterion: Criterion,
    dimension: Dimension,
    score: int,
    reason: str = "",
    consensus_reached: bool = True,
    consensus_count: int = 1,
    judge_votes: list[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Evaluate a score-based criterion.

    Args:
        criterion: The criterion to evaluate
        dimension: The dimension defining the score scale
        score: The score value (e.g., 1-3)
        reason: Optional reasoning/additional context for the score
        consensus_reached: Whether consensus was reached
        consensus_count: Number of judges that agreed
        judge_votes: List of individual judge votes

    Returns:
        Dictionary with evaluation results
    """
    if not dimension.scores:
        raise ValueError(f"Dimension '{dimension.name}' does not have scores defined")

    max_score = max(dimension.scores.keys())

    if score not in dimension.scores:
        raise ValueError(f"Score {score} is not valid for dimension '{dimension.name}'")

    # Build criterion text from dimension description and score definitions
    # If criterion is "from_scores", use dimension description
    if criterion.criterion == "from_scores":
        criterion_text = dimension.description
    else:
        criterion_text = criterion.criterion

    # Determine pass/fail based on pass_above threshold
    if dimension.pass_above is not None:
        passes = score >= dimension.pass_above
        result = "pass" if passes else "fail"
    else:
        # No pass_above defined, result is just the numeric score
        result = score

    result_dict = {
        "criterion_name": criterion.name,
        "criterion_text": criterion_text,
        "category": criterion.category,
        "dimension": criterion.dimension,
        "result": result,
        "score": score,
        "max_score": max_score,
        "reason": reason,
        "consensus_reached": consensus_reached,
        "consensus_count": consensus_count,
    }

    if judge_votes:
        result_dict["judge_votes"] = judge_votes

    return result_dict


def evaluate_rubric(rubric: Rubric, evaluations: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Evaluate a complete rubric with provided evaluation inputs.

    Args:
        rubric: The rubric to evaluate
        evaluations: Dictionary mapping criterion names to evaluation data
                    Format: {
                        "criterion_name": {
                            "type": "binary",  # or "score"
                            "passes": True,    # for binary
                            "score": 3         # for score
                        }
                    }

    Returns:
        List of evaluation results for each criterion
    """
    results = []

    for criterion in rubric.criteria:
        if criterion.name not in evaluations:
            raise ValueError(f"No evaluation provided for criterion '{criterion.name}'")

        eval_data = evaluations[criterion.name]
        eval_type = eval_data.get("type")

        if eval_type == "binary":
            passes = eval_data.get("passes", False)
            reason = eval_data.get("reason", "")
            consensus_reached = eval_data.get("consensus_reached", True)
            consensus_count = eval_data.get("consensus_count", 1)
            judge_votes = eval_data.get("judge_votes")
            tool_breakdown = eval_data.get("tool_breakdown")

            result = evaluate_binary_criterion(
                criterion,
                passes,
                reason,
                consensus_reached=consensus_reached,
                consensus_count=consensus_count,
                judge_votes=judge_votes,
                tool_breakdown=tool_breakdown,
            )
            results.append(result)

        elif eval_type == "score":
            score = eval_data.get("score")
            if score is None:
                raise ValueError(f"Score not provided for criterion '{criterion.name}'")

            reason = eval_data.get("reason", "")
            consensus_reached = eval_data.get("consensus_reached", True)
            consensus_count = eval_data.get("consensus_count", 1)
            judge_votes = eval_data.get("judge_votes")

            # Find the dimension for this criterion
            dimension = rubric.get_dimension(criterion.dimension)
            if not dimension:
                raise ValueError(f"Dimension '{criterion.dimension}' not found")

            result = evaluate_score_criterion(
                criterion,
                dimension,
                score,
                reason,
                consensus_reached=consensus_reached,
                consensus_count=consensus_count,
                judge_votes=judge_votes,
            )
            results.append(result)
        else:
            raise ValueError(
                f"Unknown evaluation type '{eval_type}' for criterion '{criterion.name}'"
            )

    return results


def calculate_total_score(results: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Calculate total score from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Tuple of (total_score, max_possible_score)
    """
    total = sum(r["score"] for r in results)
    max_total = sum(r["max_score"] for r in results)
    return total, max_total


def calculate_percentage_score(results: list[dict[str, Any]]) -> float:
    """
    Calculate percentage score from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Percentage score (0-100)
    """
    total, max_total = calculate_total_score(results)
    if max_total == 0:
        return 0.0
    return (total / max_total) * 100
