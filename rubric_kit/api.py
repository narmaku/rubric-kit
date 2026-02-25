"""Public Python API for rubric-kit.

This module provides the programmatic interface to rubric-kit, enabling
evaluation, generation, and refinement of rubrics without CLI invocation.

Usage::

    from rubric_kit import evaluate, generate, refine

    result = evaluate(
        rubric="path/to/rubric.yaml",
        input_file="path/to/chat.txt",
        model="gpt-4o",
    )
    print(f"Score: {result.summary.percentage:.1f}%")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from rubric_kit.generator import parse_dimensions_file
from rubric_kit.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.metrics import MetricsAggregator
from rubric_kit.processor import calculate_percentage_score, calculate_total_score, evaluate_rubric
from rubric_kit.schema import (
    ConsensusConfig,
    Dimension,
    ExecutionConfig,
    JudgeConfig,
    JudgePanelConfig,
    Rubric,
)
from rubric_kit.validator import load_judge_panel_config, load_rubric

logger = logging.getLogger("rubric_kit")


# =============================================================================
# Result Models
# =============================================================================


class CriterionResult(BaseModel):
    """Result for a single criterion evaluation.

    Maps directly to the dict structure returned by
    ``processor.evaluate_binary_criterion()`` and
    ``processor.evaluate_score_criterion()``.
    """

    criterion_name: str
    criterion_text: str | None = None
    category: str | None = None
    dimension: str
    result: str | int  # "pass", "fail", or integer score
    score: int
    max_score: int
    reason: str = ""
    consensus_reached: bool = True
    consensus_count: int = 1
    judge_votes: list[dict[str, Any]] | None = None
    tool_breakdown: dict[str, Any] | None = None


class ScoreSummary(BaseModel):
    """Aggregated score summary for an evaluation."""

    total_score: int
    max_score: int
    percentage: float


class EvaluationResult(BaseModel):
    """Complete result of an ``evaluate()`` call.

    Contains per-criterion results, score summary, the rubric and panel
    configuration used, and optional LLM metrics.
    """

    criteria_results: list[CriterionResult]
    summary: ScoreSummary
    rubric: Rubric
    panel_config: JudgePanelConfig
    input_type: Literal["chat_session", "qna"]
    input_source: str
    metrics: Any | None = None  # MetricsSummary (avoid import cycle)
    timestamp: datetime = Field(default_factory=datetime.now)


class GenerationResult(BaseModel):
    """Complete result of a ``generate()`` call."""

    rubric: Rubric
    model: str
    input_type: Literal["chat_session", "qna"]
    input_source: str
    metrics: Any | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class RefinementResult(BaseModel):
    """Complete result of a ``refine()`` call."""

    rubric: Rubric
    original_rubric: Rubric
    model: str
    had_feedback: bool = False
    had_context: bool = False
    metrics: Any | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ContestantResult(BaseModel):
    """Result for a single arena contestant."""

    contestant_id: str
    name: str
    description: str | None = None
    metadata: dict[str, Any] | None = None
    criteria_results: list[CriterionResult]
    summary: ScoreSummary


class ArenaRanking(BaseModel):
    """A single ranking entry in arena results."""

    rank: int
    contestant_id: str
    name: str
    percentage: float
    total_score: int
    max_score: int


class ArenaResult(BaseModel):
    """Complete result of an ``arena()`` call."""

    arena_name: str
    arena_description: str | None = None
    contestants: dict[str, ContestantResult] = Field(default_factory=dict)
    rankings: list[ArenaRanking] = Field(default_factory=list)
    rubric: Rubric | None = None
    panel_config: JudgePanelConfig | None = None
    metrics: Any | None = None
    failed_contestants: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class DryRunResult(BaseModel):
    """Result of a dry-run cost estimation."""

    total_calls: int
    prompt_tokens: int
    cost_minimal: float
    cost_conservative: float
    cost_worst_case: float
    model_estimates: dict[str, dict[str, Any]]


class ExportResult(BaseModel):
    """Result of an ``export()`` call."""

    format: Literal["pdf", "csv", "json"]
    output_path: str
    success: bool = True


# =============================================================================
# Internal Helpers
# =============================================================================


def _resolve_rubric(
    rubric: Rubric | str | Path,
    variables_file: str | Path | None = None,
    require_variables: bool = True,
) -> Rubric:
    """Resolve a rubric from either a Rubric object or a file path.

    Args:
        rubric: A Rubric object or path to a rubric YAML file.
        variables_file: Optional path to external variables file.
        require_variables: Whether to require all variables to be defined.

    Returns:
        Resolved Rubric object.

    Raises:
        RubricValidationError: If the rubric file is invalid or not found.
    """
    if isinstance(rubric, Rubric):
        return rubric
    variables_path = str(variables_file) if variables_file else None
    return load_rubric(
        str(rubric),
        variables_file=variables_path,
        require_variables=require_variables,
    )


def _resolve_panel_config(
    panel_config: JudgePanelConfig | str | Path | None,
    model: str = "gpt-4",
    base_url: str | None = None,
) -> JudgePanelConfig:
    """Resolve panel config from an object, file path, or create a default.

    Args:
        panel_config: A JudgePanelConfig, path to config YAML, or None.
        model: Model name for the default single-judge panel.
        base_url: Custom API base URL for the default panel.

    Returns:
        Resolved JudgePanelConfig object.
    """
    if isinstance(panel_config, JudgePanelConfig):
        return panel_config
    if panel_config is not None:
        return load_judge_panel_config(str(panel_config))
    return JudgePanelConfig(
        judges=[JudgeConfig(name="default", model=model, base_url=base_url)],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous"),
    )


def _resolve_input(
    input_file: str | Path | None,
    input_content: str | None,
) -> tuple[str | None, str | None]:
    """Validate and resolve input source.

    Exactly one of ``input_file`` or ``input_content`` must be provided.

    Args:
        input_file: Path to an input file.
        input_content: Raw input content string.

    Returns:
        Tuple of (file_path_as_str_or_None, content_or_None).

    Raises:
        ValueError: If both or neither are provided.
    """
    if input_file and input_content:
        raise ValueError("Provide either input_file or input_content, not both.")
    if not input_file and not input_content:
        raise ValueError("Either input_file or input_content must be provided.")
    if input_file:
        return str(input_file), None
    return None, input_content


def _resolve_dimensions(
    dimensions: list[Dimension] | str | Path | None,
) -> list[Dimension] | None:
    """Resolve dimensions from a list, file path, or None.

    Args:
        dimensions: A list of Dimension objects, path to YAML file, or None.

    Returns:
        List of Dimension objects, or None.
    """
    if dimensions is None:
        return None
    if isinstance(dimensions, list):
        return dimensions
    return parse_dimensions_file(str(dimensions))


def _build_criterion_results(
    raw_results: list[dict[str, Any]],
) -> list[CriterionResult]:
    """Convert raw processor result dicts to typed CriterionResult objects.

    Args:
        raw_results: List of dicts from ``processor.evaluate_rubric()``.

    Returns:
        List of CriterionResult objects.
    """
    return [CriterionResult(**r) for r in raw_results]


# =============================================================================
# Public API Functions
# =============================================================================


def evaluate(
    *,
    rubric: Rubric | str | Path,
    input_file: str | Path | None = None,
    input_content: str | None = None,
    input_type: Literal["chat_session", "qna"] = "chat_session",
    panel_config: JudgePanelConfig | str | Path | None = None,
    model: str = "gpt-4",
    base_url: str | None = None,
    variables_file: str | Path | None = None,
    track_metrics: bool = True,
    include_call_log: bool = False,
) -> EvaluationResult:
    """Evaluate input against a rubric using an LLM judge panel.

    Args:
        rubric: A Rubric object, or path to a rubric YAML file.
        input_file: Path to input file (chat session or Q&A YAML).
        input_content: Raw input content string (alternative to input_file).
            Exactly one of input_file or input_content must be provided.
        input_type: Type of input: ``"chat_session"`` or ``"qna"``.
        panel_config: A JudgePanelConfig object, or path to panel config YAML.
            If None, creates a single-judge panel using ``model``.
        model: Model to use when panel_config is not provided.
        base_url: Custom API base URL when panel_config is not provided.
        variables_file: Path to external variables file for rubric substitution.
        track_metrics: Whether to track LLM call metrics.
        include_call_log: Whether to include detailed call log in metrics.

    Returns:
        EvaluationResult with typed criteria results and score summary.

    Raises:
        ValueError: If neither input_file nor input_content is provided,
            or if both are provided.
        FileNotFoundError: If rubric file, input file, or panel config not found.
        RubricValidationError: If rubric is invalid.
    """
    import os
    import tempfile

    # Resolve inputs
    resolved_rubric = _resolve_rubric(rubric, variables_file=variables_file)
    resolved_panel = _resolve_panel_config(panel_config, model=model, base_url=base_url)
    file_path, content = _resolve_input(input_file, input_content)

    # Create metrics aggregator
    metrics = MetricsAggregator(include_call_log=include_call_log) if track_metrics else None

    # Determine input source for result metadata
    input_source = str(file_path) if file_path else "<in-memory>"

    # Run evaluation
    temp_file = None
    try:
        if content is not None:
            # Write content to temp file for core functions that require file paths
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(content)
                temp_file = f.name
            eval_file = temp_file
        else:
            eval_file = file_path

        logger.info("Evaluating %s from %s", input_type.replace("_", " "), input_source)

        if input_type == "qna":
            evaluations = evaluate_rubric_with_panel_from_qa(
                resolved_rubric, eval_file, resolved_panel, metrics=metrics
            )
        else:
            evaluations = evaluate_rubric_with_panel(
                resolved_rubric, eval_file, resolved_panel, metrics=metrics
            )
    finally:
        if temp_file is not None:
            os.unlink(temp_file)

    # Process scores
    results = evaluate_rubric(resolved_rubric, evaluations)
    total_score, max_score = calculate_total_score(results)
    percentage = calculate_percentage_score(results)

    # Build typed result
    criteria_results = _build_criterion_results(results)
    summary = ScoreSummary(
        total_score=total_score,
        max_score=max_score,
        percentage=round(percentage, 1),
    )
    metrics_summary = metrics.get_summary() if metrics else None

    return EvaluationResult(
        criteria_results=criteria_results,
        summary=summary,
        rubric=resolved_rubric,
        panel_config=resolved_panel,
        input_type=input_type,
        input_source=input_source,
        metrics=metrics_summary,
    )
