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

from rubric_kit.generator import (
    RubricGenerator,
    parse_chat_session,
    parse_dimensions_file,
    parse_qa_input,
)
from rubric_kit.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.metrics import MetricsAggregator, estimate_cost, estimate_tokens
from rubric_kit.output import convert_yaml_to_csv, convert_yaml_to_json
from rubric_kit.pdf_export import export_evaluation_pdf
from rubric_kit.processor import calculate_percentage_score, calculate_total_score, evaluate_rubric
from rubric_kit.prompts import EVALUATOR_CONFIG, build_binary_criterion_prompt
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


def generate(
    *,
    input_file: str | Path | None = None,
    input_content: str | None = None,
    input_type: Literal["chat_session", "qna"] = "qna",
    model: str = "gpt-4",
    base_url: str | None = None,
    num_dimensions: int | None = None,
    num_criteria: int | None = None,
    category_hints: list[str] | None = None,
    dimensions: list[Dimension] | str | Path | None = None,
    use_variables: bool = True,
    guidelines: str | None = None,
    track_metrics: bool = True,
) -> GenerationResult:
    """Generate a rubric from input content using an LLM.

    Args:
        input_file: Path to input file (Q&A or chat session).
        input_content: Raw input content string (alternative to input_file).
        input_type: Type of input: ``"qna"`` or ``"chat_session"``.
        model: LLM model to use for generation.
        base_url: Custom API base URL.
        num_dimensions: Number of dimensions to generate (None for auto).
        num_criteria: Number of criteria to generate (None for auto).
        category_hints: Optional category names to guide generation.
        dimensions: Pre-defined dimensions (list or path to YAML file).
        use_variables: Whether to extract variables from content.
        guidelines: Optional guidelines text to guide generation.
        track_metrics: Whether to track LLM call metrics.

    Returns:
        GenerationResult containing the generated Rubric.

    Raises:
        ValueError: If neither input_file nor input_content is provided.
        FileNotFoundError: If input file or dimensions file not found.
    """
    import tempfile

    file_path, content = _resolve_input(input_file, input_content)
    resolved_dims = _resolve_dimensions(dimensions)

    # Create metrics aggregator
    metrics = MetricsAggregator() if track_metrics else None

    # Create generator
    generator = RubricGenerator(model=model, base_url=base_url, metrics=metrics)

    # Determine input source for result metadata
    input_source = str(file_path) if file_path else "<in-memory>"

    # Handle inline content by writing to temp file for parsers
    temp_file = None
    try:
        if content is not None:
            suffix = ".yaml" if input_type == "qna" else ".txt"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as f:
                f.write(content)
                temp_file = f.name
            parse_path = temp_file
        else:
            parse_path = str(file_path)

        # Parse input and generate rubric
        if input_type == "qna":
            qa_input = parse_qa_input(parse_path)
            rubric = generator.generate_rubric(
                qa_input,
                num_dimensions=num_dimensions,
                num_criteria=num_criteria,
                category_hints=category_hints,
                dimensions=resolved_dims,
                use_variables=use_variables,
                guidelines=guidelines,
            )
        else:
            chat_input = parse_chat_session(parse_path)
            rubric = generator.generate_rubric_from_chat(
                chat_input,
                num_dimensions=num_dimensions,
                num_criteria=num_criteria,
                category_hints=category_hints,
                dimensions=resolved_dims,
                use_variables=use_variables,
                guidelines=guidelines,
            )
    finally:
        if temp_file is not None:
            import os

            os.unlink(temp_file)

    metrics_summary = metrics.get_summary() if metrics else None

    return GenerationResult(
        rubric=rubric,
        model=model,
        input_type=input_type,
        input_source=input_source,
        metrics=metrics_summary,
    )


def refine(
    *,
    rubric: Rubric | str | Path,
    model: str = "gpt-4",
    base_url: str | None = None,
    feedback: str | None = None,
    input_file: str | Path | None = None,
    input_content: str | None = None,
    input_type: Literal["chat_session", "qna"] | None = None,
    dimensions: list[Dimension] | str | Path | None = None,
    use_variables: bool = True,
    variables_file: str | Path | None = None,
    track_metrics: bool = True,
) -> RefinementResult:
    """Refine an existing rubric with optional feedback and context.

    Args:
        rubric: A Rubric object or path to a rubric YAML file.
        model: LLM model to use for refinement.
        base_url: Custom API base URL.
        feedback: Optional feedback text to guide refinement.
        input_file: Optional path to context file (Q&A or chat session).
        input_content: Optional raw context content string.
        input_type: Type of context input. Required if input_file or
            input_content is provided.
        dimensions: Optional dimensions to merge with existing.
        use_variables: Whether to use variables in the refined rubric.
        variables_file: Path to external variables file.
        track_metrics: Whether to track LLM call metrics.

    Returns:
        RefinementResult containing the refined Rubric.

    Raises:
        ValueError: If input is provided without input_type.
        FileNotFoundError: If rubric or input file not found.
        RubricValidationError: If rubric is invalid.
    """
    import tempfile

    resolved_rubric = _resolve_rubric(
        rubric, variables_file=variables_file, require_variables=False
    )
    resolved_dims = _resolve_dimensions(dimensions)

    # Create metrics aggregator
    metrics = MetricsAggregator() if track_metrics else None

    # Create generator
    generator = RubricGenerator(model=model, base_url=base_url, metrics=metrics)

    has_context = input_file is not None or input_content is not None

    # Refine with or without context
    if has_context:
        if input_type is None:
            raise ValueError("input_type is required when input_file or input_content is provided.")

        # Resolve context input
        file_path, content = _resolve_input(input_file, input_content)

        temp_file = None
        try:
            if content is not None:
                suffix = ".yaml" if input_type == "qna" else ".txt"
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=suffix, delete=False, encoding="utf-8"
                ) as f:
                    f.write(content)
                    temp_file = f.name
                parse_path = temp_file
            else:
                parse_path = str(file_path)

            if input_type == "qna":
                qa_input = parse_qa_input(parse_path)
                refined = generator.refine_rubric_with_qa(
                    resolved_rubric,
                    qa_input,
                    feedback=feedback,
                    dimensions_to_merge=resolved_dims,
                    use_variables=use_variables,
                )
            else:
                chat_input = parse_chat_session(parse_path)
                refined = generator.refine_rubric_with_chat(
                    resolved_rubric,
                    chat_input,
                    feedback=feedback,
                    dimensions_to_merge=resolved_dims,
                    use_variables=use_variables,
                )
        finally:
            if temp_file is not None:
                import os

                os.unlink(temp_file)
    else:
        refined = generator.refine_rubric(
            resolved_rubric,
            feedback=feedback,
            dimensions_to_merge=resolved_dims,
            use_variables=use_variables,
        )

    metrics_summary = metrics.get_summary() if metrics else None

    return RefinementResult(
        rubric=refined,
        original_rubric=resolved_rubric,
        model=model,
        had_feedback=feedback is not None,
        had_context=has_context,
        metrics=metrics_summary,
    )


def export(
    *,
    input_file: str | Path,
    output_file: str | Path,
    format: Literal["pdf", "csv", "json"],
) -> ExportResult:
    """Export evaluation results to a different format.

    Args:
        input_file: Path to evaluation YAML output file.
        output_file: Path for the exported file.
        format: Output format: ``"pdf"``, ``"csv"``, or ``"json"``.

    Returns:
        ExportResult indicating success and output path.

    Raises:
        ValueError: If format is not supported.
        FileNotFoundError: If input file not found.
    """
    in_path = str(input_file)
    out_path = str(output_file)

    if format == "pdf":
        export_evaluation_pdf(in_path, out_path)
    elif format == "csv":
        convert_yaml_to_csv(in_path, out_path)
    elif format == "json":
        convert_yaml_to_json(in_path, out_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")

    return ExportResult(format=format, output_path=out_path)


def dry_run_evaluate(
    *,
    rubric: Rubric | str | Path,
    panel_config: JudgePanelConfig | str | Path | None = None,
    model: str = "gpt-4",
    variables_file: str | Path | None = None,
) -> DryRunResult:
    """Estimate the cost of an evaluation without making LLM calls.

    Args:
        rubric: A Rubric object or path to rubric YAML file.
        panel_config: A JudgePanelConfig object or path to config file.
            If None, uses a single-judge panel with ``model``.
        model: Model to use for cost estimation when panel_config is None.
        variables_file: Path to external variables file.

    Returns:
        DryRunResult with cost estimates per model.
    """
    resolved_rubric = _resolve_rubric(rubric, variables_file=variables_file)
    resolved_panel = _resolve_panel_config(panel_config, model=model)

    judge_models = [judge.model for judge in resolved_panel.judges]

    # Cost estimation constants
    minimal_tokens = 400
    config = EVALUATOR_CONFIG
    max_tokens = config.max_tokens
    conservative_tokens = int(max_tokens * 0.1)

    estimates: dict[str, dict[str, Any]] = {}

    for criterion in resolved_rubric.criteria:
        prompt = build_binary_criterion_prompt(
            criterion, "[Sample chat content for estimation]"
        )
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ]

        for judge_model in judge_models:
            if judge_model not in estimates:
                estimates[judge_model] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "cost_minimal": 0.0,
                    "cost_conservative": 0.0,
                    "cost_worst_case": 0.0,
                }

            prompt_tokens = estimate_tokens(judge_model, messages)
            estimates[judge_model]["calls"] += 1
            estimates[judge_model]["prompt_tokens"] += prompt_tokens
            estimates[judge_model]["cost_minimal"] += estimate_cost(
                judge_model, prompt_tokens, minimal_tokens
            )
            estimates[judge_model]["cost_conservative"] += estimate_cost(
                judge_model, prompt_tokens, conservative_tokens
            )
            estimates[judge_model]["cost_worst_case"] += estimate_cost(
                judge_model, prompt_tokens, max_tokens
            )

    # Calculate totals
    total_calls = sum(m["calls"] for m in estimates.values())
    total_prompt_tokens = sum(m["prompt_tokens"] for m in estimates.values())
    total_minimal = sum(m["cost_minimal"] for m in estimates.values())
    total_conservative = sum(m["cost_conservative"] for m in estimates.values())
    total_worst = sum(m["cost_worst_case"] for m in estimates.values())

    return DryRunResult(
        total_calls=total_calls,
        prompt_tokens=total_prompt_tokens,
        cost_minimal=total_minimal,
        cost_conservative=total_conservative,
        cost_worst_case=total_worst,
        model_estimates=estimates,
    )
