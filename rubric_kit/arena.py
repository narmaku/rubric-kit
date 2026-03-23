"""Arena evaluation module for comparing multiple contestants against a shared rubric."""

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

import yaml

from rubric_kit.core.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.core.processor import (
    calculate_percentage_score,
    calculate_total_score,
    evaluate_rubric,
)
from rubric_kit.io.output import print_evaluation_config
from rubric_kit.io.validator import load_judge_panel_config, load_rubric, substitute_variables
from rubric_kit.models import converters
from rubric_kit.models.schema import (
    ArenaContestant,
    ArenaSpec,
    Criterion,
    Dimension,
    JudgePanelConfig,
    Rubric,
    ToolCalls,
    ToolSpec,
)
from rubric_kit.reports.pdf_arena import export_arena_pdf


logger = logging.getLogger(__name__)


# Default evaluator functions (can be overridden for testing via dependency injection)
def _default_evaluate_panel(rubric, input_file, panel_config):
    """Default implementation using evaluate_rubric_with_panel."""
    return evaluate_rubric_with_panel(rubric, input_file, panel_config)


def _default_evaluate_panel_qa(rubric, input_file, panel_config):
    """Default implementation using evaluate_rubric_with_panel_from_qa."""
    return evaluate_rubric_with_panel_from_qa(rubric, input_file, panel_config)


def load_arena_spec(arena_spec_file: str) -> ArenaSpec:
    """Load and validate an arena specification file."""
    if not os.path.exists(arena_spec_file):
        raise FileNotFoundError(f"Arena spec file not found: {arena_spec_file}")

    with open(arena_spec_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "arena" not in data:
        raise ValueError("Arena spec file must have an 'arena' key at the root")

    arena_data = data["arena"]
    contestants = [ArenaContestant(**c) for c in arena_data.get("contestants", [])]

    return ArenaSpec(
        name=arena_data.get("name"),
        description=arena_data.get("description"),
        rubric_file=arena_data["rubric_file"],
        judges_panel_file=arena_data["judges_panel_file"],
        contestants=contestants,
    )


def load_contestant_variables(contestant: ArenaContestant, base_dir: str) -> dict[str, str] | None:
    """Load variables for a contestant from inline definition or external file."""
    if contestant.variables:
        return contestant.variables

    if not contestant.variables_file:
        return None

    variables_path = os.path.join(base_dir, contestant.variables_file)
    if not os.path.exists(variables_path):
        raise FileNotFoundError(f"Variables file not found: {variables_path}")

    with open(variables_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("variables", data)


def apply_variables_to_rubric(base_rubric: Rubric, variables: dict[str, str]) -> Rubric:
    """Create a new Rubric with variable substitution applied to criterion text and tool params."""
    substituted_criteria = []
    for crit in base_rubric.criteria:
        new_criterion_text = substitute_variables(crit.criterion, variables)
        new_tool_calls = (
            _substitute_tool_calls(crit.tool_calls, variables) if crit.tool_calls else None
        )

        substituted_criteria.append(
            Criterion(
                name=crit.name,
                category=crit.category,
                weight=crit.weight,
                dimension=crit.dimension,
                criterion=new_criterion_text,
                tool_calls=new_tool_calls,
            )
        )

    substituted_dimensions = [
        Dimension(
            name=dim.name,
            description=substitute_variables(dim.description, variables),
            grading_type=dim.grading_type,
            scores=dim.scores,
            pass_above=dim.pass_above,
        )
        for dim in base_rubric.dimensions
    ]

    return Rubric(
        dimensions=substituted_dimensions, criteria=substituted_criteria, variables=variables
    )


def _substitute_tool_calls(tool_calls: ToolCalls, variables: dict[str, str]) -> ToolCalls:
    """Apply variable substitution to tool call parameters."""

    def substitute_params(tc: ToolSpec) -> ToolSpec:
        if tc.params is None:
            return tc
        new_params = {
            k: substitute_variables(v, variables) if isinstance(v, str) else v
            for k, v in tc.params.items()
        }
        return ToolSpec(
            name=tc.name, min_calls=tc.min_calls, max_calls=tc.max_calls, params=new_params
        )

    return ToolCalls(
        respect_order=tool_calls.respect_order,
        params_strict_mode=tool_calls.params_strict_mode,
        required=[substitute_params(tc) for tc in tool_calls.required],
        optional=[substitute_params(tc) for tc in tool_calls.optional],
        prohibited=tool_calls.prohibited,
    )


def _generate_contestant_id(index: int, filename: str) -> str:
    """Generate a unique contestant ID from a 1-based index and filename.

    Args:
        index: Zero-based index of the contestant in the input list.
        filename: Full path or basename of the output file.

    Returns:
        ID in the format ``contestant-NNN-sanitized-basename``.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    sanitized = basename.replace("output_", "").replace("_", "-")
    return f"contestant-{index + 1:03d}-{sanitized}"


def combine_outputs_to_arena(
    output_files: list[str], arena_name: str = "Combined Arena"
) -> dict[str, Any]:
    """Combine multiple evaluation output files into arena format."""
    contestants_results: dict[str, Any] = {}
    seen_ids: set[str] = set()
    shared_rubric = None
    shared_judge_panel = None

    for idx, output_file in enumerate(output_files):
        logger.info("[%d/%d] Loading: %s", idx + 1, len(output_files), output_file)

        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")

        with open(output_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data.get("mode") == "arena":
            raise ValueError(f"File is already an arena result: {output_file}")

        if not data.get("results"):
            raise ValueError(f"File missing 'results' section: {output_file}")

        contestant_id = _generate_contestant_id(idx, output_file)
        if contestant_id in seen_ids:
            raise ValueError(
                f"Duplicate contestant ID '{contestant_id}' generated from file: {output_file}"
            )
        seen_ids.add(contestant_id)
        metadata = data.get("metadata", {})
        basename = os.path.splitext(os.path.basename(output_file))[0]
        contestant_name = metadata.get("report_title", basename)
        input_info = data.get("input", {})
        summary = data.get("summary", {})

        logger.info("   ID: %s", contestant_id)
        logger.info("   Name: %s", contestant_name)
        logger.info(
            "   Score: %s/%s (%.1f%%)",
            summary.get("total_score", 0),
            summary.get("max_score", 0),
            summary.get("percentage", 0),
        )

        contestants_results[contestant_id] = {
            "name": contestant_name,
            "description": f"Loaded from {output_file}",
            "metadata": {
                "source_file": output_file,
                "original_timestamp": metadata.get("timestamp"),
                "rubric_source": metadata.get("rubric_source_file"),
                "judge_panel_source": metadata.get("judge_panel_source_file"),
            },
            "input": input_info,
            "results": data.get("results", []),
            "summary": summary,
        }

        if shared_rubric is None and data.get("rubric"):
            shared_rubric = data["rubric"]
        if shared_judge_panel is None and data.get("judge_panel"):
            shared_judge_panel = data["judge_panel"]

    rankings = _generate_rankings(contestants_results)

    return {
        "mode": "arena",
        "arena_name": arena_name,
        "arena_description": f"Combined from {len(output_files)} evaluation outputs",
        "contestants": contestants_results,
        "rankings": rankings,
        "rubric": shared_rubric,
        "judge_panel": shared_judge_panel,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_files": output_files,
            "combined_from_outputs": True,
        },
    }


def _generate_rankings(contestants_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate sorted rankings from contestant results."""
    rankings = sorted(
        [
            {
                "id": cid,
                "name": cdata["name"],
                "percentage": cdata["summary"].get("percentage", 0),
                "total_score": cdata["summary"].get("total_score", 0),
                "max_score": cdata["summary"].get("max_score", 0),
            }
            for cid, cdata in contestants_results.items()
        ],
        key=lambda x: x["percentage"],
        reverse=True,
    )

    for idx, r in enumerate(rankings, 1):
        r["rank"] = idx

    return rankings


def _save_partial_arena_results(
    output_file: str,
    arena_name: str,
    arena_spec: ArenaSpec,
    contestants_results: dict[str, Any],
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    report_title: str | None = None,
) -> None:
    """Save partial arena results after each contestant evaluation."""
    rankings = _generate_rankings(contestants_results)

    output_data = {
        "mode": "arena",
        "arena_name": arena_name,
        "arena_description": arena_spec.description,
        "contestants": contestants_results,
        "rankings": rankings,
        "rubric": converters.rubric_to_portable_dict(base_rubric),
        "judge_panel": converters.panel_config_to_portable_dict(panel_config),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "rubric_source_file": arena_spec.rubric_file,
            "judge_panel_source_file": arena_spec.judges_panel_file,
            "partial": True,
        },
    }

    if report_title:
        output_data["metadata"]["report_title"] = report_title

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def _evaluate_contestant(
    contestant: ArenaContestant,
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    base_dir: str,
    evaluate_panel: Callable | None = None,
    evaluate_panel_qa: Callable | None = None,
) -> dict[str, Any]:
    """Evaluate a single contestant and return results."""
    evaluate_panel = evaluate_panel or _default_evaluate_panel
    evaluate_panel_qa = evaluate_panel_qa or _default_evaluate_panel_qa

    contestant_vars = load_contestant_variables(contestant, base_dir)

    if contestant_vars:
        rubric = apply_variables_to_rubric(base_rubric, contestant_vars)
        logger.info("   Variables: %d", len(contestant_vars))
    else:
        rubric = base_rubric

    input_path = os.path.join(base_dir, contestant.input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("   Input: %s from %s", contestant.input_type, contestant.input_file)

    if contestant.input_type == "qna":
        evaluations = evaluate_panel_qa(rubric, input_path, panel_config)
    else:
        evaluations = evaluate_panel(rubric, input_path, panel_config)

    results = evaluate_rubric(rubric, evaluations)
    total_score, max_score = calculate_total_score(results)
    percentage = calculate_percentage_score(results)

    logger.info("   Score: %d/%d (%.1f%%)", total_score, max_score, percentage)

    return {
        "name": contestant.name,
        "description": contestant.description,
        "metadata": contestant.metadata,
        "input": {"type": contestant.input_type, "source_file": contestant.input_file},
        "results": results,
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1),
        },
    }


def _load_cached_results(output_file: str, force: bool) -> dict[str, Any]:
    """Load existing results from output file if available."""
    if not os.path.exists(output_file) or force:
        if force:
            logger.info("Force mode: will re-evaluate all contestants")
        return {}

    logger.info("Found existing results in %s", output_file)
    try:
        with open(output_file, encoding="utf-8") as f:
            existing_data = yaml.safe_load(f)
        if existing_data and existing_data.get("mode") == "arena":
            existing_results = existing_data.get("contestants", {})
            logger.info("Loaded %d cached contestant results", len(existing_results))
            logger.info("(Use --force to re-evaluate all)")
            return existing_results
    except Exception as e:
        logger.warning("Could not load existing results: %s", e)

    return {}


def run_arena_from_spec(
    arena_spec_file: str,
    output_file: str,
    report_file: str | None = None,
    report_title: str | None = None,
    force: bool = False,
    print_table: bool = True,
    evaluate_panel: Callable | None = None,
    evaluate_panel_qa: Callable | None = None,
    pdf_exporter: Callable | None = None,
    include_input: bool = False,
) -> int:
    """Run arena evaluation from specification file."""
    logger.info("Loading arena specification from %s...", arena_spec_file)
    arena_spec = load_arena_spec(arena_spec_file)
    arena_name = arena_spec.name or "Arena Evaluation"
    logger.info("Loaded arena: %s", arena_name)
    logger.info("   Contestants: %d", len(arena_spec.contestants))

    existing_results = _load_cached_results(output_file, force)

    base_dir = os.path.dirname(os.path.abspath(arena_spec_file))

    rubric_path = os.path.join(base_dir, arena_spec.rubric_file)
    logger.info("Loading shared rubric from %s...", rubric_path)
    base_rubric = load_rubric(rubric_path, require_variables=False)
    logger.info(
        "Loaded %d dimensions and %d criteria",
        len(base_rubric.dimensions),
        len(base_rubric.criteria),
    )

    panel_path = os.path.join(base_dir, arena_spec.judges_panel_file)
    logger.info("Loading judge panel from %s...", panel_path)
    panel_config = load_judge_panel_config(panel_path)
    logger.info("Loaded panel with %d judge(s)", len(panel_config.judges))
    print_evaluation_config(panel_config)

    contestants_results: dict[str, Any] = dict(existing_results)
    failed_contestants: list[str] = []
    skipped_count = 0
    evaluated_count = 0

    logger.info("=" * 80)
    logger.info("ARENA EVALUATION")
    logger.info("=" * 80)

    for idx, contestant in enumerate(arena_spec.contestants, 1):
        if contestant.id in existing_results:
            cached = existing_results[contestant.id]
            cached_pct = cached.get("summary", {}).get("percentage", 0)
            logger.info(
                "[%d/%d] %s (id: %s)",
                idx,
                len(arena_spec.contestants),
                contestant.name,
                contestant.id,
            )
            logger.info("   Skipped (cached: %.1f%%)", cached_pct)
            skipped_count += 1
            continue

        logger.info(
            "[%d/%d] Evaluating: %s (id: %s)",
            idx,
            len(arena_spec.contestants),
            contestant.name,
            contestant.id,
        )

        try:
            contestants_results[contestant.id] = _evaluate_contestant(
                contestant,
                base_rubric,
                panel_config,
                base_dir,
                evaluate_panel=evaluate_panel,
                evaluate_panel_qa=evaluate_panel_qa,
            )
            evaluated_count += 1

            _save_partial_arena_results(
                output_file,
                arena_name,
                arena_spec,
                contestants_results,
                base_rubric,
                panel_config,
                report_title,
            )
        except Exception as e:
            logger.error("   Failed: %s", e)
            failed_contestants.append(contestant.id)

    _print_evaluation_summary(evaluated_count, skipped_count, failed_contestants)

    rankings = _generate_rankings(contestants_results)
    output_data = _build_arena_output(
        arena_name,
        arena_spec,
        contestants_results,
        rankings,
        base_rubric,
        panel_config,
        arena_spec_file,
        report_title,
        failed_contestants,
    )

    logger.info("Writing arena results to %s...", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    status = (
        f" - {len(failed_contestants)} contestant(s) pending"
        if failed_contestants
        else " - complete"
    )
    logger.info("Arena results written (YAML)%s", status)

    if report_file:
        _generate_arena_report(output_file, report_file, pdf_exporter, include_input=include_input)

    if print_table:
        _print_arena_rankings(rankings)

    return 0


def run_arena_from_outputs(
    output_files: list[str],
    output_file: str,
    report_file: str | None = None,
    report_title: str | None = None,
    print_table: bool = True,
    include_input: bool = False,
) -> int:
    """Combine existing output files into arena format."""
    logger.info("Combining %d evaluation outputs into arena format...", len(output_files))

    arena_name = report_title or "Combined Arena Evaluation"
    output_data = combine_outputs_to_arena(output_files, arena_name)

    if report_title:
        output_data["metadata"]["report_title"] = report_title

    logger.info("Writing arena results to %s...", output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    logger.info("Arena results written (YAML)")

    if report_file:
        _generate_arena_report(output_file, report_file, include_input=include_input)

    if print_table:
        _print_arena_rankings(output_data["rankings"])

    return 0


def _print_evaluation_summary(evaluated: int, skipped: int, failed: list[str]) -> None:
    """Print summary of arena evaluation."""
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("   Evaluated: %d", evaluated)
    logger.info("   Skipped (cached): %d", skipped)
    logger.info("   Failed: %d", len(failed))
    if failed:
        logger.info("   Failed IDs: %s", ", ".join(failed))
        logger.info("   (Fix the issues and re-run to complete these evaluations)")


def _build_arena_output(
    arena_name: str,
    arena_spec: ArenaSpec,
    contestants_results: dict[str, Any],
    rankings: list[dict[str, Any]],
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    arena_spec_file: str,
    report_title: str | None,
    failed_contestants: list[str],
) -> dict[str, Any]:
    """Build the final arena output data structure."""
    output_data = {
        "mode": "arena",
        "arena_name": arena_name,
        "arena_description": arena_spec.description,
        "contestants": contestants_results,
        "rankings": rankings,
        "rubric": converters.rubric_to_portable_dict(base_rubric),
        "judge_panel": converters.panel_config_to_portable_dict(panel_config),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "arena_spec_file": arena_spec_file,
            "rubric_source_file": arena_spec.rubric_file,
            "judge_panel_source_file": arena_spec.judges_panel_file,
        },
    }

    if failed_contestants:
        output_data["metadata"]["partial"] = True
        output_data["metadata"]["failed_contestants"] = failed_contestants

    if report_title:
        output_data["metadata"]["report_title"] = report_title

    return output_data


def _generate_arena_report(
    output_file: str,
    report_file: str,
    pdf_exporter: Callable | None = None,
    *,
    include_input: bool = False,
) -> None:
    """Generate PDF report for arena results."""
    logger.info("Generating Arena PDF report to %s...", report_file)
    try:
        exporter = pdf_exporter or export_arena_pdf
        exporter(output_file, report_file, include_input=include_input)
        logger.info("Arena PDF report generated")
    except Exception as e:
        logger.error("PDF generation failed: %s", e)


def _print_arena_rankings(rankings: list[dict[str, Any]]) -> None:
    """Print arena rankings to console."""
    logger.info("=" * 80)
    logger.info("ARENA RANKINGS")
    logger.info("=" * 80)

    for r in rankings:
        logger.info(
            "#%d: %s - %.1f%% (%d/%d)",
            r["rank"],
            r["name"],
            r["percentage"],
            r["total_score"],
            r["max_score"],
        )
