"""Arena evaluation module for comparing multiple contestants against a shared rubric."""

import os
import sys
import yaml
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from rubric_kit.schema import (
    ArenaSpec, ArenaContestant, Rubric, Dimension, Criterion,
    ToolCalls, ToolSpec, JudgePanelConfig
)
from rubric_kit.validator import load_rubric, load_judge_panel_config, substitute_variables
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.pdf_export import export_arena_pdf
from rubric_kit import converters


# Default evaluator functions (can be overridden for testing via dependency injection)
def _default_evaluate_panel(rubric, input_file, panel_config):
    """Default implementation that imports at call time for proper mock support."""
    from rubric_kit.llm_judge import evaluate_rubric_with_panel
    return evaluate_rubric_with_panel(rubric, input_file, panel_config)


def _default_evaluate_panel_qa(rubric, input_file, panel_config):
    """Default implementation that imports at call time for proper mock support."""
    from rubric_kit.llm_judge import evaluate_rubric_with_panel_from_qa
    return evaluate_rubric_with_panel_from_qa(rubric, input_file, panel_config)


def load_arena_spec(arena_spec_file: str) -> ArenaSpec:
    """Load and validate an arena specification file."""
    if not os.path.exists(arena_spec_file):
        raise FileNotFoundError(f"Arena spec file not found: {arena_spec_file}")
    
    with open(arena_spec_file, 'r', encoding='utf-8') as f:
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
        contestants=contestants
    )


def load_contestant_variables(contestant: ArenaContestant, base_dir: str) -> Optional[Dict[str, str]]:
    """Load variables for a contestant from inline definition or external file."""
    if contestant.variables:
        return contestant.variables
    
    if not contestant.variables_file:
        return None
    
    variables_path = os.path.join(base_dir, contestant.variables_file)
    if not os.path.exists(variables_path):
        raise FileNotFoundError(f"Variables file not found: {variables_path}")
    
    with open(variables_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data.get("variables", data)


def apply_variables_to_rubric(base_rubric: Rubric, variables: Dict[str, str]) -> Rubric:
    """Create a new Rubric with variable substitution applied to criterion text and tool params."""
    substituted_criteria = []
    for crit in base_rubric.criteria:
        new_criterion_text = substitute_variables(crit.criterion, variables)
        new_tool_calls = _substitute_tool_calls(crit.tool_calls, variables) if crit.tool_calls else None
        
        substituted_criteria.append(Criterion(
            name=crit.name,
            category=crit.category,
            weight=crit.weight,
            dimension=crit.dimension,
            criterion=new_criterion_text,
            tool_calls=new_tool_calls
        ))
    
    substituted_dimensions = [
        Dimension(
            name=dim.name,
            description=substitute_variables(dim.description, variables),
            grading_type=dim.grading_type,
            scores=dim.scores,
            pass_above=dim.pass_above
        )
        for dim in base_rubric.dimensions
    ]
    
    return Rubric(
        dimensions=substituted_dimensions,
        criteria=substituted_criteria,
        variables=variables
    )


def _substitute_tool_calls(tool_calls: ToolCalls, variables: Dict[str, str]) -> ToolCalls:
    """Apply variable substitution to tool call parameters."""
    def substitute_params(tc: ToolSpec) -> ToolSpec:
        if tc.params is None:
            return tc
        new_params = {
            k: substitute_variables(v, variables) if isinstance(v, str) else v
            for k, v in tc.params.items()
        }
        return ToolSpec(name=tc.name, min_calls=tc.min_calls, max_calls=tc.max_calls, params=new_params)
    
    return ToolCalls(
        respect_order=tool_calls.respect_order,
        params_strict_mode=tool_calls.params_strict_mode,
        required=[substitute_params(tc) for tc in tool_calls.required],
        optional=[substitute_params(tc) for tc in tool_calls.optional],
        prohibited=tool_calls.prohibited
    )


def combine_outputs_to_arena(output_files: List[str], arena_name: str = "Combined Arena") -> Dict[str, Any]:
    """Combine multiple evaluation output files into arena format."""
    contestants_results: Dict[str, Any] = {}
    shared_rubric = None
    shared_judge_panel = None
    
    for idx, output_file in enumerate(output_files):
        print(f"\n[{idx + 1}/{len(output_files)}] Loading: {output_file}")
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if data.get("mode") == "arena":
            raise ValueError(f"File is already an arena result: {output_file}")
        
        if not data.get("results"):
            raise ValueError(f"File missing 'results' section: {output_file}")
        
        basename = os.path.splitext(os.path.basename(output_file))[0]
        contestant_id = basename.replace("output_", "").replace("_", "-")
        metadata = data.get("metadata", {})
        contestant_name = metadata.get("report_title", basename)
        input_info = data.get("input", {})
        summary = data.get("summary", {})
        
        print(f"   ID: {contestant_id}")
        print(f"   Name: {contestant_name}")
        print(f"   Score: {summary.get('total_score', 0)}/{summary.get('max_score', 0)} ({summary.get('percentage', 0):.1f}%)")
        
        contestants_results[contestant_id] = {
            "name": contestant_name,
            "description": f"Loaded from {output_file}",
            "metadata": {
                "source_file": output_file,
                "original_timestamp": metadata.get("timestamp"),
                "rubric_source": metadata.get("rubric_source_file"),
                "judge_panel_source": metadata.get("judge_panel_source_file")
            },
            "input": input_info,
            "results": data.get("results", []),
            "summary": summary
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
            "combined_from_outputs": True
        }
    }


def _generate_rankings(contestants_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate sorted rankings from contestant results."""
    rankings = sorted(
        [
            {
                "id": cid,
                "name": cdata["name"],
                "percentage": cdata["summary"].get("percentage", 0),
                "total_score": cdata["summary"].get("total_score", 0),
                "max_score": cdata["summary"].get("max_score", 0)
            }
            for cid, cdata in contestants_results.items()
        ],
        key=lambda x: x["percentage"],
        reverse=True
    )
    
    for idx, r in enumerate(rankings, 1):
        r["rank"] = idx
    
    return rankings


def _save_partial_arena_results(
    output_file: str,
    arena_name: str,
    arena_spec: ArenaSpec,
    contestants_results: Dict[str, Any],
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    report_title: Optional[str] = None
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
            "partial": True
        }
    }
    
    if report_title:
        output_data["metadata"]["report_title"] = report_title
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def _evaluate_contestant(
    contestant: ArenaContestant,
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    base_dir: str,
    evaluate_panel: Optional[Callable] = None,
    evaluate_panel_qa: Optional[Callable] = None
) -> Dict[str, Any]:
    """Evaluate a single contestant and return results."""
    evaluate_panel = evaluate_panel or _default_evaluate_panel
    evaluate_panel_qa = evaluate_panel_qa or _default_evaluate_panel_qa
    
    contestant_vars = load_contestant_variables(contestant, base_dir)
    
    if contestant_vars:
        rubric = apply_variables_to_rubric(base_rubric, contestant_vars)
        print(f"   Variables: {len(contestant_vars)}")
    else:
        rubric = base_rubric
    
    input_path = os.path.join(base_dir, contestant.input_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"   Input: {contestant.input_type} from {contestant.input_file}")
    
    if contestant.input_type == "qna":
        evaluations = evaluate_panel_qa(rubric, input_path, panel_config)
    else:
        evaluations = evaluate_panel(rubric, input_path, panel_config)
    
    results = evaluate_rubric(rubric, evaluations)
    total_score, max_score = calculate_total_score(results)
    percentage = calculate_percentage_score(results)
    
    print(f"   ✓ Score: {total_score}/{max_score} ({percentage:.1f}%)")
    
    return {
        "name": contestant.name,
        "description": contestant.description,
        "metadata": contestant.metadata,
        "input": {
            "type": contestant.input_type,
            "source_file": contestant.input_file
        },
        "results": results,
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1)
        }
    }


def _load_cached_results(output_file: str, force: bool) -> Dict[str, Any]:
    """Load existing results from output file if available."""
    if not os.path.exists(output_file) or force:
        if force:
            print("\n🔄 Force mode: will re-evaluate all contestants")
        return {}
    
    print(f"\n📂 Found existing results in {output_file}")
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = yaml.safe_load(f)
        if existing_data and existing_data.get("mode") == "arena":
            existing_results = existing_data.get("contestants", {})
            print(f"   ✓ Loaded {len(existing_results)} cached contestant results")
            print("   (Use --force to re-evaluate all)")
            return existing_results
    except Exception as e:
        print(f"   ⚠ Could not load existing results: {e}")
    
    return {}


def run_arena_from_spec(
    arena_spec_file: str,
    output_file: str,
    report_file: Optional[str] = None,
    report_title: Optional[str] = None,
    force: bool = False,
    print_table: bool = True,
    evaluate_panel: Optional[Callable] = None,
    evaluate_panel_qa: Optional[Callable] = None,
    pdf_exporter: Optional[Callable] = None
) -> int:
    """Run arena evaluation from specification file."""
    print(f"Loading arena specification from {arena_spec_file}...")
    arena_spec = load_arena_spec(arena_spec_file)
    arena_name = arena_spec.name or "Arena Evaluation"
    print(f"✓ Loaded arena: {arena_name}")
    print(f"   Contestants: {len(arena_spec.contestants)}")
    
    existing_results = _load_cached_results(output_file, force)
    
    base_dir = os.path.dirname(os.path.abspath(arena_spec_file))
    
    rubric_path = os.path.join(base_dir, arena_spec.rubric_file)
    print(f"\nLoading shared rubric from {rubric_path}...")
    base_rubric = load_rubric(rubric_path, require_variables=False)
    print(f"✓ Loaded {len(base_rubric.dimensions)} dimensions and {len(base_rubric.criteria)} criteria")
    
    panel_path = os.path.join(base_dir, arena_spec.judges_panel_file)
    print(f"\nLoading judge panel from {panel_path}...")
    panel_config = load_judge_panel_config(panel_path)
    print(f"✓ Loaded panel with {len(panel_config.judges)} judge(s)")
    _print_evaluation_config(panel_config)
    
    contestants_results: Dict[str, Any] = dict(existing_results)
    failed_contestants: List[str] = []
    skipped_count = 0
    evaluated_count = 0
    
    print(f"\n{'='*80}")
    print("ARENA EVALUATION")
    print(f"{'='*80}")
    
    for idx, contestant in enumerate(arena_spec.contestants, 1):
        if contestant.id in existing_results:
            cached = existing_results[contestant.id]
            cached_pct = cached.get("summary", {}).get("percentage", 0)
            print(f"\n[{idx}/{len(arena_spec.contestants)}] {contestant.name} (id: {contestant.id})")
            print(f"   ⏭️  Skipped (cached: {cached_pct:.1f}%)")
            skipped_count += 1
            continue
        
        print(f"\n[{idx}/{len(arena_spec.contestants)}] Evaluating: {contestant.name} (id: {contestant.id})")
        
        try:
            contestants_results[contestant.id] = _evaluate_contestant(
                contestant, base_rubric, panel_config, base_dir,
                evaluate_panel=evaluate_panel, evaluate_panel_qa=evaluate_panel_qa
            )
            evaluated_count += 1
            
            _save_partial_arena_results(
                output_file, arena_name, arena_spec, contestants_results,
                base_rubric, panel_config, report_title
            )
        except Exception as e:
            print(f"   ❌ Failed: {e}", file=sys.stderr)
            failed_contestants.append(contestant.id)
    
    _print_evaluation_summary(evaluated_count, skipped_count, failed_contestants)
    
    rankings = _generate_rankings(contestants_results)
    output_data = _build_arena_output(
        arena_name, arena_spec, contestants_results, rankings,
        base_rubric, panel_config, arena_spec_file, report_title, failed_contestants
    )
    
    print(f"\nWriting arena results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    
    status = f" - {len(failed_contestants)} contestant(s) pending" if failed_contestants else " - complete"
    print(f"✓ Arena results written (YAML){status}")
    
    if report_file:
        _generate_arena_report(output_file, report_file, pdf_exporter)
    
    if print_table:
        _print_arena_rankings(rankings)
    
    return 0


def run_arena_from_outputs(
    output_files: List[str],
    output_file: str,
    report_file: Optional[str] = None,
    report_title: Optional[str] = None,
    print_table: bool = True
) -> int:
    """Combine existing output files into arena format."""
    print(f"Combining {len(output_files)} evaluation outputs into arena format...")
    
    arena_name = report_title or "Combined Arena Evaluation"
    output_data = combine_outputs_to_arena(output_files, arena_name)
    
    if report_title:
        output_data["metadata"]["report_title"] = report_title
    
    print(f"\nWriting arena results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"✓ Arena results written (YAML)")
    
    if report_file:
        _generate_arena_report(output_file, report_file)
    
    if print_table:
        _print_arena_rankings(output_data["rankings"])
    
    return 0


def _print_evaluation_config(panel_config: JudgePanelConfig) -> None:
    """Print evaluation configuration details."""
    print(f"   Execution mode: {panel_config.execution.mode}")
    print(f"   Consensus mode: {panel_config.consensus.mode}")
    if panel_config.consensus.mode in ("quorum", "majority"):
        print(f"   Consensus threshold: {panel_config.consensus.threshold}")


def _print_evaluation_summary(evaluated: int, skipped: int, failed: List[str]) -> None:
    """Print summary of arena evaluation."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"   Evaluated: {evaluated}")
    print(f"   Skipped (cached): {skipped}")
    print(f"   Failed: {len(failed)}")
    if failed:
        print(f"   Failed IDs: {', '.join(failed)}")
        print("   (Fix the issues and re-run to complete these evaluations)")


def _build_arena_output(
    arena_name: str,
    arena_spec: ArenaSpec,
    contestants_results: Dict[str, Any],
    rankings: List[Dict[str, Any]],
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    arena_spec_file: str,
    report_title: Optional[str],
    failed_contestants: List[str]
) -> Dict[str, Any]:
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
            "judge_panel_source_file": arena_spec.judges_panel_file
        }
    }
    
    if failed_contestants:
        output_data["metadata"]["partial"] = True
        output_data["metadata"]["failed_contestants"] = failed_contestants
    
    if report_title:
        output_data["metadata"]["report_title"] = report_title
    
    return output_data


def _generate_arena_report(output_file: str, report_file: str, pdf_exporter: Optional[Callable] = None) -> None:
    """Generate PDF report for arena results."""
    print(f"\nGenerating Arena PDF report to {report_file}...")
    try:
        exporter = pdf_exporter or export_arena_pdf
        exporter(output_file, report_file)
        print(f"✓ Arena PDF report generated")
    except Exception as e:
        print(f"⚠ PDF generation failed: {e}", file=sys.stderr)


def _print_arena_rankings(rankings: List[Dict[str, Any]]) -> None:
    """Print arena rankings to console."""
    print(f"\n{'='*80}")
    print("ARENA RANKINGS")
    print(f"{'='*80}\n")
    
    for r in rankings:
        medal = "🥇" if r["rank"] == 1 else ("🥈" if r["rank"] == 2 else ("🥉" if r["rank"] == 3 else "  "))
        print(f"{medal} #{r['rank']}: {r['name']} - {r['percentage']:.1f}% ({r['total_score']}/{r['max_score']})")
    print()

