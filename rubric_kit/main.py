"""Main CLI entry point for rubric-kit."""

import argparse
import os
import sys
import tempfile
import traceback
import yaml
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, Any, Optional, Tuple

from rubric_kit.validator import load_rubric, load_judge_panel_config, RubricValidationError
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.output import print_table, print_tool_breakdowns, convert_yaml_to_csv, convert_yaml_to_json
from rubric_kit.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.generator import RubricGenerator, parse_qa_input, parse_chat_session, parse_dimensions_file
from rubric_kit.pdf_export import export_evaluation_pdf, export_arena_pdf
from rubric_kit.schema import (
    JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig, Rubric
)
from rubric_kit import converters
from rubric_kit.cli import create_parser
from rubric_kit.arena import run_arena_from_spec, run_arena_from_outputs
from rubric_kit.metrics import MetricsAggregator


# =============================================================================
# Error Handling
# =============================================================================

def handle_command_errors(func: Callable) -> Callable:
    """Decorator to handle common errors for command functions."""
    @wraps(func)
    def wrapper(args) -> int:
        try:
            return func(args)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RubricValidationError as e:
            print(f"Rubric validation error: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            traceback.print_exc()
            return 1
    return wrapper


# =============================================================================
# Helper Functions
# =============================================================================

def create_default_panel_config(args) -> JudgePanelConfig:
    """Create a default single-judge panel configuration.
    
    Uses LiteLLM for provider auto-detection. API keys are read from
    environment variables based on the model's provider.
    """
    return JudgePanelConfig(
        judges=[JudgeConfig(
            name="default",
            model=args.model,
            base_url=args.base_url
        )],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )


def create_generator(args, metrics=None) -> RubricGenerator:
    """Create and return a RubricGenerator instance."""
    return RubricGenerator(model=args.model, base_url=args.base_url, metrics=metrics)


def ensure_yaml_extension(output_file: str) -> str:
    """Ensure the output file has a .yaml extension."""
    base, ext = os.path.splitext(output_file)
    if ext.lower() not in ('.yaml', '.yml'):
        return f"{output_file}.yaml"
    return output_file


def resolve_text_from_args(
    inline_value: Optional[str],
    file_path: Optional[str],
    arg_name: str
) -> Optional[str]:
    """Resolve text content from either inline argument or file."""
    if inline_value:
        return inline_value
    
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{arg_name.capitalize()} file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    return None


def read_input_content(input_file: str) -> str:
    """Read and return input file content."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def _build_input_data(input_type: str, input_file: str, raw_content: Optional[str] = None) -> Dict[str, Any]:
    """
    Build structured input data for output YAML.
    
    For QnA: parses YAML and stores question, answer, context as separate keys.
    For chat_session: stores raw content under chat_session key.
    
    Args:
        input_type: Either "qna" or "chat_session"
        input_file: Path to the input file
        raw_content: Optional pre-read content (for rerun with embedded content)
        
    Returns:
        Structured input data dictionary
    """
    content = raw_content if raw_content else read_input_content(input_file)
    
    input_data = {
        "type": input_type,
        "source_file": input_file
    }
    
    if input_type == "qna":
        # Parse QnA YAML and store as structured data
        try:
            qa_data = yaml.safe_load(content)
            if isinstance(qa_data, dict):
                if "question" in qa_data:
                    input_data["question"] = qa_data["question"]
                if "answer" in qa_data:
                    input_data["answer"] = qa_data["answer"]
                if "context" in qa_data:
                    input_data["context"] = qa_data["context"]
        except yaml.YAMLError:
            # If parsing fails, store raw content
            input_data["raw_content"] = content
    else:
        # Chat session - store raw content
        input_data["chat_session"] = content
    
    return input_data


def print_evaluation_config(panel_config: JudgePanelConfig) -> None:
    """Print evaluation configuration details."""
    print(f"   Execution mode: {panel_config.execution.mode}")
    print(f"   Consensus mode: {panel_config.consensus.mode}")
    if panel_config.consensus.mode in ("quorum", "majority"):
        print(f"   Consensus threshold: {panel_config.consensus.threshold}")


def print_rubric_summary(rubric: Rubric, title: str) -> None:
    """Print a summary of the rubric."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    if rubric.variables:
        print(f"\nVariables ({len(rubric.variables)}):")
        for var_name, var_value in rubric.variables.items():
            display_value = var_value if len(var_value) <= 40 else var_value[:37] + "..."
            print(f"  • {var_name}: {display_value}")
    
    print(f"\nDimensions ({len(rubric.dimensions)}):")
    for dim in rubric.dimensions:
        print(f"  • {dim.name} ({dim.grading_type})")
    
    print(f"\nCriteria ({len(rubric.criteria)}):")
    for crit in rubric.criteria:
        print(f"  • {crit.name} [{crit.category}] - {crit.dimension}")
    print()


def write_rubric_to_file(
    rubric: Rubric, 
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Write a rubric to a YAML file (always self-contained with variables).
    
    Args:
        rubric: The rubric to write
        output_path: Path to the output file
        metadata: Optional metadata dict to include (timestamp, operation, metrics, etc.)
    """
    rubric_dict = converters.rubric_to_dict(rubric)
    
    # Add metadata if provided
    if metadata:
        rubric_dict["metadata"] = metadata
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(rubric_dict, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


# =============================================================================
# Command: evaluate
# =============================================================================

@handle_command_errors
def cmd_evaluate(args) -> int:
    """Execute the 'evaluate' subcommand."""
    # Check for dry-run mode
    if getattr(args, 'dry_run', False):
        return _cmd_evaluate_dry_run(args)
    
    # Load rubric
    print(f"Loading rubric from {args.rubric_file}...")
    variables_file = getattr(args, 'variables_file', None)
    if variables_file:
        print(f"   Using variables from: {variables_file}")
    rubric = load_rubric(args.rubric_file, variables_file=variables_file)
    print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    
    # Load or create judge panel configuration
    if args.judge_panel_config:
        print(f"\nLoading judge panel configuration from {args.judge_panel_config}...")
        panel_config = load_judge_panel_config(args.judge_panel_config)
        print(f"✓ Loaded panel with {len(panel_config.judges)} judge(s)")
    else:
        panel_config = create_default_panel_config(args)
        print(f"\n🤖 Using single judge: {args.model}")
    
    # Create metrics aggregator unless disabled
    metrics = None
    if not getattr(args, 'no_metrics', False):
        include_call_log = getattr(args, 'include_call_log', False)
        metrics = MetricsAggregator(include_call_log=include_call_log)
    
    # Determine input file and type
    input_file, input_type = _get_input_file_and_type(args)
    
    # Run evaluation
    print(f"\nEvaluating {input_type.replace('_', ' ')} from {input_file}...")
    print_evaluation_config(panel_config)
    
    evaluations = _run_evaluation(rubric, input_file, input_type, panel_config, metrics=metrics)
    print(f"✓ Evaluated {len(evaluations)} criteria")
    
    # Process scores
    results, total_score, max_score, percentage = _process_evaluation_results(rubric, evaluations)
    
    # Build and write output
    output_data = _build_evaluate_output(
        args, rubric, panel_config, results, total_score, max_score, percentage,
        input_type, input_file, metrics=metrics
    )
    
    output_file = ensure_yaml_extension(args.output_file)
    _write_yaml_output(output_file, output_data)
    
    # Print metrics summary
    if metrics:
        _print_metrics_summary(metrics)
    
    # Generate PDF report if requested
    if args.report:
        _generate_pdf_report(output_file, args.report)
    
    # Print results table
    if not args.no_table:
        _print_results_table(results)
    
    return 0


def _cmd_evaluate_dry_run(args) -> int:
    """Execute evaluate command in dry-run mode (estimate costs only)."""
    from rubric_kit.metrics import estimate_tokens, estimate_cost
    from rubric_kit.prompts import EVALUATOR_CONFIG, build_binary_criterion_prompt
    
    print("DRY RUN MODE - Estimating costs without making LLM calls\n")
    
    # Load rubric
    print(f"Loading rubric from {args.rubric_file}...")
    variables_file = getattr(args, 'variables_file', None)
    rubric = load_rubric(args.rubric_file, variables_file=variables_file)
    print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    
    # Get judge models from panel config or use CLI default
    judge_models = _get_judge_models(args)
    
    # Estimate costs per model
    model_estimates = _estimate_costs_per_model(
        rubric, judge_models, EVALUATOR_CONFIG, 
        build_binary_criterion_prompt, estimate_tokens, estimate_cost
    )
    
    # Print results
    _print_dry_run_results(model_estimates, EVALUATOR_CONFIG.max_tokens)
    
    return 0


def _get_judge_models(args) -> list:
    """Get list of judge models from panel config or CLI args."""
    if args.judge_panel_config:
        panel_config = load_judge_panel_config(args.judge_panel_config)
        models = [judge.model for judge in panel_config.judges]
        print(f"✓ Loaded panel with {len(models)} judge(s):")
        for i, model in enumerate(models, 1):
            print(f"   Judge {i}: {model}")
        return models
    
    print(f"   Using single judge: {args.model}")
    return [args.model]


def _estimate_costs_per_model(rubric, judge_models, config, build_prompt_fn, 
                               estimate_tokens_fn, estimate_cost_fn) -> dict:
    """Estimate costs for each model across all criteria."""
    MINIMAL_TOKENS = 400  # Minimal evaluation response (pass/fail + brief reason)
    max_tokens = config.max_tokens
    conservative_tokens = int(max_tokens * 0.1)
    
    estimates = {}
    
    for criterion in rubric.criteria:
        prompt = build_prompt_fn(criterion, "[Sample chat content for estimation]")
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        for model in judge_models:
            if model not in estimates:
                estimates[model] = {
                    "calls": 0, "prompt_tokens": 0,
                    "cost_minimal": 0.0, "cost_conservative": 0.0, "cost_worst_case": 0.0
                }
            
            prompt_tokens = estimate_tokens_fn(model, messages)
            estimates[model]["calls"] += 1
            estimates[model]["prompt_tokens"] += prompt_tokens
            estimates[model]["cost_minimal"] += estimate_cost_fn(model, prompt_tokens, MINIMAL_TOKENS)
            estimates[model]["cost_conservative"] += estimate_cost_fn(model, prompt_tokens, conservative_tokens)
            estimates[model]["cost_worst_case"] += estimate_cost_fn(model, prompt_tokens, max_tokens)
    
    return estimates


def _print_dry_run_results(model_estimates: dict, max_tokens: int) -> None:
    """Print formatted dry-run cost estimation results."""
    MINIMAL_TOKENS = 400
    conservative_tokens = int(max_tokens * 0.1)
    
    # Calculate totals
    total_calls = sum(m["calls"] for m in model_estimates.values())
    total_prompt_tokens = sum(m["prompt_tokens"] for m in model_estimates.values())
    
    totals = {
        "calls": total_calls,
        "prompt_tokens": total_prompt_tokens,
        "minimal": sum(m["cost_minimal"] for m in model_estimates.values()),
        "conservative": sum(m["cost_conservative"] for m in model_estimates.values()),
        "worst_case": sum(m["cost_worst_case"] for m in model_estimates.values()),
    }
    
    # Calculate completion tokens for each scenario
    completion_minimal = total_calls * MINIMAL_TOKENS
    completion_conservative = total_calls * conservative_tokens
    completion_worst = total_calls * max_tokens
    
    # Print header
    print("\n" + "=" * 70)
    print("DRY-RUN COST ESTIMATE")
    print("=" * 70)
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Total LLM calls: {totals['calls']}")
    print(f"  Max output tokens/call (configured): {max_tokens:,}")
    
    # Per-model breakdown (before summary if multiple models)
    if len(model_estimates) > 1:
        print(f"\nPer-model breakdown:")
        for model, est in model_estimates.items():
            print(f"  {model}:")
            print(f"    Calls: {est['calls']}, Prompt tokens: ~{est['prompt_tokens']:,}")
            print(f"    Costs: ${est['cost_minimal']:.4f} (minimal) | "
                  f"${est['cost_conservative']:.4f} (conservative) | "
                  f"${est['cost_worst_case']:.4f} (worst)")
    
    # Summary table with token breakdown
    scenarios = [
        ("MINIMAL", MINIMAL_TOKENS, completion_minimal, totals["minimal"], 
         "minimal pass/fail + brief reason"),
        ("CONSERVATIVE", conservative_tokens, completion_conservative, totals["conservative"], 
         "10% of max, longer reasoning"),
        ("WORST CASE", max_tokens, completion_worst, totals["worst_case"], 
         "100% of max, theoretical"),
    ]
    
    print(f"\nCost Summary:")
    print(f"  {'Scenario':<13} {'Prompt':<12} {'Completion':<14} {'Total':<14} {'Cost':<10} Description")
    print(f"  {'-'*13} {'-'*12} {'-'*14} {'-'*14} {'-'*10} {'-'*24}")
    for name, comp_per_call, comp_total, cost, desc in scenarios:
        total_tokens = total_prompt_tokens + comp_total
        print(f"  {name:<13} ~{total_prompt_tokens:<11,} ~{comp_total:<13,} ~{total_tokens:<13,} ${cost:<9.4f} {desc}")
    
    print(f"\nNote: Actual costs depend on response lengths.")


def _print_metrics_summary(metrics: MetricsAggregator) -> None:
    """Print a summary of metrics to console."""
    summary = metrics.get_summary()
    print(f"\n📊 Metrics Summary:")
    print(f"   Total LLM calls: {summary.total_calls}")
    print(f"   Total tokens: {summary.total_tokens:,} (prompt: {summary.prompt_tokens:,}, completion: {summary.completion_tokens:,})")
    print(f"   Estimated cost: ${summary.cost_usd:.4f}")
    print(f"   Total time: {summary.latency_seconds:.1f}s")


def _get_input_file_and_type(args) -> Tuple[str, str]:
    """Extract input file and type from args."""
    if args.qna_file:
        return args.qna_file, "qna"
    return args.chat_session_file, "chat_session"


def _run_evaluation(rubric, input_file: str, input_type: str, panel_config, metrics=None):
    """Run the appropriate evaluation based on input type."""
    if input_type == "qna":
        return evaluate_rubric_with_panel_from_qa(rubric, input_file, panel_config, metrics=metrics)
    return evaluate_rubric_with_panel(rubric, input_file, panel_config, metrics=metrics)


def _process_evaluation_results(rubric, evaluations):
    """Process evaluation results and calculate scores."""
    print("\nProcessing scores...")
    results = evaluate_rubric(rubric, evaluations)
    total_score, max_score = calculate_total_score(results)
    percentage = calculate_percentage_score(results)
    print(f"✓ Evaluation complete: {total_score}/{max_score} ({percentage:.1f}%)")
    return results, total_score, max_score, percentage


def _build_evaluate_output(
    args, rubric, panel_config, results, total_score, max_score, percentage,
    input_type, input_file, metrics=None
) -> Dict[str, Any]:
    """Build the output data structure for evaluate command."""
    output_data = {
        "results": results,
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1)
        },
        "rubric": converters.rubric_to_portable_dict(rubric),
        "judge_panel": converters.panel_config_to_portable_dict(panel_config),
        "input": _build_input_data(input_type, input_file),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "rubric_source_file": args.rubric_file,
            "judge_panel_source_file": args.judge_panel_config
        }
    }
    
    if args.report_title:
        output_data["metadata"]["report_title"] = args.report_title
    
    # Add metrics if collected
    if metrics is not None:
        output_data["metrics"] = metrics.to_dict()
    
    return output_data


def _literal_str_representer(dumper, data):
    """Custom representer for multiline strings using literal block scalar style."""
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


def _write_yaml_output(output_file: str, output_data: Dict[str, Any]) -> None:
    """Write output data to YAML file with proper multiline string formatting."""
    print(f"\nWriting results to {output_file} (YAML)...")
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a custom dumper that uses literal block scalar for multiline strings
    class MultilineDumper(yaml.SafeDumper):
        pass
    
    MultilineDumper.add_representer(str, _literal_str_representer)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, Dumper=MultilineDumper, sort_keys=False, 
                  default_flow_style=False, allow_unicode=True, width=120)
    print(f"✓ YAML file written (self-contained)")


def _generate_pdf_report(output_file: str, report_path: str) -> None:
    """Generate PDF report from output file."""
    print(f"\nGenerating PDF report to {report_path}...")
    try:
        export_evaluation_pdf(output_file, report_path)
        print(f"✓ PDF report generated")
    except Exception as e:
        print(f"⚠ PDF generation failed: {e}", file=sys.stderr)


def _print_results_table(results) -> None:
    """Print results table to console."""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")
    print_table(results, include_summary=True)
    print_tool_breakdowns(results)
    print()


# =============================================================================
# Command: generate
# =============================================================================

@handle_command_errors
def cmd_generate(args) -> int:
    """Execute the 'generate' subcommand."""
    # Check for dry-run mode
    if getattr(args, 'dry_run', False):
        return _cmd_generate_dry_run(args)
    
    # Parse input
    input_obj, input_type = _load_generate_input(args)
    
    # Parse category hints
    category_hints = _parse_category_hints(args)
    
    # Resolve guidelines
    guidelines = resolve_text_from_args(
        getattr(args, 'guidelines', None),
        getattr(args, 'guidelines_file', None),
        'guidelines'
    )
    if guidelines:
        _print_guidelines_info(guidelines, args)
    
    # Create metrics aggregator unless disabled
    metrics = None
    if not getattr(args, 'no_metrics', False):
        metrics = MetricsAggregator()
    
    # Initialize generator
    print(f"\n🤖 Initializing rubric generator...")
    print(f"   Model: {args.model}")
    generator = create_generator(args, metrics=metrics)
    
    # Parse dimension and criteria counts
    num_dimensions, num_criteria = _parse_dimension_criteria_counts(args)
    
    # Load dimensions file if provided
    provided_dimensions = _load_dimensions_file(args)
    
    # Generate rubric
    _print_generation_progress(num_dimensions, num_criteria)
    use_variables = not getattr(args, 'no_variables', False)
    if not use_variables:
        print("   Mode: No variables (hard-coded values)")
    
    rubric = _generate_rubric(
        generator, input_obj, input_type, num_dimensions, num_criteria,
        category_hints, provided_dimensions, use_variables, guidelines
    )
    
    _print_generation_result(rubric, use_variables)
    
    # Print metrics summary
    if metrics:
        _print_metrics_summary(metrics)
    
    # Build metadata
    metadata = _build_generate_metadata(args, input_obj, input_type, num_dimensions, num_criteria, use_variables, metrics)
    
    # Write rubric to file
    print(f"\nWriting rubric to {args.output_file}...")
    write_rubric_to_file(rubric, args.output_file, metadata=metadata)
    print(f"✓ Rubric written successfully")
    
    print_rubric_summary(rubric, "GENERATED RUBRIC SUMMARY")
    return 0


def _build_generate_metadata(
    args,
    input_obj,
    input_type: str,
    num_dimensions: Optional[int],
    num_criteria: Optional[int],
    use_variables: bool,
    metrics: Optional["MetricsAggregator"]
) -> Dict[str, Any]:
    """Build metadata dict for generated rubric.
    
    Args:
        args: CLI arguments
        input_obj: Parsed input (QAInput or ChatSessionInput)
        input_type: "qa" or "chat"
        num_dimensions: Number of dimensions (or None for auto)
        num_criteria: Number of criteria (or None for auto)
        use_variables: Whether variables were used
        metrics: Optional MetricsAggregator
        
    Returns:
        Metadata dictionary
    """
    source_file = args.qna_file if args.qna_file else args.chat_session_file
    source_type = "qna" if input_type == "qa" else "chat_session"
    
    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "operation": "generate",
        "model": args.model,
        "source_file": source_file,
        "source_type": source_type,
        "options": {
            "num_dimensions": num_dimensions,
            "num_criteria": num_criteria,
            "use_variables": use_variables,
        }
    }
    
    # Add guidelines info if provided
    guidelines = resolve_text_from_args(
        getattr(args, 'guidelines', None),
        getattr(args, 'guidelines_file', None),
        'guidelines'
    )
    if guidelines:
        metadata["options"]["has_guidelines"] = True
    
    # Add dimensions file info if provided
    if getattr(args, 'dimensions_file', None):
        metadata["options"]["dimensions_file"] = args.dimensions_file
    
    # Add category hints if provided
    if getattr(args, 'categories', None):
        metadata["options"]["categories"] = args.categories
    
    # Add metrics if collected
    if metrics is not None:
        metadata["metrics"] = metrics.to_dict()
    
    return metadata


def _cmd_generate_dry_run(args) -> int:
    """Execute generate command in dry-run mode (estimate costs only)."""
    from rubric_kit.metrics import estimate_tokens, estimate_cost
    from rubric_kit.prompts import GENERATOR_CONFIG
    
    print("DRY RUN MODE - Estimating costs without making LLM calls\n")
    
    # Parse input
    input_obj, input_type = _load_generate_input(args)
    
    # Get model
    model = args.model
    print(f"\n   Model: {model}")
    
    # Build prompts and estimate costs
    prompt_estimates = _estimate_generate_prompts(args, input_obj, input_type, GENERATOR_CONFIG)
    
    # Print results
    _print_generate_dry_run_results(model, prompt_estimates, GENERATOR_CONFIG.max_tokens)
    
    return 0


def _estimate_generate_prompts(args, input_obj, input_type, config) -> list:
    """Build prompts for generation and return list of (call_type, messages) tuples."""
    from rubric_kit.prompts import (
        build_dimension_generation_prompt,
        build_criteria_generation_prompt,
        build_chat_dimension_generation_prompt,
        build_chat_criteria_generation_prompt
    )
    from rubric_kit.schema import Dimension
    
    prompts = []
    
    if input_type == "qa":
        # Q&A based generation
        if not getattr(args, 'dimensions_file', None):
            sample_prompt = build_dimension_generation_prompt(
                question=input_obj.question,
                answer=input_obj.answer,
                num_dimensions=None,
                context=input_obj.context
            )
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": sample_prompt}
            ]
            prompts.append(("generate_dimensions", messages))
        
        # Sample dimension for criteria estimation
        sample_dims = [Dimension(name="sample", description="sample", grading_type="binary")]
        sample_prompt = build_criteria_generation_prompt(
            question=input_obj.question,
            answer=input_obj.answer,
            dimensions=sample_dims,
            num_criteria=None,
            context=input_obj.context
        )
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": sample_prompt}
        ]
        prompts.append(("generate_criteria", messages))
    else:
        # Chat based generation
        if not getattr(args, 'dimensions_file', None):
            sample_prompt = build_chat_dimension_generation_prompt(
                chat_content=input_obj.content,
                num_dimensions=None,
                context=input_obj.context
            )
            messages = [
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": sample_prompt}
            ]
            prompts.append(("generate_dimensions", messages))
        
        # Sample dimension for criteria estimation
        sample_dims = [Dimension(name="sample", description="sample", grading_type="binary")]
        sample_prompt = build_chat_criteria_generation_prompt(
            chat_content=input_obj.content,
            dimensions=sample_dims,
            num_criteria=None,
            context=input_obj.context
        )
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": sample_prompt}
        ]
        prompts.append(("generate_criteria", messages))
    
    return prompts


def _print_generate_dry_run_results(model: str, prompts: list, max_tokens: int) -> None:
    """Print formatted dry-run cost estimation for generate command."""
    from rubric_kit.metrics import estimate_tokens, estimate_cost
    
    MINIMAL_TOKENS = 2500  # Generation outputs structured JSON (dimensions + criteria + variables)
    conservative_tokens = int(max_tokens * 0.1)
    
    # Calculate prompt tokens for each call
    total_calls = len(prompts)
    total_prompt_tokens = 0
    call_details = []
    
    for call_type, messages in prompts:
        prompt_tokens = estimate_tokens(model, messages)
        total_prompt_tokens += prompt_tokens
        call_details.append((call_type, prompt_tokens))
    
    # Calculate costs for each scenario
    completion_minimal = total_calls * MINIMAL_TOKENS
    completion_conservative = total_calls * conservative_tokens
    completion_worst = total_calls * max_tokens
    
    cost_minimal = estimate_cost(model, total_prompt_tokens, completion_minimal)
    cost_conservative = estimate_cost(model, total_prompt_tokens, completion_conservative)
    cost_worst = estimate_cost(model, total_prompt_tokens, completion_worst)
    
    # Print header
    print("\n" + "=" * 70)
    print("DRY-RUN COST ESTIMATE")
    print("=" * 70)
    
    # Configuration
    print(f"\nConfiguration:")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Max output tokens/call (configured): {max_tokens:,}")
    
    # Per-call breakdown
    print(f"\nPer-call breakdown:")
    for call_type, prompt_tokens in call_details:
        print(f"  {call_type}: ~{prompt_tokens:,} prompt tokens")
    
    # Summary table with token breakdown
    scenarios = [
        ("MINIMAL", MINIMAL_TOKENS, completion_minimal, cost_minimal, 
         "minimal structured output"),
        ("CONSERVATIVE", conservative_tokens, completion_conservative, cost_conservative, 
         "10% of max, detailed output"),
        ("WORST CASE", max_tokens, completion_worst, cost_worst, 
         "100% of max, theoretical"),
    ]
    
    print(f"\nCost Summary:")
    print(f"  {'Scenario':<13} {'Prompt':<12} {'Completion':<14} {'Total':<14} {'Cost':<10} Description")
    print(f"  {'-'*13} {'-'*12} {'-'*14} {'-'*14} {'-'*10} {'-'*22}")
    for name, comp_per_call, comp_total, cost, desc in scenarios:
        total_tokens = total_prompt_tokens + comp_total
        print(f"  {name:<13} ~{total_prompt_tokens:<11,} ~{comp_total:<13,} ~{total_tokens:<13,} ${cost:<9.4f} {desc}")
    
    print(f"\nNote: Actual costs depend on response lengths.")


def _load_generate_input(args):
    """Load input for generate command."""
    if args.qna_file:
        print(f"Loading Q&A from {args.qna_file}...")
        input_obj = parse_qa_input(args.qna_file)
        print(f"✓ Loaded Q&A pair")
        print(f"   Q: {input_obj.question[:80]}{'...' if len(input_obj.question) > 80 else ''}")
        return input_obj, "qa"
    
    print(f"Loading chat session from {args.chat_session_file}...")
    input_obj = parse_chat_session(args.chat_session_file)
    print(f"✓ Loaded chat session")
    print(f"   Content length: {len(input_obj.content)} characters")
    print(f"   The LLM will analyze the session to detect tool calls and structure")
    return input_obj, "chat"


def _parse_category_hints(args):
    """Parse category hints from args."""
    if not args.categories:
        return None
    category_hints = [c.strip() for c in args.categories.split(',')]
    print(f"   Category hints: {', '.join(category_hints)}")
    return category_hints


def _print_guidelines_info(guidelines: str, args) -> None:
    """Print guidelines information."""
    guidelines_preview = guidelines[:60] + "..." if len(guidelines) > 60 else guidelines
    source = "(from file)" if getattr(args, 'guidelines_file', None) else ""
    print(f"   Guidelines{source}: {guidelines_preview}")


def _parse_dimension_criteria_counts(args) -> Tuple[Optional[int], Optional[int]]:
    """Parse dimension and criteria counts, supporting 'auto' keyword."""
    num_dimensions = None if args.num_dimensions == "auto" else int(args.num_dimensions)
    num_criteria = None if args.num_criteria == "auto" else int(args.num_criteria)
    return num_dimensions, num_criteria


def _load_dimensions_file(args):
    """Load dimensions from file if provided."""
    if not args.dimensions_file:
        return None
    print(f"\nLoading dimensions from {args.dimensions_file}...")
    dimensions = parse_dimensions_file(args.dimensions_file)
    print(f"✓ Loaded {len(dimensions)} dimensions (skipping dimension generation)")
    return dimensions


def _print_generation_progress(num_dimensions: Optional[int], num_criteria: Optional[int]) -> None:
    """Print progress message for rubric generation."""
    if num_dimensions is None and num_criteria is None:
        print("\n🔄 Generating rubric with auto-detected dimensions and criteria...")
    elif num_dimensions is None:
        print(f"\n🔄 Generating rubric with auto-detected dimensions and {num_criteria} criteria...")
    elif num_criteria is None:
        print(f"\n🔄 Generating rubric with {num_dimensions} dimensions and auto-detected criteria...")
    else:
        print(f"\n🔄 Generating rubric with {num_dimensions} dimensions and {num_criteria} criteria...")
    print("   This may take a moment...")


def _generate_rubric(generator, input_obj, input_type, num_dimensions, num_criteria,
                     category_hints, dimensions, use_variables, guidelines):
    """Generate rubric based on input type."""
    if input_type == "chat":
        return generator.generate_rubric_from_chat(
            input_obj,
            num_dimensions=num_dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            dimensions=dimensions,
            use_variables=use_variables,
            guidelines=guidelines
        )
    return generator.generate_rubric(
        input_obj,
        num_dimensions=num_dimensions,
        num_criteria=num_criteria,
        category_hints=category_hints,
        dimensions=dimensions,
        use_variables=use_variables,
        guidelines=guidelines
    )


def _print_generation_result(rubric, use_variables: bool) -> None:
    """Print generation result summary."""
    print(f"✓ Generated {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    if rubric.variables:
        print(f"   Variables extracted: {len(rubric.variables)}")
    elif use_variables:
        print("   Note: No variables extracted from content")


# =============================================================================
# Command: refine
# =============================================================================

@handle_command_errors
def cmd_refine(args) -> int:
    """Execute the 'refine' subcommand."""
    # Load existing rubric
    print(f"Loading rubric from {args.rubric_file}...")
    variables_file = getattr(args, 'variables_file', None)
    if variables_file:
        print(f"   Using variables from: {variables_file}")
    rubric = load_rubric(args.rubric_file, variables_file=variables_file, require_variables=False)
    print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    
    if rubric.variables:
        print(f"   Variables: {len(rubric.variables)}")
    else:
        print("   Note: No variables defined - LLM will extract them from context")
    
    # Create metrics aggregator unless disabled
    metrics = None
    if not getattr(args, 'no_metrics', False):
        metrics = MetricsAggregator()
    
    # Initialize generator
    print(f"\n🤖 Initializing rubric refiner...")
    print(f"   Model: {args.model}")
    generator = create_generator(args, metrics=metrics)
    
    # Load optional input context
    input_obj, input_type = _load_refine_input(args)
    
    # Load dimensions file if provided
    dimensions_to_merge = _load_dimensions_file(args)
    
    # Resolve feedback
    feedback = resolve_text_from_args(
        getattr(args, 'feedback', None),
        getattr(args, 'feedback_file', None),
        'feedback'
    )
    
    # Refine rubric
    use_variables = not getattr(args, 'no_variables', False)
    _print_refine_progress(input_type, feedback, dimensions_to_merge, use_variables, args)
    
    refined_rubric = _refine_rubric(
        generator, rubric, input_obj, input_type, feedback,
        dimensions_to_merge, use_variables
    )
    
    _print_generation_result(refined_rubric, use_variables)
    
    # Print metrics summary
    if metrics:
        _print_metrics_summary(metrics)
    
    # Build metadata
    metadata = _build_refine_metadata(args, input_obj, input_type, use_variables, feedback, metrics)
    
    # Write output
    output_path = args.output_file if args.output_file else args.rubric_file
    print(f"\nWriting refined rubric to {output_path}...")
    write_rubric_to_file(refined_rubric, output_path, metadata=metadata)
    print(f"✓ Refined rubric written successfully")
    
    print_rubric_summary(refined_rubric, "REFINED RUBRIC SUMMARY")
    return 0


def _build_refine_metadata(
    args,
    input_obj,
    input_type: Optional[str],
    use_variables: bool,
    feedback: Optional[str],
    metrics: Optional["MetricsAggregator"]
) -> Dict[str, Any]:
    """Build metadata dict for refined rubric.
    
    Args:
        args: CLI arguments
        input_obj: Optional parsed input (QAInput or ChatSessionInput)
        input_type: "qa", "chat", or None
        use_variables: Whether variables were used
        feedback: Optional feedback text
        metrics: Optional MetricsAggregator
        
    Returns:
        Metadata dictionary
    """
    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "operation": "refine",
        "model": args.model,
        "source_rubric_file": args.rubric_file,
        "options": {
            "use_variables": use_variables,
            "has_feedback": feedback is not None,
        }
    }
    
    # Add context info if provided
    if input_type and input_obj:
        if input_type == "qa":
            metadata["context_file"] = args.qna_file
            metadata["context_type"] = "qna"
        elif input_type == "chat":
            metadata["context_file"] = args.chat_session_file
            metadata["context_type"] = "chat_session"
    
    # Add dimensions file info if provided
    if getattr(args, 'dimensions_file', None):
        metadata["options"]["dimensions_file"] = args.dimensions_file
    
    # Add metrics if collected
    if metrics is not None:
        metadata["metrics"] = metrics.to_dict()
    
    return metadata


def _load_refine_input(args):
    """Load optional input context for refine command."""
    if args.qna_file:
        print(f"\nLoading Q&A from {args.qna_file}...")
        input_obj = parse_qa_input(args.qna_file)
        print(f"✓ Loaded Q&A pair")
        print(f"   Q: {input_obj.question[:80]}{'...' if len(input_obj.question) > 80 else ''}")
        return input_obj, "qa"
    
    if args.chat_session_file:
        print(f"\nLoading chat session from {args.chat_session_file}...")
        input_obj = parse_chat_session(args.chat_session_file)
        print(f"✓ Loaded chat session")
        print(f"   Content length: {len(input_obj.content)} characters")
        return input_obj, "chat"
    
    return None, None


def _print_refine_progress(input_type, feedback, dimensions_to_merge, use_variables, args) -> None:
    """Print refine progress message."""
    parts = ["\n🔄 Refining rubric"]
    if input_type:
        parts.append(f" using {input_type} context")
    if feedback:
        parts.append(" with feedback")
    if dimensions_to_merge:
        parts.append(f" (merging {len(dimensions_to_merge)} dimensions)")
    if not use_variables:
        parts.append(" (no variables)")
    print("".join(parts) + "...")
    
    if feedback:
        feedback_preview = feedback[:60] + "..." if len(feedback) > 60 else feedback
        source = "(from file)" if getattr(args, 'feedback_file', None) else ""
        print(f"   Feedback{source}: {feedback_preview}")
    if not use_variables:
        print("   Mode: No variables (hard-coded values)")
    print("   This may take a moment...")


def _refine_rubric(generator, rubric, input_obj, input_type, feedback,
                   dimensions_to_merge, use_variables):
    """Refine rubric based on input type."""
    if input_type == "qa":
        return generator.refine_rubric_with_qa(
            rubric, input_obj, feedback=feedback,
            dimensions_to_merge=dimensions_to_merge, use_variables=use_variables
        )
    if input_type == "chat":
        return generator.refine_rubric_with_chat(
            rubric, input_obj, feedback=feedback,
            dimensions_to_merge=dimensions_to_merge, use_variables=use_variables
        )
    return generator.refine_rubric(
        rubric, feedback=feedback,
        dimensions_to_merge=dimensions_to_merge, use_variables=use_variables
    )


# =============================================================================
# Command: export
# =============================================================================

@handle_command_errors
def cmd_export(args) -> int:
    """Execute the 'export' subcommand to convert YAML to various formats."""
    print(f"Loading evaluation results from {args.input_file}...")
    
    format_type = args.format.lower()
    
    if format_type == 'pdf':
        print(f"Generating PDF report to {args.output_file}...")
        export_evaluation_pdf(args.input_file, args.output_file)
        print(f"✓ PDF report exported to {args.output_file}")
    elif format_type == 'csv':
        print(f"Converting to CSV: {args.output_file}...")
        convert_yaml_to_csv(args.input_file, args.output_file)
        print(f"✓ CSV file exported to {args.output_file}")
    elif format_type == 'json':
        print(f"Converting to JSON: {args.output_file}...")
        convert_yaml_to_json(args.input_file, args.output_file)
        print(f"✓ JSON file exported to {args.output_file}")
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    return 0


# =============================================================================
# Command: rerun
# =============================================================================

@handle_command_errors
def cmd_rerun(args) -> int:
    """Execute the 'rerun' subcommand - re-evaluate using settings from a previous output."""
    print(f"Loading evaluation configuration from {args.input_file}...")
    data = _load_self_contained_yaml(args.input_file)
    
    # Rebuild rubric and panel config
    rubric = converters.rebuild_rubric_from_dict(data["rubric"])
    print(f"✓ Rebuilt rubric: {len(rubric.dimensions)} dimensions, {len(rubric.criteria)} criteria")
    
    panel_config = converters.rebuild_panel_config_from_dict(data["judge_panel"])
    print(f"✓ Rebuilt judge panel: {len(panel_config.judges)} judge(s)")
    
    # Determine input source
    input_file, input_type, input_content = _resolve_rerun_input(args, data)
    
    # Evaluate
    print_evaluation_config(panel_config)
    evaluations = _run_rerun_evaluation(rubric, input_file, input_type, input_content, panel_config)
    print(f"✓ Evaluated {len(evaluations)} criteria")
    
    # Process scores
    results, total_score, max_score, percentage = _process_evaluation_results(rubric, evaluations)
    
    # Build and write output
    output_data = _build_rerun_output(
        args, data, results, total_score, max_score, percentage,
        input_type, input_file, input_content
    )
    
    output_file = ensure_yaml_extension(args.output_file)
    _write_yaml_output(output_file, output_data)
    
    if args.report:
        _generate_pdf_report(output_file, args.report)
    
    if not args.no_table:
        _print_results_table(results)
    
    return 0


def _load_self_contained_yaml(input_file: str) -> Dict[str, Any]:
    """Load a self-contained evaluation YAML file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if not data.get("rubric"):
        raise ValueError("Input file missing 'rubric' section - not a self-contained evaluation file")
    if not data.get("judge_panel"):
        raise ValueError("Input file missing 'judge_panel' section - not a self-contained evaluation file")
    
    return data


def _resolve_rerun_input(args, data) -> Tuple[Optional[str], str, Optional[str]]:
    """Resolve input source for rerun command."""
    input_data = data.get("input", {})
    input_type = input_data.get("type", "chat_session")
    
    if args.qna_file:
        print(f"\n📥 Using new Q&A input: {args.qna_file}")
        return args.qna_file, "qna", None
    
    if args.chat_session_file:
        print(f"\n📥 Using new chat session input: {args.chat_session_file}")
        return args.chat_session_file, "chat_session", None
    
    # Check for embedded content (new structured format or legacy)
    embedded_content = _extract_embedded_content(input_data, input_type)
    if embedded_content:
        print(f"\n📥 Using embedded input content from previous evaluation")
        return None, input_type, embedded_content
    
    if input_data.get("source_file") and os.path.exists(input_data["source_file"]):
        print(f"\n📥 Using original input file: {input_data['source_file']}")
        return input_data["source_file"], input_type, None
    
    raise ValueError(
        "No input available. The original input file is not accessible and no embedded content. "
        "Use --from-chat-session or --from-qna to provide new input."
    )


def _extract_embedded_content(input_data: Dict[str, Any], input_type: str) -> Optional[str]:
    """Extract embedded content from input data (handles both new and legacy formats)."""
    # New structured format
    if input_type == "qna":
        if "question" in input_data:
            # Reconstruct YAML content for QnA
            qa_dict = {}
            if "question" in input_data:
                qa_dict["question"] = input_data["question"]
            if "answer" in input_data:
                qa_dict["answer"] = input_data["answer"]
            if "context" in input_data:
                qa_dict["context"] = input_data["context"]
            if qa_dict:
                return yaml.dump(qa_dict, default_flow_style=False, allow_unicode=True)
    else:
        if "chat_session" in input_data:
            return input_data["chat_session"]
    
    # Legacy format
    if "content" in input_data:
        return input_data["content"]
    
    return None


def _run_rerun_evaluation(rubric, input_file, input_type, input_content, panel_config):
    """Run evaluation for rerun command."""
    if input_file:
        return _run_evaluation(rubric, input_file, input_type, panel_config)
    
    # Use embedded content - write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(input_content)
        temp_file = f.name
    
    try:
        return _run_evaluation(rubric, temp_file, input_type, panel_config)
    finally:
        os.unlink(temp_file)


def _build_rerun_output(args, original_data, results, total_score, max_score, percentage,
                        input_type, input_file, input_content) -> Dict[str, Any]:
    """Build output data structure for rerun command."""
    original_input = original_data.get("input", {})
    
    # Build input data
    if input_file:
        # New input file provided
        new_input_data = _build_input_data(input_type, input_file)
    elif input_content:
        # Using embedded content from original
        source_file = original_input.get("source_file", "")
        new_input_data = _build_input_data(input_type, source_file, raw_content=input_content)
    else:
        # Preserve original input structure
        new_input_data = original_input.copy()
    
    output_data = {
        "results": results,
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1)
        },
        "rubric": original_data["rubric"],
        "judge_panel": original_data["judge_panel"],
        "input": new_input_data,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "rerun_from": args.input_file,
            "rubric_source_file": original_data.get("metadata", {}).get("rubric_source_file"),
            "judge_panel_source_file": original_data.get("metadata", {}).get("judge_panel_source_file")
        }
    }
    
    # Handle report title
    if args.report_title:
        output_data["metadata"]["report_title"] = args.report_title
    elif original_data.get("metadata", {}).get("report_title"):
        output_data["metadata"]["report_title"] = original_data["metadata"]["report_title"]
    
    return output_data


# =============================================================================
# Command: arena
# =============================================================================

@handle_command_errors
def cmd_arena(args) -> int:
    """Execute the 'arena' subcommand for comparative evaluation."""
    output_file = ensure_yaml_extension(args.output_file)
    
    if args.output_files:
        return run_arena_from_outputs(
            output_files=args.output_files,
            output_file=output_file,
            report_file=args.report,
            report_title=args.report_title,
            print_table=not args.no_table
        )
    
    # Pass functions from this module's namespace for proper test mocking
    return run_arena_from_spec(
        arena_spec_file=args.arena_spec,
        output_file=output_file,
        report_file=args.report,
        report_title=args.report_title,
        force=getattr(args, 'force', False),
        print_table=not args.no_table,
        evaluate_panel=evaluate_rubric_with_panel,
        evaluate_panel_qa=evaluate_rubric_with_panel_from_qa,
        pdf_exporter=export_arena_pdf
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    """Main CLI entry point with subcommands."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 2
    
    return args.func(args)


# Command function mapping (set after parser creation)
def _setup_command_handlers(parser) -> None:
    """Set command handlers on subparsers."""
    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                handler = {
                    'evaluate': cmd_evaluate,
                    'generate': cmd_generate,
                    'refine': cmd_refine,
                    'export': cmd_export,
                    'rerun': cmd_rerun,
                    'arena': cmd_arena,
                }.get(name)
                if handler:
                    subparser.set_defaults(func=handler)


# Monkey-patch create_parser to add handlers
_original_create_parser = create_parser

def create_parser():
    parser = _original_create_parser()
    _setup_command_handlers(parser)
    return parser


if __name__ == "__main__":
    sys.exit(main())
