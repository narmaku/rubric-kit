"""Main CLI entry point for rubric-kit."""

import argparse
import sys
import os
import yaml
import traceback
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, Any, List, Optional, Tuple

from rubric_kit.validator import load_rubric, load_judge_panel_config, RubricValidationError, substitute_variables
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.output import write_yaml, print_table, print_tool_breakdowns, convert_yaml_to_csv, convert_yaml_to_json
from rubric_kit.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.generator import RubricGenerator, parse_qa_input, parse_chat_session, parse_dimensions_file
from rubric_kit.pdf_export import export_evaluation_pdf, export_arena_pdf
from rubric_kit.schema import (
    JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig,
    Rubric, Dimension, Criterion, ToolSpec, ArenaSpec, ArenaContestant
)


# ============================================================================
# Helper Functions
# ============================================================================

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


def get_api_key(args_api_key: Optional[str]) -> str:
    """Get API key from args or environment variable."""
    api_key = args_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key required. Set OPENAI_API_KEY environment variable or use --api-key"
        )
    return api_key


def create_default_panel_config(args) -> JudgePanelConfig:
    """Create a default single-judge panel configuration."""
    api_key = get_api_key(args.api_key)
    return JudgePanelConfig(
        judges=[JudgeConfig(
            name="default",
            model=args.model,
            api_key=api_key,
            base_url=args.base_url
        )],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )


def create_generator(args) -> RubricGenerator:
    """Create and return a RubricGenerator instance."""
    api_key = get_api_key(args.api_key)
    return RubricGenerator(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url
    )


def convert_tool_spec_to_dict(tool_spec: ToolSpec) -> Dict[str, Any]:
    """Convert a ToolSpec to dictionary format."""
    tool_dict: Dict[str, Any] = {}
    if tool_spec.min_calls is not None:
        tool_dict["min_calls"] = tool_spec.min_calls
    if tool_spec.max_calls is not None:
        tool_dict["max_calls"] = tool_spec.max_calls
    # Preserve params distinction: None (not declared) vs {} (empty dict)
    if tool_spec.params is not None:
        tool_dict["params"] = tool_spec.params
    return tool_dict if tool_dict else None


def convert_criterion_to_dict(criterion: Criterion) -> Dict[str, Any]:
    """Convert a Criterion to dictionary format."""
    crit_dict: Dict[str, Any] = {
        "category": criterion.category,
        "weight": criterion.weight,
        "dimension": criterion.dimension,
        "criterion": criterion.criterion
    }
    
    if not criterion.tool_calls:
        return crit_dict
    
    required_list = [
        {tc.name: convert_tool_spec_to_dict(tc)}
        for tc in criterion.tool_calls.required
    ]
    
    optional_list = [
        {tc.name: convert_tool_spec_to_dict(tc)}
        for tc in criterion.tool_calls.optional
    ]
    
    prohibited_list = [
        {tc.name: None}
        for tc in criterion.tool_calls.prohibited
    ]
    
    crit_dict["tool_calls"] = {
        "respect_order": criterion.tool_calls.respect_order,
        "required": required_list,
        "optional": optional_list if optional_list else [],
        "prohibited": prohibited_list if prohibited_list else []
    }
    # Only include params_strict_mode if it's True (default is False)
    if criterion.tool_calls.params_strict_mode:
        crit_dict["tool_calls"]["params_strict_mode"] = True
    
    return crit_dict


def convert_rubric_to_yaml_dict(rubric: Rubric) -> Dict[str, Any]:
    """Convert a Rubric object to YAML dictionary format."""
    rubric_dict: Dict[str, Any] = {}
    
    # Include variables section first if present (sorted alphabetically for consistency)
    if rubric.variables:
        rubric_dict["variables"] = dict(sorted(rubric.variables.items()))
    
    rubric_dict["dimensions"] = []
    for dim in rubric.dimensions:
        dim_dict: Dict[str, Any] = {
            dim.name: dim.description,
            "grading_type": dim.grading_type
        }
        if dim.scores:
            dim_dict["scores"] = dim.scores
        rubric_dict["dimensions"].append(dim_dict)
    
    rubric_dict["criteria"] = {
        criterion.name: convert_criterion_to_dict(criterion)
        for criterion in rubric.criteria
    }
    
    return rubric_dict


def write_rubric_to_file(rubric: Rubric, output_path: str) -> None:
    """Write a rubric to a YAML file (always self-contained with variables)."""
    rubric_dict = convert_rubric_to_yaml_dict(rubric)
    with open(output_path, 'w') as f:
        yaml.dump(
            rubric_dict,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True
        )


def print_rubric_summary(rubric: Rubric, title: str) -> None:
    """Print a summary of the rubric."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    
    if rubric.variables:
        print(f"\nVariables ({len(rubric.variables)}):")
        for var_name, var_value in rubric.variables.items():
            # Truncate long values
            display_value = var_value if len(var_value) <= 40 else var_value[:37] + "..."
            print(f"  • {var_name}: {display_value}")
    
    print(f"\nDimensions ({len(rubric.dimensions)}):")
    for dim in rubric.dimensions:
        print(f"  • {dim.name} ({dim.grading_type})")
    
    print(f"\nCriteria ({len(rubric.criteria)}):")
    for crit in rubric.criteria:
        print(f"  • {crit.name} [{crit.category}] - {crit.dimension}")
    print()


def print_evaluation_config(panel_config: JudgePanelConfig) -> None:
    """Print evaluation configuration details."""
    print(f"   Execution mode: {panel_config.execution.mode}")
    print(f"   Consensus mode: {panel_config.consensus.mode}")
    if panel_config.consensus.mode in ("quorum", "majority"):
        print(f"   Consensus threshold: {panel_config.consensus.threshold}")


def ensure_yaml_extension(output_file: str) -> str:
    """Ensure the output file has a .yaml extension."""
    base, ext = os.path.splitext(output_file)
    if ext.lower() not in ('.yaml', '.yml'):
        return f"{output_file}.yaml"
    return output_file


def parse_dimension_criteria_counts(args) -> Tuple[Optional[int], Optional[int]]:
    """Parse dimension and criteria counts, supporting 'auto' keyword."""
    num_dimensions = None if args.num_dimensions == "auto" else int(args.num_dimensions)
    num_criteria = None if args.num_criteria == "auto" else int(args.num_criteria)
    return num_dimensions, num_criteria


def print_generation_progress(num_dimensions: Optional[int], num_criteria: Optional[int]) -> None:
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


# ============================================================================
# Command Functions
# ============================================================================

def convert_rubric_to_portable_dict(rubric: Rubric) -> Dict[str, Any]:
    """Convert a Rubric object to a portable dictionary format."""
    return {
        "dimensions": [
            {
                "name": dim.name,
                "description": dim.description,
                "grading_type": dim.grading_type,
                "scores": dim.scores,
                "pass_above": dim.pass_above
            }
            for dim in rubric.dimensions
        ],
        "criteria": [
            {
                "name": crit.name,
                "category": crit.category,
                "dimension": crit.dimension,
                "criterion": crit.criterion,
                "weight": crit.weight,
                "tool_calls": convert_tool_calls_to_dict(crit.tool_calls) if crit.tool_calls else None
            }
            for crit in rubric.criteria
        ]
    }


def convert_tool_calls_to_dict(tool_calls) -> Dict[str, Any]:
    """Convert ToolCalls object to dictionary format."""
    return {
        "respect_order": tool_calls.respect_order,
        "params_strict_mode": tool_calls.params_strict_mode,
        "required": [
            {tc.name: convert_tool_spec_to_dict(tc)}
            for tc in tool_calls.required
        ],
        "optional": [
            {tc.name: convert_tool_spec_to_dict(tc)}
            for tc in tool_calls.optional
        ],
        "prohibited": [
            {tc.name: None}
            for tc in tool_calls.prohibited
        ]
    }


def convert_panel_config_to_portable_dict(panel_config: JudgePanelConfig) -> Dict[str, Any]:
    """Convert JudgePanelConfig to a portable dictionary (no API keys)."""
    return {
        "judges": [
            {
                "name": j.name,
                "model": j.model,
                "base_url": j.base_url,
                "temperature": j.temperature,
                "max_tokens": j.max_tokens,
                "top_p": j.top_p,
                "frequency_penalty": j.frequency_penalty,
                "presence_penalty": j.presence_penalty
            }
            for j in panel_config.judges
        ],
        "execution": {
            "mode": panel_config.execution.mode,
            "batch_size": panel_config.execution.batch_size,
            "timeout": panel_config.execution.timeout
        },
        "consensus": {
            "mode": panel_config.consensus.mode,
            "threshold": panel_config.consensus.threshold,
            "on_no_consensus": panel_config.consensus.on_no_consensus
        }
    }


def read_input_content(input_file: str) -> str:
    """Read and return input file content."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


@handle_command_errors
def cmd_evaluate(args) -> int:
    """
    Execute the 'evaluate' subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
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
    
    # Determine input file and type
    if args.qna_file:
        input_file = args.qna_file
        input_type = "qna"
    else:
        input_file = args.chat_session_file
        input_type = "chat_session"
    
    # Evaluate based on input type
    if input_type == "qna":
        print(f"\nEvaluating Q&A from {input_file}...")
        print_evaluation_config(panel_config)
        evaluations = evaluate_rubric_with_panel_from_qa(
            rubric,
            input_file,
            panel_config
        )
    else:
        print(f"\nEvaluating chat session from {input_file}...")
        print_evaluation_config(panel_config)
        evaluations = evaluate_rubric_with_panel(
            rubric,
            input_file,
            panel_config
        )
    
    print(f"✓ Evaluated {len(evaluations)} criteria")
    
    # Process scores
    print("\nProcessing scores...")
    results = evaluate_rubric(rubric, evaluations)
    
    # Calculate scores
    total_score, max_score = calculate_total_score(results)
    percentage = calculate_percentage_score(results)
    print(f"✓ Evaluation complete: {total_score}/{max_score} ({percentage:.1f}%)")
    
    # Build self-contained output structure
    output_data = {
        # Results section
        "results": results,
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1)
        },
        
        # Full rubric (portable, self-contained)
        "rubric": convert_rubric_to_portable_dict(rubric),
        
        # Full judge panel config (portable, no API keys)
        "judge_panel": convert_panel_config_to_portable_dict(panel_config),
        
        # Input reference and optional content
        "input": {
            "type": input_type,
            "source_file": input_file
        },
        
        # Metadata
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "rubric_source_file": args.rubric_file,
            "judge_panel_source_file": args.judge_panel_config
        }
    }
    
    # Add report title if provided
    if args.report_title:
        output_data["metadata"]["report_title"] = args.report_title
    
    # Include input content if requested
    if args.include_input:
        print("   Including input content in output...")
        output_data["input"]["content"] = read_input_content(input_file)
    
    # Always write YAML output (source of truth)
    output_file = ensure_yaml_extension(args.output_file)
    print(f"\nWriting results to {output_file} (YAML)...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"✓ YAML file written (self-contained)")
    
    # Generate PDF report if requested
    if args.report:
        print(f"\nGenerating PDF report to {args.report}...")
        try:
            export_evaluation_pdf(output_file, args.report)
            print(f"✓ PDF report generated")
        except Exception as e:
            print(f"⚠ PDF generation failed: {e}", file=sys.stderr)
    
    # Print table if requested
    if not args.no_table:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80 + "\n")
        print_table(results, include_summary=True)
        print_tool_breakdowns(results)
        print()
    
    return 0


@handle_command_errors
def cmd_generate(args) -> int:
    """
    Execute the 'generate' subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse input based on type
    if args.qna_file:
        print(f"Loading Q&A from {args.qna_file}...")
        input_obj = parse_qa_input(args.qna_file)
        print(f"✓ Loaded Q&A pair")
        print(f"   Q: {input_obj.question[:80]}{'...' if len(input_obj.question) > 80 else ''}")
        input_type = "qa"
    else:
        print(f"Loading chat session from {args.chat_session_file}...")
        input_obj = parse_chat_session(args.chat_session_file)
        print(f"✓ Loaded chat session")
        print(f"   Content length: {len(input_obj.content)} characters")
        print(f"   The LLM will analyze the session to detect tool calls and structure")
        input_type = "chat"
    
    # Parse category hints if provided
    category_hints = None
    if args.categories:
        category_hints = [c.strip() for c in args.categories.split(',')]
        print(f"   Category hints: {', '.join(category_hints)}")
    
    # Initialize generator
    print(f"\n🤖 Initializing rubric generator...")
    print(f"   Model: {args.model}")
    generator = create_generator(args)
    
    # Parse dimension and criteria counts
    num_dimensions, num_criteria = parse_dimension_criteria_counts(args)
    
    # Load dimensions file if provided
    provided_dimensions = None
    if args.dimensions_file:
        print(f"\nLoading dimensions from {args.dimensions_file}...")
        provided_dimensions = parse_dimensions_file(args.dimensions_file)
        print(f"✓ Loaded {len(provided_dimensions)} dimensions (skipping dimension generation)")
    
    # Generate rubric
    print_generation_progress(num_dimensions, num_criteria)
    
    if input_type == "chat":
        rubric = generator.generate_rubric_from_chat(
            input_obj,
            num_dimensions=num_dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            dimensions=provided_dimensions
        )
    else:
        rubric = generator.generate_rubric(
            input_obj,
            num_dimensions=num_dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            dimensions=provided_dimensions
        )
    
    print(f"✓ Generated {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    if rubric.variables:
        print(f"   Variables extracted: {len(rubric.variables)}")
    
    # Write rubric to file
    print(f"\nWriting rubric to {args.output_file}...")
    write_rubric_to_file(rubric, args.output_file)
    print(f"✓ Rubric written successfully")
    
    # Print summary
    print_rubric_summary(rubric, "GENERATED RUBRIC SUMMARY")
    
    return 0


@handle_command_errors
def cmd_export(args) -> int:
    """
    Execute the 'export' subcommand to convert YAML to various formats.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
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


def load_self_contained_yaml(input_file: str) -> Dict[str, Any]:
    """Load a self-contained evaluation YAML file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Validate required sections
    if not data.get("rubric"):
        raise ValueError("Input file missing 'rubric' section - not a self-contained evaluation file")
    if not data.get("judge_panel"):
        raise ValueError("Input file missing 'judge_panel' section - not a self-contained evaluation file")
    
    return data


def load_arena_spec(arena_spec_file: str) -> ArenaSpec:
    """Load and validate an arena specification file."""
    if not os.path.exists(arena_spec_file):
        raise FileNotFoundError(f"Arena spec file not found: {arena_spec_file}")
    
    with open(arena_spec_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # The arena spec should be under an 'arena' key
    if "arena" not in data:
        raise ValueError("Arena spec file must have an 'arena' key at the root")
    
    arena_data = data["arena"]
    
    # Parse contestants
    contestants = []
    for c in arena_data.get("contestants", []):
        contestants.append(ArenaContestant(**c))
    
    return ArenaSpec(
        name=arena_data.get("name"),
        description=arena_data.get("description"),
        rubric_file=arena_data["rubric_file"],
        judges_panel_file=arena_data["judges_panel_file"],
        contestants=contestants
    )


def rebuild_rubric_from_dict(rubric_data: Dict[str, Any]) -> Rubric:
    """Rebuild a Rubric object from portable dictionary format."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    dimensions = []
    for dim_data in rubric_data.get("dimensions", []):
        dimensions.append(Dimension(
            name=dim_data["name"],
            description=dim_data["description"],
            grading_type=dim_data["grading_type"],
            scores=dim_data.get("scores"),
            pass_above=dim_data.get("pass_above")
        ))
    
    criteria = []
    for crit_data in rubric_data.get("criteria", []):
        tool_calls = None
        if crit_data.get("tool_calls"):
            tc_data = crit_data["tool_calls"]
            tool_calls = ToolCalls(
                respect_order=tc_data.get("respect_order", True),
                params_strict_mode=tc_data.get("params_strict_mode", False),
                required=[
                    ToolSpec(name=list(t.keys())[0], **(list(t.values())[0] or {}))
                    for t in tc_data.get("required", [])
                ],
                optional=[
                    ToolSpec(name=list(t.keys())[0], **(list(t.values())[0] or {}))
                    for t in tc_data.get("optional", [])
                ],
                prohibited=[
                    ToolSpec(name=list(t.keys())[0])
                    for t in tc_data.get("prohibited", [])
                ]
            )
        
        criteria.append(Criterion(
            name=crit_data["name"],
            category=crit_data.get("category"),
            dimension=crit_data["dimension"],
            criterion=crit_data.get("criterion"),
            weight=crit_data["weight"],
            tool_calls=tool_calls
        ))
    
    return Rubric(dimensions=dimensions, criteria=criteria)


def rebuild_panel_config_from_dict(panel_data: Dict[str, Any], api_key: Optional[str] = None) -> JudgePanelConfig:
    """Rebuild a JudgePanelConfig from portable dictionary format."""
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required. Set OPENAI_API_KEY environment variable or use --api-key")
    
    judges = []
    for j_data in panel_data.get("judges", []):
        judges.append(JudgeConfig(
            name=j_data["name"],
            model=j_data["model"],
            api_key=api_key,  # Always use provided/env API key
            base_url=j_data.get("base_url"),
            temperature=j_data.get("temperature"),
            max_tokens=j_data.get("max_tokens"),
            top_p=j_data.get("top_p"),
            frequency_penalty=j_data.get("frequency_penalty"),
            presence_penalty=j_data.get("presence_penalty")
        ))
    
    exec_data = panel_data.get("execution", {})
    execution = ExecutionConfig(
        mode=exec_data.get("mode", "sequential"),
        batch_size=exec_data.get("batch_size", 2),
        timeout=exec_data.get("timeout", 30)
    )
    
    cons_data = panel_data.get("consensus", {})
    consensus = ConsensusConfig(
        mode=cons_data.get("mode", "unanimous"),
        threshold=cons_data.get("threshold"),
        on_no_consensus=cons_data.get("on_no_consensus", "fail")
    )
    
    return JudgePanelConfig(judges=judges, execution=execution, consensus=consensus)


@handle_command_errors
def cmd_rerun(args) -> int:
    """
    Execute the 'rerun' subcommand - re-evaluate using settings from a previous output.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print(f"Loading evaluation configuration from {args.input_file}...")
    data = load_self_contained_yaml(args.input_file)
    
    # Rebuild rubric from stored data
    rubric = rebuild_rubric_from_dict(data["rubric"])
    print(f"✓ Rebuilt rubric: {len(rubric.dimensions)} dimensions, {len(rubric.criteria)} criteria")
    
    # Rebuild panel config from stored data
    panel_config = rebuild_panel_config_from_dict(data["judge_panel"], args.api_key)
    print(f"✓ Rebuilt judge panel: {len(panel_config.judges)} judge(s)")
    
    # Determine input source
    input_data = data.get("input", {})
    input_type = input_data.get("type", "chat_session")
    
    # Check for new input override
    if args.qna_file:
        input_file = args.qna_file
        input_type = "qna"
        print(f"\n📥 Using new Q&A input: {input_file}")
    elif args.chat_session_file:
        input_file = args.chat_session_file
        input_type = "chat_session"
        print(f"\n📥 Using new chat session input: {input_file}")
    elif input_data.get("content"):
        # Use embedded content
        print(f"\n📥 Using embedded input content from previous evaluation")
        input_content = input_data["content"]
        input_file = None  # Will use content directly
    elif input_data.get("source_file") and os.path.exists(input_data["source_file"]):
        input_file = input_data["source_file"]
        print(f"\n📥 Using original input file: {input_file}")
    else:
        raise ValueError(
            "No input available. The original input file is not accessible and no embedded content. "
            "Use --from-chat-session or --from-qna to provide new input."
        )
    
    # Evaluate
    print_evaluation_config(panel_config)
    
    if input_file:
        if input_type == "qna":
            evaluations = evaluate_rubric_with_panel_from_qa(rubric, input_file, panel_config)
        else:
            evaluations = evaluate_rubric_with_panel(rubric, input_file, panel_config)
    else:
        # Use embedded content - need to write to temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(input_content)
            temp_file = f.name
        try:
            if input_type == "qna":
                evaluations = evaluate_rubric_with_panel_from_qa(rubric, temp_file, panel_config)
            else:
                evaluations = evaluate_rubric_with_panel(rubric, temp_file, panel_config)
        finally:
            os.unlink(temp_file)
    
    print(f"✓ Evaluated {len(evaluations)} criteria")
    
    # Process scores
    print("\nProcessing scores...")
    results = evaluate_rubric(rubric, evaluations)
    
    total_score, max_score = calculate_total_score(results)
    percentage = calculate_percentage_score(results)
    print(f"✓ Evaluation complete: {total_score}/{max_score} ({percentage:.1f}%)")
    
    # Build output structure (same format as evaluate)
    output_data = {
        "results": results,
        "summary": {
            "total_score": total_score,
            "max_score": max_score,
            "percentage": round(percentage, 1)
        },
        "rubric": data["rubric"],  # Preserve original rubric
        "judge_panel": data["judge_panel"],  # Preserve original panel config
        "input": {
            "type": input_type,
            "source_file": input_file or input_data.get("source_file")
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "rerun_from": args.input_file,
            "rubric_source_file": data.get("metadata", {}).get("rubric_source_file"),
            "judge_panel_source_file": data.get("metadata", {}).get("judge_panel_source_file")
        }
    }
    
    # Use new report title if provided, otherwise preserve original
    if args.report_title:
        output_data["metadata"]["report_title"] = args.report_title
    elif data.get("metadata", {}).get("report_title"):
        output_data["metadata"]["report_title"] = data["metadata"]["report_title"]
    
    # Include input content if requested or if it was in original
    if args.include_input:
        if input_file:
            output_data["input"]["content"] = read_input_content(input_file)
        elif input_data.get("content"):
            output_data["input"]["content"] = input_data["content"]
    
    # Write output
    output_file = ensure_yaml_extension(args.output_file)
    print(f"\nWriting results to {output_file} (YAML)...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"✓ YAML file written (self-contained)")
    
    # Generate PDF report if requested
    if args.report:
        print(f"\nGenerating PDF report to {args.report}...")
        try:
            export_evaluation_pdf(output_file, args.report)
            print(f"✓ PDF report generated")
        except Exception as e:
            print(f"⚠ PDF generation failed: {e}", file=sys.stderr)
    
    # Print table if requested
    if not args.no_table:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80 + "\n")
        print_table(results, include_summary=True)
        print_tool_breakdowns(results)
        print()
    
    return 0


@handle_command_errors
def cmd_refine(args) -> int:
    """
    Execute the 'refine' subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load existing rubric (don't require variables - LLM can generate them)
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
    
    # Initialize generator
    print(f"\n🤖 Initializing rubric refiner...")
    print(f"   Model: {args.model}")
    generator = create_generator(args)
    
    # Parse input based on type (if provided)
    input_type = None
    input_obj = None
    
    if args.qna_file:
        print(f"\nLoading Q&A from {args.qna_file}...")
        input_obj = parse_qa_input(args.qna_file)
        print(f"✓ Loaded Q&A pair")
        print(f"   Q: {input_obj.question[:80]}{'...' if len(input_obj.question) > 80 else ''}")
        input_type = "qa"
    elif args.chat_session_file:
        print(f"\nLoading chat session from {args.chat_session_file}...")
        input_obj = parse_chat_session(args.chat_session_file)
        print(f"✓ Loaded chat session")
        print(f"   Content length: {len(input_obj.content)} characters")
        input_type = "chat"
    
    # Load dimensions file if provided
    dimensions_to_merge = None
    if args.dimensions_file:
        print(f"\nLoading dimensions from {args.dimensions_file}...")
        dimensions_to_merge = parse_dimensions_file(args.dimensions_file)
        print(f"✓ Loaded {len(dimensions_to_merge)} dimensions to merge")
    
    # Refine rubric
    feedback_msg = f" with feedback" if args.feedback else ""
    context_msg = f" using {input_type} context" if input_type else ""
    dims_msg = f" (merging {len(dimensions_to_merge)} dimensions)" if dimensions_to_merge else ""
    print(f"\n🔄 Refining rubric{context_msg}{feedback_msg}{dims_msg}...")
    if args.feedback:
        print(f"   Feedback: {args.feedback}")
    print("   This may take a moment...")
    
    if input_type == "qa":
        refined_rubric = generator.refine_rubric_with_qa(
            rubric,
            input_obj,
            feedback=args.feedback,
            dimensions_to_merge=dimensions_to_merge
        )
    elif input_type == "chat":
        refined_rubric = generator.refine_rubric_with_chat(
            rubric,
            input_obj,
            feedback=args.feedback,
            dimensions_to_merge=dimensions_to_merge
        )
    else:
        refined_rubric = generator.refine_rubric(
            rubric,
            feedback=args.feedback,
            dimensions_to_merge=dimensions_to_merge
        )
    
    print(f"✓ Refined rubric: {len(refined_rubric.dimensions)} dimensions, {len(refined_rubric.criteria)} criteria")
    if refined_rubric.variables:
        print(f"   Variables extracted: {len(refined_rubric.variables)}")
    
    # Determine output path and write
    output_path = args.output_file if args.output_file else args.rubric_file
    print(f"\nWriting refined rubric to {output_path}...")
    write_rubric_to_file(refined_rubric, output_path)
    print(f"✓ Refined rubric written successfully")
    
    # Print summary
    print_rubric_summary(refined_rubric, "REFINED RUBRIC SUMMARY")
    
    return 0


def load_contestant_variables(contestant: ArenaContestant, base_dir: str) -> Optional[Dict[str, str]]:
    """Load variables for a contestant from inline definition or external file."""
    # Inline variables take priority
    if contestant.variables:
        return contestant.variables
    
    # Load from external file if specified
    if contestant.variables_file:
        variables_path = os.path.join(base_dir, contestant.variables_file)
        if not os.path.exists(variables_path):
            raise FileNotFoundError(f"Variables file not found: {variables_path}")
        
        with open(variables_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Variables can be at root level or under 'variables' key
        if "variables" in data:
            return data["variables"]
        return data
    
    return None


def apply_variables_to_rubric(base_rubric: Rubric, variables: Dict[str, str]) -> Rubric:
    """
    Create a new Rubric with variable substitution applied to criterion text and tool params.
    
    Args:
        base_rubric: The base rubric with placeholder text
        variables: Dictionary of variable values to substitute
        
    Returns:
        New Rubric with variables substituted in criterion text and tool params
    """
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    # Apply substitution to each criterion
    substituted_criteria = []
    for crit in base_rubric.criteria:
        # Substitute in criterion text
        new_criterion_text = substitute_variables(crit.criterion, variables)
        
        # Substitute in tool call params if present
        new_tool_calls = None
        if crit.tool_calls:
            new_required = []
            for tc in crit.tool_calls.required:
                new_params = None
                if tc.params is not None:
                    new_params = {
                        k: substitute_variables(v, variables) if isinstance(v, str) else v
                        for k, v in tc.params.items()
                    }
                new_required.append(ToolSpec(
                    name=tc.name,
                    min_calls=tc.min_calls,
                    max_calls=tc.max_calls,
                    params=new_params
                ))
            
            new_optional = []
            for tc in crit.tool_calls.optional:
                new_params = None
                if tc.params is not None:
                    new_params = {
                        k: substitute_variables(v, variables) if isinstance(v, str) else v
                        for k, v in tc.params.items()
                    }
                new_optional.append(ToolSpec(
                    name=tc.name,
                    min_calls=tc.min_calls,
                    max_calls=tc.max_calls,
                    params=new_params
                ))
            
            new_tool_calls = ToolCalls(
                respect_order=crit.tool_calls.respect_order,
                params_strict_mode=crit.tool_calls.params_strict_mode,
                required=new_required,
                optional=new_optional,
                prohibited=crit.tool_calls.prohibited  # Prohibited tools don't need param substitution
            )
        
        substituted_criteria.append(Criterion(
            name=crit.name,
            category=crit.category,
            weight=crit.weight,
            dimension=crit.dimension,
            criterion=new_criterion_text,
            tool_calls=new_tool_calls
        ))
    
    # Also substitute in dimension descriptions
    substituted_dimensions = []
    for dim in base_rubric.dimensions:
        new_description = substitute_variables(dim.description, variables)
        substituted_dimensions.append(Dimension(
            name=dim.name,
            description=new_description,
            grading_type=dim.grading_type,
            scores=dim.scores,
            pass_above=dim.pass_above
        ))
    
    return Rubric(
        dimensions=substituted_dimensions,
        criteria=substituted_criteria,
        variables=variables
    )


def combine_outputs_to_arena(output_files: List[str], arena_name: str = "Combined Arena") -> Dict[str, Any]:
    """
    Combine multiple evaluation output files into arena format.
    
    Args:
        output_files: List of paths to output.yaml files
        arena_name: Name for the combined arena
        
    Returns:
        Arena-formatted data structure
    """
    contestants_results: Dict[str, Any] = {}
    shared_rubric = None
    shared_judge_panel = None
    
    for idx, output_file in enumerate(output_files):
        print(f"\n[{idx + 1}/{len(output_files)}] Loading: {output_file}")
        
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        with open(output_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Check if this is already an arena file
        if data.get("mode") == "arena":
            raise ValueError(f"File is already an arena result: {output_file}")
        
        # Validate required sections
        if not data.get("results"):
            raise ValueError(f"File missing 'results' section: {output_file}")
        
        # Generate contestant ID from filename or metadata
        basename = os.path.splitext(os.path.basename(output_file))[0]
        contestant_id = basename.replace("output_", "").replace("_", "-")
        
        # Try to get a better name from metadata
        metadata = data.get("metadata", {})
        contestant_name = metadata.get("report_title", basename)
        
        # Extract input info
        input_info = data.get("input", {})
        
        print(f"   ID: {contestant_id}")
        print(f"   Name: {contestant_name}")
        
        summary = data.get("summary", {})
        print(f"   Score: {summary.get('total_score', 0)}/{summary.get('max_score', 0)} ({summary.get('percentage', 0):.1f}%)")
        
        # Store contestant results
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
        
        # Use first file's rubric and judge_panel as shared (they should be compatible)
        if shared_rubric is None and data.get("rubric"):
            shared_rubric = data["rubric"]
        if shared_judge_panel is None and data.get("judge_panel"):
            shared_judge_panel = data["judge_panel"]
    
    # Generate rankings
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


def _save_partial_arena_results(
    output_file: str,
    arena_name: str,
    arena_spec: ArenaSpec,
    contestants_results: Dict[str, Any],
    base_rubric: Rubric,
    panel_config: JudgePanelConfig,
    args
) -> None:
    """Save partial arena results after each contestant evaluation."""
    # Generate rankings from current results
    rankings = sorted(
        [
            {
                "id": cid,
                "name": cdata["name"],
                "percentage": cdata["summary"]["percentage"],
                "total_score": cdata["summary"]["total_score"],
                "max_score": cdata["summary"]["max_score"]
            }
            for cid, cdata in contestants_results.items()
        ],
        key=lambda x: x["percentage"],
        reverse=True
    )
    
    for idx, r in enumerate(rankings, 1):
        r["rank"] = idx
    
    # Build partial output structure
    output_data = {
        "mode": "arena",
        "arena_name": arena_name,
        "arena_description": arena_spec.description,
        "contestants": contestants_results,
        "rankings": rankings,
        "rubric": convert_rubric_to_portable_dict(base_rubric),
        "judge_panel": convert_panel_config_to_portable_dict(panel_config),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "arena_spec_file": args.arena_spec,
            "rubric_source_file": arena_spec.rubric_file,
            "judge_panel_source_file": arena_spec.judges_panel_file,
            "partial": True  # Indicate this is a partial save
        }
    }
    
    if args.report_title:
        output_data["metadata"]["report_title"] = args.report_title
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


@handle_command_errors
def cmd_arena(args) -> int:
    """
    Execute the 'arena' subcommand for comparative evaluation.
    
    Supports two modes:
    1. --arena-spec: Run fresh evaluations from arena specification
    2. --from-outputs: Combine existing output.yaml files into arena format
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Mode 1: Combine existing outputs
    if args.output_files:
        print(f"Combining {len(args.output_files)} evaluation outputs into arena format...")
        
        arena_name = args.report_title or "Combined Arena Evaluation"
        output_data = combine_outputs_to_arena(args.output_files, arena_name)
        
        # Add report title if provided
        if args.report_title:
            output_data["metadata"]["report_title"] = args.report_title
        
        # Write YAML output
        output_file = ensure_yaml_extension(args.output_file)
        print(f"\nWriting arena results to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
        print(f"✓ Arena results written (YAML)")
        
        # Generate PDF report if requested
        if args.report:
            print(f"\nGenerating Arena PDF report to {args.report}...")
            try:
                export_arena_pdf(output_file, args.report)
                print(f"✓ Arena PDF report generated")
            except Exception as e:
                print(f"⚠ PDF generation failed: {e}", file=sys.stderr)
        
        # Print rankings summary
        if not args.no_table:
            print(f"\n{'='*80}")
            print("ARENA RANKINGS")
            print(f"{'='*80}\n")
            
            for r in output_data["rankings"]:
                medal = "🥇" if r["rank"] == 1 else ("🥈" if r["rank"] == 2 else ("🥉" if r["rank"] == 3 else "  "))
                print(f"{medal} #{r['rank']}: {r['name']} - {r['percentage']:.1f}% ({r['total_score']}/{r['max_score']})")
            print()
        
        return 0
    
    # Mode 2: Run fresh evaluations from arena spec
    # Load arena specification
    print(f"Loading arena specification from {args.arena_spec}...")
    arena_spec = load_arena_spec(args.arena_spec)
    arena_name = arena_spec.name or "Arena Evaluation"
    print(f"✓ Loaded arena: {arena_name}")
    print(f"   Contestants: {len(arena_spec.contestants)}")
    
    # Check for existing results (resume support)
    output_file = ensure_yaml_extension(args.output_file)
    existing_results: Dict[str, Any] = {}
    
    if os.path.exists(output_file) and not getattr(args, 'force', False):
        print(f"\n📂 Found existing results in {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f)
            if existing_data and existing_data.get("mode") == "arena":
                existing_results = existing_data.get("contestants", {})
                print(f"   ✓ Loaded {len(existing_results)} cached contestant results")
                print("   (Use --force to re-evaluate all)")
        except Exception as e:
            print(f"   ⚠ Could not load existing results: {e}")
    elif getattr(args, 'force', False):
        print("\n🔄 Force mode: will re-evaluate all contestants")
    
    # Get base directory for relative paths
    base_dir = os.path.dirname(os.path.abspath(args.arena_spec))
    
    # Load shared rubric
    rubric_path = os.path.join(base_dir, arena_spec.rubric_file)
    print(f"\nLoading shared rubric from {rubric_path}...")
    # Load rubric without variables initially - we'll apply per-contestant
    base_rubric = load_rubric(rubric_path, require_variables=False)
    print(f"✓ Loaded {len(base_rubric.dimensions)} dimensions and {len(base_rubric.criteria)} criteria")
    
    # Load shared judge panel
    panel_path = os.path.join(base_dir, arena_spec.judges_panel_file)
    print(f"\nLoading judge panel from {panel_path}...")
    panel_config = load_judge_panel_config(panel_path)
    print(f"✓ Loaded panel with {len(panel_config.judges)} judge(s)")
    print_evaluation_config(panel_config)
    
    # Dictionary to hold all contestant results (start with cached)
    contestants_results: Dict[str, Any] = dict(existing_results)
    failed_contestants: List[str] = []
    skipped_count = 0
    evaluated_count = 0
    
    # Evaluate each contestant
    print(f"\n{'='*80}")
    print("ARENA EVALUATION")
    print(f"{'='*80}")
    
    for idx, contestant in enumerate(arena_spec.contestants, 1):
        # Check if already cached
        if contestant.id in existing_results:
            cached = existing_results[contestant.id]
            cached_pct = cached.get("summary", {}).get("percentage", 0)
            print(f"\n[{idx}/{len(arena_spec.contestants)}] {contestant.name} (id: {contestant.id})")
            print(f"   ⏭️  Skipped (cached: {cached_pct:.1f}%)")
            skipped_count += 1
            continue
        
        print(f"\n[{idx}/{len(arena_spec.contestants)}] Evaluating: {contestant.name} (id: {contestant.id})")
        
        try:
            # Load contestant-specific variables if any
            contestant_vars = load_contestant_variables(contestant, base_dir)
            
            # Create a rubric with contestant-specific variables (with substitution applied)
            if contestant_vars:
                rubric = apply_variables_to_rubric(base_rubric, contestant_vars)
                print(f"   Variables: {len(contestant_vars)}")
            else:
                rubric = base_rubric
            
            # Resolve input file path
            input_path = os.path.join(base_dir, contestant.input_file)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Evaluate based on input type
            print(f"   Input: {contestant.input_type} from {contestant.input_file}")
            
            if contestant.input_type == "qna":
                evaluations = evaluate_rubric_with_panel_from_qa(
                    rubric, input_path, panel_config
                )
            else:
                evaluations = evaluate_rubric_with_panel(
                    rubric, input_path, panel_config
                )
            
            # Process scores
            results = evaluate_rubric(rubric, evaluations)
            total_score, max_score = calculate_total_score(results)
            percentage = calculate_percentage_score(results)
            
            print(f"   ✓ Score: {total_score}/{max_score} ({percentage:.1f}%)")
            
            # Store contestant results
            contestants_results[contestant.id] = {
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
            evaluated_count += 1
            
            # Save partial results after each successful evaluation
            _save_partial_arena_results(
                output_file, arena_name, arena_spec, contestants_results,
                base_rubric, panel_config, args
            )
            
        except Exception as e:
            print(f"   ❌ Failed: {e}", file=sys.stderr)
            failed_contestants.append(contestant.id)
            # Continue with next contestant instead of failing completely
            continue
    
    # Summary of evaluation
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"   Evaluated: {evaluated_count}")
    print(f"   Skipped (cached): {skipped_count}")
    print(f"   Failed: {len(failed_contestants)}")
    if failed_contestants:
        print(f"   Failed IDs: {', '.join(failed_contestants)}")
        print("   (Fix the issues and re-run to complete these evaluations)")
    
    # Generate final rankings
    rankings = sorted(
        [
            {
                "id": cid,
                "name": cdata["name"],
                "percentage": cdata["summary"]["percentage"],
                "total_score": cdata["summary"]["total_score"],
                "max_score": cdata["summary"]["max_score"]
            }
            for cid, cdata in contestants_results.items()
        ],
        key=lambda x: x["percentage"],
        reverse=True
    )
    
    # Add rank
    for idx, r in enumerate(rankings, 1):
        r["rank"] = idx
    
    # Build final output structure
    output_data = {
        "mode": "arena",
        "arena_name": arena_name,
        "arena_description": arena_spec.description,
        
        "contestants": contestants_results,
        "rankings": rankings,
        
        # Shared configuration
        "rubric": convert_rubric_to_portable_dict(base_rubric),
        "judge_panel": convert_panel_config_to_portable_dict(panel_config),
        
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "arena_spec_file": args.arena_spec,
            "rubric_source_file": arena_spec.rubric_file,
            "judge_panel_source_file": arena_spec.judges_panel_file
        }
    }
    
    # Mark as partial if there were failures
    if failed_contestants:
        output_data["metadata"]["partial"] = True
        output_data["metadata"]["failed_contestants"] = failed_contestants
    
    # Add report title if provided
    if args.report_title:
        output_data["metadata"]["report_title"] = args.report_title
    
    # Write final YAML output
    print(f"\nWriting arena results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    
    if failed_contestants:
        print(f"✓ Arena results written (YAML) - {len(failed_contestants)} contestant(s) pending")
    else:
        print(f"✓ Arena results written (YAML) - complete")
    
    # Generate PDF report if requested
    if args.report:
        print(f"\nGenerating Arena PDF report to {args.report}...")
        try:
            export_arena_pdf(output_file, args.report)
            print(f"✓ Arena PDF report generated")
        except Exception as e:
            print(f"⚠ PDF generation failed: {e}", file=sys.stderr)
    
    # Print rankings summary
    if not args.no_table:
        print(f"\n{'='*80}")
        print("ARENA RANKINGS")
        print(f"{'='*80}\n")
        
        for r in rankings:
            medal = "🥇" if r["rank"] == 1 else ("🥈" if r["rank"] == 2 else ("🥉" if r["rank"] == 3 else "  "))
            print(f"{medal} #{r['rank']}: {r['name']} - {r['percentage']:.1f}% ({r['total_score']}/{r['max_score']})")
        print()
    
    return 0


def main() -> int:
    """
    Main CLI entry point with subcommands.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Rubric Kit - Automatic rubric evaluation using LLM-as-a-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        help='Command to execute'
    )
    
    # ========== EVALUATE subcommand ==========
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a chat session against a rubric',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from Q&A YAML file (always outputs YAML)
  %(prog)s --from-qna qna.yaml --rubric-file rubric.yaml --output-file results.yaml
  
  # Evaluate from chat session file
  %(prog)s --from-chat-session chat_session.txt --rubric-file rubric.yaml --output-file results.yaml
  
  # With PDF report generation
  %(prog)s --from-chat-session chat.txt --rubric-file rubric.yaml --output-file output.yaml --report report.pdf
  
  # With custom report title
  %(prog)s --from-qna qna.yaml --rubric-file rubric.yaml --output-file output.yaml --report report.pdf --report-title "Q1 2025 Evaluation"
"""
    )
    
    # Mutually exclusive input format options
    input_group = evaluate_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--from-qna',
        dest='qna_file',
        help='Path to Q&A YAML file (must contain question, answer, and optional context keys)'
    )
    input_group.add_argument(
        '--from-chat-session',
        dest='chat_session_file',
        help='Path to chat session file (any format, will use heuristics to parse)'
    )
    
    evaluate_parser.add_argument(
        '--rubric-file',
        required=True,
        help='Path to rubric YAML file'
    )
    
    evaluate_parser.add_argument(
        '--variables-file',
        help='Path to variables YAML file (required if rubric has placeholders but no embedded variables)'
    )
    
    evaluate_parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output YAML file (source of truth artifact). Extension .yaml is added if not present.'
    )
    
    evaluate_parser.add_argument(
        '--no-table',
        action='store_true',
        help='Do not print results table to console'
    )
    
    evaluate_parser.add_argument(
        '--report',
        dest='report',
        help='Path to generate PDF report (optional)'
    )
    
    evaluate_parser.add_argument(
        '--report-title',
        dest='report_title',
        help='Custom title for the PDF report (optional)'
    )
    
    evaluate_parser.add_argument(
        '--include-input',
        action='store_true',
        dest='include_input',
        help='Include input content in output YAML (for rerun capability)'
    )
    
    evaluate_parser.add_argument(
        '--judge-panel-config',
        help='Path to judge panel configuration YAML file (optional, creates single-judge panel if not provided)'
    )
    
    evaluate_parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable) - used for default single-judge panel'
    )
    
    evaluate_parser.add_argument(
        '--base-url',
        help='Base URL for OpenAI-compatible endpoint (optional) - used for default single-judge panel'
    )
    
    evaluate_parser.add_argument(
        '--model',
        default='gpt-4',
        help='Model name to use for LLM evaluation (default: gpt-4) - used for default single-judge panel'
    )
    
    evaluate_parser.set_defaults(func=cmd_evaluate)
    
    # ========== GENERATE subcommand ==========
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate a rubric from a Q&A pair or chat session',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from Q&A YAML file
  %(prog)s --from-qna qna.yaml --output-file output_rubric.yaml
  
  # Generate from chat session file
  %(prog)s --from-chat-session session.txt --output-file output_rubric.yaml
  
  # With custom parameters
  %(prog)s --from-qna qna.yaml --output-file rubric.yaml --num-dimensions 5 --num-criteria 8
  
  # With category hints
  %(prog)s --from-chat-session session.txt --output-file rubric.yaml --categories "Tools,Output,Reasoning"
"""
    )
    
    # Mutually exclusive input format options
    generate_input_group = generate_parser.add_mutually_exclusive_group(required=True)
    generate_input_group.add_argument(
        '--from-qna',
        dest='qna_file',
        help='Path to Q&A YAML file (must contain question, answer, and optional context keys)'
    )
    generate_input_group.add_argument(
        '--from-chat-session',
        dest='chat_session_file',
        help='Path to chat session file (any format, will use heuristics to parse)'
    )
    
    generate_parser.add_argument(
        '--output-file',
        required=True,
        help='Path to output rubric YAML file'
    )
    
    generate_parser.add_argument(
        '--num-dimensions',
        type=str,
        default="auto",
        help='Number of dimensions to generate (1-10 or "auto", default: auto)'
    )
    
    generate_parser.add_argument(
        '--num-criteria',
        type=str,
        default="auto",
        help='Number of criteria to generate (1-10 or "auto", default: auto)'
    )
    
    generate_parser.add_argument(
        '--categories',
        help='Comma-separated list of category hints (e.g., "Output,Reasoning")'
    )
    
    generate_parser.add_argument(
        '--dimensions-file',
        help='Path to dimensions YAML file (skips dimension generation, uses provided dimensions)'
    )
    
    generate_parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    generate_parser.add_argument(
        '--base-url',
        help='Base URL for OpenAI-compatible endpoint (optional)'
    )
    
    generate_parser.add_argument(
        '--model',
        default='gpt-4',
        help='Model name to use for generation (default: gpt-4)'
    )
    
    generate_parser.set_defaults(func=cmd_generate)
    
    # ========== REFINE subcommand ==========
    refine_parser = subparsers.add_parser(
        'refine',
        help='Refine an existing rubric',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (overwrites original)
  %(prog)s --rubric-file rubric.yaml
  
  # With feedback
  %(prog)s --rubric-file rubric.yaml --feedback "Add more specific criteria"
  
  # With custom output path
  %(prog)s --rubric-file rubric.yaml --output-file refined_rubric.yaml
  
  # Refine using Q&A context
  %(prog)s --rubric-file rubric.yaml --from-qna qna.yaml --output-file refined.yaml
  
  # Refine using chat session context
  %(prog)s --rubric-file rubric.yaml --from-chat-session session.txt --output-file refined.yaml
"""
    )
    
    refine_parser.add_argument(
        '--rubric-file',
        required=True,
        help='Path to existing rubric YAML file'
    )
    
    refine_parser.add_argument(
        '--variables-file',
        help='Path to variables YAML file (provides variable values for rubrics with placeholders)'
    )
    
    # Mutually exclusive input format options (optional for refine)
    refine_input_group = refine_parser.add_mutually_exclusive_group(required=False)
    refine_input_group.add_argument(
        '--from-qna',
        dest='qna_file',
        help='Path to Q&A YAML file to use as context for refinement (optional)'
    )
    refine_input_group.add_argument(
        '--from-chat-session',
        dest='chat_session_file',
        help='Path to chat session file to use as context for refinement (optional)'
    )
    
    refine_parser.add_argument(
        '--output-file',
        help='Output path for refined rubric (default: overwrite original)'
    )
    
    refine_parser.add_argument(
        '--feedback',
        help='Specific feedback for refinement (optional)'
    )
    
    refine_parser.add_argument(
        '--dimensions-file',
        help='Path to dimensions YAML file (merges with existing rubric dimensions)'
    )
    
    refine_parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    refine_parser.add_argument(
        '--base-url',
        help='Base URL for OpenAI-compatible endpoint (optional)'
    )
    
    refine_parser.add_argument(
        '--model',
        default='gpt-4',
        help='Model name to use for refinement (default: gpt-4)'
    )
    
    refine_parser.set_defaults(func=cmd_refine)
    
    # ========== EXPORT subcommand ==========
    export_parser = subparsers.add_parser(
        'export',
        help='Convert evaluation YAML to PDF, CSV, or JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to PDF report
  %(prog)s results.yaml --format pdf --output report.pdf
  
  # Export to CSV
  %(prog)s results.yaml --format csv --output results.csv
  
  # Export to JSON
  %(prog)s results.yaml --format json --output results.json
"""
    )
    
    export_parser.add_argument(
        'input_file',
        help='Path to input YAML file with evaluation results'
    )
    
    export_parser.add_argument(
        '--format',
        required=True,
        choices=['pdf', 'csv', 'json'],
        help='Output format: pdf, csv, or json'
    )
    
    export_parser.add_argument(
        '--output', '-o',
        dest='output_file',
        required=True,
        help='Path to output file'
    )
    
    export_parser.set_defaults(func=cmd_export)
    
    # ========== RERUN subcommand ==========
    rerun_parser = subparsers.add_parser(
        'rerun',
        help='Re-evaluate using settings from a previous self-contained output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Re-run with embedded input content
  %(prog)s results.yaml --output-file new_results.yaml
  
  # Re-run with new input
  %(prog)s results.yaml --from-chat-session new_chat.txt --output-file new_results.yaml
  
  # Re-run with new Q&A input
  %(prog)s results.yaml --from-qna new_qna.yaml --output-file new_results.yaml
  
  # Re-run and generate PDF report
  %(prog)s results.yaml --output-file new_results.yaml --report report.pdf
"""
    )
    
    rerun_parser.add_argument(
        'input_file',
        help='Path to self-contained evaluation YAML file'
    )
    
    rerun_parser.add_argument(
        '--output-file', '-o',
        required=True,
        help='Path to output YAML file'
    )
    
    # Optional new input
    rerun_input_group = rerun_parser.add_mutually_exclusive_group(required=False)
    rerun_input_group.add_argument(
        '--from-qna',
        dest='qna_file',
        help='Path to new Q&A YAML file (overrides embedded/original input)'
    )
    rerun_input_group.add_argument(
        '--from-chat-session',
        dest='chat_session_file',
        help='Path to new chat session file (overrides embedded/original input)'
    )
    
    rerun_parser.add_argument(
        '--include-input',
        action='store_true',
        dest='include_input',
        help='Include input content in output YAML'
    )
    
    rerun_parser.add_argument(
        '--report',
        help='Path to generate PDF report (optional)'
    )
    
    rerun_parser.add_argument(
        '--report-title',
        dest='report_title',
        help='Custom title for the PDF report (optional, overrides original)'
    )
    
    rerun_parser.add_argument(
        '--no-table',
        action='store_true',
        help='Do not print results table to console'
    )
    
    rerun_parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    rerun_parser.set_defaults(func=cmd_rerun)
    
    # ========== ARENA subcommand ==========
    arena_parser = subparsers.add_parser(
        'arena',
        help='Run comparative evaluation of multiple contestants against a shared rubric',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Run fresh evaluations from arena spec
  %(prog)s --arena-spec arena.yaml --output-file arena_results.yaml
  %(prog)s --arena-spec arena.yaml --output-file arena_results.yaml --report arena_report.pdf
  
  # Mode 2: Combine existing output.yaml files into arena comparison
  %(prog)s --from-outputs output1.yaml output2.yaml output3.yaml --output-file arena_results.yaml
  %(prog)s --from-outputs *.yaml --output-file arena_results.yaml --report arena_report.pdf --report-title "Model Comparison"
"""
    )
    
    # Two modes: --arena-spec (run evaluations) OR --from-outputs (combine existing)
    arena_input_group = arena_parser.add_mutually_exclusive_group(required=True)
    arena_input_group.add_argument(
        '--arena-spec',
        dest='arena_spec',
        help='Path to arena specification YAML file (runs fresh evaluations)'
    )
    arena_input_group.add_argument(
        '--from-outputs',
        dest='output_files',
        nargs='+',
        metavar='OUTPUT_FILE',
        help='Combine multiple existing output.yaml files into arena results'
    )
    
    arena_parser.add_argument(
        '--output-file', '-o',
        required=True,
        help='Path to output YAML file with aggregated results'
    )
    
    arena_parser.add_argument(
        '--report',
        help='Path to generate Arena PDF report (optional)'
    )
    
    arena_parser.add_argument(
        '--report-title',
        dest='report_title',
        help='Custom title for the Arena PDF report (optional)'
    )
    
    arena_parser.add_argument(
        '--no-table',
        action='store_true',
        help='Do not print rankings table to console'
    )
    
    arena_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-evaluation of all contestants (ignore cached results)'
    )
    
    arena_parser.set_defaults(func=cmd_arena)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no subcommand specified, print help and return error
    if not hasattr(args, 'func'):
        parser.print_help()
        return 2
    
    # Execute the subcommand
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

