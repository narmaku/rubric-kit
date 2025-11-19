"""Main CLI entry point for rubric-kit."""

import argparse
import sys
import os
import yaml
import traceback
from functools import wraps
from typing import Callable, Dict, Any, Optional, Tuple

from rubric_kit.validator import load_rubric, load_judge_panel_config, RubricValidationError
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.output import write_results, print_table, detect_format_from_extension
from rubric_kit.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.generator import RubricGenerator, parse_qa_input, parse_chat_session
from rubric_kit.schema import (
    JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig,
    Rubric, Dimension, Criterion, ToolSpec
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
    rubric_dict: Dict[str, Any] = {"dimensions": []}
    
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
    """Write a rubric to a YAML file."""
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


def get_output_format(args, output_file: str) -> str:
    """Determine output format from args or file extension."""
    if hasattr(args, 'format') and args.format:
        return args.format
    return detect_format_from_extension(output_file)


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
    rubric = load_rubric(args.rubric_file)
    print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    
    # Load or create judge panel configuration
    if args.judge_panel_config:
        print(f"\nLoading judge panel configuration from {args.judge_panel_config}...")
        panel_config = load_judge_panel_config(args.judge_panel_config)
        print(f"✓ Loaded panel with {len(panel_config.judges)} judge(s)")
    else:
        panel_config = create_default_panel_config(args)
        print(f"\n🤖 Using single judge: {args.model}")
    
    # Evaluate based on input type
    if args.qna_file:
        print(f"\nEvaluating Q&A from {args.qna_file}...")
        print_evaluation_config(panel_config)
        evaluations = evaluate_rubric_with_panel_from_qa(
            rubric,
            args.qna_file,
            panel_config
        )
    else:
        print(f"\nEvaluating chat session from {args.chat_session_file}...")
        print_evaluation_config(panel_config)
        evaluations = evaluate_rubric_with_panel(
            rubric,
            args.chat_session_file,
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
    
    # Write results
    output_format = get_output_format(args, args.output_file)
    format_name = output_format.upper()
    print(f"\nWriting results to {args.output_file} ({format_name})...")
    write_results(results, args.output_file, format=output_format, include_summary=args.include_summary)
    print(f"✓ {format_name} file written")
    
    # Print table if requested
    if not args.no_table:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80 + "\n")
        print_table(results, include_summary=True)
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
    
    # Generate rubric
    print_generation_progress(num_dimensions, num_criteria)
    
    if input_type == "chat":
        rubric = generator.generate_rubric_from_chat(
            input_obj,
            num_dimensions=num_dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints
        )
    else:
        rubric = generator.generate_rubric(
            input_obj,
            num_dimensions=num_dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints
        )
    
    print(f"✓ Generated {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    
    # Write rubric to file
    print(f"\nWriting rubric to {args.output_file}...")
    write_rubric_to_file(rubric, args.output_file)
    print(f"✓ Rubric written successfully")
    
    # Print summary
    print_rubric_summary(rubric, "GENERATED RUBRIC SUMMARY")
    
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
    # Load existing rubric
    print(f"Loading rubric from {args.rubric_file}...")
    rubric = load_rubric(args.rubric_file)
    print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
    
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
    
    # Refine rubric
    feedback_msg = f" with feedback" if args.feedback else ""
    context_msg = f" using {input_type} context" if input_type else ""
    print(f"\n🔄 Refining rubric{context_msg}{feedback_msg}...")
    if args.feedback:
        print(f"   Feedback: {args.feedback}")
    print("   This may take a moment...")
    
    if input_type == "qa":
        refined_rubric = generator.refine_rubric_with_qa(
            rubric,
            input_obj,
            feedback=args.feedback
        )
    elif input_type == "chat":
        refined_rubric = generator.refine_rubric_with_chat(
            rubric,
            input_obj,
            feedback=args.feedback
        )
    else:
        refined_rubric = generator.refine_rubric(
            rubric,
            feedback=args.feedback
        )
    
    print(f"✓ Refined rubric: {len(refined_rubric.dimensions)} dimensions, {len(refined_rubric.criteria)} criteria")
    
    # Determine output path and write
    output_path = args.output_file if args.output_file else args.rubric_file
    print(f"\nWriting refined rubric to {output_path}...")
    write_rubric_to_file(refined_rubric, output_path)
    print(f"✓ Refined rubric written successfully")
    
    # Print summary
    print_rubric_summary(refined_rubric, "REFINED RUBRIC SUMMARY")
    
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
  # Evaluate from Q&A YAML file
  %(prog)s --from-qna qna.yaml --rubric-file rubric.yaml --output-file results.csv
  
  # Evaluate from chat session file
  %(prog)s --from-chat-session chat_session.txt --rubric-file rubric.yaml --output-file results.csv
  
  # With custom model
  %(prog)s --from-chat-session chat.txt --rubric-file rubric.yaml --output-file output.csv --model gpt-4-turbo
  
  # With custom OpenAI-compatible endpoint
  %(prog)s --from-chat-session chat.txt --rubric-file rubric.yaml --output-file output.csv --base-url https://api.example.com/v1
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
        '--output-file',
        required=True,
        help='Path to output file (CSV, JSON, or YAML). Format is auto-detected from extension (.csv, .json, .yaml, .yml)'
    )
    
    evaluate_parser.add_argument(
        '--format',
        choices=['csv', 'json', 'yaml'],
        help='Output format (csv, json, yaml). If not specified, format is detected from file extension.'
    )
    
    evaluate_parser.add_argument(
        '--no-table',
        action='store_true',
        help='Do not print results table to console'
    )
    
    evaluate_parser.add_argument(
        '--include-summary',
        action='store_true',
        help='Include summary row in CSV output'
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
