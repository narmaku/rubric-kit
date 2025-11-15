"""Main CLI entry point for rubric-kit."""

import argparse
import sys
import os
import yaml

from rubric_kit.validator import load_rubric, load_judge_panel_config, RubricValidationError
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.output import write_results, print_table
from rubric_kit.llm_judge import evaluate_rubric_with_panel, evaluate_rubric_with_panel_from_qa
from rubric_kit.generator import RubricGenerator, parse_qa_input, parse_chat_session
from rubric_kit.schema import JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig




def cmd_evaluate(args) -> int:
    """
    Execute the 'evaluate' subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Load rubric
        print(f"Loading rubric from {args.rubric_yaml}...")
        rubric = load_rubric(args.rubric_yaml)
        print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
        
        # Load or create judge panel configuration
        if args.judge_panel_config:
            print(f"\nLoading judge panel configuration from {args.judge_panel_config}...")
            panel_config = load_judge_panel_config(args.judge_panel_config)
            print(f"✓ Loaded panel with {len(panel_config.judges)} judge(s)")
        else:
            # Create default single-judge panel
            api_key = args.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Error: API key required. Set OPENAI_API_KEY environment variable or use --api-key", file=sys.stderr)
                return 1
            
            panel_config = JudgePanelConfig(
                judges=[JudgeConfig(
                    name="default",
                    model=args.model,
                    api_key=api_key,
                    base_url=args.base_url
                )],
                execution=ExecutionConfig(mode="sequential"),
                consensus=ConsensusConfig(mode="unanimous")
            )
            print(f"\n🤖 Using single judge: {args.model}")
        
        # Determine input type and evaluate
        if args.qna_file:
            print(f"\nEvaluating Q&A from {args.qna_file}...")
            print(f"   Execution mode: {panel_config.execution.mode}")
            print(f"   Consensus mode: {panel_config.consensus.mode}")
            if panel_config.consensus.mode in ("quorum", "majority"):
                print(f"   Consensus threshold: {panel_config.consensus.threshold}")
            
            evaluations = evaluate_rubric_with_panel_from_qa(
                rubric,
                args.qna_file,
                panel_config
            )
        else:
            print(f"\nEvaluating chat session from {args.chat_session_file}...")
            print(f"   Execution mode: {panel_config.execution.mode}")
            print(f"   Consensus mode: {panel_config.consensus.mode}")
            if panel_config.consensus.mode in ("quorum", "majority"):
                print(f"   Consensus threshold: {panel_config.consensus.threshold}")
            
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
        
        # Determine output format
        output_format = getattr(args, 'format', None)
        if output_format is None:
            from rubric_kit.output import detect_format_from_extension
            output_format = detect_format_from_extension(args.output_file)
        
        format_name = output_format.upper()
        print(f"\nWriting results to {args.output_file} ({format_name})...")
        write_results(results, args.output_file, format=output_format, include_summary=args.include_summary)
        print(f"✓ {format_name} file written")
        
        # Print table
        if not args.no_table:
            print("\n" + "=" * 80)
            print("RESULTS")
            print("=" * 80 + "\n")
            print_table(results, include_summary=True)
            print()
        
        return 0
        
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
        import traceback
        traceback.print_exc()
        return 1


def cmd_generate(args) -> int:
    """
    Execute the 'generate' subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Get API key
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: API key required. Set OPENAI_API_KEY environment variable or use --api-key", file=sys.stderr)
            return 1
        
        # Determine input type and parse accordingly
        if args.qna_file:
            # Parse Q&A YAML file
            print(f"Loading Q&A from {args.qna_file}...")
            qa_input = parse_qa_input(args.qna_file)
            print(f"✓ Loaded Q&A pair")
            print(f"   Q: {qa_input.question[:80]}{'...' if len(qa_input.question) > 80 else ''}")
            input_obj = qa_input
            input_type = "qa"
        else:
            # Parse chat session file
            print(f"Loading chat session from {args.chat_session_file}...")
            chat_input = parse_chat_session(args.chat_session_file)
            print(f"✓ Loaded chat session")
            print(f"   Content length: {len(chat_input.content)} characters")
            print(f"   The LLM will analyze the session to detect tool calls and structure")
            input_obj = chat_input
            input_type = "chat"
        
        # Parse category hints if provided
        category_hints = None
        if args.categories:
            category_hints = [c.strip() for c in args.categories.split(',')]
            print(f"   Category hints: {', '.join(category_hints)}")
        
        # Initialize generator
        print(f"\n🤖 Initializing rubric generator...")
        print(f"   Model: {args.model}")
        generator = RubricGenerator(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url
        )
        
        # Parse dimension and criteria counts (support "auto" keyword)
        num_dimensions = None if args.num_dimensions == "auto" else int(args.num_dimensions)
        num_criteria = None if args.num_criteria == "auto" else int(args.num_criteria)
        
        # Generate rubric
        if num_dimensions is None and num_criteria is None:
            print(f"\n🔄 Generating rubric with auto-detected dimensions and criteria...")
        elif num_dimensions is None:
            print(f"\n🔄 Generating rubric with auto-detected dimensions and {num_criteria} criteria...")
        elif num_criteria is None:
            print(f"\n🔄 Generating rubric with {num_dimensions} dimensions and auto-detected criteria...")
        else:
            print(f"\n🔄 Generating rubric with {num_dimensions} dimensions and {num_criteria} criteria...")
        print("   This may take a moment...")
        
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
        
        # Convert rubric to YAML format
        rubric_dict = {
            "dimensions": []
        }
        
        for dim in rubric.dimensions:
            dim_dict = {
                dim.name: dim.description,
                "grading_type": dim.grading_type
            }
            if dim.scores:
                dim_dict["scores"] = dim.scores
            rubric_dict["dimensions"].append(dim_dict)
        
        rubric_dict["criteria"] = {}
        for criterion in rubric.criteria:
            crit_dict = {
                "category": criterion.category,
                "weight": criterion.weight,
                "dimension": criterion.dimension,
                "criterion": criterion.criterion
            }
            # Add tool_calls if present
            if criterion.tool_calls:
                # Use list format where each item is a dict with tool name as key
                required_list = []
                for tc in criterion.tool_calls.required:
                    tool_dict = {}
                    if tc.min_calls is not None:
                        tool_dict["min_calls"] = tc.min_calls
                    if tc.max_calls is not None:
                        tool_dict["max_calls"] = tc.max_calls
                    if tc.params:
                        tool_dict["params"] = tc.params
                    # Create dict with tool name as key
                    required_list.append({tc.name: tool_dict if tool_dict else None})
                
                optional_list = []
                for tc in criterion.tool_calls.optional:
                    tool_dict = {}
                    if tc.max_calls is not None:
                        tool_dict["max_calls"] = tc.max_calls
                    if tc.params:
                        tool_dict["params"] = tc.params
                    optional_list.append({tc.name: tool_dict if tool_dict else None})
                
                prohibited_list = []
                for tc in criterion.tool_calls.prohibited:
                    prohibited_list.append({tc.name: None})
                
                crit_dict["tool_calls"] = {
                    "respect_order": criterion.tool_calls.respect_order,
                    "required": required_list,
                    "optional": optional_list if optional_list else [],
                    "prohibited": prohibited_list if prohibited_list else []
                }
            rubric_dict["criteria"][criterion.name] = crit_dict
        
        # Write to file
        print(f"\nWriting rubric to {args.output_yaml}...")
        with open(args.output_yaml, 'w') as f:
            yaml.dump(rubric_dict, f, 
                     sort_keys=False, 
                     default_flow_style=False,
                     allow_unicode=True)
        
        print(f"✓ Rubric written successfully")
        
        # Print summary
        print("\n" + "=" * 80)
        print("GENERATED RUBRIC SUMMARY")
        print("=" * 80)
        print(f"\nDimensions ({len(rubric.dimensions)}):")
        for dim in rubric.dimensions:
            print(f"  • {dim.name} ({dim.grading_type})")
        
        print(f"\nCriteria ({len(rubric.criteria)}):")
        for crit in rubric.criteria:
            print(f"  • {crit.name} [{crit.category}] - {crit.dimension}")
        print()
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_refine(args) -> int:
    """
    Execute the 'refine' subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Get API key
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: API key required. Set OPENAI_API_KEY environment variable or use --api-key", file=sys.stderr)
            return 1
        
        # Load existing rubric
        print(f"Loading rubric from {args.rubric_yaml}...")
        rubric = load_rubric(args.rubric_yaml)
        print(f"✓ Loaded {len(rubric.dimensions)} dimensions and {len(rubric.criteria)} criteria")
        
        # Initialize generator
        print(f"\n🤖 Initializing rubric refiner...")
        print(f"   Model: {args.model}")
        generator = RubricGenerator(
            api_key=api_key,
            model=args.model,
            base_url=args.base_url
        )
        
        # Refine rubric
        feedback_msg = f" with feedback" if args.feedback else ""
        print(f"\n🔄 Refining rubric{feedback_msg}...")
        if args.feedback:
            print(f"   Feedback: {args.feedback}")
        print("   This may take a moment...")
        
        refined_rubric = generator.refine_rubric(
            rubric,
            feedback=args.feedback
        )
        
        print(f"✓ Refined rubric: {len(refined_rubric.dimensions)} dimensions, {len(refined_rubric.criteria)} criteria")
        
        # Convert rubric to YAML format
        rubric_dict = {
            "dimensions": []
        }
        
        for dim in refined_rubric.dimensions:
            dim_dict = {
                dim.name: dim.description,
                "grading_type": dim.grading_type
            }
            if dim.scores:
                dim_dict["scores"] = dim.scores
            rubric_dict["dimensions"].append(dim_dict)
        
        rubric_dict["criteria"] = {}
        for criterion in refined_rubric.criteria:
            crit_dict = {
                "category": criterion.category,
                "weight": criterion.weight,
                "dimension": criterion.dimension,
                "criterion": criterion.criterion
            }
            # Add tool_calls if present
            if criterion.tool_calls:
                # Use list format where each item is a dict with tool name as key
                required_list = []
                for tc in criterion.tool_calls.required:
                    tool_dict = {}
                    if tc.min_calls is not None:
                        tool_dict["min_calls"] = tc.min_calls
                    if tc.max_calls is not None:
                        tool_dict["max_calls"] = tc.max_calls
                    if tc.params:
                        tool_dict["params"] = tc.params
                    # Create dict with tool name as key
                    required_list.append({tc.name: tool_dict if tool_dict else None})
                
                optional_list = []
                for tc in criterion.tool_calls.optional:
                    tool_dict = {}
                    if tc.max_calls is not None:
                        tool_dict["max_calls"] = tc.max_calls
                    if tc.params:
                        tool_dict["params"] = tc.params
                    optional_list.append({tc.name: tool_dict if tool_dict else None})
                
                prohibited_list = []
                for tc in criterion.tool_calls.prohibited:
                    prohibited_list.append({tc.name: None})
                
                crit_dict["tool_calls"] = {
                    "respect_order": criterion.tool_calls.respect_order,
                    "required": required_list,
                    "optional": optional_list if optional_list else [],
                    "prohibited": prohibited_list if prohibited_list else []
                }
            rubric_dict["criteria"][criterion.name] = crit_dict
        
        # Determine output path
        output_path = args.output if args.output else args.rubric_yaml
        
        # Write to file
        print(f"\nWriting refined rubric to {output_path}...")
        with open(output_path, 'w') as f:
            yaml.dump(rubric_dict, f, 
                     sort_keys=False, 
                     default_flow_style=False,
                     allow_unicode=True)
        
        print(f"✓ Refined rubric written successfully")
        
        # Print summary
        print("\n" + "=" * 80)
        print("REFINED RUBRIC SUMMARY")
        print("=" * 80)
        print(f"\nDimensions ({len(refined_rubric.dimensions)}):")
        for dim in refined_rubric.dimensions:
            print(f"  • {dim.name} ({dim.grading_type})")
        
        print(f"\nCriteria ({len(refined_rubric.criteria)}):")
        for crit in refined_rubric.criteria:
            print(f"  • {crit.name} [{crit.category}] - {crit.dimension}")
        print()
        
        return 0
        
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
        import traceback
        traceback.print_exc()
        return 1


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
  %(prog)s --from-qna qna.yaml rubric.yaml results.csv
  
  # Evaluate from chat session file
  %(prog)s --from-chat-session chat_session.txt rubric.yaml results.csv
  
  # With custom model
  %(prog)s --from-chat-session chat.txt rubric.yaml output.csv --model gpt-4-turbo
  
  # With custom OpenAI-compatible endpoint
  %(prog)s --from-chat-session chat.txt rubric.yaml output.csv --base-url https://api.example.com/v1
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
        'rubric_yaml',
        help='Path to rubric YAML file'
    )
    
    evaluate_parser.add_argument(
        'output_file',
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
  %(prog)s --from-qna qna.yaml output_rubric.yaml
  
  # Generate from chat session file
  %(prog)s --from-chat-session session.txt output_rubric.yaml
  
  # With custom parameters
  %(prog)s --from-qna qna.yaml rubric.yaml --num-dimensions 5 --num-criteria 8
  
  # With category hints
  %(prog)s --from-chat-session session.txt rubric.yaml --categories "Tools,Output,Reasoning"
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
        'output_yaml',
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
  %(prog)s rubric.yaml
  
  # With feedback
  %(prog)s rubric.yaml --feedback "Add more specific criteria"
  
  # With custom output path
  %(prog)s rubric.yaml --output refined_rubric.yaml
"""
    )
    
    refine_parser.add_argument(
        'rubric_yaml',
        help='Path to existing rubric YAML file'
    )
    
    refine_parser.add_argument(
        '--feedback',
        help='Specific feedback for refinement (optional)'
    )
    
    refine_parser.add_argument(
        '--output',
        help='Output path for refined rubric (default: overwrite original)'
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
