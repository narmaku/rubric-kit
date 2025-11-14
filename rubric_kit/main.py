"""Main CLI entry point for rubric-kit."""

import argparse
import sys
import os

from rubric_kit.validator import load_rubric, RubricValidationError
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.output import write_csv, print_table
from rubric_kit.llm_judge import evaluate_rubric_with_llm


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Rubric Kit - Automatic rubric evaluation using LLM-as-a-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s chat_session.txt rubric.yaml results.csv
  
  # With custom model
  %(prog)s chat.txt rubric.yaml output.csv --model gpt-4-turbo
  
  # With custom OpenAI-compatible endpoint
  %(prog)s chat.txt rubric.yaml output.csv --base-url https://api.example.com/v1

Chat session file format (plain text):
  User: What are the system specifications?
  Assistant: The system has 8 CPUs and 64 GB of RAM.
  
  Tool calls:
  - get_system_information() -> {"cpus": 8, "ram_gb": 64}
"""
    )
    
    parser.add_argument(
        'chat_session_file',
        help='Path to chat session file (plain text)'
    )
    
    parser.add_argument(
        'rubric_yaml',
        help='Path to rubric YAML file'
    )
    
    parser.add_argument(
        'output_file',
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--no-table',
        action='store_true',
        help='Do not print results table to console'
    )
    
    parser.add_argument(
        '--include-summary',
        action='store_true',
        help='Include summary row in CSV output'
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    parser.add_argument(
        '--base-url',
        help='Base URL for OpenAI-compatible endpoint (optional)'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-4',
        help='Model name to use for LLM evaluation (default: gpt-4)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load rubric
        print(f"Loading rubric from {args.rubric_yaml}...")
        rubric = load_rubric(args.rubric_yaml)
        print(f"✓ Loaded {len(rubric.descriptors)} descriptors and {len(rubric.criteria)} criteria")
        
        # Get API key
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: API key required. Set OPENAI_API_KEY environment variable or use --api-key", file=sys.stderr)
            return 1
        
        # Evaluate with LLM judge
        print(f"\n🤖 Using LLM judge to evaluate chat session from {args.chat_session_file}...")
        print(f"   Model: {args.model}")
        
        evaluations = evaluate_rubric_with_llm(
            rubric,
            args.chat_session_file,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model
        )
        print(f"✓ LLM evaluated {len(evaluations)} criteria")
        
        # Process scores
        print("\nProcessing scores...")
        results = evaluate_rubric(rubric, evaluations)
        
        # Calculate scores
        total_score, max_score = calculate_total_score(results)
        percentage = calculate_percentage_score(results)
        
        print(f"✓ Evaluation complete: {total_score}/{max_score} ({percentage:.1f}%)")
        
        # Write CSV
        print(f"\nWriting results to {args.output_file}...")
        write_csv(results, args.output_file, include_summary=args.include_summary)
        print(f"✓ CSV file written")
        
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


if __name__ == "__main__":
    sys.exit(main())

