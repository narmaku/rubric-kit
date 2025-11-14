"""Main CLI entry point for rubric-kit."""

import argparse
import json
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from rubric_kit.validator import load_rubric, RubricValidationError
from rubric_kit.processor import evaluate_rubric, calculate_total_score, calculate_percentage_score
from rubric_kit.output import write_csv, print_table


def load_evaluations(file_path: str) -> Dict[str, Any]:
    """
    Load evaluation data from a JSON or YAML file.
    
    Args:
        file_path: Path to the evaluations file
        
    Returns:
        Dictionary of evaluations
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Evaluations file not found: {file_path}")
    
    suffix = path.suffix.lower()
    
    try:
        with open(path, 'r') as f:
            if suffix == '.json':
                return json.load(f)
            elif suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}. Use .json, .yaml, or .yml")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Error parsing evaluations file: {e}")


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Rubric Kit - Validate and score rubric evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evaluations.json rubric.yaml results.csv
  %(prog)s eval.yaml rubric.yaml output.csv

Evaluation file format (JSON or YAML):
  {
    "criterion_name": {
      "type": "binary",
      "passes": true
    },
    "another_criterion": {
      "type": "score",
      "score": 3
    }
  }
"""
    )
    
    parser.add_argument(
        'prompt_file',
        help='Path to evaluations file (JSON or YAML)'
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
    
    args = parser.parse_args()
    
    try:
        # Load rubric
        print(f"Loading rubric from {args.rubric_yaml}...")
        rubric = load_rubric(args.rubric_yaml)
        print(f"✓ Loaded {len(rubric.descriptors)} descriptors and {len(rubric.criteria)} criteria")
        
        # Load evaluations
        print(f"\nLoading evaluations from {args.prompt_file}...")
        evaluations = load_evaluations(args.prompt_file)
        print(f"✓ Loaded evaluations for {len(evaluations)} criteria")
        
        # Evaluate rubric
        print("\nEvaluating rubric...")
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

