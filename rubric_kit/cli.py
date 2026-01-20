"""CLI argument parser configuration for rubric-kit."""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        description="Rubric Kit - Automatic rubric evaluation using LLM-as-a-Judge",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        help='Command to execute'
    )
    
    _add_evaluate_parser(subparsers)
    _add_generate_parser(subparsers)
    _add_refine_parser(subparsers)
    _add_export_parser(subparsers)
    _add_rerun_parser(subparsers)
    _add_arena_parser(subparsers)
    
    return parser


def _add_evaluate_parser(subparsers) -> None:
    """Configure the 'evaluate' subcommand parser."""
    parser = subparsers.add_parser(
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
    
    # Input format options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--from-qna', dest='qna_file',
        help='Path to Q&A YAML file (must contain question, answer, and optional context keys)'
    )
    input_group.add_argument(
        '--from-chat-session', dest='chat_session_file',
        help='Path to chat session file (any format, will use heuristics to parse)'
    )
    
    parser.add_argument('--rubric-file', required=True, help='Path to rubric YAML file')
    parser.add_argument('--variables-file', help='Path to variables YAML file (required if rubric has placeholders but no embedded variables)')
    parser.add_argument('--output-file', required=True, help='Path to output YAML file (source of truth artifact). Extension .yaml is added if not present.')
    parser.add_argument('--no-table', action='store_true', help='Do not print results table to console')
    parser.add_argument('--report', dest='report', help='Path to generate PDF report (optional)')
    parser.add_argument('--report-title', dest='report_title', help='Custom title for the PDF report (optional)')
    parser.add_argument('--judge-panel-config', help='Path to judge panel configuration YAML file (optional, creates single-judge panel if not provided)')
    parser.add_argument('--base-url', help='Base URL for OpenAI-compatible endpoint')
    parser.add_argument('--model', default='gpt-4', help='Model name (default: gpt-4). LiteLLM format: gpt-4, vertex_ai/gemini-2.5-flash, watsonx/llama-3, ollama/llama3')
    
    # Metrics options
    parser.add_argument('--dry-run', action='store_true', help='Estimate costs without making LLM calls')
    parser.add_argument('--no-metrics', action='store_true', help='Disable metrics collection in output')
    parser.add_argument('--include-call-log', action='store_true', help='Include detailed per-call metrics log in output')


def _add_generate_parser(subparsers) -> None:
    """Configure the 'generate' subcommand parser."""
    parser = subparsers.add_parser(
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
    
    # Input format options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--from-qna', dest='qna_file',
        help='Path to Q&A YAML file (must contain question, answer, and optional context keys)'
    )
    input_group.add_argument(
        '--from-chat-session', dest='chat_session_file',
        help='Path to chat session file (any format, will use heuristics to parse)'
    )
    
    parser.add_argument('--output-file', required=True, help='Path to output rubric YAML file')
    parser.add_argument('--num-dimensions', type=str, default="auto", help='Number of dimensions to generate (1-10 or "auto", default: auto)')
    parser.add_argument('--num-criteria', type=str, default="auto", help='Number of criteria to generate (1-10 or "auto", default: auto)')
    parser.add_argument('--categories', help='Comma-separated list of category hints (e.g., "Output,Reasoning")')
    parser.add_argument('--dimensions-file', help='Path to dimensions YAML file (skips dimension generation, uses provided dimensions)')
    parser.add_argument('--base-url', help='Base URL for OpenAI-compatible endpoint')
    parser.add_argument('--model', default='gpt-4', help='Model name (default: gpt-4). LiteLLM format: gpt-4, vertex_ai/gemini-2.5-flash, watsonx/llama-3, ollama/llama3')
    parser.add_argument('--no-variables', action='store_true', dest='no_variables', help='Do not extract variables - use hard-coded values directly in criteria')
    
    # Guidelines options (mutually exclusive)
    guidelines_group = parser.add_mutually_exclusive_group()
    guidelines_group.add_argument('--guidelines', help='Specific guidelines or hints to guide rubric generation (e.g., "Focus on security aspects")')
    guidelines_group.add_argument('--guidelines-file', dest='guidelines_file', help='Path to file containing guidelines (for long or multi-line content, e.g., AI persona descriptions)')
    
    # Metrics options
    parser.add_argument('--dry-run', action='store_true', help='Estimate costs without making LLM calls')
    parser.add_argument('--no-metrics', action='store_true', help='Disable metrics collection in output')


def _add_refine_parser(subparsers) -> None:
    """Configure the 'refine' subcommand parser."""
    parser = subparsers.add_parser(
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
    
    parser.add_argument('--rubric-file', required=True, help='Path to existing rubric YAML file')
    parser.add_argument('--variables-file', help='Path to variables YAML file (provides variable values for rubrics with placeholders)')
    
    # Input format options (mutually exclusive, optional for refine)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--from-qna', dest='qna_file', help='Path to Q&A YAML file to use as context for refinement (optional)')
    input_group.add_argument('--from-chat-session', dest='chat_session_file', help='Path to chat session file to use as context for refinement (optional)')
    
    parser.add_argument('--output-file', help='Output path for refined rubric (default: overwrite original)')
    
    # Feedback options (mutually exclusive)
    feedback_group = parser.add_mutually_exclusive_group()
    feedback_group.add_argument('--feedback', help='Specific feedback for refinement (optional)')
    feedback_group.add_argument('--feedback-file', dest='feedback_file', help='Path to file containing feedback (for long or multi-line content, e.g., AI persona descriptions)')
    
    parser.add_argument('--dimensions-file', help='Path to dimensions YAML file (merges with existing rubric dimensions)')
    parser.add_argument('--base-url', help='Base URL for OpenAI-compatible endpoint')
    parser.add_argument('--model', default='gpt-4', help='Model name (default: gpt-4). LiteLLM format: gpt-4, vertex_ai/gemini-2.5-flash, watsonx/llama-3, ollama/llama3')
    parser.add_argument('--no-variables', action='store_true', dest='no_variables', help='Do not extract variables - use hard-coded values directly in criteria')
    
    # Metrics options
    parser.add_argument('--no-metrics', action='store_true', help='Disable metrics collection in output')


def _add_export_parser(subparsers) -> None:
    """Configure the 'export' subcommand parser."""
    parser = subparsers.add_parser(
        'export',
        help='Convert evaluation YAML to PDF, CSV, or JSON format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to PDF report
  %(prog)s results.yaml --format pdf --output report.pdf
  
  # Export to CSV (includes metadata header)
  %(prog)s results.yaml --format csv --output results.csv
  
  # Export to JSON
  %(prog)s results.yaml --format json --output results.json
"""
    )
    
    parser.add_argument('input_file', help='Path to input YAML file with evaluation results')
    parser.add_argument('--format', required=True, choices=['pdf', 'csv', 'json'], help='Output format: pdf, csv, or json')
    parser.add_argument('--output', '-o', dest='output_file', required=True, help='Path to output file')


def _add_rerun_parser(subparsers) -> None:
    """Configure the 'rerun' subcommand parser."""
    parser = subparsers.add_parser(
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
    
    parser.add_argument('input_file', help='Path to self-contained evaluation YAML file')
    parser.add_argument('--output-file', '-o', required=True, help='Path to output YAML file')
    
    # Optional new input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--from-qna', dest='qna_file', help='Path to new Q&A YAML file (overrides embedded/original input)')
    input_group.add_argument('--from-chat-session', dest='chat_session_file', help='Path to new chat session file (overrides embedded/original input)')
    
    parser.add_argument('--report', help='Path to generate PDF report (optional)')
    parser.add_argument('--report-title', dest='report_title', help='Custom title for the PDF report (optional, overrides original)')
    parser.add_argument('--no-table', action='store_true', help='Do not print results table to console')


def _add_arena_parser(subparsers) -> None:
    """Configure the 'arena' subcommand parser."""
    parser = subparsers.add_parser(
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
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--arena-spec', dest='arena_spec', help='Path to arena specification YAML file (runs fresh evaluations)')
    input_group.add_argument('--from-outputs', dest='output_files', nargs='+', metavar='OUTPUT_FILE', help='Combine multiple existing output.yaml files into arena results')
    
    parser.add_argument('--output-file', '-o', required=True, help='Path to output YAML file with aggregated results')
    parser.add_argument('--report', help='Path to generate Arena PDF report (optional)')
    parser.add_argument('--report-title', dest='report_title', help='Custom title for the Arena PDF report (optional)')
    parser.add_argument('--no-table', action='store_true', help='Do not print rankings table to console')
    parser.add_argument('--force', action='store_true', help='Force re-evaluation of all contestants (ignore cached results)')

