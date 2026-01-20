"""Output handlers for CSV, JSON, YAML and table display."""

import csv
import json
import yaml
import os
from typing import List, Dict, Any, Tuple
from tabulate import tabulate

# Constants
DEFAULT_CSV_FIELDS = ["criterion_name", "category", "dimension", "result", "score", "max_score"]
PRIORITY_FIELDS = ["criterion_name", "category", "dimension", "criterion_text", "result", "score", "max_score", "reason"]
FORMAT_EXTENSIONS = {
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.csv': 'csv'
}


def _extract_judge_vote_value(vote: Dict[str, Any]) -> str | int:
    """Extract vote value from a judge vote dictionary."""
    if "passes" in vote:
        return "pass" if vote["passes"] else "fail"
    if "score" in vote:
        return vote["score"]
    return ""


def _expand_judge_votes(result: Dict[str, Any]) -> Dict[str, Any]:
    """Expand judge_votes into separate columns for a single result."""
    expanded_result = result.copy()
    judge_votes = result.get("judge_votes", [])
    
    if not isinstance(judge_votes, list):
        return expanded_result
    
    expanded_result.pop("judge_votes", None)
    
    for vote in judge_votes:
        judge_name = vote.get("judge", "unknown")
        vote_value = _extract_judge_vote_value(vote)
        
        if vote_value:
            expanded_result[f"judge_{judge_name}_vote"] = vote_value
        
        if "reason" in vote:
            expanded_result[f"judge_{judge_name}_reason"] = vote["reason"]
    
    return expanded_result


def _order_fieldnames(fieldnames: List[str]) -> List[str]:
    """Order fieldnames with priority fields first, judge fields last."""
    fieldnames_set = set(fieldnames)
    judge_fields = [f for f in fieldnames if f.startswith("judge_")]
    other_fields = [f for f in fieldnames if f not in PRIORITY_FIELDS and not f.startswith("judge_")]
    
    ordered = [f for f in PRIORITY_FIELDS if f in fieldnames_set]
    ordered.extend(sorted(other_fields))
    ordered.extend(sorted(judge_fields))
    
    return ordered


def _prepare_data_for_csv(
    results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Prepare results data for CSV format by expanding judge_votes.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Tuple of (expanded_results, fieldnames)
    """
    if not results:
        return [], DEFAULT_CSV_FIELDS
    
    expanded_results = [_expand_judge_votes(result) for result in results]
    
    fieldnames_set = set()
    for result in expanded_results:
        fieldnames_set.update(result.keys())
    
    fieldnames = _order_fieldnames(list(fieldnames_set))
    
    return expanded_results, fieldnames


def _calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate summary statistics from results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "criterion_name": "TOTAL",
            "score": 0,
            "max_score": 0,
            "result": "0.0%"
        }
    
    total_score = sum(r["score"] for r in results)
    max_score = sum(r["max_score"] for r in results)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    return {
        "criterion_name": "TOTAL",
        "score": total_score,
        "max_score": max_score,
        "result": f"{percentage:.1f}%"
    }


def _create_summary_row(fieldnames: List[str], summary: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary row with empty values for all fields, then update with summary data."""
    summary_row = {key: "" for key in fieldnames}
    summary_row.update(summary)
    return summary_row


def write_csv(
    results: List[Dict[str, Any]], 
    output_path: str, 
    include_summary: bool = False
) -> None:
    """
    Write evaluation results to a CSV file.
    
    Args:
        results: List of evaluation results
        output_path: Path to output CSV file
        include_summary: Whether to include summary row
    """
    expanded_results, fieldnames = _prepare_data_for_csv(results)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in expanded_results:
            writer.writerow(result)
        
        if include_summary:
            summary = _calculate_summary(results)
            summary_row = _create_summary_row(fieldnames, summary)
            writer.writerow(summary_row)


def _prepare_structured_output(
    results: List[Dict[str, Any]], 
    include_summary: bool,
    metadata: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Prepare output data structure with optional summary and metadata."""
    output_data = {"results": results}
    
    if include_summary:
        output_data["summary"] = _calculate_summary(results)
    
    if metadata:
        output_data["metadata"] = metadata
    
    return output_data


def write_json(
    results: List[Dict[str, Any]], 
    output_path: str, 
    include_summary: bool = False,
    metadata: Dict[str, Any] | None = None
) -> None:
    """
    Write evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        output_path: Path to output JSON file
        include_summary: Whether to include summary in output
        metadata: Optional metadata dictionary (rubric_file, timestamp, etc.)
    """
    output_data = _prepare_structured_output(results, include_summary, metadata)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def write_yaml(
    results: List[Dict[str, Any]], 
    output_path: str, 
    include_summary: bool = False,
    metadata: Dict[str, Any] | None = None
) -> None:
    """
    Write evaluation results to a YAML file.
    
    Args:
        results: List of evaluation results
        output_path: Path to output YAML file
        include_summary: Whether to include summary in output
        metadata: Optional metadata dictionary (rubric_file, timestamp, etc.)
    """
    output_data = _prepare_structured_output(results, include_summary, metadata)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, 
                 sort_keys=False, 
                 default_flow_style=False,
                 allow_unicode=True)


def detect_format_from_extension(output_path: str) -> str:
    """
    Detect output format from file extension.
    
    Args:
        output_path: Path to output file
        
    Returns:
        Format string: 'csv', 'json', or 'yaml'
    """
    ext = os.path.splitext(output_path)[1].lower()
    return FORMAT_EXTENSIONS.get(ext, 'csv')


def write_results(
    results: List[Dict[str, Any]], 
    output_path: str, 
    format: str | None = None,
    include_summary: bool = False,
    metadata: Dict[str, Any] | None = None
) -> None:
    """
    Write evaluation results to a file in the specified format.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format ('csv', 'json', 'yaml'). If None, detected from file extension.
        include_summary: Whether to include summary in output
        metadata: Optional metadata dictionary (rubric_file, timestamp, etc.)
    """
    if format is None:
        format = detect_format_from_extension(output_path)
    
    format = format.lower()
    
    writers = {
        'csv': write_csv,
        'json': write_json,
        'yaml': write_yaml
    }
    
    writer = writers.get(format)
    if writer is None:
        raise ValueError(f"Unsupported format: {format}. Supported formats: csv, json, yaml")
    
    # CSV doesn't support metadata (flat format), only JSON/YAML
    if format == 'csv':
        writer(results, output_path, include_summary=include_summary)
    else:
        writer(results, output_path, include_summary=include_summary, metadata=metadata)


def _format_consensus_indicator(consensus_reached: bool) -> str:
    """Format consensus indicator symbol."""
    return "✓" if consensus_reached else "⚠"


def _format_agreement(result: Dict[str, Any]) -> str:
    """Format agreement as 'X/Y' or 'N/A'."""
    consensus_count = result.get("consensus_count")
    if consensus_count is None:
        return "N/A"
    
    judge_votes = result.get("judge_votes", [])
    total_judges = len(judge_votes) if judge_votes else consensus_count
    return f"{consensus_count}/{total_judges}"


def _format_result_row(result: Dict[str, Any]) -> List[str]:
    """Format a single result into a table row."""
    score = result.get("score", 0)
    max_score = result.get("max_score", 0)
    consensus_reached = result.get("consensus_reached", True)
    
    return [
        result.get("criterion_name", ""),
        result.get("dimension", ""),
        str(result.get("result", "")),
        f"{score}/{max_score}",
        _format_consensus_indicator(consensus_reached),
        _format_agreement(result)
    ]


def _add_summary_rows(rows: List[List[str]], results: List[Dict[str, Any]]) -> None:
    """Add separator and summary rows to the table."""
    total_score = sum(r["score"] for r in results)
    max_score = sum(r["max_score"] for r in results)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    separator = ["─" * 20, "─" * 10, "─" * 10, "─" * 10, "─" * 9, "─" * 10]
    summary = ["TOTAL", "", f"{percentage:.1f}%", f"{total_score}/{max_score}", "", ""]
    
    rows.append(separator)
    rows.append(summary)


def format_table(results: List[Dict[str, Any]], include_summary: bool = True) -> str:
    """
    Format evaluation results as a pretty table.
    
    Shows only essential information for readability.
    Full details are available in the CSV output.
    
    Args:
        results: List of evaluation results
        include_summary: Whether to include summary row
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."
    
    headers = ["Criterion", "Dimension", "Result", "Score", "Consensus", "Agreement"]
    rows = [_format_result_row(result) for result in results]
    
    if include_summary:
        _add_summary_rows(rows, results)
    
    return tabulate(rows, headers=headers, tablefmt="grid")


def print_table(results: List[Dict[str, Any]], include_summary: bool = True) -> None:
    """
    Print evaluation results as a pretty table to stdout.
    
    Args:
        results: List of evaluation results
        include_summary: Whether to include summary row
    """
    table = format_table(results, include_summary=include_summary)
    print(table)


def _format_tool_breakdown_row(tool_result: Dict[str, Any]) -> List[str]:
    """Format a single tool result into a table row."""
    called = "✓" if tool_result.get("called") else "✗"
    count_ok = "✓" if tool_result.get("count_ok") else "✗"
    params_ok = tool_result.get("params_ok")
    params = "✓" if params_ok else ("✗" if params_ok is False else "N/A")
    
    score = tool_result.get("score", 0)
    max_score = tool_result.get("max_score", 0)
    
    return [
        tool_result.get("name", ""),  # Full tool name, no truncation
        tool_result.get("type", ""),
        called,
        f"{tool_result.get('count', 0)} {count_ok}",
        params,
        f"{score:.1f}/{max_score:.1f}"
    ]


def format_tool_breakdown(breakdown: Dict[str, Any]) -> str:
    """
    Format a tool breakdown as a pretty table.
    
    Args:
        breakdown: Tool breakdown dictionary with tool_results
        
    Returns:
        Formatted table string
    """
    tool_results = breakdown.get("tool_results", [])
    if not tool_results:
        return ""
    
    # Header info
    order_status = "✓" if breakdown.get("order_ok") else ("✗" if breakdown.get("order_ok") is False else "N/A")
    header = f"Score: {breakdown.get('overall_score', 0):.1f}/3 | Order: {order_status}"
    
    # Table
    headers = ["Tool", "Type", "Called", "Count", "Params", "Score"]
    rows = [_format_tool_breakdown_row(tr) for tr in tool_results]
    
    table = tabulate(rows, headers=headers, tablefmt="simple")
    
    # Issues
    issues = breakdown.get("issues", [])
    issues_text = ""
    if issues:
        issues_text = "\nIssues:\n" + "\n".join(f"  • {issue}" for issue in issues[:5])
    
    # Summary
    summary = breakdown.get("summary", "")
    summary_text = f"\nSummary: {summary}" if summary else ""
    
    return f"{header}\n{table}{issues_text}{summary_text}"


def print_tool_breakdowns(results: List[Dict[str, Any]]) -> None:
    """
    Print tool breakdowns for all results that have them.
    
    Args:
        results: List of evaluation results (some may have tool_breakdown)
    """
    breakdowns = [
        (r.get("criterion_name", "unknown"), r["tool_breakdown"])
        for r in results
        if r.get("tool_breakdown")
    ]
    
    if not breakdowns:
        return
    
    print("\n" + "=" * 60)
    print("TOOL CALLS BREAKDOWNS")
    print("=" * 60)
    
    for criterion_name, breakdown in breakdowns:
        print(f"\n--- {criterion_name} ---")
        print(format_tool_breakdown(breakdown))
    
    print()


def _load_yaml_data(input_path: str) -> Dict[str, Any]:
    """
    Load data from a YAML file.
    
    Args:
        input_path: Path to input YAML file
        
    Returns:
        Dictionary with the loaded data
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to max length, adding ellipsis if needed."""
    if not text:
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def _format_csv_header_comments(data: Dict[str, Any]) -> List[str]:
    """
    Generate CSV header comment lines with metadata and input summary.
    
    Returns list of comment lines (starting with #).
    """
    lines = ["# Evaluation Report"]
    lines.append("#")
    
    # Metadata
    metadata = data.get("metadata", {})
    if metadata.get("timestamp"):
        lines.append(f"# Evaluated: {metadata['timestamp']}")
    if metadata.get("rubric_source_file"):
        lines.append(f"# Rubric: {metadata['rubric_source_file']}")
    
    # Summary
    summary = data.get("summary", {})
    if summary:
        score = summary.get("total_score", 0)
        max_score = summary.get("max_score", 0)
        pct = summary.get("percentage", 0)
        lines.append(f"# Score: {score}/{max_score} ({pct}%)")
    
    lines.append("#")
    
    # Input content
    input_data = data.get("input", {})
    input_type = input_data.get("type", "unknown")
    source_file = input_data.get("source_file", "")
    
    if source_file:
        lines.append(f"# Input Source: {source_file}")
    lines.append(f"# Input Type: {input_type}")
    
    if input_type == "qna":
        question = input_data.get("question", "")
        answer = input_data.get("answer", "")
        context = input_data.get("context", "")
        
        if question:
            lines.append(f"# Question: {_truncate_text(question, 150)}")
        if context:
            lines.append(f"# Context: {_truncate_text(context, 100)}")
        if answer:
            lines.append(f"# Answer: {_truncate_text(answer, 300)}")
    else:
        chat_session = input_data.get("chat_session", "")
        if chat_session:
            lines.append(f"# Chat Session: {_truncate_text(chat_session, 400)}")
    
    lines.append("#")
    lines.append("# " + "=" * 60)
    lines.append("#")
    
    return lines


def convert_yaml_to_csv(input_path: str, output_path: str) -> None:
    """
    Convert evaluation results from YAML to CSV format.
    
    Includes header comments with metadata and input summary for complete,
    self-contained exports.
    
    Args:
        input_path: Path to input YAML file
        output_path: Path to output CSV file
    """
    data = _load_yaml_data(input_path)
    results = data.get("results", [])
    
    # Generate header comments
    header_comments = _format_csv_header_comments(data)
    
    # Prepare results data
    expanded_results, fieldnames = _prepare_data_for_csv(results)
    
    # Include summary if present
    include_summary = "summary" in data
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Write header comments
        for line in header_comments:
            f.write(line + '\n')
        
        # Write CSV data
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in expanded_results:
            writer.writerow(result)
        
        if include_summary:
            summary = _calculate_summary(results)
            summary_row = _create_summary_row(fieldnames, summary)
            writer.writerow(summary_row)


def convert_yaml_to_json(input_path: str, output_path: str) -> None:
    """
    Convert evaluation results from YAML to JSON format.
    
    Always includes full input content for complete, self-contained exports.
    
    Args:
        input_path: Path to input YAML file
        output_path: Path to output JSON file
    """
    data = _load_yaml_data(input_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

