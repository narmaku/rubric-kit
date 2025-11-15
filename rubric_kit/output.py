"""Output handlers for CSV, JSON, YAML and table display."""

import csv
import json
import yaml
import os
from typing import List, Dict, Any, Tuple
from tabulate import tabulate


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
        return [], ["criterion_name", "category", "dimension", "result", "score", "max_score"]
    
    # Expand judge_votes into separate columns
    expanded_results = []
    judge_names = set()
    
    for result in results:
        expanded_result = {}
        for key, value in result.items():
            if key == "judge_votes" and isinstance(value, list):
                # Extract judge votes into separate columns
                for vote in value:
                    judge_name = vote.get("judge", "unknown")
                    judge_names.add(judge_name)
                    
                    # Add vote details as separate columns
                    if "passes" in vote:
                        expanded_result[f"judge_{judge_name}_vote"] = "pass" if vote["passes"] else "fail"
                    elif "score" in vote:
                        expanded_result[f"judge_{judge_name}_vote"] = vote["score"]
                    
                    if "reason" in vote:
                        expanded_result[f"judge_{judge_name}_reason"] = vote["reason"]
            else:
                expanded_result[key] = value
        
        expanded_results.append(expanded_result)
    
    # Get all unique keys from expanded results
    fieldnames_set = set()
    for result in expanded_results:
        fieldnames_set.update(result.keys())
    fieldnames = sorted(list(fieldnames_set))
    
    # Ensure common fields come first, judge columns at the end
    priority_fields = ["criterion_name", "category", "dimension", "criterion_text", "result", "score", "max_score", "reason"]
    judge_fields = [f for f in fieldnames if f.startswith("judge_")]
    other_fields = [f for f in fieldnames if f not in priority_fields and not f.startswith("judge_")]
    
    ordered_fieldnames = []
    for field in priority_fields:
        if field in fieldnames:
            ordered_fieldnames.append(field)
    ordered_fieldnames.extend(sorted(other_fields))
    ordered_fieldnames.extend(sorted(judge_fields))
    fieldnames = ordered_fieldnames
    
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
    
    if not expanded_results:
        # Write empty CSV with headers
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write result rows
        for result in expanded_results:
            writer.writerow(result)
        
        # Add summary row if requested
        if include_summary:
            summary = _calculate_summary(results)
            summary_row = {key: "" for key in fieldnames}
            summary_row.update(summary)
            writer.writerow(summary_row)


def write_json(
    results: List[Dict[str, Any]], 
    output_path: str, 
    include_summary: bool = False
) -> None:
    """
    Write evaluation results to a JSON file.
    
    Args:
        results: List of evaluation results
        output_path: Path to output JSON file
        include_summary: Whether to include summary in output
    """
    output_data = {
        "results": results
    }
    
    if include_summary:
        output_data["summary"] = _calculate_summary(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def write_yaml(
    results: List[Dict[str, Any]], 
    output_path: str, 
    include_summary: bool = False
) -> None:
    """
    Write evaluation results to a YAML file.
    
    Args:
        results: List of evaluation results
        output_path: Path to output YAML file
        include_summary: Whether to include summary in output
    """
    output_data = {
        "results": results
    }
    
    if include_summary:
        output_data["summary"] = _calculate_summary(results)
    
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
    
    if ext == '.json':
        return 'json'
    elif ext in ('.yaml', '.yml'):
        return 'yaml'
    elif ext == '.csv':
        return 'csv'
    else:
        # Default to CSV if extension is not recognized
        return 'csv'


def write_results(
    results: List[Dict[str, Any]], 
    output_path: str, 
    format: str | None = None,
    include_summary: bool = False
) -> None:
    """
    Write evaluation results to a file in the specified format.
    
    Args:
        results: List of evaluation results
        output_path: Path to output file
        format: Output format ('csv', 'json', 'yaml'). If None, detected from file extension.
        include_summary: Whether to include summary in output
    """
    # Auto-detect format from extension if not specified
    if format is None:
        format = detect_format_from_extension(output_path)
    
    format = format.lower()
    
    if format == 'csv':
        write_csv(results, output_path, include_summary=include_summary)
    elif format == 'json':
        write_json(results, output_path, include_summary=include_summary)
    elif format == 'yaml':
        write_yaml(results, output_path, include_summary=include_summary)
    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: csv, json, yaml")


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
    
    # Prepare table data with simplified columns
    headers = ["Criterion", "Dimension", "Result", "Score", "Consensus", "Agreement"]
    rows = []
    
    for result in results:
        result_str = str(result.get("result", ""))
        score = result.get("score", 0)
        max_score = result.get("max_score", 0)
        
        # Format consensus indicator
        consensus_reached = result.get("consensus_reached", True)
        consensus_indicator = "✓" if consensus_reached else "⚠"
        
        # Format agreement as "2/3" or "N/A" if not available
        consensus_count = result.get("consensus_count")
        if consensus_count is not None:
            # Try to determine total judges from judge_votes if available
            judge_votes = result.get("judge_votes", [])
            total_judges = len(judge_votes) if judge_votes else consensus_count
            agreement = f"{consensus_count}/{total_judges}"
        else:
            agreement = "N/A"
        
        row = [
            result.get("criterion_name", ""),
            result.get("dimension", ""),
            result_str,
            f"{score}/{max_score}",
            consensus_indicator,
            agreement
        ]
        rows.append(row)
    
    # Add summary row if requested
    if include_summary:
        total_score = sum(r["score"] for r in results)
        max_score = sum(r["max_score"] for r in results)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        rows.append([
            "─" * 20,
            "─" * 10,
            "─" * 10,
            "─" * 10,
            "─" * 9,
            "─" * 10
        ])
        
        rows.append([
            "TOTAL",
            "",
            f"{percentage:.1f}%",
            f"{total_score}/{max_score}",
            "",
            ""
        ])
    
    # Format as table
    table = tabulate(rows, headers=headers, tablefmt="grid")
    return table


def print_table(results: List[Dict[str, Any]], include_summary: bool = True) -> None:
    """
    Print evaluation results as a pretty table to stdout.
    
    Args:
        results: List of evaluation results
        include_summary: Whether to include summary row
    """
    table = format_table(results, include_summary=include_summary)
    print(table)

