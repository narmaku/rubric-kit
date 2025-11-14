"""Output handlers for CSV and table display."""

import csv
from typing import List, Dict, Any
from tabulate import tabulate


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
    if not results:
        # Write empty CSV with headers
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["criterion_name", "category", "dimension", "result", "score", "max_score"])
        return
    
    # Get all unique keys from all results (not just first one)
    fieldnames_set = set()
    for result in results:
        fieldnames_set.update(result.keys())
    fieldnames = sorted(list(fieldnames_set))
    
    # Ensure common fields come first
    priority_fields = ["criterion_name", "category", "dimension", "criterion_text", "result", "score", "max_score"]
    ordered_fieldnames = []
    for field in priority_fields:
        if field in fieldnames:
            ordered_fieldnames.append(field)
            fieldnames.remove(field)
    ordered_fieldnames.extend(fieldnames)
    fieldnames = ordered_fieldnames
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write result rows
        for result in results:
            writer.writerow(result)
        
        # Add summary row if requested
        if include_summary:
            total_score = sum(r["score"] for r in results)
            max_score = sum(r["max_score"] for r in results)
            percentage = (total_score / max_score * 100) if max_score > 0 else 0
            
            summary_row = {key: "" for key in fieldnames}
            summary_row["criterion_name"] = "TOTAL"
            summary_row["score"] = total_score
            summary_row["max_score"] = max_score
            summary_row["result"] = f"{percentage:.1f}%"
            
            writer.writerow(summary_row)


def format_table(results: List[Dict[str, Any]], include_summary: bool = True) -> str:
    """
    Format evaluation results as a pretty table.
    
    Args:
        results: List of evaluation results
        include_summary: Whether to include summary row
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."
    
    # Prepare table data
    headers = ["Criterion", "Category", "Dimension", "Result", "Score", "Max Score"]
    rows = []
    
    for result in results:
        result_str = str(result.get("result", ""))
        
        # Add score description if available
        if "score_description" in result and result["score_description"]:
            result_str = f"{result_str} - {result['score_description']}"
        
        row = [
            result.get("criterion_name", ""),
            result.get("category", ""),
            result.get("dimension", ""),
            result_str,
            result.get("score", 0),
            result.get("max_score", 0)
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
            "─" * 15,
            "─" * 10,
            "─" * 5,
            "─" * 9
        ])
        
        rows.append([
            "TOTAL",
            "",
            "",
            f"{percentage:.1f}%",
            total_score,
            max_score
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

