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
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write result rows
        for result in expanded_results:
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

