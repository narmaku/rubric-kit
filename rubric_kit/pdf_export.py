"""PDF export functionality for evaluation results and rubrics."""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO


# =============================================================================
# COMMON COLORS
# =============================================================================

COLORS = {
    "primary": colors.HexColor('#2c3e50'),
    "secondary": colors.HexColor('#666666'),
    "title": colors.HexColor('#1a1a1a'),
    "success": colors.HexColor('#27ae60'),
    "danger": colors.HexColor('#c0392b'),
    "warning": colors.HexColor('#f39c12'),
    "info": colors.HexColor('#3498db'),
    "header_bg": colors.HexColor('#34495e'),
    "row_alt": colors.HexColor('#f5f5f5'),
}


def _load_evaluation_data(input_file: str) -> Dict[str, Any]:
    """
    Load evaluation data from YAML or JSON file.
    
    Args:
        input_file: Path to input file (YAML or JSON)
        
    Returns:
        Dictionary with results and optional metadata
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    ext = os.path.splitext(input_file)[1].lower()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        if ext in ('.yaml', '.yml'):
            data = yaml.safe_load(f)
        elif ext == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .yaml, .yml, or .json")
    
    return data


def _calculate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from results."""
    if not results:
        return {
            "total_score": 0,
            "max_score": 0,
            "percentage": 0.0,
            "passed": 0,
            "failed": 0,
            "total_criteria": 0
        }
    
    total_score = sum(r.get("score", 0) for r in results)
    max_score = sum(r.get("max_score", 0) for r in results)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0.0
    
    passed = sum(1 for r in results if r.get("result") == "pass" or (isinstance(r.get("result"), int) and r.get("result", 0) > 0))
    failed = len(results) - passed
    
    return {
        "total_score": total_score,
        "max_score": max_score,
        "percentage": percentage,
        "passed": passed,
        "failed": failed,
        "total_criteria": len(results)
    }


def _create_score_distribution_chart(results: List[Dict[str, Any]]) -> bytes:
    """Create a score distribution chart and return as PNG bytes."""
    scores = [r.get("score", 0) for r in results]
    max_scores = [r.get("max_score", 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Create histogram of scores
    ax.hist(scores, bins=range(0, max(max_scores) + 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of Criteria')
    ax.set_title('Score Distribution')
    ax.grid(True, alpha=0.3)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def _create_dimension_breakdown_chart(results: List[Dict[str, Any]]) -> bytes:
    """Create a dimension breakdown chart and return as PNG bytes."""
    from collections import defaultdict
    
    dimension_scores = defaultdict(lambda: {"total": 0, "max": 0})
    
    for r in results:
        dim = r.get("dimension", "Unknown")
        dimension_scores[dim]["total"] += r.get("score", 0)
        dimension_scores[dim]["max"] += r.get("max_score", 0)
    
    dimensions = list(dimension_scores.keys())
    percentages = [
        (dimension_scores[d]["total"] / dimension_scores[d]["max"] * 100) 
        if dimension_scores[d]["max"] > 0 else 0
        for d in dimensions
    ]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(dimensions, percentages, color='steelblue', alpha=0.7)
    ax.set_xlabel('Score Percentage (%)')
    ax.set_title('Score by Dimension')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=9)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def _create_title_page(metadata: Optional[Dict[str, Any]], story: List) -> None:
    """Create title page with metadata."""
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Spacer(1, 2*inch))
    
    # Use custom title from metadata if provided
    report_title = "Evaluation Report"
    if metadata and metadata.get("report_title"):
        report_title = metadata["report_title"]
    
    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.5*inch))
    
    # Metadata
    if metadata:
        meta_style = ParagraphStyle(
            'MetaStyle',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#666666'),
            alignment=TA_LEFT,
            leftIndent=1*inch,
            rightIndent=1*inch
        )
        
        if metadata.get("rubric_file"):
            story.append(Paragraph(f"<b>Rubric:</b> {metadata['rubric_file']}", meta_style))
        if metadata.get("input_file"):
            story.append(Paragraph(f"<b>Input:</b> {metadata['input_file']}", meta_style))
        if metadata.get("timestamp"):
            try:
                dt = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                story.append(Paragraph(f"<b>Date:</b> {dt.strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
            except:
                story.append(Paragraph(f"<b>Date:</b> {metadata['timestamp']}", meta_style))
        
        if metadata.get("judge_panel"):
            panel = metadata["judge_panel"]
            story.append(Paragraph(f"<b>Judges:</b> {panel.get('num_judges', 0)}", meta_style))
            if panel.get("judges"):
                judge_names = [j.get("name", "unknown") for j in panel["judges"]]
                story.append(Paragraph(f"<b>Judge Names:</b> {', '.join(judge_names)}", meta_style))
    
    story.append(PageBreak())


def _create_summary_section(stats: Dict[str, Any], story: List) -> None:
    """Create executive summary section."""
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    header_style = ParagraphStyle(
        'SummaryHeader',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    cell_style = ParagraphStyle(
        'SummaryCell',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT
    )
    
    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary table with Paragraph objects
    summary_data = [
        [Paragraph("Metric", header_style), Paragraph("Value", header_style)],
        [Paragraph("Total Score", cell_style), Paragraph(f"{stats['total_score']}/{stats['max_score']}", cell_style)],
        [Paragraph("Percentage", cell_style), Paragraph(f"{stats['percentage']:.1f}%", cell_style)],
        [Paragraph("Criteria Passed", cell_style), Paragraph(str(stats['passed']), cell_style)],
        [Paragraph("Criteria Failed", cell_style), Paragraph(str(stats['failed']), cell_style)],
        [Paragraph("Total Criteria", cell_style), Paragraph(str(stats['total_criteria']), cell_style)]
    ]
    
    page_width = letter[0]
    margin = 0.75 * inch
    usable_width = page_width - (2 * margin)
    
    summary_table = Table(summary_data, colWidths=[3*inch, usable_width - 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))


def _create_judges_panel_summary(judge_panel: Optional[Dict[str, Any]], results: List[Dict[str, Any]], story: List) -> None:
    """Create LLM Judges Panel Summary section as a table."""
    if not judge_panel:
        return
    
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    header_style = ParagraphStyle(
        'JudgeHeader',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    cell_style = ParagraphStyle(
        'JudgeCell',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT
    )
    
    story.append(Paragraph("LLM Judges Panel Summary", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Extract data
    judges = judge_panel.get("judges", [])
    execution_mode = judge_panel.get("execution", {}).get("mode", "sequential")
    consensus_cfg = judge_panel.get("consensus", {})
    consensus_mode = consensus_cfg.get("mode", "unanimous")
    threshold = consensus_cfg.get("threshold")
    
    # Calculate consensus stats
    consensus_count = sum(1 for r in results if r.get("consensus_reached", True))
    total_criteria = len(results)
    consensus_pct = (consensus_count / total_criteria * 100) if total_criteria > 0 else 0
    
    # Format judges list
    judge_names = ", ".join(
        f"{j.get('name', 'unknown')} ({j.get('model', 'unknown')})" for j in judges
    ) if judges else "N/A"
    
    # Format consensus mode with optional threshold
    consensus_display = consensus_mode
    if threshold:
        consensus_display += f" (threshold: {threshold})"
    
    # Build table data
    table_data = [
        [Paragraph("Setting", header_style), Paragraph("Value", header_style)],
        [Paragraph("Number of Judges", cell_style), Paragraph(str(len(judges)), cell_style)],
        [Paragraph("Judges", cell_style), Paragraph(judge_names, cell_style)],
        [Paragraph("Execution Mode", cell_style), Paragraph(execution_mode, cell_style)],
        [Paragraph("Consensus Mode", cell_style), Paragraph(consensus_display, cell_style)],
        [Paragraph("Consensus Reached", cell_style), Paragraph(f"{consensus_count}/{total_criteria} ({consensus_pct:.0f}%)", cell_style)],
    ]
    
    page_width = letter[0]
    margin = 0.75 * inch
    usable_width = page_width - (2 * margin)
    
    table = Table(table_data, colWidths=[2 * inch, usable_width - 2 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))


def _create_tool_breakdown_section(breakdown: Dict[str, Any], story: List) -> None:
    """Create a tool calls breakdown section for a criterion."""
    styles = getSampleStyleSheet()
    
    subheading_style = ParagraphStyle(
        'BreakdownHeading',
        parent=styles['Heading4'],
        fontSize=11,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=6,
        spaceBefore=8
    )
    
    cell_style = ParagraphStyle(
        'BreakdownCell',
        parent=styles['Normal'],
        fontSize=7,
        leading=9,
        alignment=TA_LEFT
    )
    
    header_style = ParagraphStyle(
        'BreakdownHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold'
    )
    
    issue_style = ParagraphStyle(
        'IssueStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#c0392b'),
        leftIndent=10
    )
    
    # Header with overall stats
    order_status = "✓" if breakdown.get("order_ok") else ("✗" if breakdown.get("order_ok") is False else "N/A")
    header_text = f"<b>Tool Calls Breakdown</b> | Score: {breakdown.get('overall_score', 0):.1f}/3 | Order: {order_status}"
    story.append(Paragraph(header_text, subheading_style))
    
    # Build breakdown table
    tool_results = breakdown.get("tool_results", [])
    if tool_results:
        table_data = [[
            Paragraph("Tool", header_style),
            Paragraph("Type", header_style),
            Paragraph("Called", header_style),
            Paragraph("Count", header_style),
            Paragraph("Params", header_style),
            Paragraph("Score", header_style)
        ]]
        
        for tr in tool_results:
            called_icon = "✓" if tr.get("called") else "✗"
            count_icon = "✓" if tr.get("count_ok") else "✗"
            params_ok = tr.get("params_ok")
            params_icon = "✓" if params_ok else ("✗" if params_ok is False else "N/A")
            
            table_data.append([
                Paragraph(tr.get("name", ""), cell_style),  # Full tool name, no truncation
                Paragraph(tr.get("type", ""), cell_style),
                Paragraph(called_icon, cell_style),
                Paragraph(f"{tr.get('count', 0)} {count_icon}", cell_style),
                Paragraph(params_icon, cell_style),
                Paragraph(f"{tr.get('score', 0):.1f}/{tr.get('max_score', 0):.1f}", cell_style)
            ])
        
        # Wider tool name column to fit full names
        col_widths = [3.2*inch, 0.7*inch, 0.5*inch, 0.6*inch, 0.5*inch, 0.7*inch]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5d6d7e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
        ]))
        story.append(table)
    
    # Issues section
    issues = breakdown.get("issues", [])
    if issues:
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("<b>Issues:</b>", cell_style))
        for issue in issues[:5]:  # Limit to 5 issues
            escaped_issue = issue.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(f"• {escaped_issue}", issue_style))
    
    story.append(Spacer(1, 0.1*inch))


def _create_results_table(results: List[Dict[str, Any]], story: List) -> None:
    """Create detailed results table with proper text wrapping."""
    styles = getSampleStyleSheet()
    
    # Create styles for table cells
    cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        wordWrap='CJK'
    )
    
    header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    story.append(Paragraph("Detailed Results", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Prepare table data with Paragraph objects for text wrapping
    # Header row
    table_data = [[
        Paragraph("Criterion", header_style),
        Paragraph("Dimension", header_style),
        Paragraph("Result", header_style),
        Paragraph("Score", header_style),
        Paragraph("Reason", header_style)
    ]]
    
    # Track criteria with tool breakdowns to render after table
    criteria_with_breakdowns = []
    
    # Data rows with Paragraph objects for wrapping
    for r in results:
        criterion_name = r.get("criterion_name", "")
        dimension = r.get("dimension", "")
        result = str(r.get("result", ""))
        score = f"{r.get('score', 0)}/{r.get('max_score', 0)}"
        reason = r.get("reason", "") or ""
        
        # Check for tool breakdown
        if r.get("tool_breakdown"):
            criteria_with_breakdowns.append((criterion_name, r["tool_breakdown"]))
        
        # Use Paragraph for all cells to enable wrapping
        table_data.append([
            Paragraph(criterion_name.replace('&', '&amp;'), cell_style),
            Paragraph(dimension.replace('&', '&amp;'), cell_style),
            Paragraph(result.replace('&', '&amp;'), cell_style),
            Paragraph(score, cell_style),
            Paragraph(reason.replace('&', '&amp;'), cell_style)
        ])
    
    # Adjust column widths to fit page (letter size is 8.5 inches, minus margins ~1 inch each side = 6.5 inches usable)
    # Use better proportions: Criterion (1.8"), Dimension (1.5"), Result (0.7"), Score (0.7"), Reason (2.0")
    page_width = letter[0]
    margin = 0.75 * inch
    usable_width = page_width - (2 * margin)
    
    col_widths = [
        1.8 * inch,  # Criterion
        1.5 * inch,  # Dimension
        0.7 * inch,  # Result
        0.7 * inch,  # Score
        usable_width - (1.8 + 1.5 + 0.7 + 0.7) * inch  # Reason (remaining space)
    ]
    
    results_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Render tool breakdowns for criteria that have them
    if criteria_with_breakdowns:
        breakdown_heading = ParagraphStyle(
            'BreakdownSectionHeading',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=8,
            spaceBefore=12
        )
        story.append(Paragraph("Tool Calls Breakdowns", breakdown_heading))
        
        for criterion_name, breakdown in criteria_with_breakdowns:
            criterion_label = ParagraphStyle(
                'CriterionLabel',
                parent=styles['Normal'],
                fontSize=10,
                fontName='Helvetica-Bold',
                textColor=colors.HexColor('#34495e'),
                spaceAfter=4
            )
            story.append(Paragraph(f"Criterion: {criterion_name}", criterion_label))
            _create_tool_breakdown_section(breakdown, story)
    
    story.append(Spacer(1, 0.3*inch))


def _render_tool_calls(tool_calls: Dict[str, Any], story: List, label_style, detail_style) -> None:
    """Render tool_calls specification for a criterion."""
    from reportlab.platypus import Spacer
    from reportlab.lib.units import inch
    
    # Display settings
    settings_parts = []
    if "respect_order" in tool_calls:
        settings_parts.append(f"respect_order: {tool_calls['respect_order']}")
    if "params_strict_mode" in tool_calls:
        settings_parts.append(f"params_strict_mode: {tool_calls['params_strict_mode']}")
    
    if settings_parts:
        story.append(Paragraph(f"<i>{', '.join(settings_parts)}</i>", label_style))
    
    # Render required, optional, and prohibited tools
    tool_sections = [
        ("required", tool_calls.get("required", [])),
        ("optional", tool_calls.get("optional", [])),
        ("prohibited", tool_calls.get("prohibited", []))
    ]
    
    for section_name, tools in tool_sections:
        if not tools:
            continue
        
        story.append(Paragraph(f"<b>{section_name}:</b>", label_style))
        
        for tool_entry in tools:
            # Handle both dict-style {"tool_name": {...}} and ToolSpec-style {"name": "...", ...}
            if isinstance(tool_entry, dict):
                # Check if it's dict-style (tool_name as key)
                tool_name = tool_entry.get("name")
                tool_spec = tool_entry
                
                if tool_name is None:
                    # It's dict-style {"tool_name": {...}}
                    for key, value in tool_entry.items():
                        tool_name = key
                        tool_spec = value if isinstance(value, dict) else {}
                        break
                
                if tool_name:
                    # Build tool description
                    tool_desc_parts = [f"<b>{_escape_xml(tool_name)}</b>"]
                    
                    min_calls = tool_spec.get("min_calls")
                    max_calls = tool_spec.get("max_calls")
                    if min_calls is not None or max_calls is not None:
                        calls_str = ""
                        if min_calls is not None and max_calls is not None:
                            if min_calls == max_calls:
                                calls_str = f"calls: {min_calls}"
                            else:
                                calls_str = f"calls: {min_calls}-{max_calls}"
                        elif min_calls is not None:
                            calls_str = f"min_calls: {min_calls}"
                        elif max_calls is not None:
                            calls_str = f"max_calls: {max_calls}"
                        tool_desc_parts.append(calls_str)
                    
                    story.append(Paragraph(" | ".join(tool_desc_parts), detail_style))
                    
                    # Show params if any
                    params = tool_spec.get("params")
                    if params:
                        params_str = ", ".join([f"{k}={_escape_xml(str(v))}" for k, v in params.items()])
                        story.append(Paragraph(f"params: {params_str}", detail_style))
    
    story.append(Spacer(1, 0.05*inch))


def _escape_xml(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _create_input_section(input_data: Optional[Dict[str, Any]], story: List) -> None:
    """Create Input Content section displaying Q&A or chat session content."""
    if not input_data:
        return
    
    input_type = input_data.get("type", "unknown")
    
    # Check if we have content in the new structured format
    has_qna_data = "question" in input_data or "answer" in input_data
    has_chat_data = "chat_session" in input_data
    has_legacy_content = "content" in input_data
    
    if not has_qna_data and not has_chat_data and not has_legacy_content:
        # Try to read from source file
        source_file = input_data.get("source_file")
        if source_file and os.path.exists(source_file):
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    if input_type == "qna":
                        # Parse and add to input_data
                        qa_content = yaml.safe_load(f)
                        if isinstance(qa_content, dict):
                            input_data.update(qa_content)
                            has_qna_data = True
                    else:
                        input_data["chat_session"] = f.read()
                        has_chat_data = True
            except Exception:
                return
        else:
            return
    
    if not has_qna_data and not has_chat_data and not has_legacy_content:
        return
    
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'InputHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=COLORS["primary"],
        spaceAfter=12,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'InputSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=COLORS["secondary"],
        spaceAfter=8,
        spaceBefore=12
    )
    
    content_style = ParagraphStyle(
        'InputContent',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10,
        backColor=COLORS["row_alt"],
        borderPadding=8
    )
    
    qa_label_style = ParagraphStyle(
        'QALabel',
        parent=styles['Normal'],
        fontSize=10,
        fontName='Helvetica-Bold',
        textColor=COLORS["header_bg"],
        spaceAfter=4,
        spaceBefore=8
    )
    
    qa_content_style = ParagraphStyle(
        'QAContent',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=12,
        leftIndent=15,
        rightIndent=10,
        backColor=COLORS["row_alt"],
        borderPadding=6
    )
    
    story.append(PageBreak())
    story.append(Paragraph("Input Content", heading_style))
    
    source_file = input_data.get("source_file", "")
    
    # Display source info
    if source_file:
        source_info = f"<i>Source: {_escape_xml(str(source_file))} ({input_type})</i>"
        story.append(Paragraph(source_info, content_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    # Handle Q&A format
    if input_type == "qna" and has_qna_data:
        _render_single_qa(input_data, story, qa_label_style, qa_content_style)
    elif has_chat_data:
        # New format: chat_session key
        _render_chat_content(input_data["chat_session"], story, subheading_style, content_style)
    elif has_legacy_content:
        # Legacy format: content key
        content = input_data["content"]
        if input_type == "qna":
            _render_qna_content(content, story, subheading_style, qa_label_style, qa_content_style)
        else:
            _render_chat_content(content, story, subheading_style, content_style)


def _render_qna_content(content: str, story: List, subheading_style, label_style, content_style) -> None:
    """Render Q&A YAML content in a structured format (legacy format with 'content' string)."""
    try:
        qa_data = yaml.safe_load(content)
        
        if isinstance(qa_data, dict):
            # Single Q&A pair
            _render_single_qa(qa_data, story, label_style, content_style)
        elif isinstance(qa_data, list):
            # Multiple Q&A pairs
            for i, qa in enumerate(qa_data, 1):
                if isinstance(qa, dict):
                    story.append(Paragraph(f"<b>Q&A Pair {i}</b>", label_style))
                    _render_single_qa(qa, story, label_style, content_style)
                    story.append(Spacer(1, 0.1*inch))
    except Exception:
        # Fall back to raw content display
        escaped = _escape_xml(content)
        story.append(Paragraph(escaped, content_style))


def _format_content_for_pdf(text: str) -> str:
    """Format text content for PDF display, preserving newlines and indentation."""
    escaped = _escape_xml(str(text))
    
    # Process line by line to preserve leading indentation
    lines = escaped.split('\n')
    formatted_lines = []
    for line in lines:
        # Count leading spaces and convert to non-breaking spaces
        stripped = line.lstrip(' ')
        leading_spaces = len(line) - len(stripped)
        if leading_spaces > 0:
            # Use &nbsp; for leading spaces to preserve indentation
            line = '&nbsp;' * leading_spaces + stripped
        formatted_lines.append(line)
    
    # Join with <br/> for proper line breaks in PDF
    return '<br/>'.join(formatted_lines)


def _render_single_qa(qa_data: Dict[str, Any], story: List, label_style, content_style) -> None:
    """Render a single Q&A pair with proper multiline and code block handling."""
    styles = getSampleStyleSheet()
    
    # Code style for content that looks like code (has code fences or indentation)
    code_style = ParagraphStyle(
        'QACode',
        parent=content_style,
        fontName='Courier',
        fontSize=8,
        leading=10,
        backColor=colors.HexColor('#f5f5f5'),
        borderColor=colors.HexColor('#dddddd'),
        borderWidth=0.5,
        borderPadding=8
    )
    
    # Question
    question = qa_data.get("question", "")
    if question:
        story.append(Paragraph("Question:", label_style))
        story.append(Spacer(1, 0.05*inch))
        formatted_q = _format_content_for_pdf(question)
        story.append(Paragraph(formatted_q, content_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Context (if present)
    context = qa_data.get("context", "")
    if context:
        story.append(Paragraph("Context:", label_style))
        story.append(Spacer(1, 0.05*inch))
        formatted_ctx = _format_content_for_pdf(context)
        # Truncate very long context
        if len(formatted_ctx) > 2000:
            formatted_ctx = formatted_ctx[:2000] + "... [truncated]"
        story.append(Paragraph(formatted_ctx, content_style))
        story.append(Spacer(1, 0.15*inch))
    
    # Answer
    answer = qa_data.get("answer", "")
    if answer:
        story.append(Paragraph("Answer:", label_style))
        story.append(Spacer(1, 0.1*inch))  # Add spacing between label and content
        formatted_a = _format_content_for_pdf(answer)
        # Use code style if answer contains code fences or looks like code
        if '```' in answer or answer.strip().startswith(('def ', 'class ', '#!/', 'import ', 'from ')):
            story.append(Paragraph(formatted_a, code_style))
        else:
            story.append(Paragraph(formatted_a, content_style))


def _render_chat_content(content: str, story: List, subheading_style, content_style) -> None:
    """Render chat session content."""
    story.append(Paragraph("Chat Session", subheading_style))
    
    # Split content into manageable chunks for better rendering
    lines = content.split('\n')
    
    # Process content in chunks to avoid memory issues with very large sessions
    chunk_size = 100
    for i in range(0, len(lines), chunk_size):
        chunk = '\n'.join(lines[i:i+chunk_size])
        escaped = _escape_xml(chunk)
        # Replace newlines with <br/> for PDF rendering
        escaped = escaped.replace('\n', '<br/>')
        story.append(Paragraph(escaped, content_style))
        
        if i + chunk_size < len(lines):
            story.append(Spacer(1, 0.05*inch))


def _create_rubric_appendix(rubric_data: Optional[Dict[str, Any]], story: List) -> None:
    """Create Rubric Appendix section with Dimensions and Criteria."""
    if not rubric_data:
        return
    dimensions = rubric_data.get("dimensions", [])
    criteria = rubric_data.get("criteria", [])
    
    if not dimensions and not criteria:
        return
    
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'AppendixHeading',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        spaceAfter=6
    )
    
    item_style = ParagraphStyle(
        'ItemStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        spaceAfter=4,
        leftIndent=15
    )
    
    # Start new page for appendix
    story.append(PageBreak())
    story.append(Paragraph("Rubric", heading_style))
    
    # Dimensions section
    if dimensions:
        story.append(Paragraph("Dimensions", subheading_style))
        
        for dim in dimensions:
            name = dim.get("name", "Unknown")
            description = dim.get("description", "")
            grading_type = dim.get("grading_type", "binary")
            scores = dim.get("scores")
            
            dim_text = f"<b>{name}</b> ({grading_type})"
            story.append(Paragraph(dim_text, body_style))
            
            if description:
                desc_escaped = description.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(desc_escaped, item_style))
            
            if scores:
                scores_text = "Scores: " + ", ".join([f"{k}: {v}" for k, v in scores.items()])
                story.append(Paragraph(scores_text, item_style))
        
        story.append(Spacer(1, 0.2*inch))
    
    # Criteria section
    if criteria:
        story.append(Paragraph("Criteria", subheading_style))
        
        tool_label_style = ParagraphStyle(
            'ToolLabelStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            spaceAfter=2,
            leftIndent=20,
            textColor=colors.HexColor('#555555')
        )
        
        tool_detail_style = ParagraphStyle(
            'ToolDetailStyle',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            spaceAfter=2,
            leftIndent=30,
            fontName='Courier'
        )
        
        for crit in criteria:
            name = crit.get("name", "Unknown")
            category = crit.get("category", "")
            dimension = crit.get("dimension", "")
            criterion_text = crit.get("criterion", "")
            weight = crit.get("weight", "")
            tool_calls = crit.get("tool_calls")
            
            crit_header = f"<b>{name}</b>"
            if category:
                crit_header += f" [{category}]"
            if dimension:
                crit_header += f" → {dimension}"
            if weight:
                crit_header += f" (weight: {weight})"
            
            story.append(Paragraph(crit_header, body_style))
            
            if criterion_text and criterion_text != "from_scores":
                text_escaped = criterion_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                story.append(Paragraph(text_escaped, item_style))
            
            # Render tool_calls if present
            if tool_calls:
                _render_tool_calls(tool_calls, story, tool_label_style, tool_detail_style)


def export_evaluation_pdf(input_file: str, output_file: str) -> None:
    """
    Export evaluation results to PDF format.
    
    Args:
        input_file: Path to input YAML or JSON file with evaluation results
        output_file: Path to output PDF file
    """
    # Load data
    data = _load_evaluation_data(input_file)
    results = data.get("results", [])
    metadata = data.get("metadata", {})
    rubric_data = data.get("rubric")
    judge_panel = data.get("judge_panel")
    input_data = data.get("input")
    
    if not results:
        raise ValueError("No results found in input file")
    
    # Calculate statistics
    stats = _calculate_summary_stats(results)
    
    # Create PDF document with margins
    doc = SimpleDocTemplate(
        output_file, 
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    story = []
    
    # Title page
    _create_title_page(metadata, story)
    
    # Summary section
    _create_summary_section(stats, story)
    
    # LLM Judges Panel Summary
    _create_judges_panel_summary(judge_panel, results, story)
    
    # Input Content section (after judges summary, before charts)
    _create_input_section(input_data, story)
    
    # Charts
    if len(results) > 0:
        try:
            heading_style = ParagraphStyle(
                'SectionHeading',
                parent=getSampleStyleSheet()['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12
            )
            
            story.append(Paragraph("Charts", heading_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Score distribution chart
            chart_data = _create_score_distribution_chart(results)
            chart_img = Image(BytesIO(chart_data), width=4*inch, height=2.7*inch)
            story.append(chart_img)
            story.append(Spacer(1, 0.3*inch))
            
            # Dimension breakdown chart
            chart_data2 = _create_dimension_breakdown_chart(results)
            chart_img2 = Image(BytesIO(chart_data2), width=5*inch, height=3*inch)
            story.append(chart_img2)
            story.append(PageBreak())
        except Exception:
            # If chart generation fails, continue without charts
            pass
    
    # Results table
    _create_results_table(results, story)
    
    # Rubric Appendix (at the end)
    _create_rubric_appendix(rubric_data, story)
    
    # Build PDF
    doc.build(story)


# ============================================================================
# Arena PDF Export Functions
# ============================================================================

def _create_arena_title_page(metadata: Optional[Dict[str, Any]], arena_name: str, 
                             arena_description: Optional[str], story: List) -> None:
    """Create title page for Arena comparative report with description and metadata table."""
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'ArenaTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=10,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'ArenaSubtitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#666666'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    description_style = ParagraphStyle(
        'ArenaDescription',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#444444'),
        alignment=TA_CENTER,
        spaceAfter=20,
        leftIndent=0.5*inch,
        rightIndent=0.5*inch
    )
    
    story.append(Spacer(1, 1.5*inch))
    
    # Report type identifier
    story.append(Paragraph("⚔️ Arena Comparative Evaluation Report", subtitle_style))
    
    # Use custom title from metadata if provided, else use arena name
    report_title = metadata.get("report_title", arena_name) if metadata else arena_name
    story.append(Paragraph(report_title, title_style))
    
    # Display arena description if provided
    if arena_description:
        story.append(Spacer(1, 0.2*inch))
        escaped_desc = arena_description.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        story.append(Paragraph(f"<i>{escaped_desc}</i>", description_style))
    
    story.append(Spacer(1, 0.4*inch))
    
    # Metadata table
    if metadata:
        header_style = ParagraphStyle(
            'MetaTableHeader',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.whitesmoke,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT
        )
        
        cell_style = ParagraphStyle(
            'MetaTableCell',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT
        )
        
        # Build metadata table data - collect all available metadata
        table_data = [[
            Paragraph("Property", header_style),
            Paragraph("Value", header_style)
        ]]
        
        # Arena spec file
        if metadata.get("arena_spec_file"):
            table_data.append([
                Paragraph("Arena Spec", cell_style),
                Paragraph(str(metadata['arena_spec_file']), cell_style)
            ])
        
        # Rubric source file
        if metadata.get("rubric_source_file"):
            table_data.append([
                Paragraph("Rubric File", cell_style),
                Paragraph(str(metadata['rubric_source_file']), cell_style)
            ])
        
        # Judge panel source
        if metadata.get("judge_panel_source_file"):
            table_data.append([
                Paragraph("Judge Panel", cell_style),
                Paragraph(str(metadata['judge_panel_source_file']), cell_style)
            ])
        
        # Timestamp
        if metadata.get("timestamp"):
            try:
                dt = datetime.fromisoformat(metadata['timestamp'].replace('Z', '+00:00'))
                timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                timestamp_str = str(metadata['timestamp'])
            table_data.append([
                Paragraph("Generated", cell_style),
                Paragraph(timestamp_str, cell_style)
            ])
        
        # Source files (if combined from outputs)
        if metadata.get("source_files"):
            source_files = metadata['source_files']
            if isinstance(source_files, list):
                source_str = ", ".join(str(f) for f in source_files[:3])
                if len(source_files) > 3:
                    source_str += f" (+{len(source_files) - 3} more)"
            else:
                source_str = str(source_files)
            table_data.append([
                Paragraph("Source Files", cell_style),
                Paragraph(source_str, cell_style)
            ])
        
        # Combined from outputs flag
        if metadata.get("combined_from_outputs"):
            table_data.append([
                Paragraph("Mode", cell_style),
                Paragraph("Combined from existing outputs", cell_style)
            ])
        
        # Add any other metadata fields not explicitly handled
        known_keys = {'arena_spec_file', 'rubric_source_file', 'judge_panel_source_file', 
                      'timestamp', 'source_files', 'combined_from_outputs', 'report_title'}
        for key, value in metadata.items():
            if key not in known_keys and value is not None:
                # Format the key nicely (replace underscores, title case)
                display_key = key.replace('_', ' ').title()
                display_value = str(value)
                if len(display_value) > 80:
                    display_value = display_value[:77] + "..."
                table_data.append([
                    Paragraph(display_key, cell_style),
                    Paragraph(display_value, cell_style)
                ])
        
        # Only create table if we have data beyond the header
        if len(table_data) > 1:
            page_width = letter[0]
            margin = 0.75 * inch
            usable_width = page_width - (2 * margin)
            
            table = Table(table_data, colWidths=[2*inch, usable_width - 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ]))
            
            story.append(table)
    
    story.append(PageBreak())


def _create_arena_rankings_section(rankings: List[Dict[str, Any]], story: List) -> None:
    """Create rankings table section."""
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    header_style = ParagraphStyle(
        'RankingsHeader',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.whitesmoke,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    )
    
    cell_style = ParagraphStyle(
        'RankingsCell',
        parent=styles['Normal'],
        fontSize=11,
        alignment=TA_LEFT
    )
    
    story.append(Paragraph("Rankings Summary", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Build table data
    table_data = [[
        Paragraph("Rank", header_style),
        Paragraph("Contestant", header_style),
        Paragraph("Score", header_style),
        Paragraph("Percentage", header_style)
    ]]
    
    for r in rankings:
        medal = "🥇" if r["rank"] == 1 else ("🥈" if r["rank"] == 2 else ("🥉" if r["rank"] == 3 else ""))
        rank_text = f"{medal} #{r['rank']}" if medal else f"#{r['rank']}"
        
        table_data.append([
            Paragraph(rank_text, cell_style),
            Paragraph(r["name"], cell_style),
            Paragraph(f"{r['total_score']}/{r['max_score']}", cell_style),
            Paragraph(f"{r['percentage']:.1f}%", cell_style)
        ])
    
    page_width = letter[0]
    margin = 0.75 * inch
    usable_width = page_width - (2 * margin)
    
    table = Table(table_data, colWidths=[1*inch, 3*inch, 1.5*inch, usable_width - 5.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))


def _create_comparative_bar_chart(contestants: Dict[str, Any], story: List) -> None:
    """Create comparative bar chart for all contestants."""
    from collections import defaultdict
    import numpy as np
    
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    story.append(Paragraph("Comparative Performance by Dimension", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Extract dimension scores per contestant
    dimension_scores: Dict[str, Dict[str, float]] = {}  # dimension -> {contestant_id: percentage}
    contestant_names = {}
    
    for contestant_id, cdata in contestants.items():
        contestant_names[contestant_id] = cdata["name"]
        dim_totals = defaultdict(lambda: {"total": 0, "max": 0})
        
        for r in cdata.get("results", []):
            dim = r.get("dimension", "Unknown")
            dim_totals[dim]["total"] += r.get("score", 0)
            dim_totals[dim]["max"] += r.get("max_score", 0)
        
        for dim, scores in dim_totals.items():
            if dim not in dimension_scores:
                dimension_scores[dim] = {}
            pct = (scores["total"] / scores["max"] * 100) if scores["max"] > 0 else 0
            dimension_scores[dim][contestant_id] = pct
    
    dimensions = list(dimension_scores.keys())
    contestant_ids = list(contestant_names.keys())
    
    if not dimensions or not contestant_ids:
        return
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(dimensions))
    width = 0.8 / len(contestant_ids)
    
    colors_list = plt.cm.Set2(np.linspace(0, 1, len(contestant_ids)))
    
    for i, cid in enumerate(contestant_ids):
        scores = [dimension_scores[dim].get(cid, 0) for dim in dimensions]
        offset = (i - len(contestant_ids)/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=contestant_names[cid], color=colors_list[i])
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison by Dimension')
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    chart_img = Image(buf, width=6.5*inch, height=4*inch)
    story.append(chart_img)
    story.append(Spacer(1, 0.3*inch))


def _create_radar_charts(contestants: Dict[str, Any], story: List) -> None:
    """Create individual radar/spider charts for each contestant's performance profile."""
    from collections import defaultdict
    import numpy as np
    
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    story.append(Paragraph("Performance Profiles (Radar Charts)", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Extract dimension scores per contestant
    all_dimensions = set()
    contestant_data = {}
    
    for contestant_id, cdata in contestants.items():
        dim_totals = defaultdict(lambda: {"total": 0, "max": 0})
        
        for r in cdata.get("results", []):
            dim = r.get("dimension", "Unknown")
            dim_totals[dim]["total"] += r.get("score", 0)
            dim_totals[dim]["max"] += r.get("max_score", 0)
            all_dimensions.add(dim)
        
        contestant_data[contestant_id] = {
            "name": cdata["name"],
            "scores": {
                dim: (scores["total"] / scores["max"] * 100) if scores["max"] > 0 else 0
                for dim, scores in dim_totals.items()
            },
            "percentage": cdata.get("summary", {}).get("percentage", 0)
        }
    
    dimensions = sorted(list(all_dimensions))
    
    if len(dimensions) < 3:
        return  # Need at least 3 dimensions for radar chart
    
    # Create angles for the radar chart
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    # Determine grid layout (2 columns for better use of space)
    num_contestants = len(contestant_data)
    ncols = 2
    nrows = (num_contestants + 1) // 2
    
    # Create a figure with subplots for each contestant
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), 
                              subplot_kw=dict(polar=True))
    
    # Flatten axes for easy iteration
    if num_contestants == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Color palette for contestants
    color_palette = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    for idx, (contestant_id, cdata) in enumerate(contestant_data.items()):
        ax = axes[idx]
        
        # Get scores for all dimensions (0 if not present)
        values = [cdata["scores"].get(dim, 0) for dim in dimensions]
        values += values[:1]  # Close the polygon
        
        color = color_palette[idx % len(color_palette)]
        
        # Plot the radar
        ax.plot(angles, values, 'o-', linewidth=2, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.25, color=color)
        
        # Configure the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=7, color='gray')
        ax.grid(True, alpha=0.3)
        
        # Title with contestant name and overall score
        title = f"{cdata['name']}\n({cdata['percentage']:.1f}%)"
        ax.set_title(title, size=11, fontweight='bold', pad=15)
    
    # Hide empty subplots if odd number of contestants
    for idx in range(num_contestants, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Scale height based on number of rows
    chart_height = min(3.5 * nrows, 9) * inch
    chart_img = Image(buf, width=6.5*inch, height=chart_height)
    story.append(chart_img)
    story.append(Spacer(1, 0.3*inch))


def _create_contestant_details_section(contestants: Dict[str, Any], story: List) -> None:
    """Create detailed results section for each contestant."""
    styles = getSampleStyleSheet()
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12
    )
    
    contestant_heading_style = ParagraphStyle(
        'ContestantHeading',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=12
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#666666'),
        leftIndent=15
    )
    
    story.append(PageBreak())
    story.append(Paragraph("Contestant Details", heading_style))
    
    for contestant_id, cdata in contestants.items():
        story.append(Paragraph(f"{cdata['name']} (id: {contestant_id})", contestant_heading_style))
        
        # Metadata
        if cdata.get("description"):
            story.append(Paragraph(f"<i>{cdata['description']}</i>", meta_style))
        
        if cdata.get("metadata"):
            meta_text = ", ".join([f"{k}: {v}" for k, v in cdata["metadata"].items()])
            story.append(Paragraph(f"<b>Metadata:</b> {meta_text}", meta_style))
        
        summary = cdata.get("summary", {})
        story.append(Paragraph(
            f"<b>Score:</b> {summary.get('total_score', 0)}/{summary.get('max_score', 0)} "
            f"({summary.get('percentage', 0):.1f}%)",
            meta_style
        ))
        
        story.append(Spacer(1, 0.1*inch))
        
        # Results table (compact)
        results = cdata.get("results", [])
        if results:
            _create_results_table(results, story)
        
        story.append(Spacer(1, 0.2*inch))


def export_arena_pdf(input_file: str, output_file: str) -> None:
    """
    Export arena comparative evaluation results to PDF format.
    
    Args:
        input_file: Path to input YAML file with arena results
        output_file: Path to output PDF file
    """
    # Load data
    data = _load_evaluation_data(input_file)
    
    if data.get("mode") != "arena":
        raise ValueError("Input file is not an arena evaluation result")
    
    arena_name = data.get("arena_name", "Arena Evaluation")
    arena_description = data.get("arena_description")
    contestants = data.get("contestants", {})
    rankings = data.get("rankings", [])
    metadata = data.get("metadata", {})
    rubric_data = data.get("rubric")
    judge_panel = data.get("judge_panel")
    
    if not contestants:
        raise ValueError("No contestants found in arena results")
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    story = []
    
    # Title page
    _create_arena_title_page(metadata, arena_name, arena_description, story)
    
    # Rankings summary
    if rankings:
        _create_arena_rankings_section(rankings, story)
    
    # LLM Judges Panel Summary (shared for all)
    first_contestant_results = list(contestants.values())[0].get("results", [])
    _create_judges_panel_summary(judge_panel, first_contestant_results, story)
    
    # Comparative charts
    try:
        story.append(PageBreak())
        _create_comparative_bar_chart(contestants, story)
        _create_radar_charts(contestants, story)
    except Exception:
        pass  # Continue without charts if they fail
    
    # Contestant details
    _create_contestant_details_section(contestants, story)
    
    # Rubric Appendix
    _create_rubric_appendix(rubric_data, story)
    
    # Build PDF
    doc.build(story)

