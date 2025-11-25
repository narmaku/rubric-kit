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
    """Create LLM Judges Panel Summary section."""
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
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceAfter=8
    )
    
    story.append(Paragraph("LLM Judges Panel Summary", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    judges = judge_panel.get("judges", [])
    execution = judge_panel.get("execution", {})
    consensus = judge_panel.get("consensus", {})
    
    execution_mode = execution.get("mode", "sequential")
    consensus_mode = consensus.get("mode", "unanimous")
    threshold = consensus.get("threshold")
    
    # Summary paragraph
    summary_text = f"This evaluation was performed by a panel of <b>{len(judges)}</b> LLM judge(s) "
    summary_text += f"using <b>{execution_mode}</b> execution mode with <b>{consensus_mode}</b> consensus"
    if threshold:
        summary_text += f" (threshold: {threshold})"
    summary_text += "."
    story.append(Paragraph(summary_text, body_style))
    
    # Judges list
    if judges:
        judges_text = "<b>Judges:</b> "
        judge_details = [f"{j.get('name', 'unknown')} ({j.get('model', 'unknown')})" for j in judges]
        judges_text += ", ".join(judge_details)
        story.append(Paragraph(judges_text, body_style))
    
    # Consensus summary from results
    consensus_count = sum(1 for r in results if r.get("consensus_reached", True))
    total_criteria = len(results)
    if total_criteria > 0:
        consensus_pct = (consensus_count / total_criteria) * 100
        consensus_text = f"<b>Consensus reached:</b> {consensus_count}/{total_criteria} criteria ({consensus_pct:.0f}%)"
        story.append(Paragraph(consensus_text, body_style))
    
    story.append(Spacer(1, 0.3*inch))


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
    
    # Data rows with Paragraph objects for wrapping
    for r in results:
        criterion_name = r.get("criterion_name", "")
        dimension = r.get("dimension", "")
        result = str(r.get("result", ""))
        score = f"{r.get('score', 0)}/{r.get('max_score', 0)}"
        reason = r.get("reason", "") or ""
        
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
    story.append(Spacer(1, 0.3*inch))


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
        
        for crit in criteria:
            name = crit.get("name", "Unknown")
            category = crit.get("category", "")
            dimension = crit.get("dimension", "")
            criterion_text = crit.get("criterion", "")
            weight = crit.get("weight", "")
            
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
        except Exception as e:
            # If chart generation fails, continue without charts
            pass
    
    # Results table
    _create_results_table(results, story)
    
    # Rubric Appendix (at the end)
    _create_rubric_appendix(rubric_data, story)
    
    # Build PDF
    doc.build(story)

