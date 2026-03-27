"""Evaluation PDF export functionality."""

from collections import defaultdict
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from .pdf_base import (
    _create_input_section,
    _create_judges_panel_summary,
    _create_results_table,
    _create_rubric_appendix,
    _load_evaluation_data,
)


def _calculate_summary_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate summary statistics from results."""
    if not results:
        return {
            "total_score": 0,
            "max_score": 0,
            "percentage": 0.0,
            "passed": 0,
            "failed": 0,
            "total_criteria": 0,
        }

    total_score = sum(r.get("score", 0) for r in results)
    max_score = sum(r.get("max_score", 0) for r in results)
    percentage = (total_score / max_score * 100) if max_score > 0 else 0.0

    passed = sum(
        1
        for r in results
        if r.get("result") == "pass"
        or (isinstance(r.get("result"), int) and r.get("result", 0) > 0)
    )
    failed = len(results) - passed

    return {
        "total_score": total_score,
        "max_score": max_score,
        "percentage": percentage,
        "passed": passed,
        "failed": failed,
        "total_criteria": len(results),
    }


def _create_score_distribution_chart(results: list[dict[str, Any]]) -> bytes:
    """Create a score distribution chart and return as PNG bytes."""
    scores = [r.get("score", 0) for r in results]
    max_scores = [r.get("max_score", 0) for r in results]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Create histogram of scores
    ax.hist(scores, bins=range(0, max(max_scores) + 2), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Criteria")
    ax.set_title("Score Distribution")
    ax.grid(True, alpha=0.3)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def _create_dimension_breakdown_chart(results: list[dict[str, Any]]) -> bytes:
    """Create a dimension breakdown chart and return as PNG bytes."""
    dimension_scores = defaultdict(lambda: {"total": 0, "max": 0})

    for r in results:
        dim = r.get("dimension", "Unknown")
        dimension_scores[dim]["total"] += r.get("score", 0)
        dimension_scores[dim]["max"] += r.get("max_score", 0)

    dimensions = list(dimension_scores.keys())
    percentages = [
        (dimension_scores[d]["total"] / dimension_scores[d]["max"] * 100)
        if dimension_scores[d]["max"] > 0
        else 0
        for d in dimensions
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(dimensions, percentages, color="steelblue", alpha=0.7)
    ax.set_xlabel("Score Percentage (%)")
    ax.set_title("Score by Dimension")
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis="x")

    # Add percentage labels on bars
    for _i, (bar, pct) in enumerate(zip(bars, percentages, strict=False)):
        ax.text(pct + 1, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def _create_title_page(metadata: dict[str, Any] | None, story: list) -> None:
    """Create title page with metadata."""
    from datetime import datetime

    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    story.append(Spacer(1, 2 * inch))

    # Use custom title from metadata if provided
    report_title = "Evaluation Report"
    if metadata and metadata.get("report_title"):
        report_title = metadata["report_title"]

    story.append(Paragraph(report_title, title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Metadata
    if metadata:
        meta_style = ParagraphStyle(
            "MetaStyle",
            parent=styles["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#666666"),
            alignment=TA_LEFT,
            leftIndent=1 * inch,
            rightIndent=1 * inch,
        )

        if metadata.get("rubric_file"):
            story.append(Paragraph(f"<b>Rubric:</b> {metadata['rubric_file']}", meta_style))
        if metadata.get("input_file"):
            story.append(Paragraph(f"<b>Input:</b> {metadata['input_file']}", meta_style))
        if metadata.get("timestamp"):
            try:
                dt = datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
                story.append(
                    Paragraph(f"<b>Date:</b> {dt.strftime('%Y-%m-%d %H:%M:%S')}", meta_style)
                )
            except (ValueError, TypeError):
                story.append(Paragraph(f"<b>Date:</b> {metadata['timestamp']}", meta_style))

        if metadata.get("judge_panel"):
            panel = metadata["judge_panel"]
            story.append(Paragraph(f"<b>Judges:</b> {panel.get('num_judges', 0)}", meta_style))
            if panel.get("judges"):
                judge_names = [j.get("name", "unknown") for j in panel["judges"]]
                story.append(Paragraph(f"<b>Judge Names:</b> {', '.join(judge_names)}", meta_style))

    story.append(PageBreak())


def _create_summary_section(stats: dict[str, Any], story: list) -> None:
    """Create executive summary section."""
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    header_style = ParagraphStyle(
        "SummaryHeader",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.whitesmoke,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
    )

    cell_style = ParagraphStyle(
        "SummaryCell", parent=styles["Normal"], fontSize=11, alignment=TA_LEFT
    )

    story.append(Paragraph("Executive Summary", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # Summary table with Paragraph objects
    summary_data = [
        [Paragraph("Metric", header_style), Paragraph("Value", header_style)],
        [
            Paragraph("Total Score", cell_style),
            Paragraph(f"{stats['total_score']}/{stats['max_score']}", cell_style),
        ],
        [Paragraph("Percentage", cell_style), Paragraph(f"{stats['percentage']:.1f}%", cell_style)],
        [Paragraph("Criteria Passed", cell_style), Paragraph(str(stats["passed"]), cell_style)],
        [Paragraph("Criteria Failed", cell_style), Paragraph(str(stats["failed"]), cell_style)],
        [
            Paragraph("Total Criteria", cell_style),
            Paragraph(str(stats["total_criteria"]), cell_style),
        ],
    ]

    page_width = letter[0]
    margin = 0.75 * inch
    usable_width = page_width - (2 * margin)

    from reportlab.platypus import Table, TableStyle

    summary_table = Table(summary_data, colWidths=[3 * inch, usable_width - 3 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 1), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 0.3 * inch))


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
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
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
                "SectionHeading",
                parent=getSampleStyleSheet()["Heading2"],
                fontSize=16,
                textColor=colors.HexColor("#2c3e50"),
                spaceAfter=12,
            )

            story.append(Paragraph("Charts", heading_style))
            story.append(Spacer(1, 0.2 * inch))

            # Score distribution chart
            chart_data = _create_score_distribution_chart(results)
            chart_img = Image(BytesIO(chart_data), width=4 * inch, height=2.7 * inch)
            story.append(chart_img)
            story.append(Spacer(1, 0.3 * inch))

            # Dimension breakdown chart
            chart_data2 = _create_dimension_breakdown_chart(results)
            chart_img2 = Image(BytesIO(chart_data2), width=5 * inch, height=3 * inch)
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
