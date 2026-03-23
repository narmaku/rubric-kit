"""Arena PDF export functionality for comparative evaluation results."""

from collections import defaultdict
from datetime import datetime
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    Table,
    TableStyle,
)

from rubric_kit.reports.pdf_base import (
    _create_input_section,
    _create_judges_panel_summary,
    _create_results_table,
    _create_rubric_appendix,
    _load_evaluation_data,
)


def _create_arena_title_page(
    metadata: dict[str, Any] | None, arena_name: str, arena_description: str | None, story: list
) -> None:
    """Create title page for Arena comparative report with description and metadata table."""
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        "ArenaTitle",
        parent=styles["Heading1"],
        fontSize=24,
        textColor=colors.HexColor("#1a1a1a"),
        spaceAfter=10,
        alignment=TA_CENTER,
    )

    subtitle_style = ParagraphStyle(
        "ArenaSubtitle",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#666666"),
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    description_style = ParagraphStyle(
        "ArenaDescription",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#444444"),
        alignment=TA_CENTER,
        spaceAfter=20,
        leftIndent=0.5 * inch,
        rightIndent=0.5 * inch,
    )

    story.append(Spacer(1, 1.5 * inch))

    # Report type identifier
    story.append(Paragraph("⚔️ Arena Comparative Evaluation Report", subtitle_style))

    # Use custom title from metadata if provided, else use arena name
    report_title = metadata.get("report_title", arena_name) if metadata else arena_name
    story.append(Paragraph(report_title, title_style))

    # Display arena description if provided
    if arena_description:
        story.append(Spacer(1, 0.2 * inch))
        escaped_desc = (
            arena_description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        story.append(Paragraph(f"<i>{escaped_desc}</i>", description_style))

    story.append(Spacer(1, 0.4 * inch))

    # Metadata table
    if metadata:
        header_style = ParagraphStyle(
            "MetaTableHeader",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.whitesmoke,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
        )

        cell_style = ParagraphStyle(
            "MetaTableCell", parent=styles["Normal"], fontSize=10, alignment=TA_LEFT
        )

        # Build metadata table data - collect all available metadata
        table_data = [[Paragraph("Property", header_style), Paragraph("Value", header_style)]]

        # Arena spec file
        if metadata.get("arena_spec_file"):
            table_data.append(
                [
                    Paragraph("Arena Spec", cell_style),
                    Paragraph(str(metadata["arena_spec_file"]), cell_style),
                ]
            )

        # Rubric source file
        if metadata.get("rubric_source_file"):
            table_data.append(
                [
                    Paragraph("Rubric File", cell_style),
                    Paragraph(str(metadata["rubric_source_file"]), cell_style),
                ]
            )

        # Judge panel source
        if metadata.get("judge_panel_source_file"):
            table_data.append(
                [
                    Paragraph("Judge Panel", cell_style),
                    Paragraph(str(metadata["judge_panel_source_file"]), cell_style),
                ]
            )

        # Timestamp
        if metadata.get("timestamp"):
            try:
                dt = datetime.fromisoformat(metadata["timestamp"].replace("Z", "+00:00"))
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                timestamp_str = str(metadata["timestamp"])
            table_data.append(
                [Paragraph("Generated", cell_style), Paragraph(timestamp_str, cell_style)]
            )

        # Source files (if combined from outputs)
        if metadata.get("source_files"):
            source_files = metadata["source_files"]
            if isinstance(source_files, list):
                source_str = ", ".join(str(f) for f in source_files[:3])
                if len(source_files) > 3:
                    source_str += f" (+{len(source_files) - 3} more)"
            else:
                source_str = str(source_files)
            table_data.append(
                [Paragraph("Source Files", cell_style), Paragraph(source_str, cell_style)]
            )

        # Combined from outputs flag
        if metadata.get("combined_from_outputs"):
            table_data.append(
                [
                    Paragraph("Mode", cell_style),
                    Paragraph("Combined from existing outputs", cell_style),
                ]
            )

        # Add any other metadata fields not explicitly handled
        known_keys = {
            "arena_spec_file",
            "rubric_source_file",
            "judge_panel_source_file",
            "timestamp",
            "source_files",
            "combined_from_outputs",
            "report_title",
        }
        for key, value in metadata.items():
            if key not in known_keys and value is not None:
                # Format the key nicely (replace underscores, title case)
                display_key = key.replace("_", " ").title()
                display_value = str(value)
                if len(display_value) > 80:
                    display_value = display_value[:77] + "..."
                table_data.append(
                    [Paragraph(display_key, cell_style), Paragraph(display_value, cell_style)]
                )

        # Only create table if we have data beyond the header
        if len(table_data) > 1:
            page_width = letter[0]
            margin = 0.75 * inch
            usable_width = page_width - (2 * margin)

            table = Table(table_data, colWidths=[2 * inch, usable_width - 2 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("TOPPADDING", (0, 0), (-1, 0), 8),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                        ("FONTSIZE", (0, 1), (-1, -1), 10),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 1), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
                    ]
                )
            )

            story.append(table)

    story.append(PageBreak())


def _create_arena_rankings_section(rankings: list[dict[str, Any]], story: list) -> None:
    """Create rankings table section."""
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    header_style = ParagraphStyle(
        "RankingsHeader",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.whitesmoke,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
    )

    cell_style = ParagraphStyle(
        "RankingsCell", parent=styles["Normal"], fontSize=11, alignment=TA_LEFT
    )

    story.append(Paragraph("Rankings Summary", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # Build table data
    table_data = [
        [
            Paragraph("Rank", header_style),
            Paragraph("Contestant", header_style),
            Paragraph("Score", header_style),
            Paragraph("Percentage", header_style),
        ]
    ]

    for r in rankings:
        medal = (
            "🥇"
            if r["rank"] == 1
            else ("🥈" if r["rank"] == 2 else ("🥉" if r["rank"] == 3 else ""))
        )
        rank_text = f"{medal} #{r['rank']}" if medal else f"#{r['rank']}"

        table_data.append(
            [
                Paragraph(rank_text, cell_style),
                Paragraph(r["name"], cell_style),
                Paragraph(f"{r['total_score']}/{r['max_score']}", cell_style),
                Paragraph(f"{r['percentage']:.1f}%", cell_style),
            ]
        )

    page_width = letter[0]
    margin = 0.75 * inch
    usable_width = page_width - (2 * margin)

    table = Table(table_data, colWidths=[1 * inch, 3 * inch, 1.5 * inch, usable_width - 5.5 * inch])
    table.setStyle(
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

    story.append(table)
    story.append(Spacer(1, 0.3 * inch))


def _create_comparative_bar_chart(contestants: dict[str, Any], story: list) -> None:
    """Create comparative bar chart for all contestants."""
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    story.append(Paragraph("Comparative Performance by Dimension", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # Extract dimension scores per contestant
    dimension_scores: dict[str, dict[str, float]] = {}  # dimension -> {contestant_id: percentage}
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
        offset = (i - len(contestant_ids) / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=contestant_names[cid], color=colors_list[i])

    ax.set_ylabel("Score (%)")
    ax.set_title("Performance Comparison by Dimension")
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    chart_img = Image(buf, width=6.5 * inch, height=4 * inch)
    story.append(chart_img)
    story.append(Spacer(1, 0.3 * inch))


def _create_radar_charts(contestants: dict[str, Any], story: list) -> None:
    """Create individual radar/spider charts for each contestant's performance profile."""
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    story.append(Paragraph("Performance Profiles (Radar Charts)", heading_style))
    story.append(Spacer(1, 0.2 * inch))

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
            "percentage": cdata.get("summary", {}).get("percentage", 0),
        }

    dimensions = sorted(all_dimensions)

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
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), subplot_kw={"polar": True})

    # Flatten axes for easy iteration
    if num_contestants == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette for contestants
    color_palette = [
        "#2ecc71",
        "#3498db",
        "#e74c3c",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#34495e",
        "#e67e22",
    ]

    for idx, (_contestant_id, cdata) in enumerate(contestant_data.items()):
        ax = axes[idx]

        # Get scores for all dimensions (0 if not present)
        values = [cdata["scores"].get(dim, 0) for dim in dimensions]
        values += values[:1]  # Close the polygon

        color = color_palette[idx % len(color_palette)]

        # Plot the radar
        ax.plot(angles, values, "o-", linewidth=2, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.25, color=color)

        # Configure the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, size=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7, color="gray")
        ax.grid(True, alpha=0.3)

        # Title with contestant name and overall score
        title = f"{cdata['name']}\n({cdata['percentage']:.1f}%)"
        ax.set_title(title, size=11, fontweight="bold", pad=15)

    # Hide empty subplots if odd number of contestants
    for idx in range(num_contestants, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)

    # Scale height based on number of rows
    chart_height = min(3.5 * nrows, 9) * inch
    chart_img = Image(buf, width=6.5 * inch, height=chart_height)
    story.append(chart_img)
    story.append(Spacer(1, 0.3 * inch))


def _create_contestant_details_section(
    contestants: dict[str, Any], story: list, *, include_input: bool = False
) -> None:
    """Create detailed results section for each contestant."""
    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    contestant_heading_style = ParagraphStyle(
        "ContestantHeading",
        parent=styles["Heading3"],
        fontSize=14,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=8,
        spaceBefore=12,
    )

    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#666666"),
        leftIndent=15,
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
        story.append(
            Paragraph(
                f"<b>Score:</b> {summary.get('total_score', 0)}/{summary.get('max_score', 0)} "
                f"({summary.get('percentage', 0):.1f}%)",
                meta_style,
            )
        )

        story.append(Spacer(1, 0.1 * inch))

        # Input content (Q&A / chat session) per contestant
        if include_input:
            input_data = cdata.get("input")
            if input_data:
                _create_input_section(input_data, story)

        # Results table (compact)
        results = cdata.get("results", [])
        if results:
            _create_results_table(results, story)

        story.append(Spacer(1, 0.2 * inch))


def export_arena_pdf(input_file: str, output_file: str, *, include_input: bool = False) -> None:
    """
    Export arena comparative evaluation results to PDF format.

    Args:
        input_file: Path to input YAML file with arena results
        output_file: Path to output PDF file
        include_input: If True, include the input content (Q&A / answers)
            for each contestant in the report.
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
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
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

    # Contestant details (with optional input content)
    _create_contestant_details_section(contestants, story, include_input=include_input)

    # Rubric Appendix
    _create_rubric_appendix(rubric_data, story)

    # Build PDF
    doc.build(story)
