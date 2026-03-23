"""Shared constants, styles, and helpers for PDF export functionality."""

import json
import os
from typing import Any

import matplotlib
import yaml
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


matplotlib.use("Agg")  # Non-interactive backend


# =============================================================================
# COMMON COLORS
# =============================================================================

COLORS = {
    "primary": colors.HexColor("#2c3e50"),
    "secondary": colors.HexColor("#666666"),
    "title": colors.HexColor("#1a1a1a"),
    "success": colors.HexColor("#27ae60"),
    "danger": colors.HexColor("#c0392b"),
    "warning": colors.HexColor("#f39c12"),
    "info": colors.HexColor("#3498db"),
    "header_bg": colors.HexColor("#34495e"),
    "row_alt": colors.HexColor("#f5f5f5"),
}


def _load_evaluation_data(input_file: str) -> dict[str, Any]:
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

    with open(input_file, encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif ext == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .yaml, .yml, or .json")

    return data


def _escape_xml(text: str) -> str:
    """Escape XML special characters for ReportLab Paragraph."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_content_for_pdf(text: str) -> str:
    """Format text content for PDF display, preserving newlines and indentation."""
    escaped = _escape_xml(str(text))

    # Process line by line to preserve leading indentation
    lines = escaped.split("\n")
    formatted_lines = []
    for line in lines:
        # Count leading spaces and convert to non-breaking spaces
        stripped = line.lstrip(" ")
        leading_spaces = len(line) - len(stripped)
        if leading_spaces > 0:
            # Use &nbsp; for leading spaces to preserve indentation
            line = "&nbsp;" * leading_spaces + stripped
        formatted_lines.append(line)

    # Join with <br/> for proper line breaks in PDF
    return "<br/>".join(formatted_lines)


def _render_tool_calls(tool_calls: dict[str, Any], story: list, label_style, detail_style) -> None:
    """Render tool_calls specification for a criterion."""
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
        ("prohibited", tool_calls.get("prohibited", [])),
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
                        params_str = ", ".join(
                            [f"{k}={_escape_xml(str(v))}" for k, v in params.items()]
                        )
                        story.append(Paragraph(f"params: {params_str}", detail_style))

    story.append(Spacer(1, 0.05 * inch))


def _create_tool_breakdown_section(breakdown: dict[str, Any], story: list) -> None:
    """Create a tool calls breakdown section for a criterion."""
    styles = getSampleStyleSheet()

    subheading_style = ParagraphStyle(
        "BreakdownHeading",
        parent=styles["Heading4"],
        fontSize=11,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=6,
        spaceBefore=8,
    )

    cell_style = ParagraphStyle(
        "BreakdownCell", parent=styles["Normal"], fontSize=7, leading=9, alignment=TA_LEFT
    )

    header_style = ParagraphStyle(
        "BreakdownHeader",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        textColor=colors.whitesmoke,
        fontName="Helvetica-Bold",
    )

    issue_style = ParagraphStyle(
        "IssueStyle",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#c0392b"),
        leftIndent=10,
    )

    # Header with overall stats
    order_status = (
        "✓" if breakdown.get("order_ok") else ("✗" if breakdown.get("order_ok") is False else "N/A")
    )
    header_text = f"<b>Tool Calls Breakdown</b> | Score: {breakdown.get('overall_score', 0):.1f}/3 | Order: {order_status}"
    story.append(Paragraph(header_text, subheading_style))

    # Build breakdown table
    tool_results = breakdown.get("tool_results", [])
    if tool_results:
        table_data = [
            [
                Paragraph("Tool", header_style),
                Paragraph("Type", header_style),
                Paragraph("Called", header_style),
                Paragraph("Count", header_style),
                Paragraph("Params", header_style),
                Paragraph("Score", header_style),
            ]
        ]

        for tr in tool_results:
            called_icon = "✓" if tr.get("called") else "✗"
            count_icon = "✓" if tr.get("count_ok") else "✗"
            params_ok = tr.get("params_ok")
            params_icon = "✓" if params_ok else ("✗" if params_ok is False else "N/A")

            table_data.append(
                [
                    Paragraph(tr.get("name", ""), cell_style),  # Full tool name, no truncation
                    Paragraph(tr.get("type", ""), cell_style),
                    Paragraph(called_icon, cell_style),
                    Paragraph(f"{tr.get('count', 0)} {count_icon}", cell_style),
                    Paragraph(params_icon, cell_style),
                    Paragraph(f"{tr.get('score', 0):.1f}/{tr.get('max_score', 0):.1f}", cell_style),
                ]
            )

        # Wider tool name column to fit full names
        col_widths = [3.2 * inch, 0.7 * inch, 0.5 * inch, 0.6 * inch, 0.5 * inch, 0.7 * inch]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5d6d7e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, 0), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                    ("FONTSIZE", (0, 1), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 1), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
                ]
            )
        )
        story.append(table)

    # Issues section
    issues = breakdown.get("issues", [])
    if issues:
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph("<b>Issues:</b>", cell_style))
        for issue in issues[:5]:  # Limit to 5 issues
            escaped_issue = issue.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(f"• {escaped_issue}", issue_style))

    story.append(Spacer(1, 0.1 * inch))


def _create_judges_panel_summary(
    judge_panel: dict[str, Any] | None, results: list[dict[str, Any]], story: list
) -> None:
    """Create LLM Judges Panel Summary section as a table."""
    if not judge_panel:
        return

    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    header_style = ParagraphStyle(
        "JudgeHeader",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.whitesmoke,
        fontName="Helvetica-Bold",
        alignment=TA_LEFT,
    )

    cell_style = ParagraphStyle(
        "JudgeCell", parent=styles["Normal"], fontSize=11, alignment=TA_LEFT
    )

    story.append(Paragraph("LLM Judges Panel Summary", heading_style))
    story.append(Spacer(1, 0.2 * inch))

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
    judge_names = (
        ", ".join(f"{j.get('name', 'unknown')} ({j.get('model', 'unknown')})" for j in judges)
        if judges
        else "N/A"
    )

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
        [
            Paragraph("Consensus Reached", cell_style),
            Paragraph(f"{consensus_count}/{total_criteria} ({consensus_pct:.0f}%)", cell_style),
        ],
    ]

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


def _create_results_table(results: list[dict[str, Any]], story: list) -> None:
    """Create detailed results table with proper text wrapping."""
    styles = getSampleStyleSheet()

    # Create styles for table cells
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )

    header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        textColor=colors.whitesmoke,
        fontName="Helvetica-Bold",
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=12,
    )

    story.append(Paragraph("Detailed Results", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # Prepare table data with Paragraph objects for text wrapping
    # Header row
    table_data = [
        [
            Paragraph("Criterion", header_style),
            Paragraph("Dimension", header_style),
            Paragraph("Result", header_style),
            Paragraph("Score", header_style),
            Paragraph("Reason", header_style),
        ]
    ]

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
        table_data.append(
            [
                Paragraph(criterion_name.replace("&", "&amp;"), cell_style),
                Paragraph(dimension.replace("&", "&amp;"), cell_style),
                Paragraph(result.replace("&", "&amp;"), cell_style),
                Paragraph(score, cell_style),
                Paragraph(reason.replace("&", "&amp;"), cell_style),
            ]
        )

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
        usable_width - (1.8 + 1.5 + 0.7 + 0.7) * inch,  # Reason (remaining space)
    ]

    results_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
            ]
        )
    )

    story.append(results_table)
    story.append(Spacer(1, 0.2 * inch))

    # Render tool breakdowns for criteria that have them
    if criteria_with_breakdowns:
        breakdown_heading = ParagraphStyle(
            "BreakdownSectionHeading",
            parent=styles["Heading3"],
            fontSize=14,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=8,
            spaceBefore=12,
        )
        story.append(Paragraph("Tool Calls Breakdowns", breakdown_heading))

        for criterion_name, breakdown in criteria_with_breakdowns:
            criterion_label = ParagraphStyle(
                "CriterionLabel",
                parent=styles["Normal"],
                fontSize=10,
                fontName="Helvetica-Bold",
                textColor=colors.HexColor("#34495e"),
                spaceAfter=4,
            )
            story.append(Paragraph(f"Criterion: {criterion_name}", criterion_label))
            _create_tool_breakdown_section(breakdown, story)

    story.append(Spacer(1, 0.3 * inch))


def _create_input_section(input_data: dict[str, Any] | None, story: list) -> None:
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
                with open(source_file, encoding="utf-8") as f:
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
        "InputHeading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=COLORS["primary"],
        spaceAfter=12,
        spaceBefore=20,
    )

    subheading_style = ParagraphStyle(
        "InputSubHeading",
        parent=styles["Heading3"],
        fontSize=12,
        textColor=COLORS["secondary"],
        spaceAfter=8,
        spaceBefore=12,
    )

    content_style = ParagraphStyle(
        "InputContent",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=8,
        leftIndent=10,
        rightIndent=10,
        backColor=COLORS["row_alt"],
        borderPadding=8,
    )

    qa_label_style = ParagraphStyle(
        "QALabel",
        parent=styles["Normal"],
        fontSize=10,
        fontName="Helvetica-Bold",
        textColor=COLORS["header_bg"],
        spaceAfter=4,
        spaceBefore=8,
    )

    qa_content_style = ParagraphStyle(
        "QAContent",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=12,
        leftIndent=15,
        rightIndent=10,
        backColor=COLORS["row_alt"],
        borderPadding=6,
    )

    story.append(PageBreak())
    story.append(Paragraph("Input Content", heading_style))

    source_file = input_data.get("source_file", "")

    # Display source info
    if source_file:
        source_info = f"<i>Source: {_escape_xml(str(source_file))} ({input_type})</i>"
        story.append(Paragraph(source_info, content_style))

    story.append(Spacer(1, 0.2 * inch))

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


def _render_qna_content(
    content: str, story: list, subheading_style, label_style, content_style
) -> None:
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
                    story.append(Spacer(1, 0.1 * inch))
    except Exception:
        # Fall back to raw content display
        escaped = _escape_xml(content)
        story.append(Paragraph(escaped, content_style))


def _render_single_qa(qa_data: dict[str, Any], story: list, label_style, content_style) -> None:
    """Render a single Q&A pair with proper multiline and code block handling."""
    getSampleStyleSheet()

    # Code style for content that looks like code (has code fences or indentation)
    code_style = ParagraphStyle(
        "QACode",
        parent=content_style,
        fontName="Courier",
        fontSize=8,
        leading=10,
        backColor=colors.HexColor("#f5f5f5"),
        borderColor=colors.HexColor("#dddddd"),
        borderWidth=0.5,
        borderPadding=8,
    )

    # Question
    question = qa_data.get("question", "")
    if question:
        story.append(Paragraph("Question:", label_style))
        story.append(Spacer(1, 0.05 * inch))
        formatted_q = _format_content_for_pdf(question)
        story.append(Paragraph(formatted_q, content_style))
        story.append(Spacer(1, 0.15 * inch))

    # Context (if present)
    context = qa_data.get("context", "")
    if context:
        story.append(Paragraph("Context:", label_style))
        story.append(Spacer(1, 0.05 * inch))
        formatted_ctx = _format_content_for_pdf(context)
        # Truncate very long context
        if len(formatted_ctx) > 2000:
            formatted_ctx = formatted_ctx[:2000] + "... [truncated]"
        story.append(Paragraph(formatted_ctx, content_style))
        story.append(Spacer(1, 0.15 * inch))

    # Answer
    answer = qa_data.get("answer", "")
    if answer:
        story.append(Paragraph("Answer:", label_style))
        story.append(Spacer(1, 0.1 * inch))  # Add spacing between label and content
        formatted_a = _format_content_for_pdf(answer)
        # Use code style if answer contains code fences or looks like code
        if "```" in answer or answer.strip().startswith(
            ("def ", "class ", "#!/", "import ", "from ")
        ):
            story.append(Paragraph(formatted_a, code_style))
        else:
            story.append(Paragraph(formatted_a, content_style))


def _render_chat_content(content: str, story: list, subheading_style, content_style) -> None:
    """Render chat session content."""
    story.append(Paragraph("Chat Session", subheading_style))

    # Split content into manageable chunks for better rendering
    lines = content.split("\n")

    # Process content in chunks to avoid memory issues with very large sessions
    chunk_size = 100
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i : i + chunk_size])
        escaped = _escape_xml(chunk)
        # Replace newlines with <br/> for PDF rendering
        escaped = escaped.replace("\n", "<br/>")
        story.append(Paragraph(escaped, content_style))

        if i + chunk_size < len(lines):
            story.append(Spacer(1, 0.05 * inch))


def _create_rubric_appendix(rubric_data: dict[str, Any] | None, story: list) -> None:
    """Create Rubric Appendix section with Dimensions and Criteria."""
    if not rubric_data:
        return
    dimensions = rubric_data.get("dimensions", [])
    criteria = rubric_data.get("criteria", [])

    if not dimensions and not criteria:
        return

    styles = getSampleStyleSheet()

    heading_style = ParagraphStyle(
        "AppendixHeading",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=20,
        spaceBefore=20,
    )

    subheading_style = ParagraphStyle(
        "SubHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=10,
        spaceBefore=15,
    )

    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=6
    )

    item_style = ParagraphStyle(
        "ItemStyle", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=4, leftIndent=15
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
                desc_escaped = (
                    description.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                story.append(Paragraph(desc_escaped, item_style))

            if scores:
                scores_text = "Scores: " + ", ".join([f"{k}: {v}" for k, v in scores.items()])
                story.append(Paragraph(scores_text, item_style))

        story.append(Spacer(1, 0.2 * inch))

    # Criteria section
    if criteria:
        story.append(Paragraph("Criteria", subheading_style))

        tool_label_style = ParagraphStyle(
            "ToolLabelStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            spaceAfter=2,
            leftIndent=20,
            textColor=colors.HexColor("#555555"),
        )

        tool_detail_style = ParagraphStyle(
            "ToolDetailStyle",
            parent=styles["Normal"],
            fontSize=9,
            leading=11,
            spaceAfter=2,
            leftIndent=30,
            fontName="Courier",
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
                text_escaped = (
                    criterion_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                )
                story.append(Paragraph(text_escaped, item_style))

            # Render tool_calls if present
            if tool_calls:
                _render_tool_calls(tool_calls, story, tool_label_style, tool_detail_style)
