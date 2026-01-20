"""Tests for PDF export functionality."""

import json
import os
import tempfile

import pytest
import yaml


@pytest.fixture
def sample_results():
    """Sample evaluation results for testing."""
    return [
        {
            "criterion_name": "fact_1",
            "criterion_text": "Check fact 1",
            "category": "Output",
            "dimension": "factual_correctness",
            "result": "pass",
            "score": 3,
            "max_score": 3,
            "reason": "The fact is correct",
            "consensus_reached": True,
        },
        {
            "criterion_name": "fact_2",
            "criterion_text": "Check fact 2",
            "category": "Output",
            "dimension": "factual_correctness",
            "result": "fail",
            "score": 0,
            "max_score": 2,
            "reason": "The fact is incorrect",
            "consensus_reached": True,
        },
        {
            "criterion_name": "useful_1",
            "criterion_text": "from_scores",
            "category": "Output",
            "dimension": "usefulness",
            "result": 3,
            "score": 3,
            "max_score": 3,
            "reason": "Very useful",
            "consensus_reached": False,
        },
    ]


@pytest.fixture
def sample_judge_panel():
    """Sample judge panel config for testing."""
    return {
        "judges": [
            {"name": "primary", "model": "gpt-4", "base_url": None},
            {"name": "secondary", "model": "gpt-4-turbo", "base_url": None},
        ],
        "execution": {"mode": "sequential", "batch_size": 2, "timeout": 30},
        "consensus": {"mode": "majority", "threshold": 2, "on_no_consensus": "fail"},
    }


@pytest.fixture
def sample_rubric():
    """Sample rubric data for testing."""
    return {
        "dimensions": [
            {
                "name": "factual_correctness",
                "description": "Evaluates factual accuracy of responses",
                "grading_type": "binary",
                "scores": None,
                "pass_above": None,
            },
            {
                "name": "usefulness",
                "description": "Evaluates how useful the response is",
                "grading_type": "score",
                "scores": {1: "Not useful", 2: "Somewhat useful", 3: "Very useful"},
                "pass_above": None,
            },
        ],
        "criteria": [
            {
                "name": "fact_1",
                "category": "Output",
                "dimension": "factual_correctness",
                "criterion": "Check that the response contains correct facts",
                "weight": 3,
                "tool_calls": None,
            },
            {
                "name": "useful_1",
                "category": "Output",
                "dimension": "usefulness",
                "criterion": "from_scores",
                "weight": "from_scores",
                "tool_calls": None,
            },
        ],
    }


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "timestamp": "2024-01-01T12:00:00",
        "rubric_source_file": "test_rubric.yaml",
        "judge_panel_source_file": None,
        "report_title": "Q1 2025 Custom Evaluation Report",
    }


def test_export_evaluation_pdf_from_yaml(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF from YAML file."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file with new structure
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "summary": {"total_score": 6, "max_score": 8, "percentage": 75.0},
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "input": {"type": "chat_session", "source_file": "test.txt"},
                "metadata": sample_metadata,
            },
            f,
        )

    # Create temporary PDF output
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created and has content
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_from_json(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF from JSON file."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary JSON file with new structure
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json_path = f.name
        json.dump(
            {
                "results": sample_results,
                "summary": {"total_score": 6, "max_score": 8, "percentage": 75.0},
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "input": {"type": "chat_session", "source_file": "test.txt"},
                "metadata": sample_metadata,
            },
            f,
        )

    # Create temporary PDF output
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(json_path, pdf_path)

        # Verify PDF was created
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(json_path):
            os.unlink(json_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_minimal(sample_results):
    """Test exporting PDF with minimal data (just results)."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file with minimal data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump({"results": sample_results}, f)

    # Create temporary PDF output
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Should still work with minimal data
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_invalid_input():
    """Test exporting PDF with invalid input file."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        with pytest.raises(FileNotFoundError):
            export_evaluation_pdf("nonexistent.yaml", pdf_path)
    finally:
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_with_custom_title(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF uses custom report_title from metadata."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file with custom title
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "metadata": sample_metadata,  # Contains report_title
            },
            f,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_with_rubric_appendix(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF includes rubric appendix with dimensions and criteria."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file with rubric data at top level
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "metadata": sample_metadata,
            },
            f,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created and has content
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_with_judges_panel_summary(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF includes LLM judges panel summary."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file with judge_panel at top level
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "metadata": sample_metadata,
            },
            f,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_includes_chat_session(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF always includes chat session content."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    chat_content = (
        "User: What is the capital of France?\nAssistant: The capital of France is Paris."
    )

    # Create temporary YAML file with input content (new structured format)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "input": {
                    "type": "chat_session",
                    "source_file": "test_chat.txt",
                    "chat_session": chat_content,
                },
                "metadata": sample_metadata,
            },
            f,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_includes_qna(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF always includes Q&A content."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file with Q&A input (new structured format)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "input": {
                    "type": "qna",
                    "source_file": "test_qna.yaml",
                    "question": "What is the capital of France?",
                    "answer": "The capital of France is Paris.",
                    "context": "Geography quiz",
                },
                "metadata": sample_metadata,
            },
            f,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)


def test_export_evaluation_pdf_without_input_content(
    sample_results, sample_metadata, sample_rubric, sample_judge_panel
):
    """Test exporting PDF gracefully handles missing input content."""
    from rubric_kit.pdf_export import export_evaluation_pdf

    # Create temporary YAML file without input content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml_path = f.name
        yaml.dump(
            {
                "results": sample_results,
                "rubric": sample_rubric,
                "judge_panel": sample_judge_panel,
                "input": {
                    "type": "chat_session",
                    "source_file": "nonexistent_file.txt",
                    # No content field
                },
                "metadata": sample_metadata,
            },
            f,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name

    try:
        # Should not crash even when content is missing
        export_evaluation_pdf(yaml_path, pdf_path)

        # Verify PDF was created
        assert os.path.exists(pdf_path)
        assert os.path.getsize(pdf_path) > 0
    finally:
        if os.path.exists(yaml_path):
            os.unlink(yaml_path)
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
