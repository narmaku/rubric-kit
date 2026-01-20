"""Tests for main CLI script."""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from rubric_kit.schema import Criterion, Dimension, Rubric


@pytest.fixture
def sample_rubric_file():
    """Create a sample rubric YAML file."""
    rubric_yaml = """
dimensions:
  - factual_correctness: Test correctness
    grading_type: binary
  - usefulness: Test usefulness
    grading_type: score
    scores:
      1: Not useful
      2: Somewhat useful
      3: Very useful

criteria:
  fact_1:
    category: Output
    weight: 3
    dimension: factual_correctness
    criterion: Check fact 1
  useful_1:
    category: Output
    weight: from_scores
    dimension: usefulness
    criterion: from_scores
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(rubric_yaml)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_evaluation_yaml():
    """Create a sample evaluation YAML file for export tests (new self-contained format)."""
    data = {
        "results": [
            {
                "criterion_name": "fact_1",
                "category": "Output",
                "dimension": "factual_correctness",
                "result": "pass",
                "score": 3,
                "max_score": 3,
                "reason": "Correct",
            }
        ],
        "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
        "rubric": {
            "dimensions": [
                {
                    "name": "factual_correctness",
                    "description": "Test correctness",
                    "grading_type": "binary",
                    "scores": None,
                    "pass_above": None,
                }
            ],
            "criteria": [
                {
                    "name": "fact_1",
                    "category": "Output",
                    "dimension": "factual_correctness",
                    "criterion": "Check fact 1",
                    "weight": 3,
                    "tool_calls": None,
                }
            ],
        },
        "judge_panel": {
            "judges": [{"name": "default", "model": "gpt-4", "base_url": None}],
            "execution": {"mode": "sequential", "batch_size": 2, "timeout": 30},
            "consensus": {"mode": "unanimous", "threshold": 1, "on_no_consensus": "fail"},
        },
        "input": {
            "type": "chat_session",
            "source_file": "test.txt",
            "chat_session": "User: Test question?\nAssistant: Test answer.",
        },
        "metadata": {
            "timestamp": "2024-01-01T12:00:00",
            "rubric_source_file": "test.yaml",
            "judge_panel_source_file": None,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_evaluation_yaml_with_qna():
    """Create a sample evaluation YAML file with Q&A input (new structured format)."""
    data = {
        "results": [
            {
                "criterion_name": "fact_1",
                "category": "Output",
                "dimension": "factual_correctness",
                "result": "pass",
                "score": 3,
                "max_score": 3,
                "reason": "Correct",
            }
        ],
        "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
        "rubric": {
            "dimensions": [
                {
                    "name": "factual_correctness",
                    "description": "Test correctness",
                    "grading_type": "binary",
                    "scores": None,
                    "pass_above": None,
                }
            ],
            "criteria": [
                {
                    "name": "fact_1",
                    "category": "Output",
                    "dimension": "factual_correctness",
                    "criterion": "Check fact 1",
                    "weight": 3,
                    "tool_calls": None,
                }
            ],
        },
        "judge_panel": {
            "judges": [{"name": "default", "model": "gpt-4", "base_url": None}],
            "execution": {"mode": "sequential", "batch_size": 2, "timeout": 30},
            "consensus": {"mode": "unanimous", "threshold": 1, "on_no_consensus": "fail"},
        },
        "input": {
            "type": "qna",
            "source_file": "qna.yaml",
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
            "context": "Geography quiz",
        },
        "metadata": {
            "timestamp": "2024-01-01T12:00:00",
            "rubric_source_file": "test.yaml",
            "judge_panel_source_file": None,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_chat_session_file():
    """Create a sample chat session file."""
    chat_content = """User: Test question?
Assistant: Test answer with information."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(chat_content)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_qa_file():
    """Create a sample Q&A file for generation."""
    qa_content = """Q: What is the capital of France?
A: The capital of France is Paris."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(qa_content)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


class TestEvaluateCommand:
    """Test the 'evaluate' subcommand."""

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_evaluate_command(
        self, mock_eval_llm, sample_rubric_file, sample_chat_session_file, capsys
    ):
        """Test the evaluate subcommand with LLM judge - always outputs YAML."""
        from rubric_kit.main import main

        # Mock LLM evaluations
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            # Call main with evaluate subcommand
            import sys

            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                sample_chat_session_file,
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
            ]

            result = main()

            # Should return 0 for success
            assert result == 0

            # Check that output file was created (YAML)
            assert os.path.exists(output_path)

            # Verify it's valid YAML with expected structure
            with open(output_path) as f:
                data = yaml.safe_load(f)
            assert "results" in data
            assert "metadata" in data

            # Check that table was printed
            captured = capsys.readouterr()
            assert "fact_1" in captured.out
            assert "useful_1" in captured.out

            # Verify LLM was called
            assert mock_eval_llm.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch("rubric_kit.main.export_evaluation_pdf")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_evaluate_with_report(
        self, mock_pdf, mock_eval_llm, sample_rubric_file, sample_chat_session_file
    ):
        """Test evaluate subcommand with --report flag generates PDF."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                sample_chat_session_file,
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
                "--report",
                pdf_path,
            ]

            result = main()

            assert result == 0
            assert mock_pdf.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_evaluate_with_report_title(
        self, mock_eval_llm, sample_rubric_file, sample_chat_session_file
    ):
        """Test evaluate subcommand with --report-title stores title in metadata."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                sample_chat_session_file,
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
                "--report-title",
                "Q1 2025 Evaluation",
            ]

            result = main()

            assert result == 0

            # Verify report title is in metadata
            with open(output_path) as f:
                data = yaml.safe_load(f)
            assert data["metadata"]["report_title"] == "Q1 2025 Evaluation"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_evaluate_output_is_self_contained(
        self, mock_eval_llm, sample_rubric_file, sample_chat_session_file
    ):
        """Test evaluate subcommand produces self-contained output with rubric and judge_panel at top level."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                sample_chat_session_file,
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            # Verify self-contained structure
            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Rubric at top level (not in metadata)
            assert "rubric" in data
            assert "dimensions" in data["rubric"]
            assert "criteria" in data["rubric"]
            assert len(data["rubric"]["dimensions"]) == 2
            assert len(data["rubric"]["criteria"]) == 2

            # Judge panel at top level
            assert "judge_panel" in data
            assert "judges" in data["judge_panel"]
            assert "execution" in data["judge_panel"]
            assert "consensus" in data["judge_panel"]

            # Input section
            assert "input" in data
            assert data["input"]["type"] == "chat_session"

            # Summary section
            assert "summary" in data
            assert "total_score" in data["summary"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_evaluate_always_includes_input_content(
        self, mock_eval_llm, sample_rubric_file, sample_chat_session_file
    ):
        """Test that evaluate subcommand always includes input content in output."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            # Note: No --include-input flag - content should be included by default
            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                sample_chat_session_file,
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            # Verify input content is included in new structured format
            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert "input" in data
            assert data["input"]["type"] == "chat_session"
            # New format uses chat_session key for chat sessions
            assert "chat_session" in data["input"]
            assert data["input"]["chat_session"] is not None
            assert len(data["input"]["chat_session"]) > 0
            # Verify it contains the actual chat session content
            assert "Test question" in data["input"]["chat_session"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_evaluate_with_missing_api_key(self, sample_rubric_file, sample_chat_session_file):
        """Test evaluate subcommand without API key."""
        import sys

        from rubric_kit.main import main

        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                sample_chat_session_file,
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                "output.yaml",
            ]

            result = main()

            # Should return non-zero for error
            assert result == 1

    def test_evaluate_with_missing_file(self):
        """Test evaluate subcommand with missing file."""
        import sys

        from rubric_kit.main import main

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            sys.argv = [
                "rubric-kit",
                "evaluate",
                "--from-chat-session",
                "nonexistent.txt",
                "--rubric-file",
                "nonexistent.yaml",
                "--output-file",
                "output.yaml",
            ]

            result = main()

            # Should return non-zero for error
            assert result != 0


class TestGenerateCommand:
    """Test the 'generate' subcommand."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_command(self, mock_generator_class, sample_qa_file):
        """Test the generate subcommand."""
        import sys

        from rubric_kit.main import main

        # Mock the generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock rubric generation
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="factual_correctness",
                    description="Test correctness",
                    grading_type="binary",
                )
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="Check fact 1",
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
            ]

            result = main()

            # Should return 0 for success
            assert result == 0

            # Check that output file was created
            assert os.path.exists(output_path)

            # Verify generator was called
            assert mock_generator.generate_rubric.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_with_parameters(self, mock_generator_class, sample_qa_file):
        """Test generate command with custom parameters."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
                "--num-dimensions",
                "3",
                "--num-criteria",
                "5",
                "--categories",
                "Output,Reasoning",
                "--model",
                "gpt-4-turbo",
            ]

            result = main()

            assert result == 0

            # Verify generator was called with correct parameters
            call_args = mock_generator.generate_rubric.call_args
            assert call_args[1]["num_dimensions"] == 3
            assert call_args[1]["num_criteria"] == 5
            assert call_args[1]["category_hints"] == ["Output", "Reasoning"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_generate_with_missing_api_key(self, sample_qa_file):
        """Test generate subcommand without API key."""
        import sys

        from rubric_kit.main import main

        with patch.dict(os.environ, {}, clear=True):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                output_path = f.name

            try:
                sys.argv = [
                    "rubric-kit",
                    "generate",
                    "--from-qna",
                    sample_qa_file,
                    "--output-file",
                    output_path,
                ]

                result = main()

                # Should return non-zero for error
                assert result == 1
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)


@pytest.fixture
def sample_dimensions_file():
    """Create a sample dimensions YAML file."""
    dims_yaml = """
dimensions:
  - name: tool_usage
    description: "Evaluates correct tool usage"
    grading_type: binary

  - name: factual_accuracy
    description: "Evaluates factual correctness"
    grading_type: binary
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(dims_yaml)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


class TestGenerateMetadata:
    """Test generate command metadata and metrics output."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_includes_metadata(self, mock_generator_class, sample_qa_file):
        """Test that generate command includes metadata in output YAML."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Verify metadata is included
            assert "metadata" in data
            assert "timestamp" in data["metadata"]
            assert "operation" in data["metadata"]
            assert data["metadata"]["operation"] == "generate"
            assert "model" in data["metadata"]
            assert "source_file" in data["metadata"]
            assert "source_type" in data["metadata"]
            assert data["metadata"]["source_type"] == "qna"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_includes_metrics_by_default(self, mock_generator_class, sample_qa_file):
        """Test that generate command includes metrics by default."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Verify metrics are included in metadata
            assert "metadata" in data
            assert "metrics" in data["metadata"]
            assert "summary" in data["metadata"]["metrics"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_no_metrics_excludes_metrics_but_keeps_metadata(
        self, mock_generator_class, sample_qa_file
    ):
        """Test that --no-metrics excludes metrics but keeps other metadata."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
                "--no-metrics",
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Verify metadata is still included
            assert "metadata" in data
            assert "timestamp" in data["metadata"]
            assert "operation" in data["metadata"]
            assert data["metadata"]["operation"] == "generate"
            assert "model" in data["metadata"]

            # Verify metrics are NOT included
            assert "metrics" not in data["metadata"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_from_chat_includes_metadata(
        self, mock_generator_class, sample_chat_session_file
    ):
        """Test that generate from chat includes metadata with correct source_type."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric_from_chat.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-chat-session",
                sample_chat_session_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert "metadata" in data
            assert data["metadata"]["source_type"] == "chat_session"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_metadata_includes_options(self, mock_generator_class, sample_qa_file):
        """Test that generate metadata includes generation options."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
                "--num-dimensions",
                "3",
                "--num-criteria",
                "5",
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert "metadata" in data
            assert "options" in data["metadata"]
            assert data["metadata"]["options"]["num_dimensions"] == 3
            assert data["metadata"]["options"]["num_criteria"] == 5
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRefineMetadata:
    """Test refine command metadata and metrics output."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_includes_metadata(self, mock_generator_class, sample_rubric_file):
        """Test that refine command includes metadata in output YAML."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Verify metadata is included
            assert "metadata" in data
            assert "timestamp" in data["metadata"]
            assert "operation" in data["metadata"]
            assert data["metadata"]["operation"] == "refine"
            assert "model" in data["metadata"]
            assert "source_rubric_file" in data["metadata"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_includes_metrics_by_default(self, mock_generator_class, sample_rubric_file):
        """Test that refine command includes metrics by default."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Verify metrics are included in metadata
            assert "metadata" in data
            assert "metrics" in data["metadata"]
            assert "summary" in data["metadata"]["metrics"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_no_metrics_excludes_metrics_but_keeps_metadata(
        self, mock_generator_class, sample_rubric_file
    ):
        """Test that --no-metrics excludes metrics but keeps other metadata."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
                "--no-metrics",
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Verify metadata is still included
            assert "metadata" in data
            assert "timestamp" in data["metadata"]
            assert "operation" in data["metadata"]
            assert data["metadata"]["operation"] == "refine"
            assert "model" in data["metadata"]

            # Verify metrics are NOT included
            assert "metrics" not in data["metadata"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_with_qa_context_includes_context_in_metadata(
        self, mock_generator_class, sample_rubric_file, sample_qa_file
    ):
        """Test that refine with Q&A context includes context info in metadata."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric_with_qa.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            assert "metadata" in data
            assert "context_file" in data["metadata"]
            assert "context_type" in data["metadata"]
            assert data["metadata"]["context_type"] == "qna"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestGenerateWithGuidelines:
    """Test generate command with --guidelines option."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_with_guidelines(self, mock_generator_class, sample_qa_file):
        """Test generate command passes guidelines to generator."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="factual_correctness",
                    description="Test correctness",
                    grading_type="binary",
                )
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="Check fact 1",
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        guidelines = "Focus on security aspects. Create atomic criteria."

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
                "--guidelines",
                guidelines,
            ]

            result = main()

            assert result == 0

            # Verify generator was called with guidelines
            call_args = mock_generator.generate_rubric.call_args
            assert call_args[1]["guidelines"] == guidelines
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_from_chat_with_guidelines(
        self, mock_generator_class, sample_chat_session_file
    ):
        """Test generate from chat command passes guidelines to generator."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[
                Dimension(name="tool_usage", description="Tool usage", grading_type="binary")
            ],
            criteria=[
                Criterion(
                    name="tool_1",
                    category="Tools",
                    weight=3,
                    dimension="tool_usage",
                    criterion="Check tool",
                )
            ],
        )
        mock_generator.generate_rubric_from_chat.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        guidelines = "Create granular tool usage criteria."

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-chat-session",
                sample_chat_session_file,
                "--output-file",
                output_path,
                "--guidelines",
                guidelines,
            ]

            result = main()

            assert result == 0

            # Verify generator was called with guidelines
            call_args = mock_generator.generate_rubric_from_chat.call_args
            assert call_args[1]["guidelines"] == guidelines
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_with_guidelines_file(self, mock_generator_class, sample_qa_file):
        """Test generate command reads guidelines from file."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        # Create guidelines file with multi-line content
        guidelines_content = """# AI Assistant Persona
The assistant being evaluated is a helpful coding assistant.
It should:
- Provide accurate code suggestions
- Follow best practices
- Explain reasoning clearly

Focus on evaluating these specific behaviors."""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(guidelines_content)
            guidelines_file = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
                "--guidelines-file",
                guidelines_file,
            ]

            result = main()

            assert result == 0

            # Verify generator was called with guidelines content from file
            call_args = mock_generator.generate_rubric.call_args
            assert call_args[1]["guidelines"] == guidelines_content
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists(guidelines_file):
                os.unlink(guidelines_file)

    def test_generate_guidelines_and_guidelines_file_mutually_exclusive(self, sample_qa_file):
        """Test that --guidelines and --guidelines-file cannot be used together."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Some guidelines")
            guidelines_file = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                "output.yaml",
                "--guidelines",
                "inline guidelines",
                "--guidelines-file",
                guidelines_file,
            ]

            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should exit with error due to mutually exclusive arguments
            assert exc_info.value.code != 0
        finally:
            if os.path.exists(guidelines_file):
                os.unlink(guidelines_file)


class TestGenerateWithDimensionsFile:
    """Test generate command with --dimensions-file."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_with_dimensions_file(
        self, mock_generator_class, sample_qa_file, sample_dimensions_file
    ):
        """Test generate command uses provided dimensions file."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="tool_usage", description="Evaluates tool usage", grading_type="binary"
                ),
                Dimension(
                    name="factual_accuracy", description="Evaluates facts", grading_type="binary"
                ),
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_accuracy",
                    criterion="Test",
                )
            ],
        )
        mock_generator.generate_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-qna",
                sample_qa_file,
                "--output-file",
                output_path,
                "--dimensions-file",
                sample_dimensions_file,
            ]

            result = main()

            assert result == 0

            # Verify generate_rubric was called with dimensions parameter
            call_args = mock_generator.generate_rubric.call_args
            assert "dimensions" in call_args[1]
            dims = call_args[1]["dimensions"]
            assert len(dims) == 2
            assert dims[0].name == "tool_usage"
            assert dims[1].name == "factual_accuracy"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_generate_from_chat_with_dimensions_file(
        self, mock_generator_class, sample_chat_session_file, sample_dimensions_file
    ):
        """Test generate from chat uses provided dimensions file."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="tool_usage", description="Evaluates tool usage", grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="tool_1",
                    category="Tools",
                    weight=3,
                    dimension="tool_usage",
                    criterion="Test",
                )
            ],
        )
        mock_generator.generate_rubric_from_chat.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "generate",
                "--from-chat-session",
                sample_chat_session_file,
                "--output-file",
                output_path,
                "--dimensions-file",
                sample_dimensions_file,
            ]

            result = main()

            assert result == 0

            # Verify generate_rubric_from_chat was called with dimensions parameter
            call_args = mock_generator.generate_rubric_from_chat.call_args
            assert "dimensions" in call_args[1]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRefineWithDimensionsFile:
    """Test refine command with --dimensions-file."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_with_dimensions_file(
        self, mock_generator_class, sample_rubric_file, sample_dimensions_file
    ):
        """Test refine command merges provided dimensions."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[
                Dimension(name="factual_correctness", description="Test", grading_type="binary"),
                Dimension(name="tool_usage", description="Tools", grading_type="binary"),
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="Test",
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
                "--dimensions-file",
                sample_dimensions_file,
            ]

            result = main()

            assert result == 0

            # Verify refine_rubric was called with dimensions_to_merge parameter
            call_args = mock_generator.refine_rubric.call_args
            assert "dimensions_to_merge" in call_args[1]
            dims = call_args[1]["dimensions_to_merge"]
            assert len(dims) == 2
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRefineCommand:
    """Test the 'refine' subcommand."""

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_command(self, mock_generator_class, sample_rubric_file):
        """Test the refine subcommand."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock refined rubric
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="factual_correctness",
                    description="Improved correctness description",
                    grading_type="binary",
                )
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="Improved criterion text",
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        sys.argv = ["rubric-kit", "refine", "--rubric-file", sample_rubric_file]

        result = main()

        # Should return 0 for success
        assert result == 0

        # Verify refine_rubric was called
        assert mock_generator.refine_rubric.called

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_with_feedback(self, mock_generator_class, sample_rubric_file):
        """Test refine command with feedback."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        feedback = "Add more specific criteria"
        sys.argv = [
            "rubric-kit",
            "refine",
            "--rubric-file",
            sample_rubric_file,
            "--feedback",
            feedback,
        ]

        result = main()

        assert result == 0

        # Verify feedback was passed
        call_args = mock_generator.refine_rubric.call_args
        assert call_args[1]["feedback"] == feedback

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_with_feedback_file(self, mock_generator_class, sample_rubric_file):
        """Test refine command reads feedback from file."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        # Create feedback file with multi-line content
        feedback_content = """# Refinement Instructions
The AI assistant being evaluated has the following persona:
- Expert Linux system administrator
- Should use diagnostic tools properly
- Must provide accurate system information

Please refine the rubric to:
1. Add criteria for tool usage order
2. Make factual checks more granular
3. Ensure each criterion checks exactly one fact"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(feedback_content)
            feedback_file = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--feedback-file",
                feedback_file,
            ]

            result = main()

            assert result == 0

            # Verify feedback content from file was passed
            call_args = mock_generator.refine_rubric.call_args
            assert call_args[1]["feedback"] == feedback_content
        finally:
            if os.path.exists(feedback_file):
                os.unlink(feedback_file)

    def test_refine_feedback_and_feedback_file_mutually_exclusive(self, sample_rubric_file):
        """Test that --feedback and --feedback-file cannot be used together."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Some feedback")
            feedback_file = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--feedback",
                "inline feedback",
                "--feedback-file",
                feedback_file,
            ]

            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should exit with error due to mutually exclusive arguments
            assert exc_info.value.code != 0
        finally:
            if os.path.exists(feedback_file):
                os.unlink(feedback_file)

    @patch("rubric_kit.main.RubricGenerator")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_refine_with_output(self, mock_generator_class, sample_rubric_file):
        """Test refine command with custom output path."""
        import sys

        from rubric_kit.main import main

        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        mock_rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_1", category="Output", weight=3, dimension="test", criterion="Test"
                )
            ],
        )
        mock_generator.refine_rubric.return_value = mock_rubric

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "refine",
                "--rubric-file",
                sample_rubric_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRerunCommand:
    """Test the 'rerun' subcommand."""

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_rerun_with_embedded_input(self, mock_eval_llm, sample_evaluation_yaml):
        """Test rerun subcommand uses settings from self-contained YAML."""
        import sys

        from rubric_kit.main import main

        # The sample_evaluation_yaml fixture already has embedded chat_session content

        mock_eval_llm.return_value = {"fact_1": {"type": "binary", "passes": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = ["rubric-kit", "rerun", sample_evaluation_yaml, "--output-file", output_path]

            result = main()

            assert result == 0
            assert os.path.exists(output_path)

            # Verify output has same structure
            with open(output_path) as f:
                new_data = yaml.safe_load(f)

            assert "rubric" in new_data
            assert "judge_panel" in new_data
            assert "results" in new_data
            assert new_data["metadata"].get("rerun_from") == sample_evaluation_yaml
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_rerun_with_new_input(
        self, mock_eval_llm, sample_evaluation_yaml, sample_chat_session_file
    ):
        """Test rerun with new input file overrides embedded/original input."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {"fact_1": {"type": "binary", "passes": True}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "rerun",
                sample_evaluation_yaml,
                "--from-chat-session",
                sample_chat_session_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0
            assert mock_eval_llm.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_rerun_missing_input_file(self):
        """Test rerun subcommand with missing input file."""
        import sys

        from rubric_kit.main import main

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
            sys.argv = ["rubric-kit", "rerun", "nonexistent.yaml", "--output-file", "output.yaml"]

            result = main()

            assert result != 0


class TestExportCommand:
    """Test the 'export' subcommand."""

    def test_export_to_pdf(self, sample_evaluation_yaml):
        """Test export subcommand with --format pdf."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "export",
                sample_evaluation_yaml,
                "--format",
                "pdf",
                "--output",
                pdf_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_export_to_csv(self, sample_evaluation_yaml):
        """Test export subcommand with --format csv."""
        import csv
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "export",
                sample_evaluation_yaml,
                "--format",
                "csv",
                "--output",
                csv_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(csv_path)

            # Verify CSV content (skip comment lines)
            with open(csv_path) as f:
                lines = [line for line in f if not line.startswith("#")]

            import io

            reader = csv.DictReader(io.StringIO("".join(lines)))
            rows = list(reader)
            assert len(rows) >= 1
            assert rows[0]["criterion_name"] == "fact_1"
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_export_to_json(self, sample_evaluation_yaml):
        """Test export subcommand with --format json."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "export",
                sample_evaluation_yaml,
                "--format",
                "json",
                "--output",
                json_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(json_path)

            # Verify JSON content
            with open(json_path) as f:
                data = json.load(f)
            assert "results" in data
            assert len(data["results"]) >= 1
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_export_missing_input_file(self):
        """Test export subcommand with missing input file."""
        import sys

        from rubric_kit.main import main

        sys.argv = [
            "rubric-kit",
            "export",
            "nonexistent.yaml",
            "--format",
            "pdf",
            "--output",
            "output.pdf",
        ]

        result = main()

        assert result != 0

    def test_export_requires_format(self, sample_evaluation_yaml):
        """Test export subcommand requires --format argument."""
        import sys

        from rubric_kit.main import main

        sys.argv = ["rubric-kit", "export", sample_evaluation_yaml, "--output", "output.pdf"]

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit with error due to missing required argument
        assert exc_info.value.code != 0

    def test_export_to_pdf_always_includes_input(self, sample_evaluation_yaml):
        """Test export to PDF always includes input content section."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "export",
                sample_evaluation_yaml,
                "--format",
                "pdf",
                "--output",
                pdf_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)

    def test_export_to_json_always_includes_input(self, sample_evaluation_yaml):
        """Test export to JSON always includes full input content."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "export",
                sample_evaluation_yaml,
                "--format",
                "json",
                "--output",
                json_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(json_path)

            # Verify JSON content includes full input
            with open(json_path) as f:
                data = json.load(f)
            assert "input" in data
            assert "chat_session" in data["input"]
            assert data["input"]["chat_session"] is not None
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)

    def test_export_to_csv_includes_header_comments(self, sample_evaluation_yaml):
        """Test export to CSV includes header comments with metadata and input summary."""
        import sys

        from rubric_kit.main import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "export",
                sample_evaluation_yaml,
                "--format",
                "csv",
                "--output",
                csv_path,
            ]

            result = main()

            assert result == 0
            assert os.path.exists(csv_path)

            # Verify CSV has header comments
            with open(csv_path) as f:
                content = f.read()

            # Should have comment header lines
            assert content.startswith("# Evaluation Report")
            assert "# Input Type:" in content
            # Should also have actual CSV data
            assert "criterion_name" in content
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


def test_cli_help():
    """Test CLI help message."""
    import sys

    from rubric_kit.main import main

    sys.argv = ["rubric-kit", "--help"]

    with pytest.raises(SystemExit) as exc_info:
        main()

    # Help should exit with 0
    assert exc_info.value.code == 0


def test_cli_no_subcommand():
    """Test CLI without subcommand shows help."""
    import sys

    from rubric_kit.main import main

    sys.argv = ["rubric-kit"]

    # Should either show help or return error
    result = main()
    assert result != 0 or result is None


class TestArenaCommand:
    """Test the 'arena' subcommand."""

    @pytest.fixture
    def sample_arena_spec_file(self, sample_rubric_file):
        """Create a sample arena spec YAML file."""
        # Create a judge panel file
        judge_panel_yaml = """
judge_panel:
  judges:
    - name: test_judge
      model: gpt-4
      api_key: null
  execution:
    mode: sequential
  consensus:
    mode: unanimous
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(judge_panel_yaml)
            judge_panel_path = f.name

        # Create chat session files
        session1_content = "User: Hello\nAssistant: Hi there!"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(session1_content)
            session1_path = f.name

        session2_content = "User: Test\nAssistant: Response"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(session2_content)
            session2_path = f.name

        # Create arena spec
        arena_spec = {
            "arena": {
                "name": "Test Arena",
                "description": "Test comparison",
                "rubric_file": sample_rubric_file,
                "judges_panel_file": judge_panel_path,
                "contestants": [
                    {"id": "model_a", "name": "Model A", "input_file": session1_path},
                    {
                        "id": "model_b",
                        "name": "Model B",
                        "input_file": session2_path,
                        "variables": {"test_var": "value"},
                    },
                ],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(arena_spec, f)
            arena_spec_path = f.name

        yield arena_spec_path

        # Cleanup
        os.unlink(arena_spec_path)
        os.unlink(judge_panel_path)
        os.unlink(session1_path)
        os.unlink(session2_path)

    def test_arena_cli_requires_arena_spec(self):
        """Test that arena subcommand requires --arena-spec argument."""
        import sys

        from rubric_kit.main import main

        sys.argv = ["rubric-kit", "arena"]

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit with error due to missing required argument
        assert exc_info.value.code != 0

    def test_arena_cli_parses_arena_spec(self, sample_arena_spec_file):
        """Test that arena subcommand correctly parses --arena-spec argument."""
        import sys

        from rubric_kit.main import main

        # Just test that the parser works, not the full command

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            # Patch sys.argv and verify parsing
            sys.argv = [
                "rubric-kit",
                "arena",
                "--arena-spec",
                sample_arena_spec_file,
                "--output-file",
                output_path,
            ]

            # This should parse correctly and complete (failures are handled gracefully)
            with patch.dict(os.environ, {}, clear=True):
                result = main()
                # Arena continues even when contestants fail (graceful degradation)
                # Returns 0 because it saves partial results
                assert result == 0

                # Verify output file was created with partial results
                with open(output_path) as f:
                    data = yaml.safe_load(f)
                assert data["mode"] == "arena"
                assert data["metadata"].get("partial")  # Marked as partial due to failures
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_arena_command_runs_multiple_evaluations(self, mock_eval_llm, sample_arena_spec_file):
        """Test that arena command evaluates all contestants."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "arena",
                "--arena-spec",
                sample_arena_spec_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            # Should have called evaluation for each contestant
            assert mock_eval_llm.call_count == 2

            # Verify output file exists
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_arena_output_structure(self, mock_eval_llm, sample_arena_spec_file):
        """Test that arena output has correct structure with all contestants."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "arena",
                "--arena-spec",
                sample_arena_spec_file,
                "--output-file",
                output_path,
            ]

            result = main()

            assert result == 0

            with open(output_path) as f:
                data = yaml.safe_load(f)

            # Check arena-specific structure
            assert data["mode"] == "arena"
            assert "contestants" in data
            assert "model_a" in data["contestants"]
            assert "model_b" in data["contestants"]

            # Each contestant should have results and summary
            for contestant_id in ["model_a", "model_b"]:
                contestant = data["contestants"][contestant_id]
                assert "results" in contestant
                assert "summary" in contestant
                assert "name" in contestant

            # Should have shared rubric and judge_panel
            assert "rubric" in data
            assert "judge_panel" in data

            # Should have rankings
            assert "rankings" in data
            assert len(data["rankings"]) == 2
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @patch("rubric_kit.main.evaluate_rubric_with_panel")
    @patch("rubric_kit.main.export_arena_pdf")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    def test_arena_with_report(self, mock_pdf, mock_eval_llm, sample_arena_spec_file):
        """Test arena subcommand with --report flag generates PDF."""
        import sys

        from rubric_kit.main import main

        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            output_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            sys.argv = [
                "rubric-kit",
                "arena",
                "--arena-spec",
                sample_arena_spec_file,
                "--output-file",
                output_path,
                "--report",
                pdf_path,
            ]

            result = main()

            assert result == 0
            assert mock_pdf.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
