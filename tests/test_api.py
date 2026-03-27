"""Tests for the rubric_kit.api module — public Python API layer."""

import os
import tempfile

import pytest
import yaml

from rubric_kit.models.schema import (
    ConsensusConfig,
    Criterion,
    Dimension,
    ExecutionConfig,
    JudgeConfig,
    JudgePanelConfig,
    Rubric,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def binary_dimension():
    """Create a binary dimension for testing."""
    return Dimension(
        name="factual_correctness",
        description="Evaluates factual correctness of the response",
        grading_type="binary",
    )


@pytest.fixture
def score_dimension():
    """Create a score dimension for testing."""
    return Dimension(
        name="response_quality",
        description="Evaluates overall response quality",
        grading_type="score",
        scores={1: "Poor", 2: "Adequate", 3: "Excellent"},
        pass_above=2,
    )


@pytest.fixture
def simple_rubric(binary_dimension, score_dimension):
    """Create a simple rubric with binary and score criteria."""
    return Rubric(
        dimensions=[binary_dimension, score_dimension],
        criteria=[
            Criterion(
                name="fact_check",
                category="Output",
                weight=3,
                dimension="factual_correctness",
                criterion="The response contains factually correct information.",
            ),
            Criterion(
                name="quality_check",
                category="Output",
                weight="from_scores",
                dimension="response_quality",
                criterion="from_scores",
            ),
        ],
    )


@pytest.fixture
def single_judge_panel():
    """Create a single-judge panel configuration."""
    return JudgePanelConfig(
        judges=[JudgeConfig(name="judge_1", model="gpt-4", api_key="test-key")],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous"),
    )


@pytest.fixture
def sample_rubric_yaml(simple_rubric):
    """Create a temporary rubric YAML file."""
    rubric_dict = {
        "dimensions": [
            {
                "factual_correctness": {
                    "description": "Evaluates factual correctness of the response",
                    "grading_type": "binary",
                }
            },
            {
                "response_quality": {
                    "description": "Evaluates overall response quality",
                    "grading_type": "score",
                    "scores": {1: "Poor", 2: "Adequate", 3: "Excellent"},
                    "pass_above": 2,
                }
            },
        ],
        "criteria": [
            {
                "fact_check": {
                    "category": "Output",
                    "weight": 3,
                    "dimension": "factual_correctness",
                    "criterion": "The response contains factually correct information.",
                }
            },
            {
                "quality_check": {
                    "category": "Output",
                    "weight": "from_scores",
                    "dimension": "response_quality",
                    "criterion": "from_scores",
                }
            },
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(rubric_dict, f, sort_keys=False)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_panel_yaml(single_judge_panel):
    """Create a temporary judge panel YAML file."""
    panel_dict = {
        "judge_panel": {
            "judges": [{"name": "judge_1", "model": "gpt-4"}],
            "execution": {"mode": "sequential"},
            "consensus": {"mode": "unanimous"},
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(panel_dict, f, sort_keys=False)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_dimensions_yaml():
    """Create a temporary dimensions YAML file."""
    dims_dict = {
        "dimensions": [
            {
                "name": "accuracy",
                "description": "Tests factual accuracy",
                "grading_type": "binary",
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(dims_dict, f, sort_keys=False)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def raw_processor_results():
    """Raw results as returned by processor.evaluate_rubric()."""
    return [
        {
            "criterion_name": "fact_check",
            "criterion_text": "The response contains factually correct information.",
            "category": "Output",
            "dimension": "factual_correctness",
            "result": "pass",
            "score": 3,
            "max_score": 3,
            "reason": "The response is factually correct.",
            "consensus_reached": True,
            "consensus_count": 1,
        },
        {
            "criterion_name": "quality_check",
            "criterion_text": "Evaluates overall response quality",
            "category": "Output",
            "dimension": "response_quality",
            "result": "pass",
            "score": 3,
            "max_score": 3,
            "reason": "Excellent quality response.",
            "consensus_reached": True,
            "consensus_count": 1,
        },
    ]


# =============================================================================
# Result Model Tests
# =============================================================================


class TestCriterionResult:
    """Tests for CriterionResult model."""

    def test_from_processor_dict(self, raw_processor_results):
        """CriterionResult can be constructed from processor output dict."""
        from rubric_kit.api import CriterionResult

        result = CriterionResult(**raw_processor_results[0])

        assert result.criterion_name == "fact_check"
        assert result.result == "pass"
        assert result.score == 3
        assert result.max_score == 3
        assert result.reason == "The response is factually correct."
        assert result.consensus_reached is True

    def test_with_judge_votes(self):
        """CriterionResult handles optional judge_votes field."""
        from rubric_kit.api import CriterionResult

        votes = [
            {"judge": "judge_1", "passes": True, "reason": "Correct"},
            {"judge": "judge_2", "passes": True, "reason": "Accurate"},
        ]
        result = CriterionResult(
            criterion_name="test",
            dimension="dim",
            result="pass",
            score=3,
            max_score=3,
            judge_votes=votes,
        )
        assert result.judge_votes == votes

    def test_with_score_result(self):
        """CriterionResult accepts integer result for score-type criteria."""
        from rubric_kit.api import CriterionResult

        result = CriterionResult(
            criterion_name="test",
            dimension="dim",
            result=2,
            score=2,
            max_score=3,
        )
        assert result.result == 2

    def test_defaults(self):
        """CriterionResult has sensible defaults for optional fields."""
        from rubric_kit.api import CriterionResult

        result = CriterionResult(
            criterion_name="test",
            dimension="dim",
            result="pass",
            score=3,
            max_score=3,
        )
        assert result.criterion_text is None
        assert result.category is None
        assert result.reason == ""
        assert result.consensus_reached is True
        assert result.consensus_count == 1
        assert result.judge_votes is None
        assert result.tool_breakdown is None


class TestScoreSummary:
    """Tests for ScoreSummary model."""

    def test_fields(self):
        """ScoreSummary stores score fields correctly."""
        from rubric_kit.api import ScoreSummary

        summary = ScoreSummary(total_score=5, max_score=6, percentage=83.3)
        assert summary.total_score == 5
        assert summary.max_score == 6
        assert summary.percentage == 83.3


class TestEvaluationResult:
    """Tests for EvaluationResult model."""

    def test_construction(self, simple_rubric, single_judge_panel):
        """EvaluationResult can be constructed with all required fields."""
        from rubric_kit.api import CriterionResult, EvaluationResult, ScoreSummary

        criteria_results = [
            CriterionResult(
                criterion_name="fact_check",
                dimension="factual_correctness",
                result="pass",
                score=3,
                max_score=3,
            )
        ]
        result = EvaluationResult(
            criteria_results=criteria_results,
            summary=ScoreSummary(total_score=3, max_score=3, percentage=100.0),
            rubric=simple_rubric,
            panel_config=single_judge_panel,
            input_type="chat_session",
            input_source="test.txt",
        )

        assert len(result.criteria_results) == 1
        assert result.summary.percentage == 100.0
        assert result.input_type == "chat_session"
        assert result.metrics is None
        assert result.timestamp is not None

    def test_serialization(self, simple_rubric, single_judge_panel):
        """EvaluationResult can be serialized via model_dump()."""
        from rubric_kit.api import CriterionResult, EvaluationResult, ScoreSummary

        result = EvaluationResult(
            criteria_results=[
                CriterionResult(
                    criterion_name="test",
                    dimension="dim",
                    result="pass",
                    score=3,
                    max_score=3,
                )
            ],
            summary=ScoreSummary(total_score=3, max_score=3, percentage=100.0),
            rubric=simple_rubric,
            panel_config=single_judge_panel,
            input_type="qna",
            input_source="test.yaml",
        )
        dumped = result.model_dump()
        assert isinstance(dumped, dict)
        assert "criteria_results" in dumped
        assert "summary" in dumped
        assert dumped["input_type"] == "qna"


class TestGenerationResult:
    """Tests for GenerationResult model."""

    def test_construction(self, simple_rubric):
        """GenerationResult can be constructed with required fields."""
        from rubric_kit.api import GenerationResult

        result = GenerationResult(
            rubric=simple_rubric,
            model="gpt-4",
            input_type="qna",
            input_source="test.yaml",
        )
        assert result.rubric == simple_rubric
        assert result.model == "gpt-4"
        assert result.metrics is None
        assert result.timestamp is not None


class TestRefinementResult:
    """Tests for RefinementResult model."""

    def test_construction(self, simple_rubric):
        """RefinementResult can be constructed with required fields."""
        from rubric_kit.api import RefinementResult

        result = RefinementResult(
            rubric=simple_rubric,
            original_rubric=simple_rubric,
            model="gpt-4",
        )
        assert result.had_feedback is False
        assert result.had_context is False
        assert result.metrics is None


class TestDryRunResult:
    """Tests for DryRunResult model."""

    def test_construction(self):
        """DryRunResult can be constructed with cost estimates."""
        from rubric_kit.api import DryRunResult

        result = DryRunResult(
            total_calls=10,
            prompt_tokens=5000,
            cost_minimal=0.01,
            cost_conservative=0.05,
            cost_worst_case=0.50,
            model_estimates={
                "gpt-4": {
                    "calls": 10,
                    "prompt_tokens": 5000,
                    "cost_minimal": 0.01,
                    "cost_conservative": 0.05,
                    "cost_worst_case": 0.50,
                }
            },
        )
        assert result.total_calls == 10
        assert result.cost_minimal == 0.01


class TestExportResult:
    """Tests for ExportResult model."""

    def test_construction(self):
        """ExportResult can be constructed."""
        from rubric_kit.api import ExportResult

        result = ExportResult(format="pdf", output_path="/tmp/report.pdf")
        assert result.format == "pdf"
        assert result.success is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestResolveRubric:
    """Tests for _resolve_rubric helper."""

    def test_from_object(self, simple_rubric):
        """Pass a Rubric object, get same object back."""
        from rubric_kit.api import _resolve_rubric

        result = _resolve_rubric(simple_rubric)
        assert result is simple_rubric

    def test_from_path(self, sample_rubric_yaml):
        """Pass a YAML path, get a loaded Rubric."""
        from rubric_kit.api import _resolve_rubric

        result = _resolve_rubric(sample_rubric_yaml)
        assert isinstance(result, Rubric)
        assert len(result.dimensions) == 2
        assert len(result.criteria) == 2

    def test_file_not_found(self):
        """Pass invalid path, get RubricValidationError."""
        from rubric_kit.api import _resolve_rubric
        from rubric_kit.io.validator import RubricValidationError

        with pytest.raises(RubricValidationError, match="not found"):
            _resolve_rubric("/nonexistent/rubric.yaml")

    def test_from_pathlib(self, sample_rubric_yaml):
        """Accept pathlib.Path objects."""
        from pathlib import Path

        from rubric_kit.api import _resolve_rubric

        result = _resolve_rubric(Path(sample_rubric_yaml))
        assert isinstance(result, Rubric)


class TestResolvePanelConfig:
    """Tests for _resolve_panel_config helper."""

    def test_from_object(self, single_judge_panel):
        """Pass a JudgePanelConfig, get same object back."""
        from rubric_kit.api import _resolve_panel_config

        result = _resolve_panel_config(single_judge_panel)
        assert result is single_judge_panel

    def test_default_creation(self):
        """Pass None, get a default single-judge panel."""
        from rubric_kit.api import _resolve_panel_config

        result = _resolve_panel_config(None, model="gpt-4o", base_url=None)
        assert isinstance(result, JudgePanelConfig)
        assert len(result.judges) == 1
        assert result.judges[0].model == "gpt-4o"
        assert result.judges[0].name == "default"
        assert result.execution.mode == "sequential"
        assert result.consensus.mode == "unanimous"

    def test_from_path(self, sample_panel_yaml):
        """Pass a YAML path, get a loaded JudgePanelConfig."""
        from rubric_kit.api import _resolve_panel_config

        result = _resolve_panel_config(sample_panel_yaml)
        assert isinstance(result, JudgePanelConfig)
        assert len(result.judges) == 1

    def test_default_with_base_url(self):
        """Default panel passes through base_url."""
        from rubric_kit.api import _resolve_panel_config

        result = _resolve_panel_config(None, model="gpt-4", base_url="http://localhost:8080")
        assert result.judges[0].base_url == "http://localhost:8080"


class TestResolveInput:
    """Tests for _resolve_input helper."""

    def test_with_file(self):
        """Returns file path when input_file is provided."""
        from rubric_kit.api import _resolve_input

        file_path, content = _resolve_input("/path/to/file.txt", None)
        assert file_path == "/path/to/file.txt"
        assert content is None

    def test_with_content(self):
        """Returns content when input_content is provided."""
        from rubric_kit.api import _resolve_input

        file_path, content = _resolve_input(None, "Hello, world!")
        assert file_path is None
        assert content == "Hello, world!"

    def test_both_raises_error(self):
        """Raises ValueError when both are provided."""
        from rubric_kit.api import _resolve_input

        with pytest.raises(ValueError, match="Provide either input_file or input_content"):
            _resolve_input("/path/to/file.txt", "Hello, world!")

    def test_neither_raises_error(self):
        """Raises ValueError when neither is provided."""
        from rubric_kit.api import _resolve_input

        with pytest.raises(ValueError, match="Either input_file or input_content"):
            _resolve_input(None, None)

    def test_pathlib_path(self):
        """Accepts pathlib.Path for input_file."""
        from pathlib import Path

        from rubric_kit.api import _resolve_input

        file_path, content = _resolve_input(Path("/path/to/file.txt"), None)
        assert file_path == "/path/to/file.txt"


class TestResolveDimensions:
    """Tests for _resolve_dimensions helper."""

    def test_none(self):
        """Returns None when no dimensions provided."""
        from rubric_kit.api import _resolve_dimensions

        assert _resolve_dimensions(None) is None

    def test_from_list(self, binary_dimension):
        """Pass dimension list, get same list back."""
        from rubric_kit.api import _resolve_dimensions

        dims = [binary_dimension]
        result = _resolve_dimensions(dims)
        assert result is dims

    def test_from_path(self, sample_dimensions_yaml):
        """Pass YAML path, get parsed dimension list."""
        from rubric_kit.api import _resolve_dimensions

        result = _resolve_dimensions(sample_dimensions_yaml)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Dimension)
        assert result[0].name == "accuracy"


class TestBuildCriterionResults:
    """Tests for _build_criterion_results helper."""

    def test_converts_raw_dicts(self, raw_processor_results):
        """Converts list of raw dicts to CriterionResult objects."""
        from rubric_kit.api import CriterionResult, _build_criterion_results

        results = _build_criterion_results(raw_processor_results)
        assert len(results) == 2
        assert all(isinstance(r, CriterionResult) for r in results)
        assert results[0].criterion_name == "fact_check"
        assert results[1].criterion_name == "quality_check"

    def test_empty_list(self):
        """Handles empty list."""
        from rubric_kit.api import _build_criterion_results

        results = _build_criterion_results([])
        assert results == []


# =============================================================================
# evaluate() API Function Tests
# =============================================================================


class TestEvaluate:
    """Tests for the evaluate() public API function."""

    def _mock_evaluations(self):
        """Return mock evaluations as produced by llm_judge."""
        return {
            "fact_check": {
                "type": "binary",
                "passes": True,
                "reason": "Factually correct.",
                "consensus_reached": True,
                "consensus_count": 1,
            },
            "quality_check": {
                "type": "score",
                "score": 3,
                "reason": "Excellent quality.",
                "consensus_reached": True,
                "consensus_count": 1,
            },
        }

    def test_evaluate_chat_session_from_file(self, simple_rubric, single_judge_panel, tmp_path):
        """Evaluate a chat session file returns EvaluationResult."""
        from unittest.mock import patch

        from rubric_kit.api import EvaluationResult, evaluate

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("User: What is Python?\nAssistant: A programming language.")

        with patch(
            "rubric_kit.api.evaluate_rubric_with_panel",
            return_value=self._mock_evaluations(),
        ):
            result = evaluate(
                rubric=simple_rubric,
                input_file=str(chat_file),
                input_type="chat_session",
                panel_config=single_judge_panel,
                track_metrics=False,
            )

        assert isinstance(result, EvaluationResult)
        assert len(result.criteria_results) == 2
        assert result.summary.total_score == 6
        assert result.summary.max_score == 6
        assert result.summary.percentage == 100.0
        assert result.input_type == "chat_session"
        assert result.input_source == str(chat_file)
        assert result.rubric is simple_rubric
        assert result.panel_config is single_judge_panel
        assert result.metrics is None

    def test_evaluate_qna_from_file(self, simple_rubric, single_judge_panel, tmp_path):
        """Evaluate a Q&A file returns EvaluationResult."""
        from unittest.mock import patch

        from rubric_kit.api import evaluate

        qna_file = tmp_path / "qna.yaml"
        qna_file.write_text("question: What is Python?\nanswer: A programming language.")

        with patch(
            "rubric_kit.api.evaluate_rubric_with_panel_from_qa",
            return_value=self._mock_evaluations(),
        ):
            result = evaluate(
                rubric=simple_rubric,
                input_file=str(qna_file),
                input_type="qna",
                panel_config=single_judge_panel,
                track_metrics=False,
            )

        assert result.input_type == "qna"
        assert len(result.criteria_results) == 2

    def test_evaluate_with_rubric_path(self, sample_rubric_yaml, single_judge_panel, tmp_path):
        """Evaluate accepts a rubric file path."""
        from unittest.mock import patch

        from rubric_kit.api import evaluate

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("User: Hello\nAssistant: Hi!")

        with patch(
            "rubric_kit.api.evaluate_rubric_with_panel",
            return_value=self._mock_evaluations(),
        ):
            result = evaluate(
                rubric=sample_rubric_yaml,
                input_file=str(chat_file),
                input_type="chat_session",
                panel_config=single_judge_panel,
                track_metrics=False,
            )

        assert len(result.rubric.dimensions) == 2

    def test_evaluate_with_input_content(self, simple_rubric, single_judge_panel):
        """Evaluate accepts inline content string instead of file."""
        from unittest.mock import patch

        from rubric_kit.api import evaluate

        with patch(
            "rubric_kit.api.evaluate_rubric_with_panel",
            return_value=self._mock_evaluations(),
        ):
            result = evaluate(
                rubric=simple_rubric,
                input_content="User: Hello\nAssistant: Hi!",
                input_type="chat_session",
                panel_config=single_judge_panel,
                track_metrics=False,
            )

        assert result.input_source == "<in-memory>"
        assert len(result.criteria_results) == 2

    def test_evaluate_with_metrics(self, simple_rubric, single_judge_panel, tmp_path):
        """Evaluate tracks metrics when track_metrics=True."""
        from unittest.mock import patch

        from rubric_kit.api import evaluate

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("User: Hello\nAssistant: Hi!")

        with patch(
            "rubric_kit.api.evaluate_rubric_with_panel",
            return_value=self._mock_evaluations(),
        ):
            result = evaluate(
                rubric=simple_rubric,
                input_file=str(chat_file),
                input_type="chat_session",
                panel_config=single_judge_panel,
                track_metrics=True,
            )

        # Metrics aggregator was created but no real LLM calls were made
        # (mocked), so summary will have 0 calls
        assert result.metrics is not None

    def test_evaluate_default_panel(self, simple_rubric, tmp_path):
        """Evaluate creates default single-judge panel when none provided."""
        from unittest.mock import patch

        from rubric_kit.api import evaluate

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("User: Hello\nAssistant: Hi!")

        with patch(
            "rubric_kit.api.evaluate_rubric_with_panel",
            return_value=self._mock_evaluations(),
        ):
            result = evaluate(
                rubric=simple_rubric,
                input_file=str(chat_file),
                input_type="chat_session",
                model="gpt-4o",
                track_metrics=False,
            )

        assert len(result.panel_config.judges) == 1
        assert result.panel_config.judges[0].model == "gpt-4o"

    def test_evaluate_missing_input_raises(self, simple_rubric, single_judge_panel):
        """Evaluate raises ValueError when no input is provided."""
        from rubric_kit.api import evaluate

        with pytest.raises(ValueError, match="Either input_file or input_content"):
            evaluate(
                rubric=simple_rubric,
                panel_config=single_judge_panel,
            )

    def test_evaluate_both_inputs_raises(self, simple_rubric, single_judge_panel, tmp_path):
        """Evaluate raises ValueError when both inputs are provided."""
        from rubric_kit.api import evaluate

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("content")

        with pytest.raises(ValueError, match="Provide either input_file or input_content"):
            evaluate(
                rubric=simple_rubric,
                input_file=str(chat_file),
                input_content="content",
                panel_config=single_judge_panel,
            )


# =============================================================================
# generate() API Function Tests
# =============================================================================


class TestGenerate:
    """Tests for the generate() public API function."""

    def test_generate_from_qna_file(self, simple_rubric, tmp_path):
        """Generate a rubric from a Q&A file."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import GenerationResult, generate

        qna_file = tmp_path / "qna.yaml"
        qna_file.write_text("question: What is Python?\nanswer: A programming language.")

        mock_generator = MagicMock()
        mock_generator.generate_rubric.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = generate(
                input_file=str(qna_file),
                input_type="qna",
                model="gpt-4",
                track_metrics=False,
            )

        assert isinstance(result, GenerationResult)
        assert result.rubric is simple_rubric
        assert result.model == "gpt-4"
        assert result.input_type == "qna"
        assert result.input_source == str(qna_file)
        mock_generator.generate_rubric.assert_called_once()

    def test_generate_from_chat_file(self, simple_rubric, tmp_path):
        """Generate a rubric from a chat session file."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import generate

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("User: Hello\nAssistant: Hi!")

        mock_generator = MagicMock()
        mock_generator.generate_rubric_from_chat.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = generate(
                input_file=str(chat_file),
                input_type="chat_session",
                model="gpt-4o",
                track_metrics=False,
            )

        assert result.rubric is simple_rubric
        assert result.input_type == "chat_session"
        mock_generator.generate_rubric_from_chat.assert_called_once()

    def test_generate_from_content_string(self, simple_rubric):
        """Generate a rubric from inline content."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import generate

        mock_generator = MagicMock()
        mock_generator.generate_rubric.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = generate(
                input_content="question: What is Python?\nanswer: A language.",
                input_type="qna",
                model="gpt-4",
                track_metrics=False,
            )

        assert result.input_source == "<in-memory>"
        assert result.rubric is simple_rubric

    def test_generate_with_dimensions(self, simple_rubric, binary_dimension, tmp_path):
        """Generate passes pre-defined dimensions to generator."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import generate

        qna_file = tmp_path / "qna.yaml"
        qna_file.write_text("question: Q\nanswer: A")

        mock_generator = MagicMock()
        mock_generator.generate_rubric.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            generate(
                input_file=str(qna_file),
                input_type="qna",
                model="gpt-4",
                dimensions=[binary_dimension],
                track_metrics=False,
            )

        call_kwargs = mock_generator.generate_rubric.call_args
        assert call_kwargs.kwargs.get("dimensions") == [binary_dimension]

    def test_generate_with_guidelines(self, simple_rubric, tmp_path):
        """Generate passes guidelines to generator."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import generate

        qna_file = tmp_path / "qna.yaml"
        qna_file.write_text("question: Q\nanswer: A")

        mock_generator = MagicMock()
        mock_generator.generate_rubric.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            generate(
                input_file=str(qna_file),
                input_type="qna",
                model="gpt-4",
                guidelines="Focus on accuracy",
                track_metrics=False,
            )

        call_kwargs = mock_generator.generate_rubric.call_args
        assert call_kwargs.kwargs.get("guidelines") == "Focus on accuracy"

    def test_generate_missing_input_raises(self):
        """Generate raises ValueError when no input is provided."""
        from rubric_kit.api import generate

        with pytest.raises(ValueError, match="Either input_file or input_content"):
            generate(model="gpt-4")


# =============================================================================
# refine() API Function Tests
# =============================================================================


class TestRefine:
    """Tests for the refine() public API function."""

    def test_refine_basic(self, simple_rubric):
        """Refine a rubric without context or feedback."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import RefinementResult, refine

        refined_rubric = simple_rubric  # Use same rubric for simplicity
        mock_generator = MagicMock()
        mock_generator.refine_rubric.return_value = refined_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = refine(
                rubric=simple_rubric,
                model="gpt-4",
                track_metrics=False,
            )

        assert isinstance(result, RefinementResult)
        assert result.rubric is refined_rubric
        assert result.original_rubric is simple_rubric
        assert result.model == "gpt-4"
        assert result.had_feedback is False
        assert result.had_context is False
        mock_generator.refine_rubric.assert_called_once()

    def test_refine_with_feedback(self, simple_rubric):
        """Refine with feedback text."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import refine

        mock_generator = MagicMock()
        mock_generator.refine_rubric.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = refine(
                rubric=simple_rubric,
                model="gpt-4",
                feedback="Add more criteria for tool usage",
                track_metrics=False,
            )

        assert result.had_feedback is True
        call_kwargs = mock_generator.refine_rubric.call_args
        assert call_kwargs.kwargs.get("feedback") == "Add more criteria for tool usage"

    def test_refine_with_qa_context(self, simple_rubric, tmp_path):
        """Refine with Q&A context input."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import refine

        qna_file = tmp_path / "qna.yaml"
        qna_file.write_text("question: Q\nanswer: A")

        mock_generator = MagicMock()
        mock_generator.refine_rubric_with_qa.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = refine(
                rubric=simple_rubric,
                model="gpt-4",
                input_file=str(qna_file),
                input_type="qna",
                track_metrics=False,
            )

        assert result.had_context is True
        mock_generator.refine_rubric_with_qa.assert_called_once()

    def test_refine_with_chat_context(self, simple_rubric, tmp_path):
        """Refine with chat session context input."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import refine

        chat_file = tmp_path / "chat.txt"
        chat_file.write_text("User: Hello\nAssistant: Hi!")

        mock_generator = MagicMock()
        mock_generator.refine_rubric_with_chat.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = refine(
                rubric=simple_rubric,
                model="gpt-4",
                input_file=str(chat_file),
                input_type="chat_session",
                track_metrics=False,
            )

        assert result.had_context is True
        mock_generator.refine_rubric_with_chat.assert_called_once()

    def test_refine_from_rubric_path(self, sample_rubric_yaml, simple_rubric):
        """Refine accepts a rubric file path."""
        from unittest.mock import MagicMock, patch

        from rubric_kit.api import refine

        mock_generator = MagicMock()
        mock_generator.refine_rubric.return_value = simple_rubric

        with patch("rubric_kit.api.RubricGenerator", return_value=mock_generator):
            result = refine(
                rubric=sample_rubric_yaml,
                model="gpt-4",
                track_metrics=False,
            )

        assert result.rubric is simple_rubric
        # Original rubric was loaded from file
        assert isinstance(result.original_rubric, Rubric)


# =============================================================================
# export() API Function Tests
# =============================================================================


class TestExport:
    """Tests for the export() public API function."""

    def test_export_to_csv(self, tmp_path):
        """Export evaluation results to CSV."""
        from unittest.mock import patch

        from rubric_kit.api import ExportResult, export

        input_yaml = tmp_path / "results.yaml"
        input_yaml.write_text("results: []")
        output_csv = tmp_path / "results.csv"

        with patch("rubric_kit.api.convert_yaml_to_csv") as mock_csv:
            result = export(
                input_file=str(input_yaml),
                output_file=str(output_csv),
                format="csv",
            )

        assert isinstance(result, ExportResult)
        assert result.format == "csv"
        assert result.output_path == str(output_csv)
        assert result.success is True
        mock_csv.assert_called_once_with(str(input_yaml), str(output_csv))

    def test_export_to_json(self, tmp_path):
        """Export evaluation results to JSON."""
        from unittest.mock import patch

        from rubric_kit.api import export

        input_yaml = tmp_path / "results.yaml"
        input_yaml.write_text("results: []")
        output_json = tmp_path / "results.json"

        with patch("rubric_kit.api.convert_yaml_to_json") as mock_json:
            result = export(
                input_file=str(input_yaml),
                output_file=str(output_json),
                format="json",
            )

        assert result.format == "json"
        mock_json.assert_called_once_with(str(input_yaml), str(output_json))

    def test_export_to_pdf(self, tmp_path):
        """Export evaluation results to PDF."""
        from unittest.mock import patch

        from rubric_kit.api import export

        input_yaml = tmp_path / "results.yaml"
        input_yaml.write_text("results: []")
        output_pdf = tmp_path / "report.pdf"

        with patch("rubric_kit.api.export_evaluation_pdf") as mock_pdf:
            result = export(
                input_file=str(input_yaml),
                output_file=str(output_pdf),
                format="pdf",
            )

        assert result.format == "pdf"
        mock_pdf.assert_called_once_with(str(input_yaml), str(output_pdf))


# =============================================================================
# dry_run_evaluate() API Function Tests
# =============================================================================


class TestDryRunEvaluate:
    """Tests for the dry_run_evaluate() public API function."""

    def test_basic(self, simple_rubric):
        """Dry run returns cost estimates without LLM calls."""
        from rubric_kit.api import DryRunResult, dry_run_evaluate

        result = dry_run_evaluate(
            rubric=simple_rubric,
            model="gpt-4",
        )

        assert isinstance(result, DryRunResult)
        assert result.total_calls > 0
        assert result.prompt_tokens > 0
        assert "gpt-4" in result.model_estimates
        assert result.cost_minimal >= 0
        assert result.cost_conservative >= result.cost_minimal
        assert result.cost_worst_case >= result.cost_conservative

    def test_with_panel_config(self, simple_rubric, single_judge_panel):
        """Dry run uses models from panel config."""
        from rubric_kit.api import dry_run_evaluate

        result = dry_run_evaluate(
            rubric=simple_rubric,
            panel_config=single_judge_panel,
        )

        assert "gpt-4" in result.model_estimates

    def test_with_rubric_path(self, sample_rubric_yaml):
        """Dry run accepts rubric file path."""
        from rubric_kit.api import dry_run_evaluate

        result = dry_run_evaluate(
            rubric=sample_rubric_yaml,
            model="gpt-4",
        )

        assert result.total_calls > 0
