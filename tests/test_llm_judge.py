"""Tests for LLM judge functionality."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from rubric_kit.schema import Rubric, Descriptor, Criterion


@pytest.fixture
def sample_chat_session_file():
    """Create a sample chat session file."""
    content = """User: What are the system specifications?
Assistant: The system has 8 physical CPUs and 64 GB of RAM.

Tool calls:
- get_system_information() -> {"cpus": 8, "ram_gb": 64}
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def simple_binary_criterion():
    """Create a simple binary criterion."""
    return Criterion(
        name="test_fact",
        category="Output",
        weight=3,
        dimension="factual_correctness",
        criterion="The response must indicate that number of physical CPUs is 8."
    )


@pytest.fixture
def simple_score_criterion():
    """Create a simple score criterion."""
    return Criterion(
        name="test_useful",
        category="Output",
        weight="from_scores",
        dimension="usefulness",
        criterion="from_scores"
    )


@pytest.fixture
def simple_rubric():
    """Create a simple rubric for testing."""
    descriptors = [
        Descriptor(
            name="factual_correctness",
            description="Evaluates factual correctness",
            grading_type="binary"
        ),
        Descriptor(
            name="usefulness",
            description="Evaluates usefulness",
            grading_type="score",
            scores={1: "Not useful", 2: "Somewhat useful", 3: "Very useful"}
        )
    ]
    
    criteria = [
        Criterion(
            name="test_fact",
            category="Output",
            weight=3,
            dimension="factual_correctness",
            criterion="The response must indicate that number of physical CPUs is 8."
        ),
        Criterion(
            name="test_useful",
            category="Output",
            weight="from_scores",
            dimension="usefulness",
            criterion="from_scores"
        )
    ]
    
    return Rubric(descriptors=descriptors, criteria=criteria)


def test_read_chat_session(sample_chat_session_file):
    """Test reading chat session file."""
    from rubric_kit.llm_judge import read_chat_session
    
    content = read_chat_session(sample_chat_session_file)
    assert "What are the system specifications?" in content
    assert "8 physical CPUs" in content


def test_create_binary_criterion_prompt(simple_binary_criterion):
    """Test creating prompt for binary criterion evaluation."""
    from rubric_kit.llm_judge import create_criterion_prompt
    
    chat_content = "Assistant: The system has 8 CPUs."
    prompt = create_criterion_prompt(simple_binary_criterion, chat_content, "binary")
    
    assert "factual_correctness" in prompt
    assert "PASS" in prompt
    assert "FAIL" in prompt
    assert "8" in chat_content


def test_create_score_criterion_prompt(simple_score_criterion):
    """Test creating prompt for score criterion evaluation."""
    from rubric_kit.llm_judge import create_criterion_prompt
    
    descriptor = Descriptor(
        name="usefulness",
        description="Test",
        grading_type="score",
        scores={1: "Not useful", 2: "Somewhat useful", 3: "Very useful"}
    )
    
    chat_content = "Assistant: Here's the information."
    prompt = create_criterion_prompt(simple_score_criterion, chat_content, "score", descriptor)
    
    assert "usefulness" in prompt
    assert "1" in prompt
    assert "3" in prompt


@patch('rubric_kit.llm_judge.OpenAI')
def test_evaluate_criterion_with_llm_binary(mock_openai, simple_binary_criterion):
    """Test LLM evaluation of binary criterion."""
    from rubric_kit.llm_judge import evaluate_criterion_with_llm
    
    # Mock OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="PASS"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    chat_content = "Assistant: The system has 8 CPUs."
    result = evaluate_criterion_with_llm(
        simple_binary_criterion,
        chat_content,
        "binary",
        api_key="test_key"
    )
    
    assert result["type"] == "binary"
    assert result["passes"] is True


@patch('rubric_kit.llm_judge.OpenAI')
def test_evaluate_criterion_with_llm_score(mock_openai, simple_score_criterion):
    """Test LLM evaluation of score criterion."""
    from rubric_kit.llm_judge import evaluate_criterion_with_llm
    
    # Mock OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="3"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    descriptor = Descriptor(
        name="usefulness",
        description="Test",
        grading_type="score",
        scores={1: "Bad", 2: "Good", 3: "Great"}
    )
    
    chat_content = "Assistant: Complete response."
    result = evaluate_criterion_with_llm(
        simple_score_criterion,
        chat_content,
        "score",
        api_key="test_key",
        descriptor=descriptor
    )
    
    assert result["type"] == "score"
    assert result["score"] == 3


@patch('rubric_kit.llm_judge.OpenAI')
def test_evaluate_rubric_with_llm(mock_openai, simple_rubric, sample_chat_session_file):
    """Test evaluating entire rubric with LLM."""
    from rubric_kit.llm_judge import evaluate_rubric_with_llm
    
    # Mock OpenAI responses
    mock_client = Mock()
    mock_responses = [
        Mock(choices=[Mock(message=Mock(content="PASS"))]),
        Mock(choices=[Mock(message=Mock(content="3"))])
    ]
    mock_client.chat.completions.create.side_effect = mock_responses
    mock_openai.return_value = mock_client
    
    evaluations = evaluate_rubric_with_llm(
        simple_rubric,
        sample_chat_session_file,
        api_key="test_key"
    )
    
    assert len(evaluations) == 2
    assert "test_fact" in evaluations
    assert "test_useful" in evaluations
    assert evaluations["test_fact"]["type"] == "binary"
    assert evaluations["test_useful"]["type"] == "score"


def test_parse_llm_response_binary():
    """Test parsing binary LLM responses."""
    from rubric_kit.llm_judge import parse_binary_response
    
    assert parse_binary_response("PASS") is True
    assert parse_binary_response("pass") is True
    assert parse_binary_response("The answer is PASS") is True
    assert parse_binary_response("FAIL") is False
    assert parse_binary_response("fail") is False


def test_parse_llm_response_score():
    """Test parsing score LLM responses."""
    from rubric_kit.llm_judge import parse_score_response
    
    assert parse_score_response("3") == 3
    assert parse_score_response("The score is 2") == 2
    assert parse_score_response("1") == 1
    assert parse_score_response("Score: 3/3") == 3


def test_invalid_api_configuration():
    """Test handling of invalid API configuration."""
    from rubric_kit.llm_judge import evaluate_criterion_with_llm
    
    criterion = Criterion(
        name="test",
        weight=1,
        dimension="test",
        criterion="Test"
    )
    
    with pytest.raises(ValueError, match="API key"):
        evaluate_criterion_with_llm(criterion, "content", "binary", api_key=None)
