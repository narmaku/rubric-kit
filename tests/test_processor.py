"""Tests for score processor."""

import pytest
from rubric_kit.schema import Rubric, Descriptor, Criterion, ToolCalls, ToolSpec


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
            scores={
                1: "Not useful",
                2: "Somewhat useful",
                3: "Very useful"
            }
        )
    ]
    
    criteria = [
        Criterion(
            name="fact_1",
            category="Output",
            weight=3,
            dimension="factual_correctness",
            criterion="Check fact 1"
        ),
        Criterion(
            name="fact_2",
            category="Output",
            weight=2,
            dimension="factual_correctness",
            criterion="Check fact 2"
        ),
        Criterion(
            name="useful_1",
            category="Output",
            weight="from_scores",
            dimension="usefulness",
            criterion="from_scores"
        )
    ]
    
    return Rubric(descriptors=descriptors, criteria=criteria)


def test_evaluate_binary_criterion_pass():
    """Test evaluating a binary criterion that passes."""
    from rubric_kit.processor import evaluate_binary_criterion
    
    criterion = Criterion(
        name="test",
        weight=3,
        dimension="factual_correctness",
        criterion="Test"
    )
    
    result = evaluate_binary_criterion(criterion, passes=True)
    assert result["criterion_name"] == "test"
    assert result["result"] == "pass"
    assert result["score"] == 3
    assert result["max_score"] == 3


def test_evaluate_binary_criterion_fail():
    """Test evaluating a binary criterion that fails."""
    from rubric_kit.processor import evaluate_binary_criterion
    
    criterion = Criterion(
        name="test",
        weight=3,
        dimension="factual_correctness",
        criterion="Test"
    )
    
    result = evaluate_binary_criterion(criterion, passes=False)
    assert result["result"] == "fail"
    assert result["score"] == 0
    assert result["max_score"] == 3


def test_evaluate_score_criterion():
    """Test evaluating a score-based criterion."""
    from rubric_kit.processor import evaluate_score_criterion
    
    criterion = Criterion(
        name="test",
        weight="from_scores",
        dimension="usefulness",
        criterion="from_scores"
    )
    
    descriptor = Descriptor(
        name="usefulness",
        description="Test",
        grading_type="score",
        scores={1: "Bad", 2: "Good", 3: "Great"}
    )
    
    result = evaluate_score_criterion(criterion, descriptor, score=2)
    assert result["criterion_name"] == "test"
    assert result["result"] == 2
    assert result["score"] == 2
    assert result["max_score"] == 3
    assert result["score_description"] == "Good"


def test_evaluate_rubric(simple_rubric):
    """Test evaluating a complete rubric."""
    from rubric_kit.processor import evaluate_rubric
    
    # Provide evaluation inputs
    evaluations = {
        "fact_1": {"type": "binary", "passes": True},
        "fact_2": {"type": "binary", "passes": False},
        "useful_1": {"type": "score", "score": 3}
    }
    
    results = evaluate_rubric(simple_rubric, evaluations)
    
    assert len(results) == 3
    assert results[0]["score"] == 3  # fact_1 passes with weight 3
    assert results[1]["score"] == 0  # fact_2 fails
    assert results[2]["score"] == 3  # useful_1 scored 3/3


def test_calculate_total_score():
    """Test calculating total score from results."""
    from rubric_kit.processor import calculate_total_score
    
    results = [
        {"criterion_name": "test1", "score": 3, "max_score": 3},
        {"criterion_name": "test2", "score": 2, "max_score": 3},
        {"criterion_name": "test3", "score": 0, "max_score": 2}
    ]
    
    total, max_total = calculate_total_score(results)
    assert total == 5
    assert max_total == 8


def test_calculate_percentage_score():
    """Test calculating percentage score."""
    from rubric_kit.processor import calculate_percentage_score
    
    results = [
        {"criterion_name": "test1", "score": 3, "max_score": 3},
        {"criterion_name": "test2", "score": 2, "max_score": 3}
    ]
    
    percentage = calculate_percentage_score(results)
    assert percentage == pytest.approx(83.33, rel=0.01)


def test_weight_zero_criterion():
    """Test that weight 0 criteria are handled correctly."""
    from rubric_kit.processor import evaluate_binary_criterion
    
    criterion = Criterion(
        name="test",
        weight=0,  # Disabled criterion
        dimension="factual_correctness",
        criterion="Test"
    )
    
    result = evaluate_binary_criterion(criterion, passes=True)
    assert result["score"] == 0
    assert result["max_score"] == 0

