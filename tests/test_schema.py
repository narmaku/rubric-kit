"""Tests for schema validation."""

import pytest
from pydantic import ValidationError


def test_binary_descriptor():
    """Test binary grading type descriptor."""
    from rubric_kit.schema import Descriptor
    
    descriptor = Descriptor(
        name="factual_correctness",
        description="Evaluates that the information of the final answer is factually correct.",
        grading_type="binary"
    )
    
    assert descriptor.name == "factual_correctness"
    assert descriptor.grading_type == "binary"
    assert descriptor.scores is None


def test_score_descriptor():
    """Test score grading type descriptor with scores."""
    from rubric_kit.schema import Descriptor
    
    descriptor = Descriptor(
        name="usefulness",
        description="Evaluates how useful is the final response.",
        grading_type="score",
        scores={
            1: "The response is completely useless.",
            2: "The response is useful but incomplete.",
            3: "The response is useful and complete."
        }
    )
    
    assert descriptor.name == "usefulness"
    assert descriptor.grading_type == "score"
    assert len(descriptor.scores) == 3
    assert descriptor.scores[1] == "The response is completely useless."


def test_score_descriptor_requires_scores():
    """Test that score type requires scores dict."""
    from rubric_kit.schema import Descriptor
    
    with pytest.raises(ValidationError):
        Descriptor(
            name="usefulness",
            description="Test",
            grading_type="score"
            # Missing scores
        )


def test_invalid_grading_type():
    """Test that invalid grading type raises error."""
    from rubric_kit.schema import Descriptor
    
    with pytest.raises(ValidationError):
        Descriptor(
            name="test",
            description="Test",
            grading_type="invalid"
        )


def test_output_criterion():
    """Test Output category criterion."""
    from rubric_kit.schema import Criterion
    
    criterion = Criterion(
        name="sys_info_factual_1",
        category="Output",
        weight=3,
        dimension="factual_correctness",
        criterion="The response must indicate that number of physical CPUs is 8."
    )
    
    assert criterion.name == "sys_info_factual_1"
    assert criterion.category == "Output"
    assert criterion.weight == 3
    assert criterion.dimension == "factual_correctness"
    assert criterion.tool_calls is None


def test_criterion_with_from_scores_weight():
    """Test criterion with from_scores weight."""
    from rubric_kit.schema import Criterion
    
    criterion = Criterion(
        name="useful_1",
        category="Output",
        weight="from_scores",
        dimension="usefulness",
        criterion="from_scores"
    )
    
    assert criterion.weight == "from_scores"
    assert criterion.criterion == "from_scores"


def test_criterion_weight_range():
    """Test that weight must be in range 0-3."""
    from rubric_kit.schema import Criterion
    
    with pytest.raises(ValidationError):
        Criterion(
            name="test",
            category="Output",
            weight=5,  # Invalid: > 3
            dimension="test",
            criterion="test"
        )


def test_tool_criterion():
    """Test Tools category criterion with tool_calls."""
    from rubric_kit.schema import Criterion, ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=True,
        required=[
            ToolSpec(
                name="get_system_information",
                min_calls=1,
                max_calls=1,
                params={}
            )
        ],
        optional=[],
        prohibited=[
            ToolSpec(
                name="get_weather",
                params={}
            )
        ]
    )
    
    criterion = Criterion(
        name="tool_call_1",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    assert criterion.category == "Tools"
    assert len(criterion.tool_calls.required) == 1
    assert criterion.tool_calls.required[0].name == "get_system_information"
    assert len(criterion.tool_calls.prohibited) == 1


def test_rubric_complete():
    """Test complete rubric structure."""
    from rubric_kit.schema import Rubric, Descriptor, Criterion
    
    descriptors = [
        Descriptor(
            name="factual_correctness",
            description="Test",
            grading_type="binary"
        )
    ]
    
    criteria = [
        Criterion(
            name="test_1",
            category="Output",
            weight=3,
            dimension="factual_correctness",
            criterion="Test criterion"
        )
    ]
    
    rubric = Rubric(descriptors=descriptors, criteria=criteria)
    
    assert len(rubric.descriptors) == 1
    assert len(rubric.criteria) == 1
    assert rubric.descriptors[0].name == "factual_correctness"


def test_rubric_validates_dimension_references():
    """Test that criteria must reference valid descriptors."""
    from rubric_kit.schema import Rubric, Descriptor, Criterion
    
    descriptors = [
        Descriptor(
            name="factual_correctness",
            description="Test",
            grading_type="binary"
        )
    ]
    
    criteria = [
        Criterion(
            name="test_1",
            category="Output",
            weight=3,
            dimension="nonexistent_dimension",  # Invalid reference
            criterion="Test"
        )
    ]
    
    with pytest.raises(ValidationError, match="references non-existent dimension"):
        Rubric(descriptors=descriptors, criteria=criteria)

