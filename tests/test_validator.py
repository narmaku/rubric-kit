"""Tests for YAML validator."""

import pytest
import tempfile
import os
from pathlib import Path


def test_load_valid_rubric():
    """Test loading a valid rubric YAML file."""
    from rubric_kit.validator import load_rubric
    
    yaml_content = """
descriptors:
  - factual_correctness: Evaluates factual correctness.
    grading_type: binary
  - usefulness: Evaluates usefulness.
    grading_type: score
    scores:
      1: Useless
      2: Somewhat useful
      3: Very useful

criteria:
  - sys_info_factual_1:
      category: Output
      weight: 3
      dimension: factual_correctness
      criterion: The response must indicate that number of physical CPUs is 8.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        rubric = load_rubric(temp_path)
        assert len(rubric.descriptors) == 2
        assert len(rubric.criteria) == 1
        assert rubric.descriptors[0].name == "factual_correctness"
        assert rubric.criteria[0].name == "sys_info_factual_1"
    finally:
        os.unlink(temp_path)


def test_load_invalid_yaml_syntax():
    """Test loading YAML with invalid syntax."""
    from rubric_kit.validator import load_rubric, RubricValidationError
    
    yaml_content = """
descriptors:
  - factual_correctness: Test
    grading_type: binary
    invalid_indentation
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        with pytest.raises(RubricValidationError, match="Invalid YAML syntax"):
            load_rubric(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_invalid_rubric_structure():
    """Test loading YAML with invalid rubric structure."""
    from rubric_kit.validator import load_rubric, RubricValidationError
    
    yaml_content = """
descriptors:
  - factual_correctness: Test
    grading_type: score
    # Missing scores for score type

criteria:
  - test_1:
      weight: 3
      dimension: factual_correctness
      criterion: Test
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name
    
    try:
        with pytest.raises(RubricValidationError, match="Validation error"):
            load_rubric(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_nonexistent_file():
    """Test loading a non-existent file."""
    from rubric_kit.validator import load_rubric, RubricValidationError
    
    with pytest.raises(RubricValidationError, match="File not found"):
        load_rubric("/nonexistent/path/rubric.yaml")


def test_load_example_yaml():
    """Test loading the example.yaml file."""
    from rubric_kit.validator import load_rubric
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    example_path = project_root / "example.yaml"
    
    if example_path.exists():
        rubric = load_rubric(str(example_path))
        assert len(rubric.descriptors) == 3
        assert len(rubric.criteria) == 5
        
        # Check descriptors
        descriptor_names = {d.name for d in rubric.descriptors}
        assert "factual_correctness" in descriptor_names
        assert "usefulness" in descriptor_names
        assert "tool_usage" in descriptor_names
        
        # Check criteria
        criterion_names = {c.name for c in rubric.criteria}
        assert "sys_info_factual_1" in criterion_names
        assert "useful_1" in criterion_names
        assert "tool_call_1" in criterion_names


def test_parse_nested_dict_format():
    """Test parsing the nested dict format used in example.yaml."""
    from rubric_kit.validator import parse_nested_dict
    
    nested = {
        "item1": {"field1": "value1", "field2": "value2"},
        "item2": {"field1": "value3", "field2": "value4"}
    }
    
    result = parse_nested_dict(nested)
    
    assert len(result) == 2
    assert result[0]["name"] == "item1"
    assert result[0]["field1"] == "value1"
    assert result[1]["name"] == "item2"
    assert result[1]["field1"] == "value3"


def test_parse_tool_calls():
    """Test parsing tool_calls structure."""
    from rubric_kit.validator import parse_nested_dict
    
    tool_calls_data = {
        "respect_order": True,
        "required": [
            {"get_system_information": {"min_calls": 1, "max_calls": 1, "params": {}}}
        ],
        "optional": [],
        "prohibited": [
            {"get_weather": {"params": {}}}
        ]
    }
    
    # The parse_nested_dict should handle the nested format
    required_parsed = parse_nested_dict(tool_calls_data["required"])
    prohibited_parsed = parse_nested_dict(tool_calls_data["prohibited"])
    
    assert required_parsed[0]["name"] == "get_system_information"
    assert required_parsed[0]["min_calls"] == 1
    assert prohibited_parsed[0]["name"] == "get_weather"

