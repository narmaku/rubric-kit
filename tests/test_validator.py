"""Tests for YAML validator."""

import os
import tempfile
from pathlib import Path

import pytest


def test_load_valid_rubric():
    """Test loading a valid rubric YAML file."""
    from rubric_kit.validator import load_rubric

    yaml_content = """
dimensions:
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        rubric = load_rubric(temp_path)
        assert len(rubric.dimensions) == 2
        assert len(rubric.criteria) == 1
        assert rubric.dimensions[0].name == "factual_correctness"
        assert rubric.criteria[0].name == "sys_info_factual_1"
    finally:
        os.unlink(temp_path)


def test_load_invalid_yaml_syntax():
    """Test loading YAML with invalid syntax."""
    from rubric_kit.validator import RubricValidationError, load_rubric

    yaml_content = """
dimensions:
  - factual_correctness: Test
    grading_type: binary
    invalid_indentation
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with pytest.raises(RubricValidationError, match="Invalid YAML syntax"):
            load_rubric(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_invalid_rubric_structure():
    """Test loading YAML with invalid rubric structure."""
    from rubric_kit.validator import RubricValidationError, load_rubric

    yaml_content = """
dimensions:
  - factual_correctness: Test
    grading_type: score
    # Missing scores for score type

criteria:
  - test_1:
      weight: 3
      dimension: factual_correctness
      criterion: Test
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with pytest.raises(RubricValidationError, match="Validation error"):
            load_rubric(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_nonexistent_file():
    """Test loading a non-existent file."""
    from rubric_kit.validator import RubricValidationError, load_rubric

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
        assert len(rubric.dimensions) == 3
        assert (
            len(rubric.criteria) == 7
        )  # sys_info_distro, sys_info_cpu, sys_info_memory, extra_info_network, extra_info_disks, useful_1, tool_call_1

        # Check descriptors
        descriptor_names = {d.name for d in rubric.dimensions}
        assert "factual_correctness" in descriptor_names
        assert "usefulness" in descriptor_names
        assert "tool_usage" in descriptor_names

        # Check criteria
        criterion_names = {c.name for c in rubric.criteria}
        assert "sys_info_distro" in criterion_names
        assert "sys_info_cpu" in criterion_names
        assert "sys_info_memory" in criterion_names
        assert "extra_info_network" in criterion_names
        assert "extra_info_disks" in criterion_names
        assert "useful_1" in criterion_names
        assert "tool_call_1" in criterion_names


def test_parse_nested_dict_format():
    """Test parsing the nested dict format used in example.yaml."""
    from rubric_kit.validator import parse_nested_dict

    nested = {
        "item1": {"field1": "value1", "field2": "value2"},
        "item2": {"field1": "value3", "field2": "value4"},
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
        "required": [{"get_system_information": {"min_calls": 1, "max_calls": 1, "params": {}}}],
        "optional": [],
        "prohibited": [{"get_weather": {"params": {}}}],
    }

    # The parse_nested_dict should handle the nested format
    required_parsed = parse_nested_dict(tool_calls_data["required"])
    prohibited_parsed = parse_nested_dict(tool_calls_data["prohibited"])

    assert required_parsed[0]["name"] == "get_system_information"
    assert required_parsed[0]["min_calls"] == 1
    assert prohibited_parsed[0]["name"] == "get_weather"


# ============================================================================
# Judge Panel Configuration Tests
# ============================================================================


def test_load_judge_panel_config_basic():
    """Test loading basic judge panel configuration."""
    from rubric_kit.validator import load_judge_panel_config

    yaml_content = """
judge_panel:
  judges:
    - name: judge_1
      model: gpt-4
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        panel_config = load_judge_panel_config(temp_path)
        assert len(panel_config.judges) == 1
        assert panel_config.judges[0].name == "judge_1"
        assert panel_config.judges[0].model == "gpt-4"
        assert panel_config.execution.mode == "sequential"  # Default
        assert panel_config.consensus.mode == "unanimous"  # Default
    finally:
        os.unlink(temp_path)


def test_load_judge_panel_config_multiple_judges():
    """Test loading judge panel with multiple judges."""
    from rubric_kit.validator import load_judge_panel_config

    yaml_content = """
judge_panel:
  judges:
    - name: primary
      model: gpt-4
    - name: secondary
      model: gpt-4-turbo
      base_url: https://api.example.com/v1
    - name: tertiary
      model: claude-3-5-sonnet
  execution:
    mode: parallel
    timeout: 60
  consensus:
    mode: quorum
    threshold: 2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        panel_config = load_judge_panel_config(temp_path)
        assert len(panel_config.judges) == 3
        assert panel_config.judges[0].name == "primary"
        assert panel_config.judges[1].base_url == "https://api.example.com/v1"
        assert panel_config.execution.mode == "parallel"
        assert panel_config.execution.timeout == 60
        assert panel_config.consensus.mode == "quorum"
        assert panel_config.consensus.threshold == 2
    finally:
        os.unlink(temp_path)


def test_load_judge_panel_config_with_defaults():
    """Test loading judge panel uses defaults for missing fields."""
    from rubric_kit.validator import load_judge_panel_config

    yaml_content = """
judge_panel:
  judges:
    - name: judge_1
      model: gpt-4
  consensus:
    mode: majority
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        panel_config = load_judge_panel_config(temp_path)
        # execution not specified, should use defaults
        assert panel_config.execution.mode == "sequential"
        assert panel_config.execution.batch_size == 2
        # consensus mode specified, threshold auto-calculated for majority
        assert panel_config.consensus.mode == "majority"
        assert panel_config.consensus.threshold == 1  # (1 // 2) + 1 = 1
    finally:
        os.unlink(temp_path)


def test_load_judge_panel_config_invalid():
    """Test loading invalid judge panel configuration raises error."""
    from rubric_kit.validator import RubricValidationError, load_judge_panel_config

    yaml_content = """
judge_panel:
  judges: []  # Empty judges list is invalid
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with pytest.raises(RubricValidationError):
            load_judge_panel_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_judge_panel_config_missing_section():
    """Test loading file without judge_panel section raises error."""
    from rubric_kit.validator import RubricValidationError, load_judge_panel_config

    yaml_content = """
dimensions:
  - test: Test dimension
    grading_type: binary
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with pytest.raises(RubricValidationError, match="judge_panel.*not found"):
            load_judge_panel_config(temp_path)
    finally:
        os.unlink(temp_path)


def test_load_judge_panel_config_from_rubric_yaml():
    """Test loading judge panel embedded in rubric YAML."""
    from rubric_kit.validator import load_judge_panel_config

    yaml_content = """
dimensions:
  - factual_correctness: Test
    grading_type: binary

criteria:
  - test_1:
      weight: 3
      dimension: factual_correctness
      criterion: Test criterion

judge_panel:
  judges:
    - name: judge_1
      model: gpt-4
  consensus:
    mode: unanimous
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        panel_config = load_judge_panel_config(temp_path)
        assert len(panel_config.judges) == 1
        assert panel_config.consensus.mode == "unanimous"
    finally:
        os.unlink(temp_path)
