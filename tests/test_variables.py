"""Tests for rubric variable substitution feature."""

import os
import tempfile

import pytest


class TestSubstituteVariables:
    """Tests for the substitute_variables function."""

    def test_substitute_single_variable(self):
        """Test substituting a single variable in criterion text."""
        from rubric_kit.validator import substitute_variables

        variables = {"ip_address": "10.0.187.159"}
        criterion_text = "The response correctly states the IP is '{{ip_address}}'."

        result = substitute_variables(criterion_text, variables)

        assert result == "The response correctly states the IP is '10.0.187.159'."

    def test_substitute_multiple_variables(self):
        """Test substituting multiple variables in same criterion."""
        from rubric_kit.validator import substitute_variables

        variables = {
            "os_name": "Red Hat Enterprise Linux 10.0",
            "kernel_version": "6.12.0-55.41.1.el10_0.x86_64",
        }
        criterion_text = "The OS is '{{os_name}}' with kernel '{{kernel_version}}'."

        result = substitute_variables(criterion_text, variables)

        assert (
            result
            == "The OS is 'Red Hat Enterprise Linux 10.0' with kernel '6.12.0-55.41.1.el10_0.x86_64'."
        )

    def test_substitute_repeated_variable(self):
        """Test same variable used multiple times."""
        from rubric_kit.validator import substitute_variables

        variables = {"value": "42"}
        criterion_text = "Value is {{value}} and doubled is {{value}}."

        result = substitute_variables(criterion_text, variables)

        assert result == "Value is 42 and doubled is 42."

    def test_substitute_no_variables_in_text(self):
        """Test text without any variable placeholders."""
        from rubric_kit.validator import substitute_variables

        variables = {"unused": "value"}
        criterion_text = "Plain text without variables."

        result = substitute_variables(criterion_text, variables)

        assert result == "Plain text without variables."

    def test_substitute_empty_variables_dict(self):
        """Test with empty variables dictionary."""
        from rubric_kit.validator import substitute_variables

        variables = {}
        criterion_text = "No variables defined."

        result = substitute_variables(criterion_text, variables)

        assert result == "No variables defined."

    def test_substitute_undefined_variable_raises_error(self):
        """Test that undefined variable raises clear error."""
        from rubric_kit.validator import RubricValidationError, substitute_variables

        variables = {"defined_var": "value"}
        criterion_text = "Uses {{undefined_var}} which is not defined."

        with pytest.raises(RubricValidationError, match="Undefined variable 'undefined_var'"):
            substitute_variables(criterion_text, variables)

    def test_substitute_with_special_chars_in_value(self):
        """Test variable values containing special characters."""
        from rubric_kit.validator import substitute_variables

        variables = {"path": "/dev/vda2", "percentage": "20.2%"}
        criterion_text = "Disk {{path}} is {{percentage}} full."

        result = substitute_variables(criterion_text, variables)

        assert result == "Disk /dev/vda2 is 20.2% full."

    def test_substitute_with_none_input(self):
        """Test that None input returns None."""
        from rubric_kit.validator import substitute_variables

        variables = {"var": "value"}

        result = substitute_variables(None, variables)

        assert result is None


class TestLoadRubricWithVariables:
    """Tests for loading rubrics with variables section."""

    def test_load_rubric_with_variables(self):
        """Test loading a rubric file that contains variables."""
        from rubric_kit.validator import load_rubric

        yaml_content = """
variables:
  ip_address: "10.0.187.159"
  ram_total: "1.7GB"

dimensions:
  - factual_accuracy: Evaluates factual accuracy.
    grading_type: binary

criteria:
  - ip_correct:
      category: Accuracy
      weight: 3
      dimension: factual_accuracy
      criterion: The response states the IP is '{{ip_address}}'.
  - ram_correct:
      category: Accuracy
      weight: 2
      dimension: factual_accuracy
      criterion: Total RAM is {{ram_total}}.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            rubric = load_rubric(temp_path)

            # Check variables were stored
            assert rubric.variables is not None
            assert rubric.variables["ip_address"] == "10.0.187.159"
            assert rubric.variables["ram_total"] == "1.7GB"

            # Check criterion text was substituted
            ip_criterion = next(c for c in rubric.criteria if c.name == "ip_correct")
            assert ip_criterion.criterion == "The response states the IP is '10.0.187.159'."

            ram_criterion = next(c for c in rubric.criteria if c.name == "ram_correct")
            assert ram_criterion.criterion == "Total RAM is 1.7GB."
        finally:
            os.unlink(temp_path)

    def test_load_rubric_without_variables(self):
        """Test loading a rubric file without variables section still works."""
        from rubric_kit.validator import load_rubric

        yaml_content = """
dimensions:
  - factual_accuracy: Evaluates factual accuracy.
    grading_type: binary

criteria:
  - test_criterion:
      category: Accuracy
      weight: 3
      dimension: factual_accuracy
      criterion: The response is accurate.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            rubric = load_rubric(temp_path)

            # Variables should be None or empty
            assert rubric.variables is None or rubric.variables == {}

            # Criterion should be unchanged
            assert rubric.criteria[0].criterion == "The response is accurate."
        finally:
            os.unlink(temp_path)

    def test_load_rubric_undefined_variable_error(self):
        """Test that undefined variable in criterion raises error."""
        from rubric_kit.validator import RubricValidationError, load_rubric

        yaml_content = """
variables:
  defined_var: "value"

dimensions:
  - test: Test dimension.
    grading_type: binary

criteria:
  - test_criterion:
      category: Test
      weight: 3
      dimension: test
      criterion: Uses {{undefined_var}} which is not defined.
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(RubricValidationError, match="undefined_var"):
                load_rubric(temp_path)
        finally:
            os.unlink(temp_path)

    def test_load_rubric_substitutes_variables_in_tool_params(self):
        """Test that variables are substituted in tool_calls params."""
        from rubric_kit.validator import load_rubric

        yaml_content = """
variables:
  host_ip: "10.0.187.159"
  username: "root"

dimensions:
  - tool_use: Evaluates tool usage.
    grading_type: binary

criteria:
  - tools_used:
      category: Tools
      weight: 3
      dimension: tool_use
      criterion: Tools must be called correctly.
      tool_calls:
        respect_order: false
        required:
          - get_system_info:
              min_calls: 1
              max_calls: 1
              params:
                host: "{{host_ip}}"
                user: "{{username}}"
        optional: []
        prohibited: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            rubric = load_rubric(temp_path)

            # Check variables were stored
            assert rubric.variables["host_ip"] == "10.0.187.159"
            assert rubric.variables["username"] == "root"

            # Check tool params were substituted
            tool_calls = rubric.criteria[0].tool_calls
            assert tool_calls.required[0].params["host"] == "10.0.187.159"
            assert tool_calls.required[0].params["user"] == "root"
        finally:
            os.unlink(temp_path)

    def test_load_rubric_with_external_variables_file(self):
        """Test loading rubric with placeholders and separate variables file."""
        from rubric_kit.validator import load_rubric

        # Rubric with placeholders but NO embedded variables
        rubric_content = """
dimensions:
  - factual_accuracy: Evaluates factual accuracy.
    grading_type: binary

criteria:
  - ip_correct:
      category: Accuracy
      weight: 3
      dimension: factual_accuracy
      criterion: The response states the IP is '{{ip_address}}'.
"""

        # Separate variables file
        variables_content = """
ip_address: "192.168.1.100"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(rubric_content)
            rubric_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(variables_content)
            variables_path = f.name

        try:
            rubric = load_rubric(rubric_path, variables_file=variables_path)

            # Check variables were loaded from external file
            assert rubric.variables["ip_address"] == "192.168.1.100"

            # Check criterion text was substituted
            assert rubric.criteria[0].criterion == "The response states the IP is '192.168.1.100'."
        finally:
            os.unlink(rubric_path)
            os.unlink(variables_path)

    def test_load_rubric_external_variables_override_embedded(self):
        """Test that external variables override embedded ones."""
        from rubric_kit.validator import load_rubric

        # Rubric with embedded variables
        rubric_content = """
variables:
  ip_address: "10.0.0.1"

dimensions:
  - factual_accuracy: Evaluates factual accuracy.
    grading_type: binary

criteria:
  - ip_correct:
      category: Accuracy
      weight: 3
      dimension: factual_accuracy
      criterion: The response states the IP is '{{ip_address}}'.
"""

        # External variables file with different value
        variables_content = """
ip_address: "192.168.1.100"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(rubric_content)
            rubric_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(variables_content)
            variables_path = f.name

        try:
            rubric = load_rubric(rubric_path, variables_file=variables_path)

            # External should override embedded
            assert rubric.variables["ip_address"] == "192.168.1.100"
            assert rubric.criteria[0].criterion == "The response states the IP is '192.168.1.100'."
        finally:
            os.unlink(rubric_path)
            os.unlink(variables_path)


class TestRubricSchemaWithVariables:
    """Tests for the Rubric schema with variables field."""

    def test_rubric_model_accepts_variables(self):
        """Test that Rubric model accepts variables field."""
        from rubric_kit.schema import Criterion, Dimension, Rubric

        dimensions = [Dimension(name="test", description="Test dimension", grading_type="binary")]
        criteria = [
            Criterion(
                name="test_criterion",
                category="Test",
                weight=3,
                dimension="test",
                criterion="Test criterion",
            )
        ]
        variables = {"var1": "value1", "var2": "value2"}

        rubric = Rubric(dimensions=dimensions, criteria=criteria, variables=variables)

        assert rubric.variables == {"var1": "value1", "var2": "value2"}

    def test_rubric_model_variables_optional(self):
        """Test that variables field is optional."""
        from rubric_kit.schema import Criterion, Dimension, Rubric

        dimensions = [Dimension(name="test", description="Test dimension", grading_type="binary")]
        criteria = [
            Criterion(
                name="test_criterion",
                category="Test",
                weight=3,
                dimension="test",
                criterion="Test criterion",
            )
        ]

        # Should work without variables
        rubric = Rubric(dimensions=dimensions, criteria=criteria)

        assert rubric.variables is None


class TestGeneratorWithVariables:
    """Tests for the generator producing rubrics with variables."""

    def test_generator_response_includes_variables(self):
        """Test that the generator can produce rubrics with a variables section."""
        from rubric_kit.generator import _convert_to_criteria, _convert_to_dimensions
        from rubric_kit.schema import Rubric

        # Simulate an LLM response that includes variables
        llm_response = {
            "variables": {"ip_address": "10.0.187.159", "ram_total": "1.7GB"},
            "dimensions": [
                {
                    "name": "factual_accuracy",
                    "description": "Evaluates factual accuracy",
                    "grading_type": "binary",
                }
            ],
            "criteria": [
                {
                    "name": "ip_correct",
                    "category": "Accuracy",
                    "weight": 3,
                    "dimension": "factual_accuracy",
                    "criterion": "The response states the IP is '{{ip_address}}'.",
                }
            ],
        }

        dimensions = _convert_to_dimensions(llm_response["dimensions"])
        criteria = _convert_to_criteria(llm_response["criteria"])
        variables = llm_response.get("variables")

        rubric = Rubric(dimensions=dimensions, criteria=criteria, variables=variables)

        assert rubric.variables == {"ip_address": "10.0.187.159", "ram_total": "1.7GB"}
        assert "{{ip_address}}" in rubric.criteria[0].criterion

    def test_convert_to_criteria_normalizes_tool_name_key(self):
        """Test that tool_name key is normalized to name in tool_calls."""
        from rubric_kit.generator import _convert_to_criteria

        # LLM sometimes returns 'tool_name' instead of 'name'
        llm_response = [
            {
                "name": "tools_used",
                "category": "Tools",
                "weight": 3,
                "dimension": "tool_use",
                "criterion": "All tools must be called",
                "tool_calls": {
                    "respect_order": False,
                    "required": [
                        {"tool_name": "get_system_info", "min_calls": 1, "max_calls": 1},
                        {"tool_name": "get_memory_info", "min_calls": 1},
                    ],
                    "optional": [{"tool_name": "get_network_info", "max_calls": 1}],
                    "prohibited": [],
                },
            }
        ]

        criteria = _convert_to_criteria(llm_response)

        assert len(criteria) == 1
        assert criteria[0].tool_calls is not None
        assert len(criteria[0].tool_calls.required) == 2
        assert criteria[0].tool_calls.required[0].name == "get_system_info"
        assert criteria[0].tool_calls.required[1].name == "get_memory_info"
        assert criteria[0].tool_calls.optional[0].name == "get_network_info"

    def test_convert_to_criteria_preserves_tool_params(self):
        """Test that tool params are preserved during conversion."""
        from rubric_kit.generator import _convert_to_criteria

        llm_response = [
            {
                "name": "tools_used",
                "category": "Tools",
                "weight": 3,
                "dimension": "tool_use",
                "criterion": "Tools must be called with correct params",
                "tool_calls": {
                    "respect_order": False,
                    "required": [
                        {
                            "name": "get_system_info",
                            "min_calls": 1,
                            "max_calls": 1,
                            "params": {"hostname": "{{hostname}}", "username": "{{username}}"},
                        }
                    ],
                    "optional": [],
                    "prohibited": [],
                },
            }
        ]

        criteria = _convert_to_criteria(llm_response)

        assert len(criteria) == 1
        assert criteria[0].tool_calls is not None
        assert criteria[0].tool_calls.required[0].name == "get_system_info"
        assert criteria[0].tool_calls.required[0].params == {
            "hostname": "{{hostname}}",
            "username": "{{username}}",
        }


class TestRubricYAMLOutput:
    """Tests for rubric YAML output with variables."""

    def test_convert_rubric_to_yaml_dict_includes_variables(self):
        """Test that convert_rubric_to_yaml_dict includes variables section."""
        from rubric_kit.converters import rubric_to_dict as convert_rubric_to_yaml_dict
        from rubric_kit.schema import Criterion, Dimension, Rubric

        rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test dim", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_crit",
                    category="Test",
                    weight=3,
                    dimension="test",
                    criterion="Value is {{value}}.",
                )
            ],
            variables={"value": "42", "name": "test"},
        )

        result = convert_rubric_to_yaml_dict(rubric)

        assert "variables" in result
        assert result["variables"] == {"value": "42", "name": "test"}
        # Variables should be first key
        assert list(result.keys())[0] == "variables"

    def test_convert_rubric_to_yaml_dict_no_variables(self):
        """Test that rubric without variables doesn't have variables section."""
        from rubric_kit.converters import rubric_to_dict as convert_rubric_to_yaml_dict
        from rubric_kit.schema import Criterion, Dimension, Rubric

        rubric = Rubric(
            dimensions=[Dimension(name="test", description="Test dim", grading_type="binary")],
            criteria=[
                Criterion(
                    name="test_crit",
                    category="Test",
                    weight=3,
                    dimension="test",
                    criterion="Simple criterion.",
                )
            ],
        )

        result = convert_rubric_to_yaml_dict(rubric)

        assert "variables" not in result

    def test_chat_criteria_generation_prompt_includes_variable_instructions(self):
        """Test that chat criteria generation prompt includes variable extraction instructions."""
        from rubric_kit.prompts import build_chat_criteria_generation_prompt
        from rubric_kit.schema import Dimension

        dimensions = [
            Dimension(
                name="factual_accuracy",
                description="Evaluates factual accuracy",
                grading_type="binary",
            )
        ]

        prompt = build_chat_criteria_generation_prompt(
            chat_content="Test chat content",
            dimensions=dimensions,
            num_criteria=None,
            category_hints=None,
            context=None,
        )

        # The prompt should mention variables
        assert "variables" in prompt.lower()
        assert "{{" in prompt  # Should show the variable placeholder syntax
