"""YAML validation logic for rubric files."""

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from rubric_kit.schema import (
    ConsensusConfig,
    Criterion,
    Dimension,
    ExecutionConfig,
    JudgeConfig,
    JudgePanelConfig,
    Rubric,
    ToolCalls,
    ToolSpec,
)


class RubricValidationError(Exception):
    """Custom exception for rubric validation errors."""

    pass


def substitute_variables(text: str | None, variables: dict[str, str]) -> str | None:
    """
    Substitute variable placeholders in text with their values.

    Uses {{variable_name}} syntax for placeholders.

    Args:
        text: Text containing variable placeholders (or None)
        variables: Dictionary mapping variable names to values

    Returns:
        Text with variables substituted, or None if input was None

    Raises:
        RubricValidationError: If a variable placeholder references an undefined variable
    """
    if text is None:
        return None

    if not variables:
        # Check if there are any placeholders that would be unresolved
        pattern = r"\{\{([^}]+)\}\}"
        match = re.search(pattern, text)
        if match:
            var_name = match.group(1).strip()
            raise RubricValidationError(
                f"Undefined variable '{var_name}' in criterion text. "
                f"Add it to the 'variables' section of the rubric."
            )
        return text

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1).strip()
        if var_name not in variables:
            raise RubricValidationError(
                f"Undefined variable '{var_name}' in criterion text. "
                f"Defined variables: {', '.join(sorted(variables.keys()))}"
            )
        return variables[var_name]

    pattern = r"\{\{([^}]+)\}\}"
    return re.sub(pattern, replace_var, text)


def _parse_pure_nested_item(item: dict) -> dict:
    """Parse pure nested format: {"key": {"field": "val"}}."""
    key, value = list(item.items())[0]
    return {"name": key, **value}


def _parse_mixed_string_item(item: dict) -> dict:
    """Parse mixed format: {"name": "Description", "other_field": "value"}."""
    flat_item = {}
    name_found = False

    for key, value in item.items():
        if isinstance(value, str) and not name_found:
            flat_item["name"] = key
            flat_item["description"] = value
            name_found = True
        else:
            flat_item[key] = value

    return flat_item


def _parse_null_value_item(item: dict) -> dict:
    """Parse format with null values: {"tool_name": null, "min_calls": 1, ...}."""
    flat_item = {}
    name_found = False

    for key, value in item.items():
        if value is None and not name_found:
            flat_item["name"] = key
            name_found = True
        elif key == "params" and value is None:
            flat_item[key] = {}
        else:
            flat_item[key] = value

    return flat_item


def _parse_list_item(item: dict) -> dict:
    """Parse a single dictionary item from a list."""
    nested_keys = [k for k, v in item.items() if isinstance(v, dict)]
    string_keys = [k for k, v in item.items() if isinstance(v, str)]
    has_nulls = any(v is None for v in item.values())

    if len(nested_keys) == 1 and len(item) == 1:
        return _parse_pure_nested_item(item)

    if string_keys:
        return _parse_mixed_string_item(item)

    if has_nulls:
        return _parse_null_value_item(item)

    return item


def _parse_dict_item(key: str, value: Any) -> dict:
    """Parse a single key-value pair from a dictionary."""
    if isinstance(value, dict):
        return {"name": key, **value}

    if isinstance(value, str):
        return {"name": key, "description": value}

    return {"name": key, "value": value}


def parse_nested_dict(data: list[dict] | dict) -> list[dict]:
    """
    Parse nested dictionary format to flat list format.

    Handles two formats:
    1. Nested format: [{"key1": {"field1": "val1"}}, ...]
    2. Flat format with name as first string key: [{"name": "desc", "field1": "val1"}, ...]

    For dimensions, converts:
        [{"factual_correctness": "Description text", "grading_type": "binary"}]
    To:
        [{"name": "factual_correctness", "description": "Description text", "grading_type": "binary"}]

    For criteria, converts:
        {"criterion_name": {"weight": 3, "dimension": "test"}}
    To:
        [{"name": "criterion_name", "weight": 3, "dimension": "test"}]
    """
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, dict):
                result.append(_parse_list_item(item))
            else:
                result.append(item)
        return result

    if isinstance(data, dict):
        return [_parse_dict_item(key, value) for key, value in data.items()]

    return []


def _parse_tool_specs(data: Any) -> list[ToolSpec]:
    """Parse a list of tool specifications."""
    if data is None:
        return []

    parsed_data = parse_nested_dict(data)
    return [ToolSpec(**item) for item in parsed_data]


def parse_tool_calls(tool_calls_data: dict) -> ToolCalls:
    """Parse tool_calls structure from YAML."""
    return ToolCalls(
        respect_order=tool_calls_data.get("respect_order", True),
        required=_parse_tool_specs(tool_calls_data.get("required")),
        optional=_parse_tool_specs(tool_calls_data.get("optional")),
        prohibited=_parse_tool_specs(tool_calls_data.get("prohibited")),
        params_strict_mode=tool_calls_data.get("params_strict_mode", False),
    )


def _load_yaml_file(yaml_path: str) -> dict[str, Any]:
    """Load and parse YAML file with error handling."""
    yaml_file = Path(yaml_path)

    if not yaml_file.exists():
        raise RubricValidationError(f"File not found: {yaml_path}")

    try:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RubricValidationError(f"Invalid YAML syntax: {e}") from e

    if not isinstance(data, dict):
        raise RubricValidationError("YAML must contain a dictionary")

    return data


def _parse_dimensions(dimensions_data: Any) -> list[Dimension]:
    """Parse dimensions from YAML data."""
    if not dimensions_data:
        return []

    dimensions = []
    parsed_data = parse_nested_dict(dimensions_data)

    for dim_data in parsed_data:
        try:
            dimensions.append(Dimension(**dim_data))
        except ValidationError as e:
            name = dim_data.get("name", "unknown")
            raise RubricValidationError(f"Validation error in dimension '{name}': {e}") from e

    return dimensions


def _parse_criteria(criteria_data: Any) -> list[Criterion]:
    """Parse criteria from YAML data."""
    if not criteria_data:
        return []

    criteria = []
    parsed_data = parse_nested_dict(criteria_data)

    for crit_data in parsed_data:
        if "tool_calls" in crit_data:
            try:
                crit_data["tool_calls"] = parse_tool_calls(crit_data["tool_calls"])
            except (ValidationError, KeyError) as e:
                name = crit_data.get("name", "unknown")
                raise RubricValidationError(
                    f"Validation error in tool_calls for criterion '{name}': {e}"
                ) from e

        try:
            criteria.append(Criterion(**crit_data))
        except ValidationError as e:
            name = crit_data.get("name", "unknown")
            raise RubricValidationError(f"Validation error in criterion '{name}': {e}") from e

    return criteria


def _parse_variables(variables_data: Any) -> dict[str, str] | None:
    """Parse variables from YAML data."""
    if not variables_data:
        return None

    if not isinstance(variables_data, dict):
        raise RubricValidationError("'variables' must be a dictionary")

    # Ensure all values are strings
    variables = {}
    for key, value in variables_data.items():
        if not isinstance(key, str):
            raise RubricValidationError(
                f"Variable name must be a string, got: {type(key).__name__}"
            )
        variables[key] = str(value) if value is not None else ""

    return variables


def load_variables_file(yaml_path: str) -> dict[str, str]:
    """
    Load variables from a YAML file.

    Args:
        yaml_path: Path to the variables YAML file

    Returns:
        Dictionary of variable names to values

    Raises:
        RubricValidationError: If the file is invalid
    """
    yaml_file = Path(yaml_path)

    if not yaml_file.exists():
        raise RubricValidationError(f"Variables file not found: {yaml_path}")

    try:
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RubricValidationError(f"Invalid YAML syntax in variables file: {e}") from e

    if not isinstance(data, dict):
        raise RubricValidationError("Variables file must contain a dictionary of key-value pairs")

    variables = _parse_variables(data)
    if variables is None:
        raise RubricValidationError("Variables file is empty")

    return variables


def _substitute_variables_in_tool_calls(
    tool_calls: ToolCalls, variables: dict[str, str]
) -> ToolCalls:
    """Substitute variables in tool_calls params."""

    def substitute_in_tool_spec(spec: ToolSpec) -> ToolSpec:
        if spec.params is None:
            return spec

        # Substitute variables in each param value
        new_params = {}
        for key, value in spec.params.items():
            if isinstance(value, str):
                new_params[key] = substitute_variables(value, variables)
            else:
                new_params[key] = value

        return ToolSpec(
            name=spec.name, min_calls=spec.min_calls, max_calls=spec.max_calls, params=new_params
        )

    return ToolCalls(
        respect_order=tool_calls.respect_order,
        required=[substitute_in_tool_spec(spec) for spec in tool_calls.required],
        optional=[substitute_in_tool_spec(spec) for spec in tool_calls.optional],
        prohibited=tool_calls.prohibited,  # Prohibited don't have params
        params_strict_mode=tool_calls.params_strict_mode,
    )


def _substitute_variables_in_criteria(
    criteria: list[Criterion], variables: dict[str, str] | None
) -> list[Criterion]:
    """Substitute variables in all criteria's criterion text and tool_calls params."""
    if not variables:
        # Still need to check for undefined variables
        for crit in criteria:
            if crit.criterion and crit.criterion != "from_scores":
                substitute_variables(crit.criterion, {})
        return criteria

    substituted_criteria = []
    for crit in criteria:
        new_criterion_text = crit.criterion
        new_tool_calls = crit.tool_calls

        # Substitute in criterion text
        if crit.criterion and crit.criterion != "from_scores":
            new_criterion_text = substitute_variables(crit.criterion, variables)

        # Substitute in tool_calls params
        if crit.tool_calls:
            new_tool_calls = _substitute_variables_in_tool_calls(crit.tool_calls, variables)

        substituted_criteria.append(
            Criterion(
                name=crit.name,
                category=crit.category,
                weight=crit.weight,
                dimension=crit.dimension,
                criterion=new_criterion_text,
                tool_calls=new_tool_calls,
            )
        )

    return substituted_criteria


def load_rubric(
    yaml_path: str, variables_file: str | None = None, require_variables: bool = True
) -> Rubric:
    """
    Load and validate a rubric from a YAML file.

    Supports variable substitution using {{variable_name}} syntax in criterion text
    and tool_calls params. Variables can be:
    - Embedded in the rubric YAML under the 'variables' section
    - Provided via a separate variables file (overrides embedded variables)

    Args:
        yaml_path: Path to the rubric YAML file
        variables_file: Optional path to external variables YAML file
        require_variables: If True (default), raises error for undefined variables.
            If False, keeps placeholders intact (useful for refine/generate commands).

    Returns:
        Validated Rubric object with variables substituted in criteria

    Raises:
        RubricValidationError: If the file is invalid, validation fails, or
            required variables are missing (when require_variables=True)
    """
    data = _load_yaml_file(yaml_path)

    # Parse embedded variables
    embedded_variables = _parse_variables(data.get("variables"))

    # Load external variables if provided (override embedded)
    if variables_file:
        external_variables = load_variables_file(variables_file)
        # Merge: external overrides embedded
        if embedded_variables:
            variables = {**embedded_variables, **external_variables}
        else:
            variables = external_variables
    else:
        variables = embedded_variables

    dimensions = _parse_dimensions(data.get("dimensions"))
    criteria = _parse_criteria(data.get("criteria"))

    # Substitute variables in criteria if we have them or require them
    if variables or require_variables:
        criteria = _substitute_variables_in_criteria(criteria, variables)
    # else: keep placeholders intact (for refine/generate when no variables available)

    try:
        return Rubric(dimensions=dimensions, criteria=criteria, variables=variables)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error: {e}") from e


def _parse_judges(judges_data: Any) -> list[JudgeConfig]:
    """Parse judges from YAML data."""
    if not isinstance(judges_data, list):
        raise RubricValidationError("judges must be a list")

    judges = []
    for judge_data in judges_data:
        try:
            judges.append(JudgeConfig(**judge_data))
        except ValidationError as e:
            name = judge_data.get("name", "unknown")
            raise RubricValidationError(f"Validation error in judge '{name}': {e}") from e

    return judges


def _parse_execution_config(execution_data: Any) -> ExecutionConfig:
    """Parse execution config from YAML data."""
    if not execution_data:
        return ExecutionConfig()

    try:
        return ExecutionConfig(**execution_data)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error in execution config: {e}") from e


def _parse_consensus_config(consensus_data: Any) -> ConsensusConfig:
    """Parse consensus config from YAML data."""
    if not consensus_data:
        return ConsensusConfig()

    try:
        return ConsensusConfig(**consensus_data)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error in consensus config: {e}") from e


def load_judge_panel_config(yaml_path: str) -> JudgePanelConfig:
    """
    Load and validate judge panel configuration from a YAML file.

    The judge panel configuration can be in a standalone file or embedded
    in the rubric YAML under the 'judge_panel' key.

    Args:
        yaml_path: Path to the YAML file

    Returns:
        Validated JudgePanelConfig object

    Raises:
        RubricValidationError: If the file is invalid or validation fails
    """
    data = _load_yaml_file(yaml_path)

    if "judge_panel" not in data:
        raise RubricValidationError("judge_panel section not found in YAML file")

    panel_data = data["judge_panel"]
    if not isinstance(panel_data, dict):
        raise RubricValidationError("judge_panel must be a dictionary")

    if "judges" not in panel_data:
        raise RubricValidationError("judges list not found in judge_panel")

    judges = _parse_judges(panel_data["judges"])
    execution = _parse_execution_config(panel_data.get("execution"))
    consensus = _parse_consensus_config(panel_data.get("consensus"))

    try:
        return JudgePanelConfig(judges=judges, execution=execution, consensus=consensus)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error in judge panel config: {e}") from e
