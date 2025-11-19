"""YAML validation logic for rubric files."""

import yaml
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Union
from pydantic import ValidationError

from rubric_kit.schema import (
    Rubric, Dimension, Criterion, ToolCalls, ToolSpec,
    JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig
)


class RubricValidationError(Exception):
    """Custom exception for rubric validation errors."""
    pass


def _replace_env_var(match: re.Match) -> str:
    """Replace a single environment variable match."""
    var_name = match.group(1)
    default_value = match.group(2)
    
    value = os.getenv(var_name)
    if value is not None:
        return value
    
    if default_value is not None:
        return default_value
    
    # If no default provided, keep the original syntax
    return match.group(0)


def expand_env_vars(data: Any) -> Any:
    """
    Recursively expand environment variables in YAML data.
    
    Supports the following syntax:
    - ${ENV_VAR_NAME} - expands to environment variable value
    - ${ENV_VAR_NAME:-default_value} - expands with default if not set
    
    Args:
        data: YAML data structure (dict, list, string, or primitive)
        
    Returns:
        Data with environment variables expanded
    """
    if isinstance(data, dict):
        return {key: expand_env_vars(value) for key, value in data.items()}
    
    if isinstance(data, list):
        return [expand_env_vars(item) for item in data]
    
    if isinstance(data, str):
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        return re.sub(pattern, _replace_env_var, data)
    
    return data


def _parse_pure_nested_item(item: Dict) -> Dict:
    """Parse pure nested format: {"key": {"field": "val"}}."""
    key, value = list(item.items())[0]
    return {"name": key, **value}


def _parse_mixed_string_item(item: Dict) -> Dict:
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


def _parse_null_value_item(item: Dict) -> Dict:
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


def _parse_list_item(item: Dict) -> Dict:
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


def _parse_dict_item(key: str, value: Any) -> Dict:
    """Parse a single key-value pair from a dictionary."""
    if isinstance(value, dict):
        return {"name": key, **value}
    
    if isinstance(value, str):
        return {"name": key, "description": value}
    
    return {"name": key, "value": value}


def parse_nested_dict(data: Union[List[Dict], Dict]) -> List[Dict]:
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


def _parse_tool_specs(data: Any) -> List[ToolSpec]:
    """Parse a list of tool specifications."""
    if data is None:
        return []
    
    parsed_data = parse_nested_dict(data)
    return [ToolSpec(**item) for item in parsed_data]


def parse_tool_calls(tool_calls_data: Dict) -> ToolCalls:
    """Parse tool_calls structure from YAML."""
    return ToolCalls(
        respect_order=tool_calls_data.get("respect_order", True),
        required=_parse_tool_specs(tool_calls_data.get("required")),
        optional=_parse_tool_specs(tool_calls_data.get("optional")),
        prohibited=_parse_tool_specs(tool_calls_data.get("prohibited"))
    )


def _load_yaml_file(yaml_path: str) -> Dict[str, Any]:
    """Load and parse YAML file with error handling."""
    yaml_file = Path(yaml_path)
    
    if not yaml_file.exists():
        raise RubricValidationError(f"File not found: {yaml_path}")
    
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RubricValidationError(f"Invalid YAML syntax: {e}")
    
    if not isinstance(data, dict):
        raise RubricValidationError("YAML must contain a dictionary")
    
    return expand_env_vars(data)


def _parse_dimensions(dimensions_data: Any) -> List[Dimension]:
    """Parse dimensions from YAML data."""
    if not dimensions_data:
        return []
    
    dimensions = []
    parsed_data = parse_nested_dict(dimensions_data)
    
    for dim_data in parsed_data:
        try:
            dimensions.append(Dimension(**dim_data))
        except ValidationError as e:
            name = dim_data.get('name', 'unknown')
            raise RubricValidationError(f"Validation error in dimension '{name}': {e}")
    
    return dimensions


def _parse_criteria(criteria_data: Any) -> List[Criterion]:
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
                name = crit_data.get('name', 'unknown')
                raise RubricValidationError(
                    f"Validation error in tool_calls for criterion '{name}': {e}"
                )
        
        try:
            criteria.append(Criterion(**crit_data))
        except ValidationError as e:
            name = crit_data.get('name', 'unknown')
            raise RubricValidationError(f"Validation error in criterion '{name}': {e}")
    
    return criteria


def load_rubric(yaml_path: str) -> Rubric:
    """
    Load and validate a rubric from a YAML file.
    
    Args:
        yaml_path: Path to the YAML file
        
    Returns:
        Validated Rubric object
        
    Raises:
        RubricValidationError: If the file is invalid or validation fails
    """
    data = _load_yaml_file(yaml_path)
    
    dimensions = _parse_dimensions(data.get("dimensions"))
    criteria = _parse_criteria(data.get("criteria"))
    
    try:
        return Rubric(dimensions=dimensions, criteria=criteria)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error: {e}")


def _parse_judges(judges_data: Any) -> List[JudgeConfig]:
    """Parse judges from YAML data."""
    if not isinstance(judges_data, list):
        raise RubricValidationError("judges must be a list")
    
    judges = []
    for judge_data in judges_data:
        try:
            judges.append(JudgeConfig(**judge_data))
        except ValidationError as e:
            name = judge_data.get('name', 'unknown')
            raise RubricValidationError(f"Validation error in judge '{name}': {e}")
    
    return judges


def _parse_execution_config(execution_data: Any) -> ExecutionConfig:
    """Parse execution config from YAML data."""
    if not execution_data:
        return ExecutionConfig()
    
    try:
        return ExecutionConfig(**execution_data)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error in execution config: {e}")


def _parse_consensus_config(consensus_data: Any) -> ConsensusConfig:
    """Parse consensus config from YAML data."""
    if not consensus_data:
        return ConsensusConfig()
    
    try:
        return ConsensusConfig(**consensus_data)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error in consensus config: {e}")


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
        return JudgePanelConfig(
            judges=judges,
            execution=execution,
            consensus=consensus
        )
    except ValidationError as e:
        raise RubricValidationError(f"Validation error in judge panel config: {e}")

