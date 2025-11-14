"""YAML validation logic for rubric files."""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Union
from pydantic import ValidationError

from rubric_kit.schema import Rubric, Descriptor, Criterion, ToolCalls, ToolSpec


class RubricValidationError(Exception):
    """Custom exception for rubric validation errors."""
    pass


def parse_nested_dict(data: Union[List[Dict], Dict]) -> List[Dict]:
    """
    Parse nested dictionary format to flat list format.
    
    Handles two formats:
    1. Nested format: [{"key1": {"field1": "val1"}}, ...]
    2. Flat format with name as first string key: [{"name": "desc", "field1": "val1"}, ...]
    
    For descriptors, converts:
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
                # Check if this is already a flat format (has only one nested dict value)
                # or if it's a mixed format (name as key with string value + other fields)
                nested_keys = [k for k, v in item.items() if isinstance(v, dict)]
                string_keys = [k for k, v in item.items() if isinstance(v, str)]
                
                if len(nested_keys) == 1 and len(item) == 1:
                    # Pure nested format: {"key": {"field": "val"}}
                    key, value = list(item.items())[0]
                    flat_item = {"name": key, **value}
                    result.append(flat_item)
                elif string_keys:
                    # Mixed format: {"name": "Description", "other_field": "value"}
                    # Find the first string value and use it as description
                    flat_item = {}
                    name_found = False
                    for key, value in item.items():
                        if isinstance(value, str) and not name_found:
                            flat_item["name"] = key
                            flat_item["description"] = value
                            name_found = True
                        else:
                            flat_item[key] = value
                    result.append(flat_item)
                elif any(v is None for v in item.values()):
                    # Format with null values: {"tool_name": null, "min_calls": 1, ...}
                    flat_item = {}
                    name_found = False
                    for key, value in item.items():
                        if value is None and not name_found:
                            flat_item["name"] = key
                            name_found = True
                        elif key == "params" and value is None:
                            # Convert None params to empty dict
                            flat_item[key] = {}
                        else:
                            flat_item[key] = value
                    result.append(flat_item)
                else:
                    # Some other format, keep as is
                    result.append(item)
            else:
                result.append(item)
        return result
    elif isinstance(data, dict):
        # Dictionary format: convert each key-value to separate items
        result = []
        for key, value in data.items():
            if isinstance(value, dict):
                flat_item = {"name": key, **value}
                result.append(flat_item)
            elif isinstance(value, str):
                flat_item = {"name": key, "description": value}
                result.append(flat_item)
            else:
                result.append({"name": key, "value": value})
        return result
    else:
        return []


def parse_tool_calls(tool_calls_data: Dict) -> ToolCalls:
    """Parse tool_calls structure from YAML."""
    respect_order = tool_calls_data.get("respect_order", True)
    
    required = []
    if "required" in tool_calls_data:
        required_data = parse_nested_dict(tool_calls_data["required"])
        required = [ToolSpec(**item) for item in required_data]
    
    optional = []
    if "optional" in tool_calls_data:
        optional_data = parse_nested_dict(tool_calls_data["optional"])
        optional = [ToolSpec(**item) for item in optional_data]
    
    prohibited = []
    if "prohibited" in tool_calls_data:
        prohibited_data = parse_nested_dict(tool_calls_data["prohibited"])
        prohibited = [ToolSpec(**item) for item in prohibited_data]
    
    return ToolCalls(
        respect_order=respect_order,
        required=required,
        optional=optional,
        prohibited=prohibited
    )


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
    yaml_file = Path(yaml_path)
    
    # Check if file exists
    if not yaml_file.exists():
        raise RubricValidationError(f"File not found: {yaml_path}")
    
    # Load YAML
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RubricValidationError(f"Invalid YAML syntax: {e}")
    
    if not isinstance(data, dict):
        raise RubricValidationError("YAML must contain a dictionary")
    
    # Parse descriptors
    descriptors = []
    if "descriptors" in data:
        descriptors_data = parse_nested_dict(data["descriptors"])
        for desc_data in descriptors_data:
            try:
                descriptors.append(Descriptor(**desc_data))
            except ValidationError as e:
                raise RubricValidationError(f"Validation error in descriptor '{desc_data.get('name', 'unknown')}': {e}")
    
    # Parse criteria
    criteria = []
    if "criteria" in data:
        criteria_data = parse_nested_dict(data["criteria"])
        for crit_data in criteria_data:
            # Handle tool_calls if present
            if "tool_calls" in crit_data:
                try:
                    crit_data["tool_calls"] = parse_tool_calls(crit_data["tool_calls"])
                except (ValidationError, KeyError) as e:
                    raise RubricValidationError(f"Validation error in tool_calls for criterion '{crit_data.get('name', 'unknown')}': {e}")
            
            try:
                criteria.append(Criterion(**crit_data))
            except ValidationError as e:
                raise RubricValidationError(f"Validation error in criterion '{crit_data.get('name', 'unknown')}': {e}")
    
    # Create and validate rubric
    try:
        rubric = Rubric(descriptors=descriptors, criteria=criteria)
    except ValidationError as e:
        raise RubricValidationError(f"Validation error: {e}")
    
    return rubric

