"""Conversion utilities for rubric-kit data structures.

Provides functions for converting between Pydantic models and dictionary formats
for YAML/JSON serialization.
"""

from typing import Dict, Any, Optional

from rubric_kit.schema import (
    Rubric, Dimension, Criterion, ToolSpec, ToolCalls,
    JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig
)


def tool_spec_to_dict(tool_spec: ToolSpec) -> Optional[Dict[str, Any]]:
    """Convert a ToolSpec to dictionary format.
    
    Returns None if there are no attributes to serialize.
    """
    tool_dict: Dict[str, Any] = {}
    if tool_spec.min_calls is not None:
        tool_dict["min_calls"] = tool_spec.min_calls
    if tool_spec.max_calls is not None:
        tool_dict["max_calls"] = tool_spec.max_calls
    if tool_spec.params is not None:
        tool_dict["params"] = tool_spec.params
    return tool_dict if tool_dict else None


def tool_calls_to_dict(tool_calls: ToolCalls) -> Dict[str, Any]:
    """Convert ToolCalls to dictionary format."""
    required_list = [
        {tc.name: tool_spec_to_dict(tc)}
        for tc in tool_calls.required
    ]
    optional_list = [
        {tc.name: tool_spec_to_dict(tc)}
        for tc in tool_calls.optional
    ]
    prohibited_list = [
        {tc.name: None}
        for tc in tool_calls.prohibited
    ]
    
    result: Dict[str, Any] = {
        "respect_order": tool_calls.respect_order,
        "required": required_list,
        "optional": optional_list if optional_list else [],
        "prohibited": prohibited_list if prohibited_list else []
    }
    
    if tool_calls.params_strict_mode:
        result["params_strict_mode"] = True
    
    return result


def criterion_to_dict(criterion: Criterion, include_name: bool = False) -> Dict[str, Any]:
    """Convert a Criterion to dictionary format.
    
    Args:
        criterion: The criterion to convert
        include_name: If True, include the 'name' field in output
    """
    crit_dict: Dict[str, Any] = {}
    
    if include_name:
        crit_dict["name"] = criterion.name
    
    crit_dict.update({
        "category": criterion.category,
        "weight": criterion.weight,
        "dimension": criterion.dimension,
        "criterion": criterion.criterion
    })
    
    if criterion.tool_calls:
        crit_dict["tool_calls"] = tool_calls_to_dict(criterion.tool_calls)
    
    return crit_dict


def dimension_to_dict(dimension: Dimension) -> Dict[str, Any]:
    """Convert a Dimension to dictionary format."""
    dim_dict: Dict[str, Any] = {
        dimension.name: dimension.description,
        "grading_type": dimension.grading_type
    }
    if dimension.scores:
        dim_dict["scores"] = dimension.scores
    return dim_dict


def rubric_to_dict(rubric: Rubric) -> Dict[str, Any]:
    """Convert a Rubric object to dictionary format suitable for YAML output."""
    rubric_dict: Dict[str, Any] = {}
    
    if rubric.variables:
        rubric_dict["variables"] = dict(sorted(rubric.variables.items()))
    
    rubric_dict["dimensions"] = [dimension_to_dict(dim) for dim in rubric.dimensions]
    rubric_dict["criteria"] = {
        criterion.name: criterion_to_dict(criterion)
        for criterion in rubric.criteria
    }
    
    return rubric_dict


def rubric_to_portable_dict(rubric: Rubric) -> Dict[str, Any]:
    """Convert a Rubric to portable dictionary format for embedding in output."""
    return {
        "dimensions": [
            {
                "name": dim.name,
                "description": dim.description,
                "grading_type": dim.grading_type,
                **({"scores": dim.scores} if dim.scores else {})
            }
            for dim in rubric.dimensions
        ],
        "criteria": [
            criterion_to_dict(c, include_name=True)
            for c in rubric.criteria
        ],
        **({"variables": rubric.variables} if rubric.variables else {})
    }


def panel_config_to_portable_dict(panel_config: JudgePanelConfig) -> Dict[str, Any]:
    """Convert a JudgePanelConfig to portable dictionary format."""
    return {
        "judges": [
            {
                "name": j.name,
                "model": j.model,
                "base_url": j.base_url,
                "api_key": None  # Never export actual API key
            }
            for j in panel_config.judges
        ],
        "execution": {"mode": panel_config.execution.mode},
        "consensus": {
            "mode": panel_config.consensus.mode,
            **({"threshold": panel_config.consensus.threshold} 
               if panel_config.consensus.threshold else {}),
            **({"on_no_consensus": panel_config.consensus.on_no_consensus}
               if panel_config.consensus.on_no_consensus else {})
        }
    }

