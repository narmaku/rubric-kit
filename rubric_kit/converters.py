"""Conversion utilities for rubric-kit data structures.

Provides functions for converting between Pydantic models and dictionary formats
for YAML/JSON serialization, and rebuilding objects from dictionaries.
"""

from typing import Any

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


# =============================================================================
# Rebuild Functions (Dictionary -> Pydantic Models)
# =============================================================================


def rebuild_rubric_from_dict(rubric_data: dict[str, Any]) -> Rubric:
    """Rebuild a Rubric object from portable dictionary format."""
    dimensions = [
        Dimension(
            name=dim_data["name"],
            description=dim_data["description"],
            grading_type=dim_data["grading_type"],
            scores=dim_data.get("scores"),
            pass_above=dim_data.get("pass_above"),
        )
        for dim_data in rubric_data.get("dimensions", [])
    ]

    criteria = [_rebuild_criterion(crit_data) for crit_data in rubric_data.get("criteria", [])]

    return Rubric(dimensions=dimensions, criteria=criteria)


def _rebuild_criterion(crit_data: dict[str, Any]) -> Criterion:
    """Rebuild a single Criterion from dictionary data."""
    tool_calls = None
    if crit_data.get("tool_calls"):
        tool_calls = _rebuild_tool_calls(crit_data["tool_calls"])

    return Criterion(
        name=crit_data["name"],
        category=crit_data.get("category"),
        dimension=crit_data["dimension"],
        criterion=crit_data.get("criterion"),
        weight=crit_data["weight"],
        tool_calls=tool_calls,
    )


def _rebuild_tool_calls(tc_data: dict[str, Any]) -> ToolCalls:
    """Rebuild ToolCalls from dictionary data."""
    return ToolCalls(
        respect_order=tc_data.get("respect_order", True),
        params_strict_mode=tc_data.get("params_strict_mode", False),
        required=[
            ToolSpec(name=list(t.keys())[0], **(list(t.values())[0] or {}))
            for t in tc_data.get("required", [])
        ],
        optional=[
            ToolSpec(name=list(t.keys())[0], **(list(t.values())[0] or {}))
            for t in tc_data.get("optional", [])
        ],
        prohibited=[ToolSpec(name=list(t.keys())[0]) for t in tc_data.get("prohibited", [])],
    )


def rebuild_panel_config_from_dict(panel_data: dict[str, Any]) -> JudgePanelConfig:
    """Rebuild a JudgePanelConfig from portable dictionary format.

    Uses LiteLLM for provider auto-detection. API keys are read from
    environment variables based on the model's provider.
    """
    judges = [
        JudgeConfig(
            name=j_data["name"],
            model=j_data["model"],
            base_url=j_data.get("base_url"),
            temperature=j_data.get("temperature"),
            max_tokens=j_data.get("max_tokens"),
            top_p=j_data.get("top_p"),
            frequency_penalty=j_data.get("frequency_penalty"),
            presence_penalty=j_data.get("presence_penalty"),
        )
        for j_data in panel_data.get("judges", [])
    ]

    exec_data = panel_data.get("execution", {})
    execution = ExecutionConfig(
        mode=exec_data.get("mode", "sequential"),
        batch_size=exec_data.get("batch_size", 2),
        timeout=exec_data.get("timeout", 30),
    )

    cons_data = panel_data.get("consensus", {})
    consensus = ConsensusConfig(
        mode=cons_data.get("mode", "unanimous"),
        threshold=cons_data.get("threshold"),
        on_no_consensus=cons_data.get("on_no_consensus", "fail"),
    )

    return JudgePanelConfig(judges=judges, execution=execution, consensus=consensus)


# =============================================================================
# Conversion Functions (Pydantic Models -> Dictionary)
# =============================================================================


def tool_spec_to_dict(tool_spec: ToolSpec) -> dict[str, Any] | None:
    """Convert a ToolSpec to dictionary format.

    Returns None if there are no attributes to serialize.
    """
    tool_dict: dict[str, Any] = {}
    if tool_spec.min_calls is not None:
        tool_dict["min_calls"] = tool_spec.min_calls
    if tool_spec.max_calls is not None:
        tool_dict["max_calls"] = tool_spec.max_calls
    if tool_spec.params is not None:
        tool_dict["params"] = tool_spec.params
    return tool_dict if tool_dict else None


def tool_calls_to_dict(tool_calls: ToolCalls) -> dict[str, Any]:
    """Convert ToolCalls to dictionary format."""
    required_list = [{tc.name: tool_spec_to_dict(tc)} for tc in tool_calls.required]
    optional_list = [{tc.name: tool_spec_to_dict(tc)} for tc in tool_calls.optional]
    prohibited_list = [{tc.name: None} for tc in tool_calls.prohibited]

    result: dict[str, Any] = {
        "respect_order": tool_calls.respect_order,
        "required": required_list,
        "optional": optional_list if optional_list else [],
        "prohibited": prohibited_list if prohibited_list else [],
    }

    if tool_calls.params_strict_mode:
        result["params_strict_mode"] = True

    return result


def criterion_to_dict(criterion: Criterion, include_name: bool = False) -> dict[str, Any]:
    """Convert a Criterion to dictionary format.

    Args:
        criterion: The criterion to convert
        include_name: If True, include the 'name' field in output
    """
    crit_dict: dict[str, Any] = {}

    if include_name:
        crit_dict["name"] = criterion.name

    crit_dict.update(
        {
            "category": criterion.category,
            "weight": criterion.weight,
            "dimension": criterion.dimension,
            "criterion": criterion.criterion,
        }
    )

    if criterion.tool_calls:
        crit_dict["tool_calls"] = tool_calls_to_dict(criterion.tool_calls)

    return crit_dict


def dimension_to_dict(dimension: Dimension) -> dict[str, Any]:
    """Convert a Dimension to dictionary format."""
    dim_dict: dict[str, Any] = {
        dimension.name: dimension.description,
        "grading_type": dimension.grading_type,
    }
    if dimension.scores:
        dim_dict["scores"] = dimension.scores
    return dim_dict


def rubric_to_dict(rubric: Rubric) -> dict[str, Any]:
    """Convert a Rubric object to dictionary format suitable for YAML output."""
    rubric_dict: dict[str, Any] = {}

    if rubric.variables:
        rubric_dict["variables"] = dict(sorted(rubric.variables.items()))

    rubric_dict["dimensions"] = [dimension_to_dict(dim) for dim in rubric.dimensions]
    rubric_dict["criteria"] = {
        criterion.name: criterion_to_dict(criterion) for criterion in rubric.criteria
    }

    return rubric_dict


def rubric_to_portable_dict(rubric: Rubric) -> dict[str, Any]:
    """Convert a Rubric to portable dictionary format for embedding in output."""
    return {
        "dimensions": [
            {
                "name": dim.name,
                "description": dim.description,
                "grading_type": dim.grading_type,
                **({"scores": dim.scores} if dim.scores else {}),
            }
            for dim in rubric.dimensions
        ],
        "criteria": [criterion_to_dict(c, include_name=True) for c in rubric.criteria],
        **({"variables": rubric.variables} if rubric.variables else {}),
    }


def panel_config_to_portable_dict(panel_config: JudgePanelConfig) -> dict[str, Any]:
    """Convert a JudgePanelConfig to portable dictionary format."""
    return {
        "judges": [
            {
                "name": j.name,
                "model": j.model,
                "base_url": j.base_url,
                "api_key": None,  # Never export actual API key
            }
            for j in panel_config.judges
        ],
        "execution": {"mode": panel_config.execution.mode},
        "consensus": {
            "mode": panel_config.consensus.mode,
            **(
                {"threshold": panel_config.consensus.threshold}
                if panel_config.consensus.threshold
                else {}
            ),
            **(
                {"on_no_consensus": panel_config.consensus.on_no_consensus}
                if panel_config.consensus.on_no_consensus
                else {}
            ),
        },
    }
