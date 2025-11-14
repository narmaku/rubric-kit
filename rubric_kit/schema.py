"""Pydantic models for rubric YAML validation."""

from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class ToolSpec(BaseModel):
    """Specification for a tool call."""
    name: str = Field(..., description="Name of the tool")
    min_calls: Optional[int] = Field(None, ge=0, description="Minimum number of calls")
    max_calls: Optional[int] = Field(None, ge=0, description="Maximum number of calls")
    params: Dict = Field(default_factory=dict, description="Tool parameters")

    @model_validator(mode='after')
    def validate_calls(self):
        """Validate that min_calls <= max_calls if both are specified."""
        if self.min_calls is not None and self.max_calls is not None:
            if self.min_calls > self.max_calls:
                raise ValueError(f"min_calls ({self.min_calls}) must be <= max_calls ({self.max_calls})")
        return self


class ToolCalls(BaseModel):
    """Tool calls specification for a criterion."""
    respect_order: bool = Field(True, description="Whether tool call order matters")
    required: List[ToolSpec] = Field(default_factory=list, description="Required tool calls")
    optional: List[ToolSpec] = Field(default_factory=list, description="Optional tool calls")
    prohibited: List[ToolSpec] = Field(default_factory=list, description="Prohibited tool calls")


class Descriptor(BaseModel):
    """A descriptor defines an evaluation dimension."""
    name: str = Field(..., description="Name of the descriptor")
    description: str = Field(..., description="Description of what this descriptor evaluates")
    grading_type: Literal["binary", "score"] = Field(..., description="Type of grading: binary (pass/fail) or score")
    scores: Optional[Dict[int, str]] = Field(None, description="Score definitions (required for score type)")

    @model_validator(mode='after')
    def validate_scores(self):
        """Validate that score type has scores defined."""
        if self.grading_type == "score" and not self.scores:
            raise ValueError("Descriptor with grading_type 'score' must have scores defined")
        return self


class Criterion(BaseModel):
    """A criterion defines a specific evaluation rule."""
    name: str = Field(..., description="Name of the criterion")
    category: Optional[str] = Field(None, description="Category of the criterion (e.g., Output, Tools)")
    weight: Union[int, Literal["from_scores"]] = Field(..., description="Weight of the criterion (0-3) or 'from_scores'")
    dimension: str = Field(..., description="Dimension/descriptor this criterion evaluates")
    criterion: Optional[str] = Field(None, description="The criterion text")
    tool_calls: Optional[ToolCalls] = Field(None, description="Tool call specifications (for Tools category)")

    @field_validator('weight')
    @classmethod
    def validate_weight(cls, v):
        """Validate that weight is in range 0-3 or 'from_scores'."""
        if isinstance(v, int):
            if v < 0 or v > 3:
                raise ValueError(f"Weight must be between 0 and 3, got {v}")
        elif v != "from_scores":
            raise ValueError(f"Weight must be an integer (0-3) or 'from_scores', got {v}")
        return v


class Rubric(BaseModel):
    """Complete rubric with descriptors and criteria."""
    descriptors: List[Descriptor] = Field(..., description="List of descriptors")
    criteria: List[Criterion] = Field(..., description="List of criteria")

    @model_validator(mode='after')
    def validate_dimension_references(self):
        """Validate that all criteria reference valid descriptors."""
        descriptor_names = {d.name for d in self.descriptors}
        
        for criterion in self.criteria:
            if criterion.dimension not in descriptor_names:
                raise ValueError(
                    f"Criterion '{criterion.name}' references non-existent dimension '{criterion.dimension}'"
                )
        
        return self

    def get_descriptor(self, name: str) -> Optional[Descriptor]:
        """Get a descriptor by name."""
        for descriptor in self.descriptors:
            if descriptor.name == name:
                return descriptor
        return None

