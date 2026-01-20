"""Pydantic models for rubric YAML validation."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Judge Panel Models
# ============================================================================


class JudgeConfig(BaseModel):
    """Configuration for a single judge."""

    name: str = Field(..., description="Unique name for this judge")
    model: str = Field(..., description="Model name (e.g., gpt-4, claude-3-5-sonnet)")
    api_key: str | None = Field(None, description="API key (null uses env var)")
    base_url: str | None = Field(None, description="Custom API endpoint")
    # Optional LLM inference parameters (if not provided, defaults from prompts.py are used)
    temperature: float | None = Field(
        None, ge=0.0, le=2.0, description="Temperature for inference (0.0-2.0)"
    )
    max_tokens: int | None = Field(None, ge=1, description="Maximum tokens in response")
    top_p: float | None = Field(
        None, ge=0.0, le=1.0, description="Top-p sampling parameter (0.0-1.0)"
    )
    frequency_penalty: float | None = Field(
        None, ge=-2.0, le=2.0, description="Frequency penalty (-2.0 to 2.0)"
    )
    presence_penalty: float | None = Field(
        None, ge=-2.0, le=2.0, description="Presence penalty (-2.0 to 2.0)"
    )


class ExecutionConfig(BaseModel):
    """Configuration for judge execution strategy."""

    mode: Literal["sequential", "parallel", "batched"] = Field(
        "sequential", description="How to execute judge calls"
    )
    batch_size: int = Field(2, ge=1, description="Batch size for batched mode")
    timeout: int = Field(30, ge=1, description="Timeout per judge call in seconds")


class ConsensusConfig(BaseModel):
    """Consensus configuration."""

    mode: Literal["quorum", "majority", "unanimous"] = Field(
        "unanimous", description="Consensus mode"
    )
    threshold: int | None = Field(None, ge=1, description="Required for quorum mode")
    on_no_consensus: Literal["fail", "median", "most_common"] = Field(
        "fail", description="How to handle no consensus (fail=conservative)"
    )

    @model_validator(mode="after")
    def validate_threshold(self):
        """Validate threshold is provided for quorum mode."""
        if self.mode != "quorum":
            return self

        if self.threshold is None:
            raise ValueError("threshold is required for quorum consensus mode")

        return self


class JudgePanelConfig(BaseModel):
    """Judge panel configuration."""

    judges: list[JudgeConfig] = Field(..., min_length=1, description="List of judges")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    consensus: ConsensusConfig = Field(default_factory=ConsensusConfig)

    def _calculate_threshold(self, num_judges: int) -> int:
        """Calculate threshold based on consensus mode."""
        if self.consensus.mode == "majority":
            return (num_judges // 2) + 1
        if self.consensus.mode == "unanimous":
            return num_judges
        return num_judges  # fallback (should not happen)

    @model_validator(mode="after")
    def validate_consensus_threshold(self):
        """Validate consensus threshold against number of judges."""
        num_judges = len(self.judges)

        if self.consensus.mode == "quorum":
            self._validate_quorum_threshold(num_judges)
            return self

        # Auto-calculate threshold for majority and unanimous modes
        self.consensus.threshold = self._calculate_threshold(num_judges)
        return self

    def _validate_quorum_threshold(self, num_judges: int) -> None:
        """Validate quorum threshold does not exceed number of judges."""
        if self.consensus.threshold and self.consensus.threshold > num_judges:
            raise ValueError(
                f"Consensus threshold ({self.consensus.threshold}) cannot exceed "
                f"number of judges ({num_judges})"
            )


# ============================================================================
# Rubric Models
# ============================================================================


class ToolSpec(BaseModel):
    """Specification for a tool call."""

    name: str = Field(..., description="Name of the tool")
    min_calls: int | None = Field(None, ge=0, description="Minimum number of calls")
    max_calls: int | None = Field(None, ge=0, description="Maximum number of calls")
    params: dict | None = Field(
        None,
        description="Tool parameters. None=no validation, empty dict=check no params used, dict with values=check specified params",
    )

    @model_validator(mode="after")
    def validate_calls(self):
        """Validate that min_calls <= max_calls if both are specified."""
        if self.min_calls is None or self.max_calls is None:
            return self

        if self.min_calls > self.max_calls:
            raise ValueError(
                f"min_calls ({self.min_calls}) must be <= max_calls ({self.max_calls})"
            )

        return self


class ToolCalls(BaseModel):
    """Tool calls specification for a criterion.

    Scoring is inferred from which tool lists are populated:
    - required: Tools that MUST be called (Pass = weight, Fail = 0)
    - optional: Tools that give bonus points if called (Pass = bonus, Fail = 0)
    - prohibited: Tools that MUST NOT be called (Violation = penalty)

    A criterion can have any combination of these lists.
    """

    respect_order: bool = Field(True, description="Whether tool call order matters")
    required: list[ToolSpec] = Field(default_factory=list, description="Required tool calls")
    optional: list[ToolSpec] = Field(default_factory=list, description="Optional tool calls")
    prohibited: list[ToolSpec] = Field(default_factory=list, description="Prohibited tool calls")
    params_strict_mode: bool = Field(
        False,
        description="If True, exactly the specified params must match. If False, extra params are ignored.",
    )


class Dimension(BaseModel):
    """A dimension defines an evaluation aspect."""

    name: str = Field(..., description="Name of the dimension")
    description: str = Field(..., description="Description of what this dimension evaluates")
    grading_type: Literal["binary", "score"] = Field(
        ..., description="Type of grading: binary (pass/fail) or score"
    )
    scores: dict[int, str] | None = Field(
        None, description="Score definitions (required for score type)"
    )
    pass_above: int | None = Field(
        None, description="Minimum score to count as 'pass' (for score type only)"
    )

    @model_validator(mode="after")
    def validate_scores(self):
        """Validate that score type has scores defined and pass_above is valid."""
        if self.grading_type == "score" and not self.scores:
            raise ValueError("Dimension with grading_type 'score' must have scores defined")

        if self.pass_above is None:
            return self

        if self.grading_type != "score":
            raise ValueError("pass_above can only be used with grading_type 'score'")

        if self.scores and self.pass_above not in self.scores:
            raise ValueError(
                f"pass_above value {self.pass_above} must be a valid score in the scores dictionary"
            )

        return self


class Criterion(BaseModel):
    """A criterion defines a specific evaluation rule."""

    name: str = Field(..., description="Name of the criterion")
    category: str | None = Field(
        None, description="Category of the criterion (e.g., Output, Tools)"
    )
    weight: int | Literal["from_scores"] = Field(
        ..., description="Weight of the criterion (0-3) or 'from_scores'"
    )
    dimension: str = Field(..., description="Dimension this criterion evaluates")
    criterion: str | None = Field(None, description="The criterion text")
    tool_calls: ToolCalls | None = Field(
        None, description="Tool call specifications (for Tools category)"
    )

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v):
        """Validate that weight is in range 0-3 or 'from_scores'."""
        if v == "from_scores":
            return v

        if not isinstance(v, int):
            raise ValueError(f"Weight must be an integer (0-3) or 'from_scores', got {v}")

        if v < 0 or v > 3:
            raise ValueError(f"Weight must be between 0 and 3, got {v}")

        return v


class Rubric(BaseModel):
    """Complete rubric with dimensions and criteria."""

    dimensions: list[Dimension] = Field(..., description="List of dimensions")
    criteria: list[Criterion] = Field(..., description="List of criteria")
    variables: dict[str, Any] | None = Field(
        None, description="Variables for placeholder substitution in criteria"
    )

    @field_validator("variables", mode="before")
    @classmethod
    def coerce_variables_to_strings(cls, v):
        """Coerce all variable values to strings.

        LLMs sometimes return numeric values for variables that should be strings.
        Since variables are used for text substitution, we coerce all values to strings.
        """
        if v is None:
            return v
        return {key: str(value) for key, value in v.items()}

    @model_validator(mode="after")
    def validate_dimension_references(self):
        """Validate that all criteria reference valid dimensions."""
        dimension_names = {d.name for d in self.dimensions}

        for criterion in self.criteria:
            if criterion.dimension not in dimension_names:
                raise ValueError(
                    f"Criterion '{criterion.name}' references non-existent dimension '{criterion.dimension}'"
                )

        return self

    def get_dimension(self, name: str) -> Dimension | None:
        """Get a dimension by name."""
        return next((d for d in self.dimensions if d.name == name), None)


# ============================================================================
# Arena Models
# ============================================================================


class ArenaContestant(BaseModel):
    """A contestant in an arena evaluation."""

    id: str = Field(..., description="Unique identifier for this contestant")
    name: str = Field(..., description="Display name for this contestant")
    input_type: Literal["chat_session", "qna"] = Field(
        "chat_session", description="Type of input: chat_session or qna"
    )
    input_file: str = Field(..., description="Path to input file")
    variables: dict[str, str] | None = Field(
        None, description="Inline variables for rubric placeholder substitution"
    )
    variables_file: str | None = Field(None, description="Path to external variables file")
    metadata: dict[str, str | int | float | bool] | None = Field(
        None, description="Custom metadata (e.g., model version, temperature)"
    )
    description: str | None = Field(
        None, description="Human-readable description of this contestant"
    )


class ArenaSpec(BaseModel):
    """Specification for an arena comparative evaluation."""

    name: str | None = Field(None, description="Name of this arena evaluation")
    description: str | None = Field(None, description="Description of the arena")
    rubric_file: str = Field(..., description="Path to the shared rubric file")
    judges_panel_file: str = Field(..., description="Path to the shared judges panel file")
    contestants: list[ArenaContestant] = Field(
        ..., min_length=1, description="List of contestants to evaluate"
    )

    @model_validator(mode="after")
    def validate_unique_contestant_ids(self):
        """Validate that all contestant IDs are unique."""
        ids = [c.id for c in self.contestants]
        if len(ids) != len(set(ids)):
            seen = set()
            for cid in ids:
                if cid in seen:
                    raise ValueError(f"Duplicate contestant id: '{cid}'")
                seen.add(cid)
        return self
