"""Rubric Kit - Generate high-quality Rubrics based on custom dimensions, descriptors, criteria and scoring system."""

__version__ = "0.2.0"

# Public API functions
# Result types
from rubric_kit.api import (
    ArenaRanking,
    ArenaResult,
    ContestantResult,
    CriterionResult,
    DryRunResult,
    EvaluationResult,
    ExportResult,
    GenerationResult,
    RefinementResult,
    ScoreSummary,
    dry_run_evaluate,
    evaluate,
    export,
    generate,
    refine,
)

# Core domain models
from rubric_kit.schema import (
    ConsensusConfig,
    Criterion,
    Dimension,
    ExecutionConfig,
    JudgeConfig,
    JudgePanelConfig,
    Rubric,
)

# Exceptions
from rubric_kit.validator import RubricValidationError


__all__ = [
    # API functions
    "evaluate",
    "generate",
    "refine",
    "export",
    "dry_run_evaluate",
    # Result types
    "EvaluationResult",
    "GenerationResult",
    "RefinementResult",
    "ArenaResult",
    "DryRunResult",
    "ExportResult",
    "CriterionResult",
    "ScoreSummary",
    "ContestantResult",
    "ArenaRanking",
    # Domain models
    "Rubric",
    "Dimension",
    "Criterion",
    "JudgeConfig",
    "JudgePanelConfig",
    "ExecutionConfig",
    "ConsensusConfig",
    # Exceptions
    "RubricValidationError",
]
