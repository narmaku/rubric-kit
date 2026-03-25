"""Tests for package reorganization (PR #13).

Validates that:
- All submodule direct imports work correctly
- Re-exports in __init__.py files are complete and functional
- The public API surface from rubric_kit is preserved
- New prompt submodules (config, evaluation, generation, refinement, tool_calls)
  can be imported individually
- New package structure (core/, io/, models/, cli/, prompts/, reports/) is intact
- Previously untested prompt builders (refine_with_qa, refine_with_chat,
  chat_dimension_generation) behave correctly
"""

import os
import tempfile

import pytest
import yaml


# =============================================================================
# Top-level rubric_kit public API re-exports
# =============================================================================


class TestTopLevelReExports:
    """Verify that rubric_kit.__init__ re-exports all expected names from new paths."""

    def test_api_functions_importable(self):
        """Test that all API functions are importable from rubric_kit."""
        from rubric_kit import (
            dry_run_evaluate,
            evaluate,
            export,
            generate,
            refine,
        )

        assert callable(evaluate)
        assert callable(generate)
        assert callable(refine)
        assert callable(export)
        assert callable(dry_run_evaluate)

    def test_result_types_importable(self):
        """Test that all result types are importable from rubric_kit."""
        from rubric_kit import (
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
        )

        # All should be importable classes
        for cls in [
            EvaluationResult,
            GenerationResult,
            RefinementResult,
            ArenaResult,
            DryRunResult,
            ExportResult,
            CriterionResult,
            ScoreSummary,
            ContestantResult,
            ArenaRanking,
        ]:
            assert cls is not None

    def test_domain_models_importable(self):
        """Test that domain models are importable from rubric_kit."""
        from rubric_kit import (
            Criterion,
            Dimension,
        )

        # Verify these are the correct classes by instantiating
        dim = Dimension(name="test", description="Test dim", grading_type="binary")
        assert dim.name == "test"

        crit = Criterion(
            name="test_c", category="Test", weight=1, dimension="test", criterion="Test"
        )
        assert crit.name == "test_c"

    def test_exception_importable(self):
        """Test that RubricValidationError is importable from rubric_kit."""
        from rubric_kit import RubricValidationError

        assert issubclass(RubricValidationError, Exception)

    def test_version_is_set(self):
        """Test that __version__ is accessible."""
        import rubric_kit

        assert hasattr(rubric_kit, "__version__")
        assert isinstance(rubric_kit.__version__, str)
        assert len(rubric_kit.__version__) > 0

    def test_all_exports_match_declared(self):
        """Test that all names in __all__ are actually importable."""
        import rubric_kit

        for name in rubric_kit.__all__:
            assert hasattr(rubric_kit, name), f"{name} declared in __all__ but not importable"


# =============================================================================
# Prompts package re-exports
# =============================================================================


class TestPromptsReExports:
    """Verify that rubric_kit.prompts re-exports match submodule contents."""

    def test_all_config_names_reexported(self):
        """Test that all config module names are re-exported from prompts."""
        from rubric_kit.prompts import (
            EVALUATOR_CONFIG,
            EVALUATOR_SYSTEM_PROMPT,
            GENERATOR_CONFIG,
            GENERATOR_SYSTEM_PROMPT,
            TOOL_CALL_EVALUATOR_CONFIG,
            LLMConfig,
        )
        from rubric_kit.prompts.config import (
            EVALUATOR_CONFIG as EC_DIRECT,
        )
        from rubric_kit.prompts.config import (
            EVALUATOR_SYSTEM_PROMPT as ESP_DIRECT,
        )
        from rubric_kit.prompts.config import (
            GENERATOR_CONFIG as GC_DIRECT,
        )
        from rubric_kit.prompts.config import (
            GENERATOR_SYSTEM_PROMPT as GSP_DIRECT,
        )
        from rubric_kit.prompts.config import (
            TOOL_CALL_EVALUATOR_CONFIG as TCEC_DIRECT,
        )
        from rubric_kit.prompts.config import (
            LLMConfig as LC_DIRECT,
        )

        # Re-exports should be the same objects
        assert EVALUATOR_CONFIG is EC_DIRECT
        assert EVALUATOR_SYSTEM_PROMPT is ESP_DIRECT
        assert GENERATOR_CONFIG is GC_DIRECT
        assert GENERATOR_SYSTEM_PROMPT is GSP_DIRECT
        assert TOOL_CALL_EVALUATOR_CONFIG is TCEC_DIRECT
        assert LLMConfig is LC_DIRECT

    def test_all_evaluation_names_reexported(self):
        """Test that all evaluation module names are re-exported from prompts."""
        from rubric_kit.prompts import (
            build_binary_criterion_prompt,
            build_score_criterion_prompt,
            build_tool_call_evaluation_prompt,
        )
        from rubric_kit.prompts.evaluation import (
            build_binary_criterion_prompt as BIN_DIRECT,
        )
        from rubric_kit.prompts.evaluation import (
            build_score_criterion_prompt as SCORE_DIRECT,
        )
        from rubric_kit.prompts.evaluation import (
            build_tool_call_evaluation_prompt as TC_DIRECT,
        )

        assert build_binary_criterion_prompt is BIN_DIRECT
        assert build_score_criterion_prompt is SCORE_DIRECT
        assert build_tool_call_evaluation_prompt is TC_DIRECT

    def test_all_generation_names_reexported(self):
        """Test that all generation module names are re-exported from prompts."""
        from rubric_kit.prompts import (
            build_chat_criteria_generation_prompt,
            build_chat_dimension_generation_prompt,
            build_criteria_generation_prompt,
            build_dimension_generation_prompt,
        )
        from rubric_kit.prompts.generation import (
            build_chat_criteria_generation_prompt as CCGP_DIRECT,
        )
        from rubric_kit.prompts.generation import (
            build_chat_dimension_generation_prompt as CDGP_DIRECT,
        )
        from rubric_kit.prompts.generation import (
            build_criteria_generation_prompt as CGP_DIRECT,
        )
        from rubric_kit.prompts.generation import (
            build_dimension_generation_prompt as DGP_DIRECT,
        )

        assert build_dimension_generation_prompt is DGP_DIRECT
        assert build_criteria_generation_prompt is CGP_DIRECT
        assert build_chat_dimension_generation_prompt is CDGP_DIRECT
        assert build_chat_criteria_generation_prompt is CCGP_DIRECT

    def test_all_refinement_names_reexported(self):
        """Test that all refinement module names are re-exported from prompts."""
        from rubric_kit.prompts import (
            build_refine_rubric_prompt,
            build_refine_rubric_with_chat_prompt,
            build_refine_rubric_with_qa_prompt,
        )
        from rubric_kit.prompts.refinement import (
            build_refine_rubric_prompt as RRP_DIRECT,
        )
        from rubric_kit.prompts.refinement import (
            build_refine_rubric_with_chat_prompt as RRCP_DIRECT,
        )
        from rubric_kit.prompts.refinement import (
            build_refine_rubric_with_qa_prompt as RRQP_DIRECT,
        )

        assert build_refine_rubric_prompt is RRP_DIRECT
        assert build_refine_rubric_with_qa_prompt is RRQP_DIRECT
        assert build_refine_rubric_with_chat_prompt is RRCP_DIRECT

    def test_prompts_all_list_is_complete(self):
        """Test that prompts __all__ contains all expected names."""
        import rubric_kit.prompts as prompts_pkg

        expected_names = [
            "LLMConfig",
            "EVALUATOR_SYSTEM_PROMPT",
            "GENERATOR_SYSTEM_PROMPT",
            "EVALUATOR_CONFIG",
            "TOOL_CALL_EVALUATOR_CONFIG",
            "GENERATOR_CONFIG",
            "build_binary_criterion_prompt",
            "build_score_criterion_prompt",
            "build_tool_call_evaluation_prompt",
            "build_dimension_generation_prompt",
            "build_criteria_generation_prompt",
            "build_chat_dimension_generation_prompt",
            "build_chat_criteria_generation_prompt",
            "build_refine_rubric_prompt",
            "build_refine_rubric_with_qa_prompt",
            "build_refine_rubric_with_chat_prompt",
        ]

        for name in expected_names:
            assert name in prompts_pkg.__all__, f"{name} missing from prompts.__all__"
            assert hasattr(prompts_pkg, name), f"{name} in __all__ but not importable"


# =============================================================================
# Direct submodule imports
# =============================================================================


class TestDirectSubmoduleImports:
    """Test that new subpackages and their modules are directly importable."""

    def test_core_subpackage_importable(self):
        """Test that core subpackage modules are importable."""
        import rubric_kit.core
        import rubric_kit.core.consensus
        import rubric_kit.core.execution
        import rubric_kit.core.llm_judge
        import rubric_kit.core.processor
        import rubric_kit.core.tool_evaluator

        assert rubric_kit.core is not None

    def test_io_subpackage_importable(self):
        """Test that io subpackage modules are importable."""
        import rubric_kit.io
        import rubric_kit.io.output
        import rubric_kit.io.parser
        import rubric_kit.io.validator

        assert rubric_kit.io is not None

    def test_models_subpackage_importable(self):
        """Test that models subpackage modules are importable."""
        import rubric_kit.models
        import rubric_kit.models.converters
        import rubric_kit.models.schema

        assert rubric_kit.models is not None

    def test_cli_subpackage_importable(self):
        """Test that cli subpackage modules are importable."""
        import rubric_kit.cli
        import rubric_kit.cli.commands
        import rubric_kit.cli.parser

        assert rubric_kit.cli is not None

    def test_prompts_subpackage_importable(self):
        """Test that prompts subpackage modules are importable."""
        import rubric_kit.prompts
        import rubric_kit.prompts.config
        import rubric_kit.prompts.evaluation
        import rubric_kit.prompts.generation
        import rubric_kit.prompts.refinement
        import rubric_kit.prompts.tool_calls

        assert rubric_kit.prompts is not None

    def test_reports_subpackage_importable(self):
        """Test that reports subpackage modules are importable."""
        import rubric_kit.reports
        import rubric_kit.reports.pdf_arena
        import rubric_kit.reports.pdf_base
        import rubric_kit.reports.pdf_evaluation

        assert rubric_kit.reports is not None

    def test_dunder_main_uses_cli_commands(self):
        """Test that __main__.py imports from cli.commands."""
        # Verify the import path works without actually running main()
        from rubric_kit.cli.commands import main

        assert callable(main)


# =============================================================================
# Prompt builder tests for previously untested functions
# =============================================================================


class TestRefineRubricWithQaPrompt:
    """Tests for build_refine_rubric_with_qa_prompt (previously untested)."""

    @pytest.fixture
    def sample_dimensions(self):
        from rubric_kit.models.schema import Dimension

        return [
            Dimension(name="accuracy", description="Factual accuracy", grading_type="binary"),
            Dimension(
                name="completeness",
                description="Answer completeness",
                grading_type="score",
                scores={0: "None", 1: "Some", 2: "Most", 3: "All"},
            ),
        ]

    @pytest.fixture
    def sample_criteria(self):
        from rubric_kit.models.schema import Criterion

        return [
            Criterion(
                name="fact_check",
                category="Accuracy",
                weight=3,
                dimension="accuracy",
                criterion="Must be factual",
            ),
            Criterion(
                name="coverage",
                category="Completeness",
                weight="from_scores",
                dimension="completeness",
                criterion="from_scores",
            ),
        ]

    def test_basic_qa_refinement_prompt(self, sample_dimensions, sample_criteria):
        """Test that QA refinement prompt includes question, answer, and rubric."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_qa_prompt

        prompt = build_refine_rubric_with_qa_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            question="What is the capital of France?",
            answer="Paris",
        )

        assert "What is the capital of France?" in prompt
        assert "Paris" in prompt
        assert "accuracy" in prompt
        assert "fact_check" in prompt
        assert "refine" in prompt.lower()
        assert "Q&A" in prompt

    def test_qa_refinement_with_feedback(self, sample_dimensions, sample_criteria):
        """Test QA refinement prompt includes specific feedback."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_qa_prompt

        prompt = build_refine_rubric_with_qa_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            question="Q",
            answer="A",
            feedback="Add more specific criteria for geography facts",
        )

        assert "Add more specific criteria for geography facts" in prompt

    def test_qa_refinement_with_context(self, sample_dimensions, sample_criteria):
        """Test QA refinement prompt includes additional context."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_qa_prompt

        prompt = build_refine_rubric_with_qa_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            question="Q",
            answer="A",
            context="This is a geography quiz for 5th graders",
        )

        assert "geography quiz for 5th graders" in prompt

    def test_qa_refinement_with_variables(self, sample_dimensions, sample_criteria):
        """Test QA refinement prompt includes variables when provided."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_qa_prompt

        prompt = build_refine_rubric_with_qa_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            question="Q",
            answer="A",
            variables={"capital": "Paris"},
        )

        assert "capital" in prompt
        assert "Paris" in prompt

    def test_qa_refinement_no_variables_mode(self, sample_dimensions, sample_criteria):
        """Test QA refinement prompt excludes variables when use_variables=False."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_qa_prompt

        prompt = build_refine_rubric_with_qa_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            question="Q",
            answer="A",
            use_variables=False,
        )

        assert "{{variable_name}}" not in prompt
        assert (
            "hard-coded" in prompt.lower()
            or "hardcoded" in prompt.lower()
            or "hardcode" in prompt.lower()
        )


class TestRefineRubricWithChatPrompt:
    """Tests for build_refine_rubric_with_chat_prompt (previously untested)."""

    @pytest.fixture
    def sample_dimensions(self):
        from rubric_kit.models.schema import Dimension

        return [
            Dimension(name="tool_use", description="Tool usage correctness", grading_type="binary"),
        ]

    @pytest.fixture
    def sample_criteria(self):
        from rubric_kit.models.schema import Criterion

        return [
            Criterion(
                name="tools_called",
                category="Tools",
                weight=3,
                dimension="tool_use",
                criterion="Essential tools must be called",
            ),
        ]

    def test_basic_chat_refinement_prompt(self, sample_dimensions, sample_criteria):
        """Test chat refinement prompt includes chat content and rubric."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_chat_prompt

        chat_content = "User: Show system info\nAssistant: Here is the system information..."

        prompt = build_refine_rubric_with_chat_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            chat_content=chat_content,
        )

        assert chat_content in prompt
        assert "tool_use" in prompt
        assert "tools_called" in prompt
        assert "refine" in prompt.lower()
        assert "chat session" in prompt.lower()

    def test_chat_refinement_with_feedback(self, sample_dimensions, sample_criteria):
        """Test chat refinement prompt includes specific feedback."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_chat_prompt

        prompt = build_refine_rubric_with_chat_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            chat_content="User: test\nAssistant: response",
            feedback="Make tool criteria more granular",
        )

        assert "Make tool criteria more granular" in prompt

    def test_chat_refinement_with_context(self, sample_dimensions, sample_criteria):
        """Test chat refinement prompt includes additional context."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_chat_prompt

        prompt = build_refine_rubric_with_chat_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            chat_content="User: test\nAssistant: response",
            context="This is a system administration task",
        )

        assert "system administration task" in prompt

    def test_chat_refinement_no_variables_mode(self, sample_dimensions, sample_criteria):
        """Test chat refinement prompt excludes variables when use_variables=False."""
        from rubric_kit.prompts.refinement import build_refine_rubric_with_chat_prompt

        prompt = build_refine_rubric_with_chat_prompt(
            dimensions=sample_dimensions,
            criteria=sample_criteria,
            chat_content="User: test\nAssistant: response",
            use_variables=False,
        )

        assert "{{variable_name}}" not in prompt
        assert (
            "hard-coded" in prompt.lower()
            or "hardcoded" in prompt.lower()
            or "hardcode" in prompt.lower()
        )


class TestChatDimensionGenerationPrompt:
    """Tests for build_chat_dimension_generation_prompt basic behavior (partially untested)."""

    def test_basic_chat_dimension_prompt(self):
        """Test basic chat dimension generation prompt."""
        from rubric_kit.prompts.generation import build_chat_dimension_generation_prompt

        chat_content = "User: What tools do we have?\nAssistant: We have get_info and run_check."

        prompt = build_chat_dimension_generation_prompt(
            chat_content=chat_content,
            num_dimensions=4,
        )

        assert chat_content in prompt
        assert "4" in prompt
        assert "dimension" in prompt.lower()
        assert "JSON" in prompt

    def test_chat_dimension_prompt_auto_count(self):
        """Test chat dimension prompt without specifying num_dimensions."""
        from rubric_kit.prompts.generation import build_chat_dimension_generation_prompt

        prompt = build_chat_dimension_generation_prompt(
            chat_content="User: Hello\nAssistant: Hi",
            num_dimensions=None,
        )

        # Should mention a range instead of a specific number
        assert "between" in prompt.lower() or "appropriate" in prompt.lower()

    def test_chat_dimension_prompt_with_context(self):
        """Test chat dimension prompt includes context when provided."""
        from rubric_kit.prompts.generation import build_chat_dimension_generation_prompt

        prompt = build_chat_dimension_generation_prompt(
            chat_content="User: test\nAssistant: response",
            num_dimensions=3,
            context="System monitoring scenario",
        )

        assert "System monitoring scenario" in prompt


# =============================================================================
# Reports submodule tests
# =============================================================================


class TestReportsSubmodule:
    """Test that reports submodule functions are accessible."""

    def test_export_evaluation_pdf_importable(self):
        """Test that export_evaluation_pdf is importable from reports."""
        from rubric_kit.reports.pdf_evaluation import export_evaluation_pdf

        assert callable(export_evaluation_pdf)

    def test_export_arena_pdf_importable(self):
        """Test that export_arena_pdf is importable from reports."""
        from rubric_kit.reports.pdf_arena import export_arena_pdf

        assert callable(export_arena_pdf)

    def test_pdf_base_helpers_importable(self):
        """Test that shared pdf_base helpers are importable."""
        from rubric_kit.reports.pdf_base import (
            _load_evaluation_data,
        )

        assert callable(_load_evaluation_data)

    def test_export_arena_pdf_basic(self):
        """Test that arena PDF export works with valid input."""
        from rubric_kit.reports.pdf_arena import export_arena_pdf

        arena_data = {
            "mode": "arena",
            "arena_name": "Test Arena",
            "arena_description": "Test comparison",
            "contestants": {
                "model_a": {
                    "name": "Model A",
                    "description": "Test model A",
                    "metadata": {"version": "1.0"},
                    "input": {"type": "chat_session", "source_file": "a.txt"},
                    "results": [
                        {
                            "criterion_name": "fact_1",
                            "criterion_text": "Check fact",
                            "category": "Output",
                            "dimension": "accuracy",
                            "result": "pass",
                            "score": 3,
                            "max_score": 3,
                            "reason": "Correct",
                        }
                    ],
                    "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
                },
            },
            "rankings": [
                {
                    "rank": 1,
                    "id": "model_a",
                    "name": "Model A",
                    "percentage": 100.0,
                    "total_score": 3,
                    "max_score": 3,
                },
            ],
            "rubric": {
                "dimensions": [
                    {"name": "accuracy", "description": "Test", "grading_type": "binary"}
                ],
                "criteria": [{"name": "fact_1", "dimension": "accuracy", "weight": 3}],
            },
            "judge_panel": {
                "judges": [{"name": "test", "model": "gpt-4"}],
                "execution": {"mode": "sequential"},
                "consensus": {"mode": "unanimous"},
            },
            "metadata": {"timestamp": "2025-01-01T00:00:00"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name
            yaml.dump(arena_data, f)

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            export_arena_pdf(yaml_path, pdf_path)

            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)


# =============================================================================
# Tool calls helpers (prompts.tool_calls) tests
# =============================================================================


class TestToolCallsHelpers:
    """Test individual helper functions in prompts.tool_calls module."""

    def test_format_tool_constraints_with_min_max(self):
        """Test formatting tool constraints with min and max."""
        from rubric_kit.models.schema import ToolSpec
        from rubric_kit.prompts.tool_calls import _format_tool_constraints

        tool = ToolSpec(name="test", min_calls=1, max_calls=3, params=None)
        result = _format_tool_constraints(tool)

        assert "min: 1" in result
        assert "max: 3" in result

    def test_format_tool_constraints_no_limits(self):
        """Test formatting tool constraints with no limits."""
        from rubric_kit.models.schema import ToolSpec
        from rubric_kit.prompts.tool_calls import _format_tool_constraints

        tool = ToolSpec(name="test", params=None)
        result = _format_tool_constraints(tool)

        assert result == ""

    def test_format_tool_params_none(self):
        """Test formatting tool params when None (no validation)."""
        from rubric_kit.models.schema import ToolSpec
        from rubric_kit.prompts.tool_calls import _format_tool_params

        tool = ToolSpec(name="test", params=None)
        result = _format_tool_params(tool)

        assert result == ""

    def test_format_tool_params_empty_dict(self):
        """Test formatting tool params when empty dict (no params allowed)."""
        from rubric_kit.models.schema import ToolSpec
        from rubric_kit.prompts.tool_calls import _format_tool_params

        tool = ToolSpec(name="test", params={})
        result = _format_tool_params(tool)

        assert "NO parameters" in result

    def test_format_tool_params_with_values(self):
        """Test formatting tool params with specified values."""
        from rubric_kit.models.schema import ToolSpec
        from rubric_kit.prompts.tool_calls import _format_tool_params

        tool = ToolSpec(name="test", params={"host": "example.com", "port": 8080})
        result = _format_tool_params(tool)

        assert "host" in result
        assert "example.com" in result
        assert "port" in result
        assert "8080" in result

    def test_build_required_tools_section_empty(self):
        """Test building required tools section with no required tools."""
        from rubric_kit.models.schema import ToolCalls
        from rubric_kit.prompts.tool_calls import _build_required_tools_section

        tc = ToolCalls(required=[], optional=[], prohibited=[])
        result = _build_required_tools_section(tc)

        assert result == ""

    def test_build_required_tools_section_with_tools(self):
        """Test building required tools section with tools."""
        from rubric_kit.models.schema import ToolCalls, ToolSpec
        from rubric_kit.prompts.tool_calls import _build_required_tools_section

        tc = ToolCalls(
            required=[ToolSpec(name="get_info", min_calls=1, params=None)],
            optional=[],
            prohibited=[],
        )
        result = _build_required_tools_section(tc)

        assert "Required Tools" in result
        assert "get_info" in result

    def test_build_optional_tools_section(self):
        """Test building optional tools section."""
        from rubric_kit.models.schema import ToolCalls, ToolSpec
        from rubric_kit.prompts.tool_calls import _build_optional_tools_section

        tc = ToolCalls(
            required=[],
            optional=[ToolSpec(name="bonus_tool", max_calls=2, params=None)],
            prohibited=[],
        )
        result = _build_optional_tools_section(tc)

        assert "Optional Tools" in result
        assert "bonus_tool" in result

    def test_build_prohibited_tools_section(self):
        """Test building prohibited tools section."""
        from rubric_kit.models.schema import ToolCalls, ToolSpec
        from rubric_kit.prompts.tool_calls import _build_prohibited_tools_section

        tc = ToolCalls(
            required=[],
            optional=[],
            prohibited=[ToolSpec(name="dangerous_tool", params=None)],
        )
        result = _build_prohibited_tools_section(tc)

        assert "Prohibited Tools" in result
        assert "dangerous_tool" in result

    def test_build_required_tool_lists(self):
        """Test building various formats of required tool lists."""
        from rubric_kit.models.schema import ToolCalls, ToolSpec
        from rubric_kit.prompts.tool_calls import _build_required_tool_lists

        tc = ToolCalls(
            required=[
                ToolSpec(name="tool_a", params=None),
                ToolSpec(name="tool_b", params=None),
            ],
            optional=[],
            prohibited=[],
        )
        numbered, labeled, comma_sep, bullets = _build_required_tool_lists(tc)

        assert "1. tool_a" in numbered
        assert "2. tool_b" in numbered
        assert "REQUIRED TOOL #1: tool_a" in labeled
        assert "tool_a, tool_b" in comma_sep
        assert "tool_a" in bullets
        assert "tool_b" in bullets


# =============================================================================
# Refinement helpers (prompts.refinement) tests
# =============================================================================


class TestRefinementHelpers:
    """Test helper functions in prompts.refinement module."""

    def test_rubric_to_yaml_basic(self):
        """Test converting rubric to YAML string."""
        from rubric_kit.models.schema import Criterion, Dimension
        from rubric_kit.prompts.refinement import _rubric_to_yaml

        dims = [Dimension(name="accuracy", description="Factual accuracy", grading_type="binary")]
        crits = [
            Criterion(
                name="check_fact",
                category="Accuracy",
                weight=3,
                dimension="accuracy",
                criterion="Must be factual",
            )
        ]

        result = _rubric_to_yaml(dims, crits)

        # Should be valid YAML
        parsed = yaml.safe_load(result)
        assert "dimensions" in parsed
        assert "criteria" in parsed
        assert parsed["dimensions"][0]["name"] == "accuracy"

    def test_rubric_to_yaml_with_variables(self):
        """Test converting rubric to YAML with variables included."""
        from rubric_kit.models.schema import Criterion, Dimension
        from rubric_kit.prompts.refinement import _rubric_to_yaml

        dims = [Dimension(name="test", description="Test", grading_type="binary")]
        crits = [
            Criterion(
                name="test_c",
                category="Test",
                weight=1,
                dimension="test",
                criterion="Check {{val}}",
            )
        ]
        variables = {"val": "42"}

        result = _rubric_to_yaml(dims, crits, variables)

        parsed = yaml.safe_load(result)
        assert "variables" in parsed
        assert parsed["variables"]["val"] == "42"

    def test_build_tool_calls_instruction_with_tool_calls(self):
        """Test tool calls instruction when criteria have tool_calls."""
        from rubric_kit.models.schema import Criterion, ToolCalls, ToolSpec
        from rubric_kit.prompts.refinement import _build_tool_calls_instruction

        criteria = [
            Criterion(
                name="tool_test",
                category="Tools",
                weight=3,
                dimension="tool_use",
                criterion="Must call tools",
                tool_calls=ToolCalls(
                    required=[ToolSpec(name="get_info", params=None)],
                    optional=[],
                    prohibited=[],
                ),
            )
        ]

        result = _build_tool_calls_instruction(criteria)

        assert "tool_calls" in result.lower() or "Tool Calls" in result

    def test_build_tool_calls_instruction_without_tool_calls(self):
        """Test tool calls instruction when no criteria have tool_calls."""
        from rubric_kit.models.schema import Criterion
        from rubric_kit.prompts.refinement import _build_tool_calls_instruction

        criteria = [
            Criterion(
                name="fact_check",
                category="Accuracy",
                weight=3,
                dimension="accuracy",
                criterion="Must be factual",
            )
        ]

        result = _build_tool_calls_instruction(criteria)

        assert result == ""

    def test_build_default_feedback_no_context(self):
        """Test default feedback without context type."""
        from rubric_kit.prompts.refinement import _build_default_feedback

        result = _build_default_feedback()

        assert "improve" in result.lower() or "improving" in result.lower()

    def test_build_default_feedback_qa_context(self):
        """Test default feedback for QA context."""
        from rubric_kit.prompts.refinement import _build_default_feedback

        result = _build_default_feedback("qa")

        assert "Q&A" in result or "answer" in result.lower()

    def test_build_default_feedback_chat_context(self):
        """Test default feedback for chat context."""
        from rubric_kit.prompts.refinement import _build_default_feedback

        result = _build_default_feedback("chat")

        assert "chat" in result.lower()


# =============================================================================
# Generation helpers (prompts.generation) tests
# =============================================================================


class TestGenerationHelpers:
    """Test helper functions in prompts.generation module."""

    def test_convert_criterion_to_dict_basic(self):
        """Test converting a basic criterion to dict for YAML."""
        from rubric_kit.models.schema import Criterion
        from rubric_kit.prompts.generation import _convert_criterion_to_dict_for_yaml

        crit = Criterion(
            name="fact_check",
            category="Accuracy",
            weight=3,
            dimension="accuracy",
            criterion="Must be factual",
        )

        result = _convert_criterion_to_dict_for_yaml(crit)

        assert result["name"] == "fact_check"
        assert result["category"] == "Accuracy"
        assert result["weight"] == 3
        assert result["dimension"] == "accuracy"
        assert result["criterion"] == "Must be factual"
        assert "tool_calls" not in result

    def test_convert_criterion_to_dict_with_tool_calls(self):
        """Test converting a criterion with tool_calls to dict."""
        from rubric_kit.models.schema import Criterion, ToolCalls, ToolSpec
        from rubric_kit.prompts.generation import _convert_criterion_to_dict_for_yaml

        tool_calls = ToolCalls(
            respect_order=True,
            required=[ToolSpec(name="get_info", min_calls=1, max_calls=2, params={"host": "srv"})],
            optional=[ToolSpec(name="bonus", params=None)],
            prohibited=[ToolSpec(name="bad_tool", params=None)],
        )

        crit = Criterion(
            name="tool_test",
            category="Tools",
            weight=3,
            dimension="tool_use",
            criterion="Must use tools",
            tool_calls=tool_calls,
        )

        result = _convert_criterion_to_dict_for_yaml(crit)

        assert "tool_calls" in result
        assert result["tool_calls"]["respect_order"] is True
        assert len(result["tool_calls"]["required"]) == 1
        assert len(result["tool_calls"]["optional"]) == 1
        assert len(result["tool_calls"]["prohibited"]) == 1
