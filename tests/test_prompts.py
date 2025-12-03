"""Tests for prompt templates module."""

import pytest
from rubric_kit.schema import Criterion, Dimension
from rubric_kit.prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    LLMConfig,
    EVALUATOR_CONFIG,
    TOOL_CALL_EVALUATOR_CONFIG,
    GENERATOR_CONFIG,
    build_binary_criterion_prompt,
    build_score_criterion_prompt,
    build_tool_call_evaluation_prompt,
    build_dimension_generation_prompt,
    build_criteria_generation_prompt,
    build_refine_rubric_prompt,
)


def test_evaluator_system_prompt_exists():
    """Test that evaluator system prompt is defined."""
    assert EVALUATOR_SYSTEM_PROMPT
    assert isinstance(EVALUATOR_SYSTEM_PROMPT, str)
    assert len(EVALUATOR_SYSTEM_PROMPT) > 0


def test_generator_system_prompt_exists():
    """Test that generator system prompt is defined."""
    assert GENERATOR_SYSTEM_PROMPT
    assert isinstance(GENERATOR_SYSTEM_PROMPT, str)
    assert len(GENERATOR_SYSTEM_PROMPT) > 0


# =============================================================================
# LLM Configuration Tests
# =============================================================================

def test_llm_config_dataclass():
    """Test LLMConfig dataclass creation."""
    config = LLMConfig(
        system_prompt="Test prompt",
        temperature=0.5,
        max_tokens=100
    )
    
    assert config.system_prompt == "Test prompt"
    assert config.temperature == 0.5
    assert config.max_tokens == 100


def test_llm_config_validation():
    """Test LLMConfig validates types."""
    # Should work with valid types
    config = LLMConfig(
        system_prompt="Valid",
        temperature=0.7,
        max_tokens=200
    )
    assert config is not None


def test_evaluator_config_exists():
    """Test that EVALUATOR_CONFIG is defined correctly."""
    assert EVALUATOR_CONFIG
    assert isinstance(EVALUATOR_CONFIG, LLMConfig)
    assert EVALUATOR_CONFIG.system_prompt == EVALUATOR_SYSTEM_PROMPT
    assert isinstance(EVALUATOR_CONFIG.temperature, float)
    assert isinstance(EVALUATOR_CONFIG.max_tokens, int)
    # Evaluator should be deterministic (low temperature)
    assert EVALUATOR_CONFIG.temperature <= 0.1


def test_generator_config_exists():
    """Test that GENERATOR_CONFIG is defined correctly."""
    assert GENERATOR_CONFIG
    assert isinstance(GENERATOR_CONFIG, LLMConfig)
    assert GENERATOR_CONFIG.system_prompt == GENERATOR_SYSTEM_PROMPT
    assert isinstance(GENERATOR_CONFIG.temperature, float)
    assert isinstance(GENERATOR_CONFIG.max_tokens, int)
    # Generator should be more creative (higher temperature)
    assert GENERATOR_CONFIG.temperature >= 0.5


def test_configs_are_different():
    """Test that evaluator and generator configs have different characteristics."""
    # Should have different temperatures for different personas
    assert EVALUATOR_CONFIG.temperature != GENERATOR_CONFIG.temperature
    # Should have different system prompts
    assert EVALUATOR_CONFIG.system_prompt != GENERATOR_CONFIG.system_prompt


def test_tool_call_evaluator_config_exists():
    """Test that TOOL_CALL_EVALUATOR_CONFIG is defined for complex evaluations."""
    assert TOOL_CALL_EVALUATOR_CONFIG
    assert isinstance(TOOL_CALL_EVALUATOR_CONFIG, LLMConfig)
    assert TOOL_CALL_EVALUATOR_CONFIG.system_prompt == EVALUATOR_SYSTEM_PROMPT
    assert isinstance(TOOL_CALL_EVALUATOR_CONFIG.temperature, float)
    assert isinstance(TOOL_CALL_EVALUATOR_CONFIG.max_tokens, int)
    # Should be deterministic like regular evaluator
    assert TOOL_CALL_EVALUATOR_CONFIG.temperature <= 0.1
    # Should have MORE tokens than regular evaluator for complex analysis
    assert TOOL_CALL_EVALUATOR_CONFIG.max_tokens > EVALUATOR_CONFIG.max_tokens


def test_build_binary_criterion_prompt():
    """Test building a binary criterion evaluation prompt."""
    criterion = Criterion(
        name="test_criterion",
        category="Test",
        weight=2,
        dimension="test_dimension",
        criterion="Must be correct"
    )
    
    chat_content = "User: Hello\nAssistant: Hi there"
    
    prompt = build_binary_criterion_prompt(criterion, chat_content)
    
    # Check essential components are present
    assert "test_dimension" in prompt
    assert "Test" in prompt
    assert "Must be correct" in prompt
    assert chat_content in prompt
    assert "RESULT:" in prompt
    assert "REASON:" in prompt
    assert "PASS or FAIL" in prompt
    
    # Check for strict evaluation instructions
    assert "EXPLICITLY" in prompt
    assert "Do NOT make inferences" in prompt
    assert "Do NOT consider related but different information" in prompt


def test_build_score_criterion_prompt():
    """Test building a score-based criterion evaluation prompt."""
    dimension = Dimension(
        name="completeness",
        description="How complete the answer is",
        grading_type="score",
        scores={1: "Incomplete", 2: "Partial", 3: "Complete"}
    )
    
    criterion = Criterion(
        name="test_score",
        category="Quality",
        weight="from_scores",
        dimension="completeness",
        criterion="from_scores"
    )
    
    chat_content = "User: Question\nAssistant: Answer"
    
    prompt = build_score_criterion_prompt(criterion, chat_content, dimension)
    
    # Check essential components
    assert "completeness" in prompt
    assert "Quality" in prompt
    assert chat_content in prompt
    assert "SCORE:" in prompt
    assert "REASON:" in prompt
    assert "Incomplete" in prompt
    assert "Partial" in prompt
    assert "Complete" in prompt
    assert "1:" in prompt or "1 :" in prompt  # Score descriptions


def test_build_score_criterion_prompt_requires_dimension():
    """Test that score prompt requires dimension with scores."""
    criterion = Criterion(
        name="test",
        category="Test",
        weight="from_scores",
        dimension="test_dim",
        criterion="from_scores"
    )
    
    # Dimension with empty scores dict (to bypass schema validation)
    dimension = Dimension(
        name="test_dim",
        description="Test",
        grading_type="binary"  # Use binary so scores aren't required
    )
    dimension.scores = None  # Manually set to None
    
    with pytest.raises(ValueError, match="scores"):
        build_score_criterion_prompt(criterion, "content", dimension)


def test_build_dimension_generation_prompt():
    """Test building dimension generation prompt."""
    question = "What is the capital of France?"
    answer = "Paris"
    num_dimensions = 3
    
    prompt = build_dimension_generation_prompt(
        question=question,
        answer=answer,
        num_dimensions=num_dimensions
    )
    
    # Check essential components
    assert question in prompt
    assert answer in prompt
    assert str(num_dimensions) in prompt
    assert "dimension" in prompt.lower()
    assert "JSON" in prompt


def test_build_dimension_generation_prompt_with_context():
    """Test dimension prompt includes context when provided."""
    prompt = build_dimension_generation_prompt(
        question="Q",
        answer="A",
        num_dimensions=3,
        context="Additional info"
    )
    
    assert "Additional info" in prompt
    assert "context" in prompt.lower()


def test_build_criteria_generation_prompt():
    """Test building criteria generation prompt."""
    question = "What is 2+2?"
    answer = "4"
    dimensions = [
        Dimension(
            name="correctness",
            description="Is it correct?",
            grading_type="binary"
        ),
        Dimension(
            name="completeness",
            description="Is it complete?",
            grading_type="score",
            scores={1: "No", 2: "Yes"}
        )
    ]
    num_criteria = 5
    
    prompt = build_criteria_generation_prompt(
        question=question,
        answer=answer,
        dimensions=dimensions,
        num_criteria=num_criteria
    )
    
    # Check essential components
    assert question in prompt
    assert answer in prompt
    assert str(num_criteria) in prompt
    assert "correctness" in prompt
    assert "completeness" in prompt
    assert "criterion" in prompt.lower()
    assert "JSON" in prompt


def test_build_criteria_generation_prompt_with_category_hints():
    """Test criteria prompt includes category hints."""
    dimensions = [
        Dimension(name="test", description="Test", grading_type="binary")
    ]
    
    prompt = build_criteria_generation_prompt(
        question="Q",
        answer="A",
        dimensions=dimensions,
        num_criteria=3,
        category_hints=["Output", "Accuracy"]
    )
    
    assert "Output" in prompt
    assert "Accuracy" in prompt


def test_build_criteria_generation_prompt_with_context():
    """Test criteria prompt includes context when provided."""
    dimensions = [
        Dimension(name="test", description="Test", grading_type="binary")
    ]
    
    prompt = build_criteria_generation_prompt(
        question="Q",
        answer="A",
        dimensions=dimensions,
        num_criteria=3,
        context="Extra context"
    )
    
    assert "Extra context" in prompt


def test_build_refine_rubric_prompt_basic():
    """Test building rubric refinement prompt without feedback."""
    dimensions = [
        Dimension(
            name="accuracy",
            description="Factual accuracy",
            grading_type="binary"
        )
    ]
    
    criteria = [
        Criterion(
            name="fact_check",
            category="Accuracy",
            weight=3,
            dimension="accuracy",
            criterion="Must be factual"
        )
    ]
    
    prompt = build_refine_rubric_prompt(dimensions, criteria)
    
    # Check essential components
    assert "accuracy" in prompt
    assert "fact_check" in prompt
    assert "refine" in prompt.lower()
    assert "JSON" in prompt


def test_build_refine_rubric_prompt_with_feedback():
    """Test building rubric refinement prompt with feedback."""
    dimensions = [
        Dimension(name="test", description="Test", grading_type="binary")
    ]
    
    criteria = [
        Criterion(
            name="test_crit",
            category="Test",
            weight=1,
            dimension="test",
            criterion="Test criterion"
        )
    ]
    
    feedback = "Make it more specific"
    
    prompt = build_refine_rubric_prompt(dimensions, criteria, feedback=feedback)
    
    assert feedback in prompt
    assert "test" in prompt
    assert "test_crit" in prompt


# =============================================================================
# Tool Call Evaluation Tests
# =============================================================================

def test_build_tool_call_evaluation_prompt_basic():
    """Test building tool call evaluation prompt with basic structure."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=True,
        required=[
            ToolSpec(name="get_system_info", min_calls=1, max_calls=1, params={})
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    chat_content = """Tool Call: get_system_info
Arguments: {}"""
    
    prompt = build_tool_call_evaluation_prompt(criterion, chat_content)
    
    # Check essential components
    assert "get_system_info" in prompt
    assert "tool call" in prompt.lower()
    assert chat_content in prompt
    assert "RESULT:" in prompt
    assert "REASON:" in prompt


def test_build_tool_call_evaluation_prompt_with_all_types():
    """Test tool call prompt includes required, optional, and prohibited tools."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=True,
        required=[
            ToolSpec(name="tool_a", min_calls=1, max_calls=2, params={}),
            ToolSpec(name="tool_b", min_calls=1, max_calls=1, params={})
        ],
        optional=[
            ToolSpec(name="tool_c", max_calls=1, params={})
        ],
        prohibited=[
            ToolSpec(name="bad_tool", params={})
        ]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    chat_content = "Some session content"
    prompt = build_tool_call_evaluation_prompt(criterion, chat_content)
    
    # Check all tool names are mentioned
    assert "tool_a" in prompt
    assert "tool_b" in prompt
    assert "tool_c" in prompt
    assert "bad_tool" in prompt
    
    # Check it mentions required/optional/prohibited
    assert "required" in prompt.lower()
    assert "optional" in prompt.lower()
    assert "prohibited" in prompt.lower()


def test_build_tool_call_evaluation_prompt_instructions():
    """Test that tool call prompt has specific parsing instructions."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=True,
        required=[
            ToolSpec(name="tool_first", min_calls=1, max_calls=1, params={}),
            ToolSpec(name="tool_second", min_calls=1, max_calls=1, params={})
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should have structured parsing instructions
    assert "parse" in prompt.lower() or "extract" in prompt.lower()
    assert "count" in prompt.lower()
    
    # Should mention checking order if respect_order is True
    assert "order" in prompt.lower()
    
    # Should show expected order explicitly
    assert "Expected order:" in prompt or "expected order" in prompt.lower()
    assert "1. tool_first" in prompt
    assert "2. tool_second" in prompt
    
    # Should have critical/explicit order checking instructions
    assert "MUST match" in prompt or "must match" in prompt.lower()
    assert "ACTUAL" in prompt or "actual" in prompt.lower()


def test_build_tool_call_evaluation_prompt_with_order_false():
    """Test that prompt adapts when order doesn't matter."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,  # Order doesn't matter
        required=[ToolSpec(name="test_tool", min_calls=1, max_calls=1, params={})],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should still mention order consideration
    assert "order" in prompt.lower() or "sequence" in prompt.lower()


def test_params_validation_none_no_check():
    """Test that params=None means no parameter validation."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,
        required=[
            ToolSpec(name="test_tool", min_calls=1, max_calls=1, params=None)
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should NOT include parameter checking instructions
    assert "Check parameters" not in prompt
    assert "Wrong or missing parameters" not in prompt
    # Tool should be listed without param requirements
    assert "test_tool" in prompt


def test_params_validation_empty_dict_check_no_params():
    """Test that params={} means check for NO parameters."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,
        required=[
            ToolSpec(name="test_tool", min_calls=1, max_calls=1, params={})
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should include parameter checking instructions
    assert "Check parameters" in prompt
    # Should mention that tool must be called with NO parameters
    assert "NO parameters" in prompt or "no parameters" in prompt.lower()
    assert "test_tool" in prompt
    # Should mention parameter failures
    assert "Wrong or missing parameters" in prompt


def test_params_validation_specified_params():
    """Test that params with values means check specified parameters."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,
        required=[
            ToolSpec(
                name="test_tool",
                min_calls=1,
                max_calls=1,
                params={"hostname": "example.com", "port": 8080}
            )
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should include parameter checking instructions
    assert "Check parameters" in prompt
    # Should show the specified parameters
    assert "hostname" in prompt
    assert "example.com" in prompt
    assert "port" in prompt
    assert "8080" in prompt
    # Should mention parameter failures
    assert "Wrong or missing parameters" in prompt


def test_params_strict_mode_false_allows_extra():
    """Test that params_strict_mode=False allows extra parameters."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,
        params_strict_mode=False,
        required=[
            ToolSpec(
                name="test_tool",
                min_calls=1,
                max_calls=1,
                params={"hostname": "example.com"}
            )
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should mention that extra parameters are OK
    assert "Extra parameters are OK" in prompt or "extra parameters" in prompt.lower()
    assert "STRICT MODE" not in prompt


def test_params_strict_mode_true_requires_exact():
    """Test that params_strict_mode=True requires exact parameter match."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,
        params_strict_mode=True,
        required=[
            ToolSpec(
                name="test_tool",
                min_calls=1,
                max_calls=1,
                params={"hostname": "example.com"}
            )
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # Should mention strict mode
    assert "STRICT MODE" in prompt
    # Should mention that extra parameters are NOT allowed
    assert "Extra parameters are NOT allowed" in prompt or "extra parameter" in prompt.lower()
    assert "exactly the specified params" in prompt.lower()


def test_params_mixed_validation_modes():
    """Test mixing different params validation modes in one criterion."""
    from rubric_kit.schema import ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=False,
        required=[
            ToolSpec(name="tool_no_validation", min_calls=1, max_calls=1, params=None),
            ToolSpec(name="tool_no_params", min_calls=1, max_calls=1, params={}),
            ToolSpec(
                name="tool_with_params",
                min_calls=1,
                max_calls=1,
                params={"key": "value"}
            )
        ],
        optional=[],
        prohibited=[]
    )
    
    criterion = Criterion(
        name="tool_test",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    prompt = build_tool_call_evaluation_prompt(criterion, "content")
    
    # All tools should be mentioned
    assert "tool_no_validation" in prompt
    assert "tool_no_params" in prompt
    assert "tool_with_params" in prompt
    
    # Should include parameter checking (because some tools have params requirements)
    assert "Check parameters" in prompt
    # Should mention NO parameters for tool_no_params
    assert "NO parameters" in prompt or "no parameters" in prompt.lower()
    # Should mention specified params for tool_with_params
    assert "key" in prompt
    assert "value" in prompt


# =============================================================================
# Variable Placeholder Tests - Regression for brace doubling bug
# =============================================================================

def test_refine_prompt_variable_placeholders_use_double_braces():
    """Test that refine prompts show {{var}} syntax, not {{{{var}}}} to the LLM.
    
    This is a regression test for a bug where the LLM was being told to use
    quadruple braces {{{{var}}}} in the guidance, causing it to output
    {{{{var}}}} instead of the correct {{var}} in refined rubrics.
    """
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    criteria = [
        Criterion(
            name="test_crit",
            category="Test",
            weight=1,
            dimension="accuracy",
            criterion="Check {{test_var}}"
        )
    ]
    
    prompt = build_refine_rubric_prompt(dimensions, criteria)
    
    # The guidance should tell the LLM to use double braces {{var}}, 
    # NOT quadruple braces {{{{var}}}}
    # If we find quadruple braces in the guidance text, the bug exists
    assert "{{{{" not in prompt, (
        "Bug: Prompt contains quadruple braces {{{{...}}}} which will cause "
        "the LLM to output doubled braces in refined rubrics. "
        "Variable placeholders should show as {{var}} not {{{{var}}}}."
    )
    
    # The prompt should contain proper double-brace examples
    assert "{{variable_name}}" in prompt or "{{" in prompt


def test_criteria_generation_prompt_variable_placeholders_use_double_braces():
    """Test that criteria generation prompts show {{var}} syntax to the LLM."""
    from rubric_kit.prompts import build_criteria_generation_prompt
    
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    prompt = build_criteria_generation_prompt(
        question="What is 2+2?",
        answer="4",
        dimensions=dimensions,
        num_criteria=3
    )
    
    # Should NOT contain quadruple braces in guidance
    assert "{{{{" not in prompt, (
        "Bug: Criteria generation prompt contains quadruple braces"
    )


def test_chat_criteria_generation_prompt_variable_placeholders_use_double_braces():
    """Test that chat criteria generation prompts show {{var}} syntax to the LLM."""
    from rubric_kit.prompts import build_chat_criteria_generation_prompt
    
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    prompt = build_chat_criteria_generation_prompt(
        chat_content="User: Hello\nAssistant: Hi",
        dimensions=dimensions,
        num_criteria=3
    )
    
    # Should NOT contain quadruple braces in guidance
    assert "{{{{" not in prompt, (
        "Bug: Chat criteria generation prompt contains quadruple braces"
    )


# =============================================================================
# No-Variables Mode Tests
# =============================================================================

def test_refine_prompt_with_use_variables_true_includes_variables_guidance():
    """Test that refine prompt includes variables guidance by default."""
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    criteria = [
        Criterion(
            name="test_crit",
            category="Test",
            weight=1,
            dimension="accuracy",
            criterion="Test criterion"
        )
    ]
    
    prompt = build_refine_rubric_prompt(dimensions, criteria, use_variables=True)
    
    # Should include variables guidance
    assert "variables" in prompt.lower()
    assert "{{variable_name}}" in prompt or "{{" in prompt


def test_refine_prompt_with_use_variables_false_excludes_variables_guidance():
    """Test that refine prompt excludes variables guidance when use_variables=False."""
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    criteria = [
        Criterion(
            name="test_crit",
            category="Test",
            weight=1,
            dimension="accuracy",
            criterion="Test criterion"
        )
    ]
    
    prompt = build_refine_rubric_prompt(dimensions, criteria, use_variables=False)
    
    # Should NOT include variables guidance or placeholder syntax examples
    assert "{{variable_name}}" not in prompt
    # Should instruct to use hardcoded values
    assert "hardcoded" in prompt.lower() or "hardcode" in prompt.lower() or "hard-coded" in prompt.lower()


def test_criteria_generation_prompt_with_use_variables_false():
    """Test that criteria generation prompt excludes variables when use_variables=False."""
    from rubric_kit.prompts import build_criteria_generation_prompt
    
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    prompt = build_criteria_generation_prompt(
        question="What is 2+2?",
        answer="4",
        dimensions=dimensions,
        num_criteria=3,
        use_variables=False
    )
    
    # Should NOT include variables guidance
    assert "{{variable_name}}" not in prompt
    # Should instruct to use hardcoded values
    assert "hardcoded" in prompt.lower() or "hardcode" in prompt.lower() or "hard-coded" in prompt.lower()


def test_chat_criteria_generation_prompt_with_use_variables_false():
    """Test that chat criteria generation prompt excludes variables when use_variables=False."""
    from rubric_kit.prompts import build_chat_criteria_generation_prompt
    
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    
    prompt = build_chat_criteria_generation_prompt(
        chat_content="User: Hello\nAssistant: Hi",
        dimensions=dimensions,
        num_criteria=3,
        use_variables=False
    )
    
    # Should NOT include variables guidance
    assert "{{variable_name}}" not in prompt
    # Should instruct to use hardcoded values
    assert "hardcoded" in prompt.lower() or "hardcode" in prompt.lower() or "hard-coded" in prompt.lower()


# =============================================================================
# Guidelines Parameter Tests
# =============================================================================

def test_build_dimension_generation_prompt_with_guidelines():
    """Test that dimension generation prompt includes guidelines when provided."""
    question = "What is the capital of France?"
    answer = "Paris"
    guidelines = "Focus on factual accuracy and completeness. Avoid subjective criteria."
    
    prompt = build_dimension_generation_prompt(
        question=question,
        answer=answer,
        num_dimensions=3,
        guidelines=guidelines
    )
    
    # Guidelines should appear in the prompt
    assert guidelines in prompt
    assert "Focus on factual accuracy" in prompt
    assert "Avoid subjective criteria" in prompt


def test_build_dimension_generation_prompt_without_guidelines():
    """Test that dimension generation prompt works without guidelines."""
    question = "What is the capital of France?"
    answer = "Paris"
    
    prompt = build_dimension_generation_prompt(
        question=question,
        answer=answer,
        num_dimensions=3
    )
    
    # Prompt should still be valid without guidelines
    assert question in prompt
    assert answer in prompt
    assert "dimension" in prompt.lower()


def test_build_criteria_generation_prompt_with_guidelines():
    """Test that criteria generation prompt includes guidelines when provided."""
    from rubric_kit.prompts import build_criteria_generation_prompt
    
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    guidelines = "Create criteria that check specific numerical values. Each criterion should be atomic."
    
    prompt = build_criteria_generation_prompt(
        question="What is 2+2?",
        answer="4",
        dimensions=dimensions,
        num_criteria=3,
        guidelines=guidelines
    )
    
    # Guidelines should appear in the prompt
    assert guidelines in prompt
    assert "specific numerical values" in prompt


def test_build_chat_dimension_generation_prompt_with_guidelines():
    """Test that chat dimension generation prompt includes guidelines when provided."""
    from rubric_kit.prompts import build_chat_dimension_generation_prompt
    
    chat_content = "User: Hello\nAssistant: Hi there!"
    guidelines = "Include a dimension for tool usage evaluation. Focus on response quality."
    
    prompt = build_chat_dimension_generation_prompt(
        chat_content=chat_content,
        num_dimensions=3,
        guidelines=guidelines
    )
    
    # Guidelines should appear in the prompt
    assert guidelines in prompt
    assert "tool usage evaluation" in prompt


def test_build_chat_criteria_generation_prompt_with_guidelines():
    """Test that chat criteria generation prompt includes guidelines when provided."""
    from rubric_kit.prompts import build_chat_criteria_generation_prompt
    
    dimensions = [
        Dimension(name="accuracy", description="Test", grading_type="binary")
    ]
    guidelines = "Create granular criteria for each fact. Avoid combining multiple checks."
    
    prompt = build_chat_criteria_generation_prompt(
        chat_content="User: Hello\nAssistant: Hi",
        dimensions=dimensions,
        num_criteria=3,
        guidelines=guidelines
    )
    
    # Guidelines should appear in the prompt
    assert guidelines in prompt
    assert "granular criteria" in prompt

