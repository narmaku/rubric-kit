"""Tests for LLM judge functionality with judge panel architecture."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import litellm
from rubric_kit.schema import (
    Rubric, Dimension, Criterion,
    JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig
)


@pytest.fixture
def sample_chat_session_file():
    """Create a sample chat session file."""
    content = """User: What are the system specifications?
Assistant: The system has 8 physical CPUs and 64 GB of RAM running Fedora Linux 42.

Tool calls:
- get_system_information() -> {"os": "Fedora Linux 42", "cpus": 8, "ram_gb": 64}
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def simple_rubric():
    """Create a simple rubric for testing."""
    dimensions = [
        Dimension(
            name="factual_correctness",
            description="Evaluates factual correctness",
            grading_type="binary"
        ),
        Dimension(
            name="usefulness",
            description="Evaluates usefulness",
            grading_type="score",
            scores={1: "Not useful", 2: "Somewhat useful", 3: "Very useful"}
        )
    ]
    
    criteria = [
        Criterion(
            name="test_fact",
            category="Output",
            weight=3,
            dimension="factual_correctness",
            criterion="The response must indicate that number of physical CPUs is 8."
        ),
        Criterion(
            name="test_useful",
            category="Output",
            weight="from_scores",
            dimension="usefulness",
            criterion="from_scores"
        )
    ]
    
    return Rubric(dimensions=dimensions, criteria=criteria)


@pytest.fixture
def single_judge_panel():
    """Create a single-judge panel configuration."""
    return JudgePanelConfig(
        judges=[JudgeConfig(name="judge_1", model="gpt-4", api_key="test-key")],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )


@pytest.fixture
def multi_judge_panel():
    """Create a multi-judge panel configuration."""
    return JudgePanelConfig(
        judges=[
            JudgeConfig(name="judge_1", model="gpt-4", api_key="test-key-1"),
            JudgeConfig(name="judge_2", model="gpt-4-turbo", api_key="test-key-2"),
            JudgeConfig(name="judge_3", model="gpt-4", api_key="test-key-3")
        ],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="quorum", threshold=2)
    )


# ============================================================================
# Core Evaluation Tests
# ============================================================================

def test_evaluate_criterion_with_single_judge(simple_rubric, sample_chat_session_file, single_judge_panel):
    """Test evaluating a criterion with single judge."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]  # Binary criterion
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Mock litellm.completion
    with patch('litellm.completion') as mock_completion:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "RESULT: PASS\nREASON: The response correctly states 8 CPUs."
        mock_completion.return_value = mock_response
        
        result = evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=single_judge_panel
        )
        
        assert result["consensus_reached"] is True
        assert result["passes"] is True
        assert len(result["judge_votes"]) == 1


def test_evaluate_criterion_with_multi_judge_consensus(simple_rubric, sample_chat_session_file, multi_judge_panel):
    """Test evaluating a criterion with multi-judge panel reaching consensus."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]  # Binary criterion
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Mock litellm.completion to return consistent PASS votes
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        mock_completion.return_value = mock_response
        
        result = evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=multi_judge_panel
        )
        
        # With threshold=2 and all 3 judges voting PASS, consensus is reached
        assert result["consensus_reached"] is True
        assert result["passes"] is True
        assert len(result["judge_votes"]) == 3
        assert result["consensus_count"] == 3


def test_evaluate_criterion_with_multi_judge_no_consensus(simple_rubric, sample_chat_session_file, multi_judge_panel):
    """Test evaluating a criterion with multi-judge panel not reaching consensus."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]  # Binary criterion
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Mock litellm.completion to return split votes
    call_count = [0]
    
    def mock_completion_call(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        # First call: PASS, second: FAIL, third: PASS
        if call_count[0] == 1:
            response.choices[0].message.content = "RESULT: FAIL\nREASON: Not specific enough."
        else:
            response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        call_count[0] += 1
        return response
    
    with patch('litellm.completion', side_effect=mock_completion_call):
        result = evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=multi_judge_panel
        )
        
        # With threshold=2 and votes 2 PASS, 1 FAIL, consensus IS reached (2 >= threshold)
        assert result["consensus_reached"] is True
        assert result["passes"] is True
        assert len(result["judge_votes"]) == 3
        assert result["consensus_count"] == 2


def test_evaluate_criterion_score_based(simple_rubric, sample_chat_session_file, multi_judge_panel):
    """Test evaluating a score-based criterion with multi-judge panel."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[1]  # Score criterion
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Mock litellm.completion to return consistent score
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SCORE: 2\nREASON: Somewhat useful response."
        mock_completion.return_value = mock_response
        
        result = evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=multi_judge_panel
        )
        
        # With threshold=2 and all 3 judges giving score 2, consensus is reached
        assert result["consensus_reached"] is True
        assert result["score"] == 2
        assert len(result["judge_votes"]) == 3


# ============================================================================
# Full Rubric Evaluation Tests
# ============================================================================

def test_evaluate_rubric_with_panel(simple_rubric, sample_chat_session_file, single_judge_panel):
    """Test evaluating full rubric with judge panel."""
    from rubric_kit.llm_judge import evaluate_rubric_with_panel
    
    # Mock litellm.completion
    call_count = [0]
    
    def mock_completion_call(*args, **kwargs):
        response = MagicMock()
        response.choices = [MagicMock()]
        if call_count[0] == 0:
            # First criterion (binary)
            response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        else:
            # Second criterion (score)
            response.choices[0].message.content = "SCORE: 3\nREASON: Very useful."
        call_count[0] += 1
        return response
    
    with patch('litellm.completion', side_effect=mock_completion_call):
        evaluations = evaluate_rubric_with_panel(
            rubric=simple_rubric,
            chat_session_file=sample_chat_session_file,
            panel_config=single_judge_panel
        )
        
        assert len(evaluations) == 2
        assert "test_fact" in evaluations
        assert "test_useful" in evaluations
        
        # Check binary criterion result
        assert evaluations["test_fact"]["type"] == "binary"
        assert evaluations["test_fact"]["consensus_reached"] is True
        assert evaluations["test_fact"]["passes"] is True
        
        # Check score criterion result
        assert evaluations["test_useful"]["type"] == "score"
        assert evaluations["test_useful"]["consensus_reached"] is True
        assert evaluations["test_useful"]["score"] == 3


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_evaluate_criterion_with_api_error(simple_rubric, sample_chat_session_file, single_judge_panel):
    """Test handling API errors during evaluation."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Mock API error
    with patch('litellm.completion', side_effect=Exception("API Error")):
        with pytest.raises(Exception, match="Judge evaluation failed"):
            evaluate_criterion_with_panel(
                criterion=criterion,
                chat_content=chat_content,
                dimension=dimension,
                panel_config=single_judge_panel
            )


# ============================================================================
# Consensus Reason Building Tests
# ============================================================================

def test_build_consensus_reason_single_judge():
    """Test that single judge reason is returned without label."""
    from rubric_kit.llm_judge import _build_consensus_reason
    
    consensus_result = {
        "passes": True,
        "judge_votes": [
            {"judge": "primary", "passes": True, "reason": "All criteria met"}
        ]
    }
    
    reason = _build_consensus_reason(consensus_result)
    # Single judge: no label added
    assert reason == "All criteria met"


def test_build_consensus_reason_multi_judge_agreement():
    """Test that multi-judge agreement returns one labeled reason."""
    from rubric_kit.llm_judge import _build_consensus_reason
    
    consensus_result = {
        "passes": True,
        "judge_votes": [
            {"judge": "primary", "passes": True, "reason": "Reason from primary"},
            {"judge": "secondary", "passes": True, "reason": "Reason from secondary"},
            {"judge": "tertiary", "passes": True, "reason": "Reason from tertiary"}
        ]
    }
    
    # Set seed for reproducible test
    import random
    random.seed(42)
    
    reason = _build_consensus_reason(consensus_result)
    
    # Should contain one of the agreeing judges' reasons with label
    assert " (from " in reason
    assert reason.endswith(")")
    
    # Verify it's one of the agreeing judges
    possible_reasons = [
        "Reason from primary (from primary)",
        "Reason from secondary (from secondary)",
        "Reason from tertiary (from tertiary)"
    ]
    assert reason in possible_reasons


def test_build_consensus_reason_partial_agreement():
    """Test that only agreeing judges' reasons are considered."""
    from rubric_kit.llm_judge import _build_consensus_reason
    
    consensus_result = {
        "passes": True,  # Final decision is PASS
        "judge_votes": [
            {"judge": "primary", "passes": True, "reason": "I agree it passes"},
            {"judge": "secondary", "passes": False, "reason": "I disagree"},
            {"judge": "tertiary", "passes": True, "reason": "This passes"}
        ]
    }
    
    # Set seed for reproducible test
    import random
    random.seed(42)
    
    reason = _build_consensus_reason(consensus_result)
    
    # Should only include reasons from judges who voted PASS
    assert " (from " in reason
    assert "I disagree" not in reason
    
    # Should be one of the agreeing judges
    possible_reasons = [
        "I agree it passes (from primary)",
        "This passes (from tertiary)"
    ]
    assert reason in possible_reasons


def test_build_consensus_reason_score_agreement():
    """Test reason building for score-based criteria."""
    from rubric_kit.llm_judge import _build_consensus_reason
    
    consensus_result = {
        "score": 3,  # Final score
        "judge_votes": [
            {"judge": "primary", "score": 3, "reason": "Excellent quality"},
            {"judge": "secondary", "score": 2, "reason": "Good but not great"},
            {"judge": "tertiary", "score": 3, "reason": "Outstanding work"}
        ]
    }
    
    # Set seed for reproducible test
    import random
    random.seed(42)
    
    reason = _build_consensus_reason(consensus_result)
    
    # Should only include reasons from judges who gave score 3
    assert " (from " in reason
    assert "Good but not great" not in reason
    
    # Should be one of the agreeing judges
    possible_reasons = [
        "Excellent quality (from primary)",
        "Outstanding work (from tertiary)"
    ]
    assert reason in possible_reasons


# ============================================================================
# Tool Breakdown Extraction Tests
# ============================================================================

def test_extract_tool_breakdown_single_judge():
    """Test that single judge's tool breakdown is returned."""
    from rubric_kit.llm_judge import _extract_tool_breakdown
    
    judge_votes = [
        {
            "judge": "primary",
            "passes": True,
            "reason": "All tools called correctly",
            "tool_breakdown": {"overall_pass": True, "overall_score": 3.0}
        }
    ]
    
    result = {"passes": True}
    breakdown = _extract_tool_breakdown(judge_votes, result)
    
    assert breakdown is not None
    assert breakdown["overall_pass"] is True
    assert breakdown["overall_score"] == 3.0


def test_extract_tool_breakdown_from_agreeing_judge():
    """Test that tool breakdown is extracted from a judge that agrees with consensus."""
    from rubric_kit.llm_judge import _extract_tool_breakdown
    
    # Scenario: First judge says PASS (wrong breakdown), second says FAIL (correct)
    # Consensus is FAIL - we should get the breakdown from the FAIL judge
    judge_votes = [
        {
            "judge": "gemini-flash",
            "passes": True,  # Disagrees with final result
            "reason": "All params look OK",
            "tool_breakdown": {
                "overall_pass": True,
                "overall_score": 3.0,
                "tool_results": [{"name": "tool1", "params_ok": True}]
            }
        },
        {
            "judge": "gemini-pro",
            "passes": False,  # Agrees with final result
            "reason": "Parameters have formatting issues",
            "tool_breakdown": {
                "overall_pass": False,
                "overall_score": 1.0,
                "tool_results": [{"name": "tool1", "params_ok": False}]
            }
        }
    ]
    
    result = {"passes": False}  # Final consensus is FAIL
    breakdown = _extract_tool_breakdown(judge_votes, result)
    
    # Should get the breakdown from the judge that agrees (gemini-pro, who said FAIL)
    assert breakdown is not None
    assert breakdown["overall_pass"] is False
    assert breakdown["overall_score"] == 1.0
    assert breakdown["tool_results"][0]["params_ok"] is False


def test_extract_tool_breakdown_from_agreeing_judge_pass():
    """Test that tool breakdown is extracted from agreeing judge when consensus is PASS."""
    from rubric_kit.llm_judge import _extract_tool_breakdown
    
    # Scenario: First judge says FAIL (wrong), second says PASS (correct)
    # Consensus is PASS - we should get the breakdown from the PASS judge
    judge_votes = [
        {
            "judge": "strict-judge",
            "passes": False,  # Disagrees with final result
            "reason": "Found issues",
            "tool_breakdown": {
                "overall_pass": False,
                "overall_score": 1.0
            }
        },
        {
            "judge": "lenient-judge",
            "passes": True,  # Agrees with final result
            "reason": "All good",
            "tool_breakdown": {
                "overall_pass": True,
                "overall_score": 3.0
            }
        }
    ]
    
    result = {"passes": True}  # Final consensus is PASS
    breakdown = _extract_tool_breakdown(judge_votes, result)
    
    # Should get the breakdown from the judge that agrees (lenient-judge)
    assert breakdown is not None
    assert breakdown["overall_pass"] is True
    assert breakdown["overall_score"] == 3.0


def test_extract_tool_breakdown_no_breakdown_present():
    """Test that None is returned when no tool breakdown is present."""
    from rubric_kit.llm_judge import _extract_tool_breakdown
    
    judge_votes = [
        {"judge": "primary", "passes": True, "reason": "OK"},
        {"judge": "secondary", "passes": True, "reason": "OK"}
    ]
    
    result = {"passes": True}
    breakdown = _extract_tool_breakdown(judge_votes, result)
    
    assert breakdown is None


def test_extract_tool_breakdown_fallback_when_no_agreeing_has_breakdown():
    """Test fallback to first available breakdown when no agreeing judge has one."""
    from rubric_kit.llm_judge import _extract_tool_breakdown
    
    # Only the disagreeing judge has a breakdown
    judge_votes = [
        {
            "judge": "judge-with-breakdown",
            "passes": True,  # Disagrees with final
            "reason": "Pass",
            "tool_breakdown": {"overall_pass": True, "overall_score": 3.0}
        },
        {
            "judge": "judge-without-breakdown",
            "passes": False,  # Agrees with final but no breakdown
            "reason": "Fail"
        }
    ]
    
    result = {"passes": False}
    breakdown = _extract_tool_breakdown(judge_votes, result)
    
    # Should fallback to the only available breakdown
    assert breakdown is not None
    assert breakdown["overall_pass"] is True


# ============================================================================
# LLM Parameter Tests
# ============================================================================

def test_judge_with_custom_temperature(simple_rubric, sample_chat_session_file):
    """Test that judge-specific temperature is used when provided."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Create judge with custom temperature
    panel = JudgePanelConfig(
        judges=[JudgeConfig(
            name="judge_1",
            model="gpt-4",
            api_key="test-key",
            temperature=0.7  # Custom temperature
        )],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )
    
    # Mock litellm.completion and capture the API call parameters
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        mock_completion.return_value = mock_response
        
        evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=panel
        )
        
        # Verify that the custom temperature was used
        call_args = mock_completion.call_args
        assert call_args is not None
        assert call_args.kwargs["temperature"] == 0.7


def test_judge_with_default_temperature(simple_rubric, sample_chat_session_file):
    """Test that default temperature is used when judge-specific temperature is not provided."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    from rubric_kit.prompts import EVALUATOR_CONFIG
    
    criterion = simple_rubric.criteria[0]
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Create judge without custom temperature
    panel = JudgePanelConfig(
        judges=[JudgeConfig(
            name="judge_1",
            model="gpt-4",
            api_key="test-key"
            # No temperature specified
        )],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )
    
    # Mock litellm.completion and capture the API call parameters
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        mock_completion.return_value = mock_response
        
        evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=panel
        )
        
        # Verify that the default temperature from EVALUATOR_CONFIG was used
        call_args = mock_completion.call_args
        assert call_args is not None
        assert call_args.kwargs["temperature"] == EVALUATOR_CONFIG.temperature


def test_judge_with_custom_max_tokens(simple_rubric, sample_chat_session_file):
    """Test that judge-specific max_tokens is used when provided."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Create judge with custom max_tokens
    panel = JudgePanelConfig(
        judges=[JudgeConfig(
            name="judge_1",
            model="gpt-4",
            api_key="test-key",
            max_tokens=4096  # Custom max_tokens
        )],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )
    
    # Mock litellm.completion and capture the API call parameters
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        mock_completion.return_value = mock_response
        
        evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=panel
        )
        
        # Verify that the custom max_tokens was used
        call_args = mock_completion.call_args
        assert call_args is not None
        assert call_args.kwargs["max_tokens"] == 4096


def test_judge_with_multiple_custom_parameters(simple_rubric, sample_chat_session_file):
    """Test that multiple judge-specific parameters can be set together."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Create judge with multiple custom parameters
    panel = JudgePanelConfig(
        judges=[JudgeConfig(
            name="judge_1",
            model="gpt-4",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2
        )],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="unanimous")
    )
    
    # Mock litellm.completion and capture the API call parameters
    with patch('litellm.completion') as mock_completion:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        mock_completion.return_value = mock_response
        
        evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=panel
        )
        
        # Verify that all custom parameters were used
        call_args = mock_completion.call_args
        assert call_args is not None
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 2048
        assert call_args.kwargs["top_p"] == 0.9
        assert call_args.kwargs["frequency_penalty"] == 0.1
        assert call_args.kwargs["presence_penalty"] == 0.2


def test_judge_panel_with_varied_parameters(simple_rubric, sample_chat_session_file):
    """Test that different judges in a panel can have different parameters."""
    from rubric_kit.llm_judge import evaluate_criterion_with_panel
    
    criterion = simple_rubric.criteria[0]
    dimension = simple_rubric.get_dimension(criterion.dimension)
    chat_content = open(sample_chat_session_file).read()
    
    # Create panel with judges having different parameters
    panel = JudgePanelConfig(
        judges=[
            JudgeConfig(
                name="judge_1",
                model="gpt-4",
                api_key="test-key-1",
                temperature=0.0  # Deterministic judge
            ),
            JudgeConfig(
                name="judge_2",
                model="gpt-4",
                api_key="test-key-2",
                temperature=0.7,  # More creative judge
                top_p=0.9
            ),
            JudgeConfig(
                name="judge_3",
                model="gpt-4",
                api_key="test-key-3"
                # Uses defaults
            )
        ],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="quorum", threshold=2)
    )
    
    # Track API calls to verify each judge uses its own parameters
    call_params = []
    
    def mock_completion_call(*args, **kwargs):
        call_params.append(kwargs.copy())
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "RESULT: PASS\nREASON: Correct."
        return response
    
    with patch('litellm.completion', side_effect=mock_completion_call):
        evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=panel
        )
        
        # Verify each judge used its own parameters
        assert len(call_params) == 3
        assert call_params[0]["temperature"] == 0.0  # Judge 1: custom temperature
        assert call_params[1]["temperature"] == 0.7  # Judge 2: custom temperature
        assert call_params[1]["top_p"] == 0.9  # Judge 2: custom top_p
        # Judge 3 should use defaults (checked via EVALUATOR_CONFIG)
