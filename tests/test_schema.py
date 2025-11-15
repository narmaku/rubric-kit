"""Tests for schema validation."""

import pytest
from pydantic import ValidationError


def test_binary_descriptor():
    """Test binary grading type descriptor."""
    from rubric_kit.schema import Rubric, Dimension, Criterion
    
    descriptor = Dimension(
        name="factual_correctness",
        description="Evaluates that the information of the final answer is factually correct.",
        grading_type="binary"
    )
    
    assert descriptor.name == "factual_correctness"
    assert descriptor.grading_type == "binary"
    assert descriptor.scores is None


def test_score_descriptor():
    """Test score grading type descriptor with scores."""
    from rubric_kit.schema import Rubric, Dimension, Criterion
    
    descriptor = Dimension(
        name="usefulness",
        description="Evaluates how useful is the final response.",
        grading_type="score",
        scores={
            1: "The response is completely useless.",
            2: "The response is useful but incomplete.",
            3: "The response is useful and complete."
        }
    )
    
    assert descriptor.name == "usefulness"
    assert descriptor.grading_type == "score"
    assert len(descriptor.scores) == 3
    assert descriptor.scores[1] == "The response is completely useless."


def test_score_descriptor_with_pass_above():
    """Test score descriptor with pass_above threshold."""
    from rubric_kit.schema import Dimension
    
    descriptor = Dimension(
        name="usefulness",
        description="Test",
        grading_type="score",
        scores={1: "Bad", 2: "Good", 3: "Great"},
        pass_above=2
    )
    
    assert descriptor.pass_above == 2
    
    # Test invalid pass_above on binary type
    with pytest.raises(ValidationError):
        Dimension(
            name="test",
            description="Test",
            grading_type="binary",
            pass_above=2  # Should fail: can't use pass_above with binary
        )
    
    # Test invalid pass_above value
    with pytest.raises(ValidationError):
        Dimension(
            name="test",
            description="Test",
            grading_type="score",
            scores={1: "Bad", 2: "Good", 3: "Great"},
            pass_above=5  # Should fail: 5 is not a valid score
        )


def test_score_descriptor_requires_scores():
    """Test that score type requires scores dict."""
    from rubric_kit.schema import Rubric, Dimension, Criterion
    
    with pytest.raises(ValidationError):
        Dimension(
            name="usefulness",
            description="Test",
            grading_type="score"
            # Missing scores
        )


def test_invalid_grading_type():
    """Test that invalid grading type raises error."""
    from rubric_kit.schema import Rubric, Dimension, Criterion
    
    with pytest.raises(ValidationError):
        Dimension(
            name="test",
            description="Test",
            grading_type="invalid"
        )


def test_output_criterion():
    """Test Output category criterion."""
    from rubric_kit.schema import Criterion
    
    criterion = Criterion(
        name="sys_info_factual_1",
        category="Output",
        weight=3,
        dimension="factual_correctness",
        criterion="The response must indicate that number of physical CPUs is 8."
    )
    
    assert criterion.name == "sys_info_factual_1"
    assert criterion.category == "Output"
    assert criterion.weight == 3
    assert criterion.dimension == "factual_correctness"
    assert criterion.tool_calls is None


def test_criterion_with_from_scores_weight():
    """Test criterion with from_scores weight."""
    from rubric_kit.schema import Criterion
    
    criterion = Criterion(
        name="useful_1",
        category="Output",
        weight="from_scores",
        dimension="usefulness",
        criterion="from_scores"
    )
    
    assert criterion.weight == "from_scores"
    assert criterion.criterion == "from_scores"


def test_criterion_weight_range():
    """Test that weight must be in range 0-3."""
    from rubric_kit.schema import Criterion
    
    with pytest.raises(ValidationError):
        Criterion(
            name="test",
            category="Output",
            weight=5,  # Invalid: > 3
            dimension="test",
            criterion="test"
        )


def test_tool_criterion():
    """Test Tools category criterion with tool_calls."""
    from rubric_kit.schema import Criterion, ToolCalls, ToolSpec
    
    tool_calls = ToolCalls(
        respect_order=True,
        required=[
            ToolSpec(
                name="get_system_information",
                min_calls=1,
                max_calls=1,
                params={}
            )
        ],
        optional=[],
        prohibited=[
            ToolSpec(
                name="get_weather",
                params={}
            )
        ]
    )
    
    criterion = Criterion(
        name="tool_call_1",
        category="Tools",
        weight=3,
        dimension="tool_usage",
        tool_calls=tool_calls
    )
    
    assert criterion.category == "Tools"
    assert len(criterion.tool_calls.required) == 1
    assert criterion.tool_calls.required[0].name == "get_system_information"
    assert len(criterion.tool_calls.prohibited) == 1


def test_rubric_complete():
    """Test complete rubric structure."""
    from rubric_kit.schema import Rubric, Dimension, Criterion, Criterion
    
    descriptors = [
        Dimension(
            name="factual_correctness",
            description="Test",
            grading_type="binary"
        )
    ]
    
    criteria = [
        Criterion(
            name="test_1",
            category="Output",
            weight=3,
            dimension="factual_correctness",
            criterion="Test criterion"
        )
    ]
    
    rubric = Rubric(dimensions=descriptors, criteria=criteria)
    
    assert len(rubric.dimensions) == 1
    assert len(rubric.criteria) == 1
    assert rubric.dimensions[0].name == "factual_correctness"


def test_rubric_validates_dimension_references():
    """Test that criteria must reference valid descriptors."""
    from rubric_kit.schema import Rubric, Dimension, Criterion, Criterion
    
    descriptors = [
        Dimension(
            name="factual_correctness",
            description="Test",
            grading_type="binary"
        )
    ]
    
    criteria = [
        Criterion(
            name="test_1",
            category="Output",
            weight=3,
            dimension="nonexistent_dimension",  # Invalid reference
            criterion="Test"
        )
    ]
    
    with pytest.raises(ValidationError, match="references non-existent dimension"):
        Rubric(dimensions=descriptors, criteria=criteria)


# ============================================================================
# Judge Panel Schema Tests
# ============================================================================

def test_judge_config_basic():
    """Test basic JudgeConfig creation."""
    from rubric_kit.schema import JudgeConfig
    
    judge = JudgeConfig(
        name="judge_1",
        model="gpt-4"
    )
    
    assert judge.name == "judge_1"
    assert judge.model == "gpt-4"
    assert judge.api_key is None
    assert judge.base_url is None


def test_judge_config_with_api_key_and_base_url():
    """Test JudgeConfig with custom API key and base URL."""
    from rubric_kit.schema import JudgeConfig
    
    judge = JudgeConfig(
        name="claude_judge",
        model="claude-3-5-sonnet-20241022",
        api_key="sk-test-key",
        base_url="https://api.anthropic.com/v1"
    )
    
    assert judge.name == "claude_judge"
    assert judge.model == "claude-3-5-sonnet-20241022"
    assert judge.api_key == "sk-test-key"
    assert judge.base_url == "https://api.anthropic.com/v1"


def test_judge_config_requires_name_and_model():
    """Test that JudgeConfig requires name and model."""
    from rubric_kit.schema import JudgeConfig
    
    with pytest.raises(ValidationError):
        JudgeConfig(name="test")  # Missing model
    
    with pytest.raises(ValidationError):
        JudgeConfig(model="gpt-4")  # Missing name


def test_execution_config_defaults():
    """Test ExecutionConfig default values."""
    from rubric_kit.schema import ExecutionConfig
    
    config = ExecutionConfig()
    
    assert config.mode == "sequential"
    assert config.batch_size == 2
    assert config.timeout == 30


def test_execution_config_sequential():
    """Test ExecutionConfig with sequential mode."""
    from rubric_kit.schema import ExecutionConfig
    
    config = ExecutionConfig(mode="sequential", timeout=60)
    
    assert config.mode == "sequential"
    assert config.timeout == 60


def test_execution_config_parallel():
    """Test ExecutionConfig with parallel mode."""
    from rubric_kit.schema import ExecutionConfig
    
    config = ExecutionConfig(mode="parallel")
    
    assert config.mode == "parallel"


def test_execution_config_batched():
    """Test ExecutionConfig with batched mode."""
    from rubric_kit.schema import ExecutionConfig
    
    config = ExecutionConfig(mode="batched", batch_size=3)
    
    assert config.mode == "batched"
    assert config.batch_size == 3


def test_execution_config_invalid_mode():
    """Test that invalid execution mode raises error."""
    from rubric_kit.schema import ExecutionConfig
    
    with pytest.raises(ValidationError):
        ExecutionConfig(mode="invalid")


def test_execution_config_batch_size_validation():
    """Test that batch_size must be >= 1."""
    from rubric_kit.schema import ExecutionConfig
    
    with pytest.raises(ValidationError):
        ExecutionConfig(batch_size=0)
    
    with pytest.raises(ValidationError):
        ExecutionConfig(batch_size=-1)


def test_consensus_config_defaults():
    """Test ConsensusConfig default values."""
    from rubric_kit.schema import ConsensusConfig
    
    config = ConsensusConfig()
    
    assert config.mode == "unanimous"
    assert config.threshold is None
    assert config.on_no_consensus == "fail"


def test_consensus_config_quorum():
    """Test ConsensusConfig with quorum mode."""
    from rubric_kit.schema import ConsensusConfig
    
    config = ConsensusConfig(mode="quorum", threshold=2)
    
    assert config.mode == "quorum"
    assert config.threshold == 2


def test_consensus_config_quorum_requires_threshold():
    """Test that quorum mode requires threshold."""
    from rubric_kit.schema import ConsensusConfig
    
    with pytest.raises(ValidationError, match="threshold is required for quorum"):
        ConsensusConfig(mode="quorum")  # Missing threshold


def test_consensus_config_majority():
    """Test ConsensusConfig with majority mode."""
    from rubric_kit.schema import ConsensusConfig
    
    config = ConsensusConfig(mode="majority")
    
    assert config.mode == "majority"


def test_consensus_config_unanimous():
    """Test ConsensusConfig with unanimous mode."""
    from rubric_kit.schema import ConsensusConfig
    
    config = ConsensusConfig(mode="unanimous")
    
    assert config.mode == "unanimous"


def test_consensus_config_on_no_consensus_options():
    """Test on_no_consensus options."""
    from rubric_kit.schema import ConsensusConfig
    
    config1 = ConsensusConfig(on_no_consensus="fail")
    assert config1.on_no_consensus == "fail"
    
    config2 = ConsensusConfig(on_no_consensus="median")
    assert config2.on_no_consensus == "median"
    
    config3 = ConsensusConfig(on_no_consensus="most_common")
    assert config3.on_no_consensus == "most_common"


def test_consensus_config_invalid_on_no_consensus():
    """Test that invalid on_no_consensus raises error."""
    from rubric_kit.schema import ConsensusConfig
    
    with pytest.raises(ValidationError):
        ConsensusConfig(on_no_consensus="invalid")


def test_judge_panel_config_basic():
    """Test basic JudgePanelConfig creation."""
    from rubric_kit.schema import JudgePanelConfig, JudgeConfig
    
    panel = JudgePanelConfig(
        judges=[
            JudgeConfig(name="judge_1", model="gpt-4")
        ]
    )
    
    assert len(panel.judges) == 1
    assert panel.judges[0].name == "judge_1"
    assert panel.execution.mode == "sequential"  # Default
    assert panel.consensus.mode == "unanimous"  # Default


def test_judge_panel_config_multiple_judges():
    """Test JudgePanelConfig with multiple judges."""
    from rubric_kit.schema import JudgePanelConfig, JudgeConfig, ExecutionConfig, ConsensusConfig
    
    panel = JudgePanelConfig(
        judges=[
            JudgeConfig(name="judge_1", model="gpt-4"),
            JudgeConfig(name="judge_2", model="gpt-4-turbo"),
            JudgeConfig(name="judge_3", model="claude-3-5-sonnet")
        ],
        execution=ExecutionConfig(mode="sequential"),
        consensus=ConsensusConfig(mode="quorum", threshold=2)
    )
    
    assert len(panel.judges) == 3
    assert panel.execution.mode == "sequential"
    assert panel.consensus.mode == "quorum"
    assert panel.consensus.threshold == 2


def test_judge_panel_config_requires_at_least_one_judge():
    """Test that JudgePanelConfig requires at least one judge."""
    from rubric_kit.schema import JudgePanelConfig
    
    with pytest.raises(ValidationError):
        JudgePanelConfig(judges=[])  # Empty judges list


def test_judge_panel_config_validates_quorum_threshold():
    """Test that quorum threshold cannot exceed number of judges."""
    from rubric_kit.schema import JudgePanelConfig, JudgeConfig, ConsensusConfig
    
    with pytest.raises(ValidationError, match="threshold.*cannot exceed.*number of judges"):
        JudgePanelConfig(
            judges=[
                JudgeConfig(name="judge_1", model="gpt-4"),
                JudgeConfig(name="judge_2", model="gpt-4-turbo")
            ],
            consensus=ConsensusConfig(mode="quorum", threshold=3)  # 3 > 2 judges
        )


def test_judge_panel_config_auto_calculates_majority_threshold():
    """Test that majority mode auto-calculates threshold."""
    from rubric_kit.schema import JudgePanelConfig, JudgeConfig, ConsensusConfig
    
    # 3 judges: majority = 2
    panel = JudgePanelConfig(
        judges=[
            JudgeConfig(name="judge_1", model="gpt-4"),
            JudgeConfig(name="judge_2", model="gpt-4-turbo"),
            JudgeConfig(name="judge_3", model="claude-3-5-sonnet")
        ],
        consensus=ConsensusConfig(mode="majority")
    )
    
    assert panel.consensus.threshold == 2  # (3 // 2) + 1 = 2
    
    # 4 judges: majority = 3
    panel2 = JudgePanelConfig(
        judges=[
            JudgeConfig(name="judge_1", model="gpt-4"),
            JudgeConfig(name="judge_2", model="gpt-4-turbo"),
            JudgeConfig(name="judge_3", model="claude-3-5-sonnet"),
            JudgeConfig(name="judge_4", model="gpt-4")
        ],
        consensus=ConsensusConfig(mode="majority")
    )
    
    assert panel2.consensus.threshold == 3  # (4 // 2) + 1 = 3


def test_judge_panel_config_auto_sets_unanimous_threshold():
    """Test that unanimous mode auto-sets threshold to all judges."""
    from rubric_kit.schema import JudgePanelConfig, JudgeConfig, ConsensusConfig
    
    panel = JudgePanelConfig(
        judges=[
            JudgeConfig(name="judge_1", model="gpt-4"),
            JudgeConfig(name="judge_2", model="gpt-4-turbo"),
            JudgeConfig(name="judge_3", model="claude-3-5-sonnet")
        ],
        consensus=ConsensusConfig(mode="unanimous")
    )
    
    assert panel.consensus.threshold == 3  # All judges

