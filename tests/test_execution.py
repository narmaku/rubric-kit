"""Tests for judge execution strategies."""

import pytest
from unittest.mock import Mock, patch, call
import asyncio


# ============================================================================
# Mock Judge Function
# ============================================================================

def mock_judge_function(judge_config, criterion, chat_content, dimension=None, parsed_session=None, metrics=None):
    """Mock judge function that returns a deterministic result."""
    # Return result based on judge name
    if "pass" in judge_config.name:
        return {"type": "binary", "passes": True, "reason": f"Pass from {judge_config.name}"}
    elif "fail" in judge_config.name:
        return {"type": "binary", "passes": False, "reason": f"Fail from {judge_config.name}"}
    elif "score" in judge_config.name:
        # Extract score from name like "score_2"
        score = int(judge_config.name.split("_")[1])
        return {"type": "score", "score": score, "reason": f"Score {score} from {judge_config.name}"}
    else:
        return {"type": "binary", "passes": True, "reason": f"Result from {judge_config.name}"}


# ============================================================================
# Sequential Execution Tests
# ============================================================================

def test_execute_judges_sequential():
    """Test sequential execution of judges."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [
        JudgeConfig(name="judge_pass_1", model="gpt-4"),
        JudgeConfig(name="judge_pass_2", model="gpt-4-turbo"),
        JudgeConfig(name="judge_fail_1", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="sequential",
        criterion=None,
        chat_content="test content"
    )
    
    assert len(results) == 3
    assert results[0]["judge"] == "judge_pass_1"
    assert results[0]["passes"] is True
    assert results[1]["judge"] == "judge_pass_2"
    assert results[1]["passes"] is True
    assert results[2]["judge"] == "judge_fail_1"
    assert results[2]["passes"] is False


def test_execute_judges_sequential_with_timeout():
    """Test sequential execution respects timeout."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    import time
    
    def slow_judge_function(judge_config, criterion, chat_content, dimension=None, parsed_session=None, metrics=None):
        """Judge function that takes time."""
        time.sleep(0.1)
        return {"type": "binary", "passes": True, "reason": "Slow result"}
    
    judges = [JudgeConfig(name="judge_1", model="gpt-4")]
    
    # Should complete successfully with sufficient timeout
    results = execute_judges(
        judges=judges,
        judge_function=slow_judge_function,
        execution_mode="sequential",
        timeout=1,
        criterion=None,
        chat_content="test"
    )
    
    assert len(results) == 1
    assert results[0]["passes"] is True


def test_execute_judges_sequential_single_judge():
    """Test sequential execution with single judge."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [JudgeConfig(name="judge_pass_1", model="gpt-4")]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="sequential",
        criterion=None,
        chat_content="test"
    )
    
    assert len(results) == 1
    assert results[0]["judge"] == "judge_pass_1"
    assert results[0]["passes"] is True


# ============================================================================
# Parallel Execution Tests
# ============================================================================

def test_execute_judges_parallel():
    """Test parallel execution of judges."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [
        JudgeConfig(name="judge_pass_1", model="gpt-4"),
        JudgeConfig(name="judge_pass_2", model="gpt-4-turbo"),
        JudgeConfig(name="judge_fail_1", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="parallel",
        criterion=None,
        chat_content="test content"
    )
    
    assert len(results) == 3
    # Results may not be in order for parallel, so check by judge name
    result_dict = {r["judge"]: r for r in results}
    assert result_dict["judge_pass_1"]["passes"] is True
    assert result_dict["judge_pass_2"]["passes"] is True
    assert result_dict["judge_fail_1"]["passes"] is False


def test_execute_judges_parallel_maintains_order():
    """Test that parallel execution maintains judge order in results."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [
        JudgeConfig(name="judge_1", model="gpt-4"),
        JudgeConfig(name="judge_2", model="gpt-4-turbo"),
        JudgeConfig(name="judge_3", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="parallel",
        criterion=None,
        chat_content="test"
    )
    
    # Results should be in same order as judges list
    assert results[0]["judge"] == "judge_1"
    assert results[1]["judge"] == "judge_2"
    assert results[2]["judge"] == "judge_3"


# ============================================================================
# Batched Execution Tests
# ============================================================================

def test_execute_judges_batched():
    """Test batched execution of judges."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [
        JudgeConfig(name="judge_pass_1", model="gpt-4"),
        JudgeConfig(name="judge_pass_2", model="gpt-4-turbo"),
        JudgeConfig(name="judge_fail_1", model="gpt-4"),
        JudgeConfig(name="judge_pass_3", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="batched",
        batch_size=2,
        criterion=None,
        chat_content="test"
    )
    
    assert len(results) == 4
    # Results should maintain order
    assert results[0]["judge"] == "judge_pass_1"
    assert results[1]["judge"] == "judge_pass_2"
    assert results[2]["judge"] == "judge_fail_1"
    assert results[3]["judge"] == "judge_pass_3"


def test_execute_judges_batched_uneven_batches():
    """Test batched execution with uneven batch sizes."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    # 5 judges with batch_size=2 => 3 batches (2, 2, 1)
    judges = [
        JudgeConfig(name="judge_1", model="gpt-4"),
        JudgeConfig(name="judge_2", model="gpt-4"),
        JudgeConfig(name="judge_3", model="gpt-4"),
        JudgeConfig(name="judge_4", model="gpt-4"),
        JudgeConfig(name="judge_5", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="batched",
        batch_size=2,
        criterion=None,
        chat_content="test"
    )
    
    assert len(results) == 5
    assert results[0]["judge"] == "judge_1"
    assert results[4]["judge"] == "judge_5"


def test_execute_judges_batched_single_batch():
    """Test batched execution where batch_size >= num judges."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [
        JudgeConfig(name="judge_1", model="gpt-4"),
        JudgeConfig(name="judge_2", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="batched",
        batch_size=5,  # Larger than number of judges
        criterion=None,
        chat_content="test"
    )
    
    assert len(results) == 2


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_execute_judges_with_failing_judge():
    """Test execution continues when one judge fails."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    def failing_judge_function(judge_config, criterion, chat_content, dimension=None, parsed_session=None, metrics=None):
        """Judge function that fails for specific judge."""
        if judge_config.name == "judge_error":
            raise Exception("Judge evaluation failed")
        return {"type": "binary", "passes": True, "reason": "OK"}
    
    judges = [
        JudgeConfig(name="judge_1", model="gpt-4"),
        JudgeConfig(name="judge_error", model="gpt-4"),
        JudgeConfig(name="judge_3", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=failing_judge_function,
        execution_mode="sequential",
        criterion=None,
        chat_content="test"
    )
    
    # Should have 3 results, with error result for failing judge
    assert len(results) == 3
    assert results[0]["judge"] == "judge_1"
    assert results[0]["passes"] is True
    
    assert results[1]["judge"] == "judge_error"
    assert results[1]["error"] is not None
    assert "failed" in results[1]["error"].lower()
    
    assert results[2]["judge"] == "judge_3"
    assert results[2]["passes"] is True


def test_execute_judges_invalid_mode():
    """Test that invalid execution mode raises error."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [JudgeConfig(name="judge_1", model="gpt-4")]
    
    with pytest.raises(ValueError, match="Invalid execution mode"):
        execute_judges(
            judges=judges,
            judge_function=mock_judge_function,
            execution_mode="invalid",
            criterion=None,
            chat_content="test"
        )


def test_execute_judges_empty_judges_list():
    """Test that empty judges list raises error."""
    from rubric_kit.execution import execute_judges
    
    with pytest.raises(ValueError, match="No judges provided"):
        execute_judges(
            judges=[],
            judge_function=mock_judge_function,
            execution_mode="sequential",
            criterion=None,
            chat_content="test"
        )


# ============================================================================
# Score-based Execution Tests
# ============================================================================

def test_execute_judges_with_scores():
    """Test execution with score-based criteria."""
    from rubric_kit.execution import execute_judges
    from rubric_kit.schema import JudgeConfig
    
    judges = [
        JudgeConfig(name="score_1", model="gpt-4"),
        JudgeConfig(name="score_2", model="gpt-4"),
        JudgeConfig(name="score_3", model="gpt-4")
    ]
    
    results = execute_judges(
        judges=judges,
        judge_function=mock_judge_function,
        execution_mode="sequential",
        criterion=None,
        chat_content="test"
    )
    
    assert len(results) == 3
    assert results[0]["score"] == 1
    assert results[1]["score"] == 2
    assert results[2]["score"] == 3

