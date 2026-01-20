"""Tests for consensus logic."""

import pytest


# ============================================================================
# Binary Consensus Tests
# ============================================================================


def test_binary_consensus_unanimous_all_pass():
    """Test unanimous consensus when all judges vote pass."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": True, "reason": "Good"},
        {"judge": "judge_2", "passes": True, "reason": "Good"},
        {"judge": "judge_3", "passes": True, "reason": "Good"},
    ]
    threshold = 3

    result = apply_binary_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["passes"] is True
    assert result["consensus_count"] == 3


def test_binary_consensus_unanimous_all_fail():
    """Test unanimous consensus when all judges vote fail."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": False, "reason": "Bad"},
        {"judge": "judge_2", "passes": False, "reason": "Bad"},
        {"judge": "judge_3", "passes": False, "reason": "Bad"},
    ]
    threshold = 3

    result = apply_binary_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["passes"] is False
    assert result["consensus_count"] == 3


def test_binary_consensus_quorum_reached_pass():
    """Test quorum consensus when threshold is met for pass."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": True, "reason": "Good"},
        {"judge": "judge_2", "passes": True, "reason": "Good"},
        {"judge": "judge_3", "passes": False, "reason": "Bad"},
    ]
    threshold = 2

    result = apply_binary_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["passes"] is True
    assert result["consensus_count"] == 2


def test_binary_consensus_quorum_reached_fail():
    """Test quorum consensus when threshold is met for fail."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": False, "reason": "Bad"},
        {"judge": "judge_2", "passes": False, "reason": "Bad"},
        {"judge": "judge_3", "passes": True, "reason": "Good"},
    ]
    threshold = 2

    result = apply_binary_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["passes"] is False
    assert result["consensus_count"] == 2


def test_binary_consensus_not_reached_conservative():
    """Test no consensus with conservative (fail) fallback."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": True, "reason": "Good"},
        {"judge": "judge_2", "passes": False, "reason": "Bad"},
        {"judge": "judge_3", "passes": True, "reason": "Good"},
    ]
    threshold = 3  # Needs all 3, but only 2 agree on pass

    result = apply_binary_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is False
    assert result["passes"] is False  # Conservative: fail
    assert result["consensus_count"] == 2  # Max agreement count


def test_binary_consensus_not_reached_most_common():
    """Test no consensus with most_common fallback."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": True, "reason": "Good"},
        {"judge": "judge_2", "passes": True, "reason": "Good"},
        {"judge": "judge_3", "passes": False, "reason": "Bad"},
    ]
    threshold = 3  # Needs all 3, but only 2 agree on pass

    result = apply_binary_consensus(votes, threshold, on_no_consensus="most_common")

    assert result["consensus_reached"] is False
    assert result["passes"] is True  # Most common: pass (2 out of 3)
    assert result["consensus_count"] == 2


def test_binary_consensus_single_judge():
    """Test consensus with single judge."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [{"judge": "judge_1", "passes": True, "reason": "Good"}]
    threshold = 1

    result = apply_binary_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["passes"] is True
    assert result["consensus_count"] == 1


# ============================================================================
# Score Consensus Tests
# ============================================================================


def test_score_consensus_unanimous_all_agree():
    """Test unanimous consensus when all judges give same score."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 3, "reason": "Excellent"},
        {"judge": "judge_2", "score": 3, "reason": "Excellent"},
        {"judge": "judge_3", "score": 3, "reason": "Excellent"},
    ]
    threshold = 3

    result = apply_score_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["score"] == 3
    assert result["consensus_count"] == 3


def test_score_consensus_quorum_reached():
    """Test quorum consensus when threshold is met for a score."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 2, "reason": "Good"},
        {"judge": "judge_2", "score": 2, "reason": "Good"},
        {"judge": "judge_3", "score": 3, "reason": "Excellent"},
    ]
    threshold = 2

    result = apply_score_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["score"] == 2
    assert result["consensus_count"] == 2


def test_score_consensus_not_reached_conservative():
    """Test no consensus with conservative (minimum score) fallback."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 1, "reason": "Poor"},
        {"judge": "judge_2", "score": 2, "reason": "Good"},
        {"judge": "judge_3", "score": 3, "reason": "Excellent"},
    ]
    threshold = 2  # No score has 2 votes

    result = apply_score_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is False
    assert result["score"] == 1  # Conservative: minimum score
    assert result["consensus_count"] == 1  # Each score only appears once


def test_score_consensus_not_reached_median():
    """Test no consensus with median fallback."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 1, "reason": "Poor"},
        {"judge": "judge_2", "score": 2, "reason": "Good"},
        {"judge": "judge_3", "score": 3, "reason": "Excellent"},
    ]
    threshold = 2

    result = apply_score_consensus(votes, threshold, on_no_consensus="median")

    assert result["consensus_reached"] is False
    assert result["score"] == 2  # Median of [1, 2, 3]
    assert result["consensus_count"] == 1


def test_score_consensus_not_reached_median_even_count():
    """Test no consensus with median fallback for even number of judges."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 1, "reason": "Poor"},
        {"judge": "judge_2", "score": 2, "reason": "Good"},
        {"judge": "judge_3", "score": 2, "reason": "Good"},
        {"judge": "judge_4", "score": 3, "reason": "Excellent"},
    ]
    threshold = 3  # No score has 3 votes

    result = apply_score_consensus(votes, threshold, on_no_consensus="median")

    assert result["consensus_reached"] is False
    # Median of [1, 2, 2, 3] = (2 + 2) / 2 = 2
    assert result["score"] == 2
    assert result["consensus_count"] == 2  # Score 2 appears twice


def test_score_consensus_not_reached_most_common():
    """Test no consensus with most_common fallback."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 1, "reason": "Poor"},
        {"judge": "judge_2", "score": 2, "reason": "Good"},
        {"judge": "judge_3", "score": 2, "reason": "Good"},
        {"judge": "judge_4", "score": 3, "reason": "Excellent"},
    ]
    threshold = 3  # No score has 3 votes

    result = apply_score_consensus(votes, threshold, on_no_consensus="most_common")

    assert result["consensus_reached"] is False
    assert result["score"] == 2  # Most common: appears twice
    assert result["consensus_count"] == 2


def test_score_consensus_single_judge():
    """Test consensus with single judge."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [{"judge": "judge_1", "score": 3, "reason": "Excellent"}]
    threshold = 1

    result = apply_score_consensus(votes, threshold, on_no_consensus="fail")

    assert result["consensus_reached"] is True
    assert result["score"] == 3
    assert result["consensus_count"] == 1


# ============================================================================
# Edge Cases
# ============================================================================


def test_binary_consensus_tie_with_most_common():
    """Test binary consensus tie (1-1) with most_common fallback."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": True, "reason": "Good"},
        {"judge": "judge_2", "passes": False, "reason": "Bad"},
    ]
    threshold = 2  # Needs both to agree

    result = apply_binary_consensus(votes, threshold, on_no_consensus="most_common")

    assert result["consensus_reached"] is False
    # Tie: fallback to fail (conservative when most_common can't decide)
    assert result["passes"] is False
    assert result["consensus_count"] == 1


def test_score_consensus_multiple_scores_tied():
    """Test score consensus when multiple scores are tied for most common."""
    from rubric_kit.consensus import apply_score_consensus

    votes = [
        {"judge": "judge_1", "score": 1, "reason": "Poor"},
        {"judge": "judge_2", "score": 1, "reason": "Poor"},
        {"judge": "judge_3", "score": 3, "reason": "Excellent"},
        {"judge": "judge_4", "score": 3, "reason": "Excellent"},
    ]
    threshold = 3  # No score has 3 votes

    result = apply_score_consensus(votes, threshold, on_no_consensus="most_common")

    assert result["consensus_reached"] is False
    # Tie between 1 and 3 (both appear twice): take minimum (conservative)
    assert result["score"] == 1
    assert result["consensus_count"] == 2


def test_empty_votes_raises_error():
    """Test that empty votes list raises error."""
    from rubric_kit.consensus import apply_binary_consensus

    with pytest.raises(ValueError, match="No votes provided"):
        apply_binary_consensus([], threshold=1, on_no_consensus="fail")


def test_invalid_threshold_raises_error():
    """Test that invalid threshold raises error."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [{"judge": "judge_1", "passes": True, "reason": "Good"}]

    with pytest.raises(ValueError, match="Threshold must be positive"):
        apply_binary_consensus(votes, threshold=0, on_no_consensus="fail")

    with pytest.raises(ValueError, match="Threshold must be positive"):
        apply_binary_consensus(votes, threshold=-1, on_no_consensus="fail")


def test_threshold_exceeds_votes_raises_error():
    """Test that threshold exceeding number of votes raises error."""
    from rubric_kit.consensus import apply_binary_consensus

    votes = [
        {"judge": "judge_1", "passes": True, "reason": "Good"},
        {"judge": "judge_2", "passes": True, "reason": "Good"},
    ]

    with pytest.raises(ValueError, match="Threshold.*exceeds number of votes"):
        apply_binary_consensus(votes, threshold=3, on_no_consensus="fail")
