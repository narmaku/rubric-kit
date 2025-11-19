"""Consensus logic for multi-judge evaluation."""

from typing import Dict, List, Any, Literal
from collections import Counter


def _validate_consensus_inputs(votes: List[Dict[str, Any]], threshold: int) -> None:
    """Validate inputs for consensus functions.
    
    Raises:
        ValueError: If votes is empty, threshold is invalid, or threshold exceeds votes
    """
    if not votes:
        raise ValueError("No votes provided")
    
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    
    if threshold > len(votes):
        raise ValueError(f"Threshold ({threshold}) exceeds number of votes ({len(votes)})")


def _resolve_binary_no_consensus(
    pass_count: int,
    fail_count: int,
    on_no_consensus: Literal["fail", "most_common"]
) -> bool:
    """Resolve binary consensus when threshold is not met.
    
    Args:
        pass_count: Number of pass votes
        fail_count: Number of fail votes
        on_no_consensus: Strategy to use when no consensus
        
    Returns:
        Final pass/fail decision
    """
    if on_no_consensus == "most_common":
        return pass_count > fail_count
    
    return False  # Conservative: fail


def _calculate_median_score(scores: List[int]) -> int:
    """Calculate median score from a list of scores.
    
    Args:
        scores: List of scores
        
    Returns:
        Median score (integer)
    """
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    
    if n % 2 == 0:
        # Even number: average of two middle values
        return (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) // 2
    
    # Odd number: middle value
    return sorted_scores[n // 2]


def _resolve_score_no_consensus(
    scores: List[int],
    score_counts: Counter,
    most_common_count: int,
    on_no_consensus: Literal["fail", "median", "most_common"]
) -> int:
    """Resolve score consensus when threshold is not met.
    
    Args:
        scores: List of all scores
        score_counts: Counter of score frequencies
        most_common_count: Count of the most common score
        on_no_consensus: Strategy to use when no consensus
        
    Returns:
        Final score decision
    """
    if on_no_consensus == "median":
        return _calculate_median_score(scores)
    
    if on_no_consensus == "most_common":
        max_count = max(score_counts.values())
        tied_scores = [score for score, count in score_counts.items() if count == max_count]
        return min(tied_scores)  # Conservative: minimum when tied
    
    # "fail" - conservative approach (minimum score)
    return min(scores)


def apply_binary_consensus(
    votes: List[Dict[str, Any]],
    threshold: int,
    on_no_consensus: Literal["fail", "most_common"] = "fail"
) -> Dict[str, Any]:
    """
    Apply consensus logic to binary (pass/fail) votes.
    
    Args:
        votes: List of judge votes, each with 'judge', 'passes', 'reason'
        threshold: Minimum number of judges that must agree
        on_no_consensus: How to handle no consensus ("fail" or "most_common")
        
    Returns:
        Dictionary with:
            - consensus_reached: bool
            - passes: bool (final decision)
            - consensus_count: int (number of judges who agreed on the result)
            - judge_votes: List[Dict] (all individual votes)
            
    Raises:
        ValueError: If votes is empty, threshold is invalid, or threshold exceeds votes
    """
    _validate_consensus_inputs(votes, threshold)
    
    pass_count = sum(1 for v in votes if v["passes"])
    fail_count = len(votes) - pass_count
    
    # Early return for pass consensus
    if pass_count >= threshold:
        return {
            "consensus_reached": True,
            "passes": True,
            "consensus_count": pass_count,
            "judge_votes": votes
        }
    
    # Early return for fail consensus
    if fail_count >= threshold:
        return {
            "consensus_reached": True,
            "passes": False,
            "consensus_count": fail_count,
            "judge_votes": votes
        }
    
    # No consensus reached
    max_count = max(pass_count, fail_count)
    passes = _resolve_binary_no_consensus(pass_count, fail_count, on_no_consensus)
    
    return {
        "consensus_reached": False,
        "passes": passes,
        "consensus_count": max_count,
        "judge_votes": votes
    }


def apply_score_consensus(
    votes: List[Dict[str, Any]],
    threshold: int,
    on_no_consensus: Literal["fail", "median", "most_common"] = "fail"
) -> Dict[str, Any]:
    """
    Apply consensus logic to score-based votes.
    
    Args:
        votes: List of judge votes, each with 'judge', 'score', 'reason'
        threshold: Minimum number of judges that must agree on a score
        on_no_consensus: How to handle no consensus ("fail", "median", "most_common")
        
    Returns:
        Dictionary with:
            - consensus_reached: bool
            - score: int (final score)
            - consensus_count: int (number of judges who agreed on the score)
            - judge_votes: List[Dict] (all individual votes)
            
    Raises:
        ValueError: If votes is empty, threshold is invalid, or threshold exceeds votes
    """
    _validate_consensus_inputs(votes, threshold)
    
    scores = [v["score"] for v in votes]
    score_counts = Counter(scores)
    most_common_score, most_common_count = score_counts.most_common(1)[0]
    
    # Early return for consensus
    if most_common_count >= threshold:
        return {
            "consensus_reached": True,
            "score": most_common_score,
            "consensus_count": most_common_count,
            "judge_votes": votes
        }
    
    # No consensus reached - apply fallback strategy
    final_score = _resolve_score_no_consensus(
        scores, score_counts, most_common_count, on_no_consensus
    )
    
    return {
        "consensus_reached": False,
        "score": final_score,
        "consensus_count": most_common_count,
        "judge_votes": votes
    }

