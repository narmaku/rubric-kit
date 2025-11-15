"""Consensus logic for multi-judge evaluation."""

from typing import Dict, List, Any, Literal
from collections import Counter
import statistics


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
    # Validate inputs
    if not votes:
        raise ValueError("No votes provided")
    
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    
    if threshold > len(votes):
        raise ValueError(f"Threshold ({threshold}) exceeds number of votes ({len(votes)})")
    
    # Count votes
    pass_count = sum(1 for v in votes if v["passes"])
    fail_count = len(votes) - pass_count
    
    # Check for consensus
    if pass_count >= threshold:
        return {
            "consensus_reached": True,
            "passes": True,
            "consensus_count": pass_count,
            "judge_votes": votes
        }
    elif fail_count >= threshold:
        return {
            "consensus_reached": True,
            "passes": False,
            "consensus_count": fail_count,
            "judge_votes": votes
        }
    else:
        # No consensus reached
        max_count = max(pass_count, fail_count)
        
        if on_no_consensus == "most_common":
            # Use most common vote (if tie, default to fail/conservative)
            passes = pass_count > fail_count
        else:  # "fail" - conservative approach
            passes = False
        
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
    # Validate inputs
    if not votes:
        raise ValueError("No votes provided")
    
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    
    if threshold > len(votes):
        raise ValueError(f"Threshold ({threshold}) exceeds number of votes ({len(votes)})")
    
    # Count scores
    scores = [v["score"] for v in votes]
    score_counts = Counter(scores)
    most_common_score, most_common_count = score_counts.most_common(1)[0]
    
    # Check for consensus
    if most_common_count >= threshold:
        return {
            "consensus_reached": True,
            "score": most_common_score,
            "consensus_count": most_common_count,
            "judge_votes": votes
        }
    else:
        # No consensus reached - apply fallback strategy
        if on_no_consensus == "median":
            # Use median score
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            if n % 2 == 0:
                # Even number: average of two middle values
                median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) // 2
            else:
                # Odd number: middle value
                median = sorted_scores[n // 2]
            final_score = median
        elif on_no_consensus == "most_common":
            # Use most common score (if tie, use minimum for conservative)
            # Get all scores with max count
            max_count = max(score_counts.values())
            tied_scores = [score for score, count in score_counts.items() if count == max_count]
            final_score = min(tied_scores)  # Conservative: minimum when tied
        else:  # "fail" - conservative approach (minimum score)
            final_score = min(scores)
        
        return {
            "consensus_reached": False,
            "score": final_score,
            "consensus_count": most_common_count,
            "judge_votes": votes
        }

