"""Score processing logic for rubric evaluation."""

from typing import Dict, List, Any
from rubric_kit.schema import Rubric, Criterion, Descriptor


def evaluate_binary_criterion(criterion: Criterion, passes: bool) -> Dict[str, Any]:
    """
    Evaluate a binary (pass/fail) criterion.
    
    Args:
        criterion: The criterion to evaluate
        passes: Whether the criterion passes
        
    Returns:
        Dictionary with evaluation results
    """
    weight = criterion.weight if isinstance(criterion.weight, int) else 0
    score = weight if passes else 0
    
    return {
        "criterion_name": criterion.name,
        "criterion_text": criterion.criterion,
        "category": criterion.category,
        "dimension": criterion.dimension,
        "result": "pass" if passes else "fail",
        "score": score,
        "max_score": weight
    }


def evaluate_score_criterion(
    criterion: Criterion, 
    descriptor: Descriptor, 
    score: int
) -> Dict[str, Any]:
    """
    Evaluate a score-based criterion.
    
    Args:
        criterion: The criterion to evaluate
        descriptor: The descriptor defining the score scale
        score: The score value (e.g., 1-3)
        
    Returns:
        Dictionary with evaluation results
    """
    if not descriptor.scores:
        raise ValueError(f"Descriptor '{descriptor.name}' does not have scores defined")
    
    max_score = max(descriptor.scores.keys())
    
    if score not in descriptor.scores:
        raise ValueError(f"Score {score} is not valid for descriptor '{descriptor.name}'")
    
    score_description = descriptor.scores.get(score, "")
    
    return {
        "criterion_name": criterion.name,
        "criterion_text": criterion.criterion,
        "category": criterion.category,
        "dimension": criterion.dimension,
        "result": score,
        "score": score,
        "max_score": max_score,
        "score_description": score_description
    }


def evaluate_rubric(rubric: Rubric, evaluations: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate a complete rubric with provided evaluation inputs.
    
    Args:
        rubric: The rubric to evaluate
        evaluations: Dictionary mapping criterion names to evaluation data
                    Format: {
                        "criterion_name": {
                            "type": "binary",  # or "score"
                            "passes": True,    # for binary
                            "score": 3         # for score
                        }
                    }
    
    Returns:
        List of evaluation results for each criterion
    """
    results = []
    
    for criterion in rubric.criteria:
        if criterion.name not in evaluations:
            raise ValueError(f"No evaluation provided for criterion '{criterion.name}'")
        
        eval_data = evaluations[criterion.name]
        eval_type = eval_data.get("type")
        
        if eval_type == "binary":
            passes = eval_data.get("passes", False)
            result = evaluate_binary_criterion(criterion, passes)
            results.append(result)
            
        elif eval_type == "score":
            score = eval_data.get("score")
            if score is None:
                raise ValueError(f"Score not provided for criterion '{criterion.name}'")
            
            # Find the descriptor for this criterion
            descriptor = rubric.get_descriptor(criterion.dimension)
            if not descriptor:
                raise ValueError(f"Descriptor '{criterion.dimension}' not found")
            
            result = evaluate_score_criterion(criterion, descriptor, score)
            results.append(result)
        else:
            raise ValueError(f"Unknown evaluation type '{eval_type}' for criterion '{criterion.name}'")
    
    return results


def calculate_total_score(results: List[Dict[str, Any]]) -> tuple[int, int]:
    """
    Calculate total score from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Tuple of (total_score, max_possible_score)
    """
    total = sum(r["score"] for r in results)
    max_total = sum(r["max_score"] for r in results)
    return total, max_total


def calculate_percentage_score(results: List[Dict[str, Any]]) -> float:
    """
    Calculate percentage score from evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Percentage score (0-100)
    """
    total, max_total = calculate_total_score(results)
    if max_total == 0:
        return 0.0
    return (total / max_total) * 100

