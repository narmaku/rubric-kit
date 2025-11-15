"""LLM-based judge panel for automatic criterion evaluation."""

import re
import os
from typing import Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

from rubric_kit.schema import Rubric, Criterion, Dimension, JudgePanelConfig, JudgeConfig
from rubric_kit.prompts import (
    EVALUATOR_CONFIG,
    TOOL_CALL_EVALUATOR_CONFIG,
    build_binary_criterion_prompt,
    build_score_criterion_prompt,
    build_tool_call_evaluation_prompt,
)
from rubric_kit.execution import execute_judges
from rubric_kit.consensus import apply_binary_consensus, apply_score_consensus


def read_chat_session(file_path: str) -> str:
    """
    Read chat session from a plain text file.
    
    Args:
        file_path: Path to the chat session file
        
    Returns:
        Content of the chat session
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def parse_binary_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response for binary criterion.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with 'passes' (bool) and 'reason' (str)
    """
    response_upper = response.upper().strip()
    
    # Extract result
    passes = False
    if "RESULT:" in response_upper:
        result_line = response_upper.split("RESULT:")[1].split("\n")[0].strip()
        passes = "PASS" in result_line
    elif "PASS" in response_upper:
        passes = True
    elif "FAIL" in response_upper:
        passes = False
    
    # Extract reason
    reason = ""
    if "REASON:" in response.upper():
        reason_parts = response.split("REASON:")
        if len(reason_parts) > 1:
            reason = reason_parts[1].strip()
    
    return {
        "passes": passes,
        "reason": reason
    }


def parse_score_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response for score criterion.
    
    Args:
        response: LLM response text
        
    Returns:
        Dictionary with 'score' (int) and 'reason' (str)
    """
    score = None
    reason = ""
    
    # Extract score
    if "SCORE:" in response.upper():
        score_line = response.upper().split("SCORE:")[1].split("\n")[0].strip()
        match = re.search(r'\d+', score_line)
        if match:
            score = int(match.group())
    else:
        # Fallback: extract first number from response
        match = re.search(r'\d+', response.strip())
        if match:
            score = int(match.group())
    
    if score is None:
        raise ValueError(f"Could not parse score from response: {response}")
    
    # Extract reason
    if "REASON:" in response.upper():
        reason_parts = response.split("REASON:")
        if len(reason_parts) > 1:
            reason = reason_parts[1].strip()
    
    return {
        "score": score,
        "reason": reason
    }


def _single_judge_evaluate(
    judge_config: JudgeConfig,
    criterion: Criterion,
    chat_content: str,
    dimension: Optional[Dimension]
) -> Dict[str, Any]:
    """
    Evaluate a criterion using a single judge.
    
    This is the function passed to execute_judges for each judge.
    
    Args:
        judge_config: Configuration for this judge
        criterion: The criterion to evaluate
        chat_content: The chat session content
        dimension: Optional dimension for score-based criteria
        
    Returns:
        Evaluation result dictionary
    """
    # Get API key (from judge config or environment)
    api_key = judge_config.api_key
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or provide in judge config.")
    
    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if judge_config.base_url:
        client_kwargs["base_url"] = judge_config.base_url
    
    client = OpenAI(**client_kwargs)
    
    # Determine evaluation type and build prompt
    grading_type = dimension.grading_type if dimension else "binary"
    
    if criterion.tool_calls:
        prompt = build_tool_call_evaluation_prompt(criterion, chat_content)
        evaluation_type = "tool_call"
        config = TOOL_CALL_EVALUATOR_CONFIG
    elif grading_type == "binary":
        prompt = build_binary_criterion_prompt(criterion, chat_content)
        evaluation_type = "binary"
        config = EVALUATOR_CONFIG
    else:  # score
        if not dimension:
            raise ValueError(f"Dimension required for score-based criterion")
        prompt = build_score_criterion_prompt(criterion, chat_content, dimension)
        evaluation_type = "score"
        config = EVALUATOR_CONFIG
    
    # Call LLM
    response = client.chat.completions.create(
        model=judge_config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    
    llm_response = response.choices[0].message.content.strip()
    
    # Parse response based on evaluation type
    if evaluation_type in ("binary", "tool_call"):
        parsed = parse_binary_response(llm_response)
        return {
            "passes": parsed["passes"],
            "reason": parsed["reason"]
        }
    else:  # score
        parsed = parse_score_response(llm_response)
        return {
            "score": parsed["score"],
            "reason": parsed["reason"]
        }


def evaluate_criterion_with_panel(
    criterion: Criterion,
    chat_content: str,
    dimension: Optional[Dimension],
    panel_config: JudgePanelConfig
) -> Dict[str, Any]:
    """
    Evaluate a single criterion using a judge panel.
    
    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content
        dimension: Dimension for score-based criteria (optional)
        panel_config: Judge panel configuration
        
    Returns:
        Evaluation result dictionary with consensus information:
        For binary: {"type": "binary", "passes": bool, "consensus_reached": bool, 
                    "consensus_count": int, "judge_votes": List[Dict], "reason": str}
        For score: {"type": "score", "score": int, "consensus_reached": bool,
                   "consensus_count": int, "judge_votes": List[Dict], "reason": str}
    """
    # Execute all judges
    judge_results = execute_judges(
        judges=panel_config.judges,
        judge_function=_single_judge_evaluate,
        execution_mode=panel_config.execution.mode,
        criterion=criterion,
        chat_content=chat_content,
        dimension=dimension,
        batch_size=panel_config.execution.batch_size,
        timeout=panel_config.execution.timeout
    )
    
    # Check for errors in judge results
    errors = [r for r in judge_results if "error" in r]
    if errors:
        # For now, raise an exception if any judge fails
        # Could be enhanced to handle partial failures
        error_msgs = [f"{r['judge']}: {r['error']}" for r in errors]
        raise Exception(f"Judge evaluation failed: {'; '.join(error_msgs)}")
    
    # Determine evaluation type
    grading_type = dimension.grading_type if dimension else "binary"
    
    # Apply consensus logic
    if grading_type == "binary" or criterion.tool_calls:
        # Binary consensus
        consensus_result = apply_binary_consensus(
            votes=judge_results,
            threshold=panel_config.consensus.threshold,
            on_no_consensus=panel_config.consensus.on_no_consensus
        )
        
        # Build reason from judge votes
        reason = _build_consensus_reason(consensus_result)
        
        return {
            "type": "binary",
            "passes": consensus_result["passes"],
            "consensus_reached": consensus_result["consensus_reached"],
            "consensus_count": consensus_result["consensus_count"],
            "judge_votes": consensus_result["judge_votes"],
            "reason": reason
        }
    else:  # score
        # Score consensus
        consensus_result = apply_score_consensus(
            votes=judge_results,
            threshold=panel_config.consensus.threshold,
            on_no_consensus=panel_config.consensus.on_no_consensus
        )
        
        # Build reason from judge votes
        reason = _build_consensus_reason(consensus_result)
        
        return {
            "type": "score",
            "score": consensus_result["score"],
            "consensus_reached": consensus_result["consensus_reached"],
            "consensus_count": consensus_result["consensus_count"],
            "judge_votes": consensus_result["judge_votes"],
            "reason": reason
        }


def _build_consensus_reason(consensus_result: Dict[str, Any]) -> str:
    """
    Build a human-readable reason from consensus result.
    
    Extracts reasons from judges that agreed on the final result.
    
    Args:
        consensus_result: Result from apply_binary_consensus or apply_score_consensus
        
    Returns:
        Combined reason string from agreeing judges
    """
    judge_votes = consensus_result["judge_votes"]
    
    if len(judge_votes) == 1:
        # Single judge: use their reason directly
        return judge_votes[0].get("reason", "")
    
    # Multiple judges: combine reasons from agreeing judges
    # Determine what the final decision was
    if "passes" in consensus_result:
        # Binary criterion
        final_result = consensus_result["passes"]
        agreeing_judges = [v for v in judge_votes if v.get("passes") == final_result]
    else:
        # Score criterion
        final_score = consensus_result["score"]
        agreeing_judges = [v for v in judge_votes if v.get("score") == final_score]
    
    # Combine reasons from agreeing judges
    if agreeing_judges:
        reasons = [v.get("reason", "") for v in agreeing_judges if v.get("reason")]
        if reasons:
            # Join with semicolons and truncate if too long
            combined = "; ".join(reasons)
            if len(combined) > 500:
                combined = combined[:497] + "..."
            return combined
    
    # Fallback: use first judge's reason
    return judge_votes[0].get("reason", "") if judge_votes else ""


def evaluate_rubric_with_panel(
    rubric: Rubric,
    chat_session_file: str,
    panel_config: JudgePanelConfig
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all criteria in a rubric using a judge panel.
    
    Args:
        rubric: The rubric to evaluate
        chat_session_file: Path to the chat session file
        panel_config: Judge panel configuration
        
    Returns:
        Dictionary mapping criterion names to evaluation results
    """
    # Read chat session
    chat_content = read_chat_session(chat_session_file)
    
    evaluations = {}
    
    for criterion in rubric.criteria:
        # Get dimension for this criterion
        dimension = rubric.get_dimension(criterion.dimension)
        if not dimension:
            raise ValueError(f"Dimension '{criterion.dimension}' not found for criterion '{criterion.name}'")
        
        # Evaluate criterion with panel
        evaluation = evaluate_criterion_with_panel(
            criterion=criterion,
            chat_content=chat_content,
            dimension=dimension,
            panel_config=panel_config
        )
        
        evaluations[criterion.name] = evaluation
    
    return evaluations
