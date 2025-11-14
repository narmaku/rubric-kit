"""LLM-based judge for automatic criterion evaluation."""

import re
import os
from typing import Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

from rubric_kit.schema import Rubric, Criterion, Descriptor


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


def create_criterion_prompt(
    criterion: Criterion,
    chat_content: str,
    grading_type: str,
    descriptor: Optional[Descriptor] = None
) -> str:
    """
    Create a prompt for LLM to evaluate a criterion.
    
    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content
        grading_type: Type of grading ("binary" or "score")
        descriptor: Optional descriptor for score-based criteria
        
    Returns:
        Formatted prompt for the LLM
    """
    if grading_type == "binary":
        prompt = f"""You are an expert evaluator. Your task is to evaluate whether a chat session meets a specific criterion.

**Criterion Details:**
- Dimension: {criterion.dimension}
- Category: {criterion.category}
- Criterion: {criterion.criterion}

**Chat Session:**
{chat_content}

**Instructions:**
Carefully analyze the chat session above and determine if it meets the criterion.
Respond with ONLY one word: either "PASS" or "FAIL".

**Response:**"""
        
    else:  # score
        if not descriptor or not descriptor.scores:
            raise ValueError(f"Descriptor with scores required for score-based criterion")
        
        score_descriptions = "\n".join([f"{score}: {desc}" for score, desc in sorted(descriptor.scores.items())])
        
        prompt = f"""You are an expert evaluator. Your task is to score a chat session based on a specific criterion.

**Criterion Details:**
- Dimension: {criterion.dimension}
- Category: {criterion.category}
- Description: {descriptor.description}

**Scoring Scale:**
{score_descriptions}

**Chat Session:**
{chat_content}

**Instructions:**
Carefully analyze the chat session above and assign a score from {min(descriptor.scores.keys())} to {max(descriptor.scores.keys())} based on the scoring scale.
Respond with ONLY the numeric score (e.g., "1", "2", or "3").

**Response:**"""
    
    return prompt


def parse_binary_response(response: str) -> bool:
    """
    Parse LLM response for binary criterion.
    
    Args:
        response: LLM response text
        
    Returns:
        True if PASS, False if FAIL
    """
    response_upper = response.upper().strip()
    if "PASS" in response_upper:
        return True
    elif "FAIL" in response_upper:
        return False
    else:
        # Default to FAIL if unclear
        return False


def parse_score_response(response: str) -> int:
    """
    Parse LLM response for score criterion.
    
    Args:
        response: LLM response text
        
    Returns:
        Numeric score value
    """
    # Extract first number from response
    match = re.search(r'\d+', response.strip())
    if match:
        return int(match.group())
    else:
        raise ValueError(f"Could not parse score from response: {response}")


def evaluate_criterion_with_llm(
    criterion: Criterion,
    chat_content: str,
    grading_type: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4",
    descriptor: Optional[Descriptor] = None
) -> Dict[str, Any]:
    """
    Evaluate a single criterion using LLM.
    
    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content
        grading_type: Type of grading ("binary" or "score")
        api_key: OpenAI API key (or None to use environment variable)
        base_url: Optional custom base URL for OpenAI-compatible endpoint
        model: Model name to use
        descriptor: Optional descriptor for score-based criteria
        
    Returns:
        Evaluation result dictionary
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = OpenAI(**client_kwargs)
    
    # Create prompt
    prompt = create_criterion_prompt(criterion, chat_content, grading_type, descriptor)
    
    # Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=10
    )
    
    llm_response = response.choices[0].message.content.strip()
    
    # Parse response
    if grading_type == "binary":
        passes = parse_binary_response(llm_response)
        return {
            "type": "binary",
            "passes": passes
        }
    else:  # score
        score = parse_score_response(llm_response)
        return {
            "type": "score",
            "score": score
        }


def evaluate_rubric_with_llm(
    rubric: Rubric,
    chat_session_file: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4"
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all criteria in a rubric using LLM.
    
    Args:
        rubric: The rubric to evaluate
        chat_session_file: Path to the chat session file
        api_key: OpenAI API key (or None to use environment variable)
        base_url: Optional custom base URL for OpenAI-compatible endpoint
        model: Model name to use
        
    Returns:
        Dictionary mapping criterion names to evaluation results
    """
    # Read chat session
    chat_content = read_chat_session(chat_session_file)
    
    evaluations = {}
    
    for criterion in rubric.criteria:
        # Get descriptor for this criterion
        descriptor = rubric.get_descriptor(criterion.dimension)
        if not descriptor:
            raise ValueError(f"Descriptor '{criterion.dimension}' not found for criterion '{criterion.name}'")
        
        grading_type = descriptor.grading_type
        
        # Evaluate criterion
        evaluation = evaluate_criterion_with_llm(
            criterion,
            chat_content,
            grading_type,
            api_key=api_key,
            base_url=base_url,
            model=model,
            descriptor=descriptor if grading_type == "score" else None
        )
        
        evaluations[criterion.name] = evaluation
    
    return evaluations

