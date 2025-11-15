"""Rubric generation using LLM."""

import json
import re
import yaml
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

from openai import OpenAI

from rubric_kit.schema import Rubric, Dimension, Criterion
from rubric_kit.prompts import (
    GENERATOR_CONFIG,
    build_dimension_generation_prompt,
    build_criteria_generation_prompt,
    build_refine_rubric_prompt,
    build_chat_dimension_generation_prompt,
    build_chat_criteria_generation_prompt,
)


@dataclass
class QAInput:
    """Question and Answer input for rubric generation."""
    question: str
    answer: str
    context: Optional[str] = None


@dataclass
class ChatSessionInput:
    """Chat session input for rubric generation."""
    content: str
    context: Optional[str] = None


def parse_qa_input(file_path: str) -> QAInput:
    """
    Parse Q&A input from a YAML file.
    
    Only supports YAML format with the following structure:
    ```yaml
    question: "The question text"
    answer: "The answer text"
    context: "Optional context"  # Optional
    ```
    
    Args:
        file_path: Path to Q&A YAML file
        
    Returns:
        QAInput object
        
    Raises:
        ValueError: If file is empty, missing required fields, or not valid YAML
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Q&A file not found: {file_path}")
    
    content = path.read_text().strip()
    
    if not content:
        raise ValueError("Q&A file is empty")
    
    # Parse YAML format
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("YAML file must contain a dictionary with 'question' and 'answer' keys")
        
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        context = data.get("context")
        
        if not question:
            raise ValueError("Required 'question' key not found or empty in YAML file")
        if not answer:
            raise ValueError("Required 'answer' key not found or empty in YAML file")
        
        # Handle multi-line answers (YAML block scalars)
        if isinstance(answer, str):
            answer = answer.strip()
        else:
            # If answer is not a string, convert to string
            answer = str(answer).strip()
        
        # Handle context similarly
        if context is not None and isinstance(context, str):
            context = context.strip() if context else None
        
        return QAInput(question=question, answer=answer, context=context)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}")


def parse_chat_session(file_path: str) -> ChatSessionInput:
    """
    Parse chat session input from a file.
    
    Simply reads the entire chat session as-is, allowing the LLM to understand
    any format (Cursor exports, ChatGPT exports, Claude exports, etc.)
    
    Args:
        file_path: Path to chat session file
        
    Returns:
        ChatSessionInput object with raw content
        
    Raises:
        ValueError: If file is empty
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Chat session file not found: {file_path}")
    
    content = path.read_text().strip()
    
    if not content:
        raise ValueError("Chat session file is empty")
    
    return ChatSessionInput(content=content, context=None)


def repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues in LLM output.
    
    Args:
        text: JSON string that may have common issues
        
    Returns:
        Repaired JSON string
    """
    # Remove trailing commas before closing brackets/braces
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Remove comments (// style and /* */ style)
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Fix unquoted keys (common LLM mistake)
    # This is a simple heuristic - match word characters followed by colon
    text = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', text)
    
    return text


class RubricGenerator:
    """Generate rubrics from Q&A pairs using LLM."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", base_url: Optional[str] = None):
        """
        Initialize RubricGenerator.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            base_url: Optional base URL for OpenAI-compatible endpoint
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate_dimensions(
        self,
        qa_input: QAInput,
        num_dimensions: Optional[int] = None
    ) -> List[Dimension]:
        """
        Generate evaluation dimensions from Q&A pair.
        
        Args:
            qa_input: Question and answer input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            
        Returns:
            List of Dimension objects
        """
        prompt = build_dimension_generation_prompt(
            question=qa_input.question,
            answer=qa_input.answer,
            num_dimensions=num_dimensions,
            context=qa_input.context
        )
        response = self._call_llm(prompt)
        
        # Convert response to Dimension objects
        dimensions = []
        for item in response:
            # Convert string keys to integers for scores
            if "scores" in item and item["scores"]:
                item["scores"] = {int(k): v for k, v in item["scores"].items()}
            
            dimensions.append(Dimension(**item))
        
        return dimensions
    
    def generate_criteria(
        self,
        qa_input: QAInput,
        dimensions: List[Dimension],
        num_criteria: Optional[int] = None,
        category_hints: Optional[List[str]] = None
    ) -> List[Criterion]:
        """
        Generate evaluation criteria for dimensions.
        
        Args:
            qa_input: Question and answer input
            dimensions: List of dimensions to create criteria for
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            
        Returns:
            List of Criterion objects
        """
        prompt = build_criteria_generation_prompt(
            question=qa_input.question,
            answer=qa_input.answer,
            dimensions=dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            context=qa_input.context
        )
        response = self._call_llm(prompt, categories=category_hints)
        
        # Convert response to Criterion objects
        criteria = []
        for item in response:
            criteria.append(Criterion(**item))
        
        return criteria
    
    def generate_rubric(
        self,
        qa_input: QAInput,
        num_dimensions: Optional[int] = None,
        num_criteria: Optional[int] = None,
        category_hints: Optional[List[str]] = None
    ) -> Rubric:
        """
        Generate a complete rubric from Q&A pair.
        
        Args:
            qa_input: Question and answer input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            
        Returns:
            Validated Rubric object
            
        Raises:
            ValueError: If parameters are out of range or generated rubric is invalid
        """
        # Validate parameters
        if num_dimensions is not None and not 1 <= num_dimensions <= 10:
            raise ValueError("num_dimensions must be between 1 and 10")
        if num_criteria is not None and not 1 <= num_criteria <= 10:
            raise ValueError("num_criteria must be between 1 and 10")
        
        # Phase 1: Generate dimensions
        dimensions = self.generate_dimensions(qa_input, num_dimensions)
        
        # Phase 2: Generate criteria
        criteria = self.generate_criteria(
            qa_input,
            dimensions,
            num_criteria,
            category_hints
        )
        
        # Create and validate rubric
        rubric = Rubric(dimensions=dimensions, criteria=criteria)
        
        return rubric
    
    def refine_rubric(
        self,
        rubric: Rubric,
        feedback: Optional[str] = None
    ) -> Rubric:
        """
        Refine an existing rubric with optional feedback.
        
        Args:
            rubric: Existing rubric to refine
            feedback: Optional specific feedback for refinement
            
        Returns:
            Refined Rubric object
        """
        prompt = build_refine_rubric_prompt(
            dimensions=rubric.dimensions,
            criteria=rubric.criteria,
            feedback=feedback
        )
        response = self._call_llm(prompt)
        
        # Convert response to Rubric
        dimensions = []
        for item in response["dimensions"]:
            if "scores" in item and item["scores"]:
                item["scores"] = {int(k): v for k, v in item["scores"].items()}
            dimensions.append(Dimension(**item))
        
        criteria = []
        for item in response["criteria"]:
            criteria.append(Criterion(**item))
        
        refined_rubric = Rubric(dimensions=dimensions, criteria=criteria)
        
        return refined_rubric
    
    def generate_dimensions_from_chat(
        self,
        chat_input: ChatSessionInput,
        num_dimensions: Optional[int] = None
    ) -> List[Dimension]:
        """
        Generate evaluation dimensions from chat session.
        
        Chat sessions typically need more dimensions than Q&A because they include
        both tool usage and output quality aspects. Uses auto mode by default.
        
        Args:
            chat_input: Chat session input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            
        Returns:
            List of Dimension objects
        """
        prompt = build_chat_dimension_generation_prompt(
            chat_content=chat_input.content,
            num_dimensions=num_dimensions,
            context=chat_input.context
        )
        response = self._call_llm(prompt)
        
        # Convert response to Dimension objects
        dimensions = []
        for item in response:
            # Convert string keys to integers for scores
            if "scores" in item and item["scores"]:
                item["scores"] = {int(k): v for k, v in item["scores"].items()}
            
            dimensions.append(Dimension(**item))
        
        return dimensions
    
    def generate_criteria_from_chat(
        self,
        chat_input: ChatSessionInput,
        dimensions: List[Dimension],
        num_criteria: Optional[int] = None,
        category_hints: Optional[List[str]] = None
    ) -> List[Criterion]:
        """
        Generate evaluation criteria for dimensions from chat session.
        
        Chat sessions benefit from more granular criteria to check specific facts,
        tool usage, and output quality. Uses auto mode by default.
        
        Args:
            chat_input: Chat session input
            dimensions: List of dimensions to create criteria for
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            
        Returns:
            List of Criterion objects
        """
        prompt = build_chat_criteria_generation_prompt(
            chat_content=chat_input.content,
            dimensions=dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            context=chat_input.context
        )
        response = self._call_llm(prompt, categories=category_hints)
        
        # Convert response to Criterion objects
        criteria = []
        for item in response:
            criteria.append(Criterion(**item))
        
        return criteria
    
    def generate_rubric_from_chat(
        self,
        chat_input: ChatSessionInput,
        num_dimensions: Optional[int] = None,
        num_criteria: Optional[int] = None,
        category_hints: Optional[List[str]] = None
    ) -> Rubric:
        """
        Generate a complete rubric from chat session.
        
        Uses auto mode by default to determine the appropriate number of dimensions
        and criteria based on content complexity, tool usage, and factual information.
        
        Args:
            chat_input: Chat session input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            
        Returns:
            Validated Rubric object
            
        Raises:
            ValueError: If parameters are out of range or generated rubric is invalid
        """
        # Validate parameters
        if num_dimensions is not None and not 1 <= num_dimensions <= 10:
            raise ValueError("num_dimensions must be between 1 and 10")
        if num_criteria is not None and not 1 <= num_criteria <= 10:
            raise ValueError("num_criteria must be between 1 and 10")
        
        # Phase 1: Generate dimensions
        dimensions = self.generate_dimensions_from_chat(chat_input, num_dimensions)
        
        # Phase 2: Generate criteria
        criteria = self.generate_criteria_from_chat(
            chat_input,
            dimensions,
            num_criteria,
            category_hints
        )
        
        # Create and validate rubric
        rubric = Rubric(dimensions=dimensions, criteria=criteria)
        
        return rubric
    
    def _call_llm(self, prompt: str, **kwargs) -> Any:
        """
        Call LLM and parse JSON response.
        
        Args:
            prompt: Prompt to send to LLM
            **kwargs: Additional context passed to this method
            
        Returns:
            Parsed JSON response
            
        Raises:
            ValueError: If response is not valid JSON or was truncated
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": GENERATOR_CONFIG.system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=GENERATOR_CONFIG.temperature,
            max_tokens=GENERATOR_CONFIG.max_tokens
        )
        
        # Check if response was truncated
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            raise ValueError(
                f"LLM response was truncated due to max_tokens limit ({GENERATOR_CONFIG.max_tokens}). "
                "The model needs more tokens to complete the response. "
                "Try reducing the number of dimensions or criteria, or use a model with higher token limits."
            )
        
        content = response.choices[0].message.content.strip()
        original_content = content  # Save for error reporting
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        # Try to parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to repair common JSON issues
            try:
                repaired = repair_json(content)
                return json.loads(repaired)
            except json.JSONDecodeError:
                # If repair also fails, provide detailed error message
                error_lines = original_content.split('\n')
                context_start = max(0, e.lineno - 3)
                context_end = min(len(error_lines), e.lineno + 2)
                context = '\n'.join(f"  {i+1:3d}| {line}" for i, line in enumerate(error_lines[context_start:context_end], start=context_start))
                
                raise ValueError(
                    f"LLM returned invalid JSON. Error at line {e.lineno}, column {e.colno}: {e.msg}\n"
                    f"Context:\n{context}\n\n"
                    f"Full response:\n{original_content[:500]}{'...' if len(original_content) > 500 else ''}"
                )

