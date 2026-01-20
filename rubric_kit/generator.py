"""Rubric generation using LLM."""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import litellm
import yaml

from rubric_kit.schema import Criterion, Dimension, Rubric


if TYPE_CHECKING:
    from rubric_kit.metrics import MetricsAggregator
from rubric_kit.prompts import (
    GENERATOR_CONFIG,
    build_chat_criteria_generation_prompt,
    build_chat_dimension_generation_prompt,
    build_criteria_generation_prompt,
    build_dimension_generation_prompt,
    build_refine_rubric_prompt,
    build_refine_rubric_with_chat_prompt,
    build_refine_rubric_with_qa_prompt,
)


@dataclass
class QAInput:
    """Question and Answer input for rubric generation."""

    question: str
    answer: str
    context: str | None = None


@dataclass
class ChatSessionInput:
    """Chat session input for rubric generation."""

    content: str
    context: str | None = None


def _is_simple_qa_format(content: str) -> bool:
    """Check if content appears to be in simple Q:/A: format."""
    first_line = content.split("\n")[0].strip()
    return first_line.startswith(("Q:", "q:", "A:", "a:")) or "\nQ:" in content or "\nq:" in content


def _parse_simple_qa_format(content: str) -> QAInput:
    """Parse simple Q:/A: text format."""
    lines = content.split("\n")
    question = None
    answer_lines = []
    in_answer = False

    for line in lines:
        line_stripped = line.strip()

        if line_stripped.startswith(("Q:", "q:")):
            if question is not None:
                raise ValueError("Multiple questions found in Q&A file")
            question = line_stripped[2:].strip()
            in_answer = False
        elif line_stripped.startswith(("A:", "a:")):
            answer_lines = [line_stripped[2:].strip()]
            in_answer = True
        elif in_answer:
            answer_lines.append(line)
        elif question is None and line_stripped:
            raise ValueError("Question not found")

    if question is None:
        raise ValueError("Question not found")

    if not answer_lines:
        raise ValueError("Answer not found")

    answer = "\n".join(answer_lines).strip()
    if not answer:
        raise ValueError("Answer not found")

    return QAInput(question=question, answer=answer, context=None)


def _parse_yaml_qa_format(content: str) -> QAInput:
    """Parse YAML format Q&A input."""
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}") from e

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
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()

    # Handle context similarly
    if context is not None and isinstance(context, str):
        context = context.strip() if context else None

    return QAInput(question=question, answer=answer, context=context)


def parse_qa_input(file_path: str) -> QAInput:
    """
    Parse Q&A input from a file.

    Supports two formats:
    1. YAML format with the following structure:
       ```yaml
       question: "The question text"
       answer: "The answer text"
       context: "Optional context"  # Optional
       ```

    2. Simple text format:
       ```
       Q: The question text
       A: The answer text
       ```
       (Answer can span multiple lines after "A:")

    Args:
        file_path: Path to Q&A file

    Returns:
        QAInput object

    Raises:
        ValueError: If file is empty, missing required fields, or not valid format
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Q&A file not found: {file_path}")

    content = path.read_text().strip()

    if not content:
        raise ValueError("Q&A file is empty")

    if _is_simple_qa_format(content):
        return _parse_simple_qa_format(content)

    return _parse_yaml_qa_format(content)


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


def parse_dimensions_file(file_path: str) -> list[Dimension]:
    """
    Parse dimensions from a YAML file.

    Args:
        file_path: Path to dimensions YAML file

    Returns:
        List of Dimension objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dimensions file not found: {file_path}")

    content = path.read_text()

    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML format: {e}") from e

    if not isinstance(data, dict) or "dimensions" not in data:
        raise ValueError("Dimensions file must contain a 'dimensions' key")

    dims_data = data["dimensions"]

    if not dims_data or not isinstance(dims_data, list):
        raise ValueError("Dimensions file must contain at least one dimension")

    dimensions = []
    for dim_data in dims_data:
        dimensions.append(Dimension(**dim_data))

    return dimensions


def repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues in LLM output.

    Args:
        text: JSON string that may have common issues

    Returns:
        Repaired JSON string
    """
    # Remove trailing commas before closing brackets/braces
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # Remove comments (// style and /* */ style)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Fix unquoted keys (common LLM mistake)
    # This is a simple heuristic - match word characters followed by colon
    text = re.sub(r"(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', text)

    return text


def _extract_json_from_response(content: str) -> str:
    """Extract JSON content from LLM response, removing markdown code blocks."""
    content = content.strip()

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


def _parse_json_response(content: str) -> Any:
    """Parse JSON from LLM response with error handling and repair attempts."""
    original_content = content

    # Try direct parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Save error details for potential error message
        first_error = e

    # Try repairing common issues
    try:
        repaired = repair_json(content)
        return json.loads(repaired)
    except json.JSONDecodeError:
        # Repair failed, use original error details
        error_lines = original_content.split("\n")
        context_start = max(0, first_error.lineno - 3)
        context_end = min(len(error_lines), first_error.lineno + 2)
        context = "\n".join(
            f"  {i + 1:3d}| {line}"
            for i, line in enumerate(error_lines[context_start:context_end], start=context_start)
        )

        raise ValueError(
            f"LLM returned invalid JSON. Error at line {first_error.lineno}, column {first_error.colno}: {first_error.msg}\n"
            f"Context:\n{context}\n\n"
            f"Full response:\n{original_content[:500]}{'...' if len(original_content) > 500 else ''}"
        ) from first_error


def _safe_int(value: Any) -> int | None:
    """Safely convert a value to int, returning None if not possible."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _normalize_scores(scores: Any) -> dict[int, str] | None:
    """Normalize scores from various LLM response formats to Dict[int, str].

    Handles:
    - Dict[str|int, str]: Standard format, just convert keys to int
    - List[Dict]: List of single-key dicts like [{1: "Bad"}, {2: "Good"}]
    - List[List]: List of pairs like [[1, "Bad"], [2, "Good"]]

    Skips entries with non-integer keys (e.g., "score", "description").
    """
    if scores is None:
        return None

    # Already a dict - convert keys to ints, skipping invalid keys
    if isinstance(scores, dict):
        result = {}
        for k, v in scores.items():
            int_key = _safe_int(k)
            if int_key is not None and isinstance(v, str):
                result[int_key] = v
        return result if result else None

    # List format - convert to dict
    if isinstance(scores, list):
        result = {}
        for item in scores:
            if isinstance(item, dict):
                # Single-key dict like {1: "Bad"} or {2: "Good"}
                for k, v in item.items():
                    int_key = _safe_int(k)
                    if int_key is not None and isinstance(v, str):
                        result[int_key] = v
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # Pair like [1, "Bad"]
                int_key = _safe_int(item[0])
                if int_key is not None:
                    result[int_key] = str(item[1])
        return result if result else None

    return None


def _convert_to_dimensions(response: list[dict[str, Any]]) -> list[Dimension]:
    """Convert LLM response to list of Dimension objects."""
    dimensions = []
    for item in response:
        if "scores" in item and item["scores"]:
            item["scores"] = _normalize_scores(item["scores"])
        dimensions.append(Dimension(**item))
    return dimensions


def _normalize_tool_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool spec, handling common LLM variations.

    Handles these formats:
    1. {"name": "tool_name", "params": {...}} - standard format, pass through
    2. {"tool_name": "some_tool", "params": {...}} - 'tool_name' as alias for 'name'
    3. {"some_tool": {"params": {...}}} - tool name as key, inner dict has params
    4. {"some_tool": {"min_calls": 1, ...}} - tool name as key with other fields
    """
    # If 'name' already exists, just return a copy
    if "name" in spec:
        return spec.copy()

    normalized = spec.copy()

    # Handle 'tool_name' as an alias for 'name'
    if "tool_name" in normalized:
        normalized["name"] = normalized.pop("tool_name")
        return normalized

    # Handle format where tool name is the key: {"tool_name": {"params": {...}}}
    # This is common when LLM uses tool name as a dict key
    known_fields = {"name", "tool_name", "min_calls", "max_calls", "params"}
    unknown_keys = [k for k in spec if k not in known_fields]

    # If there's exactly one unknown key and it looks like a tool name (not a known field),
    # treat it as the tool name with the value being additional properties
    if len(unknown_keys) == 1:
        tool_name_key = unknown_keys[0]
        inner_value = spec[tool_name_key]

        # Build the normalized spec
        normalized = {"name": tool_name_key}

        # If the inner value is a dict, merge its properties
        if isinstance(inner_value, dict):
            for k, v in inner_value.items():
                if k in ("min_calls", "max_calls", "params"):
                    normalized[k] = v

        # Also include any known fields from the outer dict
        for field in ("min_calls", "max_calls", "params"):
            if field in spec and field not in normalized:
                normalized[field] = spec[field]

        return normalized

    return normalized


def _normalize_tool_calls(tool_calls: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool_calls structure from LLM response."""
    if not tool_calls:
        return tool_calls

    normalized = tool_calls.copy()

    # Normalize required tools
    if "required" in normalized and isinstance(normalized["required"], list):
        normalized["required"] = [_normalize_tool_spec(spec) for spec in normalized["required"]]

    # Normalize optional tools
    if "optional" in normalized and isinstance(normalized["optional"], list):
        normalized["optional"] = [_normalize_tool_spec(spec) for spec in normalized["optional"]]

    # Normalize prohibited tools
    if "prohibited" in normalized and isinstance(normalized["prohibited"], list):
        normalized["prohibited"] = [_normalize_tool_spec(spec) for spec in normalized["prohibited"]]

    return normalized


def _convert_to_criteria(response: list[dict[str, Any]]) -> list[Criterion]:
    """Convert LLM response to list of Criterion objects."""
    criteria = []
    for item in response:
        # Normalize tool_calls if present
        if "tool_calls" in item and item["tool_calls"]:
            item = item.copy()
            item["tool_calls"] = _normalize_tool_calls(item["tool_calls"])
        criteria.append(Criterion(**item))
    return criteria


def _validate_dimension_criteria_params(
    num_dimensions: int | None, num_criteria: int | None
) -> None:
    """Validate dimension and criteria count parameters."""
    if num_dimensions is not None and not 1 <= num_dimensions <= 10:
        raise ValueError("num_dimensions must be between 1 and 10")
    if num_criteria is not None and not 1 <= num_criteria <= 10:
        raise ValueError("num_criteria must be between 1 and 10")


def _merge_dimensions(
    existing: list[Dimension], to_merge: list[Dimension] | None
) -> list[Dimension]:
    """
    Merge dimensions, with to_merge dimensions taking priority.

    Args:
        existing: Existing dimensions from rubric
        to_merge: Optional dimensions to merge in

    Returns:
        Merged list of dimensions (to_merge + any unique from existing)
    """
    if to_merge is None:
        return existing

    # Build set of names from dimensions to merge
    merged_names = {d.name for d in to_merge}

    # Start with all dimensions to merge
    result = list(to_merge)

    # Add any existing dimensions not in to_merge
    for dim in existing:
        if dim.name not in merged_names:
            result.append(dim)

    return result


class RubricGenerator:
    """Generate rubrics from Q&A pairs using LLM.

    Supports multiple LLM providers via LiteLLM:
    - OpenAI: model="gpt-4" or "gpt-4o"
    - Google Vertex AI: model="vertex_ai/gemini-2.5-flash"
    - IBM WatsonX: model="watsonx/meta-llama/llama-3-8b-instruct"
    - Anthropic: model="claude-3-5-sonnet-20241022"
    - Local Ollama: model="ollama/llama3"
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        metrics: Optional["MetricsAggregator"] = None,
    ):
        """
        Initialize RubricGenerator.

        Args:
            api_key: Optional API key. If not provided, LiteLLM will auto-detect
                    from environment variables based on the model/provider.
            model: Model name to use. Use LiteLLM format for non-OpenAI providers:
                   - "gpt-4" for OpenAI
                   - "vertex_ai/gemini-2.5-flash" for Google Vertex AI
                   - "watsonx/meta-llama/llama-3-8b-instruct" for IBM WatsonX
            base_url: Optional base URL for OpenAI-compatible endpoints
            metrics: Optional MetricsAggregator for tracking LLM call metrics
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.metrics = metrics

    def generate_dimensions(
        self, qa_input: QAInput, num_dimensions: int | None = None, guidelines: str | None = None
    ) -> list[Dimension]:
        """
        Generate evaluation dimensions from Q&A pair.

        Args:
            qa_input: Question and answer input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            guidelines: Optional specific guidelines/hints to guide dimension generation

        Returns:
            List of Dimension objects
        """
        prompt = build_dimension_generation_prompt(
            question=qa_input.question,
            answer=qa_input.answer,
            num_dimensions=num_dimensions,
            context=qa_input.context,
            guidelines=guidelines,
        )
        response = self._call_llm(prompt, call_type="generate_dimensions")
        return _convert_to_dimensions(response)

    def generate_criteria(
        self,
        qa_input: QAInput,
        dimensions: list[Dimension],
        num_criteria: int | None = None,
        category_hints: list[str] | None = None,
        use_variables: bool = True,
        guidelines: str | None = None,
    ) -> tuple[list[Criterion], dict[str, str] | None]:
        """
        Generate evaluation criteria for dimensions.

        Args:
            qa_input: Question and answer input
            dimensions: List of dimensions to create criteria for
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
            guidelines: Optional specific guidelines/hints to guide criteria generation

        Returns:
            Tuple of (List of Criterion objects, Optional variables dict)
        """
        prompt = build_criteria_generation_prompt(
            question=qa_input.question,
            answer=qa_input.answer,
            dimensions=dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            context=qa_input.context,
            use_variables=use_variables,
            guidelines=guidelines,
        )
        response = self._call_llm(prompt, call_type="generate_criteria")

        # Handle new format with variables
        if isinstance(response, dict) and "criteria" in response:
            variables = response.get("variables") if use_variables else None
            criteria = _convert_to_criteria(response["criteria"])
            return criteria, variables

        # Fallback for old format (just an array)
        return _convert_to_criteria(response), None

    def generate_rubric(
        self,
        qa_input: QAInput,
        num_dimensions: int | None = None,
        num_criteria: int | None = None,
        category_hints: list[str] | None = None,
        dimensions: list[Dimension] | None = None,
        use_variables: bool = True,
        guidelines: str | None = None,
    ) -> Rubric:
        """
        Generate a complete rubric from Q&A pair.

        Args:
            qa_input: Question and answer input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            dimensions: Optional pre-defined dimensions (skips dimension generation)
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
            guidelines: Optional specific guidelines/hints to guide rubric generation

        Returns:
            Validated Rubric object with variables extracted from the content

        Raises:
            ValueError: If parameters are out of range or generated rubric is invalid
        """
        _validate_dimension_criteria_params(num_dimensions, num_criteria)

        # Use provided dimensions or generate new ones
        if dimensions is not None:
            dims = dimensions
        else:
            dims = self.generate_dimensions(qa_input, num_dimensions, guidelines=guidelines)

        criteria, variables = self.generate_criteria(
            qa_input,
            dims,
            num_criteria,
            category_hints,
            use_variables=use_variables,
            guidelines=guidelines,
        )

        return Rubric(dimensions=dims, criteria=criteria, variables=variables)

    def refine_rubric(
        self,
        rubric: Rubric,
        feedback: str | None = None,
        dimensions_to_merge: list[Dimension] | None = None,
        use_variables: bool = True,
    ) -> Rubric:
        """
        Refine an existing rubric with optional feedback.

        Args:
            rubric: Existing rubric to refine
            feedback: Optional specific feedback for refinement
            dimensions_to_merge: Optional dimensions to merge with existing
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.

        Returns:
            Refined Rubric object with variables
        """
        # Merge dimensions if provided
        merged_dims = _merge_dimensions(rubric.dimensions, dimensions_to_merge)

        prompt = build_refine_rubric_prompt(
            dimensions=merged_dims,
            criteria=rubric.criteria,
            feedback=feedback,
            variables=rubric.variables if use_variables else None,
            use_variables=use_variables,
        )
        response = self._call_llm(prompt, call_type="refine_rubric")

        dimensions = _convert_to_dimensions(response["dimensions"])
        criteria = _convert_to_criteria(response["criteria"])
        variables = response.get("variables") if use_variables else None

        return Rubric(dimensions=dimensions, criteria=criteria, variables=variables)

    def refine_rubric_with_qa(
        self,
        rubric: Rubric,
        qa_input: QAInput,
        feedback: str | None = None,
        dimensions_to_merge: list[Dimension] | None = None,
        use_variables: bool = True,
    ) -> Rubric:
        """
        Refine an existing rubric using Q&A context.

        Args:
            rubric: Existing rubric to refine
            qa_input: Q&A input to use as context for refinement
            feedback: Optional specific feedback for refinement
            dimensions_to_merge: Optional dimensions to merge with existing
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.

        Returns:
            Refined Rubric object with variables
        """
        # Merge dimensions if provided
        merged_dims = _merge_dimensions(rubric.dimensions, dimensions_to_merge)

        prompt = build_refine_rubric_with_qa_prompt(
            dimensions=merged_dims,
            criteria=rubric.criteria,
            question=qa_input.question,
            answer=qa_input.answer,
            feedback=feedback,
            context=qa_input.context,
            variables=rubric.variables if use_variables else None,
            use_variables=use_variables,
        )
        response = self._call_llm(prompt, call_type="refine_rubric_with_qa")

        dimensions = _convert_to_dimensions(response["dimensions"])
        criteria = _convert_to_criteria(response["criteria"])
        variables = response.get("variables") if use_variables else None

        return Rubric(dimensions=dimensions, criteria=criteria, variables=variables)

    def refine_rubric_with_chat(
        self,
        rubric: Rubric,
        chat_input: ChatSessionInput,
        feedback: str | None = None,
        dimensions_to_merge: list[Dimension] | None = None,
        use_variables: bool = True,
    ) -> Rubric:
        """
        Refine an existing rubric using chat session context.

        Args:
            rubric: Existing rubric to refine
            chat_input: Chat session input to use as context for refinement
            feedback: Optional specific feedback for refinement
            dimensions_to_merge: Optional dimensions to merge with existing
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.

        Returns:
            Refined Rubric object with variables
        """
        # Merge dimensions if provided
        merged_dims = _merge_dimensions(rubric.dimensions, dimensions_to_merge)

        prompt = build_refine_rubric_with_chat_prompt(
            dimensions=merged_dims,
            criteria=rubric.criteria,
            chat_content=chat_input.content,
            feedback=feedback,
            context=chat_input.context,
            variables=rubric.variables if use_variables else None,
            use_variables=use_variables,
        )
        response = self._call_llm(prompt, call_type="refine_rubric_with_chat")

        dimensions = _convert_to_dimensions(response["dimensions"])
        criteria = _convert_to_criteria(response["criteria"])
        variables = response.get("variables") if use_variables else None

        return Rubric(dimensions=dimensions, criteria=criteria, variables=variables)

    def generate_dimensions_from_chat(
        self,
        chat_input: ChatSessionInput,
        num_dimensions: int | None = None,
        guidelines: str | None = None,
    ) -> list[Dimension]:
        """
        Generate evaluation dimensions from chat session.

        Chat sessions typically need more dimensions than Q&A because they include
        both tool usage and output quality aspects. Uses auto mode by default.

        Args:
            chat_input: Chat session input
            num_dimensions: Number of dimensions to generate (1-10), or None for auto
            guidelines: Optional specific guidelines/hints to guide dimension generation

        Returns:
            List of Dimension objects
        """
        prompt = build_chat_dimension_generation_prompt(
            chat_content=chat_input.content,
            num_dimensions=num_dimensions,
            context=chat_input.context,
            guidelines=guidelines,
        )
        response = self._call_llm(prompt, call_type="generate_dimensions_from_chat")
        return _convert_to_dimensions(response)

    def generate_criteria_from_chat(
        self,
        chat_input: ChatSessionInput,
        dimensions: list[Dimension],
        num_criteria: int | None = None,
        category_hints: list[str] | None = None,
        use_variables: bool = True,
        guidelines: str | None = None,
    ) -> tuple[list[Criterion], dict[str, str] | None]:
        """
        Generate evaluation criteria for dimensions from chat session.

        Chat sessions benefit from more granular criteria to check specific facts,
        tool usage, and output quality. Uses auto mode by default.

        Args:
            chat_input: Chat session input
            dimensions: List of dimensions to create criteria for
            num_criteria: Number of criteria to generate (1-10), or None for auto
            category_hints: Optional list of category names to guide generation
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
            guidelines: Optional specific guidelines/hints to guide criteria generation

        Returns:
            Tuple of (List of Criterion objects, Optional variables dict)
        """
        prompt = build_chat_criteria_generation_prompt(
            chat_content=chat_input.content,
            dimensions=dimensions,
            num_criteria=num_criteria,
            category_hints=category_hints,
            context=chat_input.context,
            use_variables=use_variables,
            guidelines=guidelines,
        )
        response = self._call_llm(prompt, call_type="generate_criteria_from_chat")

        # Handle new format with variables
        if isinstance(response, dict) and "criteria" in response:
            variables = response.get("variables") if use_variables else None
            criteria = _convert_to_criteria(response["criteria"])
            return criteria, variables

        # Fallback for old format (just an array)
        return _convert_to_criteria(response), None

    def generate_rubric_from_chat(
        self,
        chat_input: ChatSessionInput,
        num_dimensions: int | None = None,
        num_criteria: int | None = None,
        category_hints: list[str] | None = None,
        dimensions: list[Dimension] | None = None,
        use_variables: bool = True,
        guidelines: str | None = None,
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
            dimensions: Optional pre-defined dimensions (skips dimension generation)
            use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
            guidelines: Optional specific guidelines/hints to guide rubric generation

        Returns:
            Validated Rubric object with variables extracted from the content

        Raises:
            ValueError: If parameters are out of range or generated rubric is invalid
        """
        _validate_dimension_criteria_params(num_dimensions, num_criteria)

        # Use provided dimensions or generate new ones
        if dimensions is not None:
            dims = dimensions
        else:
            dims = self.generate_dimensions_from_chat(
                chat_input, num_dimensions, guidelines=guidelines
            )

        criteria, variables = self.generate_criteria_from_chat(
            chat_input,
            dims,
            num_criteria,
            category_hints,
            use_variables=use_variables,
            guidelines=guidelines,
        )

        return Rubric(dimensions=dims, criteria=criteria, variables=variables)

    def _call_llm(self, prompt: str, call_type: str = "generate", **kwargs) -> Any:
        """
        Call LLM via LiteLLM and parse JSON response.

        Args:
            prompt: Prompt to send to LLM
            call_type: Type of call for metrics tracking (e.g., 'generate_dimensions')
            **kwargs: Additional context passed to this method (currently unused)

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If response is not valid JSON or was truncated
        """
        api_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": GENERATOR_CONFIG.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": GENERATOR_CONFIG.temperature,
            "max_tokens": GENERATOR_CONFIG.max_tokens,
        }

        # Add explicit API key if provided
        if self.api_key:
            api_params["api_key"] = self.api_key

        # Add custom base URL if provided (for OpenAI-compatible endpoints)
        if self.base_url:
            api_params["api_base"] = self.base_url

        # Track timing for metrics
        start_time = time.time()
        response = litellm.completion(**api_params)
        latency = time.time() - start_time

        # Record metrics if aggregator is configured
        if self.metrics is not None:
            self.metrics.record_call(
                call_type=call_type,
                model=self.model,
                usage=response.usage,
                latency=latency,
                response=response,
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
        json_content = _extract_json_from_response(content)

        return _parse_json_response(json_content)
