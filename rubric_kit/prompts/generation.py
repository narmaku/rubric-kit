"""Generation prompt builders for rubric-kit.

This module contains functions that build prompts for generating evaluation
dimensions and criteria from Q&A pairs or chat sessions.
"""

from typing import Any

from rubric_kit.models.schema import Criterion, Dimension


def _convert_criterion_to_dict_for_yaml(criterion: Criterion) -> dict[str, Any]:
    """Convert a criterion to dict format for YAML display, including tool_calls if present."""
    crit_dict: dict[str, Any] = {
        "name": criterion.name,
        "category": criterion.category,
        "weight": criterion.weight,
        "dimension": criterion.dimension,
        "criterion": criterion.criterion,
    }

    if criterion.tool_calls:
        required_list = [
            {
                tc.name: {
                    "min_calls": tc.min_calls,
                    "max_calls": tc.max_calls,
                    **({"params": tc.params} if tc.params else {}),
                }
            }
            for tc in criterion.tool_calls.required
        ]
        optional_list = [
            {
                tc.name: {
                    "min_calls": tc.min_calls,
                    "max_calls": tc.max_calls,
                    **({"params": tc.params} if tc.params else {}),
                }
            }
            for tc in criterion.tool_calls.optional
        ]
        prohibited_list = [tc.name for tc in criterion.tool_calls.prohibited]

        crit_dict["tool_calls"] = {
            "respect_order": criterion.tool_calls.respect_order,
            "required": required_list,
            "optional": optional_list if optional_list else [],
            "prohibited": prohibited_list if prohibited_list else [],
        }

    return crit_dict


def build_dimension_generation_prompt(
    question: str,
    answer: str,
    num_dimensions: int | None,
    context: str | None = None,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation dimensions from a Q&A pair.

    Args:
        question: The question being evaluated
        answer: The answer being evaluated
        num_dimensions: Number of dimensions to generate, or None for auto
        context: Optional additional context
        guidelines: Optional specific guidelines/hints to guide dimension generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    count_instruction = (
        f"Generate {num_dimensions} evaluation dimensions"
        if num_dimensions is not None
        else "Generate an appropriate number of evaluation dimensions (between 3 and 10)"
    )

    return f"""Given the following Question and Answer pair, {count_instruction} for assessing answer quality.

Question: {question}

Answer: {answer}{context_info}{guidelines_section}

Each dimension should:
1. Have a unique, descriptive name (lowercase with underscores, e.g., "factual_correctness")
2. Have a **GENERIC** description of what aspect it evaluates
3. **DO NOT** mention specific data values or fields in the dimension description
4. Specify a grading_type: either "binary" (pass/fail) or "score" (numeric scale from 0 to 3)
5. For "score" type, you MUST include a "scores" dictionary with integer keys (0-3) and description values

**CRITICAL - Score Dimensions:**
- If grading_type is "score", the "scores" field is REQUIRED - do NOT set it to null or omit it
- Scores must have keys 0, 1, 2, 3 with string descriptions for each level
- If you don't need nuanced scoring, use grading_type "binary" instead

**CRITICAL - Dimension Design:**
- Dimensions should be GENERIC and reusable (e.g., "factual_correctness" not "cpu_count_correctness")
- Do NOT create separate dimensions for each piece of data
- One "factual_correctness" dimension can be used by MANY criteria checking different facts
- The CRITERIA will specify what specific values to check

IMPORTANT: Prefer "binary" grading type unless a dimension truly requires nuanced scoring.

Common dimensions to consider:
- factual_correctness: Factual accuracy of information
- completeness: Whether all key information is provided
- relevance: How well the answer addresses the question
- clarity: How clear and understandable the answer is

Return ONLY a JSON array of dimension objects. Example format:
[
  {{
    "name": "factual_correctness",
    "description": "Evaluates whether the information provided is factually accurate and correct",
    "grading_type": "binary"
  }},
  {{
    "name": "completeness",
    "description": "Evaluates how complete and comprehensive the answer is",
    "grading_type": "score",
    "scores": {{
      "0": "No relevant information provided",
      "1": "Missing most key information",
      "2": "Partially complete, missing some key details",
      "3": "Complete with all essential information"
    }}
  }}
]"""


def build_criteria_generation_prompt(
    question: str,
    answer: str,
    dimensions: list[Dimension],
    num_criteria: int | None,
    category_hints: list[str] | None = None,
    context: str | None = None,
    use_variables: bool = True,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation criteria from Q&A and dimensions.

    Args:
        question: The question being evaluated
        answer: The answer being evaluated
        dimensions: List of dimensions to create criteria for
        num_criteria: Number of criteria to generate, or None for auto
        category_hints: Optional list of category names to guide generation
        context: Optional additional context
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
        guidelines: Optional specific guidelines/hints to guide criteria generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    # Format dimensions for prompt
    dimensions_str = "\n".join(
        [f"- {d.name} ({d.grading_type}): {d.description}" for d in dimensions]
    )

    category_guidance = (
        f"\n\nPreferred categories to use: {', '.join(category_hints)}"
        if category_hints
        else "\n\nSuggested categories: Output, Reasoning, Completeness, Accuracy, Clarity"
    )

    count_instruction = (
        f"generate {num_criteria} specific evaluation criteria"
        if num_criteria is not None
        else "generate an appropriate number of specific evaluation criteria (between 5 and 10, as many as needed to thoroughly evaluate the answer)"
    )

    if use_variables:
        variables_section = """**IMPORTANT - Variables Section:**
Extract specific data values from the answer (names, numbers, identifiers, etc.) and put them in a "variables" section. Variables should ONLY contain actual, correct values - NOT examples of incorrect values. Then use {{variable_name}} placeholders in your criterion text AND tool_calls params instead of hard-coding the values. This makes the rubric reusable with different data."""

        criteria_item_2 = """2. **Use variables for specific values** - extract specific data values to the variables section and reference them using {{variable_name}} syntax"""

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The answer correctly reports RAM (~{{ram_total}}) and disk size ({{disk_size}})" - This mixes two values!
- RIGHT: Create TWO separate criteria:
  1. "The answer correctly reports RAM as ~{{ram_total}}"
  2. "The answer correctly reports disk size as {{disk_size}}"
- Each factual accuracy criterion should verify ONE atomic value against ground truth"""

        criterion_field = """- criterion: Specific text describing what to check, using {{variable_name}} for specific values (or "from_scores" for score dimensions)"""

        json_example = """Return ONLY a JSON object with "variables" and "criteria" keys. Example format:
{
  "variables": {
    "capital_city": "Paris",
    "country_name": "France"
  },
  "criteria": [
    {
      "name": "capital_accuracy",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_correctness",
      "criterion": "The answer must correctly identify {{capital_city}} as the capital of {{country_name}}"
    },
    {
      "name": "completeness_score",
      "category": "Completeness",
      "weight": "from_scores",
      "dimension": "completeness",
      "criterion": "from_scores"
    }
  ]
}

Note: Extract ALL specific data values (names, numbers, identifiers, etc.) to the variables section."""
    else:
        variables_section = """**IMPORTANT - No Variables Mode:**
Do NOT create a variables section. Use hard-coded values directly in criterion text and tool_calls params. Write specific, concrete values directly into the criteria."""

        criteria_item_2 = """2. **Use hard-coded values** - write specific values directly into criteria (e.g., "IP address is '10.0.187.159'" not "IP address is '{{ip_address}}'")"""

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The answer correctly reports RAM (~1.7GB) and disk size (50GB)" - This mixes two values!
- RIGHT: Create TWO separate criteria:
  1. "The answer correctly reports RAM as ~1.7GB"
  2. "The answer correctly reports disk size as 50GB"
- Each factual accuracy criterion should verify ONE atomic value against ground truth"""

        criterion_field = """- criterion: Specific text describing what to check with hard-coded values (or "from_scores" for score dimensions)"""

        json_example = """Return ONLY a JSON object with "criteria" key (NO variables section). Example format:
{
  "criteria": [
    {
      "name": "capital_accuracy",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_correctness",
      "criterion": "The answer must correctly identify Paris as the capital of France"
    },
    {
      "name": "completeness_score",
      "category": "Completeness",
      "weight": "from_scores",
      "dimension": "completeness",
      "criterion": "from_scores"
    }
  ]
}

Note: Use hard-coded values directly in criteria - do NOT use variable placeholders."""

    return f"""Given the following Question, Answer, and Dimensions, {count_instruction}.

Question: {question}

Answer: {answer}{context_info}{guidelines_section}

Dimensions:
{dimensions_str}{category_guidance}

{variables_section}

Criteria should be:
1. **ATOMIC** - each criterion checks exactly ONE specific thing (one fact, one value, one requirement)
{criteria_item_2}
3. **Never mix multiple values in one factual accuracy criterion** - create separate criteria for each value to check
4. Specific and measurable
5. Distributed across the provided dimensions
6. Assigned appropriate categories (e.g., Output, Reasoning, Completeness)
7. Given weights between 0-3 based on importance (3=most important, 0=informational only)
8. For score-type dimensions, use weight="from_scores" and criterion="from_scores"

{atomic_examples}

Each criterion should have:
- name: Unique identifier (lowercase with underscores)
- category: Category name (will be auto-assigned based on the criterion type)
- weight: Integer 0-3, or "from_scores" for score-type dimensions
- dimension: Must reference one of the dimension names above
{criterion_field}

**CRITICAL - Weight Constraints:**
- Criterion weight MUST be an integer from 0 to 3 (inclusive), OR the string "from_scores"
- DO NOT use weights outside the 0-3 range (e.g., 10 is INVALID)

**CRITICAL - Dimension Reference:**
- If referencing a dimension with grading_type "score", ensure that dimension has a "scores" dictionary defined

{json_example}"""


def build_chat_dimension_generation_prompt(
    chat_content: str,
    num_dimensions: int | None,
    context: str | None = None,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation dimensions from a chat session.

    Args:
        chat_content: The raw chat session content
        num_dimensions: Number of dimensions to generate, or None for auto
        context: Optional additional context
        guidelines: Optional specific guidelines/hints to guide dimension generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    count_instruction = (
        f"Generate {num_dimensions} evaluation dimensions"
        if num_dimensions is not None
        else "Generate an appropriate number of evaluation dimensions (between 5 and 10, as many as needed)"
    )

    return f"""Given the following chat session, {count_instruction} for assessing the assistant's performance.

**Chat Session:**
{chat_content}{context_info}{guidelines_section}

**Instructions:**
Analyze the chat session above to understand what happened. Consider:
- Tool usage (if tools were used): correct selection, proper ordering, completeness
- **Output accuracy**: factual correctness of information provided
- Output completeness: whether all requested information was provided
- Output quality: clarity, relevance, organization

Each dimension should:
1. Have a unique, descriptive name (lowercase with underscores, e.g., "tool_usage_correctness", "factual_accuracy")
2. Have a **GENERIC** description of what aspect it evaluates (e.g., "checks if stated facts are correct")
3. **DO NOT** mention specific tools, data values, or fields in the dimension description
4. Specify a grading_type: either "binary" (pass/fail) or "score" (numeric scale from 0 to 3)
5. For "score" type, you MUST include a "scores" dictionary with integer keys (0-3) and description values

**CRITICAL - Score Dimensions:**
- If grading_type is "score", the "scores" field is REQUIRED - do NOT set it to null or omit it
- Scores must have keys 0, 1, 2, 3 with string descriptions for each level
- If you don't need nuanced scoring, use grading_type "binary" instead

**CRITICAL - Dimension Design:**
- Dimensions should be GENERIC and reusable (e.g., "factual_accuracy" not "data_field_accuracy")
- Do NOT create separate dimensions for each category or type of data
- One "factual_accuracy" dimension can be used by MANY criteria checking different facts
- The CRITERIA will specify what specific values to check (e.g., "field X equals value Y")

IMPORTANT:
- If tools were used, include one dimension for tool usage evaluation (typically named "tool_use")
- **Prefer "binary" grading type for fact-checking dimensions**
- Typical dimensions needed: tool_use, factual_accuracy, completeness, clarity
- Use "score" type only for dimensions that genuinely need nuanced evaluation (e.g., overall clarity, completeness)

**CRITICAL - Tool Evaluation Scoring (if using score type for tool_use):**
If you use grading_type "score" for tool evaluation, use this scoring model.
The checks depend on tool_calls configuration (respect_order, params, params_strict_mode):

- 3: All applicable checks pass - tool called with correct count, correct order (if respect_order=true), correct parameters (if params specified)
- 2: Tool called with correct order and parameters, but call count outside min/max bounds
- 1: Tool called but with incorrect parameters (if params specified) OR wrong order (if respect_order=true)
- 0: Required tool not called at all

Note: If respect_order=false, order is not checked. If no params specified, params are not checked.

Return ONLY a JSON array of dimension objects. Example format:
[
  {{
    "name": "tool_use",
    "description": "Evaluates whether the assistant correctly used tools to accomplish the task",
    "grading_type": "binary"
  }},
  {{
    "name": "factual_accuracy",
    "description": "Evaluates whether stated facts and data values are correct",
    "grading_type": "binary"
  }},
  {{
    "name": "completeness",
    "description": "Evaluates whether all requested information was provided",
    "grading_type": "score",
    "scores": {{
      "0": "No relevant information provided",
      "1": "Missing most requested information",
      "2": "Some information provided but incomplete",
      "3": "All requested information comprehensively provided"
    }}
  }},
  {{
    "name": "clarity",
    "description": "Evaluates the readability and organization of the response",
    "grading_type": "score",
    "scores": {{
      "0": "Completely unintelligible or no response",
      "1": "Poorly organized or difficult to understand",
      "2": "Generally clear but could be improved",
      "3": "Exceptionally clear and well-organized"
    }}
  }}
]"""


def build_chat_criteria_generation_prompt(
    chat_content: str,
    dimensions: list[Dimension],
    num_criteria: int | None,
    category_hints: list[str] | None = None,
    context: str | None = None,
    use_variables: bool = True,
    guidelines: str | None = None,
) -> str:
    """
    Build a prompt for generating evaluation criteria from a chat session.

    Args:
        chat_content: The raw chat session content
        dimensions: List of dimensions to create criteria for
        num_criteria: Number of criteria to generate, or None for auto
        category_hints: Optional list of category names to guide generation
        context: Optional additional context
        use_variables: If True, instruct LLM to extract variables. If False, use hard-coded values.
        guidelines: Optional specific guidelines/hints to guide criteria generation

    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    guidelines_section = f"\n\n**Generation Guidelines:**\n{guidelines}" if guidelines else ""

    # Format dimensions for prompt
    dimensions_str = "\n".join(
        [f"- {d.name} ({d.grading_type}): {d.description}" for d in dimensions]
    )

    category_guidance = (
        f"\n\nPreferred categories to use: {', '.join(category_hints)}"
        if category_hints
        else "\n\nSuggested categories: Tools, Output, Reasoning, Completeness, Accuracy"
    )

    count_instruction = (
        f"generate {num_criteria} specific evaluation criteria"
        if num_criteria is not None
        else "generate an appropriate number of specific evaluation criteria (between 7 and 12, create enough to check all important aspects including tool calls and key facts)"
    )

    if use_variables:
        variables_section = """**IMPORTANT - Variables Section:**
Extract specific data values from the chat session (IP addresses, RAM amounts, percentages, identifiers, etc.) and put them in a "variables" section. Variables should ONLY contain actual, correct values - NOT examples of incorrect values. Then use {{variable_name}} placeholders in your criterion text AND tool_calls params instead of hard-coding the values."""

        criteria_item_3 = (
            """3. **Use variables** - reference values using {{variable_name}} syntax"""
        )

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The response correctly states the RAM (~{{ram_total}}) and IP address ({{ip_address}})" - Mixes two values!
- RIGHT: Create SEPARATE criteria:
  1. "The response correctly states RAM as ~{{ram_total}}"
  2. "The response correctly states IP address as {{ip_address}}"
- ONE value per factual accuracy criterion - no exceptions"""

        json_example = """Return ONLY a JSON object with "variables" and "criteria" keys. Example format:
{
  "variables": {
    "ip_address": "10.0.187.159",
    "ram_total": "1.7GB",
    "host": "server01"
  },
  "criteria": [
    {
      "name": "core_tools_called",
      "category": "Tools",
      "weight": 3,
      "dimension": "tool_use",
      "criterion": "Must call essential diagnostic tools.",
      "tool_calls": {
        "respect_order": false,
        "required": [
          {"name": "get_system_info", "min_calls": 1, "params": {"host": "{{host}}"}},
          {"name": "get_memory_info", "min_calls": 1}
        ]
      }
    },
    {
      "name": "optional_diagnostics",
      "category": "Tools",
      "weight": 1,
      "dimension": "tool_use",
      "criterion": "Extra credit for additional diagnostics.",
      "tool_calls": {
        "optional": [
          {"name": "get_network_interfaces", "min_calls": 1}
        ]
      }
    },
    {
      "name": "no_dangerous_ops",
      "category": "Tools",
      "weight": 2,
      "dimension": "tool_use",
      "criterion": "Must not call destructive operations.",
      "tool_calls": {
        "prohibited": [
          {"name": "reboot_system"}
        ]
      }
    },
    {
      "name": "ip_address_correct",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_accuracy",
      "criterion": "The response correctly states the IP address is '{{ip_address}}'."
    }
  ]
}

Note:
- Extract actual tool names from the chat session
- Scoring inferred from lists: required=pass/fail, optional=bonus, prohibited=penalty
- Extract ALL specific data values to the variables section"""
    else:
        variables_section = """**IMPORTANT - No Variables Mode:**
Do NOT create a variables section. Use hard-coded values directly in criterion text and tool_calls params. Write specific, concrete values directly into the criteria."""

        criteria_item_3 = """3. **Use hard-coded values** - write specific values directly into criteria (e.g., "IP address is '10.0.187.159'" not "IP address is '{{ip_address}}'")"""

        atomic_examples = """**CRITICAL - Atomic Factual Accuracy Criteria:**
- WRONG: "The response correctly states the RAM (~1.7GB) and IP address (10.0.187.159)" - Mixes two values!
- RIGHT: Create SEPARATE criteria:
  1. "The response correctly states RAM as ~1.7GB"
  2. "The response correctly states IP address as 10.0.187.159"
- ONE value per factual accuracy criterion - no exceptions"""

        json_example = """Return ONLY a JSON object with "criteria" key (NO variables section). Example format:
{
  "criteria": [
    {
      "name": "core_tools_called",
      "category": "Tools",
      "weight": 3,
      "dimension": "tool_use",
      "criterion": "Must call essential diagnostic tools.",
      "tool_calls": {
        "respect_order": false,
        "required": [
          {"name": "get_system_info", "min_calls": 1, "params": {"host": "server01"}},
          {"name": "get_memory_info", "min_calls": 1}
        ]
      }
    },
    {
      "name": "ip_address_correct",
      "category": "Accuracy",
      "weight": 3,
      "dimension": "factual_accuracy",
      "criterion": "The response correctly states the IP address is '10.0.187.159'."
    }
  ]
}

Note:
- Extract actual tool names from the chat session
- Use hard-coded values directly in criteria - do NOT use variable placeholders"""

    return f"""Given the following chat session and dimensions, {count_instruction}.

**Chat Session:**
{chat_content}{context_info}{guidelines_section}

**Dimensions:**
{dimensions_str}{category_guidance}

**Instructions:**
Analyze the chat session above. If you detect tool calls in the session, create criteria that evaluate them.

{variables_section}

**CRITICAL - Granular Tool Criteria:**
When evaluating tool usage, create SEPARATE criteria for different tool categories. Scoring is inferred from which lists are populated:

1. **Required tools** (use `required` list) - Core/essential tools that MUST be called
   - Pass = full weight, Fail = 0

2. **Bonus tools** (use `optional` list only) - Nice-to-have tools
   - Pass = extra credit, Fail = 0 (no penalty for not calling)

3. **Penalty tools** (use `prohibited` list only) - Tools that should NOT be called
   - No violation = 0, Violation = negative score

**Strategy for tool criteria:**
- Create ONE criterion per tool category, not one giant criterion
- Each criterion has its own weight reflecting importance
- Granular scoring: required pass/fail, optional give bonus, prohibited deduct points

Criteria should be:
1. **ATOMIC** - each criterion checks exactly ONE specific thing (one fact, one value, one tool requirement)
2. **Fact-based where possible** - create separate criteria for each distinct fact or data point
{criteria_item_3}
4. **Never mix multiple values in one factual accuracy criterion** - this is critical for reliable evaluation
5. Measurable and unambiguous
6. Distributed across the provided dimensions
7. Given weights between 0-3 based on importance (3=most important, 0=informational only)
8. For score-type dimensions, use weight="from_scores" and criterion="from_scores"

{atomic_examples}

Each criterion should have:
- name: Unique identifier (lowercase with underscores)
- category: Category name (Tools, Output, Reasoning, etc.)
- weight: Integer 0-3, or "from_scores" for score-type dimensions
- dimension: Must reference one of the dimension names above
- criterion: Specific text describing what to check
- tool_calls: (ONLY for tool usage criteria) Tool call specification

**CRITICAL - Weight Constraints:**
- Criterion weight MUST be an integer from 0 to 3 (inclusive), OR the string "from_scores"
- DO NOT use weights outside the 0-3 range (e.g., 10 is INVALID)

**CRITICAL - Dimension Reference:**
- If referencing a dimension with grading_type "score", ensure that dimension has a "scores" dictionary defined

{json_example}"""
