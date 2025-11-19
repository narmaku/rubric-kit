"""Prompt templates for LLM-based evaluation and generation.

This module centralizes all prompts and LLM configurations used in rubric-kit for:
- Criterion evaluation (binary and score-based)
- Dimension generation
- Criteria generation  
- Rubric refinement

All prompts and configurations are designed to be easily identifiable and modifiable.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import yaml

from rubric_kit.schema import Criterion, Dimension


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

EVALUATOR_SYSTEM_PROMPT = (
    "You are a precise evaluator. Follow instructions exactly. "
    "Be concise and accurate."
)

GENERATOR_SYSTEM_PROMPT = (
    "You are an expert at creating evaluation rubrics. "
    "You always respond with valid JSON only, no additional text."
)


# =============================================================================
# LLM CONFIGURATIONS
# =============================================================================

@dataclass
class LLMConfig:
    """
    Configuration for LLM API calls.
    
    Bundles together all parameters needed for a specific LLM "persona":
    - System prompt defining the role
    - Temperature controlling randomness/creativity
    - Max tokens limiting response length
    
    This makes it easy to maintain different configurations for different
    use cases (e.g., deterministic evaluation vs creative generation).
    
    Attributes:
        system_prompt: The system message defining the LLM's role
        temperature: Controls randomness (0.0=deterministic, 1.0=creative)
        max_tokens: Maximum number of tokens in the response
    """
    system_prompt: str
    temperature: float
    max_tokens: int


# Named configurations for different LLM personas
EVALUATOR_CONFIG = LLMConfig(
    system_prompt=EVALUATOR_SYSTEM_PROMPT,
    temperature=0.0,  # Deterministic for consistent evaluation
    max_tokens=8192   # Sufficient for detailed evaluations
)

TOOL_CALL_EVALUATOR_CONFIG = LLMConfig(
    system_prompt=EVALUATOR_SYSTEM_PROMPT,
    temperature=0.0,  # Deterministic for consistent evaluation  
    max_tokens=16384   # More tokens needed for structural comparison and reasoning
)

GENERATOR_CONFIG = LLMConfig(
    system_prompt=GENERATOR_SYSTEM_PROMPT,
    temperature=0.7,  # More creative for generation tasks
    max_tokens=16384   # Longer responses for generating rubrics (increased for complex rubrics)
)


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

def build_binary_criterion_prompt(
    criterion: Criterion,
    chat_content: str
) -> str:
    """
    Build a prompt for binary (pass/fail) criterion evaluation.
    
    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content to evaluate
        
    Returns:
        Formatted prompt string for the LLM
    """
    return f"""You are an expert evaluator. Your task is to evaluate whether a chat session meets a specific criterion.

**Criterion Details:**
- Dimension: {criterion.dimension}
- Category: {criterion.category}
- Criterion: {criterion.criterion}

**Chat Session:**
{chat_content}

**Instructions:**

Carefully read the criterion above and determine what it requires. Then evaluate the chat session:

**Step 1 - Understand the requirement:**
- Does the criterion check for CORRECTNESS? (words like "correctly", "accurately", "true", or specifies exact values to match)
- Or does it check for PRESENCE? (words like "includes", "mentions", "contains")

**Step 2A - If checking CORRECTNESS:**
1. Find the authoritative source in the chat (tool outputs, function results, provided data)
2. Locate the specific data point mentioned in the criterion within that source
3. Extract the exact value from the source (this is ground truth)
4. Find what the assistant claimed about this in their final response
5. Compare: Does the assistant's claim match the source exactly?
   - Even small discrepancies = FAIL
   - Wrong numbers, wrong labels, wrong units = FAIL
   - Topic mentioned but value wrong = FAIL
   - Only PASS if values match exactly

**Step 2B - If checking PRESENCE:**
1. Look for the required information in the chat session
2. The information must be EXPLICITLY stated, not implied
3. Do NOT make inferences - only PASS if the information is directly stated
4. Do NOT consider related but different information - only exact matches count
5. PASS if present, FAIL if missing or incomplete

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence. For correctness: state source value and assistant's claim. For presence: quote relevant text or state what's missing.]

**Examples:**

RESULT: PASS
REASON: Source data shows "X=10" and assistant correctly stated "X is 10".

RESULT: FAIL
REASON: Source shows "value A" but assistant incorrectly claimed "value B".

RESULT: PASS
REASON: Response explicitly includes the required information about topic Z.

RESULT: FAIL
REASON: Required information about topic Y is not mentioned in the response.

**Your Response:**"""


def build_tool_call_evaluation_prompt(
    criterion: Criterion,
    chat_content: str,
    tool_call_sequence: Optional[List[str]] = None,
    parsed_tool_calls: Optional[List[Any]] = None
) -> str:
    """
    Build a prompt for tool call evaluation.
    
    Tool call evaluation compares extracted tool calls against specifications.
    If tool_call_sequence is provided (pre-parsed), evaluation is deterministic.
    Otherwise, the judge must extract tool calls from raw chat content.
    
    Args:
        criterion: The criterion with tool_calls specification
        chat_content: The chat session content to evaluate
        tool_call_sequence: Optional pre-parsed list of tool names in order
        parsed_tool_calls: Optional pre-parsed list of ToolCall objects with parameters
        
    Returns:
        Formatted prompt string for the LLM
        
    Raises:
        ValueError: If criterion doesn't have tool_calls defined
    """
    if not criterion.tool_calls:
        raise ValueError(
            f"Criterion '{criterion.name}' must have tool_calls defined for tool call evaluation"
        )
    
    tool_calls = criterion.tool_calls
    
    # Build required tools section with parameters
    required_section = ""
    if tool_calls.required:
        required_lines = []
        for tool in tool_calls.required:
            min_max = []
            if tool.min_calls is not None:
                min_max.append(f"min: {tool.min_calls}")
            if tool.max_calls is not None:
                min_max.append(f"max: {tool.max_calls}")
            constraint = f" ({', '.join(min_max)})" if min_max else ""
            
            # Add parameter requirements if specified
            params_info = ""
            if tool.params:
                params_list = [f"{k}: {v}" for k, v in tool.params.items()]
                params_info = f" with parameters: {', '.join(params_list)}"
            
            required_lines.append(f"  - {tool.name}{constraint}{params_info}")
        required_section = "**Required Tools:**\n" + "\n".join(required_lines)
    
    # Build optional tools section
    optional_section = ""
    if tool_calls.optional:
        optional_lines = []
        for tool in tool_calls.optional:
            max_constraint = f" (max: {tool.max_calls})" if tool.max_calls is not None else ""
            optional_lines.append(f"  - {tool.name}{max_constraint}")
        optional_section = "\n\n**Optional Tools:**\n" + "\n".join(optional_lines)
    
    # Build prohibited tools section
    prohibited_section = ""
    if tool_calls.prohibited:
        prohibited_lines = [f"  - {tool.name}" for tool in tool_calls.prohibited]
        prohibited_section = "\n\n**Prohibited Tools:**\n" + "\n".join(prohibited_lines)
    
    # Build numbered list of required tools for absolute clarity
    required_tool_list = ""
    required_tool_list_numbered = ""
    if tool_calls.required:
        tool_items = []
        tool_items_numbered = []
        for i, tool in enumerate(tool_calls.required, 1):
            tool_items.append(f"REQUIRED TOOL #{i}: {tool.name}")
            tool_items_numbered.append(f"{i}. {tool.name}")
        required_tool_list = "\n".join(tool_items)
        required_tool_list_numbered = "\n".join(tool_items_numbered)
    
    # Build explicit tool name lists for injection into instructions
    required_tool_names = [tool.name for tool in tool_calls.required] if tool_calls.required else []
    required_tool_names_list = ", ".join(required_tool_names) if required_tool_names else ""
    required_tool_names_bullets = "\n".join([f"   - {name}" for name in required_tool_names]) if required_tool_names else ""
    
    # Build parameter checking instructions if any tools have params specified
    has_params_to_check = any(tool.params for tool in tool_calls.required) if tool_calls.required else False
    param_check_instructions = ""
    if has_params_to_check:
        param_check_instructions = """
   
   **Check parameters** (CRITICAL)
   - For each required tool that specifies parameters, verify the actual call used the EXACT parameter values
   - Compare expected parameters (from specification above) with actual parameters (from extracted calls)
   - Parameter names must match exactly (case-sensitive)
   - Parameter values must match exactly (no partial matches, no "close enough")
   - Missing parameters = FAIL
   - Wrong parameter values = FAIL
   - Extra parameters are OK (only required ones must match)
   - If ANY required parameter is missing or wrong → FAIL"""
    
    # If pre-parsed tool sequence is provided, use it directly
    # Also include parameters if available
    actual_calls_section = ""
    if tool_call_sequence is not None:
        call_lines = []
        for i, name in enumerate(tool_call_sequence, 1):
            # Try to find parameters for this tool call
            params_str = ""
            if parsed_tool_calls:
                # Find matching tool call by name (handle both full names and function names)
                for tc in parsed_tool_calls:
                    # Match if: full_name matches, function matches, or name matches function
                    if (tc.full_name == name or 
                        tc.function == name or 
                        name.endswith(f".{tc.function}") or 
                        tc.full_name.endswith(f".{name}")):
                        if tc.parameters:
                            params_list = []
                            for k, v in tc.parameters.items():
                                if v is None:
                                    params_list.append(f"{k}: null")
                                else:
                                    params_list.append(f"{k}: {v}")
                            if params_list:
                                params_str = f" (parameters: {', '.join(params_list)})"
                        break
            
            call_lines.append(f"{i}. {name}{params_str}")
        
        actual_calls_section = f"""
**EXTRACTED TOOL CALLS (in order):**
{chr(10).join(call_lines)}
"""
    
    # Build evaluation instructions based on respect_order setting
    if tool_calls.respect_order:
        if tool_call_sequence is not None:
            # With pre-parsed data: direct comparison
            evaluation_body = f"""**Evaluation Instructions:**

Expected order:
{required_tool_list_numbered}

The specification requires these tools IN THIS EXACT ORDER:
{required_tool_list}
{actual_calls_section}

**Your task:**

1. **Compare the extracted calls against required order**
   - Position 1: Does extracted call #1 match REQUIRED TOOL #1?
   - Position 2: Does extracted call #2 match REQUIRED TOOL #2?
   - Continue for all positions
   - If ANY position doesn't match → ORDER IS WRONG → FAIL
   
2. **Check other requirements**
   - All required tools present? The required tools are:
{required_tool_names_bullets}
   - Call counts within limits (if specified)?
   - Optional tools within limits (if any)?
   - No prohibited tools called (if any)?{param_check_instructions}

3. **Final result**
   - Order wrong → FAIL
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL
   - Wrong or missing parameters → FAIL
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence. For order failures: state both the required order and actual order using the exact tool identifiers. For missing tools: you MUST state which specific tool from this list was not called: {required_tool_names_list}. For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual. Copy the exact tool identifier from the list above, such as "{required_tool_names[0] if required_tool_names else ''}" or another tool from the list.]
"""
        else:
            # Without pre-parsed data: must extract first
            evaluation_body = f"""**Evaluation Instructions:**

Expected order:
{required_tool_list_numbered}

The specification requires these tools IN THIS EXACT ORDER:
{required_tool_list}

**Your task:**

1. **Find the tool calls in the chat session**
   - Scan through the chat session and identify all tool calls
   - Extract the tool names in the order they were called
   
2. **Write down the actual order you found**
   - List them: "First tool called: <actual_tool_name>, Second tool called: <actual_tool_name>, ..."
   - IMPORTANT: Use the actual tool names you found in the chat session, not placeholders
   
3. **Compare against the required order**
   - Position 1: Does first tool called = REQUIRED TOOL #1? (MUST match exactly)
   - Position 2: Does second tool called = REQUIRED TOOL #2? (MUST match exactly)
   - Continue for all positions
   - If ANY position doesn't match → ORDER IS WRONG → FAIL
   
4. **Check other requirements**
   - All required tools present? The required tools are:
{required_tool_names_bullets}
   - Call counts within limits (if specified)?
   - Optional tools within limits (if any)?
   - No prohibited tools called (if any)?{param_check_instructions}

5. **Final result**
   - Order wrong → FAIL
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL
   - Wrong or missing parameters → FAIL
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence. For order failures: state both the required order and actual order using the exact tool names from the specification. For missing tools: you MUST state the exact tool name that was not called from this list: {required_tool_names_list}. For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual. Use the exact tool name, not a placeholder or the word "name".]
"""
        
    else:
        if tool_call_sequence is not None:
            # With pre-parsed data: direct presence check
            evaluation_body = f"""**Evaluation Instructions:**

The specification requires these tools (ORDER DOESN'T MATTER):
{required_tool_list}
{actual_calls_section}

**Your task:**

1. **Check presence**
   - The following required tools MUST be present in the extracted calls:
{required_tool_names_bullets}
   - Check if each of these exact tool identifiers appears in the extracted calls list above
   - Order doesn't matter
   - If reporting a missing tool in your REASON, copy one of these exact identifiers: {required_tool_names_list}
   
2. **Check counts** (if limits specified)
   - Are call counts within min/max limits?
   - Are optional tools within limits (if any)?{param_check_instructions}
   
3. **Check prohibitions** (if any)
   - Were any prohibited tools called?

4. **Final result**
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL
   - Called prohibited tool → FAIL
   - Wrong or missing parameters → FAIL
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence explaining what passed or what violation occurred. If a required tool is missing, you MUST copy one of these exact tool identifiers that was not called: {required_tool_names_list}. For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual. For example, if {required_tool_names[0] if required_tool_names else 'the first tool'} was not called, write: "Required tool {required_tool_names[0] if required_tool_names else '[tool_identifier]'} was not called."]
"""
        else:
            # Without pre-parsed data: must extract first
            evaluation_body = f"""**Evaluation Instructions:**

The specification requires these tools (ORDER DOESN'T MATTER):
{required_tool_list}

**Your task:**

1. **Find all tool calls in the chat session**
   - Scan through and identify all tool calls
   - Order doesn't matter for this evaluation
   
2. **Check presence**
   - The following required tools MUST be called at least once:
{required_tool_names_bullets}
   - Check if each of these exact tool identifiers appears in the chat session
   - If reporting a missing tool in your REASON, copy one of these exact identifiers: {required_tool_names_list}
   
3. **Check counts** (if limits specified)
   - Are call counts within min/max limits?
   - Are optional tools within limits (if any)?{param_check_instructions}
   
4. **Check prohibitions** (if any)
   - Were any prohibited tools called?

5. **Final result**
   - If any of the required tools ({required_tool_names_list}) is missing → FAIL
   - Violated any limit → FAIL
   - Called prohibited tool → FAIL
   - Wrong or missing parameters → FAIL
   - Otherwise → PASS

**Your response format (2 lines only):**
RESULT: [PASS or FAIL]
REASON: [One sentence explaining what passed or what violation occurred. If a required tool is missing, you MUST copy one of these exact tool identifiers that was not called: {required_tool_names_list}. For parameter failures: state which tool had wrong/missing parameters and what was expected vs actual. For example, if {required_tool_names[0] if required_tool_names else 'the first tool'} was not called, write: "Required tool {required_tool_names[0] if required_tool_names else '[tool_identifier]'} was not called."]
"""
    
    return f"""You are an expert at evaluating tool usage in chat sessions.

**Tool Usage Specification:**
{required_section}{optional_section}{prohibited_section}

**Chat Session:**
{chat_content}

{evaluation_body}

**Your Response:**"""


def build_score_criterion_prompt(
    criterion: Criterion,
    chat_content: str,
    dimension: Dimension
) -> str:
    """
    Build a prompt for score-based criterion evaluation.
    
    Args:
        criterion: The criterion to evaluate
        chat_content: The chat session content to evaluate
        dimension: The dimension with score scale definitions
        
    Returns:
        Formatted prompt string for the LLM
        
    Raises:
        ValueError: If dimension doesn't have scores defined
    """
    if not dimension.scores:
        raise ValueError(
            f"Dimension '{dimension.name}' does not have scores defined. "
            "Score-based evaluation requires a dimension with scores."
        )
    
    score_descriptions = "\n".join([
        f"{score}: {desc}" 
        for score, desc in sorted(dimension.scores.items())
    ])
    
    return f"""You are an expert evaluator. Your task is to score a chat session based on a specific criterion.

**Criterion Details:**
- Dimension: {criterion.dimension}
- Category: {criterion.category}
- Description: {dimension.description}

**Scoring Scale:**
{score_descriptions}

**Chat Session:**
{chat_content}

**Instructions:**
Read the scoring scale carefully. Evaluate the chat session and assign the most appropriate score.
Your response MUST be in this exact format (2 lines only):
SCORE: [numeric score from {min(dimension.scores.keys())} to {max(dimension.scores.keys())}]
REASON: [One sentence explaining why this score fits. Keep it brief and specific.]

Example response:
SCORE: 3
REASON: Response includes all essential information with no gaps.

**Your Response:**"""


# =============================================================================
# GENERATION PROMPTS
# =============================================================================

def build_dimension_generation_prompt(
    question: str,
    answer: str,
    num_dimensions: Optional[int],
    context: Optional[str] = None
) -> str:
    """
    Build a prompt for generating evaluation dimensions from a Q&A pair.
    
    Args:
        question: The question being evaluated
        answer: The answer being evaluated
        num_dimensions: Number of dimensions to generate, or None for auto
        context: Optional additional context
        
    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    
    if num_dimensions is None:
        count_instruction = "Generate an appropriate number of evaluation dimensions (between 3 and 10)"
    else:
        count_instruction = f"Generate {num_dimensions} evaluation dimensions"
    
    return f"""Given the following Question and Answer pair, {count_instruction} for assessing answer quality.

Question: {question}

Answer: {answer}{context_info}

Each dimension should:
1. Have a unique, descriptive name (lowercase with underscores, e.g., "factual_correctness")
2. Have a **GENERIC** description of what aspect it evaluates
3. **DO NOT** mention specific data values or fields in the dimension description
4. Specify a grading_type: either "binary" (pass/fail) or "score" (numeric scale from 1 to 3)
5. For "score" type, include a scores dictionary with integer keys and description values

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
      "1": "Missing most key information",
      "2": "Partially complete, missing some key details",
      "3": "Complete with all essential information"
    }}
  }}
]"""


def build_criteria_generation_prompt(
    question: str,
    answer: str,
    dimensions: List[Dimension],
    num_criteria: Optional[int],
    category_hints: Optional[List[str]] = None,
    context: Optional[str] = None
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
        
    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    
    # Format dimensions for prompt
    dimensions_str = "\n".join([
        f"- {d.name} ({d.grading_type}): {d.description}"
        for d in dimensions
    ])
    
    category_guidance = ""
    if category_hints:
        category_guidance = f"\n\nPreferred categories to use: {', '.join(category_hints)}"
    else:
        category_guidance = "\n\nSuggested categories: Output, Reasoning, Completeness, Accuracy, Clarity"
    
    if num_criteria is None:
        count_instruction = "generate an appropriate number of specific evaluation criteria (between 5 and 10, as many as needed to thoroughly evaluate the answer)"
    else:
        count_instruction = f"generate {num_criteria} specific evaluation criteria"
    
    return f"""Given the following Question, Answer, and Dimensions, {count_instruction}.

Question: {question}

Answer: {answer}{context_info}

Dimensions:
{dimensions_str}{category_guidance}

Criteria should be:
1. Specific and measurable
2. Distributed across the provided dimensions
3. Assigned appropriate categories (e.g., Output, Reasoning, Completeness)
4. Given weights between 1-3 based on importance (3=most important)
5. For score-type dimensions, use weight="from_scores" and criterion="from_scores"

Each criterion should have:
- name: Unique identifier (lowercase with underscores)
- category: Category name (will be auto-assigned based on the criterion type)
- weight: Integer 1-3, or "from_scores" for score-type dimensions
- dimension: Must reference one of the dimension names above
- criterion: Specific text describing what to check (or "from_scores" for score dimensions)

Return ONLY a JSON array of criterion objects. Example format:
[
  {{
    "name": "capital_accuracy",
    "category": "Accuracy",
    "weight": 3,
    "dimension": "factual_correctness",
    "criterion": "The answer must correctly identify Paris as the capital of France"
  }},
  {{
    "name": "completeness_score",
    "category": "Completeness",
    "weight": "from_scores",
    "dimension": "completeness",
    "criterion": "from_scores"
  }}
]"""


def build_refine_rubric_prompt(
    dimensions: List[Dimension],
    criteria: List[Criterion],
    feedback: Optional[str] = None
) -> str:
    """
    Build a prompt for refining an existing rubric.
    
    Args:
        dimensions: Current dimensions in the rubric
        criteria: Current criteria in the rubric
        feedback: Optional specific feedback for refinement
        
    Returns:
        Formatted prompt string for the LLM
    """
    # Convert to dict for YAML display
    rubric_dict = {
        "dimensions": [
            {
                "name": d.name,
                "description": d.description,
                "grading_type": d.grading_type,
                **({"scores": d.scores} if d.scores else {})
            }
            for d in dimensions
        ],
        "criteria": [
            {
                "name": c.name,
                "category": c.category,
                "weight": c.weight,
                "dimension": c.dimension,
                "criterion": c.criterion
            }
            for c in criteria
        ]
    }
    
    rubric_yaml = yaml.dump(rubric_dict, sort_keys=False)
    
    feedback_section = ""
    if feedback:
        feedback_section = f"\n\nSpecific Feedback:\n{feedback}"
    else:
        feedback_section = (
            "\n\nPlease improve the rubric by:\n"
            "- Making criteria more specific and measurable\n"
            "- Improving descriptions for clarity\n"
            "- Ensuring proper weight distribution\n"
            "- Adding detail where criteria are too vague"
        )
    
    return f"""Refine the following evaluation rubric to improve its quality.

Current Rubric:
{rubric_yaml}{feedback_section}

Return the refined rubric as JSON with the same structure. Maintain all dimension names that criteria reference.

Return ONLY a JSON object with this format:
{{
  "dimensions": [
    {{
      "name": "dimension_name",
      "description": "Clear description",
      "grading_type": "binary",
      "scores": {{"1": "desc", "2": "desc"}}
    }}
  ],
  "criteria": [
    {{
      "name": "criterion_name",
      "category": "Category",
      "weight": 3,
      "dimension": "dimension_name",
      "criterion": "Specific criterion text"
    }}
  ]
}}"""


def build_chat_dimension_generation_prompt(
    chat_content: str,
    num_dimensions: Optional[int],
    context: Optional[str] = None
) -> str:
    """
    Build a prompt for generating evaluation dimensions from a chat session.
    
    Args:
        chat_content: The raw chat session content
        num_dimensions: Number of dimensions to generate, or None for auto
        context: Optional additional context
        
    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    
    if num_dimensions is None:
        count_instruction = "Generate an appropriate number of evaluation dimensions (between 5 and 10, as many as needed)"
    else:
        count_instruction = f"Generate {num_dimensions} evaluation dimensions"
    
    return f"""Given the following chat session, {count_instruction} for assessing the assistant's performance.

**Chat Session:**
{chat_content}{context_info}

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
4. Specify a grading_type: either "binary" (pass/fail) or "score" (numeric scale from 1 to 3)
5. For "score" type, include a scores dictionary with integer keys and description values

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
      "1": "Poorly organized or difficult to understand",
      "2": "Generally clear but could be improved",
      "3": "Exceptionally clear and well-organized"
    }}
  }}
]"""


def build_chat_criteria_generation_prompt(
    chat_content: str,
    dimensions: List[Dimension],
    num_criteria: Optional[int],
    category_hints: Optional[List[str]] = None,
    context: Optional[str] = None
) -> str:
    """
    Build a prompt for generating evaluation criteria from a chat session.
    
    Args:
        chat_content: The raw chat session content
        dimensions: List of dimensions to create criteria for
        num_criteria: Number of criteria to generate, or None for auto
        category_hints: Optional list of category names to guide generation
        context: Optional additional context
        
    Returns:
        Formatted prompt string for the LLM
    """
    context_info = f"\n\nAdditional Context: {context}" if context else ""
    
    # Format dimensions for prompt
    dimensions_str = "\n".join([
        f"- {d.name} ({d.grading_type}): {d.description}"
        for d in dimensions
    ])
    
    category_guidance = ""
    if category_hints:
        category_guidance = f"\n\nPreferred categories to use: {', '.join(category_hints)}"
    else:
        category_guidance = "\n\nSuggested categories: Tools, Output, Reasoning, Completeness, Accuracy"
    
    if num_criteria is None:
        count_instruction = "generate an appropriate number of specific evaluation criteria (between 7 and 10, create enough to check all important aspects including each tool call and each key fact in the response)"
    else:
        count_instruction = f"generate {num_criteria} specific evaluation criteria"
    
    return f"""Given the following chat session and dimensions, {count_instruction}.

**Chat Session:**
{chat_content}{context_info}

**Dimensions:**
{dimensions_str}{category_guidance}

**Instructions:**
Analyze the chat session above. If you detect tool calls in the session, create criteria that evaluate them.

Criteria should be:
1. **Atomic and specific** - each criterion should check ONE specific thing (e.g., "value X is correct", not "all values are correct")
2. **Fact-based where possible** - for factual information, create separate criteria for each distinct fact or data point
3. Measurable and unambiguous
4. Distributed across the provided dimensions
5. Assigned appropriate categories (Tools, Accuracy, Completeness, Output, Clarity, etc.)
6. Given weights between 1-3 based on importance (3=most important, 1=nice-to-have)
7. For score-type dimensions, use weight="from_scores" and criterion="from_scores"
8. **IMPORTANT - Tool Usage Criteria**:
   - For tool evaluation, typically create ONE comprehensive criterion (e.g., "tools_used_correctly")
   - This single criterion uses the "tool_calls" specification to check ALL aspects: selection, order, count, and params
   - The tool_calls specification should include:
     * respect_order: true/false (whether tool call order matters)
     * required: List of required tools (detect from session) with min_calls and max_calls
     * optional: List of optional tools that can be called
     * prohibited: List of tools that should not be called
   - Only create separate tool criteria if you need different importance weights for different aspects

**Strategy for creating atomic criteria:**
- If the response mentions a specific value, create a criterion checking that exact value
- Each distinct fact should have its own criterion (e.g., quantity, identifier, measurement as separate criteria)
- Multiple atomic criteria can all reference the SAME generic dimension (e.g., all fact checks use "factual_accuracy")
- Prefer many specific criteria over few general ones
- Each criterion tests ONE specific thing but references a generic dimension

Each criterion should have:
- name: Unique identifier (lowercase with underscores)
- category: Category name (Tools, Output, Reasoning, etc.)
- weight: Integer 1-3, or "from_scores" for score-type dimensions
- dimension: Must reference one of the dimension names above
- criterion: Specific text describing what to check (or "from_scores" for score dimensions)
- tool_calls: (ONLY for tool usage criteria) Tool call specification object

Return ONLY a JSON array of criterion objects. Example format:
[
  {{
    "name": "tools_used_correctly",
    "category": "Tools",
    "weight": 3,
    "dimension": "tool_use",
    "criterion": "Assistant must call the appropriate tools in the correct order with valid parameters",
    "tool_calls": {{
      "respect_order": true,
      "required": [
        {{"name": "first_tool_from_session", "min_calls": 1, "max_calls": 1}},
        {{"name": "second_tool_from_session", "min_calls": 1, "max_calls": 1}}
      ],
      "optional": [],
      "prohibited": []
    }}
  }},
  {{
    "name": "key_value_correct",
    "category": "Accuracy",
    "weight": 3,
    "dimension": "factual_accuracy",
    "criterion": "The response correctly states the primary value or identifier"
  }},
  {{
    "name": "measurement_correct",
    "category": "Accuracy",
    "weight": 3,
    "dimension": "factual_accuracy",
    "criterion": "The response correctly reports the measured quantity"
  }},
  {{
    "name": "secondary_detail_correct",
    "category": "Accuracy",
    "weight": 2,
    "dimension": "factual_accuracy",
    "criterion": "The response correctly identifies the secondary attribute or detail"
  }},
  {{
    "name": "completeness_score",
    "category": "Completeness",
    "weight": "from_scores",
    "dimension": "completeness",
    "criterion": "from_scores"
  }}
]

Note: 
- Extract actual tool names from the chat session above when creating tool_calls specifications
- Multiple criteria can reference the same dimension (e.g., all factual checks use "factual_accuracy")"""

