# Rubric Kit

Automatic rubric evaluation using LLM-as-a-Judge. Create, refine, and apply evaluation rubrics powered by AI. Generate rubrics from Q&A pairs, evaluate chat sessions, and export detailed scores.

## Features

- **✨ AI-Powered Rubric Generation**: Automatically generate high-quality rubrics from Q&A pairs
- **🔄 Rubric Refinement**: Improve existing rubrics with AI-guided feedback
- **🤖 Multi-Judge Panel**: Use multiple LLMs as judges with configurable consensus mechanisms
- **⚖️ Consensus Modes**: Quorum, majority, and unanimous consensus for reliable evaluations
- **🚀 Flexible Execution**: Sequential, parallel, or batched judge execution
- **Schema Validation**: Pydantic-based validation for rubric YAML files
- **Flexible Grading**: Support for binary (pass/fail) and score-based (1-N) grading
- **Tool Call Validation**: Define required, optional, and prohibited tool calls
- **📄 PDF Reports**: Generate comprehensive PDF reports with charts, summaries, and rubric appendix
- **📊 Export Formats**: Convert evaluation results to PDF, CSV, or JSON
- **Pretty Tables**: Display results in formatted tables in the terminal
- **OpenAI Compatible**: Works with any OpenAI-compatible endpoint
- **Comprehensive Testing**: Full test coverage with pytest (200+ tests)
- **Customizable Prompts & LLM Configs**: All prompts and LLM parameters centralized for easy modification
- **Specialized Tool Call Evaluation**: Dedicated prompts for parsing and validating tool usage patterns

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

Rubric Kit provides five main commands:
- `generate`: Create a new rubric from a Q&A pair or chat session
- `evaluate`: Evaluate a chat session against a rubric (outputs self-contained YAML)
- `refine`: Improve an existing rubric
- `export`: Convert evaluation YAML to PDF, CSV, or JSON format
- `rerun`: Re-evaluate using settings from a previous self-contained output

### Generate a Rubric

Create a rubric automatically from a Question & Answer pair:

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Basic usage
rubric-kit generate qa_input.txt output_rubric.yaml

# With custom parameters
rubric-kit generate qa_input.txt rubric.yaml \
  --num-dimensions 5 \
  --num-criteria 8 \
  --categories "Output,Reasoning,Completeness"

# With custom model
rubric-kit generate qa.txt rubric.yaml --model gpt-4-turbo
```

**Q&A Input Format:**

Simple text format:
```text
Q: What is the capital of France?
A: The capital of France is Paris.
```

Or YAML format:
```yaml
question: What is the capital of France?
answer: The capital of France is Paris.
context: Testing geography knowledge  # optional
```

**Generate Options:**
- `--num-dimensions N`: Number of dimensions to generate (1-10, default: 5)
- `--num-criteria N`: Number of criteria to generate (1-10, default: 7)
- `--categories LIST`: Comma-separated category hints (e.g., "Output,Reasoning")
- `--model MODEL`: Model to use (default: `gpt-4`)
- `--api-key KEY`: OpenAI API key
- `--base-url URL`: Custom API endpoint

### Evaluate a Chat Session

Evaluate a chat session against an existing rubric. The evaluation **always outputs a YAML file** (the source of truth artifact).

```bash
# Basic usage (always outputs YAML)
rubric-kit evaluate --from-chat-session chat_session.txt --rubric-file rubric.yaml --output-file results.yaml

# With PDF report generation
rubric-kit evaluate --from-chat-session chat.txt --rubric-file rubric.yaml --output-file output.yaml --report report.pdf

# With custom report title
rubric-kit evaluate --from-qna qna.yaml --rubric-file rubric.yaml --output-file output.yaml --report report.pdf --report-title "Q1 2025 Evaluation"

# With custom model
rubric-kit evaluate --from-chat-session chat.txt --rubric-file rubric.yaml --output-file output.yaml --model gpt-4-turbo
```

**Judge Panel Configuration:**

Rubric Kit supports multi-judge evaluation where multiple LLMs can evaluate each criterion and reach consensus. This provides more reliable and robust evaluations.

```bash
# Using a judge panel configuration file
rubric-kit evaluate chat.txt rubric.yaml output.csv --judge-panel-config panel.yaml
```

Example judge panel configuration (`panel.yaml`):

```yaml
judge_panel:
  judges:
    - name: primary
      model: gpt-4
      api_key: ${OPENAI_API_KEY}  # Can reference environment variables
    - name: secondary  
      model: gpt-4-turbo
      api_key: ${OPENAI_API_KEY}
    - name: tertiary
      model: claude-3-5-sonnet-20241022
      api_key: ${ANTHROPIC_API_KEY}
      base_url: https://api.anthropic.com/v1
  
  execution:
    mode: sequential  # Options: sequential, parallel, batched
    batch_size: 2     # Used for batched mode
    timeout: 30       # Timeout per judge in seconds
  
  consensus:
    mode: quorum           # Options: quorum, majority, unanimous
    threshold: 2           # Required for quorum mode (2 out of 3)
    on_no_consensus: fail  # Options: fail (conservative), median, most_common
```

**Execution Modes:**
- `sequential`: Judges called one by one (default, safest for rate limits)
- `parallel`: All judges called simultaneously (fastest, may hit rate limits)
- `batched`: Judges called in batches (balance between speed and rate limits)

**Consensus Modes:**
- `unanimous`: All judges must agree (threshold = number of judges)
- `majority`: More than 50% of judges must agree (threshold auto-calculated)
- `quorum`: Specific number of judges must agree (threshold configurable)

**No Consensus Handling:**
- `fail`: Use minimum score/fail (conservative, default)
- `median`: Use median of all judge scores
- `most_common`: Use most frequent score/result

**Single Judge (Default):**
If no judge panel config is provided, a single-judge panel is created automatically using CLI arguments:

```bash
# Traditional single-judge usage (backward compatible)
rubric-kit evaluate chat.txt rubric.yaml output.csv \
  --model gpt-4 \
  --api-key your-key
```

**Evaluate Options:**
- `--from-chat-session FILE`: Path to chat session file
- `--from-qna FILE`: Path to Q&A YAML file (alternative to chat session)
- `--rubric-file FILE`: Path to rubric YAML file
- `--output-file FILE`: Path to output YAML file (self-contained)
- `--report FILE`: Path to generate PDF report (optional)
- `--report-title TEXT`: Custom title for the PDF report (optional)
- `--include-input`: Embed input content in output YAML (for rerun capability)
- `--judge-panel-config FILE`: Path to judge panel configuration YAML file
- `--no-table`: Skip printing results table to console
- `--model MODEL`: Model to use for default single-judge panel (default: `gpt-4`)
- `--api-key KEY`: OpenAI API key for default single-judge panel
- `--base-url URL`: Custom API endpoint for default single-judge panel

### Refine a Rubric

Improve an existing rubric with AI-guided refinement:

```bash
# Basic usage (overwrites original)
rubric-kit refine rubric.yaml

# With specific feedback
rubric-kit refine rubric.yaml --feedback "Add more specific criteria for accuracy"

# Save to new file
rubric-kit refine rubric.yaml --output refined_rubric.yaml

# With custom model
rubric-kit refine rubric.yaml --model gpt-4-turbo
```

**Refine Options:**
- `--feedback TEXT`: Specific feedback for refinement
- `--output PATH`: Output path (default: overwrite original)
- `--model MODEL`: Model to use (default: `gpt-4`)
- `--api-key KEY`: OpenAI API key
- `--base-url URL`: Custom API endpoint

### Export Evaluation Results

Convert the evaluation YAML file to other formats (PDF, CSV, or JSON):

```bash
# Export to PDF report
rubric-kit export results.yaml --format pdf --output report.pdf

# Export to CSV
rubric-kit export results.yaml --format csv --output results.csv

# Export to JSON
rubric-kit export results.yaml --format json --output results.json
```

**Export Options:**
- `--format TYPE`: Output format: `pdf`, `csv`, or `json` (required)
- `--output FILE` / `-o FILE`: Path to output file (required)

**PDF Report Contents:**
- Custom title (from `--report-title` or metadata)
- Executive summary with scores
- LLM Judges Panel summary
- Score distribution and dimension breakdown charts
- Detailed results table
- Rubric Appendix (Dimensions and Criteria)

### Re-run Evaluation

Re-evaluate using the rubric and judge panel settings from a previous self-contained output:

```bash
# Re-run with embedded input content (if --include-input was used originally)
rubric-kit rerun results.yaml --output-file new_results.yaml

# Re-run with new input (same rubric and judge settings)
rubric-kit rerun results.yaml --from-chat-session new_chat.txt --output-file new_results.yaml

# Re-run with new Q&A input
rubric-kit rerun results.yaml --from-qna new_qna.yaml --output-file new_results.yaml

# Re-run and generate PDF report
rubric-kit rerun results.yaml --output-file new_results.yaml --report report.pdf
```

**Rerun Options:**
- `--output-file FILE` / `-o FILE`: Path to output YAML file (required)
- `--from-chat-session FILE`: Use new chat session input (optional)
- `--from-qna FILE`: Use new Q&A input (optional)
- `--include-input`: Include input content in output YAML
- `--report FILE`: Generate PDF report (optional)
- `--no-table`: Skip printing results table
- `--api-key KEY`: API key for LLM calls

## Rubric YAML Format

Define your rubric in YAML format with dimensions and criteria:

```yaml
dimensions:
  - factual_correctness: Evaluates that the information is factually correct.
    grading_type: binary  # pass/fail
  
  - usefulness: Evaluates how useful is the final response.
    grading_type: score  # must specify scores
    pass_above: 2  # optional: scores >= 2 show as "pass" in result column
    scores:
      1: The response is completely useless.
      2: The response is useful but incomplete.
      3: The response is useful and complete.
  
  - tool_usage: Evaluates whether the model selected the correct tools.
    grading_type: binary

criteria:
  sys_info_factual_1:
    category: Output
    weight: 3  # Range is 0-3, where 0 disables the criterion
    dimension: factual_correctness
    criterion: The response must indicate that number of physical CPUs is 8.
  
  useful_1:
    category: Output
    weight: from_scores  # Use score value as weight
    dimension: usefulness
    criterion: from_scores
  
  tool_call_1:
    category: Tools
    weight: 3
    dimension: tool_usage
    tool_calls:
      respect_order: true  # Whether tool call order matters
      params_strict_mode: false  # If true, exactly specified params must match
      required:
        - get_system_information:
          min_calls: 1
          max_calls: 1
          params:  # Omit for no validation, {} for no params, {key: value} for specific params
            hostname: example.com
      optional:
        - get_network_interfaces:
          max_calls: 1
          # params omitted = no validation
      prohibited:
        - get_weather:
          # params omitted = no validation
```

## Chat Session File Format

Provide a plain text file with the chat session:

```text
User: What are the system specifications?
Assistant: The system has 8 physical CPUs and 64 GB of RAM.

Tool calls:
- get_system_information() -> {"cpus": 8, "ram_gb": 64}
```

The LLM will analyze this conversation and automatically evaluate each criterion in your rubric. Include:
- User queries and assistant responses
- Tool calls and their results
- Any other relevant context

The format is flexible - just write the chat session in a natural, readable way.

## Output Formats

### Standardized Workflow

1. **Run evaluation** → Always produces `output.yaml` (source of truth)
2. **Use export command** → Convert to PDF, CSV, or JSON as needed

```bash
# Step 1: Evaluate and produce YAML
rubric-kit evaluate --from-chat-session chat.txt --rubric-file rubric.yaml --output-file results.yaml

# Step 2: Export to desired format
rubric-kit export results.yaml --format pdf --output report.pdf
rubric-kit export results.yaml --format csv --output results.csv
```

### Console Table

Results are displayed in a formatted table:

```
+----------------------+------------+---------------------+------------+---------+-----------+-----------+
| Criterion            | Dimension  | Result              | Score      | Consensus| Agreement |
+======================+============+=====================+============+=========+===========+
| sys_info_factual_1   | factual    | pass                | 3/3        | ✓       | 2/2       |
+----------------------+------------+---------------------+------------+---------+-----------+
| useful_1             | usefulness | pass                | 3/3        | ✓       | 2/2       |
+----------------------+------------+---------------------+------------+---------+-----------+
| TOTAL                |            | 100.0%              | 6/6        |         |           |
+----------------------+------------+---------------------+------------+---------+-----------+
```

**Column Descriptions:**
- **Result**: Shows `pass`/`fail` for binary criteria and score-based criteria with `pass_above` defined
- **Score**: The actual score value (used for calculating total score)
- **Consensus**: Whether judges reached consensus (✓ or ⚠)
- **Agreement**: Number of agreeing judges out of total

### YAML Output (Self-Contained)

The YAML file is the **self-contained source-of-truth artifact** containing everything needed for PDF generation, post-processing, sharing, and re-running:

```yaml
# Evaluation results
results:
  - criterion_name: sys_info_factual_1
    category: Output
    dimension: factual_correctness
    result: pass
    score: 3
    max_score: 3
    reason: The response correctly shows 8 CPUs

# Summary scores
summary:
  total_score: 6
  max_score: 6
  percentage: 100.0

# Full rubric (portable, self-contained)
rubric:
  dimensions:
    - name: factual_correctness
      description: "..."
      grading_type: binary
  criteria:
    - name: sys_info_factual_1
      category: Output
      dimension: factual_correctness
      criterion: "..."
      weight: 3

# Full judge panel config (portable, no API keys)
judge_panel:
  judges:
    - name: primary
      model: gpt-4
      base_url: null
  execution:
    mode: sequential
    batch_size: 2
    timeout: 30
  consensus:
    mode: unanimous
    threshold: 1
    on_no_consensus: fail

# Input reference (and optional content)
input:
  type: chat_session
  source_file: "chat.txt"
  content: "..."  # Only if --include-input was used

# Metadata
metadata:
  timestamp: "2025-01-01T12:00:00"
  report_title: "Q1 2025 Evaluation"
  rubric_source_file: "rubric.yaml"
```

### CSV/JSON Export

Use the `export` command to convert YAML to other formats:

```bash
rubric-kit export results.yaml --format csv --output results.csv
rubric-kit export results.yaml --format json --output results.json
```

## Development

### Running Tests

Run all tests:

```bash
pytest
```

Run specific test file:

```bash
pytest tests/test_schema.py -v
```

Run with coverage:

```bash
pytest --cov=rubric_kit --cov-report=html
```

### Project Structure

```
rubric-kit/
├── rubric_kit/                 # Main package
│   ├── __init__.py
│   ├── schema.py               # Pydantic models for validation
│   ├── validator.py            # YAML validation logic
│   ├── processor.py            # Score processing
│   ├── output.py               # YAML, CSV, JSON and table output
│   ├── pdf_export.py           # PDF report generation
│   ├── prompts.py              # LLM prompt templates
│   ├── llm_judge.py            # LLM-based criterion evaluation
│   ├── generator.py            # Rubric generation from Q&A
│   └── main.py                 # CLI entry point with subcommands
├── tests/                      # Test suite (228 tests)
│   ├── test_schema.py
│   ├── test_validator.py
│   ├── test_processor.py
│   ├── test_output.py
│   ├── test_pdf_export.py
│   ├── test_prompts.py
│   ├── test_llm_judge.py
│   ├── test_generator.py
│   └── test_main.py
├── main.py                     # Command-line entry point
├── example.yaml                # Example rubric file
├── example_chat_session.txt    # Example chat session
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Rubric Components

### Dimensions

Dimensions define evaluation aspects:

- **name**: Unique identifier
- **description**: What this dimension evaluates
- **grading_type**: Either `binary` (pass/fail) or `score` (numeric scale)
- **scores**: Required for `score` type; maps score values to descriptions
- **pass_above**: Optional for `score` type; minimum score threshold to show as "pass" in result column

### Criteria

Criteria define specific evaluation rules:

- **name**: Unique identifier
- **category**: Optional category (e.g., "Output", "Tools")
- **weight**: Score weight (0-3) or `from_scores`
- **dimension**: References a descriptor name
- **criterion**: Description of what to evaluate
- **tool_calls**: Optional tool call specifications (for Tools category)

### Tool Calls

For criteria that evaluate tool usage, rubric-kit provides **specialized tool call evaluation** with structured parsing:

- **respect_order**: Whether order matters (default: true)
- **params_strict_mode**: If true, exactly the specified params must match (no extra params allowed). Default: false
- **required**: List of required tool calls with min/max constraints
- **optional**: List of optional tool calls
- **prohibited**: List of prohibited tool calls

**Parameter Validation:**

The `params` field for each tool specification controls parameter validation:

- **`params` omitted/not declared**: No parameter validation - tool can be called with any parameters or none
- **`params: {}`** (empty dict): Explicitly check that the tool was called with **NO parameters** (fails if any params were used)
- **`params: {key: value}`**: Check that the specified parameters match exactly. Extra parameters are ignored unless `params_strict_mode: true`

**Examples:**

```yaml
tool_calls:
  respect_order: false
  params_strict_mode: false  # Allow extra params (default)
  required:
    - tool_with_params:
        params:
          hostname: example.com
          port: 8080
    - tool_no_params:
        params: {}  # Must be called with NO parameters
    - tool_any_params:
        # params not specified = no validation
```

**Tool Call Evaluation Features:**
- Automatically parses tool calls from chat sessions (handles various formats)
- Counts tool call occurrences and verifies min/max constraints
- Checks tool call order when `respect_order: true`
- Validates that all required tools were called
- Validates tool call parameters based on specification
- Ensures no prohibited tools were used
- Uses a dedicated prompt template optimized for structured parsing

## Complete Workflow Example

Here's a complete workflow showing all five commands:

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# 1. Generate a rubric from a Q&A pair
echo "Q: What is the capital of France?" > qa.txt
echo "A: The capital of France is Paris, known for the Eiffel Tower." >> qa.txt

rubric-kit generate --from-qna qa.txt --output-file geography_rubric.yaml \
  --num-dimensions 3 \
  --num-criteria 5

# 2. (Optional) Refine the generated rubric
rubric-kit refine --rubric-file geography_rubric.yaml \
  --feedback "Add more emphasis on accuracy and completeness"

# 3. Evaluate a chat session (outputs self-contained YAML)
rubric-kit evaluate --from-chat-session example_chat_session.txt \
  --rubric-file geography_rubric.yaml \
  --output-file results.yaml \
  --include-input \
  --report report.pdf \
  --report-title "Geography Evaluation Report"

# 4. Export to other formats as needed
rubric-kit export results.yaml --format csv --output results.csv
rubric-kit export results.yaml --format json --output results.json

# 5. Re-run with different input (using same rubric & judge settings)
rubric-kit rerun results.yaml --from-chat-session new_chat.txt --output-file new_results.yaml
```

**Included Files:**
- `example.yaml` - Complete rubric with 3 dimensions and 5 criteria  
- `example_chat_session.txt` - Sample chat session for evaluation
- `example_qa.txt` - Sample Q&A for rubric generation

**Try the included example:**
```bash
export OPENAI_API_KEY="your-api-key-here"
rubric-kit evaluate --from-chat-session example_chat_session.txt --rubric-file example.yaml --output-file results.yaml
```

## Customizing Prompts and LLM Behavior

All LLM prompts and configurations are centralized in `rubric_kit/prompts.py` for easy modification and customization.

### LLM Configurations

The module provides **LLM Configuration objects** that bundle together all parameters for different "personas":

```python
@dataclass
class LLMConfig:
    system_prompt: str   # The system message defining the LLM's role
    temperature: float   # Controls randomness (0.0=deterministic, 1.0=creative)
    max_tokens: int      # Maximum tokens in the response
```

**Named Configurations:**

- `EVALUATOR_CONFIG`: Deterministic evaluation (temp=0.0, max_tokens=120)
  - Used for consistent criterion evaluation
  - Precise, focused, reproducible results
  
- `GENERATOR_CONFIG`: Creative generation (temp=0.7, max_tokens=2000)
  - Used for rubric generation and refinement
  - More flexible and creative outputs

### Prompt Building Functions

- `build_binary_criterion_prompt()`: Creates prompts for pass/fail content evaluation
- `build_score_criterion_prompt()`: Creates prompts for score-based content evaluation
- `build_tool_call_evaluation_prompt()`: Creates prompts for tool call evaluation (parsing, counting, order checking)
- `build_dimension_generation_prompt()`: Creates prompts for generating dimensions
- `build_criteria_generation_prompt()`: Creates prompts for generating criteria
- `build_refine_rubric_prompt()`: Creates prompts for rubric refinement

### For Contributors

To modify LLM behavior, edit `prompts.py`:

**Adjust LLM parameters for a persona:**
```python
# In prompts.py
EVALUATOR_CONFIG = LLMConfig(
    system_prompt=EVALUATOR_SYSTEM_PROMPT,
    temperature=0.0,    # Change this for more/less randomness
    max_tokens=120      # Change this for longer/shorter responses
)
```

**Modify prompt templates:**
```python
# All prompt building functions are in prompts.py
def build_binary_criterion_prompt(criterion, chat_content):
    return f"""Your custom prompt here..."""
```

**Benefits of this approach:**
- ✅ All related settings grouped together
- ✅ Easy to test different configurations
- ✅ Simple to maintain - one place to change per persona
- ✅ Type-safe with dataclass validation
- ✅ No parameter proliferation in function signatures

## License

See LICENSE file for details.
