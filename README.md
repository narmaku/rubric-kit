# Rubric Kit

Automatic rubric evaluation using LLM-as-a-Judge. Define your rubrics with custom dimensions, descriptors, and criteria, then let an LLM automatically evaluate chat sessions against your rubric. Output detailed scores in CSV format with pretty-printed tables.

## Features

- **🤖 LLM-as-a-Judge**: Automatic criterion evaluation using OpenAI-compatible LLMs
- **Schema Validation**: Pydantic-based validation for rubric YAML files
- **Flexible Grading**: Support for binary (pass/fail) and score-based (1-N) grading
- **Tool Call Validation**: Define required, optional, and prohibited tool calls
- **CSV Export**: Export evaluation results to CSV with optional summary rows
- **Pretty Tables**: Display results in formatted tables in the terminal
- **OpenAI Compatible**: Works with any OpenAI-compatible endpoint
- **Comprehensive Testing**: Full test coverage with pytest (44 tests)

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py <chat_session_file> <rubric_yaml> <output_csv>
```

### Examples

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Basic usage
python main.py example_chat_session.txt example.yaml results.csv

# With custom model
python main.py chat.txt rubric.yaml output.csv --model gpt-4-turbo

# With custom OpenAI-compatible endpoint
python main.py chat.txt rubric.yaml output.csv \
  --model gpt-4 \
  --base-url https://your-endpoint.com/v1

# With API key as argument
python main.py chat.txt rubric.yaml output.csv --api-key sk-...
```

### Command-Line Options

**General Options:**
- `--no-table`: Skip printing the results table to console
- `--include-summary`: Include a summary row in CSV output
- `--help`: Show help message

**LLM Configuration:**
- `--api-key KEY`: OpenAI API key (or set `OPENAI_API_KEY` env var)
- `--base-url URL`: Base URL for OpenAI-compatible endpoint
- `--model MODEL`: Model name to use (default: `gpt-4`)

## Rubric YAML Format

Define your rubric in YAML format with descriptors and criteria:

```yaml
descriptors:
  - factual_correctness: Evaluates that the information is factually correct.
    grading_type: binary  # pass/fail
  
  - usefulness: Evaluates how useful is the final response.
    grading_type: score  # must specify scores
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
      required:
        - get_system_information:
          min_calls: 1
          max_calls: 1
          params:
      optional:
        - get_network_interfaces:
          max_calls: 1
          params:
      prohibited:
        - get_weather:
          params:
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

### Console Table

Results are displayed in a formatted table:

```
+----------------------+------------+---------------------+------------+---------+-------------+
| Criterion            | Category   | Dimension           | Result     | Score   | Max Score   |
+======================+============+=====================+============+=========+=============+
| sys_info_factual_1   | Output     | factual_correctness | pass       | 3       | 3           |
+----------------------+------------+---------------------+------------+---------+-------------+
| useful_1             | Output     | usefulness          | 3 - Very   | 3       | 3           |
|                      |            |                     | useful     |         |             |
+----------------------+------------+---------------------+------------+---------+-------------+
| TOTAL                |            |                     | 100.0%     | 6       | 6           |
+----------------------+------------+---------------------+------------+---------+-------------+
```

### CSV Output

Results are exported to CSV with all evaluation details:

```csv
criterion_name,category,dimension,criterion_text,result,score,max_score,score_description
sys_info_factual_1,Output,factual_correctness,The response must...,pass,3,3,
useful_1,Output,usefulness,from_scores,3,3,3,Very useful
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
│   ├── output.py               # CSV and table output
│   ├── llm_judge.py            # LLM-based criterion evaluation
│   └── main.py                 # CLI entry point
├── tests/                      # Test suite (45 tests)
│   ├── test_schema.py
│   ├── test_validator.py
│   ├── test_processor.py
│   ├── test_output.py
│   ├── test_llm_judge.py
│   └── test_main.py
├── main.py                     # Command-line entry point
├── example.yaml                # Example rubric file
├── example_chat_session.txt    # Example chat session
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Rubric Components

### Descriptors

Descriptors define evaluation dimensions:

- **name**: Unique identifier
- **description**: What this descriptor evaluates
- **grading_type**: Either `binary` or `score`
- **scores**: Required for `score` type; maps score values to descriptions

### Criteria

Criteria define specific evaluation rules:

- **name**: Unique identifier
- **category**: Optional category (e.g., "Output", "Tools")
- **weight**: Score weight (0-3) or `from_scores`
- **dimension**: References a descriptor name
- **criterion**: Description of what to evaluate
- **tool_calls**: Optional tool call specifications (for Tools category)

### Tool Calls

For criteria that evaluate tool usage:

- **respect_order**: Whether order matters (default: true)
- **required**: List of required tool calls with min/max constraints
- **optional**: List of optional tool calls
- **prohibited**: List of prohibited tool calls

## Examples

**Included Files:**
- `example.yaml` - Complete rubric with 3 descriptors and 5 criteria
- `example_chat_session.txt` - Sample chat session for evaluation

**Try it out:**
```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Run the example
python main.py example_chat_session.txt example.yaml results.csv

# View results in terminal (automatic)
# Or check results.csv for detailed breakdown
```

## License

See LICENSE file for details.
