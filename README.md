# Rubric Kit

Generate high-quality rubrics based on custom dimensions, descriptors, criteria, and scoring system. Evaluate responses against rubrics and output detailed scores in CSV format with pretty-printed tables.

## Features

- **Schema Validation**: Pydantic-based validation for rubric YAML files
- **Flexible Grading**: Support for binary (pass/fail) and score-based (1-N) grading
- **Tool Call Validation**: Define required, optional, and prohibited tool calls
- **CSV Export**: Export evaluation results to CSV with optional summary rows
- **Pretty Tables**: Display results in formatted tables in the terminal
- **Comprehensive Testing**: Full test coverage with pytest

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python main.py <evaluations_file> <rubric_yaml> <output_csv>
```

### Example

```bash
python main.py example_evaluations.json example.yaml results.csv
```

### Command-Line Options

- `--no-table`: Skip printing the results table to console
- `--include-summary`: Include a summary row in the CSV output
- `--help`: Show help message

### Example with Options

```bash
python main.py example_evaluations.json example.yaml results.csv --include-summary
```

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

## Evaluations File Format

Provide evaluation results in JSON or YAML format:

### JSON Format

```json
{
  "sys_info_factual_1": {
    "type": "binary",
    "passes": true
  },
  "useful_1": {
    "type": "score",
    "score": 3
  },
  "tool_call_1": {
    "type": "binary",
    "passes": true
  }
}
```

### YAML Format

```yaml
sys_info_factual_1:
  type: binary
  passes: true

useful_1:
  type: score
  score: 3

tool_call_1:
  type: binary
  passes: true
```

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
├── rubric_kit/           # Main package
│   ├── __init__.py
│   ├── schema.py         # Pydantic models for validation
│   ├── validator.py      # YAML validation logic
│   ├── processor.py      # Score processing
│   ├── output.py         # CSV and table output
│   └── main.py           # CLI entry point
├── tests/                # Test suite
│   ├── test_schema.py
│   ├── test_validator.py
│   ├── test_processor.py
│   ├── test_output.py
│   └── test_main.py
├── main.py               # Command-line entry point
├── example.yaml          # Example rubric file
├── example_evaluations.json  # Example evaluations
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
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

See `example.yaml` and `example_evaluations.json` for complete working examples.

## License

See LICENSE file for details.
