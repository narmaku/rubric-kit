# Rubric Kit

Rubric framework. Create, refine, and apply evaluation rubrics powered by AI.

## Features

- **Rubric Generation** - Create rubrics from Q&A pairs or chat sessions
- **Multi-Judge Panel** - Multiple LLMs with consensus mechanisms (quorum, majority, unanimous)
- **Flexible Grading** - Binary (pass/fail) and score-based (0-3 scale) grading
- **Tool Call Validation** - Define required, optional, and prohibited tool calls
- **PDF Reports** - Comprehensive reports with charts and breakdowns
- **Export Formats** - YAML (source of truth), PDF, CSV, JSON
- **Self-Contained Outputs** - Re-run evaluations from previous results
- **OpenAI Compatible** - Works with any OpenAI-compatible endpoint

## Installation

Requires Python 3.10 or higher.

```bash
pip install rubric-kit
```

For development:

```bash
git clone https://github.com/your-org/rubric-kit
cd rubric-kit
pip install -e ".[dev]"
```

## Quick Start

```bash
export OPENAI_API_KEY="your-api-key"

# Generate a rubric from Q&A
rubric-kit generate qa_input.txt rubric.yaml

# Evaluate a chat session
rubric-kit evaluate --from-chat-session chat.txt --rubric-file rubric.yaml --output-file results.yaml

# Export to PDF
rubric-kit export results.yaml --format pdf --output report.pdf
```

## Commands

| Command | Description |
|---------|-------------|
| `generate` | Create a rubric from Q&A pair or chat session |
| `evaluate` | Evaluate content against a rubric (outputs YAML) |
| `refine` | Improve an existing rubric with AI feedback |
| `export` | Convert YAML to PDF, CSV, or JSON |
| `rerun` | Re-evaluate using settings from previous output |
| `arena` | Compare multiple contestants against same rubric |

Use `rubric-kit <command> --help` for detailed options.

## YAML Formats

See [`examples/`](examples/) for complete format examples:

- [`rubric.example.yaml`](examples/rubric.example.yaml) - Rubric with dimensions and criteria
- [`judge_panel.example.yaml`](examples/judge_panel.example.yaml) - Multi-judge configuration
- [`dimensions.example.yaml`](examples/dimensions.example.yaml) - Predefined dimensions
- [`arena.example.yaml`](examples/arena.example.yaml) - Arena competition spec

### Rubric Structure

```yaml
dimensions:
  - factual_correctness: Evaluates factual accuracy
    grading_type: binary

  - quality: Evaluates response quality
    grading_type: score
    scores:
      0: Poor
      1: Fair
      2: Good
      3: Excellent

criteria:
  accuracy_check:
    category: Output
    weight: 3
    dimension: factual_correctness
    criterion: Response must correctly state X.

  tool_usage:
    category: Tools
    weight: 2
    dimension: tool_use
    tool_calls:
      respect_order: false
      required:
        - get_info:
            min_calls: 1
```

### Judge Panel

```yaml
judge_panel:
  judges:
    - name: primary
      model: gpt-4
      api_key: ${OPENAI_API_KEY}
    - name: secondary
      model: gpt-4-turbo
      api_key: ${OPENAI_API_KEY}

  execution:
    mode: parallel  # sequential, parallel, batched

  consensus:
    mode: majority  # unanimous, majority, quorum
    on_no_consensus: fail  # fail, median, most_common
```

## Output

Evaluation always produces a **self-contained YAML** file:

```yaml
results:
  - criterion_name: accuracy_check
    result: pass
    score: 3
    reason: The response correctly stated X.

summary:
  total_score: 15
  max_score: 18
  percentage: 83.3

rubric: { ... }       # Full rubric for reference
judge_panel: { ... }  # Judge configuration used
metadata: { ... }     # Timestamp, source files
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=rubric_kit --cov-report=html

# Format code
black rubric_kit tests
```

See [`CLAUDE.md`](CLAUDE.md) for contribution guidelines.

## License

See [LICENSE](LICENSE) file.
