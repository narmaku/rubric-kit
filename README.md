# Rubric Kit

Rubric framework. Create, refine, and apply evaluation rubrics powered by AI.

## Features

- **Rubric Generation** - Create rubrics from Q&A pairs or chat sessions
- **Multi-Judge Panel** - Multiple LLMs with consensus mechanisms (quorum, majority, unanimous)
- **Multi-Provider Support** - OpenAI, Google Vertex AI, IBM WatsonX, Anthropic, Ollama, and 100+ providers via LiteLLM
- **Flexible Grading** - Binary (pass/fail) and score-based (0-3 scale) grading
- **Tool Call Validation** - Define required, optional, and prohibited tool calls
- **PDF Reports** - Comprehensive reports with charts and breakdowns
- **Export Formats** - YAML (source of truth), PDF, CSV, JSON
- **Self-Contained Outputs** - Re-run evaluations from previous results
- **Cost Estimation** - Dry-run mode to estimate costs before running evaluations
- **Metrics Tracking** - Token usage, latency, and cost tracking per LLM call

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

## LLM Provider Setup

Rubric Kit uses [LiteLLM](https://docs.litellm.ai/) to support 100+ LLM providers. API keys are configured via environment variables (never in config files or CLI arguments).

### Some Supported Providers (most popular)

| Provider | Model Format | Environment Variables |
|----------|--------------|----------------------|
| **OpenAI** | `gpt-4`, `gpt-4o` | `OPENAI_API_KEY` |
| **Google AI Studio** | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| **Google Vertex AI** | `vertex_ai/gemini-2.5-flash` | `gcloud auth` or `GOOGLE_APPLICATION_CREDENTIALS` |
| **IBM WatsonX** | `watsonx/meta-llama/llama-3-8b-instruct` | `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` |
| **Anthropic** | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| **Ollama (local)** | `ollama/llama3.1` | None (uses `localhost:11434`) |
| **Ollama (remote)** | `ollama/granite4` | `OLLAMA_API_BASE` |
| **Azure OpenAI** | `azure/gpt-4` | `AZURE_API_KEY`, `AZURE_API_BASE` |

> 📚 **Full documentation**: See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for complete list of supported providers, model formats, and required environment variables.

### CLI Examples

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
rubric-kit generate --from-qna qna.yaml --output-file rubric.yaml --model gpt-4o

# Google AI Studio (Gemini)
export GEMINI_API_KEY="..."
rubric-kit generate --from-qna qna.yaml --output-file rubric.yaml --model gemini/gemini-2.5-flash

# Google Vertex AI
gcloud auth application-default login
rubric-kit generate --from-qna qna.yaml --output-file rubric.yaml --model vertex_ai/gemini-2.5-flash

# IBM WatsonX
export WATSONX_APIKEY="..."
export WATSONX_PROJECT_ID="..."
rubric-kit generate --from-qna qna.yaml --output-file rubric.yaml --model watsonx/meta-llama/llama-3-8b-instruct

# Local Ollama
rubric-kit generate --from-qna qna.yaml --output-file rubric.yaml --model ollama/llama3
```

### OpenAI-Compatible Endpoints

For custom OpenAI-compatible endpoints (vLLM, LocalAI, etc.), use `base_url`:

```yaml
judges:
  - name: custom-endpoint
    model: mistral-7b  # Model name as expected by your endpoint
    base_url: http://your-endpoint:8000/v1
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

### Common Options

| Flag | Commands | Description |
|------|----------|-------------|
| `--dry-run` | evaluate, generate | Estimate costs without making LLM calls |
| `--no-metrics` | evaluate, generate, refine | Disable metrics collection in output |
| `--include-call-log` | evaluate | Include detailed per-call metrics in output |

## YAML Formats

See [`examples/`](examples/) for complete format examples:

- [`rubric.example.yaml`](examples/rubric.example.yaml) - Rubric with dimensions and criteria
- [`judge_panel.example.yaml`](examples/judge_panel.example.yaml) - Multi-judge configuration
- [`dimensions.example.yaml`](examples/dimensions.example.yaml) - Predefined dimensions
- [`arena.example.yaml`](examples/arena.example.yaml) - Arena competition spec

### Rubric Structure

Below you will find a very basic rubric to understand how it is composed.

TODO: For Rubrics best practices, please read the [RUBRICS](RUBRICS.md) file.

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
    - name: ChatGPT-4o
      model: gpt-4o
    - name: Gemini-2.5-Flash
      model: vertex_ai/gemini-2.5-flash
    - name: Claude-4.5-Sonnet
      model: anthropic/claude-4-5-sonnet

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
input: { ... }        # Input content (Q&A or chat session)
metadata:
  timestamp: 2025-01-20T10:30:00
  metrics:            # LLM usage metrics (unless --no-metrics)
    summary:
      total_calls: 5
      prompt_tokens: 2500
      completion_tokens: 800
      estimated_cost_usd: 0.0425
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

## License

See [LICENSE](LICENSE) file.
