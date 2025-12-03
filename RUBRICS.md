# Rubric Best Practices

Best practices for creating effective evaluation rubrics—manually or with AI assistance.

## Why Use Rubrics?

| Benefit | Description |
|---------|-------------|
| **Self-Contained** | Portable across prompt variations; understandable without external context |
| **Multidimensional** | Evaluates multiple aspects independently |
| **Scalable** | Works for human evaluators or LLM judge panels |

Think of a rubric as a **checklist** or **unit test suite** - each criterion is independent and defines what success looks like, regardless of how you get there.

---

## Prompt Design

Before creating a rubric, design a clear test prompt. For AI Agents, prompts may require tool usage.

### Essential Attributes

| Attribute | Description |
|-----------|-------------|
| **Feasible** | Required tools must be enabled and functional |
| **Unambiguous** | Single Ground Truth; avoid vague requests like "give me an overview" |
| **Conversational** | Fits naturally in a conversation |

### Examples

```yaml
# ❌ Bad: Vague, no Ground Truth
question: "Give me an overview of my system."

# ❌ Bad: Ambiguous metric
question: "Which process uses the most resources?"

# ✅ Good: Specific and measurable
question: "What is the total RAM in GB and CPU core count?"
```

---

## Writing Criteria

### Core Principles

1. **Self-Contained** — Include all info needed; don't reference external data
2. **Atomic** — One requirement per criterion
3. **Objective** — Binary pass/fail, no subjective language
4. **Use-Case Focused** — Only verify what the task requires

```yaml
# ❌ Bad: External reference + compound
criterion: The response must include the correct OS, RAM, and CPU count.

# ✅ Good: Self-contained + atomic
criterion: The response must state the operating system is Fedora Linux 42.
```

### Coverage

Rubrics should cover:
- **Process** — How the model approached the problem
- **Outcome** — Correctness of the final answer
- **Tool Usage** — Required tool calls (if applicable)

### Language

| Do | Don't |
|----|-------|
| "The model must..." | "e.g.", "such as" |
| "The response must state..." | "mention", "efficient", "effective" |
| "The model must not..." | "enough", "fluent", "unnecessary" |

### Special Cases

**Decimals** — Allow rounding: "CPU usage is 45.67% or an approximate rounded value"

**Large lists (5+ items)** — Check 4 representative items + total count

---

## Tool Call Evaluation

| Type | Logic | Scoring |
|------|-------|---------|
| **required** | All must be called (AND) | Pass = weight, Fail = 0 |
| **optional** | Bonus if called (OR) | Pass = bonus, Fail = 0 |
| **prohibited** | Must NOT be called | Violation = -weight |

### Example

```yaml
core_tools:
  weight: 3
  criterion: The model must gather system information.
  tool_calls:
    respect_order: false
    required:
      - get_system_info:
          min_calls: 1
          params:
            hostname: "{{hostname}}"
    optional:
      - get_network_info:
          min_calls: 1
    prohibited:
      - delete_files
```

**Alternative tools (OR logic):** Use `optional` when multiple tools can achieve the same result.

---

## Workflow

### 1. Define the Ideal Outcome

Ask: *What is the best possible answer for this use case?*

Write it down. This becomes your Ground Truth.

### 2. Create Prompt Variants

Test at least 3 prompts that should produce the same outcome:

```yaml
question: "Give me a summary of my system."
question: "What are my system specs?"
question: "Show me the hardware configuration."
```

### 3. Evaluate with Multiple Judges

Use at least 3 LLM judges per prompt. Look for:
- Criteria that are too strict/lenient
- Ambiguous wording causing disagreement
- Missing criteria

### 4. Refine

- Tighten ambiguous criteria with specific values
- Split compound criteria
- Adjust weights for critical criteria
- Remove redundant checks

```bash
rubric-kit evaluate --rubric rubric.yaml --from-chat-session session.txt -o results.yaml
rubric-kit refine rubric.yaml --feedback "Add specific value to criterion X" -o rubric_v2.yaml
```

### Scoring Guidelines

| Score | Interpretation |
|-------|----------------|
| ≥ 70% | Acceptable |
| 50-69% | Partial; review failures |
| < 50% | Significant gaps |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Vague prompts | Define specific metrics |
| Compound criteria | Split into atomic checks |
| External references | Include all data in criterion |
| Subjective language | Use objective, measurable terms |
| Wrong weights | Weight accuracy higher than optional criteria |

---

## Summary

A good rubric is a neutral checklist that verifies:
1. The model's reasoning process
2. Correct tool usage
3. Accuracy of the final answer

Test with multiple prompts, evaluate with multiple judges, refine based on results.
