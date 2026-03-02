#!/usr/bin/env python3
"""Smoke test for rubric-kit CLI.

Exercises all subcommands and key options against dummy data to verify
that CLI functionality is intact after refactoring.

Dry-run tests require no LLM calls and no API key.
Live tests are skipped unless OPENAI_API_KEY is set.

Usage:
    python scripts/smoke_test.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

# ---- Styling ---------------------------------------------------------------

USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def green(t: str) -> str:
    return _c("0;32", t)


def red(t: str) -> str:
    return _c("0;31", t)


def cyan(t: str) -> str:
    return _c("0;36", t)


def yellow(t: str) -> str:
    return _c("0;33", t)


# ---- Test runner ------------------------------------------------------------

PASS = 0
FAIL = 0
SKIP = 0


def run_test(name: str, cmd: list[str], expect_fail: bool = False) -> bool:
    """Run a CLI command and report PASS/FAIL."""
    global PASS, FAIL
    print(f"  {cyan(f'{name:<58s}')}", end="", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if expect_fail:
        ok = result.returncode != 0
    else:
        ok = result.returncode == 0

    if ok:
        print(green("PASS"))
        PASS += 1
    else:
        print(red("FAIL"))
        output = (result.stderr or result.stdout or "").strip()
        last_lines = "\n".join(output.splitlines()[-5:])
        print(f"    Command: {' '.join(cmd)}")
        print(f"    Exit code: {result.returncode}")
        if last_lines:
            for line in last_lines.splitlines():
                print(f"      {line}")
        FAIL += 1

    return ok


def skip_test(name: str, reason: str) -> None:
    """Report a skipped test."""
    global SKIP
    print(f"  {cyan(f'{name:<58s}')}{yellow('SKIP')} ({reason})")
    SKIP += 1


def assert_file(path: str, name: str) -> None:
    """Assert a file exists and is non-empty."""
    global PASS, FAIL
    print(f"  {cyan(f'{name:<58s}')}", end="", flush=True)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        print(green("PASS"))
        PASS += 1
    else:
        print(red("FAIL") + f" (missing or empty: {path})")
        FAIL += 1


# ---- Fixtures ---------------------------------------------------------------

QNA_YAML = textwrap.dedent("""\
    question: What is the capital of France?
    answer: >
      The capital of France is Paris. It is located in the northern
      part of the country along the Seine river.
    context: Geography quiz for European capitals.
""")

CHAT_TXT = textwrap.dedent("""\
    User: What is the capital of France?

    Assistant: The capital of France is Paris. It is a major European city
    and a global center for art, fashion, gastronomy, and culture.
""")

RUBRIC_YAML = textwrap.dedent("""\
    dimensions:
      - factual_accuracy:
          description: "Evaluates whether the response is factually correct"
          grading_type: binary

      - completeness:
          description: "Evaluates whether the response covers all aspects"
          grading_type: score
          scores:
            1: "Minimal or missing information"
            2: "Partially complete"
            3: "Thorough and complete"

    criteria:
      - fact_check:
          category: Output
          weight: 3
          dimension: factual_accuracy
          criterion: "The response correctly identifies Paris as the capital of France."

      - detail_check:
          category: Output
          weight: from_scores
          dimension: completeness
          criterion: from_scores
""")

PANEL_YAML = textwrap.dedent("""\
    judge_panel:
      judges:
        - name: default
          model: gpt-4
      execution:
        mode: sequential
      consensus:
        mode: unanimous
""")

DIMENSIONS_YAML = textwrap.dedent("""\
    dimensions:
      - name: accuracy
        description: "Tests factual accuracy"
        grading_type: binary

      - name: clarity
        description: "Tests response clarity"
        grading_type: score
        scores:
          1: "Unclear"
          2: "Acceptable"
          3: "Very clear"
""")

VARIABLES_YAML = textwrap.dedent("""\
    country: France
    capital: Paris
""")

# Pre-built evaluation output (for export / rerun / arena tests)
EVAL_OUTPUT_YAML = textwrap.dedent("""\
    results:
      - criterion_name: fact_check
        criterion_text: "The response correctly identifies Paris as the capital of France."
        category: Output
        dimension: factual_accuracy
        result: pass
        score: 3
        max_score: 3
        reason: "Paris is correctly identified."
        consensus_reached: true
        consensus_count: 1
      - criterion_name: detail_check
        criterion_text: "Evaluates whether the response covers all aspects"
        category: Output
        dimension: completeness
        result: pass
        score: 3
        max_score: 3
        reason: "Good detail."
        consensus_reached: true
        consensus_count: 1
    summary:
      total_score: 6
      max_score: 6
      percentage: 100.0
    rubric:
      dimensions:
        - name: factual_accuracy
          description: "Evaluates whether the response is factually correct"
          grading_type: binary
        - name: completeness
          description: "Evaluates whether the response covers all aspects"
          grading_type: score
          scores:
            1: "Minimal or missing information"
            2: "Partially complete"
            3: "Thorough and complete"
      criteria:
        - name: fact_check
          category: Output
          weight: 3
          dimension: factual_accuracy
          criterion: "The response correctly identifies Paris as the capital of France."
        - name: detail_check
          category: Output
          weight: from_scores
          dimension: completeness
          criterion: from_scores
    judge_panel:
      judges:
        - name: default
          model: gpt-4
      execution:
        mode: sequential
      consensus:
        mode: unanimous
    input:
      type: chat_session
      source_file: chat.txt
      chat_session: |
        User: What is the capital of France?
        Assistant: The capital of France is Paris.
    metadata:
      timestamp: "2025-01-01T00:00:00"
""")


def write_fixtures(d: str) -> dict[str, str]:
    """Write all fixture files into directory d, return path map."""
    paths = {}
    fixtures = {
        "qna.yaml": QNA_YAML,
        "chat.txt": CHAT_TXT,
        "rubric.yaml": RUBRIC_YAML,
        "panel.yaml": PANEL_YAML,
        "dimensions.yaml": DIMENSIONS_YAML,
        "variables.yaml": VARIABLES_YAML,
        "eval_output.yaml": EVAL_OUTPUT_YAML,
    }
    for name, content in fixtures.items():
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(content)
        paths[name] = path
    # Second eval output for arena --from-outputs
    path2 = os.path.join(d, "eval_output_2.yaml")
    with open(path2, "w") as f:
        f.write(EVAL_OUTPUT_YAML.replace("fact_check", "fact_check_2").replace(
            "detail_check", "detail_check_2"
        ))
    paths["eval_output_2.yaml"] = path2
    return paths


# ---- Main -------------------------------------------------------------------

def main() -> int:
    tmpdir = tempfile.mkdtemp(prefix="rk_smoke_")

    try:
        f = write_fixtures(tmpdir)

        rk = [sys.executable, "-m", "rubric_kit"]

        print()
        print("=" * 60)
        print("  rubric-kit CLI Smoke Test")
        print("=" * 60)
        print(f"  Python:   {sys.executable}")
        print(f"  Temp dir: {tmpdir}")
        print()

        # ── 1. Help & version ──────────────────────────────────────
        print("  [1] Help & basic invocation")
        run_test("rubric-kit --help", [*rk, "--help"])
        run_test("rubric-kit evaluate --help", [*rk, "evaluate", "--help"])
        run_test("rubric-kit generate --help", [*rk, "generate", "--help"])
        run_test("rubric-kit refine --help", [*rk, "refine", "--help"])
        run_test("rubric-kit export --help", [*rk, "export", "--help"])
        run_test("rubric-kit rerun --help", [*rk, "rerun", "--help"])
        run_test("rubric-kit arena --help", [*rk, "arena", "--help"])
        run_test("no subcommand shows help (exit 2)",
                 rk, expect_fail=True)
        print()

        # ── 2. Evaluate: dry-run ───────────────────────────────────
        print("  [2] evaluate --dry-run (no LLM calls)")
        run_test("evaluate --dry-run --from-chat-session",
                 [*rk, "evaluate",
                  "--from-chat-session", f["chat.txt"],
                  "--rubric-file", f["rubric.yaml"],
                  "--output-file", os.path.join(tmpdir, "dry_eval.yaml"),
                  "--dry-run"])

        run_test("evaluate --dry-run --from-qna",
                 [*rk, "evaluate",
                  "--from-qna", f["qna.yaml"],
                  "--rubric-file", f["rubric.yaml"],
                  "--output-file", os.path.join(tmpdir, "dry_eval2.yaml"),
                  "--dry-run"])

        run_test("evaluate --dry-run with --judge-panel-config",
                 [*rk, "evaluate",
                  "--from-chat-session", f["chat.txt"],
                  "--rubric-file", f["rubric.yaml"],
                  "--output-file", os.path.join(tmpdir, "dry_eval3.yaml"),
                  "--judge-panel-config", f["panel.yaml"],
                  "--dry-run"])
        print()

        # ── 3. Generate: dry-run ───────────────────────────────────
        print("  [3] generate --dry-run (no LLM calls)")
        run_test("generate --dry-run --from-qna",
                 [*rk, "generate",
                  "--from-qna", f["qna.yaml"],
                  "--output-file", os.path.join(tmpdir, "dry_gen.yaml"),
                  "--dry-run"])

        run_test("generate --dry-run --from-chat-session",
                 [*rk, "generate",
                  "--from-chat-session", f["chat.txt"],
                  "--output-file", os.path.join(tmpdir, "dry_gen2.yaml"),
                  "--dry-run"])

        run_test("generate --dry-run with --dimensions-file",
                 [*rk, "generate",
                  "--from-qna", f["qna.yaml"],
                  "--output-file", os.path.join(tmpdir, "dry_gen3.yaml"),
                  "--dimensions-file", f["dimensions.yaml"],
                  "--dry-run"])

        run_test("generate --dry-run with --num-dimensions --num-criteria",
                 [*rk, "generate",
                  "--from-qna", f["qna.yaml"],
                  "--output-file", os.path.join(tmpdir, "dry_gen4.yaml"),
                  "--num-dimensions", "3",
                  "--num-criteria", "5",
                  "--dry-run"])
        print()

        # ── 4. Export (from pre-built eval output) ─────────────────
        print("  [4] export (CSV / JSON / PDF)")
        csv_out = os.path.join(tmpdir, "export.csv")
        run_test("export --format csv",
                 [*rk, "export", f["eval_output.yaml"],
                  "--format", "csv", "--output", csv_out])
        assert_file(csv_out, "  -> CSV file created")

        json_out = os.path.join(tmpdir, "export.json")
        run_test("export --format json",
                 [*rk, "export", f["eval_output.yaml"],
                  "--format", "json", "--output", json_out])
        assert_file(json_out, "  -> JSON file created")

        pdf_out = os.path.join(tmpdir, "export.pdf")
        run_test("export --format pdf",
                 [*rk, "export", f["eval_output.yaml"],
                  "--format", "pdf", "--output", pdf_out])
        assert_file(pdf_out, "  -> PDF file created")
        print()

        # ── 5. Rerun (from pre-built eval output) ─────────────────
        print("  [5] rerun (uses embedded settings)")
        has_key = bool(os.environ.get("OPENAI_API_KEY"))

        if has_key:
            rerun_out = os.path.join(tmpdir, "rerun.yaml")
            run_test("rerun with embedded input",
                     [*rk, "rerun", f["eval_output.yaml"],
                      "--output-file", rerun_out])

            rerun_new_out = os.path.join(tmpdir, "rerun_new.yaml")
            run_test("rerun with new chat session",
                     [*rk, "rerun", f["eval_output.yaml"],
                      "--from-chat-session", f["chat.txt"],
                      "--output-file", rerun_new_out])

            rerun_qna_out = os.path.join(tmpdir, "rerun_qna.yaml")
            run_test("rerun with new QnA input",
                     [*rk, "rerun", f["eval_output.yaml"],
                      "--from-qna", f["qna.yaml"],
                      "--output-file", rerun_qna_out])

            run_test("rerun with --no-table",
                     [*rk, "rerun", f["eval_output.yaml"],
                      "--output-file", os.path.join(tmpdir, "rerun_nt.yaml"),
                      "--no-table"])
        else:
            skip_test("rerun with embedded input", "OPENAI_API_KEY not set")
            skip_test("rerun with new chat session", "OPENAI_API_KEY not set")
            skip_test("rerun with new QnA input", "OPENAI_API_KEY not set")
            skip_test("rerun with --no-table", "OPENAI_API_KEY not set")
        print()

        # ── 6. Arena: --from-outputs (no LLM calls) ───────────────
        print("  [6] arena --from-outputs (no LLM calls)")
        arena_out = os.path.join(tmpdir, "arena.yaml")
        run_test("arena --from-outputs (2 files)",
                 [*rk, "arena",
                  "--from-outputs", f["eval_output.yaml"], f["eval_output_2.yaml"],
                  "--output-file", arena_out])
        assert_file(arena_out, "  -> arena YAML created")

        arena_pdf = os.path.join(tmpdir, "arena.pdf")
        run_test("arena --from-outputs with --report",
                 [*rk, "arena",
                  "--from-outputs", f["eval_output.yaml"], f["eval_output_2.yaml"],
                  "--output-file", os.path.join(tmpdir, "arena2.yaml"),
                  "--report", arena_pdf,
                  "--report-title", "Smoke Test Arena"])
        assert_file(arena_pdf, "  -> arena PDF created")

        run_test("arena --from-outputs with --no-table",
                 [*rk, "arena",
                  "--from-outputs", f["eval_output.yaml"], f["eval_output_2.yaml"],
                  "--output-file", os.path.join(tmpdir, "arena3.yaml"),
                  "--no-table"])
        print()

        # ── 7. Python API imports ──────────────────────────────────
        print("  [7] Python API imports")
        run_test("import rubric_kit",
                 [sys.executable, "-c", "import rubric_kit"])
        run_test("from rubric_kit import evaluate",
                 [sys.executable, "-c", "from rubric_kit import evaluate"])
        run_test("from rubric_kit import generate, refine",
                 [sys.executable, "-c", "from rubric_kit import generate, refine"])
        run_test("from rubric_kit import Rubric, Dimension, Criterion",
                 [sys.executable, "-c",
                  "from rubric_kit import Rubric, Dimension, Criterion"])
        run_test("from rubric_kit import EvaluationResult, ScoreSummary",
                 [sys.executable, "-c",
                  "from rubric_kit import EvaluationResult, ScoreSummary"])
        run_test("rubric_kit.__version__ is set",
                 [sys.executable, "-c",
                  "import rubric_kit; assert rubric_kit.__version__"])
        print()

        # ── 8. Error handling ──────────────────────────────────────
        print("  [8] Error handling (expected failures)")
        run_test("evaluate: missing --rubric-file (exit != 0)",
                 [*rk, "evaluate",
                  "--from-chat-session", f["chat.txt"],
                  "--output-file", "/dev/null"],
                 expect_fail=True)

        run_test("evaluate: nonexistent rubric file (exit != 0)",
                 [*rk, "evaluate",
                  "--from-chat-session", f["chat.txt"],
                  "--rubric-file", "/nonexistent/rubric.yaml",
                  "--output-file", "/dev/null"],
                 expect_fail=True)

        run_test("generate: missing input (exit != 0)",
                 [*rk, "generate",
                  "--output-file", "/dev/null"],
                 expect_fail=True)

        run_test("export: invalid format (exit != 0)",
                 [*rk, "export", f["eval_output.yaml"],
                  "--format", "xml", "--output", "/dev/null"],
                 expect_fail=True)
        print()

        # ── Summary ───────────────────────────────────────────────
        total = PASS + FAIL + SKIP
        print("=" * 60)
        print(f"  Results: {green(f'{PASS} passed')}, ", end="")
        if FAIL:
            print(f"{red(f'{FAIL} failed')}, ", end="")
        else:
            print(f"0 failed, ", end="")
        print(f"{SKIP} skipped  (total: {total})")
        print("=" * 60)
        print()

        return 1 if FAIL > 0 else 0

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
