"""Tests for Arena mode functionality."""

import os

import pytest
import yaml

from rubric_kit.schema import ArenaContestant


class TestLoadArenaSpec:
    """Test arena spec loading and validation."""

    @pytest.fixture
    def sample_arena_yaml(self, tmp_path):
        """Create a sample arena spec YAML file."""
        # Create rubric file
        rubric_content = """
dimensions:
  - test_dim: Test dimension
    grading_type: binary

criteria:
  test_crit:
    category: Output
    weight: 3
    dimension: test_dim
    criterion: Test criterion
"""
        rubric_path = tmp_path / "rubric.yaml"
        rubric_path.write_text(rubric_content)

        # Create judge panel file
        judge_panel_content = """
judge_panel:
  judges:
    - name: test_judge
      model: gpt-4
      api_key: null
  execution:
    mode: sequential
  consensus:
    mode: unanimous
"""
        judge_panel_path = tmp_path / "judges.yaml"
        judge_panel_path.write_text(judge_panel_content)

        # Create session files
        session1_path = tmp_path / "session1.txt"
        session1_path.write_text("User: Test\nAssistant: Response")

        session2_path = tmp_path / "session2.txt"
        session2_path.write_text("User: Another test\nAssistant: Another response")

        # Create arena spec
        arena_content = f"""
arena:
  name: "Test Arena"
  description: "Testing arena functionality"
  rubric_file: "{rubric_path.name}"
  judges_panel_file: "{judge_panel_path.name}"
  contestants:
    - id: model_a
      name: "Model A"
      input_file: "{session1_path.name}"
    - id: model_b
      name: "Model B"
      input_file: "{session2_path.name}"
      variables:
        test_var: "test_value"
"""
        arena_path = tmp_path / "arena.yaml"
        arena_path.write_text(arena_content)

        return arena_path

    def test_load_arena_spec_success(self, sample_arena_yaml):
        """Test loading a valid arena spec file."""
        from rubric_kit.arena import load_arena_spec

        spec = load_arena_spec(str(sample_arena_yaml))

        assert spec.name == "Test Arena"
        assert spec.description == "Testing arena functionality"
        assert len(spec.contestants) == 2
        assert spec.contestants[0].id == "model_a"
        assert spec.contestants[1].id == "model_b"
        assert spec.contestants[1].variables == {"test_var": "test_value"}

    def test_load_arena_spec_missing_file(self):
        """Test loading non-existent arena spec file."""
        from rubric_kit.arena import load_arena_spec

        with pytest.raises(FileNotFoundError):
            load_arena_spec("nonexistent_arena.yaml")

    def test_load_arena_spec_missing_arena_key(self, tmp_path):
        """Test loading arena spec without 'arena' key."""
        from rubric_kit.arena import load_arena_spec

        bad_spec = tmp_path / "bad_arena.yaml"
        bad_spec.write_text("rubric_file: test.yaml")

        with pytest.raises(ValueError, match="must have an 'arena' key"):
            load_arena_spec(str(bad_spec))


class TestContestantVariables:
    """Test contestant-specific variable loading."""

    def test_load_inline_variables(self, tmp_path):
        """Test loading inline variables from contestant."""
        from rubric_kit.arena import load_contestant_variables

        contestant = ArenaContestant(
            id="test",
            name="Test",
            input_file="session.txt",
            variables={"var1": "value1", "var2": "value2"},
        )

        variables = load_contestant_variables(contestant, str(tmp_path))

        assert variables == {"var1": "value1", "var2": "value2"}

    def test_load_external_variables_file(self, tmp_path):
        """Test loading variables from external file."""
        from rubric_kit.arena import load_contestant_variables

        # Create variables file
        vars_file = tmp_path / "vars.yaml"
        vars_file.write_text("var1: external_value1\nvar2: external_value2")

        contestant = ArenaContestant(
            id="test", name="Test", input_file="session.txt", variables_file="vars.yaml"
        )

        variables = load_contestant_variables(contestant, str(tmp_path))

        assert variables == {"var1": "external_value1", "var2": "external_value2"}

    def test_load_variables_with_nested_key(self, tmp_path):
        """Test loading variables from file with 'variables' key."""
        from rubric_kit.arena import load_contestant_variables

        # Create variables file with nested structure
        vars_file = tmp_path / "vars.yaml"
        vars_file.write_text("variables:\n  var1: nested_value")

        contestant = ArenaContestant(
            id="test", name="Test", input_file="session.txt", variables_file="vars.yaml"
        )

        variables = load_contestant_variables(contestant, str(tmp_path))

        assert variables == {"var1": "nested_value"}

    def test_inline_variables_take_priority(self, tmp_path):
        """Test that inline variables are used even when variables_file is specified."""
        from rubric_kit.arena import load_contestant_variables

        # Create variables file
        vars_file = tmp_path / "vars.yaml"
        vars_file.write_text("var1: file_value")

        contestant = ArenaContestant(
            id="test",
            name="Test",
            input_file="session.txt",
            variables={"var1": "inline_value"},
            variables_file="vars.yaml",  # This should be ignored
        )

        variables = load_contestant_variables(contestant, str(tmp_path))

        # Inline takes priority
        assert variables == {"var1": "inline_value"}

    def test_no_variables_returns_none(self, tmp_path):
        """Test that contestant without variables returns None."""
        from rubric_kit.arena import load_contestant_variables

        contestant = ArenaContestant(id="test", name="Test", input_file="session.txt")

        variables = load_contestant_variables(contestant, str(tmp_path))

        assert variables is None


class TestApplyVariablesToRubric:
    """Test variable substitution when applying contestant variables to rubric."""

    def test_apply_variables_substitutes_criterion_text(self):
        """Test that variables are substituted in criterion text."""
        from rubric_kit.arena import apply_variables_to_rubric
        from rubric_kit.schema import Criterion, Dimension, Rubric

        base_rubric = Rubric(
            dimensions=[
                Dimension(name="test_dim", description="Test dimension", grading_type="binary")
            ],
            criteria=[
                Criterion(
                    name="test_crit",
                    category="Test",
                    weight=3,
                    dimension="test_dim",
                    criterion="The RAM should be {{ram_total}} and CPU count {{cpu_count}}.",
                )
            ],
        )

        variables = {"ram_total": "64 GB", "cpu_count": "8"}

        result = apply_variables_to_rubric(base_rubric, variables)

        assert result.criteria[0].criterion == "The RAM should be 64 GB and CPU count 8."
        assert result.variables == variables

    def test_apply_variables_substitutes_dimension_description(self):
        """Test that variables are substituted in dimension descriptions."""
        from rubric_kit.arena import apply_variables_to_rubric
        from rubric_kit.schema import Criterion, Dimension, Rubric

        base_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="test_dim",
                    description="Evaluates if OS is {{os_distro}}",
                    grading_type="binary",
                )
            ],
            criteria=[
                Criterion(
                    name="test_crit",
                    category="Test",
                    weight=3,
                    dimension="test_dim",
                    criterion="Test",
                )
            ],
        )

        variables = {"os_distro": "Fedora Linux 42"}

        result = apply_variables_to_rubric(base_rubric, variables)

        assert result.dimensions[0].description == "Evaluates if OS is Fedora Linux 42"

    def test_apply_variables_substitutes_tool_params(self):
        """Test that variables are substituted in tool call params."""
        from rubric_kit.arena import apply_variables_to_rubric
        from rubric_kit.schema import Criterion, Dimension, Rubric, ToolCalls, ToolSpec

        base_rubric = Rubric(
            dimensions=[
                Dimension(name="tool_dim", description="Tool usage", grading_type="binary")
            ],
            criteria=[
                Criterion(
                    name="tool_crit",
                    category="Tools",
                    weight=3,
                    dimension="tool_dim",
                    tool_calls=ToolCalls(
                        respect_order=False,
                        required=[
                            ToolSpec(
                                name="get_system_info",
                                min_calls=1,
                                params={"hostname": "{{hostname}}", "user": "{{username}}"},
                            )
                        ],
                    ),
                )
            ],
        )

        variables = {"hostname": "server.example.com", "username": "admin"}

        result = apply_variables_to_rubric(base_rubric, variables)

        tool_params = result.criteria[0].tool_calls.required[0].params
        assert tool_params["hostname"] == "server.example.com"
        assert tool_params["user"] == "admin"


class TestArenaOutputStructure:
    """Test the arena output YAML structure."""

    @pytest.fixture
    def sample_arena_output(self):
        """Create a sample arena output structure."""
        return {
            "mode": "arena",
            "arena_name": "Test Arena",
            "arena_description": "Testing",
            "contestants": {
                "model_a": {
                    "name": "Model A",
                    "results": [{"criterion_name": "crit1", "score": 3, "max_score": 3}],
                    "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
                },
                "model_b": {
                    "name": "Model B",
                    "results": [{"criterion_name": "crit1", "score": 2, "max_score": 3}],
                    "summary": {"total_score": 2, "max_score": 3, "percentage": 66.7},
                },
            },
            "rankings": [
                {"rank": 1, "id": "model_a", "name": "Model A", "percentage": 100.0},
                {"rank": 2, "id": "model_b", "name": "Model B", "percentage": 66.7},
            ],
            "rubric": {"dimensions": [], "criteria": []},
            "judge_panel": {"judges": [], "execution": {}, "consensus": {}},
            "metadata": {"timestamp": "2025-01-01T00:00:00"},
        }

    def test_arena_output_has_required_keys(self, sample_arena_output):
        """Test that arena output has all required keys."""
        assert "mode" in sample_arena_output
        assert sample_arena_output["mode"] == "arena"
        assert "contestants" in sample_arena_output
        assert "rankings" in sample_arena_output
        assert "rubric" in sample_arena_output
        assert "judge_panel" in sample_arena_output
        assert "metadata" in sample_arena_output

    def test_rankings_are_sorted_by_percentage(self, sample_arena_output):
        """Test that rankings are sorted by percentage descending."""
        rankings = sample_arena_output["rankings"]

        percentages = [r["percentage"] for r in rankings]
        assert percentages == sorted(percentages, reverse=True)

    def test_each_contestant_has_summary(self, sample_arena_output):
        """Test that each contestant has results and summary."""
        for _contestant_id, cdata in sample_arena_output["contestants"].items():
            assert "results" in cdata
            assert "summary" in cdata
            assert "total_score" in cdata["summary"]
            assert "max_score" in cdata["summary"]
            assert "percentage" in cdata["summary"]


class TestGenerateContestantId:
    """Test contestant ID generation helper."""

    def test_generates_sequential_id_with_basename(self):
        """Test that ID follows contestant-NNN-basename format."""
        from rubric_kit.arena import _generate_contestant_id

        result = _generate_contestant_id(0, "/some/path/result_coding_001.yaml")

        assert result == "contestant-001-result-coding-001"

    def test_sequential_numbering(self):
        """Test that index produces correct zero-padded number."""
        from rubric_kit.arena import _generate_contestant_id

        assert _generate_contestant_id(0, "file_a.yaml").startswith("contestant-001-")
        assert _generate_contestant_id(1, "file_b.yaml").startswith("contestant-002-")
        assert _generate_contestant_id(9, "file_c.yaml").startswith("contestant-010-")

    def test_strips_output_prefix(self):
        """Test that 'output_' prefix is stripped from basename."""
        from rubric_kit.arena import _generate_contestant_id

        result = _generate_contestant_id(0, "output_model_a.yaml")

        assert result == "contestant-001-model-a"

    def test_same_basename_different_index_produces_unique_ids(self):
        """Test that identical filenames at different indices produce unique IDs."""
        from rubric_kit.arena import _generate_contestant_id

        id1 = _generate_contestant_id(0, "/dir_a/result_coding_001.yaml")
        id2 = _generate_contestant_id(1, "/dir_b/result_coding_001.yaml")

        assert id1 != id2
        assert id1 == "contestant-001-result-coding-001"
        assert id2 == "contestant-002-result-coding-001"


class TestCombineOutputsToArena:
    """Test combining multiple output files into arena format."""

    @pytest.fixture
    def sample_output_files(self, tmp_path):
        """Create sample output YAML files."""
        output1 = {
            "results": [
                {
                    "criterion_name": "crit1",
                    "dimension": "accuracy",
                    "score": 3,
                    "max_score": 3,
                    "result": "pass",
                    "reason": "Good",
                }
            ],
            "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
            "rubric": {"dimensions": [{"name": "accuracy"}], "criteria": [{"name": "crit1"}]},
            "judge_panel": {"judges": [{"name": "test"}]},
            "input": {"type": "chat_session", "source_file": "session1.txt"},
            "metadata": {"timestamp": "2025-01-01T00:00:00", "report_title": "Model A Evaluation"},
        }

        output2 = {
            "results": [
                {
                    "criterion_name": "crit1",
                    "dimension": "accuracy",
                    "score": 2,
                    "max_score": 3,
                    "result": "fail",
                    "reason": "Partial",
                }
            ],
            "summary": {"total_score": 2, "max_score": 3, "percentage": 66.7},
            "rubric": {"dimensions": [{"name": "accuracy"}], "criteria": [{"name": "crit1"}]},
            "judge_panel": {"judges": [{"name": "test"}]},
            "input": {"type": "chat_session", "source_file": "session2.txt"},
            "metadata": {"timestamp": "2025-01-01T01:00:00", "report_title": "Model B Evaluation"},
        }

        file1 = tmp_path / "output_model_a.yaml"
        file2 = tmp_path / "output_model_b.yaml"

        with open(file1, "w") as f:
            yaml.dump(output1, f)
        with open(file2, "w") as f:
            yaml.dump(output2, f)

        return [str(file1), str(file2)]

    def test_combine_outputs_creates_arena_format(self, sample_output_files):
        """Test that combining outputs creates valid arena format."""
        from rubric_kit.arena import combine_outputs_to_arena

        result = combine_outputs_to_arena(sample_output_files, "Test Arena")

        assert result["mode"] == "arena"
        assert result["arena_name"] == "Test Arena"
        assert len(result["contestants"]) == 2
        assert len(result["rankings"]) == 2
        assert result["rubric"] is not None
        assert result["judge_panel"] is not None

    def test_combine_outputs_generates_rankings(self, sample_output_files):
        """Test that combined outputs have correct rankings."""
        from rubric_kit.arena import combine_outputs_to_arena

        result = combine_outputs_to_arena(sample_output_files)

        rankings = result["rankings"]
        assert rankings[0]["rank"] == 1
        assert rankings[0]["percentage"] == 100.0  # Model A is first
        assert rankings[1]["rank"] == 2
        assert rankings[1]["percentage"] == 66.7  # Model B is second

    def test_combine_outputs_preserves_metadata(self, sample_output_files):
        """Test that original metadata is preserved in contestant data."""
        from rubric_kit.arena import combine_outputs_to_arena

        result = combine_outputs_to_arena(sample_output_files)

        for _contestant_id, cdata in result["contestants"].items():
            assert "metadata" in cdata
            assert "source_file" in cdata["metadata"]
            assert "original_timestamp" in cdata["metadata"]

    def test_combine_outputs_rejects_arena_files(self, tmp_path):
        """Test that combining fails if an arena file is included."""
        from rubric_kit.arena import combine_outputs_to_arena

        arena_file = tmp_path / "arena_result.yaml"
        with open(arena_file, "w") as f:
            yaml.dump({"mode": "arena", "contestants": {}}, f)

        with pytest.raises(ValueError, match="already an arena result"):
            combine_outputs_to_arena([str(arena_file)])

    def test_combine_outputs_rejects_missing_results(self, tmp_path):
        """Test that combining fails if results are missing."""
        from rubric_kit.arena import combine_outputs_to_arena

        bad_file = tmp_path / "bad.yaml"
        with open(bad_file, "w") as f:
            yaml.dump({"summary": {}}, f)

        with pytest.raises(ValueError, match="missing 'results' section"):
            combine_outputs_to_arena([str(bad_file)])

    def test_combine_outputs_with_duplicate_basenames(self, tmp_path):
        """Test that files with identical basenames produce unique contestants."""
        from rubric_kit.arena import combine_outputs_to_arena

        output_data = {
            "results": [{"criterion_name": "c1", "dimension": "d1", "score": 3, "max_score": 3}],
            "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
            "rubric": {"dimensions": [], "criteria": []},
            "judge_panel": {"judges": []},
            "metadata": {"report_title": "Same Name"},
        }

        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        dir_a.mkdir()
        dir_b.mkdir()

        file_a = dir_a / "result_coding_001.yaml"
        file_b = dir_b / "result_coding_001.yaml"

        with open(file_a, "w") as f:
            yaml.dump(output_data, f)
        with open(file_b, "w") as f:
            yaml.dump(output_data, f)

        result = combine_outputs_to_arena([str(file_a), str(file_b)])

        assert len(result["contestants"]) == 2
        assert len(result["rankings"]) == 2

    def test_combine_outputs_ids_use_sequential_format(self, sample_output_files):
        """Test that contestant IDs follow the contestant-NNN-basename format."""
        from rubric_kit.arena import combine_outputs_to_arena

        result = combine_outputs_to_arena(sample_output_files)
        contestant_ids = list(result["contestants"].keys())

        assert contestant_ids[0] == "contestant-001-model-a"
        assert contestant_ids[1] == "contestant-002-model-b"


class TestDuplicateBasenameHandling:
    """Regression tests for arena bug when output files share the same basename.

    Before the fix (commit f31959a), contestant IDs were derived solely from
    the filename, so files named the same (from different directories) would
    silently overwrite each other in the results dict, losing data.

    The fix uses ``_generate_contestant_id()`` which prefixes a sequential
    index (``contestant-NNN-...``) to guarantee uniqueness.
    """

    def _make_output(self, score: int, max_score: int = 3, title: str = "Model") -> dict:
        """Create a minimal evaluation output dict."""
        return {
            "results": [
                {
                    "criterion_name": "c1",
                    "dimension": "d1",
                    "score": score,
                    "max_score": max_score,
                    "result": "pass" if score == max_score else "fail",
                    "reason": f"Score {score}",
                }
            ],
            "summary": {
                "total_score": score,
                "max_score": max_score,
                "percentage": round(score / max_score * 100, 1) if max_score else 0.0,
            },
            "rubric": {"dimensions": [], "criteria": []},
            "judge_panel": {"judges": []},
            "input": {"type": "chat_session", "source_file": "input.txt"},
            "metadata": {"report_title": title},
        }

    def test_three_files_same_basename_all_unique_ids(self, tmp_path):
        """Three files with identical basenames produce three unique contestant IDs."""
        from rubric_kit.arena import combine_outputs_to_arena

        dirs = [tmp_path / f"run_{i}" for i in range(3)]
        files = []
        for i, d in enumerate(dirs):
            d.mkdir()
            path = d / "output.yaml"
            with open(path, "w") as f:
                yaml.dump(self._make_output(score=i + 1, title=f"Run {i + 1}"), f)
            files.append(str(path))

        result = combine_outputs_to_arena(files)

        ids = list(result["contestants"].keys())
        assert len(ids) == 3
        assert len(set(ids)) == 3, f"IDs are not unique: {ids}"

    def test_duplicate_basenames_preserve_individual_scores(self, tmp_path):
        """Each contestant retains its own score even when basenames collide."""
        from rubric_kit.arena import combine_outputs_to_arena

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        file_a = dir_a / "result.yaml"
        file_b = dir_b / "result.yaml"
        with open(file_a, "w") as f:
            yaml.dump(self._make_output(score=3, title="Winner"), f)
        with open(file_b, "w") as f:
            yaml.dump(self._make_output(score=1, title="Loser"), f)

        result = combine_outputs_to_arena([str(file_a), str(file_b)])

        scores = [c["summary"]["total_score"] for c in result["contestants"].values()]
        assert sorted(scores) == [1, 3], "Scores must not be overwritten"

    def test_duplicate_basenames_rankings_reflect_all_contestants(self, tmp_path):
        """Rankings include every contestant even when basenames are identical."""
        from rubric_kit.arena import combine_outputs_to_arena

        files = []
        for i in range(4):
            d = tmp_path / f"dir_{i}"
            d.mkdir()
            path = d / "eval.yaml"
            with open(path, "w") as f:
                yaml.dump(self._make_output(score=i, max_score=3, title=f"M{i}"), f)
            files.append(str(path))

        result = combine_outputs_to_arena(files)

        assert len(result["rankings"]) == 4
        # Rankings should be sorted by percentage descending
        percentages = [r["percentage"] for r in result["rankings"]]
        assert percentages == sorted(percentages, reverse=True)

    def test_duplicate_basenames_metadata_not_mixed(self, tmp_path):
        """Each contestant's metadata and input data remains distinct."""
        from rubric_kit.arena import combine_outputs_to_arena

        dir_a = tmp_path / "site_a"
        dir_b = tmp_path / "site_b"
        dir_a.mkdir()
        dir_b.mkdir()

        data_a = self._make_output(score=3, title="Site A Model")
        data_a["input"] = {"type": "chat_session", "source_file": "chat_a.txt"}
        data_b = self._make_output(score=1, title="Site B Model")
        data_b["input"] = {"type": "qna", "source_file": "qna_b.yaml"}

        file_a = dir_a / "output.yaml"
        file_b = dir_b / "output.yaml"
        with open(file_a, "w") as f:
            yaml.dump(data_a, f)
        with open(file_b, "w") as f:
            yaml.dump(data_b, f)

        result = combine_outputs_to_arena([str(file_a), str(file_b)])

        contestants = list(result["contestants"].values())
        input_types = {c["input"]["type"] for c in contestants}
        assert input_types == {"chat_session", "qna"}, "Input data was mixed between contestants"

    def test_id_generation_is_deterministic_on_order(self, tmp_path):
        """Same file list in same order always produces the same IDs."""
        from rubric_kit.arena import combine_outputs_to_arena

        dirs = [tmp_path / f"d{i}" for i in range(3)]
        files = []
        for i, d in enumerate(dirs):
            d.mkdir()
            path = d / "result.yaml"
            with open(path, "w") as f:
                yaml.dump(self._make_output(score=i + 1), f)
            files.append(str(path))

        result1 = combine_outputs_to_arena(files)
        result2 = combine_outputs_to_arena(files)

        ids1 = list(result1["contestants"].keys())
        ids2 = list(result2["contestants"].keys())
        assert ids1 == ids2

    def test_many_duplicates_all_tracked(self, tmp_path):
        """Stress test: 20 files with the same basename all appear as contestants."""
        from rubric_kit.arena import combine_outputs_to_arena

        files = []
        for i in range(20):
            d = tmp_path / f"run_{i:03d}"
            d.mkdir()
            path = d / "output.yaml"
            with open(path, "w") as f:
                yaml.dump(self._make_output(score=(i % 3) + 1, title=f"Run {i}"), f)
            files.append(str(path))

        result = combine_outputs_to_arena(files)

        assert len(result["contestants"]) == 20
        assert len(result["rankings"]) == 20
        ids = list(result["contestants"].keys())
        assert len(set(ids)) == 20, f"Not all IDs unique among 20 contestants"


class TestArenaFromOutputsCLI:
    """Test the arena --from-outputs CLI mode."""

    @pytest.fixture
    def sample_output_files(self, tmp_path):
        """Create sample output YAML files."""
        output1 = {
            "results": [{"criterion_name": "c1", "dimension": "d1", "score": 3, "max_score": 3}],
            "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
            "rubric": {"dimensions": [], "criteria": []},
            "judge_panel": {"judges": []},
            "metadata": {"report_title": "Model A"},
        }

        output2 = {
            "results": [{"criterion_name": "c1", "dimension": "d1", "score": 1, "max_score": 3}],
            "summary": {"total_score": 1, "max_score": 3, "percentage": 33.3},
            "rubric": {"dimensions": [], "criteria": []},
            "judge_panel": {"judges": []},
            "metadata": {"report_title": "Model B"},
        }

        file1 = tmp_path / "out1.yaml"
        file2 = tmp_path / "out2.yaml"

        with open(file1, "w") as f:
            yaml.dump(output1, f)
        with open(file2, "w") as f:
            yaml.dump(output2, f)

        return [str(file1), str(file2)]

    def test_arena_from_outputs_cli(self, sample_output_files, tmp_path):
        """Test arena --from-outputs CLI command."""
        import sys

        from rubric_kit.main import main

        output_path = str(tmp_path / "arena_result.yaml")

        sys.argv = (
            ["rubric-kit", "arena", "--from-outputs"]
            + sample_output_files
            + ["--output-file", output_path]
        )

        result = main()

        assert result == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            data = yaml.safe_load(f)

        assert data["mode"] == "arena"
        assert len(data["contestants"]) == 2
        assert len(data["rankings"]) == 2

    def test_arena_from_outputs_with_report(self, sample_output_files, tmp_path):
        """Test arena --from-outputs with PDF report generation."""
        import sys

        from rubric_kit.main import main

        output_path = str(tmp_path / "arena_result.yaml")
        pdf_path = str(tmp_path / "arena_report.pdf")

        sys.argv = (
            ["rubric-kit", "arena", "--from-outputs"]
            + sample_output_files
            + [
                "--output-file",
                output_path,
                "--report",
                pdf_path,
                "--report-title",
                "Custom Arena Title",
            ]
        )

        result = main()

        assert result == 0
        assert os.path.exists(output_path)
        assert os.path.exists(pdf_path)


class TestArenaPDFExport:
    """Test arena PDF export functionality."""

    @pytest.fixture
    def arena_output_file(self, tmp_path):
        """Create a sample arena output YAML file."""
        data = {
            "mode": "arena",
            "arena_name": "Test Arena",
            "arena_description": "Testing",
            "contestants": {
                "model_a": {
                    "name": "Model A",
                    "description": "Test model A",
                    "metadata": {"version": "1.0"},
                    "input": {"type": "chat_session", "source_file": "a.txt"},
                    "results": [
                        {
                            "criterion_name": "crit1",
                            "dimension": "accuracy",
                            "result": "pass",
                            "score": 3,
                            "max_score": 3,
                            "reason": "Good",
                        }
                    ],
                    "summary": {"total_score": 3, "max_score": 3, "percentage": 100.0},
                },
                "model_b": {
                    "name": "Model B",
                    "description": "Test model B",
                    "metadata": {"version": "2.0"},
                    "input": {"type": "chat_session", "source_file": "b.txt"},
                    "results": [
                        {
                            "criterion_name": "crit1",
                            "dimension": "accuracy",
                            "result": "fail",
                            "score": 0,
                            "max_score": 3,
                            "reason": "Failed",
                        }
                    ],
                    "summary": {"total_score": 0, "max_score": 3, "percentage": 0.0},
                },
            },
            "rankings": [
                {
                    "rank": 1,
                    "id": "model_a",
                    "name": "Model A",
                    "percentage": 100.0,
                    "total_score": 3,
                    "max_score": 3,
                },
                {
                    "rank": 2,
                    "id": "model_b",
                    "name": "Model B",
                    "percentage": 0.0,
                    "total_score": 0,
                    "max_score": 3,
                },
            ],
            "rubric": {
                "dimensions": [
                    {"name": "accuracy", "description": "Test", "grading_type": "binary"}
                ],
                "criteria": [{"name": "crit1", "dimension": "accuracy", "weight": 3}],
            },
            "judge_panel": {
                "judges": [{"name": "test", "model": "gpt-4"}],
                "execution": {"mode": "sequential"},
                "consensus": {"mode": "unanimous"},
            },
            "metadata": {"timestamp": "2025-01-01T00:00:00"},
        }

        output_path = tmp_path / "arena_output.yaml"
        with open(output_path, "w") as f:
            yaml.dump(data, f)

        return output_path

    def test_export_arena_pdf(self, arena_output_file, tmp_path):
        """Test that arena PDF export works."""
        from rubric_kit.pdf_export import export_arena_pdf

        pdf_path = tmp_path / "arena_report.pdf"

        export_arena_pdf(str(arena_output_file), str(pdf_path))

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_export_arena_pdf_rejects_non_arena(self, tmp_path):
        """Test that arena PDF export rejects non-arena files."""
        from rubric_kit.pdf_export import export_arena_pdf

        # Create a regular evaluation file
        regular_file = tmp_path / "regular.yaml"
        with open(regular_file, "w") as f:
            yaml.dump({"results": [], "rubric": {}, "judge_panel": {}}, f)

        pdf_path = tmp_path / "report.pdf"

        with pytest.raises(ValueError, match="not an arena evaluation"):
            export_arena_pdf(str(regular_file), str(pdf_path))
