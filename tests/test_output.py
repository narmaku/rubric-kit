"""Tests for output handlers."""

import pytest
import tempfile
import os
import csv
import json
import yaml
from io import StringIO


@pytest.fixture
def sample_results():
    """Sample evaluation results for testing."""
    return [
        {
            "criterion_name": "fact_1",
            "criterion_text": "Check fact 1",
            "category": "Output",
            "dimension": "factual_correctness",
            "result": "pass",
            "score": 3,
            "max_score": 3,
            "reason": "The fact is correct"
        },
        {
            "criterion_name": "fact_2",
            "criterion_text": "Check fact 2",
            "category": "Output",
            "dimension": "factual_correctness",
            "result": "fail",
            "score": 0,
            "max_score": 2,
            "reason": "The fact is incorrect"
        },
        {
            "criterion_name": "useful_1",
            "criterion_text": "from_scores",
            "category": "Output",
            "dimension": "usefulness",
            "result": 3,
            "score": 3,
            "max_score": 3,
            "score_description": "Very useful",
            "reason": "Very useful. Additional comments from LLM."
        }
    ]


def test_write_csv(sample_results):
    """Test writing results to CSV."""
    from rubric_kit.output import write_csv
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_csv(sample_results, temp_path)
        
        # Read back and verify
        with open(temp_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["criterion_name"] == "fact_1"
        assert rows[0]["score"] == "3"
        assert rows[1]["result"] == "fail"
        assert rows[2]["score_description"] == "Very useful"
        # Check that reason column is present
        assert "reason" in rows[0]
    finally:
        os.unlink(temp_path)


def test_csv_has_summary_row(sample_results):
    """Test that CSV includes summary row."""
    from rubric_kit.output import write_csv
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_csv(sample_results, temp_path, include_summary=True)
        
        with open(temp_path, 'r') as f:
            content = f.read()
            # Check for summary information
            assert "TOTAL" in content or "Summary" in content
    finally:
        os.unlink(temp_path)


def test_format_table(sample_results):
    """Test formatting results as a table."""
    from rubric_kit.output import format_table
    
    table_str = format_table(sample_results)
    
    assert isinstance(table_str, str)
    assert "fact_1" in table_str
    assert "fact_2" in table_str
    assert "useful_1" in table_str
    assert "pass" in table_str
    assert "fail" in table_str
    # Check for score columns
    assert "3" in table_str
    # Check that new simplified columns are included
    assert "Dimension" in table_str
    assert "Consensus" in table_str
    assert "Agreement" in table_str


def test_format_table_with_summary(sample_results):
    """Test formatting table with summary."""
    from rubric_kit.output import format_table
    
    table_str = format_table(sample_results, include_summary=True)
    
    assert "Total" in table_str or "TOTAL" in table_str
    # Should show 6/8 (3+0+3 out of 3+2+3)
    assert "6" in table_str
    assert "8" in table_str


def test_print_table(sample_results, capsys):
    """Test printing table to stdout."""
    from rubric_kit.output import print_table
    
    print_table(sample_results)
    
    captured = capsys.readouterr()
    assert "fact_1" in captured.out
    assert "fact_2" in captured.out


def test_csv_headers():
    """Test that CSV has correct headers."""
    from rubric_kit.output import write_csv
    
    results = [
        {
            "criterion_name": "test",
            "criterion_text": "Test criterion",
            "category": "Output",
            "dimension": "test_dim",
            "result": "pass",
            "score": 1,
            "max_score": 1,
            "reason": "Test reason"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_csv(results, temp_path)
        
        with open(temp_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)
        
        # Check that essential headers are present
        assert "criterion_name" in headers
        assert "score" in headers
        assert "result" in headers
        assert "reason" in headers
    finally:
        os.unlink(temp_path)


def test_empty_results():
    """Test handling empty results."""
    from rubric_kit.output import format_table, write_csv
    
    # Should not crash with empty results
    table_str = format_table([])
    assert isinstance(table_str, str)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_csv([], temp_path)
        assert os.path.exists(temp_path)
    finally:
        os.unlink(temp_path)


def test_write_json(sample_results):
    """Test writing results to JSON."""
    from rubric_kit.output import write_json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        write_json(sample_results, temp_path)
        
        # Read back and verify
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "results" in data
        assert len(data["results"]) == 3
        assert data["results"][0]["criterion_name"] == "fact_1"
        assert data["results"][0]["score"] == 3
        assert data["results"][1]["result"] == "fail"
        assert data["results"][2]["score_description"] == "Very useful"
    finally:
        os.unlink(temp_path)


def test_write_json_with_summary(sample_results):
    """Test that JSON includes summary when requested."""
    from rubric_kit.output import write_json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        write_json(sample_results, temp_path, include_summary=True)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "summary" in data
        assert data["summary"]["criterion_name"] == "TOTAL"
        assert data["summary"]["score"] == 6  # 3+0+3
        assert data["summary"]["max_score"] == 8  # 3+2+3
    finally:
        os.unlink(temp_path)


def test_write_yaml(sample_results):
    """Test writing results to YAML."""
    from rubric_kit.output import write_yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        write_yaml(sample_results, temp_path)
        
        # Read back and verify
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert "results" in data
        assert len(data["results"]) == 3
        assert data["results"][0]["criterion_name"] == "fact_1"
        assert data["results"][0]["score"] == 3
        assert data["results"][1]["result"] == "fail"
        assert data["results"][2]["score_description"] == "Very useful"
    finally:
        os.unlink(temp_path)


def test_write_yaml_with_summary(sample_results):
    """Test that YAML includes summary when requested."""
    from rubric_kit.output import write_yaml
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        write_yaml(sample_results, temp_path, include_summary=True)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert "summary" in data
        assert data["summary"]["criterion_name"] == "TOTAL"
        assert data["summary"]["score"] == 6  # 3+0+3
        assert data["summary"]["max_score"] == 8  # 3+2+3
    finally:
        os.unlink(temp_path)


def test_detect_format_from_extension():
    """Test format detection from file extension."""
    from rubric_kit.output import detect_format_from_extension
    
    assert detect_format_from_extension("output.csv") == "csv"
    assert detect_format_from_extension("output.json") == "json"
    assert detect_format_from_extension("output.yaml") == "yaml"
    assert detect_format_from_extension("output.yml") == "yaml"
    assert detect_format_from_extension("output.txt") == "csv"  # Default


def test_write_results_csv(sample_results):
    """Test write_results function with CSV format."""
    from rubric_kit.output import write_results
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = f.name
    
    try:
        write_results(sample_results, temp_path, format="csv")
        
        # Read back and verify
        with open(temp_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
        assert rows[0]["criterion_name"] == "fact_1"
    finally:
        os.unlink(temp_path)


def test_write_results_json(sample_results):
    """Test write_results function with JSON format."""
    from rubric_kit.output import write_results
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        write_results(sample_results, temp_path, format="json")
        
        # Read back and verify
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "results" in data
        assert len(data["results"]) == 3
    finally:
        os.unlink(temp_path)


def test_write_results_yaml(sample_results):
    """Test write_results function with YAML format."""
    from rubric_kit.output import write_results
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        write_results(sample_results, temp_path, format="yaml")
        
        # Read back and verify
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert "results" in data
        assert len(data["results"]) == 3
    finally:
        os.unlink(temp_path)


def test_write_results_auto_detect(sample_results):
    """Test write_results function with auto-detection from extension."""
    from rubric_kit.output import write_results
    
    # Test JSON auto-detection
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    # Test YAML auto-detection
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
    
    try:
        write_results(sample_results, json_path)  # Should auto-detect JSON
        write_results(sample_results, yaml_path)  # Should auto-detect YAML
        
        # Verify JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        assert "results" in json_data
        
        # Verify YAML
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        assert "results" in yaml_data
    finally:
        os.unlink(json_path)
        os.unlink(yaml_path)


def test_write_results_invalid_format(sample_results):
    """Test write_results function with invalid format."""
    from rubric_kit.output import write_results
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Unsupported format"):
            write_results(sample_results, temp_path, format="invalid")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_write_json_with_metadata(sample_results):
    """Test that JSON includes metadata when provided."""
    from rubric_kit.output import write_json
    
    metadata = {
        "rubric_file": "test_rubric.yaml",
        "input_file": "test_input.txt",
        "timestamp": "2024-01-01T12:00:00",
        "judge_panel": {
            "judges": ["judge1", "judge2"],
            "consensus_mode": "majority"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        write_json(sample_results, temp_path, metadata=metadata)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert data["metadata"]["rubric_file"] == "test_rubric.yaml"
        assert data["metadata"]["input_file"] == "test_input.txt"
        assert data["metadata"]["timestamp"] == "2024-01-01T12:00:00"
        assert "judge_panel" in data["metadata"]
        assert data["results"] == sample_results
    finally:
        os.unlink(temp_path)


def test_write_yaml_with_metadata(sample_results):
    """Test that YAML includes metadata when provided."""
    from rubric_kit.output import write_yaml
    
    metadata = {
        "rubric_file": "test_rubric.yaml",
        "input_file": "test_input.txt",
        "timestamp": "2024-01-01T12:00:00"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        write_yaml(sample_results, temp_path, metadata=metadata)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        assert "metadata" in data
        assert data["metadata"]["rubric_file"] == "test_rubric.yaml"
        assert data["metadata"]["input_file"] == "test_input.txt"
        assert data["results"] == sample_results
    finally:
        os.unlink(temp_path)


def test_write_json_without_metadata(sample_results):
    """Test that JSON works without metadata (backward compatibility)."""
    from rubric_kit.output import write_json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        write_json(sample_results, temp_path)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Should not have metadata key if not provided
        assert "metadata" not in data
        assert "results" in data
    finally:
        os.unlink(temp_path)


def test_write_results_with_metadata(sample_results):
    """Test write_results function passes metadata to writers."""
    from rubric_kit.output import write_results
    
    metadata = {
        "rubric_file": "test_rubric.yaml",
        "timestamp": "2024-01-01T12:00:00"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        write_results(sample_results, temp_path, metadata=metadata)
        
        with open(temp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "metadata" in data
        assert data["metadata"]["rubric_file"] == "test_rubric.yaml"
    finally:
        os.unlink(temp_path)


def test_convert_yaml_to_csv(sample_results):
    """Test converting YAML evaluation results to CSV."""
    from rubric_kit.output import convert_yaml_to_csv
    
    # Create source YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
        yaml.dump({"results": sample_results, "summary": {"criterion_name": "TOTAL", "score": 6}}, f)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        convert_yaml_to_csv(yaml_path, csv_path)
        
        # Verify CSV was created
        assert os.path.exists(csv_path)
        
        # Read and verify content (skip comment lines)
        with open(csv_path, 'r') as f:
            lines = [line for line in f if not line.startswith('#')]
        
        import io
        reader = csv.DictReader(io.StringIO(''.join(lines)))
        rows = list(reader)
        
        assert len(rows) >= 3  # At least 3 results
        assert rows[0]["criterion_name"] == "fact_1"
    finally:
        os.unlink(yaml_path)
        os.unlink(csv_path)


def test_convert_yaml_to_json(sample_results):
    """Test converting YAML evaluation results to JSON."""
    from rubric_kit.output import convert_yaml_to_json
    
    # Create source YAML file
    metadata = {"rubric_file": "test.yaml", "timestamp": "2024-01-01T12:00:00"}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
        yaml.dump({"results": sample_results, "metadata": metadata}, f)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        convert_yaml_to_json(yaml_path, json_path)
        
        # Verify JSON was created
        assert os.path.exists(json_path)
        
        # Read and verify content
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert "results" in data
        assert "metadata" in data
        assert len(data["results"]) == 3
        assert data["metadata"]["rubric_file"] == "test.yaml"
    finally:
        os.unlink(yaml_path)
        os.unlink(json_path)


def test_convert_yaml_to_csv_missing_file():
    """Test convert_yaml_to_csv with missing input file."""
    from rubric_kit.output import convert_yaml_to_csv
    
    with pytest.raises(FileNotFoundError):
        convert_yaml_to_csv("nonexistent.yaml", "output.csv")


def test_convert_yaml_to_json_missing_file():
    """Test convert_yaml_to_json with missing input file."""
    from rubric_kit.output import convert_yaml_to_json
    
    with pytest.raises(FileNotFoundError):
        convert_yaml_to_json("nonexistent.yaml", "output.json")


def test_convert_yaml_to_json_always_includes_input(sample_results):
    """Test convert_yaml_to_json always includes full input content."""
    from rubric_kit.output import convert_yaml_to_json
    
    chat_content = "User: Test question?\nAssistant: Test answer."
    
    # Create source YAML file with input content (new structured format)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
        yaml.dump({
            "results": sample_results,
            "input": {
                "type": "chat_session",
                "source_file": "test.txt",
                "chat_session": chat_content
            }
        }, f)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        convert_yaml_to_json(yaml_path, json_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Input content should always be included
        assert "input" in data
        assert "chat_session" in data["input"]
        assert data["input"]["chat_session"] == chat_content
    finally:
        os.unlink(yaml_path)
        os.unlink(json_path)


def test_convert_yaml_to_csv_includes_header_comments(sample_results):
    """Test convert_yaml_to_csv includes header comments with metadata and input summary."""
    from rubric_kit.output import convert_yaml_to_csv
    
    chat_content = "User: Test question?\nAssistant: Test answer."
    
    # Create source YAML file (new structured format)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
        yaml.dump({
            "results": sample_results,
            "input": {
                "type": "chat_session",
                "source_file": "test.txt",
                "chat_session": chat_content
            },
            "metadata": {
                "timestamp": "2024-01-01T12:00:00"
            },
            "summary": {
                "total_score": 6,
                "max_score": 9,
                "percentage": 66.7
            }
        }, f)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name
    
    try:
        convert_yaml_to_csv(yaml_path, csv_path)
        
        assert os.path.exists(csv_path)
        
        # Read and verify header comments
        with open(csv_path, 'r') as f:
            content = f.read()
        
        # Verify header comments are present
        assert content.startswith("# Evaluation Report")
        assert "# Input Type: chat_session" in content
        assert "# Chat Session:" in content
        assert "# Score:" in content
        
        # Also read as CSV to verify data
        with open(csv_path, 'r') as f:
            # Skip comment lines for CSV reader
            lines = [line for line in f if not line.startswith('#')]
        
        import io
        reader = csv.DictReader(io.StringIO(''.join(lines)))
        rows = list(reader)
        
        # CSV should still have the results (3 results + 1 summary row)
        assert len(rows) == 4
    finally:
        os.unlink(yaml_path)
        os.unlink(csv_path)

