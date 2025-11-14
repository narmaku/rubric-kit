"""Tests for output handlers."""

import pytest
import tempfile
import os
import csv
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
            "max_score": 3
        },
        {
            "criterion_name": "fact_2",
            "criterion_text": "Check fact 2",
            "category": "Output",
            "dimension": "factual_correctness",
            "result": "fail",
            "score": 0,
            "max_score": 2
        },
        {
            "criterion_name": "useful_1",
            "criterion_text": "from_scores",
            "category": "Output",
            "dimension": "usefulness",
            "result": 3,
            "score": 3,
            "max_score": 3,
            "score_description": "Very useful"
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
            "max_score": 1
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

