"""Tests for main CLI script."""

import pytest
import tempfile
import os
import json
import yaml


@pytest.fixture
def sample_rubric_file():
    """Create a sample rubric YAML file."""
    rubric_yaml = """
descriptors:
  - factual_correctness: Test correctness
    grading_type: binary
  - usefulness: Test usefulness
    grading_type: score
    scores:
      1: Not useful
      2: Somewhat useful
      3: Very useful

criteria:
  fact_1:
    category: Output
    weight: 3
    dimension: factual_correctness
    criterion: Check fact 1
  useful_1:
    category: Output
    weight: from_scores
    dimension: usefulness
    criterion: from_scores
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(rubric_yaml)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_evaluations_file():
    """Create a sample evaluations JSON file."""
    evaluations = {
        "fact_1": {
            "type": "binary",
            "passes": True
        },
        "useful_1": {
            "type": "score",
            "score": 3
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(evaluations, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


def test_main_function(sample_rubric_file, sample_evaluations_file, capsys):
    """Test the main function."""
    from rubric_kit.main import main
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        output_path = f.name
    
    try:
        # Call main with test files
        import sys
        sys.argv = ['rubric-kit', sample_evaluations_file, sample_rubric_file, output_path]
        
        result = main()
        
        # Should return 0 for success
        assert result == 0
        
        # Check that output file was created
        assert os.path.exists(output_path)
        
        # Check that table was printed
        captured = capsys.readouterr()
        assert "fact_1" in captured.out
        assert "useful_1" in captured.out
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_load_evaluations_json():
    """Test loading evaluations from JSON file."""
    from rubric_kit.main import load_evaluations
    
    evaluations = {
        "test_1": {"type": "binary", "passes": True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(evaluations, f)
        temp_path = f.name
    
    try:
        loaded = load_evaluations(temp_path)
        assert loaded == evaluations
    finally:
        os.unlink(temp_path)


def test_load_evaluations_yaml():
    """Test loading evaluations from YAML file."""
    from rubric_kit.main import load_evaluations
    
    evaluations = {
        "test_1": {"type": "binary", "passes": True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(evaluations, f)
        temp_path = f.name
    
    try:
        loaded = load_evaluations(temp_path)
        assert loaded == evaluations
    finally:
        os.unlink(temp_path)


def test_main_with_missing_file():
    """Test main function with missing file."""
    from rubric_kit.main import main
    import sys
    
    sys.argv = ['rubric-kit', 'nonexistent.json', 'nonexistent.yaml', 'output.csv']
    
    result = main()
    
    # Should return non-zero for error
    assert result != 0


def test_cli_help():
    """Test CLI help message."""
    from rubric_kit.main import main
    import sys
    
    sys.argv = ['rubric-kit', '--help']
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    
    # Help should exit with 0
    assert exc_info.value.code == 0

