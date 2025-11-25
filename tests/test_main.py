"""Tests for main CLI script."""

import pytest
import tempfile
import os
import yaml
import json
from unittest.mock import patch, Mock

from rubric_kit.schema import Rubric, Dimension, Criterion


@pytest.fixture
def sample_rubric_file():
    """Create a sample rubric YAML file."""
    rubric_yaml = """
dimensions:
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
def sample_evaluation_yaml():
    """Create a sample evaluation YAML file for export tests (new self-contained format)."""
    data = {
        "results": [
            {
                "criterion_name": "fact_1",
                "category": "Output",
                "dimension": "factual_correctness",
                "result": "pass",
                "score": 3,
                "max_score": 3,
                "reason": "Correct"
            }
        ],
        "summary": {
            "total_score": 3,
            "max_score": 3,
            "percentage": 100.0
        },
        "rubric": {
            "dimensions": [
                {
                    "name": "factual_correctness",
                    "description": "Test correctness",
                    "grading_type": "binary",
                    "scores": None,
                    "pass_above": None
                }
            ],
            "criteria": [
                {
                    "name": "fact_1",
                    "category": "Output",
                    "dimension": "factual_correctness",
                    "criterion": "Check fact 1",
                    "weight": 3,
                    "tool_calls": None
                }
            ]
        },
        "judge_panel": {
            "judges": [{"name": "default", "model": "gpt-4", "base_url": None}],
            "execution": {"mode": "sequential", "batch_size": 2, "timeout": 30},
            "consensus": {"mode": "unanimous", "threshold": 1, "on_no_consensus": "fail"}
        },
        "input": {
            "type": "chat_session",
            "source_file": "test.txt"
        },
        "metadata": {
            "timestamp": "2024-01-01T12:00:00",
            "rubric_source_file": "test.yaml",
            "judge_panel_source_file": None
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_chat_session_file():
    """Create a sample chat session file."""
    chat_content = """User: Test question?
Assistant: Test answer with information."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(chat_content)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_qa_file():
    """Create a sample Q&A file for generation."""
    qa_content = """Q: What is the capital of France?
A: The capital of France is Paris."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(qa_content)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


class TestEvaluateCommand:
    """Test the 'evaluate' subcommand."""
    
    @patch('rubric_kit.main.evaluate_rubric_with_panel')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_evaluate_command(self, mock_eval_llm, sample_rubric_file, sample_chat_session_file, capsys):
        """Test the evaluate subcommand with LLM judge - always outputs YAML."""
        from rubric_kit.main import main
        
        # Mock LLM evaluations
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            # Call main with evaluate subcommand
            import sys
            sys.argv = ['rubric-kit', 'evaluate', '--from-chat-session', sample_chat_session_file, '--rubric-file', sample_rubric_file, '--output-file', output_path]
            
            result = main()
            
            # Should return 0 for success
            assert result == 0
            
            # Check that output file was created (YAML)
            assert os.path.exists(output_path)
            
            # Verify it's valid YAML with expected structure
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)
            assert "results" in data
            assert "metadata" in data
            
            # Check that table was printed
            captured = capsys.readouterr()
            assert "fact_1" in captured.out
            assert "useful_1" in captured.out
            
            # Verify LLM was called
            assert mock_eval_llm.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('rubric_kit.main.evaluate_rubric_with_panel')
    @patch('rubric_kit.main.export_evaluation_pdf')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_evaluate_with_report(self, mock_pdf, mock_eval_llm, sample_rubric_file, sample_chat_session_file):
        """Test evaluate subcommand with --report flag generates PDF."""
        from rubric_kit.main import main
        import sys
        
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            pdf_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'evaluate', '--from-chat-session', sample_chat_session_file, 
                       '--rubric-file', sample_rubric_file, '--output-file', output_path,
                       '--report', pdf_path]
            
            result = main()
            
            assert result == 0
            assert mock_pdf.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    @patch('rubric_kit.main.evaluate_rubric_with_panel')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_evaluate_with_report_title(self, mock_eval_llm, sample_rubric_file, sample_chat_session_file):
        """Test evaluate subcommand with --report-title stores title in metadata."""
        from rubric_kit.main import main
        import sys
        
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'evaluate', '--from-chat-session', sample_chat_session_file, 
                       '--rubric-file', sample_rubric_file, '--output-file', output_path,
                       '--report-title', 'Q1 2025 Evaluation']
            
            result = main()
            
            assert result == 0
            
            # Verify report title is in metadata
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)
            assert data["metadata"]["report_title"] == "Q1 2025 Evaluation"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('rubric_kit.main.evaluate_rubric_with_panel')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_evaluate_output_is_self_contained(self, mock_eval_llm, sample_rubric_file, sample_chat_session_file):
        """Test evaluate subcommand produces self-contained output with rubric and judge_panel at top level."""
        from rubric_kit.main import main
        import sys
        
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'evaluate', '--from-chat-session', sample_chat_session_file, 
                       '--rubric-file', sample_rubric_file, '--output-file', output_path]
            
            result = main()
            
            assert result == 0
            
            # Verify self-contained structure
            with open(output_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Rubric at top level (not in metadata)
            assert "rubric" in data
            assert "dimensions" in data["rubric"]
            assert "criteria" in data["rubric"]
            assert len(data["rubric"]["dimensions"]) == 2
            assert len(data["rubric"]["criteria"]) == 2
            
            # Judge panel at top level
            assert "judge_panel" in data
            assert "judges" in data["judge_panel"]
            assert "execution" in data["judge_panel"]
            assert "consensus" in data["judge_panel"]
            
            # Input section
            assert "input" in data
            assert data["input"]["type"] == "chat_session"
            
            # Summary section
            assert "summary" in data
            assert "total_score" in data["summary"]
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_evaluate_with_missing_api_key(self, sample_rubric_file, sample_chat_session_file):
        """Test evaluate subcommand without API key."""
        from rubric_kit.main import main
        import sys
        
        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            sys.argv = ['rubric-kit', 'evaluate', '--from-chat-session', sample_chat_session_file, '--rubric-file', sample_rubric_file, '--output-file', 'output.yaml']
            
            result = main()
            
            # Should return non-zero for error
            assert result == 1
    
    def test_evaluate_with_missing_file(self):
        """Test evaluate subcommand with missing file."""
        from rubric_kit.main import main
        import sys
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            sys.argv = ['rubric-kit', 'evaluate', '--from-chat-session', 'nonexistent.txt', '--rubric-file', 'nonexistent.yaml', '--output-file', 'output.yaml']
            
            result = main()
            
            # Should return non-zero for error
            assert result != 0


class TestGenerateCommand:
    """Test the 'generate' subcommand."""
    
    @patch('rubric_kit.main.RubricGenerator')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_generate_command(self, mock_generator_class, sample_qa_file):
        """Test the generate subcommand."""
        from rubric_kit.main import main
        import sys
        
        # Mock the generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Mock rubric generation
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="factual_correctness",
                    description="Test correctness",
                    grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="Check fact 1"
                )
            ]
        )
        mock_generator.generate_rubric.return_value = mock_rubric
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'generate', '--from-qna', sample_qa_file, '--output-file', output_path]
            
            result = main()
            
            # Should return 0 for success
            assert result == 0
            
            # Check that output file was created
            assert os.path.exists(output_path)
            
            # Verify generator was called
            assert mock_generator.generate_rubric.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('rubric_kit.main.RubricGenerator')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_generate_with_parameters(self, mock_generator_class, sample_qa_file):
        """Test generate command with custom parameters."""
        from rubric_kit.main import main
        import sys
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="test",
                    description="Test",
                    grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="test_1",
                    category="Output",
                    weight=3,
                    dimension="test",
                    criterion="Test"
                )
            ]
        )
        mock_generator.generate_rubric.return_value = mock_rubric
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = [
                'rubric-kit', 'generate', '--from-qna', sample_qa_file, '--output-file', output_path,
                '--num-dimensions', '3',
                '--num-criteria', '5',
                '--categories', 'Output,Reasoning',
                '--model', 'gpt-4-turbo'
            ]
            
            result = main()
            
            assert result == 0
            
            # Verify generator was called with correct parameters
            call_args = mock_generator.generate_rubric.call_args
            assert call_args[1]['num_dimensions'] == 3
            assert call_args[1]['num_criteria'] == 5
            assert call_args[1]['category_hints'] == ['Output', 'Reasoning']
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_generate_with_missing_api_key(self, sample_qa_file):
        """Test generate subcommand without API key."""
        from rubric_kit.main import main
        import sys
        
        with patch.dict(os.environ, {}, clear=True):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                output_path = f.name
            
            try:
                sys.argv = ['rubric-kit', 'generate', '--from-qna', sample_qa_file, '--output-file', output_path]
                
                result = main()
                
                # Should return non-zero for error
                assert result == 1
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)


class TestRefineCommand:
    """Test the 'refine' subcommand."""
    
    @patch('rubric_kit.main.RubricGenerator')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_refine_command(self, mock_generator_class, sample_rubric_file):
        """Test the refine subcommand."""
        from rubric_kit.main import main
        import sys
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # Mock refined rubric
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="factual_correctness",
                    description="Improved correctness description",
                    grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="Improved criterion text"
                )
            ]
        )
        mock_generator.refine_rubric.return_value = mock_rubric
        
        sys.argv = ['rubric-kit', 'refine', '--rubric-file', sample_rubric_file]
        
        result = main()
        
        # Should return 0 for success
        assert result == 0
        
        # Verify refine_rubric was called
        assert mock_generator.refine_rubric.called
    
    @patch('rubric_kit.main.RubricGenerator')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_refine_with_feedback(self, mock_generator_class, sample_rubric_file):
        """Test refine command with feedback."""
        from rubric_kit.main import main
        import sys
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="test",
                    description="Test",
                    grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="test_1",
                    category="Output",
                    weight=3,
                    dimension="test",
                    criterion="Test"
                )
            ]
        )
        mock_generator.refine_rubric.return_value = mock_rubric
        
        feedback = "Add more specific criteria"
        sys.argv = ['rubric-kit', 'refine', '--rubric-file', sample_rubric_file, '--feedback', feedback]
        
        result = main()
        
        assert result == 0
        
        # Verify feedback was passed
        call_args = mock_generator.refine_rubric.call_args
        assert call_args[1]['feedback'] == feedback
    
    @patch('rubric_kit.main.RubricGenerator')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_refine_with_output(self, mock_generator_class, sample_rubric_file):
        """Test refine command with custom output path."""
        from rubric_kit.main import main
        import sys
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        mock_rubric = Rubric(
            dimensions=[
                Dimension(
                    name="test",
                    description="Test",
                    grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="test_1",
                    category="Output",
                    weight=3,
                    dimension="test",
                    criterion="Test"
                )
            ]
        )
        mock_generator.refine_rubric.return_value = mock_rubric
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'refine', '--rubric-file', sample_rubric_file, '--output-file', output_path]
            
            result = main()
            
            assert result == 0
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestRerunCommand:
    """Test the 'rerun' subcommand."""
    
    @patch('rubric_kit.main.evaluate_rubric_with_panel')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_rerun_with_embedded_input(self, mock_eval_llm, sample_evaluation_yaml):
        """Test rerun subcommand uses settings from self-contained YAML."""
        from rubric_kit.main import main
        import sys
        
        # Add embedded input content to the fixture
        with open(sample_evaluation_yaml, 'r') as f:
            data = yaml.safe_load(f)
        data["input"]["content"] = "User: Test question?\nAssistant: Test answer."
        with open(sample_evaluation_yaml, 'w') as f:
            yaml.dump(data, f)
        
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'rerun', sample_evaluation_yaml, '--output-file', output_path]
            
            result = main()
            
            assert result == 0
            assert os.path.exists(output_path)
            
            # Verify output has same structure
            with open(output_path, 'r') as f:
                new_data = yaml.safe_load(f)
            
            assert "rubric" in new_data
            assert "judge_panel" in new_data
            assert "results" in new_data
            assert new_data["metadata"].get("rerun_from") == sample_evaluation_yaml
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @patch('rubric_kit.main.evaluate_rubric_with_panel')
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'})
    def test_rerun_with_new_input(self, mock_eval_llm, sample_evaluation_yaml, sample_chat_session_file):
        """Test rerun with new input file overrides embedded/original input."""
        from rubric_kit.main import main
        import sys
        
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'rerun', sample_evaluation_yaml, 
                       '--from-chat-session', sample_chat_session_file,
                       '--output-file', output_path]
            
            result = main()
            
            assert result == 0
            assert mock_eval_llm.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_rerun_missing_input_file(self):
        """Test rerun subcommand with missing input file."""
        from rubric_kit.main import main
        import sys
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            sys.argv = ['rubric-kit', 'rerun', 'nonexistent.yaml', '--output-file', 'output.yaml']
            
            result = main()
            
            assert result != 0


class TestExportCommand:
    """Test the 'export' subcommand."""
    
    def test_export_to_pdf(self, sample_evaluation_yaml):
        """Test export subcommand with --format pdf."""
        from rubric_kit.main import main
        import sys
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            pdf_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'export', sample_evaluation_yaml, '--format', 'pdf', '--output', pdf_path]
            
            result = main()
            
            assert result == 0
            assert os.path.exists(pdf_path)
            assert os.path.getsize(pdf_path) > 0
        finally:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
    
    def test_export_to_csv(self, sample_evaluation_yaml):
        """Test export subcommand with --format csv."""
        from rubric_kit.main import main
        import sys
        import csv
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'export', sample_evaluation_yaml, '--format', 'csv', '--output', csv_path]
            
            result = main()
            
            assert result == 0
            assert os.path.exists(csv_path)
            
            # Verify CSV content
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) >= 1
            assert rows[0]["criterion_name"] == "fact_1"
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
    
    def test_export_to_json(self, sample_evaluation_yaml):
        """Test export subcommand with --format json."""
        from rubric_kit.main import main
        import sys
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            sys.argv = ['rubric-kit', 'export', sample_evaluation_yaml, '--format', 'json', '--output', json_path]
            
            result = main()
            
            assert result == 0
            assert os.path.exists(json_path)
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
            assert "results" in data
            assert len(data["results"]) >= 1
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)
    
    def test_export_missing_input_file(self):
        """Test export subcommand with missing input file."""
        from rubric_kit.main import main
        import sys
        
        sys.argv = ['rubric-kit', 'export', 'nonexistent.yaml', '--format', 'pdf', '--output', 'output.pdf']
        
        result = main()
        
        assert result != 0
    
    def test_export_requires_format(self, sample_evaluation_yaml):
        """Test export subcommand requires --format argument."""
        from rubric_kit.main import main
        import sys
        
        sys.argv = ['rubric-kit', 'export', sample_evaluation_yaml, '--output', 'output.pdf']
        
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Should exit with error due to missing required argument
        assert exc_info.value.code != 0


def test_cli_help():
    """Test CLI help message."""
    from rubric_kit.main import main
    import sys
    
    sys.argv = ['rubric-kit', '--help']
    
    with pytest.raises(SystemExit) as exc_info:
        main()
    
    # Help should exit with 0
    assert exc_info.value.code == 0


def test_cli_no_subcommand():
    """Test CLI without subcommand shows help."""
    from rubric_kit.main import main
    import sys
    
    sys.argv = ['rubric-kit']
    
    # Should either show help or return error
    result = main()
    assert result != 0 or result is None



