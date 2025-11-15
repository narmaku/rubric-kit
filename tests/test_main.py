"""Tests for main CLI script."""

import pytest
import tempfile
import os
from unittest.mock import patch, Mock

from rubric_kit.schema import Rubric, Dimension, Criterion, Criterion


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
        """Test the evaluate subcommand with LLM judge."""
        from rubric_kit.main import main
        
        # Mock LLM evaluations
        mock_eval_llm.return_value = {
            "fact_1": {"type": "binary", "passes": True},
            "useful_1": {"type": "score", "score": 3}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            # Call main with evaluate subcommand
            import sys
            sys.argv = ['rubric-kit', 'evaluate', sample_chat_session_file, sample_rubric_file, output_path]
            
            result = main()
            
            # Should return 0 for success
            assert result == 0
            
            # Check that output file was created
            assert os.path.exists(output_path)
            
            # Check that table was printed
            captured = capsys.readouterr()
            assert "fact_1" in captured.out
            assert "useful_1" in captured.out
            
            # Verify LLM was called
            assert mock_eval_llm.called
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_evaluate_with_missing_api_key(self, sample_rubric_file, sample_chat_session_file):
        """Test evaluate subcommand without API key."""
        from rubric_kit.main import main
        import sys
        
        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            sys.argv = ['rubric-kit', 'evaluate', sample_chat_session_file, sample_rubric_file, 'output.csv']
            
            result = main()
            
            # Should return non-zero for error
            assert result == 1
    
    def test_evaluate_with_missing_file(self):
        """Test evaluate subcommand with missing file."""
        from rubric_kit.main import main
        import sys
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            sys.argv = ['rubric-kit', 'evaluate', 'nonexistent.txt', 'nonexistent.yaml', 'output.csv']
            
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
            sys.argv = ['rubric-kit', 'generate', sample_qa_file, output_path]
            
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
                'rubric-kit', 'generate', sample_qa_file, output_path,
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
                sys.argv = ['rubric-kit', 'generate', sample_qa_file, output_path]
                
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
        
        sys.argv = ['rubric-kit', 'refine', sample_rubric_file]
        
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
        sys.argv = ['rubric-kit', 'refine', sample_rubric_file, '--feedback', feedback]
        
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
            sys.argv = ['rubric-kit', 'refine', sample_rubric_file, '--output', output_path]
            
            result = main()
            
            assert result == 0
            assert os.path.exists(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


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

