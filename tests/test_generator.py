"""Tests for rubric generator."""

import json
import pytest
from pathlib import Path
from rubric_kit.generator import (
    RubricGenerator, QAInput, parse_qa_input, repair_json,
    ChatSessionInput, parse_chat_session
)
from rubric_kit.schema import Rubric, Dimension, Criterion


class TestQAInputParsing:
    """Test Q&A input parsing from different formats."""
    
    def test_parse_simple_text_format(self, tmp_path):
        """Test parsing simple Q: A: text format."""
        qa_file = tmp_path / "qa.txt"
        qa_file.write_text(
            "Q: What is the capital of France?\n"
            "A: The capital of France is Paris."
        )
        
        qa_input = parse_qa_input(str(qa_file))
        
        assert qa_input.question == "What is the capital of France?"
        assert qa_input.answer == "The capital of France is Paris."
        assert qa_input.context is None
    
    def test_parse_yaml_format(self, tmp_path):
        """Test parsing YAML format with optional context."""
        qa_file = tmp_path / "qa.yaml"
        qa_file.write_text(
            "question: What are the system specifications?\n"
            "answer: The system has 8 CPUs and 64GB RAM.\n"
            "context: Testing system information retrieval\n"
        )
        
        qa_input = parse_qa_input(str(qa_file))
        
        assert qa_input.question == "What are the system specifications?"
        assert qa_input.answer == "The system has 8 CPUs and 64GB RAM."
        assert qa_input.context == "Testing system information retrieval"
    
    def test_parse_multiline_qa(self, tmp_path):
        """Test parsing Q&A with multiline answers."""
        qa_file = tmp_path / "qa.txt"
        qa_file.write_text(
            "Q: Explain photosynthesis.\n"
            "A: Photosynthesis is a process used by plants.\n"
            "It converts light energy into chemical energy.\n"
            "This process occurs in chloroplasts."
        )
        
        qa_input = parse_qa_input(str(qa_file))
        
        assert qa_input.question == "Explain photosynthesis."
        assert "Photosynthesis is a process" in qa_input.answer
        assert "chloroplasts" in qa_input.answer
    
    def test_parse_empty_file_raises_error(self, tmp_path):
        """Test that empty file raises ValueError."""
        qa_file = tmp_path / "empty.txt"
        qa_file.write_text("")
        
        with pytest.raises(ValueError, match="Q&A file is empty"):
            parse_qa_input(str(qa_file))
    
    def test_parse_missing_question_raises_error(self, tmp_path):
        """Test that missing question raises ValueError."""
        qa_file = tmp_path / "qa.txt"
        qa_file.write_text("A: Just an answer")
        
        with pytest.raises(ValueError, match="Question not found"):
            parse_qa_input(str(qa_file))
    
    def test_parse_missing_answer_raises_error(self, tmp_path):
        """Test that missing answer raises ValueError."""
        qa_file = tmp_path / "qa.txt"
        qa_file.write_text("Q: Just a question")
        
        with pytest.raises(ValueError, match="Answer not found"):
            parse_qa_input(str(qa_file))


class TestRubricGenerator:
    """Test RubricGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a RubricGenerator instance."""
        return RubricGenerator(api_key="test-key", model="gpt-4")
    
    @pytest.fixture
    def simple_qa(self):
        """Simple Q&A input for testing."""
        return QAInput(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            context=None
        )
    
    def test_generator_initialization(self):
        """Test RubricGenerator initialization."""
        gen = RubricGenerator(api_key="test-key", model="gpt-4")
        assert gen.api_key == "test-key"
        assert gen.model == "gpt-4"
        assert gen.base_url is None
    
    def test_generator_with_base_url(self):
        """Test RubricGenerator with custom base URL."""
        gen = RubricGenerator(
            api_key="test-key",
            model="gpt-4",
            base_url="https://custom.api.com/v1"
        )
        assert gen.base_url == "https://custom.api.com/v1"
    
    def test_generate_dimensions_returns_list(self, generator, simple_qa, monkeypatch):
        """Test that generate_dimensions returns list of Dimensions."""
        # Mock LLM response
        mock_response = [
            {
                "name": "factual_correctness",
                "description": "Evaluates factual accuracy",
                "grading_type": "binary"
            },
            {
                "name": "completeness",
                "description": "Evaluates answer completeness",
                "grading_type": "score",
                "scores": {
                    1: "Incomplete",
                    2: "Partially complete",
                    3: "Complete"
                }
            }
        ]
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        dimensions = generator.generate_dimensions(simple_qa, num_dimensions=2)
        
        assert len(dimensions) == 2
        assert all(isinstance(d, Dimension) for d in dimensions)
        assert dimensions[0].name == "factual_correctness"
        assert dimensions[0].grading_type == "binary"
        assert dimensions[1].name == "completeness"
        assert dimensions[1].grading_type == "score"
        assert dimensions[1].scores == {1: "Incomplete", 2: "Partially complete", 3: "Complete"}
    
    def test_generate_criteria_returns_list(self, generator, simple_qa, monkeypatch):
        """Test that generate_criteria returns list of Criteria."""
        dimensions = [
            Dimension(
                name="factual_correctness",
                description="Evaluates factual accuracy",
                grading_type="binary"
            )
        ]
        
        mock_response = [
            {
                "name": "capital_fact_1",
                "category": "Output",
                "weight": 3,
                "dimension": "factual_correctness",
                "criterion": "The answer must correctly identify Paris as the capital."
            }
        ]
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        criteria = generator.generate_criteria(simple_qa, dimensions, num_criteria=1)
        
        assert len(criteria) == 1
        assert all(isinstance(c, Criterion) for c in criteria)
        assert criteria[0].name == "capital_fact_1"
        assert criteria[0].category == "Output"
        assert criteria[0].weight == 3
    
    def test_generate_criteria_with_category_hints(self, generator, simple_qa, monkeypatch):
        """Test that category hints are passed to LLM."""
        dimensions = [
            Dimension(
                name="factual_correctness",
                description="Evaluates factual accuracy",
                grading_type="binary"
            )
        ]
        
        mock_response = [
            {
                "name": "capital_fact_1",
                "category": "Accuracy",
                "weight": 3,
                "dimension": "factual_correctness",
                "criterion": "The answer must correctly identify Paris."
            }
        ]
        
        called_with_categories = []
        
        def mock_llm_call(*args, **kwargs):
            # Capture the categories passed to LLM
            if 'categories' in kwargs:
                called_with_categories.append(kwargs['categories'])
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        criteria = generator.generate_criteria(
            simple_qa,
            dimensions,
            num_criteria=1,
            category_hints=["Accuracy", "Completeness"]
        )
        
        assert len(criteria) == 1
        assert criteria[0].category == "Accuracy"
        # Verify categories were passed to LLM
        assert len(called_with_categories) > 0
    
    def test_generate_rubric_full_workflow(self, generator, simple_qa, monkeypatch):
        """Test full rubric generation workflow."""
        # Mock dimension generation
        mock_dimensions = [
            {
                "name": "factual_correctness",
                "description": "Evaluates factual accuracy",
                "grading_type": "binary"
            }
        ]
        
        # Mock criteria generation
        mock_criteria = [
            {
                "name": "capital_fact_1",
                "category": "Output",
                "weight": 3,
                "dimension": "factual_correctness",
                "criterion": "The answer must correctly identify Paris."
            }
        ]
        
        call_count = [0]
        
        def mock_llm_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_dimensions
            else:
                return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        rubric = generator.generate_rubric(
            simple_qa,
            num_dimensions=1,
            num_criteria=1
        )
        
        assert isinstance(rubric, Rubric)
        assert len(rubric.dimensions) == 1
        assert len(rubric.criteria) == 1
        assert rubric.dimensions[0].name == "factual_correctness"
        assert rubric.criteria[0].name == "capital_fact_1"
        assert call_count[0] == 2  # Two LLM calls
    
    def test_generate_rubric_validates_output(self, generator, simple_qa, monkeypatch):
        """Test that generated rubric is validated against schema."""
        # Mock invalid response (criterion references non-existent dimension)
        mock_dimensions = [
            {
                "name": "factual_correctness",
                "description": "Evaluates factual accuracy",
                "grading_type": "binary"
            }
        ]
        
        mock_criteria = [
            {
                "name": "bad_criterion",
                "category": "Output",
                "weight": 3,
                "dimension": "nonexistent_dimension",  # Invalid!
                "criterion": "This references wrong dimension"
            }
        ]
        
        call_count = [0]
        
        def mock_llm_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_dimensions
            else:
                return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        with pytest.raises(ValueError, match="references non-existent dimension"):
            generator.generate_rubric(simple_qa, num_dimensions=1, num_criteria=1)
    
    def test_generate_rubric_respects_limits(self, generator, simple_qa):
        """Test that num_dimensions and num_criteria are enforced."""
        with pytest.raises(ValueError, match="num_dimensions must be between 1 and 10"):
            generator.generate_rubric(simple_qa, num_dimensions=0, num_criteria=5)
        
        with pytest.raises(ValueError, match="num_dimensions must be between 1 and 10"):
            generator.generate_rubric(simple_qa, num_dimensions=11, num_criteria=5)
        
        with pytest.raises(ValueError, match="num_criteria must be between 1 and 10"):
            generator.generate_rubric(simple_qa, num_dimensions=5, num_criteria=0)
        
        with pytest.raises(ValueError, match="num_criteria must be between 1 and 10"):
            generator.generate_rubric(simple_qa, num_dimensions=5, num_criteria=11)


class TestRubricRefine:
    """Test rubric refinement functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create a RubricGenerator instance."""
        return RubricGenerator(api_key="test-key", model="gpt-4")
    
    @pytest.fixture
    def existing_rubric(self):
        """Create an existing rubric for refinement."""
        return Rubric(
            dimensions=[
                Dimension(
                    name="factual_correctness",
                    description="Evaluates factual accuracy",
                    grading_type="binary"
                )
            ],
            criteria=[
                Criterion(
                    name="fact_1",
                    category="Output",
                    weight=3,
                    dimension="factual_correctness",
                    criterion="The answer must be correct."
                )
            ]
        )
    
    def test_refine_rubric_with_feedback(self, generator, existing_rubric, monkeypatch):
        """Test refining a rubric with specific feedback."""
        mock_response = {
            "dimensions": [
                {
                    "name": "factual_correctness",
                    "description": "Evaluates factual accuracy of the response",
                    "grading_type": "binary"
                },
                {
                    "name": "specificity",
                    "description": "Evaluates how specific the answer is",
                    "grading_type": "binary"
                }
            ],
            "criteria": [
                {
                    "name": "fact_1",
                    "category": "Output",
                    "weight": 3,
                    "dimension": "factual_correctness",
                    "criterion": "The answer must correctly identify the capital city."
                },
                {
                    "name": "spec_1",
                    "category": "Output",
                    "weight": 2,
                    "dimension": "specificity",
                    "criterion": "The answer must provide specific details."
                }
            ]
        }
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        refined_rubric = generator.refine_rubric(
            existing_rubric,
            feedback="Add more specific criteria and a new dimension for specificity"
        )
        
        assert isinstance(refined_rubric, Rubric)
        assert len(refined_rubric.dimensions) == 2
        assert len(refined_rubric.criteria) == 2
        assert refined_rubric.dimensions[1].name == "specificity"
    
    def test_refine_rubric_without_feedback(self, generator, existing_rubric, monkeypatch):
        """Test refining a rubric without specific feedback (general improvement)."""
        mock_response = {
            "dimensions": [
                {
                    "name": "factual_correctness",
                    "description": "Evaluates factual accuracy and precision",
                    "grading_type": "binary"
                }
            ],
            "criteria": [
                {
                    "name": "fact_1",
                    "category": "Output",
                    "weight": 3,
                    "dimension": "factual_correctness",
                    "criterion": "The answer must correctly and precisely identify the capital city with proper spelling."
                }
            ]
        }
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        refined_rubric = generator.refine_rubric(existing_rubric)
        
        assert isinstance(refined_rubric, Rubric)
        # Should improve quality without changing structure
        assert len(refined_rubric.dimensions) == 1
        assert len(refined_rubric.criteria) == 1


class TestJSONRepair:
    """Test JSON repair functionality."""
    
    def test_repair_trailing_comma_in_array(self):
        """Test removing trailing comma from array."""
        invalid_json = '[1, 2, 3,]'
        repaired = repair_json(invalid_json)
        assert json.loads(repaired) == [1, 2, 3]
    
    def test_repair_trailing_comma_in_object(self):
        """Test removing trailing comma from object."""
        invalid_json = '{"name": "test", "value": 1,}'
        repaired = repair_json(invalid_json)
        assert json.loads(repaired) == {"name": "test", "value": 1}
    
    def test_repair_unquoted_keys(self):
        """Test fixing unquoted object keys."""
        invalid_json = '{name: "test", value: 1}'
        repaired = repair_json(invalid_json)
        assert json.loads(repaired) == {"name": "test", "value": 1}
    
    def test_repair_single_line_comments(self):
        """Test removing single-line comments."""
        invalid_json = '''{
  "name": "test", // This is a comment
  "value": 1
}'''
        repaired = repair_json(invalid_json)
        assert json.loads(repaired) == {"name": "test", "value": 1}
    
    def test_repair_multiline_comments(self):
        """Test removing multiline comments."""
        invalid_json = '''{
  "name": "test", /* This is a
  multiline comment */
  "value": 1
}'''
        repaired = repair_json(invalid_json)
        assert json.loads(repaired) == {"name": "test", "value": 1}
    
    def test_repair_multiple_issues(self):
        """Test repairing JSON with multiple issues."""
        invalid_json = '''{
  name: "test", // comment
  value: 1,
}'''
        repaired = repair_json(invalid_json)
        assert json.loads(repaired) == {"name": "test", "value": 1}
    
    def test_repair_nested_objects(self):
        """Test repairing nested objects with issues."""
        invalid_json = '''{
  name: "test",
  nested: {
    key: "value",
  },
}'''
        repaired = repair_json(invalid_json)
        expected = {"name": "test", "nested": {"key": "value"}}
        assert json.loads(repaired) == expected
    
    def test_repair_preserves_valid_json(self):
        """Test that valid JSON is not modified."""
        valid_json = '{"name": "test", "value": 1}'
        repaired = repair_json(valid_json)
        assert json.loads(repaired) == {"name": "test", "value": 1}


class TestChatSessionParsing:
    """Test chat session parsing."""
    
    def test_parse_chat_session_reads_content(self, tmp_path):
        """Test parsing chat session reads raw content."""
        session_file = tmp_path / "session.txt"
        content = """# Session Export

### User:
can you give me a summary of my system?

---

### Assistant:
The system is running Fedora Linux 42.
"""
        session_file.write_text(content)
        
        chat_input = parse_chat_session(str(session_file))
        
        assert chat_input.content == content.strip()
        assert chat_input.context is None
    
    def test_parse_chat_session_empty_raises_error(self, tmp_path):
        """Test that empty chat session raises ValueError."""
        session_file = tmp_path / "empty.txt"
        session_file.write_text("")
        
        with pytest.raises(ValueError, match="Chat session file is empty"):
            parse_chat_session(str(session_file))


class TestChatSessionRubricGeneration:
    """Test rubric generation from chat sessions."""
    
    @pytest.fixture
    def generator(self):
        """Create a RubricGenerator instance."""
        return RubricGenerator(api_key="test-key", model="gpt-4")
    
    @pytest.fixture
    def chat_session_input(self):
        """Sample chat session input for testing."""
        return ChatSessionInput(
            content="""### User:
can you give me a summary of my system?

### Assistant:
#### Tool Call: get_system_information

### Assistant:
The system is running Fedora Linux 42 with 8 CPUs."""
        )
    
    def test_generate_dimensions_from_chat_session(self, generator, chat_session_input, monkeypatch):
        """Test generating dimensions from chat session."""
        mock_response = [
            {
                "name": "tool_usage",
                "description": "Evaluates correct tool usage",
                "grading_type": "binary"
            },
            {
                "name": "output_quality",
                "description": "Evaluates output quality",
                "grading_type": "score",
                "scores": {1: "Poor", 2: "Good", 3: "Excellent"}
            }
        ]
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        dimensions = generator.generate_dimensions_from_chat(
            chat_session_input,
            num_dimensions=2
        )
        
        assert len(dimensions) == 2
        assert all(isinstance(d, Dimension) for d in dimensions)
        assert dimensions[0].name == "tool_usage"
    
    def test_generate_rubric_from_chat_full_workflow(self, generator, chat_session_input, monkeypatch):
        """Test full rubric generation from chat session."""
        mock_dimensions = [
            {
                "name": "tool_usage",
                "description": "Evaluates correct tool usage",
                "grading_type": "binary"
            }
        ]
        
        mock_criteria = [
            {
                "name": "tool_order_1",
                "category": "Tools",
                "weight": 3,
                "dimension": "tool_usage",
                "criterion": "Tools must be called in correct order",
                "tool_calls": {
                    "respect_order": True,
                    "required": [
                        {"name": "get_system_information", "min_calls": 1, "max_calls": 1}
                    ],
                    "optional": [],
                    "prohibited": []
                }
            }
        ]
        
        call_count = [0]
        
        def mock_llm_call(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_dimensions
            else:
                return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        rubric = generator.generate_rubric_from_chat(
            chat_session_input,
            num_dimensions=1,
            num_criteria=1
        )
        
        assert isinstance(rubric, Rubric)
        assert len(rubric.dimensions) == 1
        assert len(rubric.criteria) == 1
