"""Tests for rubric generator."""

import json
import pytest
from pathlib import Path
from rubric_kit.generator import (
    RubricGenerator, QAInput, parse_qa_input, repair_json,
    ChatSessionInput, parse_chat_session, _normalize_tool_spec,
    parse_dimensions_file
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
        
        criteria, variables = generator.generate_criteria(simple_qa, dimensions, num_criteria=1)
        
        assert len(criteria) == 1
        assert all(isinstance(c, Criterion) for c in criteria)
        assert criteria[0].name == "capital_fact_1"
        assert criteria[0].category == "Output"
        assert criteria[0].weight == 3
        assert variables is None  # Old format doesn't have variables
    
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
        
        called_with_call_type = []
        
        def mock_llm_call(*args, **kwargs):
            # Capture the call_type passed to LLM
            if len(args) >= 2:
                called_with_call_type.append(args[1])  # call_type is second positional arg
            elif 'call_type' in kwargs:
                called_with_call_type.append(kwargs['call_type'])
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        criteria, variables = generator.generate_criteria(
            simple_qa,
            dimensions,
            num_criteria=1,
            category_hints=["Accuracy", "Completeness"]
        )
        
        assert len(criteria) == 1
        assert criteria[0].category == "Accuracy"
        # Verify call_type was passed to LLM
        assert len(called_with_call_type) > 0
        assert called_with_call_type[0] == "generate_criteria"
        assert variables is None  # Old format doesn't have variables
    
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


class TestGenerateRubricWithProvidedDimensions:
    """Test rubric generation with pre-defined dimensions."""
    
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
    
    @pytest.fixture
    def provided_dimensions(self):
        """Pre-defined dimensions for testing."""
        return [
            Dimension(
                name="factual_accuracy",
                description="Evaluates whether stated facts are correct",
                grading_type="binary"
            ),
            Dimension(
                name="completeness",
                description="Evaluates whether all requested information is provided",
                grading_type="score",
                scores={0: "None", 1: "Partial", 2: "Most", 3: "Complete"}
            )
        ]
    
    def test_generate_rubric_with_provided_dimensions(
        self, generator, simple_qa, provided_dimensions, monkeypatch
    ):
        """Test generating rubric using provided dimensions (skip dimension generation)."""
        # Mock criteria generation only (dimensions should not be generated)
        mock_criteria = [
            {
                "name": "capital_fact",
                "category": "Accuracy",
                "weight": 3,
                "dimension": "factual_accuracy",
                "criterion": "The answer correctly identifies Paris as the capital."
            }
        ]
        
        call_count = [0]
        
        def mock_llm_call(*args, **kwargs):
            call_count[0] += 1
            return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        rubric = generator.generate_rubric(
            simple_qa,
            dimensions=provided_dimensions,
            num_criteria=1
        )
        
        assert isinstance(rubric, Rubric)
        # Should use provided dimensions, not generate new ones
        assert len(rubric.dimensions) == 2
        assert rubric.dimensions[0].name == "factual_accuracy"
        assert rubric.dimensions[1].name == "completeness"
        # Only ONE LLM call for criteria (not two for dimensions + criteria)
        assert call_count[0] == 1
    
    def test_generate_rubric_from_chat_with_provided_dimensions(
        self, generator, chat_session_input, provided_dimensions, monkeypatch
    ):
        """Test generating rubric from chat using provided dimensions."""
        mock_criteria = [
            {
                "name": "tool_usage",
                "category": "Tools",
                "weight": 3,
                "dimension": "factual_accuracy",
                "criterion": "Correctly used system information tool."
            }
        ]
        
        call_count = [0]
        
        def mock_llm_call(*args, **kwargs):
            call_count[0] += 1
            return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        rubric = generator.generate_rubric_from_chat(
            chat_session_input,
            dimensions=provided_dimensions,
            num_criteria=1
        )
        
        assert isinstance(rubric, Rubric)
        assert len(rubric.dimensions) == 2
        assert rubric.dimensions[0].name == "factual_accuracy"
        # Only ONE LLM call for criteria
        assert call_count[0] == 1
    
    def test_generate_rubric_validates_criteria_against_provided_dimensions(
        self, generator, simple_qa, provided_dimensions, monkeypatch
    ):
        """Test that criteria referencing non-existent dimensions fail validation."""
        # Mock criteria that references a dimension not in provided_dimensions
        mock_criteria = [
            {
                "name": "bad_criterion",
                "category": "Output",
                "weight": 3,
                "dimension": "nonexistent_dimension",  # Not in provided_dimensions!
                "criterion": "This references wrong dimension"
            }
        ]
        
        def mock_llm_call(*args, **kwargs):
            return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        with pytest.raises(ValueError, match="references non-existent dimension"):
            generator.generate_rubric(
                simple_qa,
                dimensions=provided_dimensions,
                num_criteria=1
            )


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


class TestRefineWithProvidedDimensions:
    """Test rubric refinement with dimension merging."""
    
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
    
    @pytest.fixture
    def dimensions_to_merge(self):
        """Dimensions to merge into existing rubric."""
        return [
            Dimension(
                name="tool_usage",
                description="Evaluates correct tool usage",
                grading_type="binary"
            ),
            Dimension(
                name="clarity",
                description="Evaluates response clarity",
                grading_type="score",
                scores={0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}
            )
        ]
    
    def test_refine_rubric_merges_dimensions(
        self, generator, existing_rubric, dimensions_to_merge, monkeypatch
    ):
        """Test that refine_rubric merges provided dimensions."""
        # Mock LLM response that uses both old and new dimensions
        mock_response = {
            "dimensions": [
                {"name": "factual_correctness", "description": "Evaluates factual accuracy", "grading_type": "binary"},
                {"name": "tool_usage", "description": "Evaluates correct tool usage", "grading_type": "binary"},
                {"name": "clarity", "description": "Evaluates response clarity", "grading_type": "score",
                 "scores": {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}}
            ],
            "criteria": [
                {"name": "fact_1", "category": "Output", "weight": 3, "dimension": "factual_correctness",
                 "criterion": "The answer must be correct."}
            ]
        }
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        refined = generator.refine_rubric(
            existing_rubric,
            dimensions_to_merge=dimensions_to_merge
        )
        
        assert isinstance(refined, Rubric)
        # Should have merged dimensions (existing + new)
        dimension_names = {d.name for d in refined.dimensions}
        assert "factual_correctness" in dimension_names
        assert "tool_usage" in dimension_names
        assert "clarity" in dimension_names
    
    def test_refine_rubric_with_qa_merges_dimensions(
        self, generator, existing_rubric, dimensions_to_merge, monkeypatch
    ):
        """Test that refine_rubric_with_qa merges provided dimensions."""
        qa_input = QAInput(
            question="What is the capital?",
            answer="Paris is the capital.",
            context=None
        )
        
        mock_response = {
            "dimensions": [
                {"name": "factual_correctness", "description": "Evaluates factual accuracy", "grading_type": "binary"},
                {"name": "tool_usage", "description": "Evaluates tool usage", "grading_type": "binary"}
            ],
            "criteria": [
                {"name": "fact_1", "category": "Output", "weight": 3, "dimension": "factual_correctness",
                 "criterion": "The answer must be correct."}
            ]
        }
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        refined = generator.refine_rubric_with_qa(
            existing_rubric,
            qa_input,
            dimensions_to_merge=dimensions_to_merge
        )
        
        assert isinstance(refined, Rubric)
        dimension_names = {d.name for d in refined.dimensions}
        assert "tool_usage" in dimension_names
    
    def test_refine_rubric_with_chat_merges_dimensions(
        self, generator, existing_rubric, dimensions_to_merge, monkeypatch
    ):
        """Test that refine_rubric_with_chat merges provided dimensions."""
        chat_input = ChatSessionInput(
            content="### User: Hello\n### Assistant: Hi there!"
        )
        
        mock_response = {
            "dimensions": [
                {"name": "factual_correctness", "description": "Evaluates factual accuracy", "grading_type": "binary"},
                {"name": "clarity", "description": "Evaluates response clarity", "grading_type": "score",
                 "scores": {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}}
            ],
            "criteria": [
                {"name": "fact_1", "category": "Output", "weight": 3, "dimension": "factual_correctness",
                 "criterion": "The answer must be correct."}
            ]
        }
        
        def mock_llm_call(*args, **kwargs):
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        refined = generator.refine_rubric_with_chat(
            existing_rubric,
            chat_input,
            dimensions_to_merge=dimensions_to_merge
        )
        
        assert isinstance(refined, Rubric)
        dimension_names = {d.name for d in refined.dimensions}
        assert "clarity" in dimension_names


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


class TestNormalizeScores:
    """Test _normalize_scores handles various LLM response formats."""
    
    def test_normalize_scores_from_dict_with_string_keys(self):
        """Test normalizing scores dict with string keys."""
        from rubric_kit.generator import _normalize_scores
        
        scores = {"1": "Bad", "2": "Good", "3": "Great"}
        result = _normalize_scores(scores)
        
        assert result == {1: "Bad", 2: "Good", 3: "Great"}
        assert all(isinstance(k, int) for k in result.keys())
    
    def test_normalize_scores_from_dict_with_int_keys(self):
        """Test normalizing scores dict with int keys (passthrough)."""
        from rubric_kit.generator import _normalize_scores
        
        scores = {1: "Bad", 2: "Good", 3: "Great"}
        result = _normalize_scores(scores)
        
        assert result == {1: "Bad", 2: "Good", 3: "Great"}
    
    def test_normalize_scores_from_list_of_single_key_dicts(self):
        """Test normalizing scores from list of single-key dicts."""
        from rubric_kit.generator import _normalize_scores
        
        # LLM sometimes returns: [{1: "Bad"}, {2: "Good"}, {3: "Great"}]
        scores = [{1: "Bad"}, {2: "Good"}, {3: "Great"}]
        result = _normalize_scores(scores)
        
        assert result == {1: "Bad", 2: "Good", 3: "Great"}
    
    def test_normalize_scores_from_list_of_pairs(self):
        """Test normalizing scores from list of pairs."""
        from rubric_kit.generator import _normalize_scores
        
        # LLM might return: [[1, "Bad"], [2, "Good"], [3, "Great"]]
        scores = [[1, "Bad"], [2, "Good"], [3, "Great"]]
        result = _normalize_scores(scores)
        
        assert result == {1: "Bad", 2: "Good", 3: "Great"}
    
    def test_normalize_scores_none(self):
        """Test normalizing None scores returns None."""
        from rubric_kit.generator import _normalize_scores
        
        assert _normalize_scores(None) is None
    
    def test_normalize_scores_with_string_keys_in_list(self):
        """Test normalizing scores list with string keys."""
        from rubric_kit.generator import _normalize_scores
        
        scores = [{"1": "Bad"}, {"2": "Good"}, {"3": "Great"}]
        result = _normalize_scores(scores)
        
        assert result == {1: "Bad", 2: "Good", 3: "Great"}
    
    def test_normalize_scores_skips_invalid_keys(self):
        """Test that non-integer keys like 'score' are skipped."""
        from rubric_kit.generator import _normalize_scores
        
        # LLM might return malformed structure with non-integer keys
        scores = {"score": 1, "description": "Bad", "1": "Poor", "2": "Good"}
        result = _normalize_scores(scores)
        
        # Should only keep valid integer keys
        assert result == {1: "Poor", 2: "Good"}
    
    def test_normalize_scores_returns_none_for_all_invalid(self):
        """Test that all-invalid keys returns None."""
        from rubric_kit.generator import _normalize_scores
        
        scores = {"score": 1, "description": "Bad"}
        result = _normalize_scores(scores)
        
        assert result is None


class TestNormalizeToolSpec:
    """Test tool spec normalization from various LLM response formats."""
    
    def test_normalize_standard_format(self):
        """Test that standard format passes through unchanged."""
        spec = {"name": "get_system_info", "min_calls": 1, "params": {"host": "server01"}}
        result = _normalize_tool_spec(spec)
        
        assert result == {"name": "get_system_info", "min_calls": 1, "params": {"host": "server01"}}
    
    def test_normalize_tool_name_alias(self):
        """Test that 'tool_name' is normalized to 'name'."""
        spec = {"tool_name": "get_system_info", "min_calls": 1}
        result = _normalize_tool_spec(spec)
        
        assert result["name"] == "get_system_info"
        assert result["min_calls"] == 1
        assert "tool_name" not in result
    
    def test_normalize_tool_name_as_key_with_params(self):
        """Test format where tool name is the dict key with params as value."""
        # This is what the LLM returned: {"linux_diagnostics.get_system_info": {"params": {"host": "{{host}}"}}}
        spec = {"linux_diagnostics.get_system_info": {"params": {"host": "{{host}}"}}}
        result = _normalize_tool_spec(spec)
        
        assert result["name"] == "linux_diagnostics.get_system_info"
        assert result["params"] == {"host": "{{host}}"}
    
    def test_normalize_tool_name_as_key_with_min_calls(self):
        """Test format where tool name is the dict key with min_calls in value."""
        spec = {"get_memory_info": {"min_calls": 1, "max_calls": 2}}
        result = _normalize_tool_spec(spec)
        
        assert result["name"] == "get_memory_info"
        assert result["min_calls"] == 1
        assert result["max_calls"] == 2
    
    def test_normalize_tool_name_as_key_with_mixed_fields(self):
        """Test format where tool name is key with mixed fields in outer and inner dict."""
        spec = {"get_disk_info": {"params": {"host": "server01"}}, "min_calls": 1}
        result = _normalize_tool_spec(spec)
        
        assert result["name"] == "get_disk_info"
        assert result["params"] == {"host": "server01"}
        assert result["min_calls"] == 1
    
    def test_normalize_preserves_original(self):
        """Test that normalization doesn't modify the original dict."""
        spec = {"tool_name": "get_info", "min_calls": 1}
        original_spec = spec.copy()
        _normalize_tool_spec(spec)
        
        assert spec == original_spec


class TestGenerateRubricWithGuidelines:
    """Test rubric generation with guidelines parameter."""
    
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
    
    def test_generate_rubric_passes_guidelines(self, generator, simple_qa, monkeypatch):
        """Test that generate_rubric passes guidelines to generation methods."""
        mock_dimensions = [
            {
                "name": "factual_correctness",
                "description": "Evaluates factual accuracy",
                "grading_type": "binary"
            }
        ]
        
        mock_criteria = [
            {
                "name": "capital_fact",
                "category": "Accuracy",
                "weight": 3,
                "dimension": "factual_correctness",
                "criterion": "The answer correctly identifies Paris as the capital."
            }
        ]
        
        call_count = [0]
        prompts_received = []
        
        def mock_llm_call(prompt, *args, **kwargs):
            call_count[0] += 1
            prompts_received.append(prompt)
            if call_count[0] == 1:
                return mock_dimensions
            else:
                return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        guidelines = "Focus on geographical accuracy. Check for specific city names."
        rubric = generator.generate_rubric(
            simple_qa,
            num_dimensions=1,
            num_criteria=1,
            guidelines=guidelines
        )
        
        # Guidelines should appear in both prompts (dimensions and criteria)
        assert len(prompts_received) == 2
        assert guidelines in prompts_received[0]  # Dimension generation prompt
        assert guidelines in prompts_received[1]  # Criteria generation prompt
    
    def test_generate_rubric_from_chat_passes_guidelines(
        self, generator, chat_session_input, monkeypatch
    ):
        """Test that generate_rubric_from_chat passes guidelines to generation methods."""
        mock_dimensions = [
            {
                "name": "tool_usage",
                "description": "Evaluates tool usage",
                "grading_type": "binary"
            }
        ]
        
        mock_criteria = [
            {
                "name": "tool_called",
                "category": "Tools",
                "weight": 3,
                "dimension": "tool_usage",
                "criterion": "Tool was called correctly."
            }
        ]
        
        call_count = [0]
        prompts_received = []
        
        def mock_llm_call(prompt, *args, **kwargs):
            call_count[0] += 1
            prompts_received.append(prompt)
            if call_count[0] == 1:
                return mock_dimensions
            else:
                return mock_criteria
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        guidelines = "Evaluate tool usage patterns. Create atomic criteria."
        rubric = generator.generate_rubric_from_chat(
            chat_session_input,
            num_dimensions=1,
            num_criteria=1,
            guidelines=guidelines
        )
        
        # Guidelines should appear in both prompts
        assert len(prompts_received) == 2
        assert guidelines in prompts_received[0]  # Dimension generation prompt
        assert guidelines in prompts_received[1]  # Criteria generation prompt
    
    def test_generate_dimensions_passes_guidelines(self, generator, simple_qa, monkeypatch):
        """Test that generate_dimensions includes guidelines in prompt."""
        mock_response = [
            {
                "name": "factual_correctness",
                "description": "Evaluates factual accuracy",
                "grading_type": "binary"
            }
        ]
        
        prompt_received = []
        
        def mock_llm_call(prompt, *args, **kwargs):
            prompt_received.append(prompt)
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        guidelines = "Create dimensions for accuracy and completeness."
        dimensions = generator.generate_dimensions(
            simple_qa,
            num_dimensions=1,
            guidelines=guidelines
        )
        
        assert len(prompt_received) == 1
        assert guidelines in prompt_received[0]
    
    def test_generate_criteria_passes_guidelines(self, generator, simple_qa, monkeypatch):
        """Test that generate_criteria includes guidelines in prompt."""
        dimensions = [
            Dimension(
                name="factual_correctness",
                description="Evaluates factual accuracy",
                grading_type="binary"
            )
        ]
        
        mock_response = [
            {
                "name": "capital_fact",
                "category": "Accuracy",
                "weight": 3,
                "dimension": "factual_correctness",
                "criterion": "Check the capital city."
            }
        ]
        
        prompt_received = []
        
        def mock_llm_call(prompt, *args, **kwargs):
            prompt_received.append(prompt)
            return mock_response
        
        monkeypatch.setattr(generator, "_call_llm", mock_llm_call)
        
        guidelines = "Each criterion should check exactly one fact."
        criteria, _ = generator.generate_criteria(
            simple_qa,
            dimensions,
            num_criteria=1,
            guidelines=guidelines
        )
        
        assert len(prompt_received) == 1
        assert guidelines in prompt_received[0]


class TestParseDimensionsFile:
    """Test parsing dimensions from YAML file."""
    
    def test_parse_dimensions_file_valid(self, tmp_path):
        """Test parsing valid dimensions YAML file."""
        dims_file = tmp_path / "dimensions.yaml"
        dims_file.write_text("""
dimensions:
  - name: factual_accuracy
    description: "Evaluates whether stated facts are correct"
    grading_type: binary

  - name: completeness
    description: "Evaluates whether all requested information is provided"
    grading_type: score
    scores:
      0: "No relevant information provided"
      1: "Missing most key information"
      2: "Partially complete"
      3: "Complete and comprehensive"
""")
        
        dimensions = parse_dimensions_file(str(dims_file))
        
        assert len(dimensions) == 2
        assert all(isinstance(d, Dimension) for d in dimensions)
        assert dimensions[0].name == "factual_accuracy"
        assert dimensions[0].grading_type == "binary"
        assert dimensions[1].name == "completeness"
        assert dimensions[1].grading_type == "score"
        assert dimensions[1].scores == {
            0: "No relevant information provided",
            1: "Missing most key information",
            2: "Partially complete",
            3: "Complete and comprehensive"
        }
    
    def test_parse_dimensions_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Dimensions file not found"):
            parse_dimensions_file("/nonexistent/path/dimensions.yaml")
    
    def test_parse_dimensions_file_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises ValueError."""
        dims_file = tmp_path / "bad.yaml"
        dims_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            parse_dimensions_file(str(dims_file))
    
    def test_parse_dimensions_file_missing_dimensions_key(self, tmp_path):
        """Test that missing 'dimensions' key raises ValueError."""
        dims_file = tmp_path / "no_key.yaml"
        dims_file.write_text("""
some_other_key:
  - name: test
""")
        
        with pytest.raises(ValueError, match="must contain a 'dimensions' key"):
            parse_dimensions_file(str(dims_file))
    
    def test_parse_dimensions_file_empty_dimensions(self, tmp_path):
        """Test that empty dimensions list raises ValueError."""
        dims_file = tmp_path / "empty.yaml"
        dims_file.write_text("dimensions: []")
        
        with pytest.raises(ValueError, match="must contain at least one dimension"):
            parse_dimensions_file(str(dims_file))
    
    def test_parse_dimensions_file_missing_required_fields(self, tmp_path):
        """Test that missing required fields raises ValueError."""
        dims_file = tmp_path / "incomplete.yaml"
        dims_file.write_text("""
dimensions:
  - name: test_dim
    description: "Missing grading_type"
""")
        
        with pytest.raises(ValueError):
            parse_dimensions_file(str(dims_file))
    
    def test_parse_dimensions_file_score_without_scores(self, tmp_path):
        """Test that score type without scores dict raises ValueError."""
        dims_file = tmp_path / "bad_score.yaml"
        dims_file.write_text("""
dimensions:
  - name: test_dim
    description: "Score type but no scores"
    grading_type: score
""")
        
        with pytest.raises(ValueError, match="must have scores defined"):
            parse_dimensions_file(str(dims_file))
