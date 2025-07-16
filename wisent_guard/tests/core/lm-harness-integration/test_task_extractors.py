"""
Simple focused tests for task extractors that mirror the CLI command:
'python -m wisent_guard tasks <task> --model MockModel --layer 5 --limit 10'

These tests focus on the core functionality without complex mocking.
"""

import pytest
from unittest.mock import Mock
import os

from wisent_guard.core.benchmark_extractors import get_extractor, MBPPExtractor, GSM8KExtractor
from wisent_guard.core.activation_collection_method import ActivationCollectionLogic, PromptConstructionStrategy


class MockModel:
    """Simple mock model for testing."""
    
    def __init__(self, model_name: str = "MockModel"):
        self.model_name = model_name
        self.device = "cpu"
        self.config = Mock()
        self.config.hidden_size = 768
        self.tokenizer = Mock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        
    def format_prompt(self, prompt: str):
        """Mock prompt formatting."""
        return f"<|user|>\n{prompt}\n<|assistant|>\n"


@pytest.mark.mbpp
class TestMBPPExtractor:
    """Test MBPP extractor functionality."""
    
    def test_mbpp_extractor_available(self):
        """Test that MBPP extractor is available."""
        extractor = get_extractor('mbpp')
        assert extractor is not None
        assert isinstance(extractor, MBPPExtractor)
    
    def test_mbpp_qa_pair_extraction(self):
        """Test QA pair extraction from MBPP document."""
        extractor = MBPPExtractor()
        sample_doc = {
            'task_id': 1,
            'text': 'Write a function to find the maximum of two numbers',
            'code': 'def max_two_numbers(a, b):\n    return max(a, b)',
            'test_list': ['assert max_two_numbers(1, 2) == 2']
        }
        
        result = extractor.extract_qa_pair(sample_doc)
        
        assert result is not None
        assert 'question' in result
        assert 'formatted_question' in result
        assert 'correct_answer' in result
        assert result['question'] == 'Write a function to find the maximum of two numbers'
        assert 'def max_two_numbers' in result['correct_answer']
    
    def test_mbpp_contrastive_pair_extraction(self):
        """Test contrastive pair extraction from MBPP document."""
        extractor = MBPPExtractor()
        sample_doc = {
            'task_id': 1,
            'text': 'Write a function to find the maximum of two numbers',
            'code': 'def max_two_numbers(a, b):\n    return max(a, b)',
            'test_list': ['assert max_two_numbers(1, 2) == 2']
        }
        
        result = extractor.extract_contrastive_pair(sample_doc)
        
        assert result is not None
        assert 'question' in result
        assert 'correct_answer' in result
        assert 'incorrect_answer' in result
        
        # Check that incorrect answer is different from correct
        assert result['correct_answer'] != result['incorrect_answer']
        
        # Check that incorrect answer is based on the correct one (word removal)
        correct_words = set(result['correct_answer'].split())
        incorrect_words = set(result['incorrect_answer'].split())
        assert len(incorrect_words) <= len(correct_words)
    
    def test_mbpp_incorrect_code_generation(self):
        """Test the incorrect code generation method."""
        extractor = MBPPExtractor()
        correct_code = "def solution(x, y):\n    return x + y"
        
        incorrect_code = extractor._create_incorrect_code(correct_code)
        
        assert incorrect_code is not None
        assert isinstance(incorrect_code, str)
        assert incorrect_code != correct_code
        
        # Should have fewer or equal words than original
        assert len(incorrect_code.split()) <= len(correct_code.split())


@pytest.mark.gsm8k
class TestGSM8KExtractor:
    """Test GSM8K extractor functionality."""
    
    def test_gsm8k_extractor_available(self):
        """Test that GSM8K extractor is available."""
        extractor = get_extractor('gsm8k')
        assert extractor is not None
        assert isinstance(extractor, GSM8KExtractor)
    
    def test_gsm8k_qa_pair_extraction(self):
        """Test QA pair extraction from GSM8K document."""
        extractor = GSM8KExtractor()
        sample_doc = {
            'question': 'Janet has 3 apples. She gives 1 to her friend. How many apples does Janet have left?',
            'answer': 'Janet starts with 3 apples. She gives away 1 apple. So she has 3 - 1 = 2 apples left.\n#### 2'
        }
        
        result = extractor.extract_qa_pair(sample_doc)
        
        assert result is not None
        assert 'question' in result
        assert 'formatted_question' in result
        assert 'correct_answer' in result
        assert result['question'] == 'Janet has 3 apples. She gives 1 to her friend. How many apples does Janet have left?'
        assert result['correct_answer'] == '2'
    
    def test_gsm8k_contrastive_pair_extraction(self):
        """Test contrastive pair extraction from GSM8K document."""
        extractor = GSM8KExtractor()
        sample_doc = {
            'question': 'Janet has 3 apples. She gives 1 to her friend. How many apples does Janet have left?',
            'answer': 'Janet starts with 3 apples. She gives away 1 apple. So she has 3 - 1 = 2 apples left.\n#### 2'
        }
        
        result = extractor.extract_contrastive_pair(sample_doc)
        
        assert result is not None
        assert 'question' in result
        assert 'correct_answer' in result
        assert 'incorrect_answer' in result
        
        # Check that incorrect answer is different from correct
        assert result['correct_answer'] != result['incorrect_answer']
        
        # Check that the correct answer is properly extracted (numerical)
        assert result['correct_answer'] == '2'
    
    def test_gsm8k_numerical_answer_extraction(self):
        """Test numerical answer extraction from GSM8K format."""
        extractor = GSM8KExtractor()
        
        # Test with #### format
        sample_doc_with_marker = {
            'question': 'What is 2+2?',
            'answer': 'Some explanation here.\n#### 42'
        }
        result = extractor.extract_qa_pair(sample_doc_with_marker)
        assert result is not None
        assert result['correct_answer'] == '42'
        
        # Test without #### format
        sample_doc_without_marker = {
            'question': 'What is 2+2?',
            'answer': 'The answer is 24'
        }
        result = extractor.extract_qa_pair(sample_doc_without_marker)
        assert result is not None
        assert result['correct_answer'] == 'The answer is 24'


@pytest.mark.unit
class TestActivationCollectionWithMock:
    """Test activation collection with MockModel."""
    
    def test_contrastive_pair_creation_mbpp(self):
        """Test creating contrastive pairs for MBPP-like data."""
        mock_model = MockModel()
        
        mbpp_qa_pairs = [
            {
                'question': 'Write a function to find the maximum of two numbers',
                'correct_answer': 'def max_two_numbers(a, b):\n    return max(a, b)',
                'incorrect_answer': 'def max_two_numbers(a, b):\n    return(a, b)'
            }
        ]
        
        collector = ActivationCollectionLogic(model=mock_model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(
            mbpp_qa_pairs, 
            PromptConstructionStrategy.MULTIPLE_CHOICE
        )
        
        assert len(contrastive_pairs) == 1
        
        pair = contrastive_pairs[0]
        assert pair.prompt is not None
        assert pair.positive_response is not None
        assert pair.negative_response is not None
        assert 'python' in pair.prompt.lower() or 'function' in pair.prompt.lower()
    
    def test_contrastive_pair_creation_gsm8k(self):
        """Test creating contrastive pairs for GSM8K-like data."""
        mock_model = MockModel()
        
        gsm8k_qa_pairs = [
            {
                'question': 'Janet has 3 apples. She gives 1 to her friend. How many apples does Janet have left?',
                'correct_answer': '2',
                'incorrect_answer': '4'
            }
        ]
        
        collector = ActivationCollectionLogic(model=mock_model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(
            gsm8k_qa_pairs, 
            PromptConstructionStrategy.MULTIPLE_CHOICE
        )
        
        assert len(contrastive_pairs) == 1
        
        pair = contrastive_pairs[0]
        assert pair.prompt is not None
        assert pair.positive_response is not None
        assert pair.negative_response is not None
        assert 'better' in pair.prompt.lower()


@pytest.mark.integration
class TestCLICommandParameters:
    """Test that parameters match CLI command expectations."""
    
    def test_cli_parameter_validation(self):
        """Test CLI parameter validation for the command:
        python -m wisent_guard tasks <task> --model MockModel --layer 5 --limit 10
        """
        # Test that key parameters are accessible
        expected_params = {
            'model_name': 'MockModel',
            'layer': '5',
            'limit': 10,
            'task_names': ['mbpp', 'gsm8k']
        }
        
        # Verify that extractors exist for expected tasks
        for task_name in expected_params['task_names']:
            extractor = get_extractor(task_name)
            assert extractor is not None
        
        # Verify MockModel can be instantiated
        mock_model = MockModel(expected_params['model_name'])
        assert mock_model.model_name == expected_params['model_name']
        assert mock_model.device == 'cpu'
        assert hasattr(mock_model, 'format_prompt')
    
    def test_environment_requirements(self):
        """Test that environment requirements are met."""
        # For MBPP, we need HF_ALLOW_CODE_EVAL=1
        # This is set by the conftest.py fixture
        assert os.environ.get('HF_ALLOW_CODE_EVAL') == '1'
        
        # Test that environment supports the tasks
        assert 'HF_ALLOW_CODE_EVAL' in os.environ


@pytest.mark.integration
class TestTaskExtensibility:
    """Test that the framework can handle new tasks."""
    
    def test_generic_extractor_pattern(self):
        """Test that the extractor pattern works for any task."""
        # Test that we can create a generic extractor
        class GenericExtractor:
            def extract_qa_pair(self, doc, task_data=None):
                return {
                    'question': doc.get('question', 'Generic question'),
                    'formatted_question': doc.get('question', 'Generic question'),
                    'correct_answer': doc.get('answer', 'Generic answer')
                }
            
            def extract_contrastive_pair(self, doc, task_data=None):
                qa_pair = self.extract_qa_pair(doc, task_data)
                qa_pair['incorrect_answer'] = 'Generic incorrect answer'
                return qa_pair
        
        extractor = GenericExtractor()
        
        # Test with generic document
        generic_doc = {
            'question': 'What is 2+2?',
            'answer': '4'
        }
        
        qa_pair = extractor.extract_qa_pair(generic_doc)
        assert qa_pair is not None
        assert qa_pair['question'] == 'What is 2+2?'
        assert qa_pair['correct_answer'] == '4'
        
        contrastive_pair = extractor.extract_contrastive_pair(generic_doc)
        assert contrastive_pair is not None
        assert 'incorrect_answer' in contrastive_pair
        assert contrastive_pair['incorrect_answer'] == 'Generic incorrect answer'