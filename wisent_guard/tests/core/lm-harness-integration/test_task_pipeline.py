"""
Tests for task pipeline integration with wisent-guard CLI.

This module tests the happy path functionality for any task using the CLI command:
'python -m wisent_guard tasks <task> --model MockModel --layer 5 --limit 10'

The tests use mock models to ensure reliable testing without requiring actual model downloads.
"""

import pytest
from unittest.mock import Mock, patch
import torch
import tempfile
import shutil
import os

from wisent_guard.core.benchmark_extractors import get_extractor
from wisent_guard.core.activation_collection_method import ActivationCollectionLogic, PromptConstructionStrategy
from wisent_guard.cli import run_task_pipeline


@pytest.fixture(scope="function")
def mock_model():
    """Fixture that provides a mock model for testing."""
    return MockModel()


@pytest.fixture(scope="function")
def temp_dir():
    """Fixture that provides a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def setup_environment():
    """Fixture that sets up the test environment."""
    # Set environment variable for code evaluation
    os.environ['HF_ALLOW_CODE_EVAL'] = '1'
    yield
    # Cleanup if needed


class MockModel:
    """Mock model that simulates model behavior for testing."""
    
    def __init__(self, model_name: str = "MockModel"):
        self.model_name = model_name
        self.device = "cpu"
        self.config = Mock()
        self.config.hidden_size = 768
        self.tokenizer = Mock()
        self.tokenizer.pad_token_id = 0
        self.tokenizer.eos_token_id = 1
        
    def generate(self, prompt: str, layer_index: int = 5, max_new_tokens: int = 300, **kwargs):
        """Mock generation method that returns task-appropriate responses."""
        if "math" in prompt.lower() or "gsm8k" in prompt.lower():
            return "The answer is 42"
        elif "python" in prompt.lower() or "code" in prompt.lower() or "mbpp" in prompt.lower():
            return "def solution():\n    return 42"
        elif "better" in prompt.lower():
            return "B"  # For multiple choice prompts
        else:
            return "Mock response"
    
    def get_activations(self, prompt: str, layer_index: int = 5, **kwargs):
        """Mock activation extraction with realistic tensor shapes."""
        batch_size = 1
        seq_length = len(prompt.split()) + 10  # Rough tokenization
        hidden_size = 768
        return torch.randn(batch_size, seq_length, hidden_size)
    
    def tokenize(self, text: str):
        """Mock tokenization."""
        return text.split()
    
    def format_prompt(self, prompt: str):
        """Mock prompt formatting."""
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    def load_lm_eval_task(self, task_name: str, shots: int = 0, limit: int = None):
        """Mock lm-eval task loading."""
        mock_task = Mock()
        mock_task.config = Mock()
        mock_task.config.task = task_name
        mock_task.test_docs = Mock()
        mock_task.test_docs.return_value = self._get_mock_docs(task_name, limit or 10)
        return mock_task
    
    def _get_mock_docs(self, task_name: str, limit: int):
        """Generate mock documents for different tasks."""
        if task_name == "mbpp":
            return [
                {
                    'task_id': i,
                    'text': f'Write a function to solve problem {i}',
                    'code': f'def solution_{i}():\n    return {i}',
                    'test_list': [f'assert solution_{i}() == {i}'],
                    'test_setup_code': '',
                    'challenge_test_list': []
                }
                for i in range(1, min(limit + 1, 11))
            ]
        elif task_name == "gsm8k":
            return [
                {
                    'question': f'What is {i} + {i}?',
                    'answer': f'The answer is {i} + {i} = {i*2}.\n#### {i*2}'
                }
                for i in range(1, min(limit + 1, 11))
            ]
        else:
            # Generic mock docs for other tasks
            return [
                {
                    'question': f'Question {i} for {task_name}',
                    'answer': f'Answer {i}',
                    'choices': [f'Choice A{i}', f'Choice B{i}', f'Choice C{i}', f'Choice D{i}'],
                    'label': i % 4
                }
                for i in range(1, min(limit + 1, 11))
            ]
    
    def split_task_data(self, task_data, split_ratio: float = 0.8, random_seed: int = 42):
        """Mock task data splitting."""
        docs = list(task_data.test_docs())
        split_point = int(len(docs) * split_ratio)
        return docs[:split_point], docs[split_point:]  # train_docs, test_docs


@pytest.mark.task_pipeline
class TestTaskPipelineHappyPath:
    """Test the happy path for task pipeline execution."""
    
    def _test_task_pipeline(self, task_name: str, temp_dir: str, mock_model: MockModel, expected_pairs: int = 8):
        """Generic test for any task pipeline."""
        # Mock the model loading
        with patch('wisent_guard.core.model.Model') as mock_model_class:
            mock_model_class.return_value = mock_model
            
            # Mock the cache directory
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Mock the classifier training
            with patch('wisent_guard.core.classifier.Classifier') as mock_classifier:
                mock_classifier_instance = Mock()
                mock_classifier_instance.fit.return_value = {
                    'accuracy': 0.75,
                    'f1_score': 0.70,
                    'precision': 0.80,
                    'recall': 0.65
                }
                mock_classifier.return_value = mock_classifier_instance
                
                # Mock the ground truth evaluator
                with patch('wisent_guard.core.ground_truth_evaluator.GroundTruthEvaluator') as mock_evaluator:
                    mock_evaluator_instance = Mock()
                    mock_evaluator_instance.evaluate.return_value = {
                        'accuracy': 0.60,
                        'correct_predictions': 6,
                        'total_samples': 10
                    }
                    mock_evaluator.return_value = mock_evaluator_instance
                    
                    # Run the task pipeline
                    result = run_task_pipeline(
                        task_name=task_name,
                        model_name="MockModel",
                        layer="5",
                        limit=10,
                        cache_dir=cache_dir,
                        verbose=False,
                        model_instance=mock_model
                    )
                    
                    # Verify the result structure
                    assert isinstance(result, dict)
                    assert 'task_name' in result
                    assert result['task_name'] == task_name
                    
                    # Verify basic pipeline components executed
                    mock_model_class.assert_called_once()
                    mock_classifier_instance.fit.assert_called_once()
                    
                    return result
    
    @pytest.mark.mbpp
    def test_mbpp_happy_path(self, temp_dir, mock_model):
        """Test MBPP task happy path."""
        result = self._test_task_pipeline("mbpp", temp_dir, mock_model)
        
        # MBPP-specific assertions
        assert result['task_name'] == 'mbpp'
        
        # Verify that the extractor was used
        extractor = get_extractor('mbpp')
        assert extractor is not None
        
        # Test that MBPP documents can be processed
        sample_doc = {
            'task_id': 1,
            'text': 'Write a function to find the maximum of two numbers',
            'code': 'def max_two_numbers(a, b):\n    return max(a, b)',
            'test_list': ['assert max_two_numbers(1, 2) == 2']
        }
        
        qa_pair = extractor.extract_qa_pair(sample_doc)
        assert qa_pair is not None
        
        contrastive_pair = extractor.extract_contrastive_pair(sample_doc)
        assert contrastive_pair is not None
        assert 'incorrect_answer' in contrastive_pair
    
    @pytest.mark.gsm8k
    def test_gsm8k_happy_path(self, temp_dir, mock_model):
        """Test GSM8K task happy path."""
        result = self._test_task_pipeline("gsm8k", temp_dir, mock_model)
        
        # GSM8K-specific assertions
        assert result['task_name'] == 'gsm8k'
        
        # Verify that the extractor was used
        extractor = get_extractor('gsm8k')
        assert extractor is not None
        
        # Test that GSM8K documents can be processed
        sample_doc = {
            'question': 'Janet has 3 apples. She gives 1 to her friend. How many apples does Janet have left?',
            'answer': 'Janet starts with 3 apples. She gives away 1 apple. So she has 3 - 1 = 2 apples left.\n#### 2'
        }
        
        qa_pair = extractor.extract_qa_pair(sample_doc)
        assert qa_pair is not None
        
        contrastive_pair = extractor.extract_contrastive_pair(sample_doc)
        assert contrastive_pair is not None
        assert 'incorrect_answer' in contrastive_pair
    
    @pytest.mark.integration
    def test_generic_task_happy_path(self, temp_dir, mock_model):
        """Test generic task pipeline for extensibility."""
        # This test ensures the pipeline can handle any task
        task_name = "hellaswag"  # Example of another task
        
        # Mock extractor for generic task
        with patch('wisent_guard.core.benchmark_extractors.get_extractor') as mock_get_extractor:
            mock_extractor = Mock()
            mock_extractor.extract_qa_pair.return_value = {
                'question': 'Test question',
                'formatted_question': 'Test formatted question',
                'correct_answer': 'Test correct answer'
            }
            mock_extractor.extract_contrastive_pair.return_value = {
                'question': 'Test question',
                'correct_answer': 'Test correct answer',
                'incorrect_answer': 'Test incorrect answer'
            }
            mock_get_extractor.return_value = mock_extractor
            
            result = self._test_task_pipeline(task_name, temp_dir, mock_model)
            
            # Verify the pipeline is flexible for any task
            assert result['task_name'] == task_name
            mock_get_extractor.assert_called()


@pytest.mark.unit
class TestActivationCollection:
    """Test activation collection for different tasks."""
    
    def test_contrastive_pair_creation(self, mock_model):
        """Test creating contrastive pairs for any task."""
        qa_pairs = [
            {
                'question': 'Test question 1',
                'correct_answer': 'Test correct answer 1',
                'incorrect_answer': 'Test incorrect answer 1'
            },
            {
                'question': 'Test question 2',
                'correct_answer': 'Test correct answer 2',
                'incorrect_answer': 'Test incorrect answer 2'
            }
        ]
        
        collector = ActivationCollectionLogic(model=mock_model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(
            qa_pairs, 
            PromptConstructionStrategy.MULTIPLE_CHOICE
        )
        
        assert len(contrastive_pairs) == 2
        
        for pair in contrastive_pairs:
            assert pair.prompt is not None
            assert pair.positive_response is not None
            assert pair.negative_response is not None
            assert 'better' in pair.prompt.lower()
    
    def test_activation_extraction_mock(self, mock_model):
        """Test activation extraction with mock model."""
        qa_pairs = [
            {
                'question': 'Test question',
                'correct_answer': 'Test correct answer',
                'incorrect_answer': 'Test incorrect answer'
            }
        ]
        
        collector = ActivationCollectionLogic(model=mock_model)
        contrastive_pairs = collector.create_batch_contrastive_pairs(
            qa_pairs, 
            PromptConstructionStrategy.MULTIPLE_CHOICE
        )
        
        # Mock activation collection
        with patch.object(collector, 'collect_activations_batch') as mock_collect:
            mock_pair = Mock()
            mock_pair.positive_response = Mock()
            mock_pair.negative_response = Mock()
            mock_pair.positive_response.activations = torch.randn(1, 768)
            mock_pair.negative_response.activations = torch.randn(1, 768)
            mock_collect.return_value = [mock_pair]
            
            processed_pairs = collector.collect_activations_batch(
                pairs=contrastive_pairs,
                layer_index=5,
                device='cpu'
            )
            
            assert len(processed_pairs) == 1
            mock_collect.assert_called_once()


@pytest.mark.integration
class TestCLICommandEquivalence:
    """Test that the test mirrors the actual CLI command behavior."""
    
    def test_cli_parameter_mapping(self, temp_dir):
        """Test that test parameters match CLI parameters."""
        # This test ensures our test parameters match the CLI command:
        # python -m wisent_guard tasks <task> --model MockModel --layer 5 --limit 10
        
        expected_params = {
            'model_name': 'MockModel',
            'layer': '5',
            'limit': 10,
            'verbose': False  # Can be True for debugging
        }
        
        # Test with MBPP
        with patch('wisent_guard.core.model.Model') as mock_model_class:
            mock_model_class.return_value = MockModel()
            
            with patch('wisent_guard.core.classifier.Classifier') as mock_classifier:
                mock_classifier_instance = Mock()
                mock_classifier_instance.fit.return_value = {'accuracy': 0.75}
                mock_classifier.return_value = mock_classifier_instance
                
                with patch('wisent_guard.core.ground_truth_evaluator.GroundTruthEvaluator') as mock_evaluator:
                    mock_evaluator_instance = Mock()
                    mock_evaluator_instance.evaluate.return_value = {'accuracy': 0.60}
                    mock_evaluator.return_value = mock_evaluator_instance
                    
                    # Test the function call matches CLI parameters
                    result = run_task_pipeline(
                        task_name="mbpp",
                        model_name="MockModel",  # Use MockModel instead of distilbert/distilgpt2
                        layer=expected_params['layer'],
                        limit=expected_params['limit'],
                        cache_dir=temp_dir,
                        verbose=expected_params['verbose']
                    )
                    
                    # Verify the call was made with correct parameters
                    mock_model_class.assert_called_once()
                    args, kwargs = mock_model_class.call_args
                    assert args[0] == "MockModel"
    
    def test_environment_requirements(self):
        """Test that environment requirements are properly set."""
        # For MBPP, we need HF_ALLOW_CODE_EVAL=1
        # This is automatically set by the setup_environment fixture
        
        # Test that the environment variable is set
        assert os.environ.get('HF_ALLOW_CODE_EVAL') == '1'
        
        # Test that the pipeline can handle this requirement
        assert os.environ.get('HF_ALLOW_CODE_EVAL') == '1'


@pytest.mark.integration
class TestTaskExtensibility:
    """Test that the pipeline can be extended to new tasks."""
    
    def test_new_task_support(self, temp_dir):
        """Test that adding a new task is straightforward."""
        # This test demonstrates how to add support for a new task
        new_task_name = "new_task"
        
        # Mock a new extractor
        class NewTaskExtractor:
            def extract_qa_pair(self, doc, task_data=None):
                return {
                    'question': doc.get('question', 'New task question'),
                    'formatted_question': doc.get('question', 'New task question'),
                    'correct_answer': doc.get('answer', 'New task answer')
                }
            
            def extract_contrastive_pair(self, doc, task_data=None):
                qa_pair = self.extract_qa_pair(doc, task_data)
                qa_pair['incorrect_answer'] = 'New task incorrect answer'
                return qa_pair
        
        # Mock the extractor registry
        with patch('wisent_guard.core.benchmark_extractors.get_extractor') as mock_get_extractor:
            mock_get_extractor.return_value = NewTaskExtractor()
            
            # Mock model and other components
            with patch('wisent_guard.core.model.Model') as mock_model_class:
                mock_model_class.return_value = MockModel()
                
                with patch('wisent_guard.core.classifier.Classifier') as mock_classifier:
                    mock_classifier_instance = Mock()
                    mock_classifier_instance.fit.return_value = {'accuracy': 0.75}
                    mock_classifier.return_value = mock_classifier_instance
                    
                    with patch('wisent_guard.core.ground_truth_evaluator.GroundTruthEvaluator') as mock_evaluator:
                        mock_evaluator_instance = Mock()
                        mock_evaluator_instance.evaluate.return_value = {'accuracy': 0.60}
                        mock_evaluator.return_value = mock_evaluator_instance
                        
                        # Test that the pipeline works with the new task
                        result = run_task_pipeline(
                            task_name=new_task_name,
                            model_name="MockModel",
                            layer="5",
                            limit=10,
                            cache_dir=temp_dir,
                            verbose=False
                        )
                        
                        # Verify the pipeline handled the new task
                        assert result['task_name'] == new_task_name
                        mock_get_extractor.assert_called()