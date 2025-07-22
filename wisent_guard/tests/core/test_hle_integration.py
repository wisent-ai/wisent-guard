"""
HLE (Human-Level Evaluation) integration tests.

This module demonstrates how HLE benchmark works with WisentGuard:
- HLE supports both exactMatch (text generation) and multipleChoice (A/B/C/D/E) questions
- Uses TaskInterface pattern for non-lm-eval benchmarks
- Filters out multimodal examples (images) for text-only implementation
- Integrates with existing classifier training and evaluation pipeline
"""

import pytest

# Ensure task registration happens
from wisent_guard.core import tasks


class TestHLEBenchmark:
    """Essential HLE functionality tests that serve as documentation."""

    def test_hle_task_variants(self):
        """
        HLE provides three task variants for different use cases:
        
        - hle: All questions (both exactMatch and multipleChoice)  
        - hle_exact_match: Only text generation questions
        - hle_multiple_choice: Only A/B/C/D/E questions
        """
        from wisent_guard.core.task_interface import list_tasks, get_task
        
        available_tasks = list_tasks()
        hle_variants = [t for t in available_tasks if t.startswith('hle')]
        
        # All three variants should be available
        assert 'hle' in hle_variants
        assert 'hle_exact_match' in hle_variants
        assert 'hle_multiple_choice' in hle_variants
        
        # Test task instantiation and filtering
        exact_task = get_task('hle_exact_match')
        assert exact_task.answer_type_filter == 'exactMatch'
        
        mc_task = get_task('hle_multiple_choice') 
        assert mc_task.answer_type_filter == 'multipleChoice'

    def test_hle_data_structure(self):
        """
        HLE data structure supports both answer types:
        
        exactMatch: Free-form text requiring exact string matching
        multipleChoice: A/B/C/D/E with choices embedded in question text
        """
        from wisent_guard.core.task_interface import get_task
        
        hle_task = get_task('hle')
        data = hle_task.load_data(limit=5)
        
        # Should contain both answer types
        answer_types = {sample['answer_type'] for sample in data}
        expected_types = {'exactMatch', 'multipleChoice'}
        assert answer_types.intersection(expected_types), f"Expected types in {expected_types}, got {answer_types}"
        
        # Check data structure for each type
        for sample in data:
            assert 'question' in sample
            assert 'answer' in sample
            assert 'category' in sample
            
            if sample['answer_type'] == 'multipleChoice':
                # Multiple choice should have letter answer (A/B/C/D/E)
                assert sample['answer'] in ['A', 'B', 'C', 'D', 'E']
                # Choices should be embedded in question text
                assert any(letter in sample['question'] for letter in ['A.', 'B.', 'C.', 'D.', 'E.'])

    def test_hle_extraction_pipeline(self):
        """
        HLE extractor handles dual answer types and creates contrastive pairs:
        
        - multipleChoice: Extracts correct choice text, finds incorrect choice for contrast
        - exactMatch: Uses answer directly, generates simple incorrect version
        """
        from wisent_guard.core.benchmark_extractors import HLEExtractor
        
        extractor = HLEExtractor()
        
        # Multiple choice example
        mc_sample = {
            'question': 'Which is the largest planet?\n\nA. Earth\nB. Jupiter\nC. Mars\nD. Venus',
            'answer': 'B',
            'answer_type': 'multipleChoice',
            'category': 'Science'
        }
        
        qa_pair = extractor.extract_qa_pair(mc_sample)
        assert 'Jupiter' in qa_pair['correct_answer']  # Should extract text from choice B
        assert qa_pair['answer_letter'] == 'B'
        
        contrastive_pair = extractor.extract_contrastive_pair(mc_sample)
        assert contrastive_pair['correct_answer'] != contrastive_pair['incorrect_answer']
        
        # Exact match example
        exact_sample = {
            'question': 'What is 2+2?',
            'answer': '4',
            'answer_type': 'exactMatch',
            'category': 'Math'
        }
        
        qa_pair = extractor.extract_qa_pair(exact_sample)
        assert qa_pair['correct_answer'] == '4'
        assert qa_pair['answer_letter'] is None  # No letter for exact match

    def test_hle_model_integration(self):
        """
        HLE integrates with WisentGuard's model loading through TaskInterface pattern.
        Model.load_lm_eval_task() detects HLE and returns TaskInterface instance.
        """
        from wisent_guard.core.model import Model
        
        model = Model('distilbert/distilgpt2', device='cpu')
        
        # Should load HLE as TaskInterface task, not lm-eval task
        task_data = model.load_lm_eval_task('hle', limit=3)
        
        # Verify it's our HLE task implementation
        assert hasattr(task_data, 'get_name')
        assert task_data.get_name() == 'hle'
        assert hasattr(task_data, 'load_data')
        
        # Should support standard data splitting
        train_docs, test_docs = model.split_task_data(task_data, split_ratio=0.7)
        assert len(train_docs) + len(test_docs) > 0

    def test_hle_evaluation_configuration(self):
        """
        HLE uses different evaluation methods based on answer type:
        
        - hle_multiple_choice: log-likelihoods (A/B/C/D/E selection)
        - hle_exact_match: text-generation (string matching)
        - hle: text-generation (mixed types, defaults to more challenging)
        """
        import json
        
        eval_methods_path = "wisent_guard/parameters/benchmarks/benchmark_evaluation_methods.json"
        with open(eval_methods_path, 'r') as f:
            methods = json.load(f)
        
        # Verify evaluation method assignments
        assert methods['hle'] == 'text-generation'
        assert methods['hle_exact_match'] == 'text-generation' 
        assert methods['hle_multiple_choice'] == 'log-likelihoods'

class TestHLEArchitecture:
    """Tests documenting HLE's architectural integration."""
    
    def test_taskinterface_pattern(self):
        """
        HLE follows TaskInterface pattern for non-lm-eval benchmarks.
        
        This pattern allows custom data loading while maintaining compatibility
        with WisentGuard's existing infrastructure.
        """
        from wisent_guard.core.tasks.hle_task import HLETask
        from wisent_guard.core.task_interface import TaskInterface
        
        # HLE implements TaskInterface
        assert issubclass(HLETask, TaskInterface)
        
        # Required methods are implemented
        task = HLETask(limit=2)
        assert callable(task.load_data)
        assert callable(task.get_extractor)
        assert callable(task.get_name)
        assert callable(task.get_description)
        assert callable(task.get_categories)
        
        # Test polymorphic behavior
        data = task.load_data()
        extractor = task.get_extractor()
        
        assert isinstance(data, list)
        assert hasattr(extractor, 'extract_qa_pair')

    def test_text_only_filtering(self):
        """
        HLE implementation filters out multimodal examples for initial release.
        
        The dataset contains ~2500 examples with 342 having images.
        We focus on ~2158 text-only examples for this implementation.
        """
        from wisent_guard.core.tasks.hle_task import HLETask
        
        task = HLETask()
        
        # Sample data should not contain image references
        sample_data = task._generate_sample_data_fallback(limit=5)
        
        for sample in sample_data:
            # Should not have image fields
            assert 'image' not in sample
            assert 'image_1' not in sample
            assert 'image_2' not in sample
            
            # Should be text-based questions
            assert isinstance(sample['question'], str)
            assert len(sample['question']) > 0