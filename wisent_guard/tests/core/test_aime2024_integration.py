"""
AIME 2024 integration tests.

This module demonstrates how AIME 2024 benchmark works with WisentGuard:
- AIME 2024: 30 high-difficulty AIME contest problems from 2024
- Uses TaskInterface pattern for non-lm-eval benchmarks  
- Reuses GSM8KExtractor with enhanced integer answer handling
- Integrates with existing mathematical reasoning evaluation pipeline
"""

import pytest
from unittest.mock import patch, MagicMock

# Ensure task registration happens
from wisent_guard.core import tasks


class TestAIME2024Benchmark:
    """Essential AIME 2024 functionality tests that serve as documentation."""

    def test_aime2024_task_availability(self):
        """
        AIME 2024 provides high-difficulty mathematical contest problems:
        
        - aime2024: 30 AIME contest problems from 2024
        - Integer answers (0-999 range typical for AIME)
        - Advanced mathematical reasoning required
        """
        from wisent_guard.core.task_interface import list_tasks, get_task
        
        available_tasks = list_tasks()
        assert 'aime2024' in available_tasks
        
        # Test task instantiation and metadata
        aime_task = get_task('aime2024')
        assert aime_task.get_name() == "aime2024"
        assert "AIME contest problems" in aime_task.get_description()
        
        categories = aime_task.get_categories()
        assert "mathematics" in categories
        assert "reasoning" in categories
        assert "contest" in categories

    def test_aime2024_data_structure(self):
        """
        AIME 2024 data structure follows contest problem format:
        
        - ID: Contest identifier (e.g., "2024-I-4", "2024-II-7")  
        - Problem: Mathematical problem statement with LaTeX
        - Solution: Detailed solution process
        - Answer: Integer answer (0-999 range)
        """
        from wisent_guard.core.task_interface import get_task
        
        # Use mock to avoid network dependency in tests
        mock_data = [
            {
                "ID": "2024-I-4",
                "Problem": "Let $x,y$ and $z$ be positive real numbers that satisfy...",
                "Solution": "Denote $\\log_2(x) = a$, $\\log_2(y) = b$...",
                "Answer": 33
            },
            {
                "ID": "2024-II-7", 
                "Problem": "Find the number of ways to arrange...",
                "Solution": "Using combinatorial analysis...",
                "Answer": 256
            }
        ]
        
        with patch('datasets.load_dataset') as mock_load:
            mock_dataset = MagicMock()
            mock_dataset.__iter__ = MagicMock(return_value=iter(mock_data))
            mock_dataset.__len__ = MagicMock(return_value=len(mock_data))
            mock_load.return_value = mock_dataset
            
            aime_task = get_task('aime2024')
            data = aime_task.load_data(limit=2)
            
            # Verify data structure
            for sample in data:
                assert 'ID' in sample          # Contest problem identifier
                assert 'Problem' in sample     # Problem statement
                assert 'Solution' in sample    # Detailed solution  
                assert 'Answer' in sample      # Integer answer
                assert isinstance(sample['Answer'], int)

    def test_aime2024_extraction_pipeline(self):
        """
        AIME 2024 uses enhanced GSM8KExtractor for mathematical reasoning:
        
        - Handles integer answers (key enhancement over string answers)
        - Extracts problem text from 'Problem' field (vs 'question' in GSM8K)
        - Converts integer answers to strings for evaluation consistency
        """
        from wisent_guard.core.benchmark_extractors import GSM8KExtractor
        
        extractor = GSM8KExtractor()
        
        # AIME 2024 format example
        aime_sample = {
            "ID": "2024-I-4",
            "Problem": "Let $x,y$ and $z$ be positive real numbers that satisfy equations...",
            "Answer": 33
        }
        
        qa_pair = extractor.extract_qa_pair(aime_sample)
        
        # Should extract problem and convert integer answer to string
        assert qa_pair['question'] == aime_sample['Problem']
        assert qa_pair['correct_answer'] == "33"  # Integer → String conversion
        
        # Test edge cases for integer answers
        edge_cases = [
            {"Problem": "Find x when x = 0", "Answer": 0},           # Zero
            {"Problem": "Large calculation", "Answer": 999},         # AIME max
            {"Problem": "Simple addition", "Answer": 42}             # Typical
        ]
        
        for case in edge_cases:
            result = extractor.extract_qa_pair(case)
            assert result is not None
            assert result['correct_answer'] == str(case['Answer'])

    def test_aime2024_backward_compatibility(self):
        """
        Enhanced GSM8KExtractor maintains compatibility with existing formats:
        
        - GSM8K: doc['question'] -> doc['answer'] (string with "#### N")
        - MATH-500: doc['problem'] -> doc['answer'] (string)  
        - AIME 2024: doc['Problem'] -> doc['Answer'] (integer)
        """
        from wisent_guard.core.benchmark_extractors import GSM8KExtractor
        
        extractor = GSM8KExtractor()
        
        # GSM8K format (original)
        gsm8k_sample = {
            "question": "Janet's ducks lay 16 eggs per day...",
            "answer": "She sells 3 * 4 = 12 for $12. #### 12"
        }
        result = extractor.extract_qa_pair(gsm8k_sample)
        assert result['correct_answer'] == "12"  # Extracts from "#### 12"
        
        # MATH-500 format  
        math500_sample = {
            "problem": "Find the value of x in the equation...",
            "answer": "The answer is 42."
        }
        result = extractor.extract_qa_pair(math500_sample)
        assert result['correct_answer'] == "The answer is 42."
        
        # AIME 2024 format (new)
        aime_sample = {
            "Problem": "Calculate the sum...",
            "Answer": 156
        }
        result = extractor.extract_qa_pair(aime_sample)
        assert result['correct_answer'] == "156"  # Integer converted to string

    def test_aime2024_model_integration(self):
        """
        AIME 2024 integrates with WisentGuard's model loading through TaskInterface pattern.
        Model.load_lm_eval_task() detects aime2024 and returns TaskInterface instance.
        """
        from wisent_guard.core.model import Model
        from wisent_guard.core.tasks.aime2024_task import AIME2024Task
        
        model = Model('distilbert/distilgpt2', device='cpu')
        
        # Should load AIME 2024 as TaskInterface task, not lm-eval task
        task_data = model.load_lm_eval_task('aime2024', limit=1)
        
        # Verify it's our AIME 2024 task implementation
        assert isinstance(task_data, AIME2024Task)
        assert hasattr(task_data, 'get_name')
        assert task_data.get_name() == 'aime2024'

    def test_aime2024_evaluation_configuration(self):
        """
        AIME 2024 uses text-generation evaluation for mathematical reasoning:
        
        - Method: text-generation (generates answer, compares with ground truth)
        - Ground truth: Integer answers from contest problems
        - Evaluation: Numerical equivalence checking
        """
        import json
        import os
        
        eval_methods_path = os.path.join(
            os.path.dirname(__file__), 
            "../../parameters/benchmarks/benchmark_evaluation_methods.json"
        )
        
        if os.path.exists(eval_methods_path):
            with open(eval_methods_path, 'r') as f:
                methods = json.load(f)
            
            # Verify AIME 2024 uses text-generation
            assert methods['aime2024'] == 'text-generation'


class TestAIME2024Architecture:
    """Tests documenting AIME 2024's architectural integration."""
    
    def test_taskinterface_pattern(self):
        """
        AIME 2024 follows TaskInterface pattern for non-lm-eval benchmarks.
        
        This pattern allows custom data loading while maintaining compatibility
        with WisentGuard's existing mathematical reasoning infrastructure.
        """
        from wisent_guard.core.tasks.aime2024_task import AIME2024Task
        from wisent_guard.core.task_interface import TaskInterface
        
        # AIME 2024 implements TaskInterface
        assert issubclass(AIME2024Task, TaskInterface)
        
        # Required methods are implemented
        task = AIME2024Task(limit=1)
        assert callable(task.load_data)
        assert callable(task.get_extractor) 
        assert callable(task.get_name)
        assert callable(task.get_description)
        assert callable(task.get_categories)
        
        # Test polymorphic behavior
        extractor = task.get_extractor()
        assert hasattr(extractor, 'extract_qa_pair')

    def test_system_integration(self):
        """
        AIME 2024 is properly registered across all WisentGuard components:
        
        - CLI: Listed in ALLOWED_TASKS
        - Benchmarks: Configured in CORE_BENCHMARKS  
        - Extractors: Registered with GSM8KExtractor
        - Tasks: Available through TaskInterface registry
        """
        # Test CLI registration
        from wisent_guard.cli import ALLOWED_TASKS
        assert "aime2024" in ALLOWED_TASKS
        
        # Test benchmarks configuration
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../core/lm-harness-integration"))
        from only_benchmarks import CORE_BENCHMARKS
        assert "aime2024" in CORE_BENCHMARKS
        assert CORE_BENCHMARKS["aime2024"]["task"] == "aime2024"
        
        # Test extractor registration
        from wisent_guard.core.benchmark_extractors import EXTRACTORS, GSM8KExtractor
        assert "aime2024" in EXTRACTORS
        assert EXTRACTORS["aime2024"] == GSM8KExtractor
        
        # Test task registry
        from wisent_guard.core.task_interface import list_tasks
        assert "aime2024" in list_tasks()

    def test_contest_mathematics_focus(self):
        """
        AIME 2024 represents high-difficulty contest mathematics:
        
        - Problem types: Number theory, algebra, geometry, combinatorics
        - Difficulty: Advanced high school / early undergraduate level
        - Answer format: Integers in range 0-999 (AIME contest rules)
        - Reasoning: Multi-step mathematical problem solving
        """
        from wisent_guard.core.task_interface import get_task
        
        task = get_task('aime2024')
        info = task.get_task_info()
        
        # Verify task metadata reflects contest nature
        assert info["task_name"] == "aime2024"
        assert "AIME contest problems" in info["description"]
        assert info["source"] == "Maxwell-Jia/AIME_2024"
        assert info["task_type"] == "text_generation"
        assert info["evaluation_method"] == "mathematical_equivalence"
        
        # Verify categorization
        categories = task.get_categories()
        assert "mathematics" in categories
        assert "contest" in categories
        assert "reasoning" in categories


@pytest.mark.integration  
class TestAIME2024FullPipeline:
    """Integration tests for complete AIME 2024 pipeline."""
    
    @pytest.mark.slow
    def test_minimal_pipeline_execution(self):
        """Test minimal pipeline execution (requires network access)."""
        pytest.skip("Requires network access and is slow - run manually for full validation")
        
        # This test would run the full pipeline but is skipped by default
        # To run manually: python -m wisent_guard tasks aime2024 --limit 2
        
        from wisent_guard.core.task_interface import get_task
        
        task = get_task("aime2024")
        data = task.load_data(limit=1)
        
        assert len(data) == 1
        assert "Problem" in data[0]
        assert "Answer" in data[0]
        
        extractor = task.get_extractor()
        qa_pair = extractor.extract_qa_pair(data[0])
        
        assert qa_pair is not None
        assert "question" in qa_pair
        assert "correct_answer" in qa_pair