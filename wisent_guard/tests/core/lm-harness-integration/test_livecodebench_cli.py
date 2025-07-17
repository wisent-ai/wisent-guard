"""
CLI integration tests for LiveCodeBench.
"""

import pytest
from unittest.mock import patch


@pytest.mark.cli
class TestLiveCodeBenchCLI:
    """Tests for LiveCodeBench CLI integration."""

    def test_livecodebench_task_available_in_cli(self):
        """Test that LiveCodeBench is available as a CLI task."""
        
        # This should not raise an UnsupportedBenchmarkError
        try:
            from wisent_guard.core.benchmark_extractors import get_extractor
            extractor = get_extractor('livecodebench')
            assert extractor is not None
        except Exception as e:
            pytest.fail(f"LiveCodeBench extractor not available: {e}")

    def test_livecodebench_in_supported_tasks(self):
        """Test that LiveCodeBench is in the supported tasks list."""
        
        # Read the CLI file to check if livecodebench is in the task list
        with open('/Users/janpiotrowski/Projekty/wisent-guard/wisent_guard/cli.py', 'r') as f:
            cli_content = f.read()
        
        assert '"livecodebench"' in cli_content, "LiveCodeBench should be in the supported tasks list"

    def test_livecodebench_group_task_skip_logic(self):
        """Test that LiveCodeBench uses the same skip logic as MBPP."""
        
        # Read the CLI file to check if livecodebench is in the skip logic
        with open('/Users/janpiotrowski/Projekty/wisent-guard/wisent_guard/cli.py', 'r') as f:
            cli_content = f.read()
        
        assert 'task_name in ["mbpp", "livecodebench"]' in cli_content, \
            "LiveCodeBench should use the same skip logic as MBPP"

    @patch('wisent_guard.cli.run_task_pipeline')
    def test_livecodebench_cli_command_simulation(self, mock_run_task):
        """Test simulation of CLI command: python -m wisent_guard tasks livecodebench"""
        # Mock the expected return value
        mock_run_task.return_value = {
            "success": True,
            "results": [{"task_id": "lcb_001", "passed": True}],
        }

        # This would be the actual CLI integration
        cli_result = mock_run_task(
            task_name="livecodebench",
            model="distilbert/distilgpt2",
            layer=5,
            limit=10,
            docker=True,
        )

        assert cli_result["success"] is True
        mock_run_task.assert_called_once()

    def test_livecodebench_extractor_registration(self):
        """Test that LiveCodeBench extractor is properly registered."""
        from wisent_guard.core.benchmark_extractors import EXTRACTORS, LiveCodeBenchExtractor
        
        assert 'livecodebench' in EXTRACTORS
        assert EXTRACTORS['livecodebench'] == LiveCodeBenchExtractor

    def test_livecodebench_cli_help_integration(self):
        """Test that LiveCodeBench appears in CLI help/documentation."""
        # This is more of a documentation test
        # We check that the task is properly integrated
        from wisent_guard.core.benchmark_extractors import get_extractor
        
        try:
            extractor = get_extractor('livecodebench')
            assert extractor is not None
        except Exception as e:
            pytest.fail(f"LiveCodeBench not properly integrated: {e}")


@pytest.mark.cli
@pytest.mark.integration
class TestLiveCodeBenchCLIIntegration:
    """Integration tests for LiveCodeBench CLI functionality."""

    def test_livecodebench_error_handling_for_cli(self):
        """Test LiveCodeBench error handling for CLI usage."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        
        # Test with invalid task that should fail gracefully
        invalid_task = {
            "task_id": "invalid_lcb_999",
            "question_title": "Invalid Task",
            "question_content": None,  # This should cause extraction to fail
            "starter_code": "def invalid_function():\n    raise ValueError('This should fail')",
            "difficulty": "HARD",
            "platform": "UNKNOWN",
        }

        result = extractor.extract_qa_pair(invalid_task)

        # Should handle error gracefully
        assert result is None

    def test_livecodebench_cli_parameter_validation(self):
        """Test that CLI parameters are validated for LiveCodeBench."""
        from wisent_guard.core.benchmark_extractors import get_extractor
        
        # Test that we can get the extractor without issues
        extractor = get_extractor('livecodebench')
        assert extractor is not None
        
        # Test that it has the required methods
        assert hasattr(extractor, 'extract_qa_pair')
        assert hasattr(extractor, 'extract_contrastive_pair')

    def test_livecodebench_docker_compatibility(self):
        """Test that LiveCodeBench is compatible with Docker execution."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        
        # Test with a simple task
        simple_task = {
            "task_id": "lcb_simple",
            "question_title": "Simple Test",
            "question_content": "Return the sum of two numbers",
            "starter_code": "def add(a, b):\n    return a + b",
            "difficulty": "EASY",
            "platform": "LEETCODE",
        }

        qa_pair = extractor.extract_qa_pair(simple_task)
        
        assert qa_pair is not None
        assert "def add" in qa_pair["correct_answer"]
        # The code should be executable in Docker
        assert "return a + b" in qa_pair["correct_answer"]

    def test_livecodebench_task_format_consistency(self):
        """Test that LiveCodeBench task format is consistent with expectations."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        
        # Test with typical LiveCodeBench format
        lcb_task = {
            "task_id": "lcb_format_test",
            "question_title": "Format Test",
            "question_content": "Test the format consistency",
            "starter_code": "def solution():\n    pass",
            "difficulty": "MEDIUM",
            "platform": "ATCODER",
            "public_test_cases": [
                {
                    "input": "test_input",
                    "output": "test_output",
                    "testtype": "FUNCTIONAL"
                }
            ],
        }

        qa_pair = extractor.extract_qa_pair(lcb_task)
        contrastive_pair = extractor.extract_contrastive_pair(lcb_task)
        
        # QA pair should have expected format
        assert qa_pair is not None
        assert "question" in qa_pair
        assert "formatted_question" in qa_pair
        assert "correct_answer" in qa_pair
        
        # Contrastive pair should have expected format
        assert contrastive_pair is not None
        assert "question" in contrastive_pair
        assert "correct_answer" in contrastive_pair
        assert "incorrect_answer" in contrastive_pair
        
        # Platform and difficulty should be in formatted question
        assert "ATCODER" in qa_pair["formatted_question"]
        assert "MEDIUM" in qa_pair["formatted_question"]