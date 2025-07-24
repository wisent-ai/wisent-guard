"""
CLI integration tests for math benchmarks.

Tests the actual CLI commands that users run, validating:
1. CLI structure and argument parsing 
2. Task recognition and validation
3. Error handling and messaging
4. Full execution (when model is pre-configured)

Commands tested:
1. Basic classifier: `python -m wisent_guard tasks gsm8k --model TEST_MODEL --layer 5 --limit 10`
2. Steering: `python -m wisent_guard tasks gsm8k --model TEST_MODEL --steering-mode --steering-method CAA`

This validates the complete pipeline from CLI parsing to model execution.
Full execution tests are skipped by default until model configuration is resolved.
"""

import pytest
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import List

# Import allowed tasks from centralized configuration
from wisent_guard.parameters.task_config import TEST_ALLOWED_TASKS as ALLOWED_TASKS


# Use tiny testing model for fast, reliable CI/CD testing
TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"

# Test with limited samples for speed (minimum 5 to ensure 80/20 split gives >0 training samples)
TEST_LIMIT = 5

class TestBenchmarkCLIIntegration:
    """Test actual CLI commands for math benchmarks.
    
    These tests validate the CLI structure and argument parsing.
    Full execution tests require pre-configured models.
    """
    
    def setup_method(self):
        """Set up for each test - ensure clean environment."""
        # Set environment variables to avoid interactive prompts
        os.environ["WISENT_GUARD_TEST_MODE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp(prefix="wisent_guard_test_")
        self.original_cwd = os.getcwd()
        
    def teardown_method(self):
        """Clean up after each test."""
        # Restore original directory
        os.chdir(self.original_cwd)
        
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except OSError:
            pass  # Ignore cleanup errors
        
        # Clean up environment variables
        for env_var in ["WISENT_GUARD_TEST_MODE", "HF_HUB_DISABLE_PROGRESS_BARS", "TOKENIZERS_PARALLELISM"]:
            if env_var in os.environ:
                del os.environ[env_var]
    
    def run_cli_command(self, cmd_args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        """Run wisent-guard CLI command and return result.
        
        Args:
            cmd_args: Command arguments (without 'python -m wisent_guard')
            timeout: Timeout in seconds
            
        Returns:
            CompletedProcess with stdout, stderr, returncode
        """
        full_cmd = [sys.executable, "-m", "wisent_guard"] + cmd_args
        
        print(f"🔧 Running: {' '.join(full_cmd)}")
        
        try:
            # Set up environment with project root in PYTHONPATH
            project_root = Path(__file__).parent.parent.parent
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project_root) + ":" + env.get("PYTHONPATH", "")
            
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.temp_dir,  # Run in temporary directory to avoid file conflicts
                env=env  # Include PYTHONPATH so subprocess can find wisent_guard
            )
            
            print(f"📤 Return code: {result.returncode}")
            if result.stdout:
                print(f"📝 STDOUT: {result.stdout[:200]}...")
            if result.stderr:
                # Check if stderr contains ERROR/FATAL level messages
                error_log_patterns = ["- ERROR -", "- FATAL -"]
                has_errors = any(pattern in result.stderr for pattern in error_log_patterns)
                if has_errors:
                    print(f"❌ STDERR (HAS ERRORS): {result.stderr[:200]}...")
                else:
                    print(f"📋 STDERR (NORMAL LOGS): {result.stderr[:200]}...")
                
            return result
            
        except subprocess.TimeoutExpired:
            pytest.fail(f"CLI command timed out after {timeout}s: {' '.join(full_cmd)}")
        except Exception as e:
            pytest.fail(f"Failed to run CLI command: {e}")
    
    @pytest.mark.slow
    @pytest.mark.parametrize("task_name", ALLOWED_TASKS)
    def test_basic_classifier_full_execution(self, task_name):
        """SLOW TEST: Full execution of basic classifier functionality on all allowed tasks.
        
        Command: python -m wisent_guard tasks {task_name} --model TEST_MODEL --layer 5 --limit 3
        
        This test downloads models and processes data, making it slow (~60s per task).
        Run with: pytest -m slow or pytest -m "not slow" to exclude.
        """
        cmd_args = [
            "tasks", task_name,
            "--model", TEST_MODEL,
            "--layer", "4",  # hf-internal-testing/tiny-random-gpt2 has 6 layers (0-5)
            "--limit", str(TEST_LIMIT)
        ]
        
        result = self.run_cli_command(cmd_args, timeout=300)  # Longer timeout for model download
        
        # Should succeed with pre-configured model
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        
        # Check for ERROR or FATAL level log messages in stderr
        error_log_patterns = ["- ERROR -", "- FATAL -"]
        error_lines = []
        for line in result.stderr.split('\n'):
            if any(pattern in line for pattern in error_log_patterns):
                error_lines.append(line.strip())
        
        assert len(error_lines) == 0, f"Found ERROR/FATAL log messages in stderr: {error_lines}"
        
        # Verify expected output patterns (check both stdout and stderr)
        full_output = (result.stdout + result.stderr).lower()
        assert task_name in full_output, f"Should mention {task_name} task in output: {full_output[:300]}"
        
        # Should contain processing information
        processing_indicators = ["model", "loading", "processing", "samples", "questions", "results", "pipeline"]
        found_processing = any(indicator in full_output for indicator in processing_indicators)
        assert found_processing, f"Should contain processing indicators: {full_output[:300]}"
        
        print(f"✅ {task_name} basic classifier FULL EXECUTION test passed!")
    
    @pytest.mark.slow
    @pytest.mark.parametrize("task_name", ALLOWED_TASKS)
    def test_steering_functionality_full_execution(self, task_name):
        """SLOW TEST: Full execution of steering functionality on all allowed tasks.
        
        Command: python -m wisent_guard tasks {task_name} --model TEST_MODEL --layer 5 --limit 3 
                 --steering-mode --steering-method CAA --steering-strength 1.5
        
        This test downloads models and processes data with steering, making it slow (~60s per task).
        Run with: pytest -m slow or pytest -m "not slow" to exclude.
        """
        cmd_args = [
            "tasks", task_name,
            "--model", TEST_MODEL,
            "--layer", "4",  # hf-internal-testing/tiny-random-gpt2 has 6 layers (0-5)
            "--limit", str(TEST_LIMIT),
            "--steering-mode",
            "--steering-method", "CAA", 
            "--steering-strength", "1.5"
        ]
        
        result = self.run_cli_command(cmd_args, timeout=300)  # Longer timeout
        
        # Should succeed with pre-configured model
        assert result.returncode == 0, f"Steering CLI command failed: {result.stderr}"
        
        # Check for ERROR or FATAL level log messages in stderr
        error_log_patterns = ["- ERROR -", "- FATAL -"]
        error_lines = []
        for line in result.stderr.split('\n'):
            if any(pattern in line for pattern in error_log_patterns):
                error_lines.append(line.strip())
        
        assert len(error_lines) == 0, f"Found ERROR/FATAL log messages in stderr: {error_lines}"
        
        # Verify steering worked (check both stdout and stderr)
        full_output = (result.stdout + result.stderr).lower()
        
        # Task name should be mentioned somewhere
        assert task_name.lower() in full_output, f"Should mention {task_name} task: {full_output[:300]}"
        
        # Should have some processing indicators (steering may not explicitly mention steering in output)
        processing_indicators = ["results", "pipeline", "processing", "samples"]
        found_processing = any(indicator in full_output for indicator in processing_indicators)
        assert found_processing, f"Should contain processing indicators: {full_output[:300]}"
        
        print(f"✅ {task_name} steering functionality FULL EXECUTION test passed!")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v", "-s"])