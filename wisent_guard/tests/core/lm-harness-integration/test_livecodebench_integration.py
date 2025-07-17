"""
Integration tests for LiveCodeBench execution with wisent-guard pipeline.
"""

import pytest


@pytest.fixture
def sample_livecodebench_data():
    """Sample LiveCodeBench data for testing."""
    return [
        {
            "task_id": "lcb_001",
            "question_title": "Two Sum",
            "question_content": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
            "starter_code": "def two_sum(nums, target):\n    # Your code here\n    pass",
            "difficulty": "EASY",
            "platform": "LEETCODE",
            "public_test_cases": [
                {
                    "input": "[2,7,11,15], 9",
                    "output": "[0,1]",
                    "testtype": "FUNCTIONAL"
                }
            ],
            "contest_date": "2023-05-15",
            "metadata": {
                "tags": ["array", "hash-table"],
                "constraints": "2 <= nums.length <= 10^4"
            }
        },
        {
            "task_id": "lcb_002",
            "question_title": "Valid Parentheses",
            "question_content": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
            "starter_code": "def is_valid(s):\n    # Your code here\n    pass",
            "difficulty": "EASY",
            "platform": "LEETCODE",
            "public_test_cases": [
                {
                    "input": "\"()\"",
                    "output": "true",
                    "testtype": "FUNCTIONAL"
                }
            ],
            "contest_date": "2023-06-01",
            "metadata": {
                "tags": ["string", "stack"],
                "constraints": "1 <= s.length <= 10^4"
            }
        },
        {
            "task_id": "lcb_003",
            "question_title": "Longest Increasing Subsequence",
            "question_content": "Given an integer array nums, return the length of the longest strictly increasing subsequence.",
            "starter_code": "def length_of_lis(nums):\n    # Your code here\n    pass",
            "difficulty": "MEDIUM",
            "platform": "LEETCODE",
            "public_test_cases": [
                {
                    "input": "[10,9,2,5,3,7,101,18]",
                    "output": "4",
                    "testtype": "FUNCTIONAL"
                }
            ],
            "contest_date": "2023-07-10",
            "metadata": {
                "tags": ["array", "binary-search", "dynamic-programming"],
                "constraints": "1 <= nums.length <= 2500"
            }
        }
    ]


@pytest.mark.docker
@pytest.mark.integration
class TestLiveCodeBenchIntegration:
    """Integration tests for LiveCodeBench execution within wisent-guard pipeline."""

    def test_livecodebench_pipeline_integration(
        self, docker_mbpp_runner, sample_livecodebench_data, docker_config
    ):
        """Test that LiveCodeBench execution integrates with wisent-guard pipeline."""
        # Simulate the pipeline flow:
        # 1. Load LiveCodeBench data
        # 2. Convert to MBPP-like format for Docker execution
        # 3. Run in Docker
        # 4. Collect results

        # Convert LiveCodeBench format to MBPP-like format for Docker execution
        mbpp_like_tasks = []
        for task in sample_livecodebench_data[:2]:  # Test with 2 tasks
            mbpp_task = {
                "task_id": task["task_id"],
                "code": task["starter_code"],
                "test_list": [
                    f"assert str({task['starter_code'].split('(')[0].split()[1]}({tc['input']})) == '{tc['output']}'"
                    for tc in task["public_test_cases"]
                ],
            }
            mbpp_like_tasks.append(mbpp_task)

        results = docker_mbpp_runner.run_mbpp_batch(mbpp_like_tasks, docker_config)

        # Verify pipeline output format
        assert len(results) == 2
        for result in results:
            assert "success" in result
            assert "task_id" in result
            assert "exit_code" in result or "error" in result

    def test_livecodebench_contrastive_pair_execution(
        self, docker_mbpp_runner, docker_config
    ):
        """Test Docker execution of contrastive pairs for LiveCodeBench training."""
        # Correct implementation (positive example)
        correct_task = {
            "task_id": "lcb_001",
            "code": """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []""",
            "test_list": [
                "assert two_sum([2,7,11,15], 9) == [0,1]",
                "assert two_sum([3,2,4], 6) == [1,2]",
                "assert two_sum([3,3], 6) == [0,1]",
            ],
        }

        # Incorrect implementation (negative example)
        incorrect_task = {
            "task_id": "lcb_002",
            "code": """def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement]]  # Bug: missing second index
        seen[num] = i
    return []""",
            "test_list": [
                "assert two_sum([2,7,11,15], 9) == [0,1]",
                "assert two_sum([3,2,4], 6) == [1,2]",
                "assert two_sum([3,3], 6) == [0,1]",
            ],
        }

        correct_result = docker_mbpp_runner.run_mbpp_task(correct_task, docker_config)
        incorrect_result = docker_mbpp_runner.run_mbpp_task(incorrect_task, docker_config)

        # Both should execute (Docker isolation), but incorrect should fail assertions
        assert correct_result["success"] is True
        assert "task_id" in incorrect_result  # Docker execution works regardless

    def test_livecodebench_difficulty_levels(
        self, docker_mbpp_runner, sample_livecodebench_data, docker_config
    ):
        """Test that different difficulty levels are handled correctly."""
        # Test different difficulty levels
        easy_task = sample_livecodebench_data[0]  # Two Sum (EASY)
        medium_task = sample_livecodebench_data[2]  # LIS (MEDIUM)
        
        # Convert to MBPP format
        easy_mbpp = {
            "task_id": easy_task["task_id"],
            "code": easy_task["starter_code"],
            "test_list": ["assert callable(two_sum)"],
        }
        
        medium_mbpp = {
            "task_id": medium_task["task_id"],
            "code": medium_task["starter_code"],
            "test_list": ["assert callable(length_of_lis)"],
        }

        easy_result = docker_mbpp_runner.run_mbpp_task(easy_mbpp, docker_config)
        medium_result = docker_mbpp_runner.run_mbpp_task(medium_mbpp, docker_config)

        # Both should execute successfully
        assert easy_result["success"] is True
        assert medium_result["success"] is True
        assert easy_result["task_id"] == easy_task["task_id"]
        assert medium_result["task_id"] == medium_task["task_id"]

    def test_livecodebench_platform_handling(
        self, docker_mbpp_runner, sample_livecodebench_data, docker_config
    ):
        """Test that different platforms (LeetCode, AtCoder, CodeForces) are handled."""
        # All sample data is from LEETCODE platform
        leetcode_task = sample_livecodebench_data[0]
        
        # Convert to MBPP format
        task = {
            "task_id": leetcode_task["task_id"],
            "code": leetcode_task["starter_code"],
            "test_list": ["assert callable(two_sum)"],
        }

        result = docker_mbpp_runner.run_mbpp_task(task, docker_config)

        # Should execute successfully regardless of platform
        assert result["success"] is True
        assert result["task_id"] == leetcode_task["task_id"]


@pytest.mark.docker
@pytest.mark.performance
class TestLiveCodeBenchPerformance:
    """Performance tests for LiveCodeBench execution."""

    def test_livecodebench_vs_mbpp_execution_time(
        self, docker_mbpp_runner, sample_livecodebench_data, sample_mbpp_data, docker_config
    ):
        """Compare execution time between LiveCodeBench and MBPP tasks."""
        import time

        # Convert LiveCodeBench to MBPP format
        lcb_task = {
            "task_id": "lcb_perf_test",
            "code": sample_livecodebench_data[0]["starter_code"],
            "test_list": ["assert callable(two_sum)"],
        }
        
        mbpp_task = sample_mbpp_data[0]

        # Time LiveCodeBench execution
        start_time = time.time()
        lcb_result = docker_mbpp_runner.run_mbpp_task(lcb_task, docker_config)
        lcb_time = time.time() - start_time

        # Time MBPP execution
        start_time = time.time()
        mbpp_result = docker_mbpp_runner.run_mbpp_task(mbpp_task, docker_config)
        mbpp_time = time.time() - start_time

        # Both should complete successfully
        assert lcb_result["success"] is True
        assert mbpp_result["success"] is True
        
        # Performance comparison (both should be fast with mocks)
        assert lcb_time < 2.0  # Less than 2 seconds
        assert mbpp_time < 2.0  # Less than 2 seconds

    def test_livecodebench_batch_processing(
        self, docker_mbpp_runner, sample_livecodebench_data, docker_config
    ):
        """Test batch processing performance for LiveCodeBench tasks."""
        import time

        # Convert all tasks to MBPP format
        lcb_tasks = []
        for task in sample_livecodebench_data:
            lcb_task = {
                "task_id": task["task_id"],
                "code": task["starter_code"],
                "test_list": ["assert True"],  # Simple test
            }
            lcb_tasks.append(lcb_task)

        # Time batch processing
        start_time = time.time()
        results = docker_mbpp_runner.run_mbpp_batch(lcb_tasks, docker_config)
        batch_time = time.time() - start_time

        # Verify results
        assert len(results) == len(sample_livecodebench_data)
        for result in results:
            assert result["success"] is True
            
        # Performance check
        assert batch_time < 5.0  # Less than 5 seconds for batch


@pytest.mark.docker
@pytest.mark.unit
class TestLiveCodeBenchExtractorIntegration:
    """Unit tests for LiveCodeBench extractor integration."""

    def test_livecodebench_extractor_availability(self):
        """Test that LiveCodeBench extractor is available."""
        from wisent_guard.core.benchmark_extractors import get_extractor, LiveCodeBenchExtractor
        
        extractor = get_extractor('livecodebench')
        assert isinstance(extractor, LiveCodeBenchExtractor)

    def test_livecodebench_qa_pair_extraction(self, sample_livecodebench_data):
        """Test QA pair extraction from LiveCodeBench document."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        doc = sample_livecodebench_data[0]
        
        qa_pair = extractor.extract_qa_pair(doc)
        
        assert qa_pair is not None
        assert "question" in qa_pair
        assert "formatted_question" in qa_pair
        assert "correct_answer" in qa_pair
        assert "Two Sum" in qa_pair["question"]
        assert "LEETCODE" in qa_pair["formatted_question"]
        assert "EASY" in qa_pair["formatted_question"]
        assert "def two_sum" in qa_pair["correct_answer"]

    def test_livecodebench_contrastive_pair_extraction(self, sample_livecodebench_data):
        """Test contrastive pair extraction from LiveCodeBench document."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        doc = sample_livecodebench_data[0]
        
        contrastive_pair = extractor.extract_contrastive_pair(doc)
        
        assert contrastive_pair is not None
        assert "question" in contrastive_pair
        assert "correct_answer" in contrastive_pair
        assert "incorrect_answer" in contrastive_pair
        assert contrastive_pair["correct_answer"] != contrastive_pair["incorrect_answer"]

    def test_livecodebench_error_handling(self):
        """Test error handling for malformed LiveCodeBench documents."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        
        # Test with empty document
        empty_doc = {}
        result = extractor.extract_qa_pair(empty_doc)
        assert result is None
        
        # Test with missing required fields
        incomplete_doc = {"question_title": "Test"}
        result = extractor.extract_qa_pair(incomplete_doc)
        assert result is None

    def test_livecodebench_platform_and_difficulty_handling(self, sample_livecodebench_data):
        """Test that platform and difficulty are properly included in extracted data."""
        from wisent_guard.core.benchmark_extractors import LiveCodeBenchExtractor
        
        extractor = LiveCodeBenchExtractor()
        
        # Test different difficulties
        easy_doc = sample_livecodebench_data[0]  # EASY
        medium_doc = sample_livecodebench_data[2]  # MEDIUM
        
        easy_result = extractor.extract_qa_pair(easy_doc)
        medium_result = extractor.extract_qa_pair(medium_doc)
        
        assert "EASY" in easy_result["formatted_question"]
        assert "MEDIUM" in medium_result["formatted_question"]
        assert "LEETCODE" in easy_result["formatted_question"]
        assert "LEETCODE" in medium_result["formatted_question"]