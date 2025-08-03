"""
LiveCodeBench task implementation for task-agnostic architecture.
"""

from typing import Any, Dict, List, Optional

from ..benchmark_extractors import LiveCodeBenchExtractor
from ..data_loaders import LiveCodeBenchLoader
from ..task_interface import TaskInterface


class LiveCodeBenchTask(TaskInterface):
    """LiveCodeBench task implementation."""

    def __init__(self, release_version: str = "release_v1", limit: Optional[int] = None):
        self._extractor = LiveCodeBenchExtractor()
        self._data_loader = LiveCodeBenchLoader()
        self._release_version = release_version
        self._validate_release_version(release_version)
        self._data = None  # Cache for loaded data
        self._limit = limit  # Store limit for later use

    def _validate_release_version(self, release_version: str) -> None:
        """Validate release version."""
        try:
            valid_versions = set(self._data_loader.list_available_versions())
            if release_version not in valid_versions:
                raise ValueError(f"Invalid release version: {release_version}. Valid versions: {valid_versions}")
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception:
            # If we can't load versions (e.g., due to dataset issues), just log a warning
            import logging

            logging.warning(
                f"Could not validate release version {release_version} due to data loader issues. Proceeding with fallback data."
            )

    def _get_version_info(self) -> Dict[str, Any]:
        """Get version-specific information."""
        try:
            return self._data_loader.get_version_info(self._release_version)
        except Exception:
            # Return default info if data loader fails
            return {
                "version": self._release_version,
                "description": f"LiveCodeBench {self._release_version} (fallback mode)",
                "contest_start": "2023-01-01",
                "contest_end": "2023-12-31",
            }

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LiveCodeBench data for the specified release version."""
        try:
            # Load real LiveCodeBench data
            problems = self._data_loader.load_problems(release_version=self._release_version, limit=limit)

            # Convert to dictionary format
            return [problem.to_dict() for problem in problems]

        except Exception as e:
            # Fallback to sample data if loading fails
            import logging

            logging.warning(f"Failed to load real LiveCodeBench data: {e}. Using sample data.")
            return self._generate_sample_data_fallback(limit)

    def _generate_sample_data_fallback(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate sample data for the specified number of problems."""
        base_problems = [
            {
                "task_id": "lcb_001",
                "question_title": "Two Sum",
                "question_content": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
                "starter_code": "def two_sum(nums, target):\n    # Your code here\n    pass",
                "difficulty": "EASY",
                "platform": "LEETCODE",
                "public_test_cases": [{"input": "[2,7,11,15], 9", "output": "[0,1]", "testtype": "FUNCTIONAL"}],
                "contest_date": "2023-05-15",
                "metadata": {"tags": ["array", "hash-table"], "constraints": "2 <= nums.length <= 10^4"},
            },
            {
                "task_id": "lcb_002",
                "question_title": "Valid Parentheses",
                "question_content": "Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
                "starter_code": "def is_valid(s):\n    # Your code here\n    pass",
                "difficulty": "EASY",
                "platform": "LEETCODE",
                "public_test_cases": [{"input": '"()"', "output": "true", "testtype": "FUNCTIONAL"}],
                "contest_date": "2023-06-01",
                "metadata": {"tags": ["string", "stack"], "constraints": "1 <= s.length <= 10^4"},
            },
            {
                "task_id": "lcb_003",
                "question_title": "Longest Increasing Subsequence",
                "question_content": "Given an integer array nums, return the length of the longest strictly increasing subsequence.",
                "starter_code": "def length_of_lis(nums):\n    # Your code here\n    pass",
                "difficulty": "MEDIUM",
                "platform": "LEETCODE",
                "public_test_cases": [{"input": "[10,9,2,5,3,7,101,18]", "output": "4", "testtype": "FUNCTIONAL"}],
                "contest_date": "2023-07-10",
                "metadata": {
                    "tags": ["array", "binary-search", "dynamic-programming"],
                    "constraints": "1 <= nums.length <= 2500",
                },
            },
            {
                "task_id": "lcb_004",
                "question_title": "Merge Two Sorted Lists",
                "question_content": "You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list.",
                "starter_code": "def merge_two_lists(list1, list2):\n    # Your code here\n    pass",
                "difficulty": "EASY",
                "platform": "LEETCODE",
                "public_test_cases": [
                    {"input": "[1,2,4], [1,3,4]", "output": "[1,1,2,3,4,4]", "testtype": "FUNCTIONAL"}
                ],
                "contest_date": "2023-08-01",
                "metadata": {
                    "tags": ["linked-list", "recursion"],
                    "constraints": "0 <= list1.length, list2.length <= 50",
                },
            },
            {
                "task_id": "lcb_005",
                "question_title": "Best Time to Buy and Sell Stock",
                "question_content": "You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit.",
                "starter_code": "def max_profit(prices):\n    # Your code here\n    pass",
                "difficulty": "EASY",
                "platform": "LEETCODE",
                "public_test_cases": [{"input": "[7,1,5,3,6,4]", "output": "5", "testtype": "FUNCTIONAL"}],
                "contest_date": "2023-09-15",
                "metadata": {"tags": ["array", "dynamic-programming"], "constraints": "1 <= prices.length <= 10^5"},
            },
        ]

        # Generate limited sample data for fallback
        if limit:
            base_problems = base_problems[:limit]

        # Add version-specific metadata
        for problem in base_problems:
            problem["release_version"] = self._release_version

        return base_problems

    def get_extractor(self):
        """Get the LiveCodeBench extractor."""
        return self._extractor

    def get_name(self) -> str:
        """Get the task name."""
        return "livecodebench"

    def get_description(self) -> str:
        """Get the task description."""
        version_info = self._get_version_info()
        return f"LiveCodeBench {self._release_version}: Contamination-free coding benchmark with {version_info['problems']} problems ({version_info['date_range']}) from LeetCode, AtCoder, and CodeForces"

    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return ["coding", "reasoning", "algorithms", "data-structures"]

    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # LiveCodeBench doesn't have separate validation sets

    def has_test_docs(self) -> bool:
        """Check if task has test documents."""
        return True  # All samples are considered test docs

    def test_docs(self) -> List[Dict[str, Any]]:
        """Get test documents."""
        if self._data is None:
            self._data = self.load_data(limit=self._limit)
        return self._data

    def validation_docs(self) -> List[Dict[str, Any]]:
        """Get validation documents."""
        return []  # No separate validation set

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text prompt."""
        # Combine problem description with starter code
        question = doc.get("question_content", "")
        starter = doc.get("starter_code", "")
        return f"{question}\n\n{starter}"


# TODO: In a real implementation, this would integrate with the actual LiveCodeBench library
# Example integration:
# from livecodebench import LiveCodeBench
#
# class LiveCodeBenchTask(TaskInterface):
#     def __init__(self):
#         self._lcb = LiveCodeBench()
#         # self._extractor = LiveCodeBenchExtractor()  # Not needed with model outputs approach
#
#     def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
#         return self._lcb.load_problems(limit=limit)
