"""
LiveCodeBench task implementation for task-agnostic architecture.
"""

from typing import Dict, Any, List, Optional
from ..task_interface import TaskInterface
from ..benchmark_extractors import LiveCodeBenchExtractor, BenchmarkExtractor


class LiveCodeBenchTask(TaskInterface):
    """LiveCodeBench task implementation."""
    
    def __init__(self, release_version: str = "release_v1"):
        self._extractor = LiveCodeBenchExtractor()
        self._release_version = release_version
        self._validate_release_version(release_version)
    
    def _validate_release_version(self, release_version: str) -> None:
        """Validate release version."""
        valid_versions = {
            "release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6",
            "release_latest", "v1", "v2", "v1_v3", "v4_v5"
        }
        if release_version not in valid_versions:
            raise ValueError(f"Invalid release version: {release_version}. Valid versions: {valid_versions}")
    
    def _get_version_info(self) -> Dict[str, Any]:
        """Get version-specific information."""
        version_info = {
            "release_v1": {"problems": 400, "date_range": "May 2023 - Mar 2024"},
            "release_v2": {"problems": 511, "date_range": "May 2023 - May 2024"},
            "release_v3": {"problems": 612, "date_range": "May 2023 - Jul 2024"},
            "release_v4": {"problems": 713, "date_range": "May 2023 - Sep 2024"},
            "release_v5": {"problems": 880, "date_range": "May 2023 - Jan 2025"},
            "release_v6": {"problems": 1055, "date_range": "May 2023 - Apr 2025"},
            "release_latest": {"problems": 1055, "date_range": "May 2023 - Apr 2025"},
            "v1": {"problems": 400, "date_range": "May 2023 - Mar 2024"},
            "v2": {"problems": 511, "date_range": "May 2023 - May 2024"},
            "v1_v3": {"problems": 612, "date_range": "May 2023 - Jul 2024"},
            "v4_v5": {"problems": 880, "date_range": "Sep 2024 - Jan 2025"}
        }
        return version_info.get(self._release_version, {"problems": 400, "date_range": "May 2023 - Mar 2024"})
    
    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load LiveCodeBench data for the specified release version."""
        # Get version-specific information
        version_info = self._get_version_info()
        expected_problems = version_info["problems"]
        
        # TODO: Replace with real LiveCodeBench integration
        # For now, generate sample data based on release version
        sample_data = self._generate_sample_data(expected_problems)
        
        if limit:
            sample_data = sample_data[:limit]
        
        return sample_data
    
    def _generate_sample_data(self, expected_problems: int) -> List[Dict[str, Any]]:
        """Generate sample data for the specified number of problems."""
        base_problems = [
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
            },
            {
                "task_id": "lcb_004",
                "question_title": "Merge Two Sorted Lists",
                "question_content": "You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list.",
                "starter_code": "def merge_two_lists(list1, list2):\n    # Your code here\n    pass",
                "difficulty": "EASY",
                "platform": "LEETCODE",
                "public_test_cases": [
                    {
                        "input": "[1,2,4], [1,3,4]",
                        "output": "[1,1,2,3,4,4]",
                        "testtype": "FUNCTIONAL"
                    }
                ],
                "contest_date": "2023-08-01",
                "metadata": {
                    "tags": ["linked-list", "recursion"],
                    "constraints": "0 <= list1.length, list2.length <= 50"
                }
            },
            {
                "task_id": "lcb_005",
                "question_title": "Best Time to Buy and Sell Stock",
                "question_content": "You are given an array prices where prices[i] is the price of a given stock on the ith day. Find the maximum profit.",
                "starter_code": "def max_profit(prices):\n    # Your code here\n    pass",
                "difficulty": "EASY",
                "platform": "LEETCODE",
                "public_test_cases": [
                    {
                        "input": "[7,1,5,3,6,4]",
                        "output": "5",
                        "testtype": "FUNCTIONAL"
                    }
                ],
                "contest_date": "2023-09-15",
                "metadata": {
                    "tags": ["array", "dynamic-programming"],
                    "constraints": "1 <= prices.length <= 10^5"
                }
            }
        ]
        
        # Generate additional problems to simulate the expected count
        # For now, we'll duplicate and modify the base problems
        result = []
        for i in range(expected_problems):
            base_index = i % len(base_problems)
            problem = base_problems[base_index].copy()
            
            # Modify task_id to make it unique
            problem["task_id"] = f"lcb_{i+1:03d}"
            
            # Add version-specific metadata
            problem["release_version"] = self._release_version
            
            result.append(problem)
        
        return result
    
    def get_extractor(self) -> BenchmarkExtractor:
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


# TODO: In a real implementation, this would integrate with the actual LiveCodeBench library
# Example integration:
# from livecodebench import LiveCodeBench
# 
# class LiveCodeBenchTask(TaskInterface):
#     def __init__(self):
#         self._lcb = LiveCodeBench()
#         self._extractor = LiveCodeBenchExtractor()
#     
#     def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
#         return self._lcb.load_problems(limit=limit)