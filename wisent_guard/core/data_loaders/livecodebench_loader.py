"""
LiveCodeBench data loader that integrates with the real LiveCodeBench dataset.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class LiveCodeBenchProblem:
    """Represents a LiveCodeBench problem with all metadata."""

    task_id: str
    question_title: str
    question_content: str
    platform: str
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: str
    public_test_cases: List[Dict[str, Any]]
    private_test_cases: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    release_version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with existing task interface."""
        return {
            "task_id": self.task_id,
            "question_title": self.question_title,
            "question_content": self.question_content,
            "starter_code": self.starter_code,
            "difficulty": self.difficulty,
            "platform": self.platform,
            "public_test_cases": self.public_test_cases,
            "private_test_cases": self.private_test_cases,
            "contest_date": self.contest_date.isoformat(),
            "release_version": self.release_version,
            "metadata": self.metadata,
        }


class LiveCodeBenchLoader:
    """Loads real LiveCodeBench data from HuggingFace datasets."""

    def __init__(self):
        self.dataset_name = "livecodebench/code_generation_lite"
        self.available_versions = {
            "release_v1": {"problems": 400, "date_range": "May 2023 - Mar 2024"},
            "release_v2": {"problems": 511, "date_range": "May 2023 - May 2024"},
            "release_v3": {"problems": 612, "date_range": "May 2023 - Jul 2024"},
            "release_v4": {"problems": 713, "date_range": "May 2023 - Sep 2024"},
            "release_v5": {"problems": 880, "date_range": "May 2023 - Jan 2025"},
            "release_v6": {"problems": 1055, "date_range": "May 2023 - Apr 2025"},
            "release_latest": {"problems": 1055, "date_range": "May 2023 - Apr 2025"},
        }

    def load_problems(
        self,
        release_version: str = "release_v1",
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[LiveCodeBenchProblem]:
        """
        Load LiveCodeBench problems from the specified release version.

        Args:
            release_version: Version to load (e.g., "release_v1", "release_v2")
            limit: Maximum number of problems to load
            start_date: Start date filter (YYYY-MM-DD format)
            end_date: End date filter (YYYY-MM-DD format)

        Returns:
            List of LiveCodeBenchProblem objects
        """
        if release_version not in self.available_versions:
            raise ValueError(
                f"Invalid release version: {release_version}. "
                f"Available versions: {list(self.available_versions.keys())}"
            )

        logger.info(f"Loading LiveCodeBench {release_version} from HuggingFace...")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(self.dataset_name, split="test", version_tag=release_version)

            logger.info(f"Successfully loaded {len(dataset)} problems from {release_version}")

            # Convert to our format
            problems = []
            for i, problem_data in enumerate(dataset):
                if limit and i >= limit:
                    break

                # Parse contest date
                contest_date = datetime.fromisoformat(problem_data["contest_date"])

                # Apply date filters
                if start_date:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    if contest_date < start_dt:
                        continue

                if end_date:
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    if contest_date > end_dt:
                        continue

                # Parse test cases
                public_test_cases = self._parse_test_cases(problem_data["public_test_cases"])
                private_test_cases = self._parse_test_cases(problem_data["private_test_cases"])

                # Parse metadata
                metadata = (
                    json.loads(problem_data["metadata"])
                    if isinstance(problem_data["metadata"], str)
                    else problem_data["metadata"]
                )

                problem = LiveCodeBenchProblem(
                    task_id=f"lcb_{release_version}_{problem_data['question_id']}",
                    question_title=problem_data["question_title"],
                    question_content=problem_data["question_content"],
                    platform=problem_data["platform"],
                    question_id=problem_data["question_id"],
                    contest_id=problem_data["contest_id"],
                    contest_date=contest_date,
                    starter_code=problem_data["starter_code"],
                    difficulty=problem_data["difficulty"],
                    public_test_cases=public_test_cases,
                    private_test_cases=private_test_cases,
                    metadata=metadata,
                    release_version=release_version,
                )

                problems.append(problem)

            logger.info(f"Processed {len(problems)} problems after filtering")
            return problems

        except Exception as e:
            logger.error(f"Error loading LiveCodeBench dataset: {e}")
            raise RuntimeError(f"Failed to load LiveCodeBench {release_version}: {e}")

    def _parse_test_cases(self, test_cases_data: Any) -> List[Dict[str, Any]]:
        """Parse test cases from dataset format."""
        if isinstance(test_cases_data, str):
            try:
                test_cases = json.loads(test_cases_data)
            except json.JSONDecodeError:
                logger.warning("Failed to parse test cases JSON, returning empty list")
                return []
        else:
            test_cases = test_cases_data

        if not isinstance(test_cases, list):
            return []

        return test_cases

    def get_version_info(self, release_version: str) -> Dict[str, Any]:
        """Get information about a specific release version."""
        return self.available_versions.get(release_version, {})

    def list_available_versions(self) -> List[str]:
        """List all available release versions."""
        return list(self.available_versions.keys())
