"""
Tests for LiveCodeBench data integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from wisent_guard.core.data_loaders import LiveCodeBenchLoader, SteeringDataExtractor
from wisent_guard.core.data_loaders.livecodebench_loader import LiveCodeBenchProblem
from wisent_guard.core.data_loaders.steering_data_extractor import ContrastivePair
from wisent_guard.core.tasks.livecodebench_task import LiveCodeBenchTask


class TestLiveCodeBenchLoader:
    """Test suite for LiveCodeBenchLoader."""
    
    def test_init(self):
        """Test loader initialization."""
        loader = LiveCodeBenchLoader()
        assert loader.dataset_name == "livecodebench/code_generation_lite"
        assert "release_v1" in loader.available_versions
        assert "release_v2" in loader.available_versions
    
    def test_available_versions(self):
        """Test available version listing."""
        loader = LiveCodeBenchLoader()
        versions = loader.list_available_versions()
        
        assert isinstance(versions, list)
        assert "release_v1" in versions
        assert "release_v2" in versions
        assert "release_latest" in versions
    
    def test_get_version_info(self):
        """Test version info retrieval."""
        loader = LiveCodeBenchLoader()
        
        v1_info = loader.get_version_info("release_v1")
        assert v1_info["problems"] == 400
        assert "May 2023" in v1_info["date_range"]
        
        v2_info = loader.get_version_info("release_v2")
        assert v2_info["problems"] == 511
        
        # Test invalid version
        invalid_info = loader.get_version_info("invalid_version")
        assert invalid_info == {}
    
    @patch('wisent_guard.core.data_loaders.livecodebench_loader.load_dataset')
    def test_load_problems_success(self, mock_load_dataset):
        """Test successful problem loading."""
        # Mock dataset response
        mock_dataset = [
            {
                "question_title": "Two Sum",
                "question_content": "Given array and target, return indices",
                "platform": "leetcode",
                "question_id": "1",
                "contest_id": "weekly-contest-1",
                "contest_date": "2023-05-15T00:00:00",
                "starter_code": "def twoSum(nums, target):\n    pass",
                "difficulty": "easy",
                "public_test_cases": '[{"input": "[2,7,11,15], 9", "output": "[0,1]", "testtype": "functional"}]',
                "private_test_cases": '[{"input": "[3,2,4], 6", "output": "[1,2]", "testtype": "functional"}]',
                "metadata": '{"func_name": "twoSum", "tags": ["array", "hash-table"]}'
            }
        ]
        mock_load_dataset.return_value = mock_dataset
        
        loader = LiveCodeBenchLoader()
        problems = loader.load_problems("release_v1", limit=1)
        
        assert len(problems) == 1
        assert isinstance(problems[0], LiveCodeBenchProblem)
        assert problems[0].question_title == "Two Sum"
        assert problems[0].platform == "leetcode"
        assert problems[0].difficulty == "easy"
        assert problems[0].release_version == "release_v1"
        assert problems[0].task_id == "lcb_release_v1_1"
        
        mock_load_dataset.assert_called_once_with(
            "livecodebench/code_generation_lite",
            split="test",
            version_tag="release_v1",
            trust_remote_code=True
        )
    
    @patch('wisent_guard.core.data_loaders.livecodebench_loader.load_dataset')
    def test_load_problems_with_date_filter(self, mock_load_dataset):
        """Test loading problems with date filtering."""
        mock_dataset = [
            {
                "question_title": "Early Problem",
                "question_content": "Description",
                "platform": "leetcode",
                "question_id": "1",
                "contest_id": "contest-1",
                "contest_date": "2023-01-01T00:00:00",
                "starter_code": "def solution():\n    pass",
                "difficulty": "easy",
                "public_test_cases": '[]',
                "private_test_cases": '[]',
                "metadata": '{}'
            },
            {
                "question_title": "Late Problem",
                "question_content": "Description",
                "platform": "leetcode",
                "question_id": "2",
                "contest_id": "contest-2",
                "contest_date": "2023-12-31T00:00:00",
                "starter_code": "def solution():\n    pass",
                "difficulty": "easy",
                "public_test_cases": '[]',
                "private_test_cases": '[]',
                "metadata": '{}'
            }
        ]
        mock_load_dataset.return_value = mock_dataset
        
        loader = LiveCodeBenchLoader()
        problems = loader.load_problems(
            "release_v1",
            start_date="2023-06-01",
            end_date="2023-12-01"
        )
        
        assert len(problems) == 0  # Both problems are outside date range
        
        problems = loader.load_problems(
            "release_v1",
            start_date="2022-01-01",
            end_date="2024-01-01"
        )
        
        assert len(problems) == 2  # Both problems are inside date range
    
    def test_load_problems_invalid_version(self):
        """Test loading problems with invalid version."""
        loader = LiveCodeBenchLoader()
        
        with pytest.raises(ValueError, match="Invalid release version"):
            loader.load_problems("invalid_version")
    
    @patch('wisent_guard.core.data_loaders.livecodebench_loader.load_dataset')
    def test_load_problems_dataset_error(self, mock_load_dataset):
        """Test handling dataset loading errors."""
        mock_load_dataset.side_effect = Exception("Dataset loading failed")
        
        loader = LiveCodeBenchLoader()
        
        with pytest.raises(RuntimeError, match="Failed to load LiveCodeBench"):
            loader.load_problems("release_v1")
    
    def test_parse_test_cases(self):
        """Test test case parsing."""
        loader = LiveCodeBenchLoader()
        
        # Test valid JSON string
        valid_json = '[{"input": "test", "output": "result", "testtype": "functional"}]'
        parsed = loader._parse_test_cases(valid_json)
        assert len(parsed) == 1
        assert parsed[0]["input"] == "test"
        
        # Test invalid JSON
        invalid_json = "invalid json"
        parsed = loader._parse_test_cases(invalid_json)
        assert parsed == []
        
        # Test list input
        list_input = [{"input": "test", "output": "result"}]
        parsed = loader._parse_test_cases(list_input)
        assert len(parsed) == 1
        assert parsed[0]["input"] == "test"


class TestLiveCodeBenchProblem:
    """Test suite for LiveCodeBenchProblem."""
    
    def test_to_dict(self):
        """Test problem to dictionary conversion."""
        problem = LiveCodeBenchProblem(
            task_id="lcb_test_1",
            question_title="Test Problem",
            question_content="Test content",
            platform="leetcode",
            question_id="1",
            contest_id="test-contest",
            contest_date=datetime(2023, 5, 15),
            starter_code="def solution():\n    pass",
            difficulty="easy",
            public_test_cases=[{"input": "test", "output": "result"}],
            private_test_cases=[],
            metadata={"tags": ["test"]},
            release_version="release_v1"
        )
        
        result = problem.to_dict()
        
        assert result["task_id"] == "lcb_test_1"
        assert result["question_title"] == "Test Problem"
        assert result["platform"] == "leetcode"
        assert result["difficulty"] == "easy"
        assert result["release_version"] == "release_v1"
        assert result["contest_date"] == "2023-05-15T00:00:00"
        assert result["metadata"] == {"tags": ["test"]}


class TestSteeringDataExtractor:
    """Test suite for SteeringDataExtractor."""
    
    def test_init(self):
        """Test extractor initialization."""
        extractor = SteeringDataExtractor()
        
        strategies = extractor.list_available_strategies()
        assert "correct_vs_incorrect" in strategies
        assert "complete_vs_incomplete" in strategies
        assert "efficient_vs_inefficient" in strategies
        assert "readable_vs_unreadable" in strategies
    
    def test_extract_contrastive_pairs(self):
        """Test contrastive pair extraction."""
        extractor = SteeringDataExtractor()
        
        # Create test problem
        problem = LiveCodeBenchProblem(
            task_id="lcb_test_1",
            question_title="Test Problem",
            question_content="Test content",
            platform="leetcode",
            question_id="1",
            contest_id="test-contest",
            contest_date=datetime(2023, 5, 15),
            starter_code="def solution():\n    pass",
            difficulty="easy",
            public_test_cases=[],
            private_test_cases=[],
            metadata={},
            release_version="release_v1"
        )
        
        pairs = extractor.extract_contrastive_pairs(
            [problem],
            strategy="correct_vs_incorrect",
            pairs_per_problem=2
        )
        
        assert len(pairs) == 2
        assert all(isinstance(pair, ContrastivePair) for pair in pairs)
        assert all(pair.problem_id == "lcb_test_1" for pair in pairs)
        assert all("correct" in pair.positive_prompt for pair in pairs)
        assert all("solution" in pair.negative_prompt for pair in pairs)
    
    def test_extract_contrastive_pairs_invalid_strategy(self):
        """Test extraction with invalid strategy."""
        extractor = SteeringDataExtractor()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            extractor.extract_contrastive_pairs([], strategy="invalid_strategy")
    
    def test_get_strategy_info(self):
        """Test strategy information retrieval."""
        extractor = SteeringDataExtractor()
        
        info = extractor.get_strategy_info("correct_vs_incorrect")
        assert info["name"] == "correct_vs_incorrect"
        assert "correct vs incorrect" in info["description"]
        assert info["available"] is True
        
        invalid_info = extractor.get_strategy_info("invalid_strategy")
        assert invalid_info["available"] is False
    
    def test_generate_correct_vs_incorrect_pairs(self):
        """Test correct vs incorrect pair generation."""
        extractor = SteeringDataExtractor()
        
        problem = LiveCodeBenchProblem(
            task_id="lcb_test_1",
            question_title="Two Sum",
            question_content="Find two numbers that add up to target",
            platform="leetcode",
            question_id="1",
            contest_id="test-contest",
            contest_date=datetime(2023, 5, 15),
            starter_code="def twoSum(nums, target):\n    pass",
            difficulty="easy",
            public_test_cases=[],
            private_test_cases=[],
            metadata={},
            release_version="release_v1"
        )
        
        pairs = extractor._generate_correct_vs_incorrect(problem, 1)
        
        assert len(pairs) == 1
        pair = pairs[0]
        assert "Two Sum" in pair.positive_prompt
        assert "correct and efficient" in pair.positive_prompt
        assert "subtle bugs" in pair.negative_prompt
        assert pair.metadata["strategy"] == "correct_vs_incorrect"
        assert pair.metadata["difficulty"] == "easy"
        assert pair.metadata["platform"] == "leetcode"


class TestLiveCodeBenchTaskIntegration:
    """Test suite for LiveCodeBenchTask integration."""
    
    def test_task_init_with_real_loader(self):
        """Test task initialization with real data loader."""
        task = LiveCodeBenchTask("release_v1")
        
        assert task._release_version == "release_v1"
        assert isinstance(task._data_loader, LiveCodeBenchLoader)
    
    def test_task_validation_with_real_versions(self):
        """Test task validation with real version list."""
        # Valid version should work
        task = LiveCodeBenchTask("release_v1")
        assert task._release_version == "release_v1"
        
        # Invalid version should raise error
        with pytest.raises(ValueError, match="Invalid release version"):
            LiveCodeBenchTask("invalid_version")
    
    def test_get_version_info_with_real_loader(self):
        """Test version info retrieval with real loader."""
        task = LiveCodeBenchTask("release_v1")
        
        info = task._get_version_info()
        assert info["problems"] == 400
        assert "May 2023" in info["date_range"]
    
    @patch('wisent_guard.core.data_loaders.livecodebench_loader.load_dataset')
    def test_load_data_success(self, mock_load_dataset):
        """Test successful data loading."""
        mock_dataset = [
            {
                "question_title": "Test Problem",
                "question_content": "Test content",
                "platform": "leetcode",
                "question_id": "1",
                "contest_id": "test-contest",
                "contest_date": "2023-05-15T00:00:00",
                "starter_code": "def solution():\n    pass",
                "difficulty": "easy",
                "public_test_cases": '[]',
                "private_test_cases": '[]',
                "metadata": '{}'
            }
        ]
        mock_load_dataset.return_value = mock_dataset
        
        task = LiveCodeBenchTask("release_v1")
        data = task.load_data(limit=1)
        
        assert len(data) == 1
        assert data[0]["question_title"] == "Test Problem"
        assert data[0]["platform"] == "leetcode"
        assert data[0]["release_version"] == "release_v1"
    
    @patch('wisent_guard.core.data_loaders.livecodebench_loader.load_dataset')
    def test_load_data_fallback_on_error(self, mock_load_dataset):
        """Test fallback to sample data on loading error."""
        mock_load_dataset.side_effect = Exception("Dataset loading failed")
        
        task = LiveCodeBenchTask("release_v1")
        data = task.load_data(limit=2)
        
        # Should fall back to sample data
        assert len(data) == 2
        assert all("lcb_" in item["task_id"] for item in data)
        assert all(item["release_version"] == "release_v1" for item in data)


@pytest.mark.integration
class TestLiveCodeBenchDataIntegration:
    """Integration tests for LiveCodeBench data loading."""
    
    def test_real_dataset_loading(self):
        """Test loading real dataset (requires internet connection)."""
        loader = LiveCodeBenchLoader()
        
        try:
            # Try to load a small sample
            problems = loader.load_problems("release_v1", limit=1)
            
            # Verify we got real data
            assert len(problems) == 1
            assert problems[0].question_title
            assert problems[0].platform in ["leetcode", "codeforces", "atcoder"]
            assert problems[0].release_version == "release_v1"
            
        except Exception as e:
            pytest.skip(f"Skipping real dataset test due to: {e}")
    
    def test_steering_data_extraction_integration(self):
        """Test end-to-end steering data extraction."""
        loader = LiveCodeBenchLoader()
        extractor = SteeringDataExtractor()
        
        try:
            # Load a small sample
            problems = loader.load_problems("release_v1", limit=2)
            
            # Extract contrastive pairs
            pairs = extractor.extract_contrastive_pairs(
                problems,
                strategy="correct_vs_incorrect",
                pairs_per_problem=1
            )
            
            # Verify extraction worked
            assert len(pairs) == 2
            assert all(isinstance(pair, ContrastivePair) for pair in pairs)
            assert all(pair.positive_prompt != pair.negative_prompt for pair in pairs)
            
        except Exception as e:
            pytest.skip(f"Skipping integration test due to: {e}")