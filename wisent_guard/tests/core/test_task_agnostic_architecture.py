"""
Tests for task-agnostic architecture.
"""

import pytest
from wisent_guard.core.task_interface import TaskInterface, TaskRegistry, register_task, get_task, list_tasks
from wisent_guard.core.tasks.livecodebench_task import LiveCodeBenchTask
from wisent_guard.core.benchmark_extractors import BenchmarkExtractor


class MockTask(TaskInterface):
    """Mock task for testing."""
    
    def load_data(self, limit=None):
        data = [{"id": i, "content": f"Mock data {i}"} for i in range(10)]
        return data[:limit] if limit else data
    
    def get_extractor(self):
        from wisent_guard.core.benchmark_extractors import DefaultExtractor
        return DefaultExtractor()
    
    def get_name(self):
        return "mock_task"
    
    def get_description(self):
        return "Mock task for testing"
    
    def get_categories(self):
        return ["test", "mock"]


class TestTaskInterface:
    """Test TaskInterface and registry functionality."""
    
    def test_task_registry_basic_operations(self):
        """Test basic task registry operations."""
        registry = TaskRegistry()
        
        # Test registration
        registry.register_task("test_task", MockTask)
        assert "test_task" in registry.list_tasks()
        
        # Test getting task
        task = registry.get_task("test_task")
        assert isinstance(task, MockTask)
        assert task.get_name() == "mock_task"
        
        # Test getting non-existent task
        with pytest.raises(ValueError, match="Task 'nonexistent' not found"):
            registry.get_task("nonexistent")
    
    def test_task_registry_info(self):
        """Test task information retrieval."""
        registry = TaskRegistry()
        registry.register_task("test_task", MockTask)
        
        # Test task info
        info = registry.get_task_info("test_task")
        assert info["name"] == "mock_task"
        assert info["description"] == "Mock task for testing"
        assert info["categories"] == ["test", "mock"]
        
        # Test list all task info
        all_info = registry.list_task_info()
        assert len(all_info) == 1
        assert all_info[0]["name"] == "mock_task"
    
    def test_global_task_registry(self):
        """Test global task registry functions."""
        # Register a task globally
        register_task("global_test", MockTask)
        
        # Test that it's available globally
        assert "global_test" in list_tasks()
        
        # Test getting the task
        task = get_task("global_test")
        assert isinstance(task, MockTask)


class TestLiveCodeBenchTask:
    """Test LiveCodeBenchTask implementation."""
    
    def test_livecodebench_task_basic_properties(self):
        """Test basic properties of LiveCodeBenchTask."""
        task = LiveCodeBenchTask()
        
        assert task.get_name() == "livecodebench"
        assert "LiveCodeBench" in task.get_description()
        assert "coding" in task.get_categories()
        assert "reasoning" in task.get_categories()
        assert "algorithms" in task.get_categories()
        assert "data-structures" in task.get_categories()
    
    def test_livecodebench_task_data_loading(self):
        """Test data loading functionality."""
        task = LiveCodeBenchTask()
        
        # Test loading all data
        all_data = task.load_data()
        assert len(all_data) == 5  # Sample data has 5 items
        
        # Test loading with limit
        limited_data = task.load_data(limit=3)
        assert len(limited_data) == 3
        
        # Test data structure
        for item in all_data:
            assert "task_id" in item
            assert "question_title" in item
            assert "question_content" in item
            assert "starter_code" in item
            assert "difficulty" in item
            assert "platform" in item
    
    def test_livecodebench_task_extractor(self):
        """Test extractor functionality."""
        task = LiveCodeBenchTask()
        extractor = task.get_extractor()
        
        assert isinstance(extractor, BenchmarkExtractor)
        
        # Test with sample data
        data = task.load_data(limit=1)
        qa_pair = extractor.extract_qa_pair(data[0])
        
        assert qa_pair is not None
        assert "question" in qa_pair
        assert "formatted_question" in qa_pair
        assert "correct_answer" in qa_pair
        
        # Test contrastive pair extraction
        contrastive_pair = extractor.extract_contrastive_pair(data[0])
        assert contrastive_pair is not None
        assert "question" in contrastive_pair
        assert "correct_answer" in contrastive_pair
        assert "incorrect_answer" in contrastive_pair
    
    def test_livecodebench_task_data_content(self):
        """Test the content of LiveCodeBench sample data."""
        task = LiveCodeBenchTask()
        data = task.load_data()
        
        # Check that we have the expected sample problems
        task_ids = [item["task_id"] for item in data]
        assert "lcb_001" in task_ids  # Two Sum
        assert "lcb_002" in task_ids  # Valid Parentheses
        assert "lcb_003" in task_ids  # Longest Increasing Subsequence
        
        # Check difficulty levels
        difficulties = [item["difficulty"] for item in data]
        assert "EASY" in difficulties
        assert "MEDIUM" in difficulties
        
        # Check platforms
        platforms = [item["platform"] for item in data]
        assert "LEETCODE" in platforms
        
        # Check that starter code is present
        for item in data:
            assert "def " in item["starter_code"]
            assert "pass" in item["starter_code"]


class TestTaskAgnosticArchitecture:
    """Test the overall task-agnostic architecture."""
    
    def test_task_registration_integration(self):
        """Test that tasks register correctly on import."""
        # Import should automatically register tasks
        
        # Clear and re-register
        registry = TaskRegistry()
        registry.register_task("livecodebench", LiveCodeBenchTask)
        
        # Test that LiveCodeBench is registered
        assert "livecodebench" in registry.list_tasks()
        
        # Test that we can get the task
        task = registry.get_task("livecodebench")
        assert isinstance(task, LiveCodeBenchTask)
    
    def test_task_architecture_extensibility(self):
        """Test that the architecture is extensible."""
        # Define a custom task
        class CustomTask(TaskInterface):
            def load_data(self, limit=None):
                return [{"custom": "data"}]
            
            def get_extractor(self):
                from wisent_guard.core.benchmark_extractors import DefaultExtractor
                return DefaultExtractor()
            
            def get_name(self):
                return "custom"
            
            def get_description(self):
                return "Custom task"
            
            def get_categories(self):
                return ["custom"]
        
        # Register the custom task
        register_task("custom_task", CustomTask)
        
        # Test that it works
        task = get_task("custom_task")
        assert isinstance(task, CustomTask)
        assert task.get_name() == "custom"
        assert task.load_data() == [{"custom": "data"}]
    
    def test_task_interface_contract(self):
        """Test that all tasks follow the interface contract."""
        # Test LiveCodeBenchTask
        task = LiveCodeBenchTask()
        
        # Test required methods exist and return expected types
        assert isinstance(task.get_name(), str)
        assert isinstance(task.get_description(), str)
        assert isinstance(task.get_categories(), list)
        assert isinstance(task.load_data(), list)
        assert isinstance(task.get_extractor(), BenchmarkExtractor)
        
        # Test data loading with limit
        limited_data = task.load_data(limit=2)
        assert len(limited_data) == 2
        
        # Test extractor integration
        extractor = task.get_extractor()
        if limited_data:
            qa_pair = extractor.extract_qa_pair(limited_data[0])
            assert qa_pair is not None or qa_pair is None  # Should not raise exception


class TestTaskAgnosticCLI:
    """Test task-agnostic CLI functionality."""
    
    def test_cli_task_loading(self):
        """Test CLI task loading functionality."""
        # Register LiveCodeBench task
        register_task("test_livecodebench", LiveCodeBenchTask)
        
        # Test task loading
        task = get_task("test_livecodebench")
        assert task.get_name() == "livecodebench"
        
        # Test data loading
        data = task.load_data(limit=3)
        assert len(data) == 3
        
        # Test extractor
        extractor = task.get_extractor()
        results = []
        for item in data:
            qa_pair = extractor.extract_qa_pair(item)
            if qa_pair:
                results.append(qa_pair)
        
        assert len(results) > 0
        assert all("question" in result for result in results)
        assert all("correct_answer" in result for result in results)
    
    def test_cli_error_handling(self):
        """Test CLI error handling."""
        # Test getting non-existent task
        with pytest.raises(ValueError, match="not found"):
            get_task("nonexistent_task")
        
        # Test that the error message includes available tasks
        try:
            get_task("nonexistent_task")
        except ValueError as e:
            assert "Available tasks:" in str(e)
    
    def test_cli_task_info(self):
        """Test CLI task information display."""
        from wisent_guard.core.task_interface import list_task_info
        
        # Register a task
        register_task("info_test", LiveCodeBenchTask)
        
        # Get task info
        all_info = list_task_info()
        
        # Find our task
        task_info = None
        for info in all_info:
            if info["name"] == "livecodebench":
                task_info = info
                break
        
        assert task_info is not None
        assert task_info["name"] == "livecodebench"
        assert "LiveCodeBench" in task_info["description"]
        assert "coding" in task_info["categories"]