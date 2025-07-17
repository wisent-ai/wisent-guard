"""Tests for the optimized Docker executor (Level 1 optimizations)."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from wisent_guard.core.docker.optimized_docker_executor import (
    OptimizedDockerExecutor,
    TaskComplexity,
    ResourceProfile,
    TaskComplexityAnalyzer
)


class TestTaskComplexityAnalyzer:
    """Test task complexity analysis."""
    
    def test_simple_code_complexity(self):
        """Test simple code is classified correctly."""
        simple_code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
"""
        complexity = TaskComplexityAnalyzer.analyze_code_complexity(simple_code)
        assert complexity == TaskComplexity.SIMPLE
    
    def test_medium_code_complexity(self):
        """Test medium complexity code."""
        medium_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

numbers = [1, 2, 3, 4, 5]
result = [fibonacci(x) for x in numbers]
print(result)
"""
        complexity = TaskComplexityAnalyzer.analyze_code_complexity(medium_code)
        assert complexity in [TaskComplexity.MEDIUM, TaskComplexity.COMPLEX]
    
    def test_complex_code_complexity(self):
        """Test complex code is classified correctly."""
        complex_code = """
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current_dist > distances[current]:
            continue
            
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

# Complex graph processing
graph = defaultdict(list)
for i in range(100):
    for j in range(i+1, min(i+10, 100)):
        graph[i].append((j, j-i))

result = dijkstra(graph, 0)
print(len(result))
"""
        complexity = TaskComplexityAnalyzer.analyze_code_complexity(complex_code)
        assert complexity == TaskComplexity.COMPLEX


class TestResourceProfile:
    """Test resource profile allocation."""
    
    def test_simple_resource_profile(self):
        """Test simple task resource profile."""
        profile = ResourceProfile.for_complexity(TaskComplexity.SIMPLE)
        assert profile.memory_limit == "64m"
        assert profile.cpu_limit == 0.25
        assert profile.timeout == 5
        assert profile.batch_size == 10
    
    def test_medium_resource_profile(self):
        """Test medium task resource profile."""
        profile = ResourceProfile.for_complexity(TaskComplexity.MEDIUM)
        assert profile.memory_limit == "128m"
        assert profile.cpu_limit == 0.5
        assert profile.timeout == 10
        assert profile.batch_size == 5
    
    def test_complex_resource_profile(self):
        """Test complex task resource profile."""
        profile = ResourceProfile.for_complexity(TaskComplexity.COMPLEX)
        assert profile.memory_limit == "256m"
        assert profile.cpu_limit == 1.0
        assert profile.timeout == 30
        assert profile.batch_size == 2


class TestOptimizedDockerExecutor:
    """Test optimized Docker executor."""
    
    @pytest.fixture
    def mock_docker_available(self):
        """Mock Docker availability check."""
        with patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            yield mock_run
    
    @pytest.fixture
    def mock_image_exists(self):
        """Mock Docker image existence check."""
        with patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run') as mock_run:
            mock_run.return_value.stdout = "image_id_123\n"
            yield mock_run
    
    @pytest.fixture
    def executor(self, mock_docker_available, mock_image_exists):
        """Create executor instance for testing."""
        return OptimizedDockerExecutor(
            build_if_missing=False,
            enable_batching=True,
            enable_resource_optimization=True
        )
    
    def test_executor_initialization(self, executor):
        """Test executor initializes correctly."""
        assert executor.enable_batching is True
        assert executor.enable_resource_optimization is True
        assert executor.execution_stats['total_tasks'] == 0
    
    def test_performance_stats_tracking(self, executor):
        """Test performance statistics tracking."""
        stats = executor.get_performance_stats()
        assert 'total_tasks' in stats
        assert 'containers_created' in stats
        assert 'batched_executions' in stats
        assert 'resource_optimizations' in stats
        assert 'total_time' in stats
        assert 'avg_time_per_task' in stats
        assert 'container_efficiency' in stats
    
    def test_stats_reset(self, executor):
        """Test statistics reset."""
        executor.execution_stats['total_tasks'] = 5
        executor.reset_stats()
        assert executor.execution_stats['total_tasks'] == 0
    
    @patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run')
    @patch('wisent_guard.core.docker.optimized_docker_executor.tempfile.NamedTemporaryFile')
    def test_execute_single_with_resource_optimization(self, mock_tempfile, mock_run, executor):
        """Test single execution with resource optimization."""
        # Mock file operations
        mock_file = Mock()
        mock_file.name = "/tmp/test_code.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file
        
        # Mock successful execution
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Hello World"
        mock_run.return_value.stderr = ""
        
        simple_code = "print('Hello World')"
        result = executor.execute_single(simple_code)
        
        assert result['success'] is True
        assert result['stdout'] == "Hello World"
        assert 'memory_limit' in result
        assert 'cpu_limit' in result
        assert executor.execution_stats['total_tasks'] == 1
        assert executor.execution_stats['resource_optimizations'] == 1
    
    @patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run')
    @patch('wisent_guard.core.docker.optimized_docker_executor.tempfile.NamedTemporaryFile')
    def test_execute_batch_with_complexity_grouping(self, mock_tempfile, mock_run, executor):
        """Test batch execution with complexity grouping."""
        # Mock file operations
        mock_file = Mock()
        mock_file.name = "/tmp/test_code.py"
        mock_tempfile.return_value.__enter__.return_value = mock_file
        
        # Mock successful execution
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Output"
        mock_run.return_value.stderr = ""
        
        # Mix of simple and complex tasks
        tasks = [
            ("print('simple')", ""),
            ("print('also simple')", ""),
            ("""
import heapq
def complex_algo():
    heap = []
    for i in range(100):
        heapq.heappush(heap, i)
    return heap
print(complex_algo())
""", "")
        ]
        
        results = executor.execute_batch(tasks)
        
        assert len(results) == 3
        assert all(result['success'] for result in results)
        assert executor.execution_stats['total_tasks'] == 3
        assert executor.execution_stats['batched_executions'] > 0
    
    @patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run')
    def test_timeout_handling(self, mock_run, executor):
        """Test timeout handling."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired("docker", 5)
        
        result = executor.execute_single("while True: pass")
        
        assert result['success'] is False
        assert 'timed out' in result['error']
        assert result['return_code'] == 124
    
    def test_docker_not_available(self):
        """Test handling when Docker is not available."""
        with patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()
            
            with pytest.raises(RuntimeError, match="Docker is not installed"):
                OptimizedDockerExecutor(build_if_missing=False)
    
    def test_docker_not_running(self):
        """Test handling when Docker is not running."""
        with patch('wisent_guard.core.docker.optimized_docker_executor.subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            
            with pytest.raises(RuntimeError, match="Docker is not running"):
                OptimizedDockerExecutor(build_if_missing=False)


class TestPerformanceComparison:
    """Test performance improvements."""
    
    @pytest.fixture
    def original_executor(self):
        """Mock original executor for comparison."""
        return Mock()
    
    @pytest.fixture
    def optimized_executor(self, mock_docker_available, mock_image_exists):
        """Create optimized executor for comparison."""
        return OptimizedDockerExecutor(
            build_if_missing=False,
            enable_batching=True,
            enable_resource_optimization=True
        )
    
    def test_container_efficiency_improvement(self, optimized_executor):
        """Test that batching improves container efficiency."""
        # Mock execution for testing
        with patch.object(optimized_executor, '_execute_with_profile') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'stdout': 'test',
                'stderr': '',
                'return_code': 0,
                'execution_time': 0.1
            }
            
            # Execute 10 simple tasks
            tasks = [("print('test')", "")] * 10
            results = optimized_executor.execute_batch(tasks)
            
            stats = optimized_executor.get_performance_stats()
            
            # Should have better efficiency than 1 container per task
            assert stats['container_efficiency'] > 1.0
            assert stats['total_tasks'] == 10
            assert stats['containers_created'] < 10  # Fewer containers due to batching
    
    def test_resource_right_sizing_benefits(self, optimized_executor):
        """Test that resource right-sizing provides appropriate allocations."""
        simple_code = "print('hello')"
        complex_code = """
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        if current_dist > distances[current]:
            continue
        for neighbor, weight in graph[current]:
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    
    return distances

graph = defaultdict(list)
for i in range(100):
    for j in range(i+1, min(i+10, 100)):
        graph[i].append((j, j-i))

result = dijkstra(graph, 0)
print(len(result))
"""
        
        # Analyze complexity
        simple_complexity = TaskComplexityAnalyzer.analyze_code_complexity(simple_code)
        complex_complexity = TaskComplexityAnalyzer.analyze_code_complexity(complex_code)
        
        # Get resource profiles
        simple_profile = ResourceProfile.for_complexity(simple_complexity)
        complex_profile = ResourceProfile.for_complexity(complex_complexity)
        
        # Verify resource right-sizing
        assert simple_profile.memory_limit < complex_profile.memory_limit
        assert simple_profile.cpu_limit <= complex_profile.cpu_limit
        assert simple_profile.timeout <= complex_profile.timeout
        assert simple_profile.batch_size >= complex_profile.batch_size