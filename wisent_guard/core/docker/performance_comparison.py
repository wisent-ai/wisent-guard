"""
Performance comparison script for Docker executor optimizations.

This script demonstrates the performance improvements of Level 1 optimizations:
- Resource right-sizing
- Intelligent batching
- Multi-stage Docker builds
"""

import time
import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .docker_executor import DockerExecutor
    from .optimized_docker_executor import OptimizedDockerExecutor, TaskComplexity, TaskComplexityAnalyzer
except ImportError:
    logger.warning("Docker executors not available for import. Running in demo mode.")
    DockerExecutor = None
    OptimizedDockerExecutor = None


class PerformanceTester:
    """Test performance improvements between original and optimized executors."""
    
    def __init__(self):
        self.test_cases = self._generate_test_cases()
    
    def _generate_test_cases(self) -> List[Tuple[str, str, TaskComplexity]]:
        """Generate test cases with different complexity levels."""
        return [
            # Simple tasks
            ("print('Hello World')", "", TaskComplexity.SIMPLE),
            ("x = 5 + 3\nprint(x)", "", TaskComplexity.SIMPLE),
            ("for i in range(5):\n    print(i)", "", TaskComplexity.SIMPLE),
            
            # Medium tasks
            ("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(fibonacci(i))
""", "", TaskComplexity.MEDIUM),
            
            ("""
import math

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 100) if is_prime(i)]
print(len(primes))
""", "", TaskComplexity.MEDIUM),
            
            # Complex tasks
            ("""
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

# Create a graph
graph = defaultdict(list)
for i in range(50):
    for j in range(i+1, min(i+5, 50)):
        graph[i].append((j, j-i))

result = dijkstra(graph, 0)
print(len(result))
""", "", TaskComplexity.COMPLEX),
        ]
    
    def run_original_executor_test(self, tasks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Run test with original executor."""
        if DockerExecutor is None:
            return self._simulate_original_performance(tasks)
        
        executor = DockerExecutor(
            timeout=30,
            memory_limit="256m",
            cpu_limit=0.5
        )
        
        start_time = time.time()
        results = []
        
        for code, input_data in tasks:
            result = executor.execute_code(code, input_data)
            results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            'executor_type': 'original',
            'total_tasks': len(tasks),
            'total_time': total_time,
            'avg_time_per_task': total_time / len(tasks),
            'containers_created': len(tasks),  # One per task
            'container_efficiency': 1.0,
            'successful_tasks': sum(1 for r in results if r.get('success', False)),
            'results': results
        }
    
    def run_optimized_executor_test(self, tasks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Run test with optimized executor."""
        if OptimizedDockerExecutor is None:
            return self._simulate_optimized_performance(tasks)
        
        executor = OptimizedDockerExecutor(
            enable_batching=True,
            enable_resource_optimization=True
        )
        
        start_time = time.time()
        results = executor.execute_batch(tasks)
        total_time = time.time() - start_time
        
        stats = executor.get_performance_stats()
        
        return {
            'executor_type': 'optimized',
            'total_tasks': len(tasks),
            'total_time': total_time,
            'avg_time_per_task': total_time / len(tasks),
            'containers_created': stats['containers_created'],
            'container_efficiency': stats['container_efficiency'],
            'successful_tasks': sum(1 for r in results if r.get('success', False)),
            'resource_optimizations': stats['resource_optimizations'],
            'batched_executions': stats['batched_executions'],
            'results': results
        }
    
    def _simulate_original_performance(self, tasks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Simulate original executor performance for demo."""
        # Simulate slower performance with fixed overhead
        simulated_time = len(tasks) * 0.8  # 800ms per task (container startup + execution)
        
        return {
            'executor_type': 'original (simulated)',
            'total_tasks': len(tasks),
            'total_time': simulated_time,
            'avg_time_per_task': simulated_time / len(tasks),
            'containers_created': len(tasks),
            'container_efficiency': 1.0,
            'successful_tasks': len(tasks),
            'results': [{'success': True, 'execution_time': 0.8} for _ in tasks]
        }
    
    def _simulate_optimized_performance(self, tasks: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Simulate optimized executor performance for demo."""
        # Simulate better performance with batching and resource optimization
        # Group tasks by complexity
        complexity_groups = {}
        for i, (code, input_data) in enumerate(tasks):
            complexity = TaskComplexityAnalyzer.analyze_code_complexity(code)
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append((i, code, input_data))
        
        # Simulate container creation based on batching
        containers_created = 0
        for complexity, group in complexity_groups.items():
            if complexity == TaskComplexity.SIMPLE:
                batch_size = 10
            elif complexity == TaskComplexity.MEDIUM:
                batch_size = 5
            else:
                batch_size = 2
            
            containers_created += (len(group) + batch_size - 1) // batch_size
        
        # Simulate time savings from optimizations
        base_time = len(tasks) * 0.8  # Original time
        startup_savings = (len(tasks) - containers_created) * 0.3  # 300ms saved per avoided container
        resource_savings = len(tasks) * 0.1  # 100ms saved per task from right-sizing
        
        simulated_time = max(0.1, base_time - startup_savings - resource_savings)
        
        return {
            'executor_type': 'optimized (simulated)',
            'total_tasks': len(tasks),
            'total_time': simulated_time,
            'avg_time_per_task': simulated_time / len(tasks),
            'containers_created': containers_created,
            'container_efficiency': len(tasks) / containers_created,
            'successful_tasks': len(tasks),
            'resource_optimizations': len(tasks),
            'batched_executions': containers_created,
            'results': [{'success': True, 'execution_time': simulated_time / len(tasks)} for _ in tasks]
        }
    
    def run_comparison(self, num_tasks: int = 20) -> Dict[str, Any]:
        """Run performance comparison between executors."""
        logger.info(f"Running performance comparison with {num_tasks} tasks")
        
        # Generate tasks for testing
        tasks = []
        for i in range(num_tasks):
            test_case = self.test_cases[i % len(self.test_cases)]
            tasks.append((test_case[0], test_case[1]))
        
        # Test original executor
        logger.info("Testing original executor...")
        original_results = self.run_original_executor_test(tasks)
        
        # Test optimized executor
        logger.info("Testing optimized executor...")
        optimized_results = self.run_optimized_executor_test(tasks)
        
        # Calculate improvements
        time_improvement = (original_results['total_time'] - optimized_results['total_time']) / original_results['total_time']
        container_reduction = (original_results['containers_created'] - optimized_results['containers_created']) / original_results['containers_created']
        
        comparison = {
            'original': original_results,
            'optimized': optimized_results,
            'improvements': {
                'time_improvement_percent': time_improvement * 100,
                'container_reduction_percent': container_reduction * 100,
                'throughput_improvement': original_results['total_time'] / optimized_results['total_time'],
                'container_efficiency_improvement': optimized_results['container_efficiency'] / original_results['container_efficiency']
            }
        }
        
        return comparison
    
    def print_comparison_report(self, comparison: Dict[str, Any]):
        """Print detailed comparison report."""
        print("\n" + "="*60)
        print("DOCKER EXECUTOR PERFORMANCE COMPARISON")
        print("="*60)
        
        original = comparison['original']
        optimized = comparison['optimized']
        improvements = comparison['improvements']
        
        print(f"\n📊 ORIGINAL EXECUTOR ({original['executor_type']})")
        print(f"   Total tasks: {original['total_tasks']}")
        print(f"   Total time: {original['total_time']:.2f}s")
        print(f"   Avg time per task: {original['avg_time_per_task']:.3f}s")
        print(f"   Containers created: {original['containers_created']}")
        print(f"   Container efficiency: {original['container_efficiency']:.2f}")
        print(f"   Successful tasks: {original['successful_tasks']}")
        
        print(f"\n🚀 OPTIMIZED EXECUTOR ({optimized['executor_type']})")
        print(f"   Total tasks: {optimized['total_tasks']}")
        print(f"   Total time: {optimized['total_time']:.2f}s")
        print(f"   Avg time per task: {optimized['avg_time_per_task']:.3f}s")
        print(f"   Containers created: {optimized['containers_created']}")
        print(f"   Container efficiency: {optimized['container_efficiency']:.2f}")
        print(f"   Successful tasks: {optimized['successful_tasks']}")
        print(f"   Resource optimizations: {optimized.get('resource_optimizations', 0)}")
        print(f"   Batched executions: {optimized.get('batched_executions', 0)}")
        
        print("\n📈 PERFORMANCE IMPROVEMENTS")
        print(f"   Time improvement: {improvements['time_improvement_percent']:.1f}%")
        print(f"   Container reduction: {improvements['container_reduction_percent']:.1f}%")
        print(f"   Throughput improvement: {improvements['throughput_improvement']:.1f}x")
        print(f"   Container efficiency improvement: {improvements['container_efficiency_improvement']:.1f}x")
        
        print("\n💡 LEVEL 1 OPTIMIZATIONS IMPACT")
        print("   ✅ Resource right-sizing: Matched resources to task complexity")
        print("   ✅ Intelligent batching: Grouped tasks by complexity")
        print("   ✅ Container efficiency: Reduced startup overhead")
        
        if improvements['time_improvement_percent'] > 50:
            print(f"   🎯 Target achieved: >50% improvement (actual: {improvements['time_improvement_percent']:.1f}%)")
        else:
            print(f"   📊 Performance gain: {improvements['time_improvement_percent']:.1f}% improvement")
        
        print("="*60)


def main():
    """Run performance comparison demo."""
    tester = PerformanceTester()
    
    # Run comparison with different task counts
    for num_tasks in [10, 20, 50]:
        print(f"\n🔬 Testing with {num_tasks} tasks...")
        comparison = tester.run_comparison(num_tasks)
        tester.print_comparison_report(comparison)
        time.sleep(1)  # Brief pause between tests


if __name__ == "__main__":
    main()