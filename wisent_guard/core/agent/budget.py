"""
Budget and resource management for wisent-guard agent operations.

This module provides utilities for managing time budgets, resource allocation,
and optimizing task execution within specified constraints.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import math


class ResourceType(Enum):
    """Types of resources that can be budgeted."""
    TIME = "time"
    MEMORY = "memory" 
    COMPUTE = "compute"
    TOKENS = "tokens"


@dataclass
class ResourceBudget:
    """Represents a budget for a specific resource type."""
    resource_type: ResourceType
    total_budget: float
    used_budget: float = 0.0
    unit: str = ""
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return max(0.0, self.total_budget - self.used_budget)
    
    @property
    def usage_percentage(self) -> float:
        """Calculate percentage of budget used."""
        if self.total_budget <= 0:
            return 0.0
        return (self.used_budget / self.total_budget) * 100.0
    
    def can_afford(self, cost: float) -> bool:
        """Check if we can afford a given cost."""
        return self.remaining_budget >= cost
    
    def spend(self, amount: float) -> bool:
        """Spend from the budget. Returns True if successful."""
        if self.can_afford(amount):
            self.used_budget += amount
            return True
        return False


@dataclass
class TaskEstimate:
    """Estimates for a specific task."""
    task_name: str
    time_seconds: float
    memory_mb: float = 0.0
    compute_units: float = 0.0
    tokens: int = 0
    
    def scale(self, factor: float) -> 'TaskEstimate':
        """Scale all estimates by a factor."""
        return TaskEstimate(
            task_name=self.task_name,
            time_seconds=self.time_seconds * factor,
            memory_mb=self.memory_mb * factor,
            compute_units=self.compute_units * factor,
            tokens=int(self.tokens * factor)
        )


class BudgetManager:
    """Manages budgets and resource allocation for agent operations."""
    
    def __init__(self):
        self.budgets: Dict[ResourceType, ResourceBudget] = {}
        self.task_estimates: Dict[str, TaskEstimate] = {}
        self._default_estimates = self._get_default_task_estimates()
    
    def set_time_budget(self, minutes: float) -> None:
        """Set a time budget in minutes."""
        self.budgets[ResourceType.TIME] = ResourceBudget(
            resource_type=ResourceType.TIME,
            total_budget=minutes * 60.0,  # Convert to seconds
            unit="seconds"
        )
    
    def set_budget(self, resource_type: ResourceType, amount: float, unit: str = "") -> None:
        """Set a budget for any resource type."""
        self.budgets[resource_type] = ResourceBudget(
            resource_type=resource_type,
            total_budget=amount,
            unit=unit
        )
    
    def get_budget(self, resource_type: ResourceType) -> Optional[ResourceBudget]:
        """Get budget for a specific resource type."""
        return self.budgets.get(resource_type)
    
    def optimize_task_allocation(self, 
                               task_candidates: List[str],
                               primary_resource: ResourceType = ResourceType.TIME,
                               max_tasks: Optional[int] = None) -> List[str]:
        """
        Optimize task allocation within budget constraints.
        
        Args:
            task_candidates: List of candidate task names
            primary_resource: Primary resource to optimize for
            max_tasks: Maximum number of tasks to select
            
        Returns:
            List of selected tasks that fit within budget
        """
        budget = self.budgets.get(primary_resource)
        if not budget:
            return task_candidates[:max_tasks] if max_tasks else task_candidates
        
        # Calculate cost for each task
        task_costs = []
        for task in task_candidates:
            cost = self._estimate_task_cost(task, primary_resource)
            if cost > 0:
                task_costs.append((task, cost))
        
        # Sort by cost (ascending) to prioritize cheaper tasks
        task_costs.sort(key=lambda x: x[1])
        
        # Select tasks that fit within budget
        selected_tasks = []
        remaining_budget = budget.remaining_budget
        
        for task, cost in task_costs:
            if cost <= remaining_budget:
                selected_tasks.append(task)
                remaining_budget -= cost
                
                if max_tasks and len(selected_tasks) >= max_tasks:
                    break
        
        return selected_tasks
    
    def calculate_max_tasks_for_budget(self, 
                                     task_type: str = "default",
                                     time_budget_minutes: float = 5.0) -> int:
        """
        Calculate maximum number of tasks that can fit within a time budget.
        
        Args:
            task_type: Type of task to estimate
            time_budget_minutes: Time budget in minutes
            
        Returns:
            Maximum number of tasks
        """
        time_budget_seconds = time_budget_minutes * 60.0
        
        # Get estimate for this task type
        task_estimate = self._estimate_task_cost(task_type, ResourceType.TIME)
        
        if task_estimate <= 0:
            return 1  # Fallback to at least 1 task
        
        max_tasks = max(1, int(time_budget_seconds / task_estimate))
        return max_tasks
    
    def estimate_completion_time(self, tasks: List[str]) -> float:
        """
        Estimate total completion time for a list of tasks.
        
        Args:
            tasks: List of task names
            
        Returns:
            Estimated time in seconds
        """
        total_time = 0.0
        for task in tasks:
            total_time += self._estimate_task_cost(task, ResourceType.TIME)
        return total_time
    
    def track_task_execution(self, task_name: str, start_time: float, end_time: float) -> None:
        """
        Track actual execution time for a task to improve future estimates.
        
        Args:
            task_name: Name of the task
            start_time: Start timestamp
            end_time: End timestamp
        """
        actual_time = end_time - start_time
        
        # Update our estimates based on actual performance
        if task_name in self.task_estimates:
            # Use exponential moving average to update estimates
            current_estimate = self.task_estimates[task_name].time_seconds
            alpha = 0.3  # Learning rate
            new_estimate = alpha * actual_time + (1 - alpha) * current_estimate
            self.task_estimates[task_name].time_seconds = new_estimate
        else:
            # First time seeing this task
            self.task_estimates[task_name] = TaskEstimate(
                task_name=task_name,
                time_seconds=actual_time
            )
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get a summary of all budgets and their usage."""
        summary = {}
        for resource_type, budget in self.budgets.items():
            summary[resource_type.value] = {
                "total": budget.total_budget,
                "used": budget.used_budget,
                "remaining": budget.remaining_budget,
                "percentage_used": budget.usage_percentage,
                "unit": budget.unit
            }
        return summary
    
    def _estimate_task_cost(self, task_name: str, resource_type: ResourceType) -> float:
        """Estimate the cost of a task for a specific resource type."""
        # Check if we have a specific estimate for this task
        if task_name in self.task_estimates:
            estimate = self.task_estimates[task_name]
            if resource_type == ResourceType.TIME:
                return estimate.time_seconds
            elif resource_type == ResourceType.MEMORY:
                return estimate.memory_mb
            elif resource_type == ResourceType.COMPUTE:
                return estimate.compute_units
            elif resource_type == ResourceType.TOKENS:
                return float(estimate.tokens)
        
        # Fall back to default estimates
        return self._get_default_cost_estimate(task_name, resource_type)
    
    def _get_default_cost_estimate(self, task_name: str, resource_type: ResourceType) -> float:
        """Get default cost estimate for a task using device benchmarking."""
        if resource_type == ResourceType.TIME:
            # Use device-specific benchmarks for time estimates
            try:
                from .device_benchmarks import estimate_task_time
                
                # Map task names to benchmark types
                task_mapping = {
                    "benchmark": "benchmark_eval",
                    "eval": "benchmark_eval", 
                    "classifier": "classifier_training",
                    "training": "classifier_training",
                    "generation": "data_generation",
                    "synthetic": "data_generation",
                    "steering": "steering",
                    "model_loading": "model_loading"
                }
                
                # Find the best matching task type
                benchmark_type = None
                for pattern, task_type in task_mapping.items():
                    if pattern in task_name.lower():
                        benchmark_type = task_type
                        break
                
                if benchmark_type:
                    # Get quantity based on task type
                    if benchmark_type in ["benchmark_eval", "classifier_training"]:
                        quantity = 100  # Base unit for these tasks
                    else:
                        quantity = 1
                    
                    return estimate_task_time(benchmark_type, quantity)
                else:
                    # Use benchmark_eval as default
                    return estimate_task_time("benchmark_eval", 100)
                    
            except Exception as e:
                raise RuntimeError(f"Device benchmark estimate failed for task '{task_name}': {e}. Run device benchmark first with: python -m wisent_guard.core.agent.budget benchmark")
        
        elif resource_type == ResourceType.MEMORY:
            raise RuntimeError(f"Memory estimation not implemented for task '{task_name}'")
        
        elif resource_type == ResourceType.COMPUTE:
            raise RuntimeError(f"Compute estimation not implemented for task '{task_name}'")
        
        elif resource_type == ResourceType.TOKENS:
            raise RuntimeError(f"Token estimation not implemented for task '{task_name}'")
        
        raise RuntimeError(f"Unknown resource type: {resource_type}")
    
    def _get_default_task_estimates(self) -> Dict[str, TaskEstimate]:
        """Get default task estimates for common operations."""
        # No default estimates - all estimates must come from device benchmarks
        return {}


# Global budget manager instance
_budget_manager = BudgetManager()


def get_budget_manager() -> BudgetManager:
    """Get the global budget manager instance."""
    return _budget_manager


def set_time_budget(minutes: float) -> None:
    """Convenience function to set time budget."""
    _budget_manager.set_time_budget(minutes)


def calculate_max_tasks_for_time_budget(task_type: str = "benchmark_evaluation", 
                                       time_budget_minutes: float = 5.0) -> int:
    """
    Calculate maximum number of tasks that can fit within a time budget.
    
    Args:
        task_type: Type of task to estimate (benchmark_evaluation, classifier_training, etc.)
        time_budget_minutes: Time budget in minutes
        
    Returns:
        Maximum number of tasks
    """
    # Use device benchmarking for more accurate estimates
    try:
        from .device_benchmarks import estimate_task_time
        
        # Map task types to benchmark types
        benchmark_mapping = {
            "benchmark_evaluation": "benchmark_eval",
            "classifier_training": "classifier_training",
            "data_generation": "data_generation",
            "steering": "steering",
            "model_loading": "model_loading"
        }
        
        benchmark_type = benchmark_mapping.get(task_type, "benchmark_eval")
        
        # Get time per task
        if benchmark_type in ["benchmark_eval", "classifier_training"]:
            time_per_task = estimate_task_time(benchmark_type, 100) / 100  # Per unit
        else:
            time_per_task = estimate_task_time(benchmark_type, 1)
        
        time_budget_seconds = time_budget_minutes * 60.0
        max_tasks = max(1, int(time_budget_seconds / time_per_task))
        
        return max_tasks
        
    except Exception as e:
        raise RuntimeError(f"Budget calculation failed for task '{task_type}': {e}. Run device benchmark first with: python -m wisent_guard.core.agent.budget benchmark")


def optimize_tasks_for_budget(task_candidates: List[str], 
                            time_budget_minutes: float = 5.0,
                            max_tasks: Optional[int] = None) -> List[str]:
    """
    Optimize task selection within a time budget.
    
    Args:
        task_candidates: List of candidate task names
        time_budget_minutes: Time budget in minutes
        max_tasks: Maximum number of tasks to select
        
    Returns:
        List of selected tasks that fit within budget
    """
    _budget_manager.set_time_budget(time_budget_minutes)
    return _budget_manager.optimize_task_allocation(
        task_candidates, 
        ResourceType.TIME, 
        max_tasks
    )


def optimize_benchmarks_for_budget(task_candidates: List[str], 
                                 time_budget_minutes: float = 5.0,
                                 max_tasks: Optional[int] = None,
                                 prefer_fast: bool = False) -> List[str]:
    """
    Optimize benchmark selection within a time budget using priority and loading time data.
    
    Args:
        task_candidates: List of candidate benchmark names
        time_budget_minutes: Time budget in minutes
        max_tasks: Maximum number of tasks to select
        prefer_fast: Whether to prefer fast benchmarks
        
    Returns:
        List of selected benchmarks that fit within budget
    """
    try:
        # Import benchmark data
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lm-harness-integration'))
        from only_benchmarks import BENCHMARKS
        
        # Get benchmark information with loading times
        benchmark_info = []
        for task in task_candidates:
            if task in BENCHMARKS:
                config = BENCHMARKS[task]
                loading_time = config.get('loading_time', 60.0)  # seconds
                priority = config.get('priority', 'unknown')
                
                # Calculate priority score for selection
                priority_score = 0
                if priority == 'high':
                    priority_score = 3
                elif priority == 'medium':
                    priority_score = 2
                elif priority == 'low':
                    priority_score = 1
                
                # Calculate efficiency score (priority per second)
                efficiency_score = priority_score / max(loading_time, 1.0)
                
                benchmark_info.append({
                    'task': task,
                    'loading_time': loading_time,
                    'priority': priority,
                    'priority_score': priority_score,
                    'efficiency_score': efficiency_score
                })
            else:
                # Fallback for unknown benchmarks
                benchmark_info.append({
                    'task': task,
                    'loading_time': 60.0,
                    'priority': 'unknown',
                    'priority_score': 0,
                    'efficiency_score': 0.0
                })
        
        # Sort by efficiency (prefer fast) or priority (prefer high priority)
        if prefer_fast:
            benchmark_info.sort(key=lambda x: x['efficiency_score'], reverse=True)
        else:
            benchmark_info.sort(key=lambda x: (x['priority_score'], -x['loading_time']), reverse=True)
        
        # Select benchmarks that fit within budget
        selected_benchmarks = []
        total_time = 0.0
        time_budget_seconds = time_budget_minutes * 60.0
        
        for info in benchmark_info:
            if total_time + info['loading_time'] <= time_budget_seconds:
                selected_benchmarks.append(info['task'])
                total_time += info['loading_time']
                
                if max_tasks and len(selected_benchmarks) >= max_tasks:
                    break
        
        return selected_benchmarks
        
    except Exception as e:
        print(f"   ⚠️ Priority-aware budget optimization failed: {e}")
        print(f"   🔄 Falling back to basic budget optimization...")
        return optimize_tasks_for_budget(task_candidates, time_budget_minutes, max_tasks)


def estimate_completion_time_minutes(tasks: List[str]) -> float:
    """
    Estimate total completion time for tasks in minutes.
    
    Args:
        tasks: List of task names
        
    Returns:
        Estimated time in minutes
    """
    seconds = _budget_manager.estimate_completion_time(tasks)
    return seconds / 60.0


def track_task_performance(task_name: str, start_time: float, end_time: float) -> None:
    """
    Track actual task performance to improve future estimates.
    
    Args:
        task_name: Name of the task
        start_time: Start timestamp
        end_time: End timestamp
    """
    _budget_manager.track_task_execution(task_name, start_time, end_time)


def run_device_benchmark(force_rerun: bool = False) -> None:
    """
    Run device performance benchmark and save results.
    
    Args:
        force_rerun: Force re-run even if cached results exist
    """
    from .device_benchmarks import ensure_benchmark_exists
    
    print("🚀 Running device performance benchmark...")
    benchmark = ensure_benchmark_exists(force_rerun=force_rerun)
    
    print("\n✅ Benchmark Results:")
    print("=" * 50)
    print(f"Device ID: {benchmark.device_id[:12]}...")
    print(f"Device Type: {benchmark.device_type}")
    print(f"Model Loading: {benchmark.model_loading_seconds:.1f}s")
    print(f"Evaluation: {benchmark.benchmark_eval_seconds_per_100_examples:.1f}s per 100 examples")
    print(f"Classifier Training: {benchmark.classifier_training_seconds_per_100_samples:.1f}s per 100 samples")
    print(f"Steering: {benchmark.steering_seconds_per_example:.1f}s per example")
    print(f"Data Generation: {benchmark.data_generation_seconds_per_example:.1f}s per example")
    print(f"\nResults saved to: device_benchmarks.json")
    
    # Show some example estimates
    print("\n📊 Example Time Estimates:")
    print("-" * 30)
    print(f"Loading model: {benchmark.model_loading_seconds:.1f}s")
    print(f"100 eval examples: {benchmark.benchmark_eval_seconds_per_100_examples:.1f}s")
    print(f"Training classifier (200 samples): {(benchmark.classifier_training_seconds_per_100_samples * 2):.1f}s")
    print(f"10 steering examples: {(benchmark.steering_seconds_per_example * 10):.1f}s")


def get_device_info() -> Dict[str, str]:
    """Get current device information."""
    from .device_benchmarks import get_current_device_info
    return get_current_device_info()


def estimate_task_time_direct(task_type: str, quantity: int = 1) -> float:
    """
    Direct estimate of task time using device benchmarks.
    
    Args:
        task_type: Type of task ("model_loading", "benchmark_eval", etc.)
        quantity: Number of items
        
    Returns:
        Estimated time in seconds
    """
    from .device_benchmarks import estimate_task_time
    return estimate_task_time(task_type, quantity)


# CLI functionality for budget management
def main():
    """CLI entry point for budget management and benchmarking."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="wisent-guard budget management and device benchmarking"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run device benchmark')
    benchmark_parser.add_argument(
        '--force', '-f', 
        action='store_true', 
        help='Force re-run benchmark even if cached results exist'
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show device information')
    
    # Estimate command
    estimate_parser = subparsers.add_parser('estimate', help='Estimate task time')
    estimate_parser.add_argument('task_type', help='Type of task')
    estimate_parser.add_argument('quantity', type=int, help='Number of items')
    
    # Budget command
    budget_parser = subparsers.add_parser('budget', help='Calculate budget allocations')
    budget_parser.add_argument('--time-minutes', '-t', type=float, default=5.0, help='Time budget in minutes')
    budget_parser.add_argument('--task-type', default='benchmark_evaluation', help='Task type to optimize for')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'benchmark':
            run_device_benchmark(force_rerun=args.force)
            
        elif args.command == 'info':
            print("🖥️ Current Device Information")
            print("=" * 40)
            device_info = get_device_info()
            for key, value in device_info.items():
                print(f"{key}: {value}")
                
        elif args.command == 'estimate':
            estimated_seconds = estimate_task_time_direct(args.task_type, args.quantity)
            print(f"⏱️ Estimated time for {args.quantity}x {args.task_type}: {estimated_seconds:.1f} seconds ({estimated_seconds/60:.2f} minutes)")
            
        elif args.command == 'budget':
            max_tasks = calculate_max_tasks_for_time_budget(args.task_type, args.time_minutes)
            
            # Map task types to benchmark types for direct estimation
            benchmark_mapping = {
                "benchmark_evaluation": "benchmark_eval",
                "classifier_training": "classifier_training", 
                "data_generation": "data_generation",
                "steering": "steering",
                "model_loading": "model_loading"
            }
            
            benchmark_type = benchmark_mapping.get(args.task_type, "benchmark_eval")
            
            # Get time per individual task unit
            if benchmark_type in ["benchmark_eval", "classifier_training"]:
                task_time = estimate_task_time_direct(benchmark_type, 100) / 100  # Per unit
            else:
                task_time = estimate_task_time_direct(benchmark_type, 1)
            
            total_time = max_tasks * task_time
            
            print(f"💰 Budget Analysis:")
            print(f"Time budget: {args.time_minutes:.1f} minutes ({args.time_minutes * 60:.0f} seconds)")
            print(f"Task type: {args.task_type} (mapped to {benchmark_type})")
            print(f"Time per task: {task_time:.2f} seconds")
            print(f"Max tasks: {max_tasks}")
            print(f"Total estimated time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
            print(f"Budget utilization: {(total_time / (args.time_minutes * 60)) * 100:.1f}%")
            
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
