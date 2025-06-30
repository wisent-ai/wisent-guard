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
                print(f"   ⚠️ Error getting device benchmark estimate: {e}")
                # Fallback to hardcoded estimates if benchmarking fails
                if "benchmark" in task_name.lower() or "eval" in task_name.lower():
                    return 30.0  # 30 seconds per benchmark task
                elif "classifier" in task_name.lower() or "training" in task_name.lower():
                    return 120.0  # 2 minutes per classifier training
                elif "generation" in task_name.lower() or "synthetic" in task_name.lower():
                    return 60.0  # 1 minute per generation task
                else:
                    return 30.0  # Default fallback
        
        elif resource_type == ResourceType.MEMORY:
            return 512.0  # 512 MB default
        
        elif resource_type == ResourceType.COMPUTE:
            return 1.0  # 1 compute unit default
        
        elif resource_type == ResourceType.TOKENS:
            return 1000  # 1000 tokens default
        
        return 0.0
    
    def _get_default_task_estimates(self) -> Dict[str, TaskEstimate]:
        """Get default task estimates for common operations."""
        return {
            "benchmark_evaluation": TaskEstimate(
                task_name="benchmark_evaluation",
                time_seconds=30.0,
                memory_mb=256.0,
                tokens=500
            ),
            "classifier_training": TaskEstimate(
                task_name="classifier_training", 
                time_seconds=120.0,
                memory_mb=1024.0,
                tokens=2000
            ),
            "data_generation": TaskEstimate(
                task_name="data_generation",
                time_seconds=60.0,
                memory_mb=512.0,
                tokens=1500
            ),
            "model_loading": TaskEstimate(
                task_name="model_loading",
                time_seconds=30.0,
                memory_mb=2048.0,
                tokens=0
            )
        }


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
        print(f"   ⚠️ Error using device benchmark, falling back to budget manager: {e}")
        return _budget_manager.calculate_max_tasks_for_budget(task_type, time_budget_minutes)


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
