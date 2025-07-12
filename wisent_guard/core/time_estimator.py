"""
Time estimation for optimization tasks.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class OptimizationTimeEstimator:
    """Estimates time for full optimization pipeline based on model and task counts."""
    
    # Empirical timing data (in seconds per task)
    TIMING_DATA = {
        "distilgpt2": {
            "classification": {
                "base": 15,  # Base time per task
                "per_sample": 0.1,  # Additional time per sample
                "per_layer": 2,  # Additional time per layer tested
            },
            "sample_size": {
                "base": 10,
                "per_size_point": 5,  # Time per sample size tested
                "per_sample": 0.05,
            },
            "classifier_training": {
                "base": 8,
                "per_sample": 0.02,
            },
            "control_vector": {
                "base": 12,
                "per_sample": 0.03,
                "activation_extraction": 0.1,  # Per sample
            }
        },
        # Default timings for unknown models
        "default": {
            "classification": {
                "base": 20,
                "per_sample": 0.15,
                "per_layer": 3,
            },
            "sample_size": {
                "base": 15,
                "per_size_point": 8,
                "per_sample": 0.08,
            },
            "classifier_training": {
                "base": 10,
                "per_sample": 0.03,
            },
            "control_vector": {
                "base": 15,
                "per_sample": 0.04,
                "activation_extraction": 0.15,
            }
        }
    }
    
    def __init__(self, model_name: str, tasks: List[str], verbose: bool = True):
        self.model_name = model_name
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.verbose = verbose
        
        # Get timing data for this model
        self.timing = self.TIMING_DATA.get(model_name, self.TIMING_DATA["default"])
        
        # Track actual times for better estimates
        self.actual_times = {
            "classification": [],
            "sample_size": [],
            "classifier_training": [],
            "control_vector": []
        }
        
        self.start_time = None
        self.phase_start_time = None
        self.current_phase = None
        self.completed_tasks = 0
        
    def estimate_total_time(
        self,
        classification_limit: int = 200,
        sample_sizes: List[int] = None,
        sample_size_limit: int = 1000,
        skip_classification: bool = False,
        skip_sample_size: bool = False,
        skip_classifier_training: bool = False,
        skip_control_vectors: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate total time for full optimization.
        
        Returns:
            Tuple of (total_seconds, phase_breakdown)
        """
        if sample_sizes is None:
            sample_sizes = [5, 10, 20, 50, 100, 200, 500]
        
        phase_times = {}
        
        # Classification optimization
        if not skip_classification:
            # Estimate layers to test (typically tests ~5 layers)
            layers_to_test = 5
            time_per_task = (
                self.timing["classification"]["base"] +
                self.timing["classification"]["per_sample"] * classification_limit +
                self.timing["classification"]["per_layer"] * layers_to_test
            )
            phase_times["classification"] = time_per_task * self.num_tasks
        else:
            phase_times["classification"] = 0
        
        # Sample size optimization
        if not skip_sample_size:
            time_per_task = (
                self.timing["sample_size"]["base"] +
                self.timing["sample_size"]["per_size_point"] * len(sample_sizes) +
                self.timing["sample_size"]["per_sample"] * sample_size_limit
            )
            phase_times["sample_size"] = time_per_task * self.num_tasks
        else:
            phase_times["sample_size"] = 0
        
        # Classifier training
        if not skip_classifier_training:
            # Assume optimal sample size ~200
            avg_sample_size = 200
            time_per_task = (
                self.timing["classifier_training"]["base"] +
                self.timing["classifier_training"]["per_sample"] * avg_sample_size
            )
            phase_times["classifier_training"] = time_per_task * self.num_tasks
        else:
            phase_times["classifier_training"] = 0
        
        # Control vector training
        if not skip_control_vectors:
            avg_sample_size = 200
            time_per_task = (
                self.timing["control_vector"]["base"] +
                self.timing["control_vector"]["per_sample"] * avg_sample_size +
                self.timing["control_vector"]["activation_extraction"] * avg_sample_size
            )
            phase_times["control_vector"] = time_per_task * self.num_tasks
        else:
            phase_times["control_vector"] = 0
        
        total_time = sum(phase_times.values())
        
        return total_time, phase_times
    
    def start_optimization(self):
        """Mark the start of optimization."""
        self.start_time = time.time()
        
    def start_phase(self, phase: str):
        """Mark the start of a phase."""
        self.current_phase = phase
        self.phase_start_time = time.time()
        self.completed_tasks = 0
        
    def task_completed(self):
        """Mark a task as completed and update estimates."""
        if self.phase_start_time and self.current_phase:
            elapsed = time.time() - self.phase_start_time
            self.completed_tasks += 1
            
            # Calculate actual time per task
            actual_per_task = elapsed / self.completed_tasks
            self.actual_times[self.current_phase].append(actual_per_task)
    
    def get_phase_progress(self) -> Dict[str, any]:
        """Get current phase progress and estimates."""
        if not self.current_phase or not self.phase_start_time:
            return {}
        
        elapsed = time.time() - self.phase_start_time
        
        # Use actual times if available
        if self.actual_times[self.current_phase]:
            avg_time_per_task = sum(self.actual_times[self.current_phase]) / len(self.actual_times[self.current_phase])
        else:
            # Use initial estimate
            avg_time_per_task = elapsed / max(1, self.completed_tasks)
        
        remaining_tasks = self.num_tasks - self.completed_tasks
        estimated_remaining = avg_time_per_task * remaining_tasks
        
        return {
            "phase": self.current_phase,
            "completed": self.completed_tasks,
            "total": self.num_tasks,
            "elapsed": elapsed,
            "estimated_remaining": estimated_remaining,
            "avg_time_per_task": avg_time_per_task,
            "eta": datetime.now() + timedelta(seconds=estimated_remaining)
        }
    
    def get_overall_progress(self, phase_times: Dict[str, float]) -> Dict[str, any]:
        """Get overall optimization progress."""
        if not self.start_time:
            return {}
        
        total_elapsed = time.time() - self.start_time
        
        # Calculate completed time based on phases
        completed_time = 0
        for phase, estimated in phase_times.items():
            if phase in self.actual_times and len(self.actual_times[phase]) > 0:
                # Phase completed
                completed_time += estimated
            elif phase == self.current_phase and self.completed_tasks > 0:
                # Current phase partially complete
                progress = self.completed_tasks / self.num_tasks
                completed_time += estimated * progress
        
        total_estimated = sum(phase_times.values())
        remaining_time = max(0, total_estimated - completed_time)
        
        # Adjust based on actual vs estimated
        if completed_time > 0:
            actual_ratio = total_elapsed / max(1, completed_time)
            remaining_time *= actual_ratio
        
        return {
            "elapsed": total_elapsed,
            "estimated_total": total_estimated,
            "estimated_remaining": remaining_time,
            "percent_complete": (completed_time / total_estimated * 100) if total_estimated > 0 else 0,
            "eta": datetime.now() + timedelta(seconds=remaining_time)
        }
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    @staticmethod
    def format_eta(eta: datetime) -> str:
        """Format ETA datetime."""
        return eta.strftime("%H:%M:%S")