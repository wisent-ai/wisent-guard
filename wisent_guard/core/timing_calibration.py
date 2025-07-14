"""Runtime timing calibration for optimization time estimation"""
import time
import json
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from .hyperparameter_optimizer import HyperparameterOptimizer, OptimizationConfig
from .model_config_manager import ModelConfigManager


class TimingCalibrator:
    """Measures actual optimization timing on the current system"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.timings = {
            "per_task_per_sample": None,  # seconds per task per sample
            "per_layer": None,  # seconds per layer tested
            "classifier_training_per_sample": None,  # seconds per training sample
            "control_vector_per_layer": None,  # seconds per layer for control vectors
        }
    
    def run_calibration(
        self,
        model_name: str,
        calibration_tasks: list = None,
        calibration_samples: int = 20,
        calibration_layers: int = 2
    ) -> Dict[str, float]:
        """
        Run calibration tests to measure actual timing.
        
        Args:
            model_name: Model to calibrate timing for
            calibration_tasks: Tasks to use for calibration (default: small subset)
            calibration_samples: Number of samples to test with
            calibration_layers: Number of layers to test
            
        Returns:
            Dictionary of timing measurements
        """
        if calibration_tasks is None:
            # Use a small, fast subset of tasks for calibration
            calibration_tasks = ["arc_easy", "hellaswag"]
        
        if self.verbose:
            print(f"\n🔧 Running timing calibration for {model_name}...")
            print(f"   Tasks: {calibration_tasks}")
            print(f"   Samples: {calibration_samples}")
            print(f"   Layers: {calibration_layers}")
        
        # Get total layers in model
        try:
            from . import Model
            model = Model(name=model_name)
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                if hasattr(model.model.config, 'num_hidden_layers'):
                    total_layers = model.model.config.num_hidden_layers
                elif hasattr(model.model.config, 'n_layer'):
                    total_layers = model.model.config.n_layer
                else:
                    total_layers = 24  # Default estimate
            else:
                total_layers = 24  # Default estimate
        except:
            total_layers = 24  # Default estimate
        
        # Sample a subset of layers for calibration
        layer_indices = np.linspace(0, total_layers - 1, calibration_layers, dtype=int)
        
        # Measure classification optimization time
        start_time = time.time()
        
        # Run a small optimization using HyperparameterOptimizer
        try:
            from wisent_guard.core import Model
            model = Model(name=model_name)
            
            # Create optimization config for calibration
            config = OptimizationConfig(
                layer_range=(layer_indices[0], layer_indices[-1]),
                aggregation_methods=["average"],  # Just test one method for speed
                threshold_range=[0.5],  # Just test one threshold for speed
                max_time_per_task_seconds=60  # 1 minute max for calibration
            )
            
            # Test optimization on one task
            if calibration_tasks:
                test_task = calibration_tasks[0]
                optimizer = HyperparameterOptimizer(
                    model=model,
                    task_name=test_task,
                    limit=calibration_samples,
                    device=None,
                    verbose=False
                )
                
                # Run optimization
                try:
                    optimizer.optimize(config)
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  Calibration task failed: {e}")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Calibration error: {e}")
        
        classification_time = time.time() - start_time
        
        # Calculate timing metrics
        total_operations = len(calibration_tasks) * calibration_layers * calibration_samples
        self.timings["per_task_per_sample"] = classification_time / (len(calibration_tasks) * calibration_samples)
        self.timings["per_layer"] = classification_time / calibration_layers
        
        # Estimate other timings based on classification timing
        # These are rough estimates based on typical ratios
        self.timings["classifier_training_per_sample"] = self.timings["per_task_per_sample"] * 0.1
        self.timings["control_vector_per_layer"] = self.timings["per_layer"] * 0.5
        
        if self.verbose:
            print(f"\n✅ Calibration complete in {classification_time:.1f}s")
            print(f"   Per task per sample: {self.timings['per_task_per_sample']:.3f}s")
            print(f"   Per layer: {self.timings['per_layer']:.3f}s")
        
        return self.timings
    
    def save_to_file(self, filepath: Path):
        """Save calibration results to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.timings, f, indent=2)
        if self.verbose:
            print(f"💾 Saved calibration to {filepath}")
    
    def load_from_file(self, filepath: Path) -> bool:
        """Load calibration results from a file"""
        if not filepath.exists():
            return False
        
        try:
            with open(filepath, 'r') as f:
                self.timings = json.load(f)
            if self.verbose:
                print(f"📂 Loaded calibration from {filepath}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load calibration: {e}")
            return False
    
    def estimate_optimization_time(
        self,
        num_tasks: int,
        num_layers: int,
        samples_per_task: int = 1000,
        include_sample_size_opt: bool = True,
        include_classifier_training: bool = True,
        include_control_vectors: bool = True,
        num_cv_layers: int = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate total optimization time based on calibration.
        
        Returns:
            Tuple of (total_seconds, breakdown_dict)
        """
        if self.timings["per_task_per_sample"] is None:
            # Use fallback timings if no calibration available
            self.timings = {
                "per_task_per_sample": 0.005,
                "per_layer": 5.0,
                "classifier_training_per_sample": 0.002,
                "control_vector_per_layer": 10.0,
            }
        
        breakdown = {}
        
        # Classification optimization time
        classification_time = (
            num_tasks * num_layers * samples_per_task * self.timings["per_task_per_sample"]
        )
        breakdown["classification"] = classification_time
        
        # Sample size optimization (if enabled)
        if include_sample_size_opt:
            # Typically tests 3-5 different sample sizes
            sample_size_time = num_tasks * 4 * 100 * self.timings["per_task_per_sample"]
            breakdown["sample_size"] = sample_size_time
        else:
            breakdown["sample_size"] = 0
        
        # Classifier training time (if enabled)
        if include_classifier_training:
            # Training uses the optimized sample size
            training_samples = min(samples_per_task, 500)  # Typically capped
            classifier_time = num_tasks * training_samples * self.timings["classifier_training_per_sample"]
            breakdown["classifier_training"] = classifier_time
        else:
            breakdown["classifier_training"] = 0
        
        # Control vector generation (if enabled)
        if include_control_vectors:
            cv_layers = num_cv_layers or num_layers
            cv_time = cv_layers * self.timings["control_vector_per_layer"]
            breakdown["control_vectors"] = cv_time
        else:
            breakdown["control_vectors"] = 0
        
        total_time = sum(breakdown.values())
        
        return total_time, breakdown