"""Runtime timing calibration for optimization time estimation"""
import time
import json
import subprocess
import sys
from typing import Dict, Optional, Tuple
from pathlib import Path


class TimingCalibrator:
    """Measures actual optimization timing on the current system"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.timings = {
            "training_time": None,  # Time for training command
            "steering_time": None,  # Time for steering command
        }
    
    def run_calibration(self, model_name: str) -> Dict[str, float]:
        """
        Run calibration by measuring training and steering times.
        Uses exactly one task, one layer, and 10 samples.
        
        Args:
            model_name: Model to calibrate timing for
            
        Returns:
            Dictionary with training_time and steering_time
        """
        # Get model layer count
        from . import Model
        model = Model(name=model_name)
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            if hasattr(model.model.config, 'num_hidden_layers'):
                total_layers = model.model.config.num_hidden_layers
            elif hasattr(model.model.config, 'n_layer'):
                total_layers = model.model.config.n_layer
            else:
                raise RuntimeError(f"Cannot determine number of layers for model {model_name}")
        else:
            raise RuntimeError(f"Cannot access model config for {model_name}")
        
        # Use middle layer for calibration
        calibration_layer = total_layers // 2
        
        if self.verbose:
            print(f"\n🔧 Running timing calibration for {model_name}...")
            print(f"   Task: arc_easy")
            print(f"   Layer: {calibration_layer}")
            print(f"   Samples: 10")
        
        # 1. Measure training time
        if self.verbose:
            print(f"\n📊 Measuring training time...")
        
        start_time = time.time()
        
        cmd = [
            sys.executable, "-m", "wisent_guard.cli", 
            "optimize-classification",
            model_name,
            "--tasks", "arc_easy",
            "--limit", "10",
            "--layer-range", f"{calibration_layer},{calibration_layer}",
            "--no-save",
            "--skip-timing-estimation"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training calibration failed:\n{result.stderr}")
            
        self.timings["training_time"] = time.time() - start_time
        
        # 2. Skip steering calibration - it's too complex for quick calibration
        # Users can run steering separately if needed
        if self.verbose:
            print(f"\n📊 Skipping steering calibration (too complex for quick estimate)")
        
        self.timings["steering_time"] = None
        
        if self.verbose:
            print(f"\n✅ Calibration complete!")
            print(f"   Training time: {self.timings['training_time']:.3f}s")
            print(f"   Steering time: {self.timings['steering_time']:.3f}s")
        
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
        sample_sizes: list = None,
        sample_size_limit: int = 1000,
        include_sample_size_opt: bool = True,
        include_classifier_training: bool = True,
        include_control_vectors: bool = True,
        num_cv_layers: int = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate total optimization time based on calibration.
        
        Linear scaling from base measurements: 1 task, 1 layer, 10 samples.
        
        Returns:
            Tuple of (total_seconds, breakdown_dict)
        """
        if self.timings["training_time"] is None:
            raise RuntimeError("No calibration data available. Run calibration first.")
        
        # Base measurements from calibration
        base_training = self.timings["training_time"]  # Time for 1 task, 1 layer, 10 samples
        base_steering = self.timings["steering_time"]  # Time for 1 task, 1 layer, 10 samples
        
        breakdown = {}
        
        # Classification optimization: scales linearly with tasks, layers, and samples
        classification_time = base_training * num_tasks * num_layers * (samples_per_task / 10)
        breakdown["classification"] = classification_time
        
        # Sample size optimization: tests multiple sample sizes on ONE layer per task
        if include_sample_size_opt and sample_sizes:
            # Calculate average sample size from the provided list
            avg_sample_size = sum(sample_sizes) / len(sample_sizes)
            # Each test uses sample_size_limit samples from the dataset
            sample_size_time = base_training * num_tasks * len(sample_sizes) * (min(avg_sample_size, sample_size_limit) / 10)
            breakdown["sample_size"] = sample_size_time
        else:
            breakdown["sample_size"] = 0
        
        # Classifier training: one run per task with full samples
        if include_classifier_training:
            classifier_time = base_training * num_tasks * (samples_per_task / 10)
            breakdown["classifier_training"] = classifier_time
        else:
            breakdown["classifier_training"] = 0
        
        # Control vector generation: skip if no steering calibration
        if include_control_vectors and base_steering is not None:
            cv_layers = num_cv_layers or num_layers
            control_vectors_time = base_steering * num_tasks * cv_layers * (samples_per_task / 10)
            breakdown["control_vectors"] = control_vectors_time
        else:
            breakdown["control_vectors"] = 0
        
        total_time = sum(breakdown.values())
        
        return total_time, breakdown