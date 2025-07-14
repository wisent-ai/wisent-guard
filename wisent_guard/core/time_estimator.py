"""Time estimation for optimization operations using runtime calibration"""
import time
from typing import Dict, Tuple, Optional
from pathlib import Path

from .timing_calibration import TimingCalibrator


class OptimizationTimeEstimator:
    """Estimates time required for optimization operations using calibration"""
    
    def __init__(
        self, 
        model_name: str, 
        verbose: bool = True,
        skip_calibration: bool = False,
        calibration_file: Optional[Path] = None,
        calibrate_only: bool = False
    ):
        self.model_name = model_name
        self.verbose = verbose
        self.calibrator = TimingCalibrator(verbose=verbose)
        
        # Get number of layers in the model
        from . import Model
        model = Model(name=model_name)
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            if hasattr(model.model.config, 'num_hidden_layers'):
                self.total_layers = model.model.config.num_hidden_layers
            elif hasattr(model.model.config, 'n_layer'):
                self.total_layers = model.model.config.n_layer
            else:
                raise RuntimeError(f"Cannot determine number of layers for model {model_name}")
        else:
            raise RuntimeError(f"Cannot access model configuration for {model_name}")
        
        # Handle calibration
        if skip_calibration:
            raise RuntimeError("Calibration cannot be skipped. Accurate timing requires calibration.")
        
        if calibration_file and calibration_file.exists():
            # Load from file
            if not self.calibrator.load_from_file(calibration_file):
                raise RuntimeError(f"Failed to load calibration from {calibration_file}")
            self.timing = self.calibrator.timings
            if self.timing["training_time"] is None or self.timing["steering_time"] is None:
                raise RuntimeError(f"Calibration file {calibration_file} contains invalid data")
        else:
            # Run calibration
            if verbose:
                print(f"\n🔧 Running timing calibration for {model_name}...")
            
            self.timing = self.calibrator.run_calibration(model_name)
            
            # Save calibration if file path provided
            if calibration_file:
                self.calibrator.save_to_file(calibration_file)
        
        self.calibrate_only = calibrate_only
    
    def estimate_classification_time(
        self,
        num_tasks: int,
        sample_limit: int = 200,
        layers: Optional[list] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate time for classification optimization.
        
        Returns:
            Tuple of (total_seconds, breakdown)
        """
        num_layers = len(layers) if layers else min(5, self.total_layers)
        
        total_time, breakdown = self.calibrator.estimate_optimization_time(
            num_tasks=num_tasks,
            num_layers=num_layers,
            samples_per_task=sample_limit,
            include_sample_size_opt=False,
            include_classifier_training=False,
            include_control_vectors=False
        )
        
        return total_time, {"classification": total_time}
    
    def estimate_full_optimization_time(
        self,
        num_tasks: int,
        classification_limit: int = 200,
        sample_sizes: list = None,
        sample_size_limit: int = 1000,
        include_sample_size_opt: bool = True,
        include_classifier_training: bool = True,
        include_control_vectors: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate time for full optimization pipeline.
        
        Returns:
            Tuple of (total_seconds, breakdown)
        """
        # Typical number of layers tested in classification
        num_layers = min(5, self.total_layers)
        
        # Control vectors typically test more layers
        cv_layers = min(10, self.total_layers)
        
        # Sample sizes must be provided
        if sample_sizes is None:
            raise RuntimeError("sample_sizes must be provided for full optimization time estimation")
        
        return self.calibrator.estimate_optimization_time(
            num_tasks=num_tasks,
            num_layers=num_layers,
            samples_per_task=classification_limit,
            sample_sizes=sample_sizes,
            sample_size_limit=sample_size_limit,
            include_sample_size_opt=include_sample_size_opt,
            include_classifier_training=include_classifier_training,
            include_control_vectors=include_control_vectors,
            num_cv_layers=cv_layers
        )
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f} minutes"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            if minutes > 0:
                return f"{hours:.0f} hours {minutes:.0f} minutes"
            else:
                return f"{hours:.0f} hours"
    
    def print_time_breakdown(self, total_time: float, breakdown: Dict[str, float]):
        """Print a formatted time breakdown"""
        print(f"\n⏱️  ESTIMATED OPTIMIZATION TIME:")
        print(f"   Total: {self.format_time(total_time)}")
        
        if len(breakdown) > 1:
            print("\n   Breakdown:")
            for phase, time_sec in breakdown.items():
                if time_sec > 0:
                    print(f"   - {phase.replace('_', ' ').title()}: {self.format_time(time_sec)}")