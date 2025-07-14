"""Time estimation for optimization operations using runtime calibration"""
import time
from typing import Dict, Tuple, Optional
from pathlib import Path

from .timing_calibration import TimingCalibrator


class OptimizationTimeEstimator:
    """Estimates time required for optimization operations using calibration"""
    
    # Fallback timing estimates (in seconds) - used when calibration is skipped
    FALLBACK_TIMING = {
        "per_task_per_sample": 0.005,  # ~5ms per sample per task
        "per_layer": 5.0,  # ~5s per layer
        "classifier_training_per_sample": 0.002,  # ~2ms per training sample
        "control_vector_per_layer": 10.0,  # ~10s per layer
    }
    
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
        try:
            from . import Model
            model = Model(name=model_name)
            # Try to detect number of layers from model
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                if hasattr(model.model.config, 'num_hidden_layers'):
                    self.total_layers = model.model.config.num_hidden_layers
                elif hasattr(model.model.config, 'n_layer'):
                    self.total_layers = model.model.config.n_layer
                else:
                    self.total_layers = 24  # Default estimate
            else:
                self.total_layers = 24  # Default estimate
        except:
            self.total_layers = 24  # Default estimate
        
        # Handle calibration
        if calibration_file and calibration_file.exists():
            # Load from file
            if self.calibrator.load_from_file(calibration_file):
                self.timing = self.calibrator.timings
            else:
                self.timing = self.FALLBACK_TIMING
        elif skip_calibration:
            # Use fallback timing
            self.timing = self.FALLBACK_TIMING
            if verbose:
                print("⏭️  Skipping calibration, using fallback estimates")
        else:
            # Run calibration
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
        
        return self.calibrator.estimate_optimization_time(
            num_tasks=num_tasks,
            num_layers=num_layers,
            samples_per_task=classification_limit,
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