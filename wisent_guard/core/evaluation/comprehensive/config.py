"""
Configuration management for comprehensive evaluation pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import torch


@dataclass
class ComprehensiveEvaluationConfig:
    """Configuration for comprehensive evaluation pipeline."""
    
    # Model configuration
    model_name: str = "distilbert/distilgpt2"  # Start with lightweight model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dataset configuration (fully configurable)
    train_dataset: str = "math500"
    val_dataset: str = "aime2024" 
    test_dataset: str = "aime2025"
    train_limit: int = 50
    val_limit: int = 25
    test_limit: int = 25
    
    # Probe training configuration
    probe_layers: List[int] = None  # Will default based on model
    probe_c_values: List[float] = None  # Will default to [0.1, 1.0, 10.0]
    
    # Steering configuration for grid search
    steering_methods: List[str] = None  # Will default to ["baseline"]
    steering_layers: List[int] = None   # Will default based on model
    steering_strengths: List[float] = None  # Will default to [1.0]
    
    # Optimization weights
    benchmark_weight: float = 0.7  # Weight for benchmark performance in combined score
    probe_weight: float = 0.3      # Weight for probe performance in combined score
    
    # Output configuration
    output_dir: str = "outputs/comprehensive_evaluation_results"
    experiment_name: str = "comprehensive_wisent_evaluation"
    
    # Wandb configuration
    wandb_project: str = "wisent-guard-comprehensive-evaluation"
    wandb_tags: List[str] = None
    wandb_entity: Optional[str] = None
    enable_wandb: bool = True
    
    # Technical configuration
    batch_size: int = 4  # Smaller for distilgpt2
    max_length: int = 128  # Shorter for distilgpt2
    seed: int = 42
    verbose: bool = False
    
    def __post_init__(self):
        """Set defaults based on model choice."""
        # Set layer defaults based on model
        if "distilgpt2" in self.model_name.lower():
            if self.probe_layers is None:
                self.probe_layers = [2, 3, 4, 5]  # distilgpt2 has 6 layers (0-5)
            if self.steering_layers is None:
                self.steering_layers = [3, 4, 5]
        elif "llama" in self.model_name.lower():
            if self.probe_layers is None:
                self.probe_layers = [8, 16, 24, 32]  # Assuming 32-layer model
            if self.steering_layers is None:
                self.steering_layers = [16, 24, 32]
        else:
            # Generic defaults
            if self.probe_layers is None:
                self.probe_layers = [4, 6, 8]
            if self.steering_layers is None:
                self.steering_layers = [6, 8]
        
        # Set other defaults
        if self.probe_c_values is None:
            self.probe_c_values = [0.1, 1.0, 10.0]
        if self.steering_methods is None:
            self.steering_methods = ["baseline"]  # Start with baseline (no steering)
        if self.steering_strengths is None:
            self.steering_strengths = [1.0]  # Start with strength 1.0
        if self.wandb_tags is None:
            self.wandb_tags = ["comprehensive_evaluation", self.train_dataset, self.val_dataset, self.test_dataset]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "test_dataset": self.test_dataset,
            "train_limit": self.train_limit,
            "val_limit": self.val_limit,
            "test_limit": self.test_limit,
            "probe_layers": self.probe_layers,
            "probe_c_values": self.probe_c_values,
            "steering_methods": self.steering_methods,
            "steering_layers": self.steering_layers,
            "steering_strengths": self.steering_strengths,
            "benchmark_weight": self.benchmark_weight,
            "probe_weight": self.probe_weight,
            "output_dir": self.output_dir,
            "experiment_name": self.experiment_name,
            "wandb_project": self.wandb_project,
            "wandb_tags": self.wandb_tags,
            "enable_wandb": self.enable_wandb,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "seed": self.seed,
            "verbose": self.verbose
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ComprehensiveEvaluationConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model-specific information."""
        return {
            "model_name": self.model_name,
            "num_probe_layers": len(self.probe_layers),
            "num_steering_layers": len(self.steering_layers),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset configuration summary."""
        return {
            "train_dataset": self.train_dataset,
            "val_dataset": self.val_dataset,
            "test_dataset": self.test_dataset,
            "train_samples": self.train_limit,
            "val_samples": self.val_limit,
            "test_samples": self.test_limit,
            "total_samples": self.train_limit + self.val_limit + self.test_limit
        }
    
    def get_hyperparameter_search_space_size(self) -> int:
        """Calculate total number of hyperparameter combinations."""
        return (len(self.steering_methods) * 
                len(self.steering_layers) * 
                len(self.steering_strengths) *
                len(self.probe_layers) *
                len(self.probe_c_values))