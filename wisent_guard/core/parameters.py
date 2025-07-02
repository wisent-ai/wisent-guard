"""
Parameter management for Wisent Guard models.

Loads model-specific parameters from JSON files and allows CLI overrides.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ModelParameters:
    """Model-specific parameter management."""
    
    def __init__(self, model_name: str, layer_override: Optional[int] = None):
        self.model_name = model_name
        self.layer_override = layer_override
        self._params = self._load_parameters()
    
    def _load_parameters(self) -> Dict[str, Any]:
        """Load parameters from JSON file."""
        # Convert model name to file path
        # meta-llama/Llama-3.1-8B-Instruct -> parameters/meta-llama/Llama-3.1-8B-Instruct.json
        param_file = Path(__file__).parent.parent / "parameters" / f"{self.model_name}.json"
        
        # Default parameters
        defaults = {
            "layer": 15
        }
        
        # Try to load from file
        if param_file.exists():
            try:
                with open(param_file, 'r') as f:
                    file_params = json.load(f)
                defaults.update(file_params)
            except Exception as e:
                print(f"⚠️ Failed to load parameters from {param_file}: {e}")
                print("   Using default parameters")
        else:
            print(f"⚠️ Parameter file not found: {param_file}")
            print("   Using default parameters")
        
        return defaults
    
    @property
    def layer(self) -> int:
        """Get the layer to use (CLI override takes precedence)."""
        if self.layer_override is not None:
            return self.layer_override
        return self._params.get("layer", 15)
    
    @property
    def default_steering_method(self) -> str:
        """Get the default steering method."""
        steering = self._params.get("steering", {})
        return steering.get("default_method", "CAA")
    
    @property
    def default_steering_strength(self) -> float:
        """Get the default steering strength."""
        steering = self._params.get("steering", {})
        return steering.get("default_strength", 1.0)
    
    def get_steering_config(self, method: str) -> Dict[str, Any]:
        """Get configuration for a specific steering method."""
        steering = self._params.get("steering", {})
        method_config = steering.get(method.lower(), {})
        return method_config
    
    def get_summary(self) -> str:
        """Get a summary of active parameters."""
        summary = f"📋 Model Parameters ({self.model_name}):\n"
        summary += f"   Layer: {self.layer}"
        if self.layer_override is not None:
            summary += f" (CLI override from {self._params.get('layer', 15)})"
        summary += f"\n   Default Steering: {self.default_steering_method} (strength: {self.default_steering_strength})"
        return summary


def load_model_parameters(model_name: str, layer_override: Optional[int] = None) -> ModelParameters:
    """Load parameters for a specific model."""
    return ModelParameters(model_name, layer_override) 