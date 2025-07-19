"""
Model Configuration Manager for storing and retrieving optimal parameters per model.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logger = logging.getLogger(__name__)


class ModelConfigManager:
    """Manages model-specific configuration files for optimal parameters."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the ModelConfigManager.
        
        Args:
            config_dir: Directory to store config files. If None, uses default location.
        """
        if config_dir is None:
            # Use ~/.wisent-guard/model_configs/ as default
            home_dir = os.path.expanduser("~")
            self.config_dir = os.path.join(home_dir, ".wisent-guard", "model_configs")
        else:
            self.config_dir = config_dir
            
        # Create directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
    def _sanitize_model_name(self, model_name: str) -> str:
        """
        Convert model name to a safe filename.
        
        Args:
            model_name: Original model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            
        Returns:
            Sanitized filename (e.g., "meta-llama_Llama-3.1-8B-Instruct")
        """
        # Replace problematic characters
        sanitized = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        # Remove any other problematic characters
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "._-")
        return sanitized
    
    def _get_config_path(self, model_name: str) -> str:
        """Get the full path to the config file for a model."""
        sanitized_name = self._sanitize_model_name(model_name)
        return os.path.join(self.config_dir, f"{sanitized_name}.json")
    
    def save_model_config(
        self,
        model_name: str,
        classification_layer: int,
        steering_layer: Optional[int] = None,
        token_aggregation: str = "average",
        detection_threshold: float = 0.6,
        optimization_method: str = "manual",
        optimization_metrics: Optional[Dict[str, Any]] = None,
        task_specific_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> str:
        """
        Save optimal parameters for a model.
        
        Args:
            model_name: Name/path of the model
            classification_layer: Optimal layer for classification
            steering_layer: Optimal layer for steering (defaults to classification_layer)
            token_aggregation: Token aggregation method
            detection_threshold: Detection threshold
            optimization_method: How these parameters were determined
            optimization_metrics: Metrics from optimization process
            task_specific_overrides: Task-specific parameter overrides
            
        Returns:
            Path to the saved config file
        """
        if steering_layer is None:
            steering_layer = classification_layer
            
        config_data = {
            "model_name": model_name,
            "created_date": datetime.now().isoformat(),
            "optimization_method": optimization_method,
            "optimal_parameters": {
                "classification_layer": classification_layer,
                "steering_layer": steering_layer,
                "token_aggregation": token_aggregation,
                "detection_threshold": detection_threshold
            },
            "task_specific_overrides": task_specific_overrides or {},
            "optimization_metrics": optimization_metrics or {},
            "config_version": "1.0"
        }
        
        config_path = self._get_config_path(model_name)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"✅ Model configuration saved: {config_path}")
            logger.info(f"   • Classification layer: {classification_layer}")
            logger.info(f"   • Steering layer: {steering_layer}")
            logger.info(f"   • Token aggregation: {token_aggregation}")
            logger.info(f"   • Detection threshold: {detection_threshold}")
            
            return config_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save model configuration: {e}")
            raise
    
    def load_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load optimal parameters for a model.
        
        Args:
            model_name: Name/path of the model
            
        Returns:
            Configuration dictionary if found, None otherwise
        """
        config_path = self._get_config_path(model_name)
        
        if not os.path.exists(config_path):
            return None
            
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            logger.debug(f"📄 Loaded model configuration: {config_path}")
            return config_data
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load model configuration: {e}")
            return None
    
    def has_model_config(self, model_name: str) -> bool:
        """Check if a model has a saved configuration."""
        config_path = self._get_config_path(model_name)
        return os.path.exists(config_path)
    
    def update_model_config(self, model_name: str, config_data: Dict[str, Any]) -> str:
        """
        Update an existing model configuration.
        
        Args:
            model_name: Name/path of the model
            config_data: Updated configuration dictionary
            
        Returns:
            Path to the saved config file
        """
        config_path = self._get_config_path(model_name)
        
        # Update timestamp
        config_data["updated_date"] = datetime.now().isoformat()
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"✅ Model configuration updated: {config_path}")
            return config_path
            
        except Exception as e:
            logger.error(f"❌ Failed to update model configuration: {e}")
            raise
    
    def get_optimal_parameters(
        self, 
        model_name: str, 
        task_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get optimal parameters for a model, with optional task-specific overrides.
        
        Args:
            model_name: Name/path of the model
            task_name: Specific task name for overrides
            
        Returns:
            Dictionary of optimal parameters or None if no config exists
        """
        config = self.load_model_config(model_name)
        if not config:
            return None
        
        # Start with base optimal parameters
        optimal_params = config.get("optimal_parameters", {}).copy()
        
        # Apply task-specific overrides if available
        if task_name and "task_specific_overrides" in config:
            task_overrides = config["task_specific_overrides"].get(task_name, {})
            optimal_params.update(task_overrides)
        
        return optimal_params
    
    def get_optimal_sample_size(
        self,
        model_name: str,
        task_name: str,
        layer: int
    ) -> Optional[int]:
        """
        Get optimal sample size for a specific task and layer.
        
        Args:
            model_name: Name/path of the model
            task_name: Task name
            layer: Layer index
            
        Returns:
            Optimal sample size or None if not found
        """
        config = self.load_model_config(model_name)
        if not config:
            return None
            
        # Check if optimal_sample_sizes exists
        if "optimal_sample_sizes" not in config:
            return None
            
        # Navigate the nested structure: optimal_sample_sizes[task][layer]
        task_sizes = config["optimal_sample_sizes"].get(task_name, {})
        sample_size = task_sizes.get(str(layer), None)
        
        return sample_size
    
    def list_model_configs(self) -> List[Dict[str, Any]]:
        """
        List all available model configurations.
        
        Returns:
            List of configuration summaries
        """
        configs = []
        
        if not os.path.exists(self.config_dir):
            return configs
        
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json'):
                try:
                    config_path = os.path.join(self.config_dir, filename)
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    summary = {
                        "model_name": config_data.get("model_name", "unknown"),
                        "created_date": config_data.get("created_date", "unknown"),
                        "optimization_method": config_data.get("optimization_method", "unknown"),
                        "classification_layer": config_data.get("optimal_parameters", {}).get("classification_layer"),
                        "steering_layer": config_data.get("optimal_parameters", {}).get("steering_layer"),
                        "config_file": filename
                    }
                    configs.append(summary)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to read config file {filename}: {e}")
        
        return configs
    
    def remove_model_config(self, model_name: str) -> bool:
        """
        Remove a model configuration.
        
        Args:
            model_name: Name/path of the model
            
        Returns:
            True if removed successfully, False otherwise
        """
        config_path = self._get_config_path(model_name)
        
        if not os.path.exists(config_path):
            logger.warning(f"⚠️ No configuration found for model: {model_name}")
            return False
        
        try:
            os.remove(config_path)
            logger.info(f"✅ Removed model configuration: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to remove model configuration: {e}")
            return False


# Convenience functions for easy access
_default_manager = None

def get_default_manager() -> ModelConfigManager:
    """Get the default ModelConfigManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelConfigManager()
    return _default_manager

def save_model_config(model_name: str, **kwargs) -> str:
    """Save model configuration using default manager."""
    return get_default_manager().save_model_config(model_name, **kwargs)

def load_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    """Load model configuration using default manager."""
    return get_default_manager().load_model_config(model_name)

def get_optimal_parameters(model_name: str, task_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get optimal parameters using default manager."""
    return get_default_manager().get_optimal_parameters(model_name, task_name) 