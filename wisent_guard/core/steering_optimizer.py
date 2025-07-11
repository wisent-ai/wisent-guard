"""
Steering Parameter Optimizer for Wisent-Guard.

Framework for optimizing steering-specific parameters including:
1. Optimal steering layer (may differ from classification layer)
2. Optimal steering strength and dynamics
3. Steering method selection and configuration
4. Task-specific steering parameter tuning

This module provides the structure for steering optimization but requires
implementation of the actual optimization logic.
"""

import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .model_config_manager import ModelConfigManager

logger = logging.getLogger(__name__)


class SteeringMethod(Enum):
    """Available steering methods for optimization."""
    CAA = "CAA"
    HPR = "HPR" 
    DAC = "DAC"
    BIPO = "BiPO"
    KSTEERING = "KSteering"


@dataclass
class SteeringOptimizationResult:
    """Results from optimizing steering parameters for a single task."""
    task_name: str
    best_steering_layer: int
    best_steering_method: str
    best_steering_strength: float
    optimal_parameters: Dict[str, Any]  # Method-specific parameters
    steering_effectiveness_score: float  # How well steering changes outputs
    classification_accuracy_impact: float  # Impact on classification performance
    optimization_time_seconds: float
    total_configurations_tested: int
    error_message: Optional[str] = None


@dataclass
class SteeringOptimizationSummary:
    """Summary of steering optimization across tasks/methods."""
    model_name: str
    optimization_type: str  # "single_task", "multi_task", "method_comparison" 
    total_configurations_tested: int
    optimization_time_minutes: float
    best_overall_method: str
    best_overall_layer: int
    best_overall_strength: float
    method_performance_ranking: Dict[str, float]  # method -> effectiveness score
    layer_effectiveness_analysis: Dict[int, float]  # layer -> avg effectiveness
    task_results: List[SteeringOptimizationResult]
    optimization_date: str


class SteeringOptimizer:
    """
    Framework for optimizing steering parameters.
    
    This class provides the structure for steering optimization but requires
    implementation of the actual optimization algorithms for each steering method.
    """
    
    def __init__(self, model_name: str, device: str = None, verbose: bool = False):
        """
        Initialize steering optimizer.
        
        Args:
            model_name: Name/path of the model to optimize steering for
            device: Device to run optimization on
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.config_manager = ModelConfigManager()
        
        # Load classification parameters if available (steering often builds on classification)
        self.classification_config = self.config_manager.load_model_config(model_name)
        if self.classification_config:
            self.base_classification_layer = self.classification_config.get("optimal_parameters", {}).get("classification_layer")
            logger.info(f"📊 Found existing classification layer: {self.base_classification_layer}")
        else:
            self.base_classification_layer = None
            logger.warning("⚠️ No existing classification configuration found")
    
    def optimize_steering_method_comparison(
        self,
        task_name: str,
        methods_to_test: Optional[List[SteeringMethod]] = None,
        layer_range: Optional[str] = None,
        strength_range: Optional[List[float]] = None,
        limit: int = 100,
        max_time_minutes: float = 30.0
    ) -> SteeringOptimizationSummary:
        """
        Compare different steering methods to find the best one for a task.
        
        Args:
            task_name: Task to optimize steering for
            methods_to_test: List of steering methods to compare
            layer_range: Range of layers to test for steering
            strength_range: Range of steering strengths to test
            limit: Maximum samples for testing
            max_time_minutes: Maximum optimization time
            
        Returns:
            SteeringOptimizationSummary with method comparison results
        """
        if methods_to_test is None:
            methods_to_test = list(SteeringMethod)
        
        logger.info(f"🎯 Comparing {len(methods_to_test)} steering methods for task: {task_name}")
        
        # TODO: Implement steering method comparison logic
        raise NotImplementedError(
            "Steering method comparison optimization not yet implemented. "
            "This requires implementing steering effectiveness measurement and "
            "parameter optimization for each steering method."
        )
    
    def optimize_steering_layer(
        self,
        task_name: str,
        steering_method: SteeringMethod = SteeringMethod.CAA,
        layer_search_range: Optional[Tuple[int, int]] = None,
        strength: float = 1.0,
        limit: int = 100
    ) -> SteeringOptimizationResult:
        """
        Find optimal steering layer for a specific method and task.
        
        Args:
            task_name: Task to optimize for
            steering_method: Steering method to use
            layer_search_range: (min_layer, max_layer) to search
            strength: Fixed steering strength to use during layer search
            limit: Maximum samples for testing
            
        Returns:
            SteeringOptimizationResult with optimal layer
        """
        logger.info(f"🔍 Optimizing steering layer for {task_name} using {steering_method.value}")
        
        if layer_search_range is None:
            # Default: search around classification layer if available
            if self.base_classification_layer:
                min_layer = max(1, self.base_classification_layer - 3)
                max_layer = self.base_classification_layer + 3
                layer_search_range = (min_layer, max_layer)
            else:
                # TODO: Auto-detect model layer count and use reasonable range
                layer_search_range = (10, 20)  # Default fallback
        
        # TODO: Implement layer optimization logic
        raise NotImplementedError(
            "Steering layer optimization not yet implemented. "
            "This requires implementing steering vector training and "
            "effectiveness measurement across different layers."
        )
    
    def optimize_steering_strength(
        self,
        task_name: str,
        steering_method: SteeringMethod = SteeringMethod.CAA,
        layer: Optional[int] = None,
        strength_range: Optional[Tuple[float, float]] = None,
        strength_steps: int = 10,
        limit: int = 100
    ) -> SteeringOptimizationResult:
        """
        Find optimal steering strength for a specific method, layer, and task.
        
        Args:
            task_name: Task to optimize for
            steering_method: Steering method to use
            layer: Steering layer to use (defaults to classification layer)
            strength_range: (min_strength, max_strength) to search
            strength_steps: Number of strength values to test
            limit: Maximum samples for testing
            
        Returns:
            SteeringOptimizationResult with optimal strength
        """
        if layer is None:
            layer = self.base_classification_layer or 15  # Default fallback
        
        if strength_range is None:
            strength_range = (0.1, 2.0)  # Default strength range
        
        logger.info(f"⚡ Optimizing steering strength for {task_name}")
        logger.info(f"   Method: {steering_method.value}, Layer: {layer}")
        logger.info(f"   Strength range: {strength_range}, Steps: {strength_steps}")
        
        # TODO: Implement strength optimization logic
        raise NotImplementedError(
            "Steering strength optimization not yet implemented. "
            "This requires implementing steering effectiveness measurement "
            "and systematic strength testing."
        )
    
    def optimize_method_specific_parameters(
        self,
        task_name: str,
        steering_method: SteeringMethod,
        base_layer: Optional[int] = None,
        base_strength: float = 1.0,
        limit: int = 100
    ) -> SteeringOptimizationResult:
        """
        Optimize method-specific parameters for a steering approach.
        
        Args:
            task_name: Task to optimize for
            steering_method: Specific steering method to optimize
            base_layer: Base steering layer to use
            base_strength: Base steering strength to use
            limit: Maximum samples for testing
            
        Returns:
            SteeringOptimizationResult with optimized method parameters
        """
        logger.info(f"🔧 Optimizing {steering_method.value}-specific parameters for {task_name}")
        
        if steering_method == SteeringMethod.CAA:
            return self._optimize_caa_parameters(task_name, base_layer, base_strength, limit)
        elif steering_method == SteeringMethod.HPR:
            return self._optimize_hpr_parameters(task_name, base_layer, base_strength, limit)
        elif steering_method == SteeringMethod.DAC:
            return self._optimize_dac_parameters(task_name, base_layer, base_strength, limit)
        elif steering_method == SteeringMethod.BIPO:
            return self._optimize_bipo_parameters(task_name, base_layer, base_strength, limit)
        elif steering_method == SteeringMethod.KSTEERING:
            return self._optimize_ksteering_parameters(task_name, base_layer, base_strength, limit)
        else:
            raise ValueError(f"Unknown steering method: {steering_method}")
    
    def _optimize_caa_parameters(
        self, 
        task_name: str, 
        layer: Optional[int], 
        strength: float, 
        limit: int
    ) -> SteeringOptimizationResult:
        """Optimize CAA (Concept Activation Analysis) specific parameters."""
        # TODO: Implement CAA parameter optimization
        # CAA typically doesn't have many hyperparameters beyond layer/strength
        # but may include normalization options, vector aggregation methods, etc.
        raise NotImplementedError("CAA parameter optimization not yet implemented")
    
    def _optimize_hpr_parameters(
        self, 
        task_name: str, 
        layer: Optional[int], 
        strength: float, 
        limit: int
    ) -> SteeringOptimizationResult:
        """Optimize HPR (Householder Pseudo-Rotation) specific parameters."""
        # TODO: Implement HPR parameter optimization
        # HPR has beta parameter and potentially rotation-specific settings
        raise NotImplementedError("HPR parameter optimization not yet implemented")
    
    def _optimize_dac_parameters(
        self, 
        task_name: str, 
        layer: Optional[int], 
        strength: float, 
        limit: int
    ) -> SteeringOptimizationResult:
        """Optimize DAC (Dynamic Activation Composition) specific parameters."""
        # TODO: Implement DAC parameter optimization  
        # DAC has dynamic control settings, entropy thresholds, etc.
        raise NotImplementedError("DAC parameter optimization not yet implemented")
    
    def _optimize_bipo_parameters(
        self, 
        task_name: str, 
        layer: Optional[int], 
        strength: float, 
        limit: int
    ) -> SteeringOptimizationResult:
        """Optimize BiPO (Bi-directional Preference Optimization) specific parameters."""
        # TODO: Implement BiPO parameter optimization
        # BiPO has learning rate, beta, epochs, and other training-specific parameters
        raise NotImplementedError("BiPO parameter optimization not yet implemented")
    
    def _optimize_ksteering_parameters(
        self, 
        task_name: str, 
        layer: Optional[int], 
        strength: float, 
        limit: int
    ) -> SteeringOptimizationResult:
        """Optimize K-Steering specific parameters."""
        # TODO: Implement K-Steering parameter optimization
        # K-Steering has many parameters: num_labels, hidden_dim, learning_rate, 
        # classifier_epochs, target/avoid labels, alpha, etc.
        raise NotImplementedError("K-Steering parameter optimization not yet implemented")
    
    def run_comprehensive_steering_optimization(
        self,
        tasks: Optional[List[str]] = None,
        methods: Optional[List[SteeringMethod]] = None,
        limit: int = 100,
        max_time_per_task_minutes: float = 20.0,
        save_results: bool = True
    ) -> SteeringOptimizationSummary:
        """
        Run comprehensive steering optimization across multiple tasks and methods.
        
        Args:
            tasks: List of tasks to optimize (if None, uses classification-optimized tasks)
            methods: List of steering methods to test
            limit: Sample limit per task
            max_time_per_task_minutes: Time limit per task
            save_results: Whether to save results to config
            
        Returns:
            SteeringOptimizationSummary with comprehensive results
        """
        logger.info(f"🚀 Starting comprehensive steering optimization")
        
        if tasks is None:
            # Use tasks that were successfully optimized for classification
            if self.classification_config:
                task_overrides = self.classification_config.get("task_specific_overrides", {})
                tasks = list(task_overrides.keys())
                if not tasks:
                    logger.warning("No classification-optimized tasks found, using default task set")
                    tasks = ["truthfulqa_mc1", "gsm8k", "squad2"]  # Default fallback
            else:
                tasks = ["truthfulqa_mc1", "gsm8k", "squad2"]  # Default fallback
        
        if methods is None:
            methods = [SteeringMethod.CAA, SteeringMethod.HPR]  # Start with simpler methods
        
        logger.info(f"📊 Tasks: {tasks}")
        logger.info(f"🔧 Methods: [methods.value for method in methods]")
        
        # TODO: Implement comprehensive optimization loop
        # This should:
        # 1. For each task and method combination
        # 2. Find optimal layer, strength, and method-specific parameters
        # 3. Measure steering effectiveness vs classification accuracy tradeoff
        # 4. Aggregate results and find best overall parameters
        # 5. Save task-specific steering configurations
        
        raise NotImplementedError(
            "Comprehensive steering optimization not yet implemented. "
            "This requires implementing all the individual optimization methods "
            "and result aggregation logic."
        )
    
    def evaluate_steering_effectiveness(
        self,
        task_name: str,
        steering_method: SteeringMethod,
        layer: int,
        strength: float,
        method_params: Dict[str, Any],
        test_samples: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate how effectively steering changes model outputs.
        
        Args:
            task_name: Task being evaluated
            steering_method: Steering method being used
            layer: Steering layer
            strength: Steering strength
            method_params: Method-specific parameters
            test_samples: Test samples to evaluate on
            
        Returns:
            Dictionary with effectiveness metrics
        """
        # TODO: Implement steering effectiveness measurement
        # This should measure:
        # - How much steering changes the model's outputs
        # - Whether steering moves outputs in the desired direction
        # - Impact on classification accuracy
        # - Consistency of steering effects across samples
        
        raise NotImplementedError(
            "Steering effectiveness evaluation not yet implemented. "
            "This requires implementing steering application and "
            "output change measurement."
        )


# Convenience functions for CLI integration
def run_steering_optimization(
    model_name: str,
    optimization_type: str = "method_comparison",
    task_name: str = "truthfulqa_mc1",
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    **kwargs
) -> Union[SteeringOptimizationResult, SteeringOptimizationSummary]:
    """
    Convenience function to run steering optimization.
    
    Args:
        model_name: Model to optimize steering for
        optimization_type: Type of optimization ("method_comparison", "layer", "strength", "comprehensive")
        task_name: Task to optimize for (for single-task optimization)
        limit: Sample limit
        device: Device to use
        verbose: Enable verbose logging
        **kwargs: Additional arguments for specific optimization types
        
    Returns:
        SteeringOptimizationResult or SteeringOptimizationSummary depending on optimization type
    """
    optimizer = SteeringOptimizer(
        model_name=model_name,
        device=device,
        verbose=verbose
    )
    
    if optimization_type == "method_comparison":
        return optimizer.optimize_steering_method_comparison(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "layer":
        return optimizer.optimize_steering_layer(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "strength":
        return optimizer.optimize_steering_strength(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "comprehensive":
        return optimizer.run_comprehensive_steering_optimization(
            limit=limit,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimization type: {optimization_type}")


# TODO: Integration with existing steering methods
# 
# The following integration points need to be implemented:
#
# 1. CAA Integration:
#    - Load existing CAA implementation from wisent_guard.core.steering_methods.caa
#    - Implement parameter optimization for CAA vectors
#    - Measure CAA steering effectiveness
#
# 2. HPR Integration:
#    - Load HPR implementation and optimize beta parameter
#    - Test rotation effectiveness across different layers
#
# 3. DAC Integration:
#    - Optimize dynamic control parameters and entropy thresholds
#    - Test adaptive steering strength adjustment
#
# 4. BiPO Integration:
#    - Optimize learning parameters for preference-based steering
#    - Implement bi-directional steering evaluation
#
# 5. K-Steering Integration:
#    - Optimize classifier parameters and label configurations
#    - Test multi-label steering effectiveness
#
# 6. Effectiveness Metrics:
#    - Implement steering strength measurement
#    - Develop steering direction accuracy metrics
#    - Create steering consistency evaluation
#    - Measure classification accuracy preservation 