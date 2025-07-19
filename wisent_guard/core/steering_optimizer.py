"""
Steering Parameter Optimizer for Wisent-Guard.

Optimizes steering-specific parameters including:
1. Optimal steering layer (may differ from classification layer)
2. Optimal steering strength and dynamics
3. Steering method selection and configuration
4. Task-specific steering parameter tuning

This module builds on top of classification optimization to find optimal
steering configurations for each model and task.
"""

import logging
import json
import time
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from .model_config_manager import ModelConfigManager

logger = logging.getLogger(__name__)


def get_default_steering_configs() -> List['SteeringMethodConfig']:
    """Get default steering method configurations with parameter variations."""
    return [
        # CAA variations
        SteeringMethodConfig(
            name="CAA",
            method=SteeringMethod.CAA,
            params={}
        ),
        SteeringMethodConfig(
            name="CAA_L2",
            method=SteeringMethod.CAA,
            params={"normalization_method": "l2_unit"}
        ),
        
        # HPR variations
        SteeringMethodConfig(
            name="HPR",
            method=SteeringMethod.HPR,
            params={"hpr_beta": 1.0}
        ),
        SteeringMethodConfig(
            name="HPR_Beta0.5",
            method=SteeringMethod.HPR,
            params={"hpr_beta": 0.5}
        ),
        
        # BiPO variations
        SteeringMethodConfig(
            name="BiPO",
            method=SteeringMethod.BIPO,
            params={"bipo_beta": 0.1, "bipo_epochs": 50}
        ),
        SteeringMethodConfig(
            name="BiPO_Beta0.05",
            method=SteeringMethod.BIPO,
            params={"bipo_beta": 0.05, "bipo_epochs": 50}
        ),
        
        # KSteering variations
        SteeringMethodConfig(
            name="KSteering",
            method=SteeringMethod.KSTEERING,
            params={
                "ksteering_alpha": 5.0,
                "ksteering_target_labels": "0",
                "ksteering_avoid_labels": ""
            }
        ),
        SteeringMethodConfig(
            name="KSteering_Alpha3",
            method=SteeringMethod.KSTEERING,
            params={
                "ksteering_alpha": 3.0,
                "ksteering_target_labels": "0",
                "ksteering_avoid_labels": ""
            }
        ),
        
        # DAC variations
        SteeringMethodConfig(
            name="DAC",
            method=SteeringMethod.DAC,
            params={
                "dac_dynamic_control": True,
                "dac_entropy_threshold": 1.0
            }
        ),
    ]


class SteeringMethod(Enum):
    """Available steering methods for optimization."""
    CAA = "CAA"
    HPR = "HPR" 
    DAC = "DAC"
    BIPO = "BiPO"
    KSTEERING = "KSteering"


@dataclass
class SteeringMethodConfig:
    """Configuration for a specific steering method with parameter variations."""
    name: str  # Display name like "CAA_L2"
    method: SteeringMethod
    params: Dict[str, Any]  # Method-specific parameters
    
    def __post_init__(self):
        """Ensure method is SteeringMethod enum."""
        if isinstance(self.method, str):
            self.method = SteeringMethod(self.method)


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
        methods_to_test: Optional[Union[List[SteeringMethod], List[SteeringMethodConfig]]] = None,
        layer_range: Optional[str] = None,
        strength_range: Optional[List[float]] = None,
        limit: int = 100,
        max_time_minutes: float = 30.0,
        split_ratio: float = 0.8
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
            split_ratio: Train/test split ratio
            
        Returns:
            SteeringOptimizationSummary with method comparison results
        """
        # Handle both old-style method list and new config list
        if methods_to_test is None:
            # Use all default configurations
            method_configs = get_default_steering_configs()
        else:
            method_configs = []
            for item in methods_to_test:
                if isinstance(item, SteeringMethodConfig):
                    method_configs.append(item)
                elif isinstance(item, SteeringMethod):
                    # Convert simple method to config with default params
                    method_configs.append(SteeringMethodConfig(
                        name=item.value,
                        method=item,
                        params={}
                    ))
                elif isinstance(item, str):
                    # Convert string to SteeringMethod enum
                    try:
                        method = SteeringMethod(item)
                        method_configs.append(SteeringMethodConfig(
                            name=method.value,
                            method=method,
                            params={}
                        ))
                    except ValueError:
                        logger.warning(f"Unknown steering method: {item}")
                else:
                    logger.warning(f"Unknown method type: {type(item)}, value: {item}")
        
        if strength_range is None:
            strength_range = [0.5, 1.0, 1.5, 2.0]
            
        logger.info(f"🎯 Comparing {len(method_configs)} steering method configurations for task: {task_name}")
        
        start_time = time.time()
        task_results = []
        all_results = {}
        
        # Determine layer search range
        if layer_range:
            layers_to_test = self._parse_layer_range(layer_range)
        elif self.base_classification_layer:
            # Search around classification layer
            min_layer = max(1, self.base_classification_layer - 2)
            max_layer = min(32, self.base_classification_layer + 2)  # Assume max 32 layers
            layers_to_test = list(range(min_layer, max_layer + 1))
        else:
            # Default range for common models
            layers_to_test = [10, 12, 14, 16, 18, 20]
        
        configurations_tested = 0
        best_overall_score = 0.0
        best_overall_config = None
        
        # Test each method configuration
        for method_config in method_configs:
            method_results = []
            
            for layer in layers_to_test:
                for strength in strength_range:
                    if time.time() - start_time > max_time_minutes * 60:
                        logger.warning(f"⏰ Time limit reached, stopping optimization")
                        break
                        
                    try:
                        # Run evaluation for this configuration
                        score = self._evaluate_steering_configuration(
                            task_name=task_name,
                            method=method_config.method,
                            layer=layer,
                            strength=strength,
                            limit=limit,
                            split_ratio=split_ratio,
                            method_params=method_config.params
                        )
                        
                        configurations_tested += 1
                        config_result = {
                            'method': method_config.name,  # Use display name
                            'method_type': method_config.method.value,
                            'layer': layer,
                            'strength': strength,
                            'score': score,
                            'params': method_config.params
                        }
                        method_results.append(config_result)
                        
                        if score > best_overall_score:
                            best_overall_score = score
                            best_overall_config = config_result
                            
                        if self.verbose:
                            logger.info(f"   {method_config.name} L{layer} S{strength}: {score:.3f}")
                            
                    except Exception as e:
                        logger.error(f"   Error testing {method_config.name} L{layer} S{strength}: {e}")
                        
            all_results[method_config.name] = method_results
        
        # Analyze results
        method_performance = {}
        layer_effectiveness = {}
        
        for method, results in all_results.items():
            if results:
                scores = [r['score'] for r in results]
                method_performance[method] = max(scores)
                
                # Aggregate by layer
                for result in results:
                    layer = result['layer']
                    if layer not in layer_effectiveness:
                        layer_effectiveness[layer] = []
                    layer_effectiveness[layer].append(result['score'])
        
        # Average layer effectiveness
        for layer in layer_effectiveness:
            layer_effectiveness[layer] = sum(layer_effectiveness[layer]) / len(layer_effectiveness[layer])
        
        # Create optimization result
        optimization_time = time.time() - start_time
        
        if best_overall_config:
            result = SteeringOptimizationResult(
                task_name=task_name,
                best_steering_layer=best_overall_config['layer'],
                best_steering_method=best_overall_config['method'],
                best_steering_strength=best_overall_config['strength'],
                optimal_parameters={
                    'split_ratio': split_ratio,
                    'limit': limit,
                    **best_overall_config.get('params', {})
                },
                steering_effectiveness_score=best_overall_config['score'],
                classification_accuracy_impact=0.0,  # Not measured here
                optimization_time_seconds=optimization_time,
                total_configurations_tested=configurations_tested
            )
            task_results.append(result)
        
        summary = SteeringOptimizationSummary(
            model_name=self.model_name,
            optimization_type="method_comparison",
            total_configurations_tested=configurations_tested,
            optimization_time_minutes=optimization_time / 60,
            best_overall_method=best_overall_config['method'] if best_overall_config else "none",
            best_overall_layer=best_overall_config['layer'] if best_overall_config else 0,
            best_overall_strength=best_overall_config['strength'] if best_overall_config else 0.0,
            method_performance_ranking=method_performance,
            layer_effectiveness_analysis=layer_effectiveness,
            task_results=task_results,
            optimization_date=datetime.now().isoformat()
        )
        
        # Save the results
        self._save_steering_optimization_results(summary)
        
        return summary
    
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
        limit: int = 100,
        method_params: Optional[Dict[str, Any]] = None
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
        
        # Load steering parameters from config
        import json
        import os
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'steering_optimization_parameters.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                steering_config = json.load(f)
        else:
            steering_config = {}
        
        # Get default layer if not provided
        if layer is None:
            layer = self._get_classification_layer(task_name)
            logger.info(f"   Using classification layer: {layer}")
        
        # Default strength range from config or fallback
        if strength_range is None:
            default_strengths = steering_config.get('steering_strengths', {}).get('default', [0.5, 1.0, 1.5, 2.0])
            strength_range = (min(default_strengths), max(default_strengths))
        
        # Generate strength values to test
        import numpy as np
        strengths = np.linspace(strength_range[0], strength_range[1], strength_steps)
        
        # Get method-specific parameters from config if not provided
        if method_params is None:
            method_configs = steering_config.get('steering_methods', [])
            for config in method_configs:
                if config['method'] == steering_method.value and config['name'] == steering_method.value:
                    method_params = config.get('params', {})
                    break
            if method_params is None:
                method_params = {}
        
        results = []
        best_score = -float('inf')
        best_strength = 0.0
        
        logger.info(f"   Testing {len(strengths)} strength values...")
        
        # Test each strength value
        for strength in strengths:
            try:
                # Run evaluation with this strength
                from ..cli import run_task_pipeline
                
                # Build kwargs for run_task_pipeline
                pipeline_kwargs = {
                    'task_name': task_name,
                    'model_name': self.model_name,
                    'limit': limit,
                    'device': self.device,
                    'verbose': False,
                    'steering_mode': True,
                    'steering_method': steering_method.value,
                    'steering_strength': float(strength),
                    'layer': str(layer),
                    'output_mode': "likelihoods",  # Get likelihoods for evaluation
                    'allow_small_dataset': True
                }
                
                # Add method-specific parameters
                for param, value in method_params.items():
                    pipeline_kwargs[param] = value
                
                result = run_task_pipeline(**pipeline_kwargs)
                
                # Extract score from evaluation results
                score = 0.0
                steering_effect = 0.0
                accuracy = 0.0
                
                if isinstance(result, dict):
                    # Get evaluation results - try both nested and direct access
                    eval_results = None
                    
                    # First try: result[task_name]['evaluation_results']
                    task_result = result.get(task_name, {})
                    if 'evaluation_results' in task_result:
                        eval_results = task_result['evaluation_results']
                    # Second try: result['evaluation_results'] (direct from run_task_pipeline)
                    elif 'evaluation_results' in result:
                        eval_results = result['evaluation_results']
                    else:
                        eval_results = {}
                    
                    # Get accuracy score (but don't use it as the primary metric)
                    accuracy = eval_results.get('accuracy', 0.0)
                    if isinstance(accuracy, str) or accuracy is None:
                        accuracy = 0.0
                    
                    # Calculate steering effect from likelihood changes
                    baseline_likes = eval_results.get('baseline_likelihoods', [])
                    steered_likes = eval_results.get('steered_likelihoods', [])
                    
                    if self.verbose:
                        logger.debug(f"   Found {len(baseline_likes)} baseline and {len(steered_likes)} steered likelihoods")
                        if baseline_likes and len(baseline_likes) > 0:
                            logger.debug(f"   First few baseline likes: {baseline_likes[:3]}")
                            logger.debug(f"   First few steered likes: {steered_likes[:3]}")
                        logger.debug(f"   Full eval_results keys: {list(eval_results.keys())}")
                        logger.debug(f"   Accuracy value: {eval_results.get('accuracy')}")
                        
                        # Check if we're getting the right data structure
                        if isinstance(result, dict):
                            logger.debug(f"   Result keys: {list(result.keys())}")
                            if 'evaluation_results' in result:
                                logger.debug(f"   Direct evaluation_results found")
                    
                    if baseline_likes and steered_likes:
                        # Filter out inf and nan values
                        valid_pairs = []
                        for b, s in zip(baseline_likes, steered_likes):
                            if np.isfinite(b) and np.isfinite(s):
                                valid_pairs.append((b, s))
                        
                        if valid_pairs:
                            changes = [abs(s - b) for b, s in valid_pairs]
                            steering_effect = sum(changes) / len(changes) if changes else 0.0
                            
                            # Cap steering effect to prevent infinity
                            steering_effect = min(steering_effect, 100.0)
                            
                            # Also calculate how many preferences changed
                            preference_changes = 0
                            for i in range(0, len(baseline_likes), 2):  # Assuming binary choices
                                if i+1 < len(baseline_likes):
                                    if np.isfinite(baseline_likes[i]) and np.isfinite(baseline_likes[i+1]) and \
                                       np.isfinite(steered_likes[i]) and np.isfinite(steered_likes[i+1]):
                                        baseline_pref = 0 if baseline_likes[i] > baseline_likes[i+1] else 1
                                        steered_pref = 0 if steered_likes[i] > steered_likes[i+1] else 1
                                        if baseline_pref != steered_pref:
                                            preference_changes += 1
                            
                            # Use steering effect as the primary score
                            score = steering_effect
                            
                            # Add bonus if accuracy is valid and good
                            if np.isfinite(accuracy) and accuracy > 0.5:
                                score += accuracy * 0.5
                        else:
                            # No valid likelihood pairs
                            score = 0.0
                            steering_effect = 0.0
                    else:
                        # Fallback to accuracy if no likelihood data
                        score = accuracy if np.isfinite(accuracy) else 0.0
                
                results.append({
                    'strength': float(strength),
                    'score': score,
                    'steering_effect': steering_effect,
                    'evaluation_results': eval_results if isinstance(result, dict) else {}
                })
                
                if score > best_score:
                    best_score = score
                    best_strength = float(strength)
                
                logger.info(f"   Strength {strength:.2f}: score={score:.3f}, effect={steering_effect:.3f}, accuracy={accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"   Error testing strength {strength}: {e}")
                results.append({
                    'strength': float(strength),
                    'score': 0.0,
                    'error': str(e)
                })
        
        return SteeringOptimizationResult(
            task_name=task_name,
            best_steering_layer=layer,
            best_steering_method=steering_method.value,
            best_steering_strength=best_strength,
            optimal_parameters={'strength': best_strength},
            steering_effectiveness_score=best_score,
            classification_accuracy_impact=best_score,  # Using same score for now
            optimization_time_seconds=0.0,  # TODO: Track actual time
            total_configurations_tested=len(results),
            error_message=None
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
    
    def _parse_layer_range(self, layer_range: str) -> List[int]:
        """Parse layer range string like '10-20' or '10,12,14'."""
        if '-' in layer_range:
            start, end = map(int, layer_range.split('-'))
            return list(range(start, end + 1))
        elif ',' in layer_range:
            return [int(x.strip()) for x in layer_range.split(',')]
        else:
            return [int(layer_range)]
    
    def _evaluate_steering_configuration(
        self,
        task_name: str,
        method: SteeringMethod,
        layer: int,
        strength: float,
        limit: int,
        split_ratio: float,
        method_params: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate a single steering configuration and return its effectiveness score.
        
        Args:
            method_params: Additional method-specific parameters
        
        Returns:
            Effectiveness score (0.0 to 1.0)
        """
        try:
            # Import CLI runner to test configuration
            from wisent_guard.cli import run_task_pipeline
            
            # Prepare kwargs with method-specific parameters
            kwargs = {
                'task_name': task_name,
                'model_name': self.model_name,
                'layer': str(layer),
                'limit': limit,
                'steering_mode': True,
                'steering_method': method.value,
                'steering_strength': strength,
                'split_ratio': split_ratio,
                'device': self.device,
                'verbose': False,
                'allow_small_dataset': True
            }
            
            # Add method-specific parameters
            if method_params:
                # Map parameter names to CLI argument names
                param_mapping = {
                    'normalization_method': 'normalization_method',
                    'hpr_beta': 'hpr_beta',
                    'dac_dynamic_control': 'dac_dynamic_control',
                    'dac_entropy_threshold': 'dac_entropy_threshold',
                    'bipo_beta': 'bipo_beta',
                    'bipo_epochs': 'bipo_epochs',
                    'ksteering_alpha': 'ksteering_alpha',
                    'ksteering_target_labels': 'ksteering_target_labels',
                    'ksteering_avoid_labels': 'ksteering_avoid_labels'
                }
                
                for param_key, param_value in method_params.items():
                    if param_key in param_mapping:
                        kwargs[param_mapping[param_key]] = param_value
            
            # Run steering evaluation
            result = run_task_pipeline(**kwargs)
            
            # Extract evaluation score
            # Priority: accuracy > likelihood change > 0.0
            if 'accuracy' in result and result['accuracy'] != 'N/A':
                return float(result['accuracy'])
            elif 'evaluation_results' in result:
                eval_results = result['evaluation_results']
                if 'accuracy' in eval_results and eval_results['accuracy'] != 'N/A':
                    return float(eval_results['accuracy'])
                # Could also use likelihood changes as a metric
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Configuration evaluation failed: {e}")
            return 0.0
    
    def _save_steering_optimization_results(self, summary: SteeringOptimizationSummary):
        """Save optimization results to configuration."""
        config = self.config_manager.load_model_config(self.model_name) or {
            'model_name': self.model_name,
            'created_date': datetime.now().isoformat(),
            'config_version': '2.0'
        }
        
        # Add steering optimization results
        if 'steering_optimization' not in config:
            config['steering_optimization'] = {}
        
        # Save overall best configuration
        config['steering_optimization']['best_method'] = summary.best_overall_method
        config['steering_optimization']['best_layer'] = summary.best_overall_layer
        config['steering_optimization']['best_strength'] = summary.best_overall_strength
        config['steering_optimization']['optimization_date'] = summary.optimization_date
        config['steering_optimization']['method_ranking'] = summary.method_performance_ranking
        
        # Save task-specific results
        if 'task_specific_steering' not in config:
            config['task_specific_steering'] = {}
        
        for task_result in summary.task_results:
            config['task_specific_steering'][task_result.task_name] = {
                'method': task_result.best_steering_method,
                'layer': task_result.best_steering_layer,
                'strength': task_result.best_steering_strength,
                'score': task_result.steering_effectiveness_score,
                'parameters': task_result.optimal_parameters
            }
        
        # Update configuration
        self.config_manager.update_model_config(self.model_name, config)
        logger.info(f"✅ Steering optimization results saved for {self.model_name}")
    
    def load_optimal_steering_config(self, task_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load optimal steering configuration for a model/task.
        
        Args:
            task_name: Optional task name for task-specific configuration
            
        Returns:
            Dictionary with optimal steering parameters or None
        """
        config = self.config_manager.load_model_config(self.model_name)
        if not config:
            return None
        
        # Check for task-specific configuration first
        if task_name and 'task_specific_steering' in config:
            task_config = config['task_specific_steering'].get(task_name)
            if task_config:
                return task_config
        
        # Fall back to overall best configuration
        if 'steering_optimization' in config:
            steering_opt = config['steering_optimization']
            return {
                'method': steering_opt.get('best_method'),
                'layer': steering_opt.get('best_layer'),
                'strength': steering_opt.get('best_strength')
            }
        
        return None
    
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
        # Use the internal evaluation method
        score = self._evaluate_steering_configuration(
            task_name=task_name,
            method=steering_method,
            layer=layer,
            strength=strength,
            limit=len(test_samples),
            split_ratio=0.8
        )
        
        return {
            'effectiveness_score': score,
            'accuracy': score,  # For now, use the same score
            'consistency': 1.0 if score > 0.5 else 0.5,
            'direction_accuracy': score
        }


# Convenience functions for CLI integration
def run_steering_optimization(
    model_name: str,
    optimization_type: str = "auto",
    task_name: str = None,
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    **kwargs
) -> Union[SteeringOptimizationResult, SteeringOptimizationSummary, Dict[str, Any]]:
    """
    Convenience function to run steering optimization.
    
    Args:
        model_name: Model to optimize steering for
        optimization_type: Type of optimization ("auto", "method_comparison", "layer", "strength", "comprehensive")
        task_name: Task to optimize for (if None and optimization_type="auto", uses all classification-optimized tasks)
        limit: Sample limit
        device: Device to use
        verbose: Enable verbose logging
        use_classification_config: Whether to use existing classification config as starting point
        **kwargs: Additional arguments for specific optimization types
        
    Returns:
        SteeringOptimizationResult, SteeringOptimizationSummary, or auto-optimization results
    """
    optimizer = SteeringOptimizer(
        model_name=model_name,
        device=device,
        verbose=verbose
    )
    
    if optimization_type == "auto":
        # Automatic optimization based on classification config
        return run_auto_steering_optimization(
            model_name=model_name,
            task_name=task_name,
            limit=limit,
            device=device,
            verbose=verbose,
            use_classification_config=use_classification_config,
            **kwargs
        )
    elif optimization_type == "method_comparison":
        if not task_name:
            raise ValueError("task_name required for method comparison")
        return optimizer.optimize_steering_method_comparison(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "layer":
        if not task_name:
            raise ValueError("task_name required for layer optimization")
        
        # Convert string steering_method to enum if needed
        if 'steering_method' in kwargs and isinstance(kwargs['steering_method'], str):
            kwargs['steering_method'] = SteeringMethod[kwargs['steering_method']]
        
        return optimizer.optimize_steering_layer(
            task_name=task_name,
            limit=limit,
            **kwargs
        )
    elif optimization_type == "strength":
        if not task_name:
            raise ValueError("task_name required for strength optimization")
        
        # Convert string steering_method to enum if needed
        if 'steering_method' in kwargs and isinstance(kwargs['steering_method'], str):
            kwargs['steering_method'] = SteeringMethod[kwargs['steering_method']]
        
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


def run_auto_steering_optimization(
    model_name: str,
    task_name: Optional[str] = None,
    limit: int = 100,
    device: str = None,
    verbose: bool = False,
    use_classification_config: bool = True,
    max_time_minutes: float = 60.0,
    methods_to_test: Optional[List[str]] = None,
    strength_range: Optional[List[float]] = None,
    layer_range: Optional[str] = None
) -> Dict[str, Any]:
    """
    Automatically optimize steering configuration.
    
    This function can work either standalone or building on existing classification config.
    
    Args:
        model_name: Model to optimize
        task_name: Specific task to optimize (required if no classification config)
        limit: Sample limit per evaluation
        device: Device to use
        verbose: Enable verbose logging
        use_classification_config: Use classification layer as starting point
        max_time_minutes: Maximum time for optimization
        methods_to_test: List of steering methods to test (defaults to ["CAA", "HPR"])
        strength_range: List of strengths to test (defaults to [0.5, 1.0, 1.5, 2.0])
        layer_range: Explicit layer range to search (e.g. "0-5" or "0,2,4")
        
    Returns:
        Dictionary with optimization results and saved configuration paths
    """
    optimizer = SteeringOptimizer(
        model_name=model_name,
        device=device,
        verbose=verbose
    )
    
    # Load classification config if requested
    config_manager = ModelConfigManager()
    classification_config = None
    if use_classification_config:
        classification_config = config_manager.load_model_config(model_name)
        if not classification_config:
            logger.info("ℹ️  No classification config found, proceeding with standalone steering optimization")
    
    # Determine tasks to optimize
    if task_name:
        tasks_to_optimize = [task_name]
    elif classification_config:
        # First try task-specific overrides
        if 'task_specific_overrides' in classification_config:
            tasks_to_optimize = list(classification_config['task_specific_overrides'].keys())
        
        # If no task-specific overrides, check for tasks from optimization metrics
        if not tasks_to_optimize and 'optimization_metrics' in classification_config:
            # Try to get tasks from sample sizes
            if 'optimal_sample_sizes' in classification_config:
                tasks_to_optimize = list(classification_config['optimal_sample_sizes'].keys())
        
        if not tasks_to_optimize:
            return {"error": "No task specified and no classification tasks found in config"}
    else:
        # Require explicit task name if no classification config
        return {"error": "Task name required when not using classification config"}
    
    # Default methods and strengths
    if methods_to_test is None:
        # Use all default configurations
        method_configs = get_default_steering_configs()
    else:
        # Convert string methods to configs
        method_configs = []
        for method in methods_to_test:
            if method == "CAA":
                # Add both CAA variations
                method_configs.append(SteeringMethodConfig("CAA", SteeringMethod.CAA, {}))
                method_configs.append(SteeringMethodConfig("CAA_L2", SteeringMethod.CAA, {"normalization_method": "l2_unit"}))
            elif method == "HPR":
                # Add both HPR variations
                method_configs.append(SteeringMethodConfig("HPR", SteeringMethod.HPR, {"hpr_beta": 1.0}))
                method_configs.append(SteeringMethodConfig("HPR_Beta0.5", SteeringMethod.HPR, {"hpr_beta": 0.5}))
            elif method == "DAC":
                method_configs.append(SteeringMethodConfig("DAC", SteeringMethod.DAC, {"dac_dynamic_control": True, "dac_entropy_threshold": 1.0}))
            elif method == "BiPO":
                # Add both BiPO variations
                method_configs.append(SteeringMethodConfig("BiPO", SteeringMethod.BIPO, {"bipo_beta": 0.1, "bipo_epochs": 50}))
                method_configs.append(SteeringMethodConfig("BiPO_Beta0.05", SteeringMethod.BIPO, {"bipo_beta": 0.05, "bipo_epochs": 50}))
            elif method == "KSteering":
                # Add both KSteering variations
                method_configs.append(SteeringMethodConfig("KSteering", SteeringMethod.KSTEERING, {"ksteering_alpha": 5.0, "ksteering_target_labels": "0", "ksteering_avoid_labels": ""}))
                method_configs.append(SteeringMethodConfig("KSteering_Alpha3", SteeringMethod.KSTEERING, {"ksteering_alpha": 3.0, "ksteering_target_labels": "0", "ksteering_avoid_labels": ""}))
            else:
                logger.warning(f"Unknown steering method: {method}")
    
    if strength_range is None:
        strength_range = [0.5, 1.0, 1.5, 2.0]
    
    if verbose:
        logger.info(f"🚀 Starting automatic steering optimization")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Tasks: {tasks_to_optimize}")
        logger.info(f"   Method configurations: {[cfg.name for cfg in method_configs]}")
        logger.info(f"   Time limit: {max_time_minutes} minutes")
    
    results = {
        'model_name': model_name,
        'optimization_date': datetime.now().isoformat(),
        'tasks_optimized': [],
        'overall_best': None,
        'config_saved': False
    }
    
    # Optimize each task
    time_per_task = max_time_minutes / len(tasks_to_optimize)
    
    for task in tasks_to_optimize:
        if verbose:
            logger.info(f"\n📊 Optimizing steering for task: {task}")
        
        # Determine layer range
        task_layer_range = layer_range  # Use provided layer range
        if not task_layer_range and classification_config and use_classification_config:
            # Only use classification config if no explicit layer range provided
            task_overrides = classification_config.get('task_specific_overrides', {}).get(task, {})
            class_layer = task_overrides.get('classification_layer')
            
            if not class_layer:
                # Use global classification layer
                class_layer = classification_config.get('optimal_parameters', {}).get('classification_layer')
            
            if class_layer:
                # Search around classification layer
                task_layer_range = f"{max(0, class_layer-2)}-{class_layer+2}"
                if verbose:
                    logger.info(f"   Using layer range around classification layer {class_layer}: {task_layer_range}")
        
        # If still no layer range, use default based on model type
        if not task_layer_range:
            # Default to searching early to middle layers
            task_layer_range = "0-5"
            if verbose:
                logger.info(f"   Using default layer range: {task_layer_range}")
        
        # Run optimization for this task
        try:
            summary = optimizer.optimize_steering_method_comparison(
                task_name=task,
                methods_to_test=method_configs,
                layer_range=task_layer_range,
                strength_range=strength_range,
                limit=limit,
                max_time_minutes=time_per_task
            )
            
            # Store results
            task_result = {
                'task': task,
                'best_method': summary.best_overall_method,
                'best_layer': summary.best_overall_layer,
                'best_strength': summary.best_overall_strength,
                'score': summary.task_results[0].steering_effectiveness_score if summary.task_results else 0.0
            }
            results['tasks_optimized'].append(task_result)
            
            # Update overall best
            if not results['overall_best'] or task_result['score'] > results['overall_best']['score']:
                results['overall_best'] = task_result
                
        except Exception as e:
            logger.error(f"❌ Failed to optimize task {task}: {e}")
            results['tasks_optimized'].append({
                'task': task,
                'error': str(e)
            })
    
    # Save configuration
    if results['tasks_optimized'] and not any('error' in r for r in results['tasks_optimized']):
        results['config_saved'] = True
        results['config_path'] = config_manager._get_config_path(model_name)
        
        if verbose:
            logger.info(f"\n✅ Steering optimization complete!")
            logger.info(f"   Configuration saved to: {results['config_path']}")
            logger.info(f"   Overall best: {results['overall_best']['best_method']} "
                       f"L{results['overall_best']['best_layer']} "
                       f"S{results['overall_best']['best_strength']}")
    
    return results


def get_optimal_steering_params(
    model_name: str,
    task_name: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get optimal steering parameters for a model/task.
    
    Args:
        model_name: Model name
        task_name: Optional task name for task-specific params
        
    Returns:
        Dictionary with steering parameters or None
    """
    optimizer = SteeringOptimizer(model_name)
    return optimizer.load_optimal_steering_config(task_name)


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