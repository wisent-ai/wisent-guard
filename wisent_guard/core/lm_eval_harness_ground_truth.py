"""
LM-Eval-Harness Ground Truth Evaluation

This module provides ground truth evaluation using the lm-eval-harness framework.
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class LMEvalHarnessGroundTruth:
    """
    Ground truth evaluator using lm-eval-harness tasks.
    
    This class orchestrates the evaluation of classifiers on lm-eval-harness tasks
    by routing to appropriate evaluation methods based on the task type.
    """
    
    def __init__(self, task_name: str, evaluation_method: str = None, model=None):
        """
        Initialize the LM-eval-harness ground truth evaluator.
        
        Args:
            task_name: Name of the lm-eval task
            evaluation_method: Evaluation method ("log-likelihoods", "text-generation", "perplexity")
            model: The model instance for activation extraction
        """
        self.task_name = task_name
        self.evaluation_method = evaluation_method
        self.model = model
        
        # Load evaluation method from benchmark configuration if not provided
        if not self.evaluation_method:
            self.evaluation_method = self._get_evaluation_method_for_task(task_name)
    
    def evaluate_classifier_on_task(self, 
                                   classifier, 
                                   task_name: str, 
                                   num_samples: int = 100,
                                   model=None,
                                   layer: int = 15,
                                   token_aggregation: str = "average") -> Dict[str, Any]:
        """
        Evaluate a classifier on the specified lm-eval task.
        
        Args:
            classifier: The classifier to evaluate
            task_name: Name of the lm-eval task
            num_samples: Number of samples to evaluate
            model: The model instance (overrides self.model if provided) 
            layer: Layer to extract activations from
            token_aggregation: Token aggregation method ("average", "final", "first", "max", "min")
            
        Returns:
            Dict containing evaluation results
        """
        # Use provided model or fall back to self.model
        evaluation_model = model or self.model
        
        # Route to appropriate evaluation method
        if self.evaluation_method == "log-likelihoods":
            return self._evaluate_log_likelihoods(classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        elif self.evaluation_method == "text-generation":
            return self._evaluate_text_generation(classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        elif self.evaluation_method == "perplexity":
            return self._evaluate_perplexity(classifier, task_name, num_samples, evaluation_model, layer, token_aggregation)
        else:
            return {
                "ground_truth": "UNKNOWN",
                "method_used": "lm-eval-harness-unsupported",
                "confidence": 0.0,
                "details": f"Unsupported evaluation method: {self.evaluation_method}",
                "task_name": task_name,
                "evaluation_method": self.evaluation_method
            }
    
    def _evaluate_log_likelihoods(self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average") -> Dict[str, Any]:
        """Evaluate classifier using log-likelihoods approach."""
        try:
            from .log_likelihoods_evaluator import LogLikelihoodsEvaluator
            
            # Create evaluator with model
            evaluator = LogLikelihoodsEvaluator(task_name, model=model)
            
            # Evaluate classifier
            results = evaluator.evaluate_classifier_on_task(
                classifier, 
                task_name, 
                num_samples=num_samples,
                model=model,
                layer=layer,
                token_aggregation=token_aggregation
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in log-likelihoods evaluation: {e}")
            return {
                "ground_truth": "UNKNOWN",
                "method_used": "lm-eval-harness-error",
                "confidence": 0.0,
                "details": f"Log-likelihoods evaluation failed: {str(e)}",
                "task_name": task_name,
                "evaluation_method": "log-likelihoods"
            }
    
    def _evaluate_text_generation(self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average") -> Dict[str, Any]:
        """Evaluate classifier using text generation approach."""
        # Placeholder for text generation evaluation
        # This would involve generating text and then classifying it
        return {
            "ground_truth": "UNKNOWN",
            "method_used": "lm-eval-harness-text-generation",
            "confidence": 0.0,
            "details": "Text generation evaluation not yet implemented",
            "task_name": task_name,
            "evaluation_method": "text-generation"
        }
    
    def _evaluate_perplexity(self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average") -> Dict[str, Any]:
        """Evaluate classifier using perplexity approach."""
        # Placeholder for perplexity evaluation
        # This would involve computing perplexity and then classifying
        return {
            "ground_truth": "UNKNOWN",
            "method_used": "lm-eval-harness-perplexity",
            "confidence": 0.0,
            "details": "Perplexity evaluation not yet implemented",
            "task_name": task_name,
            "evaluation_method": "perplexity"
        }
    
    def _get_evaluation_method_for_task(self, task_name: str) -> str:
        """Get the evaluation method for a task from the benchmark configuration."""
        try:
            import json
            eval_methods_path = "wisent_guard/parameters/benchmarks/benchmark_evaluation_methods.json"
            with open(eval_methods_path, 'r') as f:
                benchmark_methods = json.load(f)
                return benchmark_methods.get(task_name, "text-generation")
        except Exception as e:
            logger.debug(f"Could not load benchmark evaluation methods: {e}")
            return "text-generation" 