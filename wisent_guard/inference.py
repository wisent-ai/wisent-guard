"""
Safe inference functionality for wisent-guard.
Clean implementation using enhanced core primitives.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from .core import Model, SteeringMethod, SteeringType

logger = logging.getLogger(__name__)


class SafeInference:
    """
    Safe inference engine using enhanced core primitives.
    """
    
    def __init__(
        self,
        model_name: str,
        layer: int = 15,
        steering_type: SteeringType = SteeringType.LOGISTIC,
        device: Optional[str] = None,
        threshold: float = 0.5
    ):
        """
        Initialize safe inference engine.
        
        Args:
            model_name: Language model name or path
            layer: Layer for activation extraction
            steering_type: Type of steering method
            device: Target device
            threshold: Safety threshold
        """
        self.model = Model(name=model_name, device=device)
        self.layer = layer
        self.steering_method = SteeringMethod(
            method_type=steering_type,
            device=device
        )
        self.threshold = threshold
        self.is_trained = False
        
        logger.info(f"Initialized SafeInference with {model_name}")
    
    def train_safety_filter(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Train the safety filter.
        
        Args:
            harmful_texts: List of harmful examples
            harmless_texts: List of harmless examples
            
        Returns:
            Training results
        """
        from .core import ContrastivePairSet, Layer
        
        # Create phrase pairs
        phrase_pairs = []
        min_len = min(len(harmful_texts), len(harmless_texts))
        
        for i in range(min_len):
            phrase_pairs.append({
                "harmful": harmful_texts[i],
                "harmless": harmless_texts[i]
            })
        
        # Create and train
        pair_set = ContrastivePairSet.from_phrase_pairs(
            name="safety_training",
            phrase_pairs=phrase_pairs,
            task_type="safety_inference"
        )
        
        layer_obj = Layer(index=self.layer, type="transformer")
        results = pair_set.train_classifier(self.steering_method.classifier, layer_obj)
        
        self.is_trained = True
        logger.info(f"Safety filter training completed: {results}")
        return results
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text with safety filtering.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, safety_info)
        """
        if not self.is_trained:
            logger.warning("Safety filter not trained, generating without filtering")
            response, _ = self.model.generate(prompt, self.layer, max_new_tokens, **generation_kwargs)
            return response, {"filtered": False, "reason": "No safety filter"}
        
        # Generate with safety monitoring
        def safety_callback(token_text: str, activations) -> bool:
            safety_result = self.steering_method.check_safety(token_text, self.threshold)
            return not safety_result.get("is_harmful", False)
        
        response, activations = self.model.generate_monitored(
            prompt=prompt,
            layer=self.layer,
            max_new_tokens=max_new_tokens,
            safety_callback=safety_callback,
            **generation_kwargs
        )
        
        # Final safety check
        final_safety = self.steering_method.check_safety(response, self.threshold)
        
        return response, final_safety
    
    def check_safety(self, text: str) -> Dict[str, Any]:
        """
        Check if text is safe.
        
        Args:
            text: Text to check
            
        Returns:
            Safety assessment
        """
        if not self.is_trained:
            return {"is_harmful": False, "reason": "No safety filter trained"}
        
        return self.steering_method.check_safety(text, self.threshold)
    
    def save_model(self, path: str) -> None:
        """Save trained safety filter."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained safety filter")
        
        self.steering_method.save_model(path)
        logger.info(f"Safety filter saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained safety filter."""
        self.steering_method.load_model(path)
        self.is_trained = True
        logger.info(f"Safety filter loaded from {path}")


def create_safe_inference(
    model_name: str,
    harmful_examples: List[str],
    harmless_examples: List[str],
    layer: int = 15,
    steering_type: SteeringType = SteeringType.LOGISTIC,
    device: Optional[str] = None
) -> SafeInference:
    """
    Create and train a safe inference engine in one step.
    
    Args:
        model_name: Language model name
        harmful_examples: Harmful text examples
        harmless_examples: Harmless text examples
        layer: Layer for activation extraction
        steering_type: Type of steering method
        device: Target device
        
    Returns:
        Trained SafeInference
    """
    inference = SafeInference(
        model_name=model_name,
        layer=layer,
        steering_type=steering_type,
        device=device
    )
    
    inference.train_safety_filter(harmful_examples, harmless_examples)
    return inference 