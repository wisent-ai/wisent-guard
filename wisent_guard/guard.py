"""
Wisent Guard: Real-time safety monitoring for language models.
Clean implementation using enhanced core primitives.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from .core import Model, SteeringMethod, SteeringType, Layer

logger = logging.getLogger(__name__)


class WisentGuard:
    """
    Main guard class for real-time safety monitoring.
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
        Initialize Wisent Guard.
        
        Args:
            model_name: Language model name or path
            layer: Layer for activation extraction
            steering_type: Type of steering method
            device: Target device
            threshold: Safety threshold
        """
        self.model = Model(name=model_name, device=device)
        self.layer_obj = Layer(index=layer, type="transformer")
        self.steering_method = SteeringMethod(
            method_type=steering_type,
            device=device
        )
        self.threshold = threshold
        self.is_trained = False
        
        logger.info(f"Initialized WisentGuard with {model_name} on layer {layer}")
    
    def train_from_pairs(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str]
    ) -> Dict[str, Any]:
        """
        Train the guard from harmful/harmless text pairs.
        
        Args:
            harmful_texts: List of harmful text examples
            harmless_texts: List of harmless text examples
            
        Returns:
            Training results
        """
        from .core import ContrastivePairSet
        
        # Create phrase pairs
        phrase_pairs = []
        min_len = min(len(harmful_texts), len(harmless_texts))
        
        for i in range(min_len):
            phrase_pairs.append({
                "harmful": harmful_texts[i],
                "harmless": harmless_texts[i]
            })
        
        # Create ContrastivePairSet
        pair_set = ContrastivePairSet.from_phrase_pairs(
            name="guard_training",
            phrase_pairs=phrase_pairs,
            task_type="safety_training"
        )
        
        # Train classifier
        results = pair_set.train_classifier(
            self.steering_method.classifier,
            self.layer_obj
        )
        
        self.is_trained = True
        logger.info(f"Guard training completed: {results}")
        return results
    
    def check_safety(self, text: str) -> Dict[str, Any]:
        """
        Check if text is safe.
        
        Args:
            text: Text to check
            
        Returns:
            Safety assessment
        """
        if not self.is_trained:
            raise ValueError("Guard must be trained before use")
        
        return self.steering_method.check_safety(text, self.threshold)
    
    def generate_safe(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text with real-time safety monitoring.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, safety_info)
        """
        if not self.is_trained:
            raise ValueError("Guard must be trained before use")
        
        def safety_callback(token_text: str, activations) -> bool:
            """Callback to check safety during generation."""
            safety_result = self.steering_method.check_safety(token_text, self.threshold)
            return not safety_result.get("is_harmful", False)
        
        # Generate with monitoring
        response, activations = self.model.generate_monitored(
            prompt=prompt,
            layer=self.layer_obj.index,
            max_new_tokens=max_new_tokens,
            safety_callback=safety_callback,
            **generation_kwargs
        )
        
        # Final safety check
        final_safety = self.check_safety(response)
        
        return response, final_safety
    
    def save(self, path: str) -> None:
        """Save trained guard."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained guard")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.steering_method.save_model(str(save_path))
        logger.info(f"Guard saved to {path}")
    
    def load(self, path: str) -> None:
        """Load trained guard."""
        self.steering_method.load_model(path)
        self.is_trained = True
        logger.info(f"Guard loaded from {path}")


def create_guard(
    model_name: str,
    harmful_examples: List[str],
    harmless_examples: List[str],
    layer: int = 15,
    steering_type: SteeringType = SteeringType.LOGISTIC,
    device: Optional[str] = None
) -> WisentGuard:
    """
    Create and train a guard in one step.
        
        Args:
        model_name: Language model name
        harmful_examples: Harmful text examples
        harmless_examples: Harmless text examples
        layer: Layer for activation extraction
        steering_type: Type of steering method
        device: Target device
        
        Returns:
        Trained WisentGuard
    """
    guard = WisentGuard(
        model_name=model_name,
        layer=layer,
        steering_type=steering_type,
        device=device
    )
    
    guard.train_from_pairs(harmful_examples, harmless_examples)
    return guard 