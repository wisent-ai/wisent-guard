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
        
        # Generate with token-level classification
        response, token_scores, classification = self.generate_with_classification(
            prompt, max_new_tokens, **generation_kwargs
        )
        
        # Prepare safety info
        avg_score = sum(token_scores) / len(token_scores) if token_scores else 0.0
        safety_info = {
            "classification": classification,
            "token_scores": token_scores,
            "average_score": avg_score,
            "is_harmful": classification == "HALLUCINATION",
            "filtered": False
        }
        
        return response, safety_info
    
    def generate_with_classification(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> Tuple[str, List[float], str]:
        """
        Generate text with token-level hallucination classification.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (response_text, token_scores, classification)
        """
        import torch
        from .core import Layer, Activations, ActivationAggregationMethod
        
        # Generate response
        response, _ = self.model.generate(prompt, self.layer, max_new_tokens, **generation_kwargs)
        
        if not response.strip():
            return response, [], "UNKNOWN"
        
        # Tokenize the full prompt + response to get individual tokens
        full_text = f"{prompt}{response}"
        inputs = self.model.tokenizer(full_text, return_tensors="pt")
        tokens = self.model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get prompt length to identify response tokens
        prompt_inputs = self.model.tokenizer(prompt, return_tensors="pt")
        prompt_length = len(prompt_inputs['input_ids'][0])
        
        # Extract activations for each response token
        token_scores = []
        layer_obj = Layer(index=self.layer, type="transformer")
        
        try:
            # Get model device
            model_device = next(self.model.hf_model.parameters()).device
            inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Get hidden states for the full sequence
            with torch.no_grad():
                outputs = self.model.hf_model(**inputs_on_device, output_hidden_states=True)
            
            # Extract activations for response tokens only
            if self.layer + 1 < len(outputs.hidden_states):
                hidden_states = outputs.hidden_states[self.layer + 1]
            else:
                hidden_states = outputs.hidden_states[-1]
            
            # Score each response token
            for token_idx in range(prompt_length, len(tokens)):
                if token_idx < hidden_states.shape[1]:
                    # Extract activation for this token
                    token_activation = hidden_states[0, token_idx, :].cpu()
                    
                    # Create Activations object
                    activation_obj = Activations(
                        tensor=token_activation.unsqueeze(0),
                        layer=layer_obj,
                        aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                    )
                    
                    # Get feature vector for classifier
                    features = activation_obj.extract_features_for_classifier()
                    
                    # Get prediction probability from classifier
                    if hasattr(self.steering_method, 'classifier') and self.steering_method.classifier:
                        try:
                            # Predict probability of being harmful (class 1)
                            prob = self.steering_method.classifier.predict_proba([features.numpy()])[0][1]
                            token_scores.append(float(prob))
                        except Exception as e:
                            logger.warning(f"Classifier error for token {token_idx}: {e}")
                            token_scores.append(0.5)
                    else:
                        token_scores.append(0.5)
        
        except Exception as e:
            logger.warning(f"Error during token scoring: {e}")
            # Fallback: assign neutral scores
            response_tokens = self.model.tokenizer(response, return_tensors="pt")['input_ids'][0]
            token_scores = [0.5] * len(response_tokens)
        
        # Classify overall response
        if token_scores:
            avg_score = sum(token_scores) / len(token_scores)
            classification = "HALLUCINATION" if avg_score > 0.6 else "TRUTHFUL"
        else:
            classification = "UNKNOWN"
        
        return response, token_scores, classification
    
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