import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations
from wisent_guard.core.layer import Layer

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
        threshold: float = 0.5,
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
        self.steering_method = SteeringMethod(method_type=steering_type, device=device)
        self.threshold = threshold
        self.is_trained = False

        logger.info(f"Initialized SafeInference with {model_name}")

    def train_safety_filter(self, harmful_texts: List[str], harmless_texts: List[str]) -> Dict[str, Any]:
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
            phrase_pairs.append({"harmful": harmful_texts[i], "harmless": harmless_texts[i]})

        # Create and train
        pair_set = ContrastivePairSet.from_phrase_pairs(
            name="safety_training", phrase_pairs=phrase_pairs, task_type="safety_inference"
        )

        layer_obj = Layer(index=self.layer, type="transformer")
        results = pair_set.train_classifier(self.steering_method.classifier, layer_obj)

        self.is_trained = True
        logger.info(f"Safety filter training completed: {results}")
        return results

    def generate(self, prompt: str, max_new_tokens: int = 50, **generation_kwargs) -> Tuple[str, Dict[str, Any]]:
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
            "filtered": False,
        }

        return response, safety_info

    def generate_with_classification(
        self, prompt: str, max_new_tokens: int = 50, **generation_kwargs
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

        # Generate response
        response, _ = self.model.generate(prompt, self.layer, max_new_tokens, **generation_kwargs)

        if not response.strip():
            return response, [], "UNKNOWN"

        # Tokenize the full prompt + response to get individual tokens
        full_text = f"{prompt}{response}"
        inputs = self.model.tokenizer(full_text, return_tensors="pt")
        tokens = self.model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get prompt length to identify response tokens
        prompt_inputs = self.model.tokenizer(prompt, return_tensors="pt")
        prompt_length = len(prompt_inputs["input_ids"][0])

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
                        aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                    )

                    # Get feature vector for classifier
                    features = activation_obj.extract_features_for_classifier()

                    # Get prediction probability from classifier
                    if hasattr(self.steering_method, "classifier") and self.steering_method.classifier:
                        try:
                            # Predict probability of being harmful (class 1)
                            # Our classifier returns a single float, not an array like sklearn
                            prob = self.steering_method.classifier.predict_proba([features.numpy()])
                            # Handle both single float and array returns
                            if isinstance(prob, (list, tuple)) or hasattr(prob, "__getitem__"):
                                if len(prob) > 0:
                                    if hasattr(prob[0], "__len__") and len(prob[0]) > 1:
                                        prob = float(prob[0][1])  # Binary classification - positive class
                                    else:
                                        prob = float(prob[0])
                                else:
                                    prob = 0.5
                            else:
                                prob = float(prob)

                            # Ensure we have a valid float
                            if prob is None or not isinstance(prob, (int, float)):
                                raise ValueError(
                                    f"Classifier returned invalid probability: {prob} (type: {type(prob)}). Expected float but got {type(prob).__name__}"
                                )

                            # Additional validation
                            if not (0.0 <= prob <= 1.0):
                                raise ValueError(f"Classifier probability {prob} is out of valid range [0.0, 1.0]")

                            token_scores.append(float(prob))
                        except Exception as e:
                            logger.warning(f"Classifier error for token {token_idx}: {e}")
                            token_scores.append(0.5)
                    else:
                        token_scores.append(0.5)

        except Exception as e:
            logger.warning(f"Error during token scoring: {e}")
            # Fallback: assign neutral scores
            response_tokens = self.model.tokenizer(response, return_tensors="pt")["input_ids"][0]
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
    device: Optional[str] = None,
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
    inference = SafeInference(model_name=model_name, layer=layer, steering_type=steering_type, device=device)

    inference.train_safety_filter(harmful_examples, harmless_examples)
    return inference


def generate_with_classification_and_handling(
    model,
    prompt,
    layer,
    max_new_tokens,
    steering_method,
    token_aggregation="average",
    threshold=0.6,
    verbose=False,
    detection_handler=None,
):
    """
    Generate text with token-level classification and optional detection handling.

    Args:
        model: Model object
        prompt: Input prompt
        layer: Layer index for activation extraction
        max_new_tokens: Maximum tokens to generate
        steering_method: Trained steering method with classifier
        token_aggregation: How to aggregate token scores ("average", "final", "first", "max", "min")
        threshold: Classification threshold
        verbose: Whether to print debug info
        detection_handler: Optional DetectionHandler for handling detected issues

    Returns:
        Tuple of (final_response_text, token_scores, classification, was_handled)
    """
    from .core.detection_handling import DetectionAction
    from .core.parser import aggregate_token_scores

    # Generate initial response with classification
    original_response, token_scores, classification = generate_with_classification(
        model, prompt, layer, max_new_tokens, steering_method, token_aggregation, threshold, verbose
    )

    # If no handler provided or no detection, return as-is
    if not detection_handler:
        return original_response, token_scores, classification, False

    # Determine if content should be handled (detected as problematic)
    is_problematic = classification == "HALLUCINATION"

    if not is_problematic:
        # Content is fine, return as-is
        return original_response, token_scores, classification, False

    # Content is problematic, apply handling
    if verbose:
        print(f"      🚨 Detected problematic content, applying {detection_handler.action.value} handling...")

    # Calculate confidence score for the detection
    if token_scores:
        confidence_score = aggregate_token_scores(token_scores, token_aggregation)
    else:
        confidence_score = 0.5

    # Create regeneration function if needed
    regenerate_function = None
    if detection_handler.action == DetectionAction.REGENERATE_UNTIL_SAFE:

        def regenerate():
            # Generate a new response
            new_response, new_token_scores, new_classification = generate_with_classification(
                model, prompt, layer, max_new_tokens, steering_method, token_aggregation, threshold, verbose
            )
            # Only return if it's not problematic
            if new_classification != "HALLUCINATION":
                return new_response
            # Still problematic, raise exception to trigger retry
            raise Exception(f"Generated response still classified as {new_classification}")

        regenerate_function = regenerate

    # Handle the detection
    final_response = detection_handler.handle_detection(
        original_response=original_response,
        detection_type="hallucination",  # Could be made configurable
        confidence_score=confidence_score,
        original_prompt=prompt,
        regenerate_function=regenerate_function,
    )

    # If regeneration was successful, get new token scores and classification
    if (
        detection_handler.action == DetectionAction.REGENERATE_UNTIL_SAFE
        and final_response != original_response
        and not final_response.startswith("I apologize")
    ):  # Not a fallback placeholder
        # Re-classify the final response
        final_response_clean, final_token_scores, final_classification = generate_with_classification(
            model, prompt, layer, max_new_tokens, steering_method, token_aggregation, threshold, verbose=False
        )

        if final_response_clean.strip() == final_response.strip():
            return final_response, final_token_scores, final_classification, True

    # For placeholder or pass-through, return original scores but indicate handling occurred
    return final_response, token_scores, classification, True


def generate_with_classification(
    model, prompt, layer, max_new_tokens, steering_method, token_aggregation="average", threshold=0.6, verbose=False
):
    """
    Generate text with token-level classification.

    Args:
        model: Model object
        prompt: Input prompt
        layer: Layer index for activation extraction
        max_new_tokens: Maximum tokens to generate
        steering_method: Trained steering method with classifier
        token_aggregation: How to aggregate token scores ("average", "final", "first", "max", "min")
        threshold: Classification threshold
        verbose: Whether to print debug info

    Returns:
        Tuple of (response_text, token_scores, classification)
    """
    import torch

    from .core import Layer
    from .core.parser import aggregate_token_scores

    # Generate response and get token-by-token activations
    response, activations_dict = model.generate(prompt, layer, max_new_tokens)

    if not response.strip():
        return response, [], "UNKNOWN"

    # Tokenize the full prompt + response to get individual tokens
    full_text = f"{prompt}{response}"
    inputs = model.tokenizer(full_text, return_tensors="pt")
    tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get prompt length to identify response tokens
    prompt_inputs = model.tokenizer(prompt, return_tensors="pt")
    prompt_length = len(prompt_inputs["input_ids"][0])

    # Extract activations for each response token
    token_scores = []
    layer_obj = Layer(index=layer, type="transformer")

    try:
        # Get model device
        model_device = next(model.hf_model.parameters()).device
        inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}

        # Get hidden states for the full sequence
        with torch.no_grad():
            outputs = model.hf_model(**inputs_on_device, output_hidden_states=True)

        # Extract activations for response tokens only
        if layer + 1 < len(outputs.hidden_states):
            hidden_states = outputs.hidden_states[layer + 1]  # [batch_size, seq_len, hidden_dim]
        else:
            hidden_states = outputs.hidden_states[-1]

        # Score each response token
        for token_idx in range(prompt_length, len(tokens)):
            if token_idx < hidden_states.shape[1]:
                # Extract activation for this token
                token_activation = hidden_states[0, token_idx, :].cpu()

                # Create Activations object
                activation_obj = Activations(
                    tensor=token_activation.unsqueeze(0),  # Add batch dimension
                    layer=layer_obj,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )

                # Get feature vector for classifier
                features = activation_obj.extract_features_for_classifier()

                # Get prediction probability from classifier
                if hasattr(steering_method, "classifier") and steering_method.classifier:
                    try:
                        # Predict probability of being harmful (class 1)
                        # Our classifier returns a single float, not an array like sklearn
                        prob = steering_method.classifier.predict_proba([features.numpy()])
                        # Handle both single float and array returns
                        if isinstance(prob, (list, tuple)) or hasattr(prob, "__getitem__"):
                            if len(prob) > 0:
                                if hasattr(prob[0], "__len__") and len(prob[0]) > 1:
                                    prob = float(prob[0][1])  # Binary classification - positive class
                                else:
                                    prob = float(prob[0])
                            else:
                                prob = 0.5
                        else:
                            prob = float(prob)

                        # Ensure we have a valid float
                        if prob is None or not isinstance(prob, (int, float)):
                            raise ValueError(
                                f"Classifier returned invalid probability: {prob} (type: {type(prob)}). Expected float but got {type(prob).__name__}"
                            )

                        # Additional validation
                        if not (0.0 <= prob <= 1.0):
                            raise ValueError(f"Classifier probability {prob} is out of valid range [0.0, 1.0]")

                        token_scores.append(float(prob))
                    except Exception as e:
                        if verbose:
                            print(f"      ⚠️  Classifier error for token {token_idx}: {e}")
                        token_scores.append(0.5)  # Neutral score
                else:
                    token_scores.append(0.5)  # Neutral score if no classifier

    except Exception as e:
        if verbose:
            print(f"      ⚠️  Error during token scoring: {e}")
        # Fallback: assign neutral scores
        response_tokens = model.tokenizer(response, return_tensors="pt")["input_ids"][0]
        token_scores = [0.5] * len(response_tokens)

    # Classify overall response using specified aggregation method
    if token_scores:
        aggregated_score = aggregate_token_scores(token_scores, token_aggregation)
        classification = "HALLUCINATION" if aggregated_score > threshold else "TRUTHFUL"
    else:
        aggregated_score = 0.5
        classification = "UNKNOWN"

    return response, token_scores, classification


def generate_with_multi_layer_classification(
    model, prompt, layers, max_new_tokens, steering_methods, token_aggregation="average", threshold=0.6, verbose=False
):
    """
    Generate text with token-level classification across multiple layers.

    Args:
        model: Model object
        prompt: Input prompt
        layers: List of layer indices for activation extraction
        max_new_tokens: Maximum tokens to generate
        steering_methods: Dict mapping layer indices to trained steering methods
        token_aggregation: How to aggregate token scores ("average", "final", "first", "max", "min")
        threshold: Classification threshold
        verbose: Whether to print debug info

    Returns:
        Tuple of (response_text, layer_results_dict)
        where layer_results_dict = {layer: {"token_scores": [...], "classification": "..."}}
    """
    import torch

    from .core import Layer
    from .core.parser import aggregate_token_scores

    # Generate response once (same for all layers)
    response, _ = model.generate(prompt, layers[0], max_new_tokens)

    if not response.strip():
        # Return empty results for all layers
        layer_results = {}
        for layer in layers:
            layer_results[layer] = {"token_scores": [], "classification": "UNKNOWN", "aggregated_score": 0.5}
        return response, layer_results

    # Tokenize the full prompt + response to get individual tokens
    full_text = f"{prompt}{response}"
    inputs = model.tokenizer(full_text, return_tensors="pt")
    tokens = model.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Get prompt length to identify response tokens
    prompt_inputs = model.tokenizer(prompt, return_tensors="pt")
    prompt_length = len(prompt_inputs["input_ids"][0])

    # Get model device and move inputs
    model_device = next(model.hf_model.parameters()).device
    inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}

    # Get hidden states for all layers at once
    with torch.no_grad():
        outputs = model.hf_model(**inputs_on_device, output_hidden_states=True)

    # Process each layer
    layer_results = {}

    for layer in layers:
        if verbose:
            print(f"    Processing layer {layer}...")

        # Extract activations for this layer
        token_scores = []
        layer_obj = Layer(index=layer, type="transformer")

        try:
            # Get hidden states for this layer
            if layer + 1 < len(outputs.hidden_states):
                hidden_states = outputs.hidden_states[layer + 1]
            else:
                hidden_states = outputs.hidden_states[-1]

            # Score each response token for this layer
            for token_idx in range(prompt_length, len(tokens)):
                if token_idx < hidden_states.shape[1]:
                    # Extract activation for this token
                    token_activation = hidden_states[0, token_idx, :].cpu()

                    # Create Activations object
                    activation_obj = Activations(
                        tensor=token_activation.unsqueeze(0),
                        layer=layer_obj,
                        aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                    )

                    # Get feature vector for classifier
                    features = activation_obj.extract_features_for_classifier()

                    # Get prediction probability from classifier for this layer
                    steering_method = steering_methods.get(layer)
                    if steering_method and hasattr(steering_method, "classifier") and steering_method.classifier:
                        try:
                            prob = steering_method.classifier.predict_proba([features.numpy()])
                            # Handle both single float and array returns
                            if isinstance(prob, (list, tuple)) or hasattr(prob, "__getitem__"):
                                prob = float(prob[0]) if len(prob) > 0 else 0.5
                            else:
                                prob = float(prob)
                            token_scores.append(prob)
                        except Exception as e:
                            if verbose:
                                print(f"      ⚠️  Layer {layer} classifier error for token {token_idx}: {e}")
                            token_scores.append(0.5)
                    else:
                        token_scores.append(0.5)

        except Exception as e:
            if verbose:
                print(f"      ⚠️  Error during layer {layer} token scoring: {e}")
            # Fallback: assign neutral scores
            response_tokens = model.tokenizer(response, return_tensors="pt")["input_ids"][0]
            token_scores = [0.5] * len(response_tokens)

        # Classify overall response for this layer
        if token_scores:
            aggregated_score = aggregate_token_scores(token_scores, token_aggregation)
            classification = "HALLUCINATION" if aggregated_score > threshold else "TRUTHFUL"
        else:
            aggregated_score = 0.5
            classification = "UNKNOWN"

        layer_results[layer] = {
            "token_scores": token_scores,
            "classification": classification,
            "aggregated_score": aggregated_score,
        }

        if verbose:
            print(f"      Layer {layer}: {classification} (score: {aggregated_score:.3f})")

    return response, layer_results


def generate_with_multi_layer_classification_and_handling(
    model,
    prompt,
    layers,
    max_new_tokens,
    steering_methods,
    token_aggregation="average",
    threshold=0.6,
    verbose=False,
    detection_handler=None,
):
    """
    Generate text with multi-layer classification and optional detection handling.

    Args:
        model: Model object
        prompt: Input prompt
        layers: List of layer indices for activation extraction
        max_new_tokens: Maximum tokens to generate
        steering_methods: Dict mapping layer indices to trained steering methods
        token_aggregation: How to aggregate token scores
        threshold: Classification threshold
        verbose: Whether to print debug info
        detection_handler: Optional DetectionHandler for handling detected issues

    Returns:
        Tuple of (final_response_text, layer_results_dict, was_handled)
    """
    from .core.detection_handling import DetectionAction

    # Generate initial response with multi-layer classification
    original_response, layer_results = generate_with_multi_layer_classification(
        model, prompt, layers, max_new_tokens, steering_methods, token_aggregation, threshold, verbose
    )

    # If no handler provided, return as-is
    if not detection_handler:
        return original_response, layer_results, False

    # Determine if content should be handled based on ANY layer detecting issues
    # Use the first layer for primary detection decision (could be made configurable)
    primary_layer = layers[0]
    is_problematic = layer_results[primary_layer]["classification"] == "HALLUCINATION"

    if not is_problematic:
        return original_response, layer_results, False

    # Content is problematic, apply handling
    if verbose:
        print(
            f"      🚨 Detected problematic content (layer {primary_layer}), applying {detection_handler.action.value} handling..."
        )

    # Calculate confidence score for the detection using primary layer
    confidence_score = layer_results[primary_layer]["aggregated_score"]

    # Create regeneration function if needed
    regenerate_function = None
    if detection_handler.action == DetectionAction.REGENERATE_UNTIL_SAFE:

        def regenerate():
            # Generate a new response
            new_response, new_layer_results = generate_with_multi_layer_classification(
                model, prompt, layers, max_new_tokens, steering_methods, token_aggregation, threshold, verbose
            )
            # Only return if primary layer is not problematic
            if new_layer_results[primary_layer]["classification"] != "HALLUCINATION":
                return new_response
            # Still problematic, raise exception to trigger retry
            raise Exception(
                f"Generated response still classified as {new_layer_results[primary_layer]['classification']}"
            )

        regenerate_function = regenerate

    # Handle the detection
    final_response = detection_handler.handle_detection(
        original_response=original_response,
        detection_type="hallucination",
        confidence_score=confidence_score,
        original_prompt=prompt,
        regenerate_function=regenerate_function,
    )

    # If regeneration was successful, get new results
    if (
        detection_handler.action == DetectionAction.REGENERATE_UNTIL_SAFE
        and final_response != original_response
        and not final_response.startswith("I apologize")
    ):
        # Re-classify the final response across all layers
        final_response_clean, final_layer_results = generate_with_multi_layer_classification(
            model, prompt, layers, max_new_tokens, steering_methods, token_aggregation, threshold, verbose=False
        )

        if final_response_clean.strip() == final_response.strip():
            return final_response, final_layer_results, True

    # For placeholder or pass-through, return original results but indicate handling occurred
    return final_response, layer_results, True
