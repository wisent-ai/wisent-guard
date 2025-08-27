from typing import Any, Dict, List, Optional

import torch

from wisent_guard.core.activations.core import ActivationAggregationStrategy, Activations
from wisent_guard.core.layer import Layer


class ActivationMonitor:
    """
    Monitor for tracking and analyzing model activations.
    Integrated into the activations primitive.
    """

    def __init__(self):
        """Initialize activation monitor."""
        self.current_activations = {}
        self.activation_history = []

    def store_activations(self, activations: Dict[int, Activations], text: Optional[str] = None) -> None:
        """Store activations with optional text context."""
        self.current_activations = activations
        self.activation_history.append({"text": text, "activations": activations})

    def analyze_activations(self, activations: Optional[Dict[int, Activations]] = None) -> Dict[int, Dict[str, Any]]:
        """Analyze activations for patterns and statistics."""
        if activations is None:
            activations = self.current_activations

        analysis = {}
        for layer_idx, activation in activations.items():
            stats = activation.get_statistics()
            features = activation.extract_features_for_classifier()

            analysis[layer_idx] = {
                "statistics": stats,
                "feature_vector": features,
                "tensor_shape": activation.tensor.shape,
                "device": str(activation.tensor.device),
            }

        return analysis

    def compare_with_baseline(
        self, baseline_activations: Dict[int, Activations], current_activations: Optional[Dict[int, Activations]] = None
    ) -> Dict[int, Dict[str, float]]:
        """Compare current activations with baseline."""
        if current_activations is None:
            current_activations = self.current_activations

        comparisons = {}
        for layer_idx in current_activations:
            if layer_idx in baseline_activations:
                current = current_activations[layer_idx]
                baseline = baseline_activations[layer_idx]

                comparisons[layer_idx] = {
                    "cosine_similarity": current.cosine_similarity(baseline),
                    "dot_product": current.dot_product_similarity(baseline),
                    "euclidean_distance": current.euclidean_distance(baseline),
                }

        return comparisons

    def detect_anomalies(
        self, threshold: float = 0.8, activations: Optional[Dict[int, Activations]] = None
    ) -> Dict[int, bool]:
        """Detect anomalies in activations based on historical patterns."""
        if activations is None:
            activations = self.current_activations

        if len(self.activation_history) < 2:
            return dict.fromkeys(activations, False)

        anomalies = {}
        for layer_idx in activations:
            historical_activations = []
            for entry in self.activation_history[:-1]:
                if layer_idx in entry["activations"]:
                    historical_activations.append(entry["activations"][layer_idx])

            if not historical_activations:
                anomalies[layer_idx] = False
                continue

            current = activations[layer_idx]
            similarities = [current.cosine_similarity(hist) for hist in historical_activations]
            avg_similarity = sum(similarities) / len(similarities)
            anomalies[layer_idx] = avg_similarity < threshold

        return anomalies

    def clear_history(self) -> None:
        """Clear activation history."""
        self.activation_history.clear()

    def save_activations(self, filepath: str) -> None:
        """Save current activations to file."""
        import torch

        if not self.current_activations:
            raise ValueError("No activations to save")

        save_data = {}
        for layer_idx, activation in self.current_activations.items():
            save_data[layer_idx] = activation.tensor

        torch.save(save_data, filepath)

    def load_activations(self, filepath: str) -> Dict[int, Activations]:
        """Load activations from file."""

        loaded_data = torch.load(filepath)
        activations = {}
        for layer_idx, tensor in loaded_data.items():
            layer_obj = Layer(index=layer_idx, type="transformer")
            activations[layer_idx] = Activations(tensor=tensor, layer=layer_obj)

        self.current_activations = activations
        return activations


class TestActivationCache:
    """
    Cache for test-time activations with associated questions and responses.
    Allows saving activations from one layer and reusing them for different layer classifiers.
    """

    def __init__(self):
        self.activations = []  # List of dicts with 'question', 'response', 'activations', 'layer'

    def add_activation(self, question: str, response: str, activations: "Activations", layer: int):
        """Add a test activation to the cache."""
        self.activations.append(
            {"question": question, "response": response, "activations": activations, "layer": layer}
        )

    def save_to_file(self, filepath: str) -> None:
        """Save cached activations to file."""

        import torch

        if not self.activations:
            raise ValueError("No activations to save")

        # Prepare data for saving
        save_data = {
            "metadata": {
                "num_samples": len(self.activations),
                "layers": list(set(item["layer"] for item in self.activations)),
                "created_at": __import__("datetime").datetime.now().isoformat(),
            },
            "activations": [],
        }

        for i, item in enumerate(self.activations):
            activation_data = {
                "index": i,
                "question": item["question"],
                "response": item["response"],
                "layer": item["layer"],
                "activation_tensor": item["activations"].tensor,
                "activation_shape": list(item["activations"].tensor.shape),
                "aggregation_method": item["activations"].aggregation_method.value
                if hasattr(item["activations"].aggregation_method, "value")
                else str(item["activations"].aggregation_method),
            }
            save_data["activations"].append(activation_data)

        # Save using torch
        torch.save(save_data, filepath)
        print(f"💾 Saved {len(self.activations)} test activations to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> "TestActivationCache":
        """Load cached activations from file."""
        import torch

        from wisent_guard.core.layer import Layer

        loaded_data = torch.load(filepath, map_location="cpu")
        cache = cls()

        metadata = loaded_data.get("metadata", {})
        print(f"📁 Loading {metadata.get('num_samples', 0)} test activations from {filepath}")

        for item in loaded_data["activations"]:
            # Reconstruct Activations object
            layer_obj = Layer(index=item["layer"], type="transformer")

            # Reconstruct aggregation method

            agg_method = ActivationAggregationStrategy.LAST_TOKEN  # default
            try:
                if item.get("aggregation_method"):
                    agg_method = ActivationAggregationStrategy(item["aggregation_method"])
            except:
                pass

            activations = Activations(
                tensor=item["activation_tensor"], layer=layer_obj, aggregation_strategy=agg_method
            )

            cache.add_activation(
                question=item["question"], response=item["response"], activations=activations, layer=item["layer"]
            )

        return cache

    def get_activations_for_layer(self, target_layer: int) -> List[Dict]:
        """Get all activations for a specific layer."""
        return [item for item in self.activations if item["layer"] == target_layer]

    def __len__(self):
        return len(self.activations)

    def __repr__(self):
        layers = list(set(item["layer"] for item in self.activations))
        return f"TestActivationCache({len(self.activations)} samples, layers: {layers})"
