"""
Vector operations for contrastive learning in wisent-guard.
Clean implementation using enhanced core primitives.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import Activations, ContrastivePairSet, Layer, Model

logger = logging.getLogger(__name__)


class ContrastiveVectors:
    """
    Clean vector operations using enhanced core primitives.
    """

    def __init__(self, model_name: str, layers: List[int], device: Optional[str] = None, save_dir: str = "./vectors"):
        """
        Initialize contrastive vectors.

        Args:
            model_name: Language model name or path
            layers: List of layer indices
            device: Target device
            save_dir: Directory to save vectors
        """
        self.model = Model(name=model_name, device=device)
        self.layers = [Layer(index=idx, type="transformer") for idx in layers]
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for computed vectors
        self.vectors = {}
        self.pair_sets = {}

        logger.info(f"Initialized ContrastiveVectors for layers {layers}")

    def create_pair_set(self, name: str, harmful_texts: List[str], harmless_texts: List[str]) -> ContrastivePairSet:
        """
        Create a contrastive pair set.

        Args:
            name: Name for the pair set
            harmful_texts: List of harmful examples
            harmless_texts: List of harmless examples

        Returns:
            ContrastivePairSet instance
        """
        # Create phrase pairs
        phrase_pairs = []
        min_len = min(len(harmful_texts), len(harmless_texts))

        for i in range(min_len):
            phrase_pairs.append({"harmful": harmful_texts[i], "harmless": harmless_texts[i]})

        pair_set = ContrastivePairSet.from_phrase_pairs(
            name=name, phrase_pairs=phrase_pairs, task_type="vector_computation"
        )

        self.pair_sets[name] = pair_set
        logger.info(f"Created pair set '{name}' with {len(phrase_pairs)} pairs")
        return pair_set

    def compute_vectors(self, pair_set_name: str, layer_idx: Optional[int] = None) -> Dict[int, Any]:
        """
        Compute contrastive vectors for a pair set.

        Args:
            pair_set_name: Name of the pair set
            layer_idx: Specific layer (if None, computes for all layers)

        Returns:
            Dictionary of computed vectors by layer
        """
        if pair_set_name not in self.pair_sets:
            raise ValueError(f"Pair set '{pair_set_name}' not found")

        pair_set = self.pair_sets[pair_set_name]
        computed_vectors = {}

        layers_to_compute = [layer_idx] if layer_idx is not None else [l.index for l in self.layers]

        for layer_index in layers_to_compute:
            layer_obj = Layer(index=layer_index, type="transformer")

            # Extract activations
            pair_set.extract_activations_with_model(self.model, layer_obj)

            # Compute contrastive vector
            vector = pair_set.compute_contrastive_vector(layer_obj)
            if vector is not None:
                computed_vectors[layer_index] = vector

                # Store in vectors dict
                if pair_set_name not in self.vectors:
                    self.vectors[pair_set_name] = {}
                self.vectors[pair_set_name][layer_index] = vector

        logger.info(f"Computed vectors for '{pair_set_name}' on layers {list(computed_vectors.keys())}")
        return computed_vectors

    def get_similarity(self, text: str, category: str, layer_idx: Optional[int] = None) -> float:
        """
        Get similarity between text and stored vectors.

        Args:
            text: Text to compare
            category: Vector category
            layer_idx: Layer index (uses first available if None)

        Returns:
            Similarity score
        """
        if category not in self.vectors:
            logger.warning(f"Category '{category}' not found")
            return 0.0

        # Use specified layer or first available
        if layer_idx is None:
            available_layers = list(self.vectors[category].keys())
            if not available_layers:
                return 0.0
            layer_idx = available_layers[0]

        if layer_idx not in self.vectors[category]:
            logger.warning(f"Layer {layer_idx} not found for category '{category}'")
            return 0.0

        # Extract activations for the text
        _, activations = self.model.generate(text, layer_idx, max_new_tokens=1)

        # Get stored vector
        stored_vector = self.vectors[category][layer_idx]

        # Compare using Activations similarity methods
        # Create Activations object for the extracted activations
        layer_obj = Layer(index=layer_idx, type="transformer")
        activations_obj = Activations(tensor=activations, layer=layer_obj)

        if hasattr(stored_vector, "cosine_similarity"):
            return stored_vector.cosine_similarity(activations_obj)
        # If stored vector is a tensor, create Activations object
        vector_activations = Activations(tensor=stored_vector, layer=layer_obj)
        return vector_activations.cosine_similarity(activations_obj)

    def evaluate_vectors(self, pair_set_name: str, layer_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate computed vectors.

        Args:
            pair_set_name: Name of the pair set
            layer_idx: Layer index

        Returns:
            Evaluation results
        """
        if pair_set_name not in self.pair_sets:
            raise ValueError(f"Pair set '{pair_set_name}' not found")

        pair_set = self.pair_sets[pair_set_name]

        if layer_idx is None:
            layer_idx = self.layers[0].index

        layer_obj = Layer(index=layer_idx, type="transformer")

        # Use ContrastivePairSet's evaluation method
        return pair_set.evaluate_with_vectors(None, layer_obj)

    def save_vectors(self, categories: Optional[List[str]] = None) -> None:
        """
        Save vectors to disk.

        Args:
            categories: Categories to save (saves all if None)
        """
        import torch

        categories_to_save = categories or list(self.vectors.keys())

        for category in categories_to_save:
            if category in self.vectors:
                save_path = self.save_dir / f"{category}_vectors.pt"
                torch.save(self.vectors[category], save_path)
                logger.info(f"Saved vectors for '{category}' to {save_path}")

    def load_vectors(self, categories: Optional[List[str]] = None, layers: Optional[List[int]] = None) -> bool:
        """
        Load vectors from disk.

        Args:
            categories: Categories to load (loads all available if None)
            layers: Layers to load (loads all if None)

        Returns:
            Success status
        """
        import torch

        try:
            # Find available vector files
            vector_files = list(self.save_dir.glob("*_vectors.pt"))

            if not vector_files:
                logger.warning("No vector files found")
                return False

            for vector_file in vector_files:
                category = vector_file.stem.replace("_vectors", "")

                if categories and category not in categories:
                    continue

                loaded_vectors = torch.load(vector_file, map_location=self.model.device)

                # Filter by layers if specified
                if layers:
                    loaded_vectors = {k: v for k, v in loaded_vectors.items() if k in layers}

                self.vectors[category] = loaded_vectors
                logger.info(f"Loaded vectors for '{category}' from {vector_file}")

            return True

        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
            return False

    def get_available_categories(self) -> List[str]:
        """Get available vector categories."""
        return list(self.vectors.keys())

    def clear_vectors(self, categories: Optional[List[str]] = None) -> None:
        """
        Clear vectors from memory.

        Args:
            categories: Categories to clear (clears all if None)
        """
        if categories:
            for category in categories:
                if category in self.vectors:
                    del self.vectors[category]
                    logger.info(f"Cleared vectors for '{category}'")
        else:
            self.vectors.clear()
            logger.info("Cleared all vectors")


def create_vectors(
    model_name: str, layers: List[int], device: Optional[str] = None, save_dir: str = "./vectors"
) -> ContrastiveVectors:
    """
    Create a ContrastiveVectors instance.

    Args:
        model_name: Language model name
        layers: List of layer indices
        device: Target device
        save_dir: Directory to save vectors

    Returns:
        ContrastiveVectors instance
    """
    return ContrastiveVectors(model_name=model_name, layers=layers, device=device, save_dir=save_dir)
