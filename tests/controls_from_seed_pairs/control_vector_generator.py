"""
Control vector generator that processes JSON seed pairs and creates serialized control vectors.
"""
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, List

from wisent_guard.core import Model
from wisent_guard.core.layer import Layer
from wisent_guard.core.steering_methods.caa import CAA
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from json_loader import SeedPairLoader

logger = logging.getLogger(__name__)


class ControlVectorGenerator:
    """Generates control vectors from seed pair JSON files and serializes them."""

    def __init__(self, model_name: str = "unsloth/Qwen3-4B-bnb-4bit", layer_index: int = 17, device: str = "auto"):
        """
        Initialize the control vector generator.

        Args:
            model_name: HuggingFace model name to use for activation extraction
            layer_index: Layer index to extract activations from
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.layer_index = layer_index

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.loader = SeedPairLoader()

    def load_model(self):
        """Load the model for activation extraction."""
        logger.info(f"Loading model: {self.model_name}")
        self.model = Model(name=self.model_name, device=self.device)
        logger.info(f"Model loaded on device: {self.device}")

    def extract_activations_for_pair_set(self, pair_set: ContrastivePairSet) -> ContrastivePairSet:
        """
        Extract activations for all pairs in a ContrastivePairSet.

        Args:
            pair_set: ContrastivePairSet to extract activations for

        Returns:
            Updated ContrastivePairSet with activations
        """
        if self.model is None:
            self.load_model()

        logger.info(f"Extracting activations for {len(pair_set.pairs)} pairs in {pair_set.name}")

        # Create Layer object
        layer = Layer(self.layer_index)

        # Use the built-in method from ContrastivePairSet
        pair_set.extract_activations_with_model(self.model, layer)

        logger.info(f"Finished extracting activations for {pair_set.name}")
        return pair_set

    def generate_control_vector(self, pair_set: ContrastivePairSet) -> Dict[str, Any]:
        """
        Generate a control vector from a ContrastivePairSet with activations.

        Args:
            pair_set: ContrastivePairSet with extracted activations

        Returns:
            Dictionary containing control vector and metadata
        """
        logger.info(f"Generating control vector for {pair_set.name}")

        # Initialize CAA steering method
        caa = CAA(device=self.device)

        # Train CAA on the contrastive pairs
        training_stats = caa.train(pair_set, self.layer_index)

        # Get the steering vector
        steering_vector = caa.get_steering_vector()

        # Prepare serializable data
        control_vector_data = {
            "trait_name": pair_set.name,
            "model_name": self.model_name,
            "layer_index": self.layer_index,
            "device": self.device,
            "steering_vector": steering_vector.cpu().numpy().tolist(),  # Convert to plain list
            "vector_shape": list(steering_vector.shape),
            "vector_norm": torch.norm(steering_vector).item(),
            "method": "CAA",
            "training_stats": training_stats,
            "num_pairs": len(pair_set.pairs),
            "metadata": {
                "task_type": pair_set.task_type,
                "generated_by": "wisent-guard control vector generator",
            }
        }

        logger.info(f"Generated control vector for {pair_set.name} (norm: {control_vector_data['vector_norm']:.4f})")
        return control_vector_data

    def save_control_vector_json(self, control_vector_data: Dict[str, Any], output_path: str):
        """
        Save control vector data to JSON file.

        Args:
            control_vector_data: Control vector data dictionary
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(control_vector_data, f, indent=2)

        logger.info(f"Saved control vector to {output_path}")

    def process_single_json_file(self, json_path: str, output_dir: str) -> bool:
        """
        Process a single JSON file and generate its control vector.

        Args:
            json_path: Path to input JSON file
            output_dir: Directory to save output control vector JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            json_path = Path(json_path)
            trait_name = json_path.stem

            logger.info(f"Processing {trait_name} from {json_path}")

            # Load the JSON file
            pair_set = self.loader.load_json_file(json_path)

            # Extract activations
            pair_set = self.extract_activations_for_pair_set(pair_set)

            # Generate control vector
            control_vector_data = self.generate_control_vector(pair_set)

            # Save to JSON
            output_path = Path(output_dir) / f"{trait_name}_control_vector.json"
            self.save_control_vector_json(control_vector_data, output_path)

            return True

        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")
            return False

    def process_single_json_file_mc(self, json_path: str, output_dir: str) -> bool:
        """
        Process a single JSON file using multiple-choice format and generate its control vector.

        Args:
            json_path: Path to input JSON file
            output_dir: Directory to save output control vector JSON

        Returns:
            True if successful, False otherwise
        """
        try:
            json_path = Path(json_path)
            trait_name = json_path.stem

            logger.info(f"Processing {trait_name} from {json_path} in multiple-choice format")

            # Load model first for multiple-choice formatting
            if self.model is None:
                self.load_model()

            # Load the JSON file in multiple-choice format
            pair_set = self.loader.load_multiple_choice_json(json_path, self.model)

            # Extract activations
            pair_set = self.extract_activations_for_pair_set(pair_set)

            # Generate control vector
            control_vector_data = self.generate_control_vector(pair_set)

            # Save to JSON
            output_path = Path(output_dir) / f"{trait_name}_mc_control_vector.json"
            self.save_control_vector_json(control_vector_data, output_path)

            return True

        except Exception as e:
            logger.error(f"Failed to process {json_path} in MC format: {e}")
            return False



if __name__ == "__main__":
    # Test the generator
    logging.basicConfig(level=logging.INFO)

    generator = ControlVectorGenerator(
        model_name="unsloth/Qwen3-4B-bnb-4bit",  # 4-bit quantized model
        layer_index=17,
        device="auto"
    )

    # Test with single file - regular format
    test_file = "/home/bc/Desktop/Documents/wisent-guard/tests/controls_from_seed_pairs/seed_pairs/helpful.json"
    output_dir = "/home/bc/Desktop/Documents/wisent-guard/tests/controls_from_seed_pairs/control_vectors"

    success = generator.process_single_json_file(test_file, output_dir)
    print(f"Regular format test: {'Success' if success else 'Failed'}")

    # Test with multiple-choice format
    success_mc = generator.process_single_json_file_mc(test_file, output_dir)
    print(f"Multiple-choice format test: {'Success' if success_mc else 'Failed'}")