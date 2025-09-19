"""
JSON loader for converting seed pair JSON files to ContrastivePairSet.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.response import PositiveResponse, NegativeResponse

logger = logging.getLogger(__name__)


class SeedPairLoader:
    """Loads JSON files from wisent-backend seed pairs format into ContrastivePairSet."""

    def __init__(self):
        pass

    def load_json_file(self, json_path: str) -> ContrastivePairSet:
        """
        Load a single JSON file and convert to ContrastivePairSet.

        Args:
            json_path: Path to the JSON file

        Returns:
            ContrastivePairSet with loaded pairs
        """
        json_path = Path(json_path)
        trait_name = json_path.stem  # Get filename without extension

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = []

        # Handle the JSON structure from wisent-backend
        if 'pairs' in data:
            for pair_data in data['pairs']:
                # Extract question (prompt), positive, and negative responses
                question = pair_data.get('question', '')
                positive = pair_data.get('positive', '')
                negative = pair_data.get('negative', '')

                if question and positive and negative:
                    # Create proper Response objects
                    pos_response = PositiveResponse(text=positive)
                    neg_response = NegativeResponse(text=negative)

                    contrastive_pair = ContrastivePair(
                        prompt=question,
                        positive_response=pos_response,
                        negative_response=neg_response,
                        label=trait_name,
                        trait_description=f"Trait: {trait_name}"
                    )
                    pairs.append(contrastive_pair)

        logger.info(f"Loaded {len(pairs)} pairs from {json_path}")

        return ContrastivePairSet(name=trait_name, pairs=pairs, task_type="seed_pairs")

    def load_all_json_files(self, json_dir: str) -> Dict[str, ContrastivePairSet]:
        """
        Load all JSON files from a directory.

        Args:
            json_dir: Directory containing JSON files

        Returns:
            Dictionary mapping trait names to ContrastivePairSet objects
        """
        json_dir = Path(json_dir)
        json_files = list(json_dir.glob("*.json"))

        logger.info(f"Found {len(json_files)} JSON files in {json_dir}")

        pair_sets = {}

        for json_file in json_files:
            try:
                pair_set = self.load_json_file(json_file)
                trait_name = json_file.stem
                pair_sets[trait_name] = pair_set
                logger.info(f"Successfully loaded {trait_name} with {len(pair_set.pairs)} pairs")
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")
                continue

        logger.info(f"Successfully loaded {len(pair_sets)} trait datasets")
        return pair_sets

    def load_multiple_choice_json(self, json_path: str, model) -> ContrastivePairSet:
        """
        Load JSON and convert to multiple-choice format using ContrastivePairSet method.

        Args:
            json_path: Path to the JSON file
            model: Model object with formatting capabilities

        Returns:
            ContrastivePairSet with multiple-choice formatted pairs
        """
        json_path = Path(json_path)
        trait_name = json_path.stem

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create empty ContrastivePairSet
        pair_set = ContrastivePairSet(name=f"{trait_name}_mc", task_type="multiple_choice")

        # Convert pairs to questions format
        questions = []
        if 'pairs' in data:
            for pair_data in data['pairs']:
                question = pair_data.get('question', '')
                positive = pair_data.get('positive', '')
                negative = pair_data.get('negative', '')

                if question and positive and negative:
                    questions.append({
                        "question": question,
                        "choice_a": positive,  # Helpful choice
                        "choice_b": negative   # Harmful choice
                    })

        # Use the existing create_multiple_choice_questions method
        pair_set.create_multiple_choice_questions(questions, model)

        logger.info(f"Loaded {len(pair_set.pairs)} MC pairs from {json_path}")
        return pair_set


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    loader = SeedPairLoader()

    # Test loading a single file
    test_file = "/home/bc/Desktop/Documents/wisent-guard/tests/controls_from_seed_pairs/seed_pairs/helpful.json"
    pair_set = loader.load_json_file(test_file)
    print(f"Loaded {pair_set.name} with {len(pair_set.pairs)} pairs")

    # Show first pair as example
    if pair_set.pairs:
        first_pair = pair_set.pairs[0]
        print(f"Example pair:")
        print(f"  Prompt: {first_pair.prompt}")
        print(f"  Positive: {first_pair.positive_response.text}")
        print(f"  Negative: {first_pair.negative_response.text}")