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
    """Loads JSON file from wisent-backend seed pair format into ContrastivePairSet for multi choice question creation."""

    def __init__(self):
        pass

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

    here = Path(__file__).parent
    test_file = here / "seed_pairs/helpful.json"

    pair_set = loader.load_multiple_choice_json(test_file)
    print(f"Loaded {pair_set.name} with {len(pair_set.pairs)} pairs")

    # Show first pair as example
    if pair_set.pairs:
        first_pair = pair_set.pairs[0]
        print(f"Example pair:")
        print(f"  Prompt: {first_pair.prompt}")
        print(f"  Positive: {first_pair.positive_response.text}")
        print(f"  Negative: {first_pair.negative_response.text}")