"""
Math500 nonsense generator.
"""

import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add wisent-guard to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wisent_guard.core.tasks.math500_task import Math500Task
from base_nosense import BaseNosenseGenerator


class Math500NosenseGenerator(BaseNosenseGenerator):
    """Generate nonsensical Math500 data."""

    def __init__(self):
        """Initialize with Math500Task."""
        original_task = Math500Task()
        super().__init__(original_task)

    def make_nonsense(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Math500 data to nonsense."""
        nonsense_data = []

        for item in data:
            nonsense_item = {}
            # Generate random answer once per item
            new_answer = str(random.randint(1, 1000))

            # Copy all fields but make them nonsensical
            for key, value in item.items():
                if key == 'problem':
                    # Make problem text nonsensical only
                    nonsense_item[key] = self.generate_random_words(value)

                elif key == 'answer':
                    # Use random answer
                    nonsense_item[key] = new_answer

                elif key == 'solution':
                    # Make solution nonsensical
                    if isinstance(value, str):
                        nonsense_item[key] = self.generate_random_words(value)
                    else:
                        nonsense_item[key] = value

                elif key == 'type':
                    # Keep type as original
                    nonsense_item[key] = value

                elif isinstance(value, str):
                    # Make other string fields nonsensical
                    nonsense_item[key] = self.generate_random_words(value)

                else:
                    # Keep non-string values as-is
                    nonsense_item[key] = value

            nonsense_data.append(nonsense_item)

        return nonsense_data


def test_math500_nosense():
    """Test Math500 nonsense generation."""
    print("Testing Math500 Nonsense Generation")
    print("=" * 40)

    generator = Math500NosenseGenerator()

    # Load a small sample of original data
    original_data = generator.load_original_data(limit=3)
    print(f"Loaded {len(original_data)} original items")

    # Generate nonsense version
    nonsense_data = generator.make_nonsense(original_data)
    print(f"Generated {len(nonsense_data)} nonsense items")

    # Show comparison
    for i, (orig, nonsense) in enumerate(zip(original_data, nonsense_data)):
        print(f"\n--- Item {i+1} ---")
        print(f"Original problem: {orig.get('problem', 'N/A')[:100]}...")
        print(f"Original answer: {orig.get('answer', 'N/A')}")
        print(f"Nonsense problem: {nonsense.get('problem', 'N/A')[:100]}...")
        print(f"Nonsense answer: {nonsense.get('answer', 'N/A')}")


if __name__ == "__main__":
    test_math500_nosense()