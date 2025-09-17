"""
Base class for creating nonsensical versions of benchmark data.
"""

import random
import string
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseNosenseGenerator(ABC):
    """Base class for generating nonsensical benchmark data."""

    def __init__(self, original_task):
        """
        Initialize with original task.

        Args:
            original_task: The original task object (e.g., Math500Task)
        """
        self.original_task = original_task

    def generate_random_words(self, text: str) -> str:
        """Replace words with random nonsense words."""
        words = text.split()
        nonsense_words = []

        for word in words:
            # Keep punctuation at the end
            if word and word[-1] in '.,!?;:':
                punctuation = word[-1]
                word = word[:-1]
            else:
                punctuation = ""

            # Generate random word of similar length
            if len(word) > 0:
                length = len(word)
                # Keep some words short, some longer
                length = max(3, min(length, 8))
                nonsense_word = ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
                nonsense_words.append(nonsense_word + punctuation)
            else:
                nonsense_words.append(word + punctuation)

        return ' '.join(nonsense_words)

    def extract_number(self, text: str) -> str:
        """Extract the first number from text."""
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        return numbers[0] if numbers else "42"

    def ensure_number_in_text(self, text: str, number: str) -> str:
        """Ensure the text contains the specified number."""
        if number not in text:
            # Add number at the end
            text += f" The answer is {number}."
        return text

    @abstractmethod
    def make_nonsense(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert real data to nonsense data.

        Args:
            data: Original benchmark data

        Returns:
            Nonsensical version of the data
        """
        pass

    def load_original_data(self, limit: int = None) -> List[Dict[str, Any]]:
        """Load original data from the task."""
        return self.original_task.load_data(limit=limit)

    def generate_nonsense_data(self, limit: int = None) -> List[Dict[str, Any]]:
        """Generate nonsense version of the benchmark data."""
        original_data = self.load_original_data(limit=limit)
        return self.make_nonsense(original_data)