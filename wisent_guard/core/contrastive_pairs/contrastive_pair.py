from typing import Optional

import numpy as np


class ContrastivePair:
    def __init__(
        self,
        prompt: str,
        positive_response: str,
        negative_response: str,
        positive_activations: Optional[np.ndarray] = None,
        negative_activations: Optional[np.ndarray] = None,
        label: Optional[str] = None,
        trait_description: Optional[str] = None,
    ):
        self.prompt = prompt
        self.positive_response = positive_response
        self.negative_response = negative_response
        self.positive_activations = positive_activations
        self.negative_activations = negative_activations
        self.label = label
        self.trait_description = trait_description
