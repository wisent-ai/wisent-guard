class ContrastivePair:
    def __init__(self, prompt, positive_response, negative_response, positive_activations=None, negative_activations=None, label=None):
        self.prompt = prompt
        self.positive_response = positive_response
        self.negative_response = negative_response
        self.positive_activations = positive_activations
        self.negative_activations = negative_activations
        self.label = label 