class Layer:
    def __init__(self, index, type=None):
        self.index = index
        self.type = type 

    def get_activations(self, model, prompt, **kwargs):
        _, activations = model.generate(prompt, self.index, **kwargs)
        return activations 