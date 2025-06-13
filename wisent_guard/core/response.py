class Response:
    def __init__(self, text, activations=None, label=None):
        self.text = text
        self.activations = activations
        self.label = label

class PositiveResponse(Response):
    pass

class NegativeResponse(Response):
    pass 