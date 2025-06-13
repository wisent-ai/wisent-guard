class Representation:
    def __init__(self, tensor, layer, aggregation_method=None):
        self.tensor = tensor
        self.layer = layer
        self.aggregation_method = aggregation_method 