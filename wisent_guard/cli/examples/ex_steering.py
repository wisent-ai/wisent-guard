# Example usage of EvaluatorRotator with steering methods. Now we can train models using different steering strategies. Now
# we only support CAA method, but more will come soon.

from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator
import time

#1 Create a rotator, it will auto-discover available steering methods in the specified location
rot = SteeringMethodRotator()

#2 List available steering methods
methods = SteeringMethodRotator.list_methods()

print("Available steering methods:")
for m in methods:
    print(f"- {m['name']}: {m.get('description', '')} (class: {m['class']})")
    time.sleep(1)

# #3 Get a specific steering method by name
method_name = "caa"

rot.use(method_name)
print(f"Using steering method: {method_name}")

#4 Train the selected steering method on a ContrastivePairSet.
# 4.1 First, create a dummy ContrastivePairSet with radnom activations
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair
from wisent_guard.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
from wisent_guard.core.activations.core.atoms import LayerActivations
import numpy as np

# Create dummy data
pairs = []
for i in range(10):
    pos_resp = PositiveResponse(
        model_response=f"This is a positive response_{i}.",
        layers_activations=LayerActivations(
            {
                "layer1": np.random.randn(10),
                "layer2": np.random.randn(5),
            }
        ),
        label="harmless",
    )
    neg_resp = NegativeResponse(
        model_response=f"This is a negative response_{i}.",
        layers_activations=LayerActivations(
            {
                "layer1": np.random.randn(10),
                "layer2": np.random.randn(5),
            }
        ),
        label="toxic",
    )
    pair = ContrastivePair(prompt=f"Some prompt_{i}", positive_response=pos_resp, negative_response=neg_resp)
    pairs.append(pair)

cps = ContrastivePairSet("dummy_data", pairs=pairs)

# 4.2 Now train the steering method on the ContrastivePairSet
res = rot.train(cps)

print("Training completed.")

#5 The results should be LayerActivations with steering vectors. We should see one vector per layer.
for layer, activations in res.items():
    print(f"Layer: {layer}, Steering Vector: {activations}")