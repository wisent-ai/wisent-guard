# Example how to use WisentModel for inference with steeiring vectors
from wisent_guard.core.models.wisent_model import WisentModel

# 1. Create dummy steering vectors for testing
# In practice, you would load or compute these based on your needs

import torch
steering_vectors = {
    "12": torch.randn(2048),  # Example steering vector for layer 10
    "14": torch.randn(2048),  # Example steering vector for layer 11
}

# 2. Initialize the WisentModel with your base model and steering vectors

PATH_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Example model name
model = WisentModel(
    model_name=PATH_MODEL,  # Example model name
    layers=steering_vectors,
    device="cuda"  # or "cpu"
)

# 2.1 set_steering_from_raw
model.set_steering_from_raw(steering_vectors, scale=-2.5, normalize=False)

# 3. Use the model for inference
prompt = "What is the capital of France?"
response = model.generate(
    inputs=[[{"role": "user", "content": prompt}]],
    max_new_tokens=50,
    temperature=0.01,
    top_p=0.9,
    use_steering=True  
)
print("Response:", response[0])

print("\n\nNow testing without steering...\n")

# 4. without steering
with model.detached():
    response_no_steering = model.generate(
        inputs=[[{"role": "user", "content": prompt}]],
        max_new_tokens=50,
        temperature=0.01,
        top_p=0.9,
        use_steering=False  
    )
print("Response without steering:", response_no_steering[0])

