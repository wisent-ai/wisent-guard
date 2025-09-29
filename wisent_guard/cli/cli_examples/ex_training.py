# Example how to train steeiring vectors on custom tasks
import time
from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator

# 1. Create a rotator for data loaders, it will auto-discover available data loaders in the specified location
data_loader_rot = DataLoaderRotator()

# 2. List available data loaders
data_loaders = DataLoaderRotator.list_loaders()

print("Available data loaders:")
for dl in data_loaders:
    print(f"- {dl['name']}: {dl.get('description', '')} (class: {dl['class']})")
    time.sleep(1)

# 3. Get a specific data loader by name. We use cutom.
data_loader_name = "custom"
data_loader_rot.use(data_loader_name)
print(f"Using data loader: {data_loader_name}")

# 4. Load data from the file
absolute_path = "./wisent_guard/cli/cli_examples/custom_dataset.json"
data = data_loader_rot.load(path=absolute_path, limit=5)

print("Sample loaded training pairs:")
for pair in data['train_qa_pairs'].pairs:
    print(f"Q: {pair.prompt} \n A+: {pair.positive_response.model_response} | A-: {pair.negative_response.model_response}")
    print("\n")
    time.sleep(1)

# 5. Create a rotator for steering methods, it will auto-discover available steering methods in the specified location
steering_rot = SteeringMethodRotator()

# 6. List available steering methods
steering_methods = SteeringMethodRotator.list_methods()

print("Available steering methods:")
for sm in steering_methods:
    print(f"- {sm['name']}: {sm.get('description', '')} (class: {sm['class']})")
    time.sleep(1)

# 7. Get a specific steering method by name. We use "caa"
steering_method_name = "caa"
steering_rot.use(steering_method_name)

caa_method = steering_rot._method # Ugly access to the method instance, fix later
print(f"Using steering method: {steering_method_name}")

# 8. Create a WisentModel instance (replace with your actual model initialization)
PATH_TO_YOUR_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your model path or name
model = WisentModel(model_name=PATH_TO_YOUR_MODEL, layers={}, device="cuda")

# 9. Create a WisentSteeringTrainer instance
training_data = data['train_qa_pairs']
trainer = WisentSteeringTrainer(model=model, pair_set=training_data, steering_method=caa_method)

# 10. Train the steering method on the loaded data
training_result = trainer.run(
    layers_spec="10-12",  # Specify which layers to use, e.g., "10-12"
    aggregation="continuation_token",  # How to aggregate activations
    return_full_sequence=False,  # Whether to return full sequence activations
    normalize_layers=True,  # Whether to normalize activations per layer
    save_dir="./steering_output"  # Directory to save the results
)
