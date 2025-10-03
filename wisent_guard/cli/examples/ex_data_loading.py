# Example usage of DataLoaderRotator. Now we can load tasks from lm-eval-harness (only supported one task, some can have subtasks) or custom QA datasets.
from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
import time

#1 Create a rotator, it will auto-discover available data loaders in the specified location
rot = DataLoaderRotator()

#2 List available data loaders
loaders = DataLoaderRotator.list_loaders()

print("Available data loaders:")
for ld in loaders:
    print(f"- {ld['name']}: {ld.get('description', '')} (class: {ld['class']})")

#3 Use a specific data loader by name (e.g., "lm_eval")
lm_loader = rot.use("lm_eval")

#4 Load data for a specific task (e.g., "hellaswag") with optional limits
# note: task extractor must be written to support given task, see: wisent_guard/core/contrastive_pairs/lm_eval_pairs/lm_task_extractors.py
res = rot.load(task="winogrande", limit=10)

#5 Inspect the loaded data
print(res)

print("\nSample training pairs:")
for pair in res['train_qa_pairs'].pairs:
    print(f"Q: {pair.prompt} \n A+: {pair.positive_response.model_response} | A-: {pair.negative_response.model_response}")
    print("\n")
    time.sleep(1)  

# --------------------------------- Custom QA dataset example ---------------------------------

custom_loader = rot.use("custom")  
absolute_path = "./wisent_guard/cli/cli_examples/custom_dataset.json"
custom_res = rot.load(path=absolute_path, limit=5)

print("\nCustom dataset sample training pairs:")
for pair in custom_res['train_qa_pairs'].pairs:
    print(f"Q: {pair.prompt} \n A+: {pair.positive_response.model_response} | A-: {pair.negative_response.model_response}")
    print("\n")
    time.sleep(1)  
