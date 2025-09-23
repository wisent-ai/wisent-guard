from lm_eval.tasks import get_task_dict
import ast

# Load HellaSwag task
tasks = get_task_dict(["hellaswag"])
hellaswag_task = tasks["hellaswag"]

# Load training data
docs = list(hellaswag_task.training_docs())
split_used = "training"

# Check if we have any data
if not docs:
    print("❌ No data found in training split")
    print(f"Available methods: {[method for method in dir(hellaswag_task) if 'docs' in method]}")
    exit(1)

print(f"HellaSwag dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF HELLASWAG DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
for key in docs[0].keys():
    print(f"{key} type: {type(docs[0].get(key))}")

# Display first few examples
for i, doc in enumerate(docs[:3]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")

    for key, value in doc.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value} (type: {type(value)})")
        