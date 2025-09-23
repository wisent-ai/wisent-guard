from lm_eval.tasks import get_task_dict

# Load PROST task
tasks = get_task_dict(["prost"])
prost_task = tasks["prost"]

# Try different data splits
docs = None
split_used = None

# Try training split first
try:
    if hasattr(prost_task, 'training_docs') and prost_task.training_docs() is not None:
        docs = list(prost_task.training_docs())
        split_used = "training"
except:
    pass

# Try test split if training doesn't work
if docs is None:
    try:
        if hasattr(prost_task, 'test_docs') and prost_task.test_docs() is not None:
            docs = list(prost_task.test_docs())
            split_used = "test"
    except:
        pass

# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(prost_task, 'validation_docs') and prost_task.validation_docs() is not None:
            docs = list(prost_task.validation_docs())
            split_used = "validation"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (training, test, validation)")
    print(f"Available methods: {[method for method in dir(prost_task) if 'docs' in method]}")
    exit(1)

print(f"PROST dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF PROST DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
for key in docs[0].keys():
    print(f"{key} type: {type(docs[0].get(key))}")

# Display first few examples
for i, doc in enumerate(docs[:5]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")

    for key, value in doc.items():
        print(f"{key}: {value} (type: {type(value)})")