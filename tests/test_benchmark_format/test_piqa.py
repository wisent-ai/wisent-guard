from lm_eval.tasks import get_task_dict

# Load PIQA task
tasks = get_task_dict(["piqa"])
piqa_task = tasks["piqa"]

# Try different data splits
docs = None
split_used = None

# Try test split first
try:
    if hasattr(piqa_task, 'test_docs') and piqa_task.test_docs() is not None:
        docs = list(piqa_task.test_docs())
        split_used = "test"
except:
    pass

# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(piqa_task, 'validation_docs') and piqa_task.validation_docs() is not None:
            docs = list(piqa_task.validation_docs())
            split_used = "validation"
    except:
        pass

# Try train split if validation doesn't work
if docs is None:
    try:
        if hasattr(piqa_task, 'training_docs') and piqa_task.training_docs() is not None:
            docs = list(piqa_task.training_docs())
            split_used = "training"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (test, validation, training)")
    print(f"Available methods: {[method for method in dir(piqa_task) if 'docs' in method]}")
    exit(1)

print(f"PIQA dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF PIQA DATA:")
print("="*80)

# Check data types for key fields
print(f"goal type: {type(docs[0].get('goal'))}")
print(f"sol1 type: {type(docs[0].get('sol1'))}")
print(f"sol2 type: {type(docs[0].get('sol2'))}")
print(f"label type: {type(docs[0].get('label'))}")

# Display first few examples
for i, doc in enumerate(docs[:5]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")
    print(f"Goal: {doc.get('goal', 'N/A')}")
    print(f"Solution 1: {doc.get('sol1', 'N/A')}")
    print(f"Solution 2: {doc.get('sol2', 'N/A')}")
    print(f"Label: {doc.get('label', 'N/A')}")
    print(f"Correct Solution: {doc.get('sol1', 'N/A') if doc.get('label', 0) == 0 else doc.get('sol2', 'N/A')}")
    print(f"Incorrect Solution: {doc.get('sol2', 'N/A') if doc.get('label', 0) == 0 else doc.get('sol1', 'N/A')}")

