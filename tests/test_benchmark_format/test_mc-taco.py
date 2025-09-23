from lm_eval.tasks import get_task_dict

# Load MC-TACO task
tasks = get_task_dict(["mc_taco"])
mc_taco_task = tasks["mc_taco"]

# Try different data splits
docs = None
split_used = None

# Try training split first
try:
    if hasattr(mc_taco_task, 'training_docs') and mc_taco_task.training_docs() is not None:
        docs = list(mc_taco_task.training_docs())
        split_used = "training"
except:
    pass

# Try test split if training doesn't work
if docs is None:
    try:
        if hasattr(mc_taco_task, 'test_docs') and mc_taco_task.test_docs() is not None:
            docs = list(mc_taco_task.test_docs())
            split_used = "test"
    except:
        pass

# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(mc_taco_task, 'validation_docs') and mc_taco_task.validation_docs() is not None:
            docs = list(mc_taco_task.validation_docs())
            split_used = "validation"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (training, test, validation)")
    print(f"Available methods: {[method for method in dir(mc_taco_task) if 'docs' in method]}")
    exit(1)

print(f"MC-TACO dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF MC-TACO DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
for key in docs[0].keys():
    print(f"{key} type: {type(docs[0].get(key))}")

# Display first few examples
for i, doc in enumerate(docs[:8]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")

    for key, value in doc.items():
        print(f"{key}: {value} (type: {type(value)})")