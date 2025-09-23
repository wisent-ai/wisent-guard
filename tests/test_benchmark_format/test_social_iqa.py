from lm_eval.tasks import get_task_dict

# Load Social IQA task
tasks = get_task_dict(["social_iqa"])
social_iqa_task = tasks["social_iqa"]

# Try different data splits
docs = None
split_used = None

# Try training split first
try:
    if hasattr(social_iqa_task, 'training_docs') and social_iqa_task.training_docs() is not None:
        docs = list(social_iqa_task.training_docs())
        split_used = "training"
except:
    pass

# Try test split if training doesn't work
if docs is None:
    try:
        if hasattr(social_iqa_task, 'test_docs') and social_iqa_task.test_docs() is not None:
            docs = list(social_iqa_task.test_docs())
            split_used = "test"
    except:
        pass

# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(social_iqa_task, 'validation_docs') and social_iqa_task.validation_docs() is not None:
            docs = list(social_iqa_task.validation_docs())
            split_used = "validation"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (training, test, validation)")
    print(f"Available methods: {[method for method in dir(social_iqa_task) if 'docs' in method]}")
    exit(1)

print(f"Social IQA dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF SOCIAL IQA DATA:")
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