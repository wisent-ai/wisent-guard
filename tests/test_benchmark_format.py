from lm_eval.tasks import get_task_dict

# Load task
tasks = get_task_dict(["winogrande"])
anli_task = tasks["winogrande"]

# Try different splits in order: validation, test, train
docs = []
split_used = None

# Try validation split first
try:
    docs = list(anli_task.validation_docs())
    if docs:
        split_used = "validation"
except:
    pass

# If no validation data, try test split
if not docs:
    try:
        docs = list(anli_task.test_docs())
        if docs:
            split_used = "test"
    except:
        pass

# If no test data, try train split
if not docs:
    try:
        docs = list(anli_task.training_docs())
        if docs:
            split_used = "train"
    except:
        pass

# Check if we have any data
if not docs:
    print("❌ No data found in any split (validation, test, train)")
    print(f"Available methods: {[method for method in dir(anli_task) if 'docs' in method]}")
    exit(1)

print(f"Dataset loaded: {len(docs)} samples from {split_used} split")

print("\n" + "="*80)
print("EXAMPLES OF DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
for key in docs[0].keys():
    print(f"{key} type: {type(docs[0].get(key))}")

# Display first few examples
for i, doc in enumerate(docs[:20]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")

    # Try to extract formatted question using doc_to_text if available
    if hasattr(anli_task, "doc_to_text"):
        question = anli_task.doc_to_text(doc)
        print(f"Formatted question: {question}")

    for key, value in doc.items():
            print(f"{key}: {value} (type: {type(value)})")