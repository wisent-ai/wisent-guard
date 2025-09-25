from lm_eval.tasks import get_task_dict

# Load xcopa task
tasks = get_task_dict(["xcopa_eu"])
xcopa_task = tasks["xcopa_eu"]

# Try different splits in order: train, test, validation
docs = []
split_used = None

# Try train split first
try:
    docs = list(xcopa_task.training_docs())
    if docs:
        split_used = "train"
except:
    pass

# If no train data, try test split
if not docs:
    try:
        docs = list(xcopa_task.test_docs())
        if docs:
            split_used = "test"
    except:
        pass

# If no test data, try validation split
if not docs:
    try:
        docs = list(xcopa_task.validation_docs())
        if docs:
            split_used = "validation"
    except:
        pass

# Check if we have any data
if not docs:
    print("❌ No data found in any split (train, test, validation)")
    print(f"Available methods: {[method for method in dir(xcopa_task) if 'docs' in method]}")
    exit(1)

print(f"xcopa dataset loaded: {len(docs)} samples from {split_used} split")

print("\n" + "="*80)
print("EXAMPLES OF XCOPA DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
for key in docs[0].keys():
    print(f"{key} type: {type(docs[0].get(key))}")

# Display first few examples
for i, doc in enumerate(docs[:30]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")

    # Try to extract formatted question using doc_to_text if available
    if hasattr(xcopa_task, "doc_to_text"):
        question = xcopa_task.doc_to_text(doc)
        print(f"Formatted question: {question}")

    for key, value in doc.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value} (type: {type(value)})")