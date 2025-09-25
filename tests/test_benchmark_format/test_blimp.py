from lm_eval.tasks import get_task_dict

# Load BLIMP task
tasks = get_task_dict(["blimp"])
blimp_task = tasks["blimp"]

# Try different splits in order: train, test, validation
docs = []
split_used = None

# Try train split first
try:
    docs = list(blimp_task.training_docs())
    if docs:
        split_used = "train"
except:
    pass

# If no train data, try test split
if not docs:
    try:
        docs = list(blimp_task.test_docs())
        if docs:
            split_used = "test"
    except:
        pass

# If no test data, try validation split
if not docs:
    try:
        docs = list(blimp_task.validation_docs())
        if docs:
            split_used = "validation"
    except:
        pass

# Check if we have any data
if not docs:
    print("❌ No data found in any split (train, test, validation)")
    print(f"Available methods: {[method for method in dir(blimp_task) if 'docs' in method]}")
    exit(1)

print(f"BLIMP dataset loaded: {len(docs)} samples from {split_used} split")

# Count label distribution
label_counts = {}
for doc in docs:
    label = doc.get('label')
    if label is not None:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

print("\n" + "="*80)
print("LABEL DISTRIBUTION:")
print("="*80)
total_samples = len(docs)
for label, count in sorted(label_counts.items()):
    percentage = (count / total_samples) * 100 if total_samples > 0 else 0
    print(f"Label {label}: {count} samples ({percentage:.2f}%)")

print("\n" + "="*80)
print("EXAMPLES OF BLIMP DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
for key in docs[0].keys():
    print(f"{key} type: {type(docs[0].get(key))}")

# Display first few examples
for i, doc in enumerate(docs[:40]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")

    for key, value in doc.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value} (type: {type(value)})")