from lm_eval.tasks import get_task_dict

# Load Arithmetic task
tasks = get_task_dict(["arithmetic_5ds"])
arithmetic_task = tasks["arithmetic_5ds"]

# Try different data splits
docs = None
split_used = None

# Try test split first
try:
    if hasattr(arithmetic_task, 'test_docs') and arithmetic_task.test_docs() is not None:
        docs = list(arithmetic_task.test_docs())
        split_used = "test"
except:
    pass


# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(arithmetic_task, 'validation_docs') and arithmetic_task.validation_docs() is not None:
            docs = list(arithmetic_task.validation_docs())
            split_used = "validation"
    except:
        pass



# Try train split if validation doesn't work
if docs is None:
    try:
        if hasattr(arithmetic_task, 'training_docs') and arithmetic_task.training_docs() is not None:
            docs = list(arithmetic_task.training_docs())
            split_used = "training"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (test, validation, training)")
    print(f"Available methods: {[method for method in dir(arithmetic_task) if 'docs' in method]}")
    exit(1)

print(f"Arithmetic dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF ARITHMETIC DATA:")
print("="*80)

# First, let's see what fields are actually available
if docs:
    print(f"First sample keys: {list(docs[0].keys())}")
    print(f"First sample: {docs[0]}")

print("\n" + "="*50)

# Display first few examples
for i, doc in enumerate(docs[:20]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")
    print(f"Context: {doc.get('context', 'N/A')}")
    print(f"Completion: {doc.get('completion', 'N/A')}")
    print(f"Split origin: {doc.get('_split_origin', 'N/A')}")
    # Try alternative field names
    print(f"Alternative fields:")
    print(f"  - question: {doc.get('question')}")
    print(f"  - answer: {doc.get('answer')}")
    print(f"  - target: {doc.get('target')}")
    print(f"  - problem: {doc.get('problem')}")
    print(f"  - solution: {doc.get('solution')}")
    if 'idx' in doc:
        print(f"Index: {doc.get('idx', 'N/A')}")

print(f"\n" + "="*80)
print("SAMPLE DATA STRUCTURE:")
print("="*80)

# Show the structure of one complete sample
if docs:
    import json
    sample = docs[0]
    print(json.dumps(sample, indent=2))

print(f"\n" + "="*80)
print("FIELD ANALYSIS:")
print("="*80)

# Analyze the data structure
if docs:
    sample = docs[0]
    print("Top-level keys:", list(sample.keys()))
    print("\nData structure explanation:")
    print("- context: The arithmetic problem/question")
    print("- completion: The expected answer/solution")
    print("- _split_origin: Source split information")

    # Show a few more samples to understand the pattern
    print(f"\n" + "="*80)
    print("MORE SAMPLE DATA STRUCTURES:")
    print("="*80)

    for i in range(min(10, len(docs))):
        print(f"\n--- SAMPLE {i+1} STRUCTURE ---")
        print(json.dumps(docs[i], indent=2))

print(f"\n" + "="*80)
print("PROBLEM TYPES ANALYSIS:")
print("="*80)

# Analyze problem types
if docs:
    problem_types = {}
    for doc in docs[:50]:  # Analyze first 50 problems
        context = doc.get('context', '')

        # Categorize by operation types
        if '+' in context:
            problem_types.setdefault('addition', []).append(context)
        elif '-' in context:
            problem_types.setdefault('subtraction', []).append(context)
        elif '*' in context or '×' in context:
            problem_types.setdefault('multiplication', []).append(context)
        elif '/' in context or '÷' in context:
            problem_types.setdefault('division', []).append(context)
        else:
            problem_types.setdefault('other', []).append(context)

    for p_type, problems in problem_types.items():
        print(f"\n{p_type.upper()} PROBLEMS ({len(problems)} examples):")
        for p in problems[:3]:  # Show first 3 examples
            print(f"  - {p}")
        if len(problems) > 3:
            print(f"  ... and {len(problems) - 3} more")

print(f"\n" + "="*80)
print("COMPLETION ANALYSIS:")
print("="*80)

# Analyze completions/answers
if docs:
    completions = [doc.get('completion', '') for doc in docs[:20]]
    print("Sample completions:")
    for i, comp in enumerate(completions):
        context = docs[i].get('context', '')
        print(f"  Problem: {context}")
        print(f"  Answer: {comp}")
        print()