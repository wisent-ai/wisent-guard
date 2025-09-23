from lm_eval.tasks import get_task_dict

# Load MultiRC task
tasks = get_task_dict(["multirc"])
multirc_task = tasks["multirc"]

# Additionally try to load data in the alternative way
print("Trying alternative loading method...")
try:
    # Load data directly from lm-eval without creating a Model instance
    from lm_eval.tasks import get_task_dict

    # Get task directly from lm-eval
    task_name = "multirc"
    task_dict = get_task_dict([task_name])
    if task_name not in task_dict:
        print(f"Warning: Task '{task_name}' not found in lm-eval")
        alt_docs = []
    else:
        task = task_dict[task_name]

        # Get the task's test documents
        alt_docs = []
        if hasattr(task, "test_docs"):
            # For lm-eval versions with test_docs method
            alt_docs = list(task.test_docs())
            print(f"Loaded {len(alt_docs)} docs using test_docs method")
        elif hasattr(task, "dataset"):
            # For newer lm-eval versions
            dataset = task.dataset
            if hasattr(dataset, "test"):
                alt_docs = list(dataset.test)
                print(f"Loaded {len(alt_docs)} docs from dataset.test")
            elif hasattr(dataset, "validation"):
                alt_docs = list(dataset.validation)
                print(f"Loaded {len(alt_docs)} docs from dataset.validation")
            else:
                # Fallback to the main dataset
                alt_docs = list(dataset)
                print(f"Loaded {len(alt_docs)} docs from main dataset")
        else:
            print("No suitable data loading method found")

    print(f"Alternative method loaded: {len(alt_docs)} samples")
    if alt_docs and len(alt_docs) > 0:
        print(f"Alternative method sample fields: {list(alt_docs[0].keys())}")

except Exception as e:
    print(f"Alternative loading method failed: {e}")
    alt_docs = []

# Try different data splits
docs = None
split_used = None

# Try test split first
try:
    if hasattr(multirc_task, 'test_docs') and multirc_task.test_docs() is not None:
        docs = list(multirc_task.test_docs())
        split_used = "test"
except:
    pass

# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(multirc_task, 'validation_docs') and multirc_task.validation_docs() is not None:
            docs = list(multirc_task.validation_docs())
            split_used = "validation"
    except:
        pass

# Try train split if validation doesn't work
if docs is None:
    try:
        if hasattr(multirc_task, 'training_docs') and multirc_task.training_docs() is not None:
            docs = list(multirc_task.training_docs())
            split_used = "training"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (test, validation, training)")
    print(f"Available methods: {[method for method in dir(multirc_task) if 'docs' in method]}")
    exit(1)

print(f"MultiRC dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF MULTIRC DATA:")
print("="*80)

# Check data types for available fields
print(f"paragraph type: {type(docs[0].get('paragraph'))}")
print(f"question type: {type(docs[0].get('question'))}")
print(f"answer type: {type(docs[0].get('answer'))}")
print(f"label type: {type(docs[0].get('label'))}")
print(f"idx type: {type(docs[0].get('idx'))}")

# Display first few examples
for i, doc in enumerate(docs[:5]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")
    print(f"Paragraph: {doc.get('paragraph', 'N/A')[:200]}...")
    print(f"Question: {doc.get('question', 'N/A')}")
    print(f"Answer: {doc.get('answer', 'N/A')}")
    print(f"Label: {doc.get('label', 'N/A')}")
    print(f"Index: {doc.get('idx', 'N/A')}")

    # Show task_data.doc_to_text output
    if hasattr(multirc_task, 'doc_to_text'):
        formatted_text = multirc_task.doc_to_text(doc)
        print(f"task_data.doc_to_text output: {formatted_text}")

        # #here - Modify formatted_text to include the answer
        answer = doc.get('answer', '')
        formatted_text_with_answer = formatted_text + answer
        print(f"Modified formatted_text (with answer): {formatted_text_with_answer}")

        # Create formatted_text_with_answer manually without using doc_to_text
        paragraph = doc.get('paragraph', '')
        question = doc.get('question', '')
        manual_formatted = f"{paragraph} \nQuestion: {question}\nAnswer: {answer}"
        print(f"Manual formatted_text (paragraph+question+answer): {manual_formatted}")
    else:
        print("task_data.doc_to_text method not available")

    # Show task_data.doc_to_target output if available
    if hasattr(multirc_task, 'doc_to_target'):
        target = multirc_task.doc_to_target(doc)
        print(f"task_data.doc_to_target output: {target}")
    else:
        print("task_data.doc_to_target method not available")

    # Show index structure if available
    idx = doc.get('idx', {})
    if isinstance(idx, dict):
        print(f"  Paragraph Index: {idx.get('paragraph', 'N/A')}")
        print(f"  Question Index: {idx.get('question', 'N/A')}")
        print(f"  Answer Index: {idx.get('answer', 'N/A')}")

print(f"\n" + "="*80)
print("DATASET STATISTICS:")
print("="*80)
print(f"Total samples: {len(docs)}")

# Count labels
label_counts = {}
for doc in docs:
    label = doc.get('label')
    label_counts[label] = label_counts.get(label, 0) + 1

print(f"Label distribution: {label_counts}")

# Show unique questions and answers
unique_questions = set()
unique_answers = set()
for doc in docs[:100]:  # Check first 100 for efficiency
    unique_questions.add(doc.get('question', ''))
    unique_answers.add(doc.get('answer', ''))

print(f"Unique questions (first 100 samples): {len(unique_questions)}")
print(f"Unique answers (first 100 samples): {len(unique_answers)}")
print(f"Sample answers: {list(unique_answers)[:10]}")

