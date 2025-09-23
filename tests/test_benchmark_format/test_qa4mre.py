from lm_eval.tasks import get_task_dict

# Load QA4MRE task (using 2012 variant as example)
tasks = get_task_dict(["qa4mre_2013"])
qa4mre_task = tasks["qa4mre_2013"]

# Try different data splits
docs = None
split_used = None

# Try test split first
try:
    if hasattr(qa4mre_task, 'test_docs') and qa4mre_task.test_docs() is not None:
        docs = list(qa4mre_task.test_docs())
        split_used = "test"
except:
    pass

# Try validation split if test doesn't work
if docs is None:
    try:
        if hasattr(qa4mre_task, 'validation_docs') and qa4mre_task.validation_docs() is not None:
            docs = list(qa4mre_task.validation_docs())
            split_used = "validation"
    except:
        pass

# Try train split if validation doesn't work
if docs is None:
    try:
        if hasattr(qa4mre_task, 'training_docs') and qa4mre_task.training_docs() is not None:
            docs = list(qa4mre_task.training_docs())
            split_used = "training"
    except:
        pass

# Check if we have any data
if docs is None:
    print("❌ No data found in any split (test, validation, training)")
    print(f"Available methods: {[method for method in dir(qa4mre_task) if 'docs' in method]}")
    exit(1)

print(f"QA4MRE dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF QA4MRE DATA:")
print("="*80)

# Check data types
print(f"correct_answer_id type: {type(docs[0].get('correct_answer_id'))}")
print(f"answer_str type: {type(docs[0].get('answer_options'))}")

# Display first few examples
for i, doc in enumerate(docs[:5]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")
    print(f"Topic: {doc.get('topic_name', 'N/A')}")
    print(f"Document: {doc.get('document_str', 'N/A')[:200]}...")
    print(f"Question: {doc.get('question_str', 'N/A')}")
    print(f"Answer Options: {doc.get('answer_options', 'N/A')}")
    print(f"Correct Answer ID: {doc.get('correct_answer_id', 'N/A')}")
    print(f"Correct Answer: {doc.get('correct_answer_str', 'N/A')}")
    print(f"Test ID: {doc.get('test_id', 'N/A')}")
    print(f"Document ID: {doc.get('document_id', 'N/A')}")
    print(f"Question ID: {doc.get('question_id', 'N/A')}")
    print(f"Topic ID: {doc.get('topic_id', 'N/A')}")



