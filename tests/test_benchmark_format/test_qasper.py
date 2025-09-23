from datasets import load_dataset

# Load Qasper dataset from Hugging Face
dataset = load_dataset("allenai/qasper")

# Try different data splits
docs = None
split_used = None

# Try test split first
if "test" in dataset:
    docs = list(dataset["test"])
    split_used = "test"
elif "validation" in dataset:
    docs = list(dataset["validation"])
    split_used = "validation"
elif "train" in dataset:
    docs = list(dataset["train"])
    split_used = "train"
else:
    print("❌ No data found in any split")
    print(f"Available splits: {list(dataset.keys())}")
    exit(1)

print(f"Qasper dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF QASPER DATA:")
print("="*80)

# Display first few examples
for i, doc in enumerate(docs[:3]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")
    print(f"Title: {doc.get('title', 'N/A')}")
    print(f"Abstract: {doc.get('abstract', 'N/A')[:200]}...")
    print(f"Full text (first 200 chars): {str(doc.get('full_text', 'N/A'))[:200]}...")

    # Display questions and answers
    qas = doc.get('qas', [])
    print(f"Number of questions: {len(qas)}")

    for j, qa in enumerate(qas[:2]):  # Show first 2 questions per document
        print(f"  Question {j+1}: {qa.get('question', 'N/A')}")
        answers = qa.get('answers', [])
        print(f"  Number of answers: {len(answers)}")
        for k, answer in enumerate(answers[:1]):  # Show first answer
            print(f"    Answer {k+1}: {answer.get('answer', {}).get('free_form_answer', 'N/A')}")


