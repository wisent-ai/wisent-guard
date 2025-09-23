from lm_eval.tasks import get_task_dict

# Load BoolQ task
tasks = get_task_dict(["boolq"])
boolq_task = tasks["boolq"]

# Get validation documents (samples)
docs = list(boolq_task.validation_docs())

print(f"BoolQ dataset loaded: {len(docs)} samples")
print("\n" + "="*80)
print("EXAMPLES OF BOOLQ DATA:")
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
    print(f"Passage: {doc.get('passage', 'N/A')[:200]}...")
    print(f"Question: {doc.get('question', 'N/A')}")
    print(f"Answer: {doc.get('answer', 'N/A')}")
    print(f"Label: {doc.get('label', 'N/A')}")
    # Try alternative field names
    print(f"Alternative answer fields:")
    print(f"  - answer: {doc.get('answer')}")
    print(f"  - label: {doc.get('label')}")
    print(f"  - target: {doc.get('target')}")
    print(f"  - correct: {doc.get('correct')}")
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
    print("- passage: The reading passage text")
    print("- question: A yes/no question about the passage")
    print("- answer: True/False answer")
    print("- label: Boolean label (True=1, False=0)")
    if 'idx' in sample:
        print("- idx: Index information")

    # Show a few more samples to understand the pattern
    print(f"\n" + "="*80)
    print("MORE SAMPLE DATA STRUCTURES:")
    print("="*80)

    for i in range(min(5, len(docs))):
        print(f"\n--- SAMPLE {i+1} STRUCTURE ---")
        print(json.dumps(docs[i], indent=2))

print(f"\n" + "="*80)
print("ANSWER DISTRIBUTION:")
print("="*80)

# Analyze answer distribution
if docs:
    true_count = sum(1 for doc in docs if doc.get('answer') == True or doc.get('label') == 1)
    false_count = sum(1 for doc in docs if doc.get('answer') == False or doc.get('label') == 0)

    print(f"True answers: {true_count}")
    print(f"False answers: {false_count}")
    print(f"Total samples: {len(docs)}")
    print(f"True percentage: {true_count/len(docs)*100:.1f}%")
    print(f"False percentage: {false_count/len(docs)*100:.1f}%")

print(f"\n" + "="*80)
print("SAMPLE QUESTIONS BY TYPE:")
print("="*80)

# Show different types of questions
if docs:
    question_types = {}
    for doc in docs[:50]:  # Analyze first 50 questions
        question = doc.get('question', '')

        # Categorize by question starters
        if question.lower().startswith('is '):
            question_types.setdefault('is_questions', []).append(question)
        elif question.lower().startswith('are '):
            question_types.setdefault('are_questions', []).append(question)
        elif question.lower().startswith('was '):
            question_types.setdefault('was_questions', []).append(question)
        elif question.lower().startswith('were '):
            question_types.setdefault('were_questions', []).append(question)
        elif question.lower().startswith('do '):
            question_types.setdefault('do_questions', []).append(question)
        elif question.lower().startswith('does '):
            question_types.setdefault('does_questions', []).append(question)
        elif question.lower().startswith('did '):
            question_types.setdefault('did_questions', []).append(question)
        elif question.lower().startswith('can '):
            question_types.setdefault('can_questions', []).append(question)
        elif question.lower().startswith('will '):
            question_types.setdefault('will_questions', []).append(question)
        else:
            question_types.setdefault('other_questions', []).append(question)

    for q_type, questions in question_types.items():
        print(f"\n{q_type.upper()} ({len(questions)} examples):")
        for q in questions[:3]:  # Show first 3 examples
            print(f"  - {q}")
        if len(questions) > 3:
            print(f"  ... and {len(questions) - 3} more")