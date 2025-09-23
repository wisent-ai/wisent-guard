from lm_eval.tasks import get_task_dict
import ast

# Load RACE task
tasks = get_task_dict(["race"])
race_task = tasks["race"]

# Load test data
docs = list(race_task.test_docs())
split_used = "test"

# Check if we have any data
if not docs:
    print("❌ No data found in test split")
    print(f"Available methods: {[method for method in dir(race_task) if 'docs' in method]}")
    exit(1)

print(f"RACE dataset loaded: {len(docs)} samples from {split_used} split")
print("\n" + "="*80)
print("EXAMPLES OF RACE DATA:")
print("="*80)

# Check data types and structure
print(f"First doc keys: {list(docs[0].keys())}")
print(f"Article type: {type(docs[0].get('article'))}")
print(f"Problems type: {type(docs[0].get('problems'))}")

# Display first few examples
for i, doc in enumerate(docs[:3]):
    print(f"\n--- EXAMPLE {i+1} ---")
    print(f"All fields: {list(doc.keys())}")
    print(f"Article: {doc.get('article', 'N/A')}...")

    # Parse the problems Python literal string
    problems_str = doc.get('problems', '')
    try:
        problems = ast.literal_eval(problems_str)
        print(f"Number of problems: {len(problems)}")

        for j, problem in enumerate(problems):
            print(f"\n  Problem {j+1}:")
            question = problem.get('question', 'N/A')
            options = problem.get('options', 'N/A')
            answer = problem.get('answer', 'N/A')
            print(f"  Question: {question} (type: {type(question)})")
            print(f"  Options: {options} (type: {type(options)})")
            print(f"  Answer: {answer} (type: {type(answer)})")

    except Exception as e:
        print(f"Failed to parse problems: {e}")



