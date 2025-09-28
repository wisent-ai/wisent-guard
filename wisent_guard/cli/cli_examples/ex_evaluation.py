# Example usage of EvaluatioRotator. Now we can evaluate models using different evaluation strategies. 
from wisent_guard.cli.evaluators.evaluator_rotator import EvaluatorRotator
import time

#1 Create a rotator, it will auto-discover available evaluators in the specified location
rot = EvaluatorRotator()

# #2 List available evaluators
evaluators = EvaluatorRotator.list_evaluators()

print("Available evaluators:")
for ev in evaluators:
    print(f"- {ev['name']}: {ev.get('description', '')} (class: {ev['class']})")

#3 Use a specific evaluator by name (e.g., "nlp"). To add custom evaluators, implement them
# in wisent_guard/core/evaluators/oracles and they will be auto-discovered.
rot.use("nlp")

#4 Evaluate a sample response against an expected answer
res = rot.evaluate("The answer is probably 42", expected="The answer is 42")
print(res)
time.sleep(1) 

#5 Evaluate a batch of responses
responses = ["The answer is probably 42", "I think it's 7", "Definitely 100"]
expected_answers = ["The answer is 12", "The answer is 7", "The answer is 99"]
batch_res = rot.evaluate_batch(responses, expected_answers)

for r in batch_res:
    print(r)
    time.sleep(1)  

#7 Example of using interactive evaluation
rot.use("interactive")
interactive_res = rot.evaluate("The sky is blue.", expected="The sky is blue.")
print(interactive_res)
