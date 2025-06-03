import os
import json
from typing import Dict, List, Tuple
from lm_eval import evaluator
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

class BenchmarkLoader:
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.task = evaluator.get_task(benchmark_name)()
        self.docs = list(self.task.validation_docs())

    def get_train_test_split(self, test_size=0.2, random_state=42):
        from sklearn.model_selection import train_test_split
        return train_test_split(self.docs, test_size=test_size, random_state=random_state)

    def get_prompt_pairs(self, data):
        prompt_pairs = []
        for doc in data:
            text = self.task.doc_to_text(doc)
            target = self.task.doc_to_target(doc)

            # fake negative just to show structure (you will need real negatives for training!)
            prompt_pairs.append((f"{text} {target}", f"{text} wrong_answer"))
        return prompt_pairs

class BenchmarkEvaluator:
    def __init__(self, model_name: str, layer_num: int):
        self.model_name = model_name
        self.layer_num = layer_num
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_model_responses(self, questions: List[str]) -> List[str]:
        responses = []
        for question in tqdm(questions, desc="Generating responses"):
            input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_new_tokens=50)
                decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                responses.append(decoded)
        return responses

    def evaluate_responses(self, test_data: List[Dict], responses: List[str], classifier=None) -> Dict:
        correct_answers = [example["choices"][example["answer"]] for example in test_data]

        # Simple matching-based accuracy
        matched = 0
        for target, pred in zip(correct_answers, responses):
            if target.lower() in pred.lower():
                matched += 1
        acc = matched / len(test_data)

        metrics = {"accuracy": acc}

        if classifier:
            harmful_count = 0
            for response in responses:
                pred = classifier.predict({"activations": [0.1] * 768})  # fake data stub
                if pred["is_harmful"]:
                    harmful_count += 1
            metrics["detected_harmful"] = harmful_count
            metrics["detection_rate"] = harmful_count / len(responses)

        return metrics


def save_results(results: Dict, benchmark_name: str, model_name: str, layer_num: int):
    os.makedirs("benchmark_results", exist_ok=True)
    fname = f"benchmark_results/{benchmark_name}_{model_name}_layer{layer_num}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    benchmark_name = "truthfulqa_mc"
    model_name = "meta-llama/Llama-3.1-8B"
    layer_num = 15

    loader = BenchmarkLoader(benchmark_name)
    train_data, test_data = loader.get_train_test_split()

    evaluator = BenchmarkEvaluator(model_name, layer_num)

    questions = [loader.task.doc_to_text(doc) for doc in test_data]
    responses = evaluator.get_model_responses(questions)
    results = evaluator.evaluate_responses(test_data, responses)

    save_results(results, benchmark_name, model_name, layer_num)
    print("Benchmark completed. Results saved.")