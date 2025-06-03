import json
import logging
import itertools
import os
import random
from typing import Any, Dict, List
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM
from wisent_guard.monitor import ActivationMonitor
from wisent_guard.benchmarking.benchmark_runner import BenchmarkRunner
from wisent_guard.utils.logger import get_logger
from lm_eval.tasks import get_task

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    def __init__(self, random_seed=42, log_level="info"):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        self.logger = get_logger("wisent_guard", log_level)

    def load_dataset(self, benchmark_names: List[str]) -> Dict[str, Any]:
        datasets = {}
        for benchmark in benchmark_names:
            task_cls = get_task(benchmark)
            task = task_cls()
            datasets[benchmark] = task
        return datasets

    def set_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_samples(self, task) -> List[Dict[str, Any]]:
        docs = list(itertools.islice(task.validation_docs(), 200))
        samples = []
        for doc in docs:
            prompt = task.doc_to_text(doc)
            target = task.doc_to_target(doc)
            samples.append({"prompt": prompt, "response": target, "label": 1})

            if hasattr(task, "doc_to_choice"):
                choices = task.doc_to_choice(doc)
                negative_choices = [choice for choice in choices if choice != target]
                for choice in negative_choices[:1]:
                    samples.append({"prompt": prompt, "response": choice, "label": 0})
        return samples

    def extract_activations(self, prompt: str, response: str) -> np.ndarray:
        self.monitor.reset()
        inputs = self.tokenizer(prompt + response, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            self.model(**inputs)

        activations_dict = self.monitor.get_activations()
        if not activations_dict:
            raise ValueError("No activations captured. Check if the hook is properly registered and triggered.")
        layer_acts = list(activations_dict.values())[0]
        return layer_acts.mean(dim=0).cpu().numpy()

    def run_benchmark(
        self,
        benchmark_names: List[str],
        model,
        tokenizer,
        model_name: str,
        layer: int,
        device: str,
        output_dir: str
    ):
        self.device = device
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.monitor = ActivationMonitor(model=model, layers=[layer], device=device)
        self.set_model(model, tokenizer)

        benchmarks = self.load_dataset(benchmark_names)

        all_results = {}
        for name, task in benchmarks.items():
            samples = self.generate_samples(task)
            labels = [s["label"] for s in samples]
            print("LABEL COUNTS:", {l: labels.count(l) for l in set(labels)})

            train_samples, test_samples = train_test_split(
                samples, test_size=0.2, random_state=self.random_seed,
                stratify=[s["label"] for s in samples]
            )

            X_train = [self.extract_activations(s["prompt"], s["response"]) for s in train_samples]
            y_train = [s["label"] for s in train_samples]

            classifier = LogisticRegression(max_iter=1000, random_state=self.random_seed)
            classifier.fit(X_train, y_train)
            print("TRAINING ACCURACY:", classifier.score(X_train, y_train))

            X_test = [self.extract_activations(s["prompt"], s["response"]) for s in test_samples]
            y_test = [s["label"] for s in test_samples]
            y_pred = classifier.predict(X_test)

            hallucinations_total = sum(1 for label in y_test if label == 0)
            hallucinations_caught = sum(1 for pred, true in zip(y_pred, y_test) if pred == 0 and true == 0)

            results = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "hallucinations_caught": hallucinations_caught,
                "hallucinations_total": hallucinations_total,
                "hallucination_detection_rate": hallucinations_caught / hallucinations_total if hallucinations_total else 0
            }

            all_results[name] = results

            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{name}_results.json"), 'w') as f:
                json.dump(results, f, indent=4)

        with open(os.path.join(output_dir, "combined_results.json"), 'w') as f:
            json.dump(all_results, f, indent=4)

        # print summary
        print("\n=== FINAL METRICS ===")
        for bench, result in all_results.items():
            print(f"Benchmark: {bench}")
            for key, val in result.items():
                print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

        return all_results