import os
import json
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from .guard import ActivationGuard
from .utils.helpers import ensure_dir
import lm_eval
import lm_eval.tasks

class BenchmarkRunner:
    def __init__(
        self,
        model: str,
        layer: int,
        save_dir: str = "./benchmark_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.layer = layer
        self.save_dir = save_dir
        ensure_dir(save_dir)
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Initialize guard
        self.guard = ActivationGuard(
            model=self.model,
            tokenizer=self.tokenizer,
            layers=[layer],
            device=device
        )

    def run_benchmark(self, benchmark_name: str) -> Dict:
        """Run a specific benchmark from lm-evaluation-harness"""
        # Get the benchmark task
        task = lm_eval.tasks.get_task(benchmark_name)
        
        # Prepare train/test split
        train_data, test_data = self._split_data(task)
        
        # Train the guard on positive/negative examples
        self._train_guard(train_data)
        
        # Evaluate on test set
        results = self._evaluate(test_data)
        
        # Save results
        self._save_results(benchmark_name, results)
        
        return results

    def _split_data(self, task) -> Tuple[List, List]:
        """Split benchmark data into train/test sets (80/20)"""
        # Get all examples
        examples = list(task.get_all_examples())
        
        # Shuffle and split
        import random
        random.shuffle(examples)
        split_idx = int(0.8 * len(examples))
        
        return examples[:split_idx], examples[split_idx:]

    def _train_guard(self, train_data: List):
        """Train the guard on positive/negative examples"""
        positive_examples = []
        negative_examples = []
        
        for example in train_data:
            # Get good and bad responses for this benchmark
            good_response = self._get_good_response(example)
            bad_response = self._get_bad_response(example)
            
            if good_response and bad_response:
                positive_examples.append((example, good_response))
                negative_examples.append((example, bad_response))
        
        # Train the guard
        self.guard.train_on_contrastive_pairs(
            positive_examples,
            negative_examples
        )

    def _get_good_response(self, example) -> str:
        """Get the good (truthful/correct) response for an example"""
        return example.get_target()  # This will vary by benchmark

    def _get_bad_response(self, example) -> str:
        """Generate or get a bad (hallucinated/incorrect) response"""
        # For TruthfulQA, we can use the opposite of the good response
        good_response = self._get_good_response(example)
        return f"The opposite of {good_response}"  # This is a simple example

    def _evaluate(self, test_data: List) -> Dict:
        """Evaluate the guard on test data"""
        results = {
            "total_examples": len(test_data),
            "caught_hallucinations": 0,
            "accuracy": 0.0
        }
        
        for example in test_data:
            # Get model prediction
            prediction = self._get_model_prediction(example)
            
            # Check if guard caught hallucination
            if self.guard.detect_hallucination(prediction):
                results["caught_hallucinations"] += 1
        
        results["accuracy"] = results["caught_hallucinations"] / len(test_data)
        return results

    def _get_model_prediction(self, example) -> str:
        """Get model prediction for an example"""
        prompt = example.get_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.guard.device)
        outputs = self.model.generate(
            **inputs,
            max_length=100,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _save_results(self, benchmark_name: str, results: Dict):
        """Save benchmark results"""
        filename = os.path.join(self.save_dir, f"{benchmark_name}_results.json")
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
