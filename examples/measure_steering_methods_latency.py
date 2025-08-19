#!/usr/bin/env python3
"""
Example: Measure inference latency for all steering methods in wisent-guard.

This example script benchmarks the inference time overhead of different steering methods:
1. Loads the Llama 3.1 8B model
2. Uses Italian/non-Italian contrastive pairs from examples/data/synthetic_pairs_italian_test.json
3. Measures inference time overhead for each steering method
4. Outputs detailed timing results in JSON format

Usage:
    python examples/measure_steering_methods_latency.py

Results are saved to 'steering_inference_benchmark.json' in the current directory.

Last results:
================================================================================
STEERING METHODS INFERENCE BENCHMARK SUMMARY
================================================================================
Baseline (no steering): 1432.88ms
--------------------------------------------------------------------------------
CAA                 : 1460.25ms (+  1.9%) | Training:    25.7ms
DAC                 : 1494.97ms (+  4.3%) | Training:     0.6ms
HPR                 : 1488.76ms (+  3.9%) | Training:   160.5ms
BiPO                : 1467.65ms (+  2.4%) | Training:   170.5ms
KSteering           : 1506.31ms (+  5.1%) | Training:    38.7ms
TokenSteered        : 1485.21ms (+  3.7%) | Training:     0.5ms
ControlVectorSteering: 1426.05ms (+ -0.5%) | Training:     0.4ms
DACAttention        : 10508.34ms (+ 622.0%)
================================================================================
"""

import json
import logging
import statistics
import time
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair

# Import core classes
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.response import NegativeResponse, PositiveResponse

# Import all steering methods
from wisent_guard.core.steering_methods_tensor.dac_attention import DAC as DACAttention

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SteeringBenchmark:
    """Benchmark class for measuring steering method inference times."""

    def __init__(self, model_path: str, layer_index: int = 15):
        self.model_path = model_path
        self.layer_index = layer_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("✓ Model loaded successfully")

        # Generation parameters
        self.gen_params = {
            "max_new_tokens": 50,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

    def load_contrastive_pairs(self, json_path: str) -> Tuple[ContrastivePairSet, List[str]]:
        """Load contrastive pairs from JSON file and extract test prompts."""
        logger.info(f"Loading contrastive pairs from {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        pair_set = ContrastivePairSet(name=data["name"])
        test_prompts = []

        for pair_data in data["pairs"]:
            prompt = pair_data["prompt"]
            positive_text = pair_data["positive_response"]
            negative_text = pair_data["negative_response"]

            # Extract activations for this pair
            pos_activation = self._extract_activations(positive_text)
            neg_activation = self._extract_activations(negative_text)

            # Create response objects
            pos_response = PositiveResponse(text=positive_text)
            pos_response.activations = pos_activation

            neg_response = NegativeResponse(text=negative_text)
            neg_response.activations = neg_activation

            # Create contrastive pair
            pair = ContrastivePair(prompt=prompt, positive_response=pos_response, negative_response=neg_response)

            pair_set.pairs.append(pair)
            test_prompts.append(prompt)

        logger.info(f"✓ Loaded {len(pair_set.pairs)} contrastive pairs")
        return pair_set, test_prompts

    def _extract_activations(self, text: str) -> torch.Tensor:
        """Extract activations from model for given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=100).to(self.device)

        activations = []

        def hook(module, input, output):
            # Get last token activation and convert to float32 for consistency
            activations.append(output[0][:, -1, :].clone().float())

        # Register hook on target layer
        handle = self.model.model.layers[self.layer_index].register_forward_hook(hook)

        with torch.no_grad():
            self.model(**inputs)

        handle.remove()
        return activations[0].squeeze(0)

    def create_steering_methods(self) -> Dict[str, Any]:
        """Create instances of all steering methods."""
        methods = {}

        # Standard steering methods
        # methods["CAA"] = CAA(device=str(self.device))
        # methods["DAC"] = DAC(device=str(self.device))
        # methods["HPR"] = HPR(device=str(self.device), epochs=10)
        # methods["BiPO"] = BiPO(device=str(self.device), num_epochs=10, batch_size=2)
        # methods["KSteering"] = KSteering(device=str(self.device), num_labels=1, classifier_epochs=20)
        # methods["TokenSteered"] = TokenSteeringWrapper(CAA(device=str(self.device)))
        # methods["ControlVectorSteering"] = ControlVectorSteering(device=str(self.device))

        # # Tensor-based methods
        methods["DACAttention"] = DACAttention(device=str(self.device))

        return methods

    def train_method(self, method_name: str, method: Any, pair_set: ContrastivePairSet) -> float:
        """Train a steering method and return training time."""
        logger.info(f"  Training {method_name}...")

        start_time = time.perf_counter()

        try:
            # Some methods need model reference
            if hasattr(method, "set_model_reference"):
                method.set_model_reference(self.model)

            # DACAttention needs model injection to use our Llama model instead of loading Mistral
            if hasattr(method, "load_model_with_reference"):
                method.load_model_with_reference(self.model, self.tokenizer)

            # Handle tensor-based methods and regular methods with train_property
            if hasattr(method, "train_property"):
                # Check if this is tensor-based (DACAttention) or regular DAC
                if method_name == "DACAttention":
                    # DACAttention doesn't need layer_index
                    method.train_property("italian_style", pair_set)
                else:
                    # Regular DAC needs layer_index
                    method.train_property("italian_style", pair_set, self.layer_index)
            else:
                # Standard training method
                method.train(pair_set, self.layer_index)

        except Exception as e:
            logger.warning(f"  Failed to train {method_name}: {e}")
            return -1.0

        training_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        logger.info(f"  ✓ {method_name} trained in {training_time:.2f}ms")
        return training_time

    def measure_baseline_inference(self, prompts: List[str]) -> Dict[str, Any]:
        """Measure baseline inference time without steering."""
        logger.info("Measuring baseline inference times...")

        times = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100).to(self.device)

            start_time = time.perf_counter()

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.gen_params)

            inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            times.append(inference_time)

        return {
            "inference_times_ms": times,
            "avg_time_ms": statistics.mean(times),
            "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
        }

    def measure_steered_inference(
        self, method_name: str, method: Any, prompts: List[str], strength: float = 1.0
    ) -> Dict[str, Any]:
        """Measure inference time with steering applied."""
        logger.info(f"Measuring inference times for {method_name}...")

        times = []

        # Special handling for DACAttention dynamic steering
        if method_name == "DACAttention":
            # Use the proper dynamic steering method instead of manual hooks
            # This will test the full dynamic capabilities: KL divergence, multiple forward passes, adaptive alphas
            property_weights = {"italian_style": 1.0}  # Use the trained property

            # No hook setup needed - DACAttention handles everything internally
            steering_hook = None
        else:
            # Regular steering methods
            class SteeringHook:
                def __init__(self, steering_method, layer_idx, strength):
                    self.steering_method = steering_method
                    self.layer_idx = layer_idx
                    self.strength = strength
                    self.hooks = []

                def add_hook(self, model):
                    def steering_hook(module, input, output):
                        hidden_states = output[0]
                        last_token = hidden_states[:, -1:, :]
                        original_dtype = hidden_states.dtype

                        try:
                            if hasattr(self.steering_method, "apply_steering"):
                                # Convert to float32 for steering computation, then back to original dtype
                                last_token_f32 = last_token.float()
                                steered = self.steering_method.apply_steering(last_token_f32, self.strength)
                                steered = steered.to(original_dtype)
                                hidden_states[:, -1:, :] = steered
                            elif hasattr(self.steering_method, "apply_steering_tensor"):
                                # Special handling for tensor-based methods
                                last_token_f32 = last_token.float()
                                steered = self.steering_method.apply_steering_tensor(
                                    last_token_f32, self.strength, layer_index=self.layer_idx
                                )
                                steered = steered.to(original_dtype)
                                hidden_states[:, -1:, :] = steered
                        except Exception as e:
                            # If steering fails, just return original activations
                            logger.warning(f"Steering failed during inference: {e}")

                        return (hidden_states,) + output[1:]

                    handle = model.model.layers[self.layer_idx].register_forward_hook(steering_hook)
                    self.hooks.append(handle)

                def remove_hooks(self):
                    for handle in self.hooks:
                        handle.remove()
                    self.hooks = []

            steering_hook = SteeringHook(method, self.layer_index, strength)

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100).to(self.device)

            start_time = time.perf_counter()

            if method_name == "DACAttention":
                # Use DACAttention's native dynamic steering generation
                # This tests the full dynamic capabilities: multiple forward passes, KL divergence, adaptive alphas
                generated_text = method.generate_with_steering(
                    prompt=prompt,
                    property_weights=property_weights,
                    max_new_tokens=self.gen_params["max_new_tokens"],
                    timing_strategy="dynamic",  # Explicit dynamic mode
                )
            else:
                # Regular hook-based steering for other methods
                steering_hook.add_hook(self.model)

                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **self.gen_params)

                # Remove hook
                steering_hook.remove_hooks()

            inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
            times.append(inference_time)

        return {
            "inference_times_ms": times,
            "avg_time_ms": statistics.mean(times),
            "std_time_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
        }

    def run_benchmark(self, contrastive_pairs_path: str, output_path: str = "steering_inference_benchmark_dac.json"):
        """Run the complete benchmark."""
        logger.info("Starting steering inference benchmark...")

        # Load contrastive pairs and test prompts
        pair_set, test_prompts = self.load_contrastive_pairs(contrastive_pairs_path)

        # Limit to 25 prompts for consistent testing
        test_prompts = test_prompts[:25]

        # Create steering methods
        methods = self.create_steering_methods()

        # Measure baseline
        baseline_results = self.measure_baseline_inference(test_prompts)

        # Results dictionary
        results = {
            "benchmark_info": {
                "model": self.model_path,
                "contrastive_pairs_source": contrastive_pairs_path,
                "num_test_prompts": len(test_prompts),
                "layer_index": self.layer_index,
                "steering_strength": 1.0,
                "max_new_tokens": self.gen_params["max_new_tokens"],
                "device": str(self.device),
            },
            "results": {"baseline_no_steering": baseline_results},
        }

        # Train and benchmark each method
        for method_name, method in methods.items():
            logger.info(f"Benchmarking {method_name}...")

            # Train the method
            training_time = self.train_method(method_name, method, pair_set)

            if training_time < 0:
                logger.warning(f"Skipping {method_name} due to training failure")
                continue

            # Measure inference time
            try:
                inference_results = self.measure_steered_inference(method_name, method, test_prompts)

                # Calculate overhead vs baseline
                overhead = (inference_results["avg_time_ms"] / baseline_results["avg_time_ms"] - 1.0) * 100

                results["results"][method_name] = {
                    "training_time_ms": training_time,
                    **inference_results,
                    "overhead_vs_baseline_percent": overhead,
                }

                logger.info(
                    f"✓ {method_name} completed - Avg: {inference_results['avg_time_ms']:.2f}ms, Overhead: {overhead:.1f}%"
                )

            except Exception as e:
                logger.warning(f"Failed to benchmark {method_name}: {e}")

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info(f"✓ Benchmark completed! Results saved to {output_path}")
        return results


def main():
    """Main function to run the benchmark."""
    model_path = "meta-llama/Llama-3.1-8B-Instruct"  # Use HF model identifier
    pairs_path = "/workspace/wisent-guard/examples/data/synthetic_pairs_italian_test.json"

    benchmark = SteeringBenchmark(model_path, layer_index=15)
    results = benchmark.run_benchmark(pairs_path)

    # Print summary
    print("\n" + "=" * 80)
    print("STEERING METHODS INFERENCE BENCHMARK SUMMARY")
    print("=" * 80)

    baseline_avg = results["results"]["baseline_no_steering"]["avg_time_ms"]
    print(f"Baseline (no steering): {baseline_avg:.2f}ms")
    print("-" * 80)

    for method_name, method_results in results["results"].items():
        if method_name == "baseline_no_steering":
            continue

        avg_time = method_results["avg_time_ms"]
        overhead = method_results["overhead_vs_baseline_percent"]
        training_time = method_results["training_time_ms"]

        print(f"{method_name:20}: {avg_time:7.2f}ms (+{overhead:5.1f}%) | Training: {training_time:7.1f}ms")

    print("=" * 80)


if __name__ == "__main__":
    main()
