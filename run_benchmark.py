# Integrated with Wisent-Guard v1

import argparse
from wisent_guard.benchmarks import BenchmarkLoader, BenchmarkEvaluator, save_results
from wisent_guard.classifier import ActivationClassifier
from wisent_guard.monitor import ActivationMonitor


def main(benchmark_name: str, model_name: str, layer_num: int, use_classifier: bool = True):
    # 1. Load benchmark and split into train/test
    loader = BenchmarkLoader(benchmark_name)
    train_data, test_data = loader.get_train_test_split()

    # 2. Prepare training pairs (positive/negative samples)
    prompt_pairs = loader.get_prompt_pairs(train_data)

    classifier = None
    evaluator = BenchmarkEvaluator(model_name, layer_num)

    # You must extract questions from test data to use as prompts
    test_questions = [loader.task.doc_to_text(example) for example in test_data]

    if use_classifier:
        print("Using real activation vectors for classifier training...")

        # Load the model and hook activations
        monitor = ActivationMonitor(evaluator.model, "llama", [layer_num])

        # Generate model responses (this will also trigger activation hooks)
        responses = evaluator.get_model_responses(test_questions)

        # Extract activations
        activation_data = monitor.activations[layer_num]  # Expecting Dict[layer] -> List[Vector]

        # Ensure alignment
        if len(activation_data) != len(test_data):
            raise ValueError("Mismatch between activation count and test data!")

        harmless_acts = []
        harmful_acts = []
        for act, ex in zip(activation_data, test_data):
            label = ex.get("label", "harmless")
            if label == "harmless":
                harmless_acts.append({"activations": act})
            else:
                harmful_acts.append({"activations": act})

        classifier = ActivationClassifier.create_from_activations(
            harmless_activations=harmless_acts,
            harmful_activations=harmful_acts,
            model_type="mlp",
            threshold=0.5
        )
    else:
        # No classifier — just generate outputs
        responses = evaluator.get_model_responses(test_questions)

    # 3. Evaluate
    results = evaluator.evaluate_responses(test_data, responses, classifier)

    # 4. Save results
    save_results(results, benchmark_name, model_name, layer_num)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark evaluation with Wisent-Guard")
    parser.add_argument("benchmark_name", help="Name of the benchmark to evaluate")
    parser.add_argument("model_name", help="Name of the model to evaluate")
    parser.add_argument("layer_num", type=int, help="Layer number to monitor")
    parser.add_argument("--no-classifier", action="store_true", help="Disable classifier evaluation")

    args = parser.parse_args()

    results = main(
        benchmark_name=args.benchmark_name,
        model_name=args.model_name,
        layer_num=args.layer_num,
        use_classifier=not args.no_classifier
    )

    # Output
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")