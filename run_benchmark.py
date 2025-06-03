# Integrated with Wisent-Guard v1

import argparse
from wisent_guard.benchmarks import BenchmarkLoader, BenchmarkEvaluator, save_results
from wisent_guard.classifier import ActivationClassifier


def main(benchmark_name: str, model_name: str, layer_num: int, use_classifier: bool = True):
    # 1. Load benchmark and split into train/test
    loader = BenchmarkLoader(benchmark_name)
    train_data, test_data = loader.get_train_test_split()

    # 2. Prepare training pairs (positive/negative samples)
    prompt_pairs = loader.get_prompt_pairs(train_data)

    classifier = None
    if use_classifier:
        positive_examples = [pair[0] for pair in prompt_pairs]
        negative_examples = [pair[1] for pair in prompt_pairs]

        # This is a placeholder. You must use real activations for real evaluation!
        print("Using fake activation vectors for placeholder classifier training...")
        classifier = ActivationClassifier.create_from_activations(
            harmless_activations=[{"activations": [float(i)] * 768} for i in range(len(positive_examples))],
            harmful_activations=[{"activations": [float(i)] * 768} for i in range(len(negative_examples))],
            model_type="mlp",
            threshold=0.5
        )

    # 3. Run model and evaluate
    evaluator = BenchmarkEvaluator(model_name, layer_num)
    
    # You must extract questions from test data to use as prompts
    test_questions = [loader.task.doc_to_text(example) for example in test_data]
    responses = evaluator.get_model_responses(test_questions)

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
    print("\n Evaluation Results:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    