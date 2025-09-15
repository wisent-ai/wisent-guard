"""
Minimal-yet-flexible cli for steering.

It loads a model, prepares train/test data, builds contrastive pairs,
extracts activations, constructs a steering method, and evaluates with LM Harness.

Example:
    python3 -m tools.cli_steering_test \
        --model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
        --tasks hellaswag \
        --layer 11 \
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from wisent_guard.core import Model
from wisent_guard.cli_bricks.cli_activation import (
    make_collector,
    create_contrastive_pairs,
    extract_activations_for_pairs,
    build_pair_set_with_real_activations,
)
from wisent_guard.cli_bricks.cli_data import load_train_test_data, LoadDataResult
from wisent_guard.core.steering_methods.steering_evaluation import run_lm_harness_evaluation

try:
    from wisent_guard.cli_bricks.cli_steering import build_steering_for_mode
except Exception as e:
    raise ImportError(
        "Failed to import build_steering_for_mode from wisent_guard.cli_bricks.cli_steering. "
        "Ensure that wisent_guard is installed correctly and that the cli_bricks module is accessible."
    ) from e


def auto_device(requested: str) -> str:
    """Return 'cuda' if available and requested is 'auto', else return requested."""
    if requested.lower() != "auto":
        return requested
    try:
        import torch  
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Wisent Guard pipeline with CLI args.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", "-m", required=True, help="HF model path or ID (as used by wisent_guard.core.Model).")
    p.add_argument("--tasks", "-t", required=True,
                   help="Comma-separated task names (e.g., 'hellaswag' or 'hellaswag,mmlu').")
    p.add_argument("--layer", type=int, default=11, help="Layer index to extract activations / apply steering.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Device to use. 'auto' picks CUDA if available.")
    p.add_argument("--seed", type=int, default=2024, help="Random seed for data split.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Data settings
    p.add_argument("--shots", type=int, default=2, help="Few-shot shots.")
    p.add_argument("--split-ratio", type=float, default=0.7, help="Train/test split ratio.")
    p.add_argument("--limit", type=int, default=10, help="Global limit for dataset size (if supported).")
    p.add_argument("--training-limit", type=int, default=10, help="Limit for training examples.")
    p.add_argument("--testing-limit", type=int, default=5, help="Limit for testing examples.")
    p.add_argument("--prompt-strategy", default="multiple_choice",
                   help="Prompt construction strategy (passed to create_contrastive_pairs).")
    p.add_argument("--token-target", default="last_token",
                   help="Token targeting strategy for activations (e.g., 'last_token').")

    # Steering settings (mirroring your defaults)
    p.add_argument("--method", default="CAA", help="Steering method name for build_steering_for_mode.")
    p.add_argument("--normalization-method", default="none", help="Normalization method (or 'none').")
    p.add_argument("--target-norm", type=float, default=None, help="Target norm if normalization is used.")
    p.add_argument("--dac-dynamic-control", action="store_true", help="Enable DAC dynamic control.")
    p.add_argument("--dac-entropy-threshold", type=float, default=0.5)
    p.add_argument("--bipo-beta", type=float, default=0.5)
    p.add_argument("--bipo-lr", type=float, default=0.01)
    p.add_argument("--bipo-epochs", type=int, default=10)
    p.add_argument("--k-num-labels", type=int, default=2)
    p.add_argument("--k-hidden", type=int, default=128)
    p.add_argument("--k-lr", type=float, default=0.01)
    p.add_argument("--k-epochs", type=int, default=10)
    p.add_argument("--k-target", default="1")
    p.add_argument("--k-avoid", default="0")
    p.add_argument("--k-alpha", type=float, default=0.5)

    # Evaluation settings
    p.add_argument("--steering-strength", type=float, default=1.0)
    p.add_argument("--use-test-split", action="store_true", default=True, help="Evaluate on the test split.")
    p.add_argument("--output-mode", choices=["metrics", "predictions", "both"], default="both",
                   help="Output mode for run_lm_harness_evaluation.")

    # I/O and cache
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Cache dir; if not set, a temp directory will be created.")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Optional path to dump the evaluation result JSON.")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    log = logging.getLogger("wisent_cli")

    # Prepare cache dir
    if args.cache_dir is None:
        tmp_root = Path(tempfile.mkdtemp(prefix="cli_data_tests_"))
        cache_dir = tmp_root / "benchmark_cache"
    else:
        cache_dir = args.cache_dir
    (cache_dir / "data").mkdir(parents=True, exist_ok=True)
    log.info("Using cache dir: %s", cache_dir)

    # Device
    device = auto_device(args.device)
    log.info("Using device: %s", device)

    # Tasks
    task_names: List[str] = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not task_names:
        raise ValueError("No valid tasks provided via --tasks")

    # Load model
    log.info("Loading model: %s", args.model)
    model = Model(name=str(args.model))

    # Load data
    log.info("Loading train/test data for tasks: %s", ", ".join(task_names))
    data: LoadDataResult = load_train_test_data(
        model=model,
        task_names=task_names,
        shots=args.shots,
        split_ratio=args.split_ratio,
        limit=args.limit,
        training_limit=args.training_limit,
        testing_limit=args.testing_limit,
        seed=args.seed,
        verbose=args.verbose,
    )
    layer = args.layer
    qa_pairs             = data["qa_pairs"]
    test_qa_pairs_source = data["test_qa_pairs_source"]
    task_data            = data["task_data"]

    # Build contrastive pairs and extract activations
    log.info("Creating collector and contrastive pairs...")
    collector = make_collector(model)

    contrastive_pairs = create_contrastive_pairs(
        collector=collector,
        prompt_construction_strategy=args.prompt_strategy,
        qa_pairs=qa_pairs,
        verbose=args.verbose,
    )
    log.info("Extracting activations (layer=%d)...", layer)

    processed_pairs = extract_activations_for_pairs(
        collector=collector,
        contrastive_pairs=contrastive_pairs,
        layer=layer,
        device=device,
        token_targeting_strategy=args.token_target,
        latency_tracker=None,
        verbose=args.verbose,
    )

    # Build pair set
    pair_set_task_name = task_names[0] if len(task_names) == 1 else ",".join(task_names)
    pair_set = build_pair_set_with_real_activations(
        processed_pairs=processed_pairs,
        task_name=pair_set_task_name,
        verbose=args.verbose,
    )

    # Build steering
    log.info("Building steering method: %s", args.method)
    steering = build_steering_for_mode(
        method_name=args.method,
        device=device,
        normalization_method=args.normalization_method,
        target_norm=args.target_norm,
        dac_dynamic_control=args.dac_dynamic_control,
        dac_entropy_threshold=args.dac_entropy_threshold,
        bipo_beta=args.bipo_beta,
        bipo_lr=args.bipo_lr,
        bipo_epochs=args.bipo_epochs,
        k_num_labels=args.k_num_labels,
        k_hidden=args.k_hidden,
        k_lr=args.k_lr,
        k_epochs=args.k_epochs,
        k_target=args.k_target,
        k_avoid=args.k_avoid,
        k_alpha=args.k_alpha,
        save_path=None,
        load_path=None,
        layer_idx=layer,
        pair_set=pair_set,
        verbose=args.verbose,
    )

    # Evaluate
    log.info("Running LM Harness evaluation...")
    res: Dict[str, Any] = run_lm_harness_evaluation(
        task_data=task_data,
        test_qa_pairs=test_qa_pairs_source,
        model=model,
        steering_methods=[steering],
        layers=[layer],
        steering_strength=args.steering_strength,
        use_test_split=args.use_test_split,
        verbose=args.verbose,
        output_mode=args.output_mode,
    )

    # Print a compact summary to stdout
    try:
        # Heuristic: show top-level keys and, if metrics exist, show them
        summary_keys = ", ".join(list(res.keys())[:6])
        log.info("Result keys: %s%s", summary_keys, " ..." if len(res.keys()) > 6 else "")
        metrics = res.get("metrics") or res.get("results") or {}
        if metrics:
            log.info("Metrics summary: %s", {k: metrics[k] for k in list(metrics)[:8]})
    except Exception:
        pass

    # Optional JSON dump
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=str)
        log.info("Wrote results to %s", args.output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
