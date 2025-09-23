from __future__ import annotations

"""
Run examples:

IMPORTANT: Provide correct path to the model if not using a HF model id/or if the model is local.

  # minimal
  python3 -m wisent_guard.cli_bricks.cli_tests.cli_control_vector_creator \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --task-name truthfulqa_mc1 \
    --layer 9 --steering-method CAA \
    --save-vector /path/to/control_vector.pt

  # with custom data from JSON
  python3 -m wisent_guard.cli_bricks.cli_tests.cli_control_vector_creator \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --data-json /path/to/data.json \
    --layer 9 --steering-method CAA \
    --save-vector /path/to/control_vector.pt
"""

import argparse
import sys
import json
import torch
from pathlib import Path

try:
    from wisent_guard.cli_bricks.cli_run_task_steering import (
        PipelineSettings,
        ModelSettings,
        DataSettings,
        ActivationSettings,
        SteeringSettings,
        TrackingSettings,
        EvaluationSettings,
        _build_or_reuse_model,
        _load_data,
        _compute_activations,
        _build_steering_obj,
    )
    from wisent_guard.cli_bricks.cli_steering import CAAConfig, HPRConfig, DACConfig, BiPOConfig, KSteeringConfig
except Exception as e:
    raise ImportError("Failed to import cli_run_task_steering") from e

# Type alias for Python 3.10 compatibility
from typing import Union, Optional
SteeringConfig = Union[CAAConfig, HPRConfig, DACConfig, BiPOConfig, KSteeringConfig]


def _csv_ints_to_tuple(val: Optional[str]) -> Optional[tuple[int, ...]]:
    """
    Convert a comma-separated string of integers into a tuple of integers.

    Arguments:
        val: A string like "1,2,3" or None.
    Returns:
        A tuple of integers (1, 2, 3) or None if input is None.
    """
    if val is None:
        return None
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


def _maybe_int_or_str(x: str) -> Union[int, str]:
    """
    Convert a string to an integer if possible, otherwise return the string.

    Arguments:
        x: Input string.

    Returns:
        int if conversion is successful, else the original string.
    """
    try:
        return int(x)
    except ValueError:
        return x


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser for the CLI.
    """
    p = argparse.ArgumentParser(
        prog="wisent_guard.cli_control_vector_creator",
        description="Create and save control vectors using various steering methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model-name", required=True, help="HF model id or local path")
    p.add_argument("--device", default="cuda", help="cuda or cpu")

    # Data sources (mutually exclusive)
    data_group = p.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--task-name", help="Task name (e.g., truthfulqa_mc1)")
    data_group.add_argument("--data-json", help="Path to JSON file with contrastive pairs")

    # Data parameters
    p.add_argument("--split-ratio", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=100, help="Total data limit")
    p.add_argument("--training-limit", type=int, default=100, help="Training data limit")
    p.add_argument("--seed", type=int, default=42)

    # Activation
    p.add_argument("--prompt-strategy", default="multiple_choice")
    p.add_argument("--token-target", default="choice_token")
    p.add_argument("--token-aggregation", default="average")
    p.add_argument("--layer", type=_maybe_int_or_str, default=9, help="Layer index for activation extraction")

    # Steering method
    p.add_argument(
        "--steering-method",
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
        help="Steering method to use"
    )
    p.add_argument("--steering-strength", type=float, default=1.0)

    # CAA specific
    p.add_argument("--normalization-method", default="none", help="CAA normalization method")
    p.add_argument("--target-norm", type=float, default=None, help="CAA target norm")

    # KSteering specific
    p.add_argument("--target-labels", default=None, help="comma-separated target labels (KSteering)")
    p.add_argument("--avoid-labels", default=None, help="comma-separated avoid labels (KSteering)")
    p.add_argument("--alpha", type=float, default=0.5, help="KSteering alpha parameter")

    # Output
    p.add_argument("--save-vector", required=True, help="Path to save the control vector (.pt file)")
    p.add_argument("--verbose", action="store_true")

    return p


def load_json_data(json_path: str) -> list[dict]:
    """
    Load contrastive pairs from JSON file.

    Expected format:
    [
        {
            "question": "What is the capital of France?",
            "correct_answer": "Paris",
            "incorrect_answer": "London"
        },
        ...
    ]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON data must be a list of objects")

    # Validate format
    required_keys = {"question", "correct_answer", "incorrect_answer"}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be a dictionary")
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            raise ValueError(f"Item {i} missing required keys: {missing_keys}")

    return data


def create_steering_config(args) -> SteeringConfig:
    """Create steering configuration based on arguments."""
    if args.steering_method == "CAA":
        return CAAConfig(
            method="CAA",
            device=args.device,
            normalization_method=args.normalization_method,
            target_norm=args.target_norm,
        )
    elif args.steering_method == "HPR":
        return HPRConfig(method="HPR", device=args.device)
    elif args.steering_method == "DAC":
        return DACConfig(method="DAC", device=args.device)
    elif args.steering_method == "BiPO":
        return BiPOConfig(method="BiPO", device=args.device)
    elif args.steering_method == "KSteering":
        return KSteeringConfig(
            method="KSteering",
            device=args.device,
            target_labels=_csv_ints_to_tuple(args.target_labels) or (1,),
            avoid_labels=_csv_ints_to_tuple(args.avoid_labels) or (0,),
            alpha=args.alpha,
        )
    else:
        raise ValueError(f"Unknown steering method: {args.steering_method}")


def settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    """
    Convert command-line arguments to pipeline settings.
    """
    # Determine task name
    task_name = args.task_name if args.task_name else "custom_json_data"

    return PipelineSettings(
        model=ModelSettings(model_name=args.model_name, device=args.device),
        data=DataSettings(
            task_name=task_name,
            split_ratio=args.split_ratio,
            limit=args.limit,
            training_limit=args.training_limit,
            testing_limit=0,  # We don't need testing data for vector creation
            seed=args.seed,
        ),
        activation=ActivationSettings(
            prompt_strategy=args.prompt_strategy,
            token_target=args.token_target,
            token_aggregation=args.token_aggregation,
            layer=args.layer,
        ),
        steering=SteeringSettings(
            config=create_steering_config(args),
            steering_method=args.steering_method,
            steering_strength=args.steering_strength,
        ),
        tracking=TrackingSettings(verbose=args.verbose),
        evaluation=EvaluationSettings(output_mode="none"),  # No evaluation needed
    )


def save_control_vector(steering_method, layer: int, save_path: str, task_name: str, verbose: bool = False):
    """
    Save the trained steering method as a control vector.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract the control vector depending on the method type
    if hasattr(steering_method, 'vector'):
        # CAA, HPR, DAC methods typically have a 'vector' attribute
        control_vector = steering_method.vector
    elif hasattr(steering_method, 'control_vector'):
        # Some methods might use 'control_vector'
        control_vector = steering_method.control_vector
    elif hasattr(steering_method, 'steering_vector'):
        # Alternative naming
        control_vector = steering_method.steering_vector
    else:
        # For other methods, try to extract the main tensor
        raise ValueError(f"Cannot extract control vector from {type(steering_method).__name__}")

    # Prepare save data
    save_data = {
        "vector": control_vector,
        "layer": layer,
        "task": task_name,
        "method": steering_method.__class__.__name__,
        "shape": control_vector.shape if control_vector is not None else None,
    }

    # Add method-specific data
    if hasattr(steering_method, 'normalization_method'):
        save_data["normalization_method"] = steering_method.normalization_method
    if hasattr(steering_method, 'target_norm'):
        save_data["target_norm"] = steering_method.target_norm

    # Save to file
    torch.save(save_data, save_path)

    if verbose:
        print(f"✅ Control vector saved to: {save_path}")
        print(f"   Method: {save_data['method']}")
        print(f"   Layer: {layer}")
        print(f"   Shape: {save_data['shape']}")
        print(f"   Task: {task_name}")


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        print(f"🚀 Creating control vector using {args.steering_method}")
        print(f"   Model: {args.model_name}")
        print(f"   Layer: {args.layer}")
        print(f"   Device: {args.device}")
        if args.task_name:
            print(f"   Task: {args.task_name}")
        if args.data_json:
            print(f"   Data JSON: {args.data_json}")
        print()

    try:
        settings = settings_from_args(args)

        # Step 1: Build model
        if args.verbose:
            print("📦 Loading model...")
        model = _build_or_reuse_model(settings.model)

        # Step 2: Load data
        if args.verbose:
            print("📊 Loading data...")

        if args.data_json:
            # Load custom JSON data
            json_data = load_json_data(args.data_json)

            # Convert JSON data to qa_pairs format
            qa_pairs = []
            for item in json_data:
                qa_pairs.append({
                    "question": item["question"],
                    "correct_answer": item["correct_answer"],
                    "incorrect_answer": item["incorrect_answer"],
                    # Add any other fields that might be expected
                    "task": "custom_json_data"
                })

            if args.verbose:
                print(f"   Loaded {len(qa_pairs)} contrastive pairs from JSON")
        else:
            qa_pairs, _, _ = _load_data(settings, model)

        # Step 3: Compute activations
        if args.verbose:
            print("🧠 Computing activations...")
        pair_set = _compute_activations(settings, model, qa_pairs, trackers=None)

        # Step 4: Build and train steering method
        if args.verbose:
            print(f"🎯 Training {args.steering_method} steering method...")
        steering_obj = _build_steering_obj(settings, pair_set)

        # Step 5: Save control vector
        if args.verbose:
            print("💾 Saving control vector...")
        save_control_vector(
            steering_obj,
            settings.activation.layer,
            args.save_vector,
            settings.data.task_name,
            args.verbose
        )

        print("🎉 Control vector creation completed successfully!")

    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())