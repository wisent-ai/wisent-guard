from __future__ import annotations

"""
Run examples:

IMPORTANT: Provide correct path to the model if not using a HF model id/or if the model is local.

  # minimal
  python3 -m wisent_guard.cli_bricks.cli_tests.cli_steering_test \
    --model-name meta-llama/Llama-3.2-1B-Instruct \
    --task-name truthfulqa_mc1
    --layer 9 --steering-method CAA \
"""

import argparse
import sys

import json

try:
    from wisent_guard.cli.cli_run_task_steering import (
        PipelineSettings,
        ModelSettings,
        DataSettings,
        ActivationSettings,
        SteeringSettings,
        TrackingSettings,
        EvaluationSettings,
        run_task_steering_pipeline,
    )
except Exception as e:
    raise ImportError("Failed to import cli_run_task_steering") from e


def _csv_ints_to_tuple(val: str | None) -> tuple[int, ...] | None:
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


def _maybe_int_or_str(x: str) -> int | str:
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
        prog="wisent_guard.cli_run",
        description="Run LM steering pipeline (thin wrapper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model-name", required=True, help="HF model id or local path")
    p.add_argument("--device", default="cuda", help="cuda or cpu")

    # Data
    p.add_argument("--task-name", default="truthfulqa_mc1")
    p.add_argument("--split-ratio", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--training-limit", type=int, default=50)
    p.add_argument("--testing-limit", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    # Activation
    p.add_argument("--prompt-strategy", default="multiple_choice")
    p.add_argument("--token-target", default="choice_token")
    p.add_argument("--token-aggregation", default="average")
    p.add_argument("--layer", type=_maybe_int_or_str, default=9)

    # Steering
    p.add_argument(
        "--steering-method",
        default="CAA",
        choices=["CAA", "HPR", "DAC", "BiPO", "KSteering"],
    )
    p.add_argument("--steering-strength", type=float, default=1.0)
    p.add_argument("--target-labels", default=None, help="comma-separated (KSteering)")
    p.add_argument("--avoid-labels", default=None, help="comma-separated (KSteering)")
    p.add_argument("--alpha", type=float, default=None, help="KSteering alpha")

    # Verbosity / output
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--show-memory-usage", action="store_true")
    p.add_argument("--show-timing-summary", action="store_true")
    p.add_argument(
        "--output-mode", default="both", choices=["both", "summary", "full"]
    )
    p.add_argument("--save-json", default=None)
    p.add_argument("--print-keys", action="store_true")

    return p


def settings_from_args(args: argparse.Namespace) -> PipelineSettings:
    """
    Convert command-line arguments to pipeline settings.
    """
    return PipelineSettings(
        model=ModelSettings(model_name=args.model_name, device=args.device),
        data=DataSettings(
            task_name=args.task_name,
            split_ratio=args.split_ratio,
            limit=args.limit,
            training_limit=args.training_limit,
            testing_limit=args.testing_limit,
            seed=args.seed,
        ),
        activation=ActivationSettings(
            prompt_strategy=args.prompt_strategy,
            token_target=args.token_target,
            token_aggregation=args.token_aggregation,
            layer=args.layer,
        ),
        steering=SteeringSettings(
            steering_method=args.steering_method,
            steering_strength=args.steering_strength,
            default_target_labels=(
                tuple(_csv_ints_to_tuple(args.target_labels) or (1,))
            ),
            default_avoid_labels=(
                tuple(_csv_ints_to_tuple(args.avoid_labels) or (0,))
            ),
            default_alpha=(0.5 if args.alpha is None else float(args.alpha)),
        ),
        tracking=TrackingSettings(
            verbose=args.verbose,
            show_memory_usage=args.show_memory_usage,
            show_timing_summary=args.show_timing_summary,
        ),
        evaluation=EvaluationSettings(output_mode=args.output_mode),
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings = settings_from_args(args)

    try:
        results = run_task_steering_pipeline(settings, trackers=None)
    except KeyboardInterrupt: 
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:  
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if args.print_keys and isinstance(results, dict):
        print("Result keys:", ", ".join(results.keys()))

    if args.save_json and isinstance(results, dict):
        try:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            if args.verbose:
                print(f"Saved results to {args.save_json}")
        except Exception as e:
            print(f"Failed to save results to {args.save_json}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__": 
    raise SystemExit(main())
