#!/usr/bin/env python3
"""
Main script to generate control vectors from all JSON seed pairs.

This script processes all JSON files in the seed_pairs directory and generates
corresponding control vectors serialized as JSON with plain activations.

Usage:
    python generate_all_control_vectors.py [--model MODEL] [--layer LAYER] [--device DEVICE] [--limit LIMIT]

Example:
    python generate_all_control_vectors.py --model microsoft/DialoGPT-small --layer 5 --limit 10
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

from control_vector_generator import ControlVectorGenerator

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('control_vector_generation.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate control vectors from JSON seed pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-dir',
        default='seed_pairs',
        help='Directory containing input JSON files'
    )

    parser.add_argument(
        '--output-dir',
        default='control_vectors',
        help='Directory to save output control vector JSONs'
    )

    parser.add_argument(
        '--model',
        default='unsloth/Qwen3-4B-bnb-4bit',
        help='HuggingFace model name to use for activation extraction'
    )

    parser.add_argument(
        '--layer',
        type=int,
        default=17,
        help='Layer index to extract activations from'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use for computation'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )

    parser.add_argument(
        '--resume-from',
        default=None,
        help='Resume from specific trait name (alphabetically)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--multiple-choice',
        action='store_true',
        help='Use multiple-choice format for prompts'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger.info("=== Control Vector Generation Pipeline ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Limit: {args.limit}")
    logger.info(f"Resume from: {args.resume_from}")
    logger.info(f"Multiple-choice format: {args.multiple_choice}")

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist!")
        sys.exit(1)

    # Count JSON files
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        logger.error(f"No JSON files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(json_files)} JSON files")

    # Apply limit if specified
    if args.limit:
        json_files = json_files[:args.limit]
        logger.info(f"Limited to {len(json_files)} files")

    # Apply resume filter if specified
    if args.resume_from:
        json_files = [f for f in json_files if f.stem >= args.resume_from]
        logger.info(f"Resuming from {args.resume_from}: {len(json_files)} files to process")

    # Sort files for consistent processing order
    json_files.sort()

    # Initialize generator
    generator = ControlVectorGenerator(
        model_name=args.model,
        layer_index=args.layer,
        device=args.device
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    start_time = time.time()
    results = {}

    for i, json_file in enumerate(json_files):
        trait_name = json_file.stem

        # Check if already processed
        suffix = "_mc_control_vector.json" if args.multiple_choice else "_control_vector.json"
        output_file = output_dir / f"{trait_name}{suffix}"
        if output_file.exists():
            logger.info(f"Skipping {trait_name} (already exists)")
            results[trait_name] = True
            continue

        logger.info(f"Processing {i+1}/{len(json_files)}: {trait_name}")

        try:
            if args.multiple_choice:
                success = generator.process_single_json_file_mc(json_file, output_dir)
            else:
                success = generator.process_single_json_file(json_file, output_dir)
            results[trait_name] = success

            if success:
                logger.info(f"✓ Successfully processed {trait_name}")
            else:
                logger.error(f"✗ Failed to process {trait_name}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Saving progress...")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing {trait_name}: {e}")
            results[trait_name] = False
            continue

    # Save results summary
    end_time = time.time()
    duration = end_time - start_time

    successful = sum(1 for success in results.values() if success)
    total = len(results)

    summary = {
        "total_files": len(json_files),
        "processed_files": total,
        "successful_files": successful,
        "failed_files": total - successful,
        "success_rate": successful / total if total > 0 else 0,
        "duration_seconds": duration,
        "model": args.model,
        "layer": args.layer,
        "device": args.device,
        "results": results
    }

    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    logger.info("=== Generation Complete ===")
    logger.info(f"Total files: {len(json_files)}")
    logger.info(f"Processed: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {total - successful}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Duration: {duration/60:.1f} minutes")
    logger.info(f"Summary saved to: {summary_file}")

    if successful < total:
        logger.warning(f"{total - successful} files failed to process. Check the logs for details.")
        sys.exit(1)
    else:
        logger.info("All files processed successfully!")


if __name__ == "__main__":
    main()