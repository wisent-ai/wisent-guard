#!/usr/bin/env python3
"""
Generate reference DAC steering vectors using the original DAC implementation.

This script creates reference steering vectors by:
1. Loading ITA (Italian) and ENG (English) datasets
2. Computing mean activations for each dataset using the original DAC implementation
3. Computing difference vectors (ITA - ENG) for language steering
4. Saving the vectors and evaluation results for validation testing

Dataset Configuration:
    - For fast testing, we limit both ITA and ENG datasets to 20 examples each
    - Datasets used: ITA_train.json (Italian responses) and ENG_train.json (English responses)
    - Original datasets: /workspace/wisent-guard/Dynamic-Activation-Composition/data/dataset/
    - Reference data: /workspace/wisent-guard/tests/steering_validation/dac/reference_data/

Usage:
    python generate_data_with_original_dac_implementation.py

Expected Output:
    - mean_activations_ita.pt: Mean activations for Italian responses (all layers)
    - mean_activations_eng.pt: Mean activations for English responses (all layers)
    - diff_activations_ita_eng.pt: Difference vector (ITA - ENG) for steering across all layers
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path

import torch

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from const import (
    MODEL_NAME,
    ACTIVATIONS_A_PATH,
    ACTIVATIONS_B_PATH,
    DIFF_ACTIVATIONS_PATH,
    DATASET_A_NAME,
    DATASET_B_NAME,
    MAX_EXAMPLES,
    MAX_NEW_TOKENS,
    ICL_EXAMPLES,
    REFERENCE_DATA_PATH,
)

# Add the Dynamic-Activation-Composition directory to the path
DAC_DIR = Path(__file__).parent.parent.parent.parent / "Dynamic-Activation-Composition"
sys.path.insert(0, str(DAC_DIR))

# Change to DAC directory for proper path resolution
original_cwd = os.getcwd()
os.chdir(DAC_DIR)

try:
    from diff_main import main as diff_main
except ImportError as e:
    print(f"Error importing DAC modules: {e}")
    print(f"Make sure the Dynamic-Activation-Composition directory exists at: {DAC_DIR}")
    os.chdir(original_cwd)
    raise


def copy_datasets_to_reference():
    """Copy the dataset files to the reference_data directory."""
    REFERENCE_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Copy dataset files
    source_dir = DAC_DIR / "data" / "dataset"

    files_to_copy = [
        (
            source_dir / DATASET_A_NAME / f"{DATASET_A_NAME}_train.json",
            REFERENCE_DATA_PATH / f"{DATASET_A_NAME.lower()}_train.json",
        ),
        (
            source_dir / DATASET_B_NAME / f"{DATASET_B_NAME}_train.json",
            REFERENCE_DATA_PATH / f"{DATASET_B_NAME.lower()}_train.json",
        ),
    ]

    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"   ✓ Copied {src.name} to reference_data/")
        else:
            print(f"   ⚠️  Source file not found: {src}")


def main():
    """
    Generate DAC reference vectors using the original implementation via diff_main.
    """
    start_time = time.time()

    print("=" * 70)
    print("DAC Reference Vector Generation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Datasets: {DATASET_A_NAME} vs {DATASET_B_NAME}")
    print(f"Examples per dataset: {MAX_EXAMPLES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"ICL examples: {ICL_EXAMPLES}")
    print(f"DAC operates on: ALL layers simultaneously")
    print(f"Output directory: {REFERENCE_DATA_PATH}")
    print("=" * 70)

    try:
        # Step 1: Copy datasets to reference directory
        print("\n[1/3] Copying datasets to reference_data directory...")
        copy_datasets_to_reference()

        # Step 2: Run diff_main to generate vectors and evaluation results
        print(f"\n[2/3] Running diff_main.py to generate vectors and evaluation results...")
        print(f"   Working directory: {os.getcwd()}")

        # Call diff_main with appropriate parameters
        diff_main(
            model_name=MODEL_NAME,
            load_in_8bit=False,
            dataset_A_name=f"{DATASET_A_NAME}_train",
            dataset_B_name=f"{DATASET_B_NAME}_train",
            icl_examples=0,
            pre_append_instruction=False,
            support=MAX_EXAMPLES,  # Use only 4 examples for computing mean activations (like original)
            max_new_tokens=MAX_NEW_TOKENS,
            debug=False,
            eval_dataset=None,
            skip_eval=True,  # Skip evaluation to save time, we only need the vectors
        )

        # Step 3: Copy generated vectors to reference_data directory
        print(f"\n[3/3] Copying generated vectors to reference_data directory...")

        # Find the output directory created by diff_main
        output_dir = Path(f"./output/{MODEL_NAME.split('/')[1]}/{DATASET_A_NAME}/diff")
        print(f"   Looking in: {output_dir}")

        # List all files in the output directory for debugging
        if output_dir.exists():
            print(f"   Found files in output directory:")
            for f in output_dir.glob("*.pt"):
                print(f"      - {f.name}")

        # Expected file names from diff_main
        mean_a_file = output_dir / f"mean_activations_A_icl{ICL_EXAMPLES}_tok{MAX_NEW_TOKENS}_{DATASET_A_NAME}.pt"
        mean_b_file = output_dir / f"mean_activations_B_icl{ICL_EXAMPLES}_tok{MAX_NEW_TOKENS}_{DATASET_B_NAME}.pt"
        diff_file = (
            output_dir / f"diff_mean_act_icl{ICL_EXAMPLES}_tok{MAX_NEW_TOKENS}_{DATASET_A_NAME}-{DATASET_B_NAME}.pt"
        )
        results_file = (
            output_dir / f"results_icl{ICL_EXAMPLES}_tok{MAX_NEW_TOKENS}_{DATASET_A_NAME}-{DATASET_B_NAME}.json"
        )

        print(f"   Looking for files:")
        print(f"      - {mean_a_file.name}")
        print(f"      - {mean_b_file.name}")
        print(f"      - {diff_file.name}")

        # Copy vectors to reference_data with standard names
        if mean_a_file.exists():
            shutil.copy2(mean_a_file, ACTIVATIONS_A_PATH)
            print(f"   ✓ Copied mean activations A to: {ACTIVATIONS_A_PATH}")
        else:
            print(f"   ⚠️  Mean activations A not found: {mean_a_file}")

        if mean_b_file.exists():
            shutil.copy2(mean_b_file, ACTIVATIONS_B_PATH)
            print(f"   ✓ Copied mean activations B to: {ACTIVATIONS_B_PATH}")
        else:
            print(f"   ⚠️  Mean activations B not found: {mean_b_file}")

        if diff_file.exists():
            shutil.copy2(diff_file, DIFF_ACTIVATIONS_PATH)
            print(f"   ✓ Copied difference activations to: {DIFF_ACTIVATIONS_PATH}")
        else:
            print(f"   ⚠️  Difference activations not found: {diff_file}")

        # Load and display vector information
        if ACTIVATIONS_A_PATH.exists() and ACTIVATIONS_B_PATH.exists() and DIFF_ACTIVATIONS_PATH.exists():
            mean_a = torch.load(ACTIVATIONS_A_PATH)
            mean_b = torch.load(ACTIVATIONS_B_PATH)
            diff = torch.load(DIFF_ACTIVATIONS_PATH)

            # Summary
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 70)
            print("SUCCESS: DAC reference vectors generated!")
            print("=" * 70)
            print(f"Total time: {elapsed_time:.1f} seconds")
            print(f"Mean activations A shape: {mean_a.shape}")
            print(f"Mean activations B shape: {mean_b.shape}")
            print(f"Difference activations shape: {diff.shape}")
            print(f"Vector norm (A): {torch.norm(mean_a).item():.4f}")
            print(f"Vector norm (B): {torch.norm(mean_b).item():.4f}")
            print(f"Difference norm: {torch.norm(diff).item():.4f}")
            print("\nFiles saved in reference_data/:")
            print(f"   - {ACTIVATIONS_A_PATH.name}")
            print(f"   - {ACTIVATIONS_B_PATH.name}")
            print(f"   - {DIFF_ACTIVATIONS_PATH.name}")
            print("=" * 70)
        else:
            print("\n⚠️  Some expected files were not generated")

    except Exception as e:
        print(f"\n❌ Error during vector generation: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
