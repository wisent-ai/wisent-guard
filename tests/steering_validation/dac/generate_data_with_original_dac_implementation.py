#!/usr/bin/env python3
"""
Generate reference DAC steering vectors using the original DAC implementation.

This script creates reference steering vectors by:
1. Loading ITA (Italian) and ENG (English) datasets
2. Computing mean activations for each dataset using the original DAC implementation
3. Computing difference vectors (ITA - ENG) for language steering
4. Saving the vectors for validation testing

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
import time
from pathlib import Path

import torch

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from model_utils import (
    create_dac_datasets,
    create_dac_original_mean_activations,
    setup_dac_model,
    validate_vector_shapes,
)
from const import (
    MODEL_NAME,
    ACTIVATIONS_A_PATH,
    ACTIVATIONS_B_PATH,
    DIFF_ACTIVATIONS_PATH,
    DATASET_A_NAME,
    DATASET_B_NAME,
    MAX_EXAMPLES,
    REFERENCE_DATA_PATH,
)


def main():
    """
    Generate DAC reference vectors using the original implementation.
    """
    start_time = time.time()

    print("=" * 70)
    print("DAC Reference Vector Generation")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Datasets: {DATASET_A_NAME} vs {DATASET_B_NAME}")
    print(f"Examples per dataset: {MAX_EXAMPLES}")
    print(f"DAC operates on: ALL layers simultaneously")
    print(f"Output directory: {REFERENCE_DATA_PATH}")
    print("=" * 70)

    # Check if reference data already exists
    if ACTIVATIONS_A_PATH.exists() and ACTIVATIONS_B_PATH.exists() and DIFF_ACTIVATIONS_PATH.exists():
        print("\n⚠️  Reference vectors already exist!")
        print(f"   - {ACTIVATIONS_A_PATH.name}")
        print(f"   - {ACTIVATIONS_B_PATH.name}")
        print(f"   - {DIFF_ACTIVATIONS_PATH.name}")

        response = input("\nDo you want to regenerate them? (y/N): ").strip().lower()
        if response != "y":
            print("Using existing reference vectors.")
            return

    try:
        # Step 1: Load datasets
        print("\n[1/5] Loading datasets...")
        dataset_a, dataset_b, instruction_a, instruction_b = create_dac_datasets()
        print(f"   Dataset A ({DATASET_A_NAME}): {len(dataset_a)} examples")
        print(f"   Dataset B ({DATASET_B_NAME}): {len(dataset_b)} examples")
        print(f"   Instruction A: '{instruction_a}'")
        print(f"   Instruction B: '{instruction_b}'")

        # Step 2: Setup model
        print(f"\n[2/5] Setting up model...")
        model, tokenizer, config, device = setup_dac_model(MODEL_NAME)

        # Step 3: Generate activations using original DAC implementation
        print(f"\n[3/5] Computing mean activations with original DAC implementation...")
        mean_activations_a, mean_activations_b, diff_activations = create_dac_original_mean_activations(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device=device,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            instruction_a=instruction_a,
            instruction_b=instruction_b,
        )

        # Step 4: Validate shapes
        print(f"\n[4/5] Validating vector shapes...")
        validate_vector_shapes(mean_activations_a, mean_activations_b, diff_activations)

        # Step 5: Save vectors
        print(f"\n[5/5] Saving reference vectors...")

        # Ensure output directory exists
        REFERENCE_DATA_PATH.mkdir(parents=True, exist_ok=True)

        # Save all three vectors
        torch.save(mean_activations_a, ACTIVATIONS_A_PATH)
        print(f"   ✓ Saved mean activations A: {ACTIVATIONS_A_PATH}")

        torch.save(mean_activations_b, ACTIVATIONS_B_PATH)
        print(f"   ✓ Saved mean activations B: {ACTIVATIONS_B_PATH}")

        torch.save(diff_activations, DIFF_ACTIVATIONS_PATH)
        print(f"   ✓ Saved difference activations: {DIFF_ACTIVATIONS_PATH}")

        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("SUCCESS: DAC reference vectors generated!")
        print("=" * 70)
        print(f"Total time: {elapsed_time:.1f} seconds")
        print(f"Mean activations A shape: {mean_activations_a.shape}")
        print(f"Mean activations B shape: {mean_activations_b.shape}")
        print(f"Difference activations shape: {diff_activations.shape}")
        print(f"Vector norm (A): {torch.norm(mean_activations_a).item():.4f}")
        print(f"Vector norm (B): {torch.norm(mean_activations_b).item():.4f}")
        print(f"Difference norm: {torch.norm(diff_activations).item():.4f}")
        print("\nFiles saved:")
        print(f"   - {ACTIVATIONS_A_PATH}")
        print(f"   - {ACTIVATIONS_B_PATH}")
        print(f"   - {DIFF_ACTIVATIONS_PATH}")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error during vector generation: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
