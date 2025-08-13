#!/usr/bin/env python3
"""
Generate reference DAC steering vectors and text completions using the original DAC implementation.

IMPORTANT: This script uses REDUCED datasets (20 examples each) for faster testing and CI/CD.
The original DAC implementation normally processes full datasets, but we reduced them to:
- Speed up testing and development cycles
- Maintain compatibility with resource-constrained environments
- Preserve data quality while reducing computation time
- Enable rapid iteration during debugging and validation

This script creates comprehensive reference data by:
1. Loading ITA (Italian) and ENG (English) datasets
2. Computing mean activations for each dataset using the original DAC implementation
3. Computing difference vectors (ITA - ENG) for language steering
4. Generating reference text completions using dynamic steering
5. Saving the vectors, text completions, and adaptation metrics for validation testing

Dataset Configuration:
    - REDUCED DATASETS: Limited to 20 examples each (instead of full ~1000+ examples)
    - Datasets used: ITA_train.json (Italian responses) and ENG_train.json (English responses)
    - Original datasets: /workspace/wisent-guard/Dynamic-Activation-Composition/data/dataset/
    - Reference data: /workspace/wisent-guard/tests/steering_validation/dac/reference_data/
    - This reduction maintains statistical validity while enabling faster testing

Usage:
    python generate_data_with_original_dac_implementation.py

Expected Output:
    - mean_activations_ita.pt: Mean activations for Italian responses (all layers)
    - mean_activations_eng.pt: Mean activations for English responses (all layers)
    - diff_activations_ita_eng.pt: Difference vector (ITA - ENG) for steering across all layers
    - text_completions_unsteered.json: Baseline text generation without steering
    - text_completions_dynamic_steering.json: Dynamic steering text generation results
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path

import torch

# Import aggressive_memory_cleanup
utils_path = Path(__file__).parent.parent / "utils.py"
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import aggressive_memory_cleanup

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
    DYNAMIC_CONFIG,
)

# Add the Dynamic-Activation-Composition directory to the path
DAC_DIR = Path(__file__).parent.parent.parent.parent / "Dynamic-Activation-Composition"
sys.path.insert(0, str(DAC_DIR))

# Change to DAC directory for proper path resolution
original_cwd = os.getcwd()
os.chdir(DAC_DIR)

try:
    from diff_main import main as diff_main
    from src.delta import generate_token_step, generate_dynamic_edited_model
    from src.utils.model_utils import load_model_and_tokenizer
    from src.utils.prompt_helper import tokenize_ICL
except ImportError as e:
    print(f"Error importing DAC modules: {e}")
    print(f"Make sure the Dynamic-Activation-Composition directory exists at: {DAC_DIR}")
    os.chdir(original_cwd)
    raise


# Test prompts for language steering validation
TEST_PROMPTS = [
    "What is the weather like?",
    "Tell me about the food",
    "Describe a typical day",
    "How are you today?",
    "What do you think about music?",
]

# Dynamic steering configuration imported from const.py (uses starting_alpha=4.0 for stronger steering)


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


def generate_reference_text_completions():
    """
    Generate reference text completions using original DAC implementation.

    Creates both unsteered baseline and dynamic steering results.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Generating Reference Text Completions")
    print("=" * 70)
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Top-p values: {DYNAMIC_CONFIG['top_p_values']}")
    print(f"Starting alpha: {DYNAMIC_CONFIG['starting_alpha']}")
    print(f"Max new tokens: {DYNAMIC_CONFIG['max_new_tokens']}")
    print("=" * 70)

    # Load model and tokenizer using original DAC approach
    print("\n[4.1] Loading model and tokenizer...")
    model, tokenizer, config, device = load_model_and_tokenizer(
        model_name=MODEL_NAME,
        load_in_8bit=False,  # Use full precision for consistency
    )

    # Load the difference tensor for single-property dynamic steering (like original DAC)
    if not DIFF_ACTIVATIONS_PATH.exists():
        raise FileNotFoundError(f"Difference tensor not found: {DIFF_ACTIVATIONS_PATH}")

    diff_activations = torch.load(DIFF_ACTIVATIONS_PATH, map_location=device)  # ITA - ENG difference
    print(f"   ✓ Loaded difference tensor (ITA-ENG): {diff_activations.shape}")

    # Load datasets for building ICL prompts
    print("\n[4.1.5] Loading datasets for ICL prompt building...")
    ita_dataset_path = REFERENCE_DATA_PATH / "ita_train.json"
    eng_dataset_path = REFERENCE_DATA_PATH / "eng_train.json"

    with open(ita_dataset_path, "r", encoding="utf-8") as f:
        ita_dataset = json.load(f)
    with open(eng_dataset_path, "r", encoding="utf-8") as f:
        eng_dataset = json.load(f)

    def build_icl_prompt_for_generation(query: str, icl_examples: int = 4, use_italian_examples: bool = False):
        """Build ICL prompt for text generation (matching original DAC approach)"""
        if icl_examples == 0:
            return query

        icl_parts = []
        dataset_to_use = ita_dataset if use_italian_examples else eng_dataset

        # Use first N examples as ICL context
        for i in range(min(icl_examples, len(dataset_to_use))):
            example = dataset_to_use[i]
            icl_parts.append(f"Q: {example['input']}")
            icl_parts.append(f"A: {example['output']}")
            icl_parts.append("")  # Empty line between examples

        # Add the query we want to complete
        icl_parts.append(f"Q: {query}")
        icl_parts.append("A:")

        return "\\n".join(icl_parts)

    # Generate unsteered completions (baseline - no ICL)
    print("\\n[4.2] Generating unsteered baseline completions...")
    unsteered_results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"   Processing unsteered prompt {i + 1}/{len(TEST_PROMPTS)}: {prompt[:30]}...")

        # Build simple prompt (no ICL for baseline)
        simple_prompt = f"Q: {prompt}\\nA:"
        prompt_tokens = tokenizer(simple_prompt, return_tensors="pt", add_special_tokens=True)
        prompt_tensor = prompt_tokens["input_ids"].to(device)

        # Generate unsteered text
        generated_ids = generate_token_step(
            model=model,
            prompt=prompt_tensor,
            max_new_tokens=DYNAMIC_CONFIG["max_new_tokens"],
        )

        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[prompt_tensor.shape[1] :].squeeze(), skip_special_tokens=True)

        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "method": "unsteered_baseline",
            "tokens_generated": len(generated_ids) - prompt_tensor.shape[1],
            "model": MODEL_NAME,
        }
        unsteered_results.append(result)

    # Save unsteered results
    unsteered_path = REFERENCE_DATA_PATH / "text_completions_unsteered.json"
    with open(unsteered_path, "w", encoding="utf-8") as f:
        json.dump(unsteered_results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved unsteered completions to: {unsteered_path}")

    # Generate dynamic steering completions
    print("\n[4.3] Generating dynamic steering completions...")
    dynamic_steering_results = []

    for top_p in DYNAMIC_CONFIG["top_p_values"]:
        print(f"   \n--- Processing top_p = {top_p} ---")

        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"   Processing dynamic prompt {i + 1}/{len(TEST_PROMPTS)} (top_p={top_p}): {prompt[:30]}...")

            # Build ICL prompt with Italian examples (for steering toward Italian)
            icl_prompt = build_icl_prompt_for_generation(prompt, icl_examples=4, use_italian_examples=True)
            print(f"      ICL prompt preview: {icl_prompt[:100]}...")

            # Tokenize ICL prompt
            prompt_tokens = tokenizer(icl_prompt, return_tensors="pt", add_special_tokens=True)
            prompt_tensor = prompt_tokens["input_ids"].to(device)

            # Generate with single-property dynamic steering (like original DAC)
            try:
                generated_ids, alpha_used, real_kls = generate_dynamic_edited_model(
                    model=model,
                    config=config,
                    no_icl_prompt=prompt_tensor,  # Now contains ICL examples
                    diff_mean_activations=diff_activations,  # Single difference vector (ITA - ENG)
                    max_new_tokens=DYNAMIC_CONFIG["max_new_tokens"],
                    starting_alpha=DYNAMIC_CONFIG["starting_alpha"],
                    top_p=top_p,
                )

                # Decode generated text
                generated_text = tokenizer.decode(
                    generated_ids[prompt_tensor.shape[1] :].squeeze(), skip_special_tokens=True
                )

                result = {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "method": "dynamic_steering_original",
                    "top_p": top_p,
                    "starting_alpha": DYNAMIC_CONFIG["starting_alpha"],
                    "alpha_history": alpha_used,  # Single alpha adaptation sequence
                    "kl_history": real_kls,  # Single KL divergence sequence
                    "tokens_generated": len(generated_ids) - prompt_tensor.shape[1],
                    "model": MODEL_NAME,
                    "steering_property": "language_ita_eng",
                }
                dynamic_steering_results.append(result)

            except Exception as e:
                print(f"      ⚠️ Error generating dynamic steering for '{prompt}': {e}")
                # Add error result for completeness
                result = {
                    "prompt": prompt,
                    "generated_text": "",
                    "method": "dynamic_steering_original",
                    "top_p": top_p,
                    "error": str(e),
                    "model": MODEL_NAME,
                }
                dynamic_steering_results.append(result)

    # Save dynamic steering results
    dynamic_path = REFERENCE_DATA_PATH / "text_completions_dynamic_steering.json"
    with open(dynamic_path, "w", encoding="utf-8") as f:
        json.dump(dynamic_steering_results, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved dynamic steering completions to: {dynamic_path}")

    # Summary statistics
    successful_generations = len([r for r in dynamic_steering_results if "error" not in r])
    total_generations = len(dynamic_steering_results)

    print("\n" + "=" * 70)
    print("TEXT GENERATION SUMMARY")
    print("=" * 70)
    print(f"Unsteered completions: {len(unsteered_results)}")
    print(f"Dynamic steering completions: {successful_generations}/{total_generations}")
    print(f"Top-p configurations tested: {len(DYNAMIC_CONFIG['top_p_values'])}")
    print("=" * 70)

    # Clean up model to free GPU memory
    print("   Cleaning up GPU memory...")
    del model
    aggressive_memory_cleanup()

    return unsteered_results, dynamic_steering_results


def main():
    """
    Generate DAC reference vectors and text completions using the original implementation.
    """
    start_time = time.time()

    print("=" * 70)
    print("DAC Reference Data Generation")
    print("=" * 70)
    print("PHASE 1: Vector Generation")
    print(f"Model: {MODEL_NAME}")
    print(f"Datasets: {DATASET_A_NAME} vs {DATASET_B_NAME}")
    print(f"Examples per dataset: {MAX_EXAMPLES}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"ICL examples: {ICL_EXAMPLES}")
    print(f"DAC operates on: ALL layers simultaneously")
    print("PHASE 2: Text Generation")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    print(f"Dynamic top-p values: {DYNAMIC_CONFIG['top_p_values']}")
    print(f"Output directory: {REFERENCE_DATA_PATH}")
    print("=" * 70)

    try:
        # Step 1: Copy datasets to reference directory
        print("\n[1/5] Copying datasets to reference_data directory...")
        copy_datasets_to_reference()

        # Step 2: Run diff_main to generate vectors and evaluation results
        print(f"\n[2/5] Running diff_main.py to generate vectors and evaluation results...")
        print(f"   Working directory: {os.getcwd()}")

        # Call diff_main with appropriate parameters (using ICL_EXAMPLES from const.py)
        diff_main(
            model_name=MODEL_NAME,
            load_in_8bit=False,
            dataset_A_name=f"{DATASET_A_NAME}_train",
            dataset_B_name=f"{DATASET_B_NAME}_train",
            icl_examples=ICL_EXAMPLES,  # Now correctly using ICL_EXAMPLES=4
            pre_append_instruction=False,
            support=MAX_EXAMPLES,  # Use 20 examples for computing mean activations
            max_new_tokens=MAX_NEW_TOKENS,
            debug=False,
            eval_dataset=None,
            skip_eval=True,  # Skip evaluation to save time, we only need the vectors
        )

        # Step 3: Copy generated vectors to reference_data directory
        print(f"\n[3/5] Copying generated vectors to reference_data directory...")

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

        # Step 4: Generate text completions using original DAC dynamic steering
        if DIFF_ACTIVATIONS_PATH.exists():
            # IMPORTANT: Clean up GPU memory from diff_main before loading model again
            print("\n   Performing aggressive GPU memory cleanup...")
            aggressive_memory_cleanup()

            print("[4/5] Generating reference text completions...")

            # Change back to original directory for proper imports
            os.chdir(original_cwd)
            unsteered_results, dynamic_results = generate_reference_text_completions()
            os.chdir(DAC_DIR)  # Change back for any remaining operations

            # Step 5: Display vector information and final summary
            print("\n[5/5] Final validation and summary...")
            mean_a = torch.load(ACTIVATIONS_A_PATH)
            mean_b = torch.load(ACTIVATIONS_B_PATH)
            diff = torch.load(DIFF_ACTIVATIONS_PATH)

            # Final summary
            elapsed_time = time.time() - start_time
            print("\n" + "=" * 70)
            print("SUCCESS: DAC reference data generation complete!")
            print("=" * 70)
            print(f"Total time: {elapsed_time:.1f} seconds")
            print("\n📊 VECTOR GENERATION:")
            print(f"Mean activations A shape: {mean_a.shape}")
            print(f"Mean activations B shape: {mean_b.shape}")
            print(f"Difference activations shape: {diff.shape}")
            print(f"Vector norm (A): {torch.norm(mean_a).item():.4f}")
            print(f"Vector norm (B): {torch.norm(mean_b).item():.4f}")
            print(f"Difference norm: {torch.norm(diff).item():.4f}")
            print("\n📝 TEXT GENERATION:")
            print(f"Unsteered completions: {len(unsteered_results)}")
            print(f"Dynamic steering completions: {len([r for r in dynamic_results if 'error' not in r])}")
            print(f"Top-p configurations: {len(DYNAMIC_CONFIG['top_p_values'])}")
            print("\n📁 Files saved in reference_data/:")
            print(f"   - {ACTIVATIONS_A_PATH.name}")
            print(f"   - {ACTIVATIONS_B_PATH.name}")
            print(f"   - {DIFF_ACTIVATIONS_PATH.name}")
            print(f"   - text_completions_unsteered.json")
            print(f"   - text_completions_dynamic_steering.json")
            print("=" * 70)
        else:
            print("\n⚠️  Vector generation failed - cannot generate text completions")

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
