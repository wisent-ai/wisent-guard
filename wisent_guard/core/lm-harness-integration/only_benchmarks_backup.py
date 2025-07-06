#!/usr/bin/env python3
"""
Script to process ALL core benchmarks from lm-eval-harness v1.0 list.
Tests each benchmark with wisent guard CLI using --limit 5.
FAILS HARD on first benchmark failure - exits with code 1 immediately.

Covers 70+ comprehensive benchmarks across all major categories:
- Benchmark Suites: GLUE, SuperGLUE, BIG-Bench
- Hallucination/Truthfulness: TruthfulQA (mc1, mc2, gen)
- Reasoning: HellaSwag, PIQA, Winogrande, LogiQA, SWAG, StoryCloze
- QA/Reading: CoQA, DROP, TriviaQA, NaturalQs, WebQs, HeadQA, QASPER
- Knowledge: MMLU, ARC, SciQ, Social-i-QA
- Math: GSM8k, Hendrycks Math, Arithmetic, Asdiv
- Coding: HumanEval, MBPP  
- Bias/Toxicity: ToxiGen, CrowS-Pairs, Hendrycks Ethics
- Adversarial: ANLI
- Multilingual: XNLI, XCopa, XStoryCloze, XWinograd, PAWS-X, MMMLU, MGSM, Belebele
- Medical: MedQA, PubMedQA, HeadQA
- Language Modeling: Lambada variants, Wikitext
- Long Context: NarrativeQA, SCROLLS
- Temporal: MCTACO, PROST
- Linguistic: BLIMP, Unscramble
- Translation: WMT
- Dialogue: BaBI, MuTual

⚠️  FAIL-HARD MODE: Script exits with code 1 on first benchmark failure.
"""

import json
import os
import sys
import subprocess
import tempfile
from typing import Dict, List, Optional
from pathlib import Path

# Import the sample retrieval function from populate_tasks
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from populate_tasks import get_task_samples_for_analysis

# Core benchmarks from lm-eval-harness v1.0 list
# Mapping display names to actual task names in lm-eval with risk/skill tags
# Tags auto-determined from README content using keyword analysis
CORE_BENCHMARKS = {
    # Benchmark Suites
    "glue": {
        "task": "glue",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "superglue": {
        "task": "superglue",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    # SuperGLUE individual tasks (also available separately)
    "cb": {
        "task": "cb",
        "tags": ["reasoning", "general knowledge", "adversarial robustness"]
    },
    "copa": {
        "task": "copa",
        "tags": ["reasoning", "general knowledge", "creative writing"]
    },
    "multirc": {
        "task": "multirc",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "record": {
        "task": "record",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "wic": {
        "task": "wic",
        "tags": ["reasoning", "general knowledge", "multilingual"]
    },
    "wsc": {
        "task": "wsc",
        "tags": ["reasoning", "general knowledge", "adversarial robustness"]
    },
    
    # Hallucination and Truthfulness
    "truthfulqa_mc1": {
        "task": "truthfulqa_mc1",
        "tags": ["hallucination", "deception", "general knowledge"]
    },
    "truthfulqa_mc2": {
        "task": "truthfulqa_mc2",
        "tags": ["hallucination", "deception", "general knowledge"]
    },
    "truthfulqa_gen": {
        "task": "truthfulqa_gen",
        "tags": ["hallucination", "deception", "general knowledge"]
    },
    
    # Reasoning and Comprehension
    "hellaswag": {
        "task": "hellaswag",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "piqa": {
        "task": "piqa",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "winogrande": {
        "task": "winogrande",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "openbookqa": {
        "task": "openbookqa",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "swag": {
        "task": "swag",
        "tags": ["reasoning", "general knowledge", "creative writing"]
    },
    "storycloze": {
        "task": "storycloze",
        "tags": ["reasoning", "creative writing", "general knowledge"]
    },
    "logiqa": {
        "task": "logiqa",
        "tags": ["reasoning", "general knowledge", "mathematics"]
    },
    "wsc273": {
        "task": "wsc273",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    
    # Reading Comprehension and QA
    "coqa": {
        "task": "coqa",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "drop": {
        "task": "drop",
        "tags": ["reasoning", "mathematics", "general knowledge"]
    },
    "boolq": {
        "task": "boolq",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "race": {
        "task": "race",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "squad2": {
        "task": "squad2",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "triviaqa": {
        "task": "triviaqa",
        "tags": ["general knowledge", "reasoning", "history"]
    },
    "naturalqs": {
        "task": "nq_open",
        "tags": ["general knowledge", "reasoning", "history"]
    },
    "webqs": {
        "task": "webqs",
        "tags": ["general knowledge", "reasoning", "history"]
    },
    "headqa": {
        "task": "headqa",
        "tags": ["medical", "reasoning", "general knowledge"]
    },
    "qasper": {
        "task": "qasper",
        "tags": ["science", "reasoning", "long context"]
    },
    "qa4mre": {
        "task": "qa4mre",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "mutual": {
        "task": "mutual",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    
    # Knowledge and Academic
    "mmlu": {
        "task": "mmlu",
        "tags": ["general knowledge", "science", "reasoning"]
    },
    "ai2_arc": {
        "task": "ai2_arc",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "arc_easy": {
        "task": "arc_easy",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "arc_challenge": {
        "task": "arc_challenge",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "sciq": {
        "task": "sciq",
        "tags": ["science", "reasoning", "general knowledge"]
    },
    "social_i_qa": {
        "task": "social_i_qa",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    
    # Mathematics
    "gsm8k": {
        "task": "gsm8k",
        "tags": ["mathematics", "reasoning", "hallucination"]
    },
    "math_qa": {
        "task": "math_qa",
        "tags": ["mathematics", "reasoning", "general knowledge"]
    },
    "hendrycks_math": {
        "task": "hendrycks_math",
        "tags": ["mathematics", "reasoning", "science"]
    },
    "arithmetic": {
        "task": "arithmetic",
        "tags": ["mathematics", "reasoning", "general knowledge"]
    },
    "asdiv": {
        "task": "asdiv",
        "tags": ["mathematics", "reasoning", "general knowledge"]
    },
    
    # Coding
    "humaneval": {
        "task": "humaneval",
        "tags": ["coding", "reasoning", "mathematics"]
    },
    "mbpp": {
        "task": "mbpp",
        "tags": ["coding", "reasoning", "mathematics"]
    },
    
    # Bias and Toxicity
    "toxigen": {
        "task": "toxigen",
        "tags": ["toxicity", "bias", "harmfulness"]
    },
    "crows_pairs": {
        "task": "crows_pairs",
        "tags": ["bias", "toxicity", "harmfulness"]
    },
    "hendrycks_ethics": {
        "task": "hendrycks_ethics",
        "tags": ["bias", "harmfulness", "general knowledge"]
    },
    
    # Adversarial
    "anli": {
        "task": "anli",
        "tags": ["adversarial robustness", "reasoning", "general knowledge"]
    },
    
    # Multilinguality
    "xnli": {
        "task": "xnli",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "xcopa": {
        "task": "xcopa",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "xstorycloze": {
        "task": "xstorycloze",
        "tags": ["multilingual", "reasoning", "creative writing"]
    },
    "xwinograd": {
        "task": "xwinograd",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "paws_x": {
        "task": "paws_x",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    "mmmlu": {
        "task": "mmmlu",
        "tags": ["multilingual", "general knowledge", "reasoning"]
    },
    "mgsm": {
        "task": "mgsm",
        "tags": ["multilingual", "mathematics", "reasoning"]
    },
    "belebele": {
        "task": "belebele",
        "tags": ["multilingual", "reasoning", "general knowledge"]
    },
    
    # Medical and Law
    "medqa": {
        "task": "medqa",
        "tags": ["medical", "reasoning", "general knowledge"]
    },
    "pubmedqa": {
        "task": "pubmedqa",
        "tags": ["medical", "reasoning", "general knowledge"]
    },
    
    # Language Modeling and Generation
    "lambada": {
        "task": "lambada",
        "tags": ["creative writing", "general knowledge", "reasoning"]
    },
    "lambada_cloze": {
        "task": "lambada_cloze",
        "tags": ["creative writing", "general knowledge", "reasoning"]
    },
    "lambada_multilingual": {
        "task": "lambada_multilingual",
        "tags": ["multilingual", "creative writing", "general knowledge"]
    },
    "wikitext": {
        "task": "wikitext",
        "tags": ["general knowledge", "creative writing", "long context"]
    },
    
    # Long Context
    "narrativeqa": {
        "task": "narrativeqa",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "scrolls": {
        "task": "scrolls",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    
    # Temporal and Event Understanding
    "mctaco": {
        "task": "mctaco",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
    "prost": {
        "task": "prost",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    
    # Linguistic Understanding
    "blimp": {
        "task": "blimp",
        "tags": ["reasoning", "general knowledge", "science"]
    },
    "unscramble": {
        "task": "unscramble",
        "tags": ["reasoning", "general knowledge", "creative writing"]
    },
    
    # Translation
    "wmt": {
        "task": "wmt",
        "tags": ["multilingual", "creative writing", "general knowledge"]
    },
    
    # Comprehensive Suites
    "big_bench": {
        "task": "big_bench",
        "tags": ["general knowledge", "reasoning", "creative writing"]
    },
    
    # Dialogue and Conversation
    "babi": {
        "task": "babi",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
}

def test_single_benchmark_direct(benchmark_name: str, benchmark_config: dict) -> bool:
    """
    Test a single benchmark directly using wisent guard CLI.
    
    Args:
        benchmark_name: Display name of the benchmark
        benchmark_config: Config dict with 'task' and 'tags' keys
    
    Returns:
        True if successful, False otherwise
    """
    task_name = benchmark_config["task"]
    tags = benchmark_config["tags"]
    
    print(f"\n{'='*60}")
    print(f"🎯 Testing: {benchmark_name} ({task_name})")
    print(f"🏷️  Tags: {', '.join(tags)}")
    print(f"{'='*60}")
    
    try:
        # Create output directory for this benchmark
        output_dir = f"test_results/{benchmark_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run wisent guard CLI with tasks command
        cmd = [
            "python", "-m", "wisent_guard.cli",
            "tasks", task_name,
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--layer", "15",
            "--limit", "5",
            "--classifier-type", "logistic",
            "--verbose"
        ]
        
        print(f"🧪 Running command:")
        print(f"   {' '.join(cmd)}")
        
        # Set working directory to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout per benchmark (some are large suites)
            cwd=project_root
        )
        
        # Save full output
        output_file = os.path.join(output_dir, "output.txt")
        with open(output_file, 'w') as f:
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write(f"RETURN CODE: {result.returncode}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")
        
        if result.returncode == 0:
            print(f"✅ Successfully tested {benchmark_name}")
            print(f"📊 Output preview: {result.stdout[:300]}...")
            
            # Look for contrastive pairs in the output
            contrastive_pairs = extract_contrastive_pairs_from_output(result.stdout)
            if contrastive_pairs:
                print(f"🔍 Found {len(contrastive_pairs)} contrastive pairs")
                
                # Save contrastive pairs
                pairs_file = os.path.join(output_dir, "contrastive_pairs.json")
                with open(pairs_file, 'w') as f:
                    json.dump(contrastive_pairs, f, indent=2)
                print(f"💾 Contrastive pairs saved to: {pairs_file}")
            
            return True
        else:
            print(f"❌ Failed to test {benchmark_name} - WILL CAUSE SCRIPT TO EXIT")
            print(f"Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            print(f"💾 Full output saved to: {output_file}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout testing {benchmark_name} - WILL CAUSE SCRIPT TO EXIT")
        return False
    except Exception as e:
        print(f"💥 Exception testing {benchmark_name}: {e} - WILL CAUSE SCRIPT TO EXIT")
        return False

def extract_contrastive_pairs_from_output(output: str) -> List[Dict]:
    """
    Extract contrastive pairs from CLI output.
    
    Args:
        output: The stdout from the CLI command
    
    Returns:
        List of contrastive pairs found in the output
    """
    pairs = []
    lines = output.split('\n')
    
    # Look for patterns indicating contrastive pairs
    for i, line in enumerate(lines):
        # Look for question/answer pairs
        if "Question:" in line or "Prompt:" in line:
            question = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            
            # Look for correct/incorrect responses in following lines
            correct_answer = None
            incorrect_answer = None
            
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j].strip()
                if "Correct:" in next_line or "Good:" in next_line:
                    correct_answer = next_line.split(":", 1)[1].strip() if ":" in next_line else next_line
                elif "Incorrect:" in next_line or "Bad:" in next_line:
                    incorrect_answer = next_line.split(":", 1)[1].strip() if ":" in next_line else next_line
                elif "---" in next_line or next_line == "":
                    break
            
            if question and correct_answer and incorrect_answer:
                pairs.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer
                })
    
    return pairs

def test_benchmark_creation(benchmark_name: str, benchmark_config: dict) -> tuple[bool, list]:
    """
    Test creating a dataset for a benchmark to see if it works.
    
    Args:
        benchmark_name: Display name of the benchmark
        benchmark_config: Config dict with 'task' and 'tags' keys
    
    Returns:
        Tuple of (success: bool, actual_tags: list)
    """
    task_name = benchmark_config["task"]
    tags = benchmark_config["tags"]
    
    print(f"\n🔍 Testing dataset creation for {benchmark_name} ({task_name})...")
    if tags:
        print(f"🏷️  Predefined tags: {', '.join(tags)}")
    else:
        print(f"🏷️  Tags will be auto-determined from README")
    
    try:
        # Get samples using the existing function
        result = get_task_samples_for_analysis(task_name, num_samples=5)
        
        if "error" in result:
            print(f"❌ Error retrieving samples: {result['error']} - WILL CAUSE SCRIPT TO EXIT")
            return False, tags
        
        if not result.get("samples"):
            print(f"❌ No samples found for {task_name} - WILL CAUSE SCRIPT TO EXIT")
            return False, tags
        
        print(f"✅ Successfully retrieved {len(result['samples'])} samples")
        
        # Use README-determined tags if available
        actual_tags = tags
        if 'readme_tags' in result and result['readme_tags']:
            actual_tags = result['readme_tags']
            print(f"🏷️  README-determined tags: {', '.join(actual_tags)}")
        
        # Show a sample
        if result["samples"]:
            sample = result["samples"][0]
            print(f"📋 Sample question: {sample.get('question', '')[:100]}...")
            print(f"✅ Correct answer: {sample.get('correct_answer', '')}")
            if sample.get('choices'):
                print(f"🔤 Choices: {len(sample['choices'])} options")
        
        return True, actual_tags
        
    except Exception as e:
        print(f"💥 Exception testing {benchmark_name}: {e} - WILL CAUSE SCRIPT TO EXIT")
        return False, tags

def main():
    """Main function to test ALL core benchmarks comprehensively."""
    print("🚀 Testing ALL core benchmarks comprehensively")
    print(f"📋 Processing {len(CORE_BENCHMARKS)} comprehensive benchmarks across all categories...")
    print("🎯 Categories: Benchmark Suites, Hallucination, Reasoning, QA/Reading, Knowledge, Math, Coding")
    print("🎯           Bias/Toxicity, Adversarial, Multilingual, Medical, Language Modeling, Long Context")
    print("🎯           Temporal, Linguistic, Translation, Dialogue")
    print("⚠️  FAIL-HARD MODE: Script will exit with code 1 on first benchmark failure!")
    
    # Create output directory
    os.makedirs("test_results", exist_ok=True)
    
    # Track results
    results = {
        "dataset_creation": {"successful": [], "failed": []},
        "cli_testing": {"successful": [], "failed": []},
        "benchmark_tags": {}
    }
    
    total_benchmarks = len(CORE_BENCHMARKS)
    current_idx = 0
    
    for benchmark_name, benchmark_config in CORE_BENCHMARKS.items():
        current_idx += 1
        task_name = benchmark_config["task"]
        tags = benchmark_config["tags"]
        
        # Store benchmark tags for analysis (will be updated with actual tags later)
        results["benchmark_tags"][benchmark_name] = {
            "task": task_name,
            "tags": tags,
            "original_tags": tags
        }
        
        print(f"\n{'='*80}")
        print(f"🎯 BENCHMARK {current_idx}/{total_benchmarks}: {benchmark_name}")
        print(f"{'='*80}")
        
        # Step 1: Test dataset creation
        print(f"\n{'='*80}")
        print(f"🎯 STEP 1: Testing dataset creation for {benchmark_name}")
        print(f"{'='*80}")
        
        dataset_success, actual_tags = test_benchmark_creation(benchmark_name, benchmark_config)
        
        # Update benchmark config with actual tags if they were determined from README
        if actual_tags != benchmark_config["tags"]:
            print(f"🔄 Updated tags from README: {actual_tags}")
            benchmark_config = benchmark_config.copy()
            benchmark_config["tags"] = actual_tags
            # Update results tracking with actual tags
            results["benchmark_tags"][benchmark_name]["tags"] = actual_tags
        
        if dataset_success:
            results["dataset_creation"]["successful"].append(benchmark_name)
            
            # Step 2: Test CLI integration
            print(f"\n{'='*80}")
            print(f"🎯 STEP 2: Testing CLI integration for {benchmark_name}")
            print(f"{'='*80}")
            
            cli_success = test_single_benchmark_direct(benchmark_name, benchmark_config)
            
            if cli_success:
                results["cli_testing"]["successful"].append(benchmark_name)
            else:
                results["cli_testing"]["failed"].append(benchmark_name)
                print(f"\n💥 FATAL ERROR: CLI testing failed for {benchmark_name}")
                print(f"🚨 Script failing hard as requested!")
                print(f"❌ Benchmark: {benchmark_name} ({task_name})")
                print(f"🏷️  Tags: {', '.join(tags)}")
                sys.exit(1)
        else:
            results["dataset_creation"]["failed"].append(benchmark_name)
            results["cli_testing"]["failed"].append(benchmark_name)
            print(f"\n💥 FATAL ERROR: Dataset creation failed for {benchmark_name}")
            print(f"🚨 Script failing hard as requested!")
            print(f"❌ Benchmark: {benchmark_name} ({task_name})")
            print(f"🏷️  Tags: {', '.join(tags)}")
            sys.exit(1)
        
        print(f"\n{'='*80}")
        print(f"📊 CURRENT STATUS")
        print(f"{'='*80}")
        print(f"✅ Dataset creation successful: {len(results['dataset_creation']['successful'])}")
        print(f"❌ Dataset creation failed: {len(results['dataset_creation']['failed'])}")
        print(f"✅ CLI testing successful: {len(results['cli_testing']['successful'])}")
        print(f"❌ CLI testing failed: {len(results['cli_testing']['failed'])}")
        
        # Continue to next benchmark
        print(f"\n✅ Successfully completed testing {benchmark_name}")
        print(f"🔄 Moving to next benchmark...\n")
    
    # Print final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n🔍 Dataset Creation Results:")
    print(f"✅ Successful: {len(results['dataset_creation']['successful'])}")
    for name in results["dataset_creation"]["successful"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")
    
    print(f"\n❌ Failed: {len(results['dataset_creation']['failed'])}")
    for name in results["dataset_creation"]["failed"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")
    
    print(f"\n🧪 CLI Testing Results:")
    print(f"✅ Successful: {len(results['cli_testing']['successful'])}")
    for name in results["cli_testing"]["successful"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")
    
    print(f"\n❌ Failed: {len(results['cli_testing']['failed'])}")
    for name in results["cli_testing"]["failed"]:
        tags = results["benchmark_tags"][name]["tags"]
        print(f"  - {name} ({', '.join(tags)})")
    
    # Save results
    results_file = "test_results/benchmark_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Success message (only reached if all benchmarks pass)
    print(f"\n🎉 SUCCESS! All {len(CORE_BENCHMARKS)} benchmarks passed!")
    print("✅ No failures detected - all benchmarks working with wisent guard CLI")
    print("🚀 Ready for production use!")

if __name__ == "__main__":
    main()
