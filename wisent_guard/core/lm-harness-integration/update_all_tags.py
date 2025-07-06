#!/usr/bin/env python3
"""
Script to update all benchmark tags in only_benchmarks.py using README-based tag determination.
"""

import os
import sys
import json
from typing import Dict, List

# Import the tag determination function
from populate_tasks import get_benchmark_tags_with_llama

def update_benchmark_tags():
    """Update all benchmark tags using README-based determination."""
    
    # Import the benchmarks from only_benchmarks.py
    from only_benchmarks import CORE_BENCHMARKS
    
    print("🔄 Updating ALL benchmark tags using README-based logic...")
    print(f"📋 Processing {len(CORE_BENCHMARKS)} benchmarks...")
    
    updated_benchmarks = {}
    successful_updates = 0
    failed_updates = 0
    
    for benchmark_name, config in CORE_BENCHMARKS.items():
        task_name = config["task"]
        current_tags = config["tags"]
        
        print(f"\n🎯 Processing: {benchmark_name} ({task_name})")
        print(f"   Current tags: {current_tags}")
        
        try:
            # Get README content for better analysis
            import requests
            task_dir_map = {
                'superglue': 'super_glue',
                'super_glue': 'super_glue',
                'super-glue': 'super_glue',
                'glue': 'glue',
                'mmlu': 'mmlu',
                'truthfulqa': 'truthfulqa',
                'hellaswag': 'hellaswag',
                'arc': 'arc',
                'winogrande': 'winogrande'
            }
            task_dir = task_dir_map.get(task_name.lower(), task_name.lower())
            readme_url = f"https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/{task_dir}/README.md"
            
            readme_content = ""
            try:
                response = requests.get(readme_url, timeout=5)
                if response.status_code == 200:
                    readme_content = response.text
            except:
                pass  # Use empty content if README fetch fails
            
            # Use the LLM function to get tags
            new_tags = get_benchmark_tags_with_llama(task_name, readme_content)
            
            print(f"   ✅ Intelligent tags: {new_tags}")
            updated_benchmarks[benchmark_name] = {
                "task": task_name,
                "tags": new_tags
            }
            successful_updates += 1
                
        except Exception as e:
            print(f"   ❌ Error processing {benchmark_name}: {e}")
            print(f"   📌 Keeping current tags: {current_tags}")
            updated_benchmarks[benchmark_name] = {
                "task": task_name,
                "tags": current_tags if current_tags else ["general knowledge", "reasoning", "science"]
            }
            failed_updates += 1
    
    print(f"\n📊 Update Summary:")
    print(f"✅ Successful README updates: {successful_updates}")
    print(f"⚠️  Kept existing/default tags: {failed_updates}")
    
    # Generate the updated CORE_BENCHMARKS dictionary content
    print(f"\n📝 Generating updated only_benchmarks.py content...")
    
    # Create the new CORE_BENCHMARKS content
    benchmarks_content = "CORE_BENCHMARKS = {\n"
    
    # Group benchmarks by category (based on comments in original file)
    categories = {
        "Benchmark Suites": ["glue", "superglue"],
        "SuperGLUE individual tasks": ["cb", "copa", "multirc", "record", "wic", "wsc"],
        "Hallucination and Truthfulness": ["truthfulqa_mc1", "truthfulqa_mc2", "truthfulqa_gen"],
        "Reasoning and Comprehension": ["hellaswag", "piqa", "winogrande", "openbookqa", "swag", "storycloze", "logiqa", "wsc273"],
        "Reading Comprehension and QA": ["coqa", "drop", "boolq", "race", "squad2", "triviaqa", "naturalqs", "webqs", "headqa", "qasper", "qa4mre", "mutual"],
        "Knowledge and Academic": ["mmlu", "ai2_arc", "arc_easy", "arc_challenge", "sciq", "social_i_qa"],
        "Mathematics": ["gsm8k", "math_qa", "hendrycks_math", "arithmetic", "asdiv"],
        "Coding": ["humaneval", "mbpp"],
        "Bias and Toxicity": ["toxigen", "crows_pairs", "hendrycks_ethics"],
        "Adversarial": ["anli"],
        "Multilinguality": ["xnli", "xcopa", "xstorycloze", "xwinograd", "paws_x", "mmmlu", "mgsm", "belebele"],
        "Medical and Law": ["medqa", "pubmedqa"],
        "Language Modeling and Generation": ["lambada", "lambada_cloze", "lambada_multilingual", "wikitext"],
        "Long Context": ["narrativeqa", "scrolls"],
        "Temporal and Event Understanding": ["mctaco", "prost"],
        "Linguistic Understanding": ["blimp", "unscramble"],
        "Translation": ["wmt"],
        "Comprehensive Suites": ["big_bench"],
        "Dialogue and Conversation": ["babi"]
    }
    
    # Add benchmarks by category
    for category, benchmark_names in categories.items():
        benchmarks_content += f"    # {category}\n"
        for benchmark_name in benchmark_names:
            if benchmark_name in updated_benchmarks:
                config = updated_benchmarks[benchmark_name]
                tags_str = ', '.join([f'"{tag}"' for tag in config["tags"]])
                benchmarks_content += f'    "{benchmark_name}": {{\n'
                benchmarks_content += f'        "task": "{config["task"]}",\n'
                benchmarks_content += f'        "tags": [{tags_str}]\n'
                benchmarks_content += f'    }},\n'
        benchmarks_content += "\n"
    
    benchmarks_content = benchmarks_content.rstrip(",\n") + "\n}\n"
    
    # Save the updated content
    output_file = "updated_benchmarks.py"
    with open(output_file, 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\nUpdated benchmark definitions with README-based tags.\n"""\n\n')
        f.write(benchmarks_content)
    
    print(f"💾 Updated benchmarks saved to: {output_file}")
    
    # Show some examples
    print(f"\n📋 Example updates:")
    for i, (name, config) in enumerate(list(updated_benchmarks.items())[:5]):
        print(f"   {name}: {config['tags']}")
    
    return updated_benchmarks

if __name__ == "__main__":
    updated_benchmarks = update_benchmark_tags() 