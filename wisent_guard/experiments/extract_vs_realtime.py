#!/usr/bin/env python3
"""
REAL vs POST-GENERATION ACTIVATION SCORING EXPERIMENT

This experiment tests whether post-generation activation extraction 
produces the same scores as real-time generation scoring.

Approach:
1. Run the CLI command: python -m wisent_guard.cli tasks truthful_qa --model meta-llama/Llama-3.1-8B-Instruct --layer 15 --limit 5 --verbose
2. Extract the generated responses and their real-time scores
3. Re-score the same generated text using post-generation extraction
4. Compare the results
"""

import subprocess
import sys
import os
import re
import torch
import numpy as np
from typing import List, Dict, Tuple

# Add the parent directory to the path so we can import wisent_guard modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wisent_guard.core.model import Model
from wisent_guard.core.layer import Layer


def run_cli_command() -> Tuple[List[Dict], str]:
    """Run the TruthfulQA CLI command and capture the output."""
    print("🚀 Running TruthfulQA CLI command with classifier saving...")
    
    cmd = [
        "python", "-m", "wisent_guard.cli", "tasks", "truthful_qa",
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--layer", "15",
        "--limit", "5", 
        "--verbose",
        "--save-classifier", "trained_classifier.pkl"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"❌ CLI command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return [], result.stdout
        
        print("✅ CLI command completed successfully!")
        return parse_cli_output(result.stdout), result.stdout
        
    except subprocess.TimeoutExpired:
        print("❌ CLI command timed out after 600 seconds")
        return [], ""
    except Exception as e:
        print(f"❌ Error running CLI command: {e}")
        return [], ""


def parse_cli_output(output: str) -> List[Dict]:
    """Parse the CLI output to extract generated responses and scores."""
    print("📊 Parsing CLI output for generated responses...")
    
    results = []
    
    # Look for test results in the output
    lines = output.split('\n')
    
    current_question = None
    current_response = None
    current_scores = None
    collecting_response = False
    response_lines = []
    
    for line in lines:
        # Look for question patterns in the specific CLI format
        if "📝 Question:" in line:
            question_match = re.search(r'📝 Question:\s*(.+)', line)
            if question_match:
                current_question = question_match.group(1).strip()
        
        # Start collecting response
        if "🤖 Generated:" in line:
            response_match = re.search(r'🤖 Generated:\s*(.+)', line)
            if response_match:
                # Start collecting the response
                collecting_response = True
                response_lines = [response_match.group(1).strip()]
        
        # Continue collecting response lines until we hit token scores
        elif collecting_response and not "🔍 Token Scores:" in line:
            # Add non-empty lines to the response
            if line.strip():
                response_lines.append(line.strip())
        
        # Look for token scores - this ends response collection
        elif "🔍 Token Scores:" in line:
            if collecting_response:
                # Join all response lines with newlines to preserve structure
                current_response = '\n'.join(response_lines)
                collecting_response = False
                response_lines = []
            
            scores_match = re.search(r'🔍 Token Scores:\s*\[([^\]]+)\]', line)
            if scores_match:
                try:
                    scores_str = scores_match.group(1)
                    # Parse scores like ['0.618', '0.544', ...]
                    scores = [float(s.strip().strip("'\"")) for s in scores_str.split(',')]
                    current_scores = scores
                except Exception as e:
                    print(f"      ⚠️ Error parsing scores: {e}")
                    pass
        
        # If we have all three pieces, save the result
        if current_question and current_response and current_scores:
            results.append({
                'question': current_question,
                'response': current_response, 
                'realtime_scores': current_scores,
                'avg_realtime_score': np.mean(current_scores)
            })
            
            print(f"      ✅ Captured: Question='{current_question[:30]}...', Response='{current_response[:30]}...', Scores={len(current_scores)} tokens")
            print(f"      📊 Full response text: '{current_response}'")
            print(f"      📊 Response length: {len(current_response)} characters")
            
            # Let's manually tokenize to see how many tokens this truncated text should have
            if current_response:
                temp_model = Model("meta-llama/Llama-3.1-8B-Instruct")
                temp_tokens = temp_model.tokenizer.encode(current_response, return_tensors="pt")
                print(f"      📊 Manual tokenization: {len(temp_tokens[0])} tokens from response text")
                print(f"      📊 CLI claims: {len(current_scores)} scores from 109 total tokens")
                print(f"      🔍 ISSUE: CLI display shows truncated response but scored full 109-token generation!")
                del temp_model
            
            # Reset for next result
            current_question = None
            current_response = None  
            current_scores = None
    
    print(f"   ✅ Extracted {len(results)} test results from CLI output")
    return results


def extract_post_generation_scores(cli_results: List[Dict]) -> List[Dict]:
    """Extract post-generation scores using the EXACT SAME trained classifier."""
    print("🔍 Loading the EXACT same classifier that CLI trained and saved...")
    
    # Load the same model
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    layer = 15
    
    # Load the exact same classifier that the CLI trained and saved
    from wisent_guard.core.steering import SteeringMethod, SteeringType
    steering_method = SteeringMethod(
        method_type=SteeringType.LOGISTIC,
        device=model.device
    )
    
    # Load the saved classifier
    import pickle
    with open("trained_classifier_pkl_layer_15.pkl", "rb") as f:
        steering_method = pickle.load(f)
    
    print(f"   ✅ Loaded classifier from trained_classifier_pkl_layer_15.pkl")
    print(f"   🔍 Loaded object type: {type(steering_method)}")
    if isinstance(steering_method, dict):
        print(f"   🔍 Dict keys: {list(steering_method.keys())}")
        if 'classifier' in steering_method:
            print(f"   🔍 Classifier type: {type(steering_method['classifier'])}")
    else:
        print(f"   🔍 Object attributes: {dir(steering_method)}")
    
    # Now use this EXACT same classifier for post-generation scoring
    post_gen_results = []
    
    for i, result in enumerate(cli_results):
        question = result['question']
        response = result['response']
        
        print(f"   📋 Processing result {i+1}: {question[:50]}...")
        
        try:
            # Use the EXACT same methodology as generate_with_classification
            from wisent_guard.inference import generate_with_classification
            
            # Instead of generating, we'll extract activations token by token from existing text
            full_text = f"{question}{response}"
            
            # Tokenize exactly like the real-time method
            inputs = model.tokenizer(full_text, return_tensors="pt")
            tokens = model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Get prompt length
            prompt_inputs = model.tokenizer(question, return_tensors="pt")
            prompt_length = len(prompt_inputs['input_ids'][0])
            
            post_token_scores = []
            
            # Get model device and process inputs (exact same as generate_with_classification)
            model_device = next(model.hf_model.parameters()).device
            inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Get hidden states
            with torch.no_grad():
                outputs = model.hf_model(**inputs_on_device, output_hidden_states=True)
            
            # Extract from same layer
            if layer + 1 < len(outputs.hidden_states):
                hidden_states = outputs.hidden_states[layer + 1]
            else:
                hidden_states = outputs.hidden_states[-1]
            
            # Score each response token using EXACT same methodology as generate_with_classification
            for token_idx in range(prompt_length, min(len(tokens), hidden_states.shape[1])):
                # Extract activation
                token_activation = hidden_states[0, token_idx, :].cpu()
                
                # Create Activations object
                from wisent_guard.core.activations import Activations, ActivationAggregationMethod
                layer_obj = Layer(index=layer, type="transformer")
                
                activation_obj = Activations(
                    tensor=token_activation.unsqueeze(0),
                    layer=layer_obj,
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                )
                
                # Get features
                features = activation_obj.extract_features_for_classifier()
                
                # DEBUG: Compare activation extraction for first few tokens
                if token_idx - prompt_length < 3:
                    print(f"         🔍 Token {token_idx - prompt_length}: activation_shape={token_activation.shape}, norm={torch.norm(token_activation):.3f}")
                    print(f"         🔍 Features shape={features.shape}, norm={torch.norm(features):.3f}, mean={features.mean():.3f}")
                
                # Handle the loaded classifier correctly (it's a dict, not SteeringMethod)
                if isinstance(steering_method, dict) and 'classifier' in steering_method:
                    classifier = steering_method['classifier']
                    try:
                        prob = classifier.predict_proba([features.numpy()])
                        if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) > 0:
                            if hasattr(prob[0], '__len__') and len(prob[0]) > 1:
                                # Binary classification - take positive class probability
                                prob = float(prob[0][1])
                            else:
                                prob = float(prob[0])
                        else:
                            prob = float(prob)
                        post_token_scores.append(prob)
                        
                        # DEBUG: Show classifier output for first few tokens
                        if token_idx - prompt_length < 3:
                            print(f"         🔍 Classifier output: {prob:.3f}")
                            
                    except Exception as e:
                        print(f"         ⚠️ Classifier error: {e}")
                        post_token_scores.append(0.5)
                elif hasattr(steering_method, 'classifier') and steering_method.classifier:
                    # Handle SteeringMethod object
                    try:
                        prob = steering_method.classifier.predict_proba([features.numpy()])
                        if isinstance(prob, (list, tuple, np.ndarray)) and len(prob) > 0:
                            if hasattr(prob[0], '__len__') and len(prob[0]) > 1:
                                prob = float(prob[0][1])
                            else:
                                prob = float(prob[0])
                        else:
                            prob = float(prob)
                        post_token_scores.append(prob)
                    except Exception as e:
                        print(f"         ⚠️ Classifier error: {e}")
                        post_token_scores.append(0.5)
                else:
                    print(f"         ⚠️ No classifier found in loaded object")
                    post_token_scores.append(0.5)
            
            post_gen_results.append({
                'question': question,
                'response': response,
                'post_gen_scores': post_token_scores,
                'avg_post_gen_score': np.mean(post_token_scores) if post_token_scores else 0
            })
            
            print(f"      ✅ Extracted {len(post_token_scores)} token scores using SAME classifier")
            print(f"      📊 First 5 post-gen scores: {[f'{s:.3f}' for s in post_token_scores[:5]]}")
            print(f"      📊 Token positions: prompt_length={prompt_length}, total_tokens={len(tokens)}, response_tokens={len(tokens)-prompt_length}")
            
            # DEBUG: Check if we're using the exact same classifier object
            print(f"      🔍 Classifier type: {type(steering_method.classifier)}")
            print(f"      🔍 Classifier trained: {hasattr(steering_method.classifier, 'predict_proba')}")
            
            # DEBUG: Check first few feature vectors
            if len(post_token_scores) >= 3:
                print(f"      🔍 Checking feature extraction for first 3 tokens...")
                for debug_idx in range(min(3, len(tokens) - prompt_length)):
                    token_idx = prompt_length + debug_idx
                    if token_idx < hidden_states.shape[1]:
                        token_activation = hidden_states[0, token_idx, :].cpu()
                        activation_obj = Activations(
                            tensor=token_activation.unsqueeze(0),
                            layer=layer_obj,
                            aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                        )
                        features = activation_obj.extract_features_for_classifier()
                        print(f"         Token {debug_idx}: shape={features.shape}, norm={torch.norm(features):.3f}, mean={features.mean():.3f}")
            
            # DEBUG: Compare question formatting
            print(f"      🔍 Question used: '{question[:100]}...'")
            print(f"      🔍 Full text length: {len(full_text)} chars")
            print(f"      🔍 Tokenization: {len(tokens)} tokens, prompt={prompt_length}, response={len(tokens)-prompt_length}")
            
        except Exception as e:
            print(f"      ❌ Error extracting scores: {e}")
            import traceback
            traceback.print_exc()
            post_gen_results.append({
                'question': question,
                'response': response, 
                'post_gen_scores': [],
                'avg_post_gen_score': 0
            })
    
    return post_gen_results


def compare_results(cli_results: List[Dict], post_gen_results: List[Dict]):
    """Compare real-time vs post-generation results."""
    print("\n📊 COMPARING REAL-TIME vs POST-GENERATION RESULTS")
    print("=" * 60)
    
    for i, (cli_result, pg_result) in enumerate(zip(cli_results, post_gen_results)):
        print(f"\n📋 Result {i+1}:")
        print(f"   Question: {cli_result['question'][:60]}...")
        print(f"   Response: {cli_result['response'][:60]}...")
        
        rt_scores = cli_result['realtime_scores']
        pg_scores = pg_result['post_gen_scores']
        
        if rt_scores and pg_scores:
            min_len = min(len(rt_scores), len(pg_scores))
            
            if min_len > 0:
                rt_subset = rt_scores[:min_len]
                pg_subset = pg_scores[:min_len]
                
                # Calculate metrics
                differences = [abs(r - p) for r, p in zip(rt_subset, pg_subset)]
                avg_diff = np.mean(differences)
                correlation = np.corrcoef(rt_subset, pg_subset)[0, 1] if min_len > 1 else 1.0
                
                print(f"   📊 Token scores comparison ({min_len} tokens):")
                print(f"      • Real-time: {[f'{s:.3f}' for s in rt_subset[:5]]}{'...' if len(rt_subset) > 5 else ''}")
                print(f"      • Post-gen:  {[f'{s:.3f}' for s in pg_subset[:5]]}{'...' if len(pg_subset) > 5 else ''}")
                print(f"      • Raw differences: {[f'{abs(r-p):.3f}' for r, p in zip(rt_subset[:5], pg_subset[:5])]}")
                print(f"      • Avg difference: {avg_diff:.6f}")
                print(f"      • Correlation: {correlation:.6f}")
                
                # Verdict
                if avg_diff < 0.001 and correlation > 0.99:
                    print("      ✅ EXCELLENT: Nearly identical!")
                elif avg_diff < 0.1 and correlation > 0.8:
                    print("      ⚠️  GOOD: Mostly consistent")
                else:
                    print("      ❌ POOR: Significant differences")
        else:
            print("   ❌ Cannot compare - missing scores")


def main():
    """Main experiment function."""
    print("🧪 REAL-TIME vs POST-GENERATION ACTIVATION SCORING EXPERIMENT")
    print("=" * 70)
    print("Testing whether post-generation extraction matches real-time scoring")
    print()
    
    try:
        # Step 1: Run CLI command to get real-time results
        cli_results, raw_output = run_cli_command()
        
        if not cli_results:
            print("❌ No results extracted from CLI output")
            print("\n📄 Raw CLI output:")
            print(raw_output)
            return
        
        # Step 2: Extract post-generation scores for the same text
        post_gen_results = extract_post_generation_scores(cli_results)
        
        # Step 3: Compare results
        compare_results(cli_results, post_gen_results)
        
        print("\n🎉 Experiment completed!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
