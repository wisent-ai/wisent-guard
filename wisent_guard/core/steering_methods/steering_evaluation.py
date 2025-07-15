"""
Steering evaluation module for lm-harness integration.
Provides functions to evaluate steering methods using the lm-harness framework.
"""

import torch
from typing import Dict, List, Any, Optional


def run_lm_harness_evaluation(task_data, test_qa_pairs, model, steering_methods, layers, steering_strength=1.0, use_test_split=True, verbose=False, output_mode="both"):
    """
    Run lm-harness evaluation on test data with steering applied.
    Now includes comparison between unsteered baseline and steered results.
    
    Args:
        output_mode: str, one of "likelihoods", "responses", or "both"
                    - "likelihoods": only show log-likelihood comparison
                    - "responses": only show response generation comparison  
                    - "both": show both comparisons (default)
    """
    
    import torch
    from lm_eval import evaluate
    from lm_eval.api.model import LM
    
    try:
        class SteeredModelWrapper(LM):
            """Wrapper to make wisent-guard model compatible with lm-harness evaluation."""
            
            def __init__(self, wisent_model, steering_methods, layers, steering_strength=1.0):
                self.wisent_model = wisent_model
                self.steering_methods = steering_methods if steering_methods else []
                self.layers = layers if layers else []
                self.steering_strength = steering_strength
                self._rank = 0
                self._world_size = 1
                
            @property
            def eot_token_id(self):
                return self.wisent_model.tokenizer.eos_token_id
            
            @property
            def max_length(self):
                return 2048  # Reasonable default
            
            @property
            def max_gen_toks(self):
                return 256  # For text generation
            
            @property
            def batch_size(self):
                return 1  # Process one at a time for now
            
            @property
            def device(self):
                return self.wisent_model.device
                
            def tok_encode(self, string: str):
                return self.wisent_model.tokenizer.encode(string, add_special_tokens=False)
            
            def tok_decode(self, tokens):
                return self.wisent_model.tokenizer.decode(tokens)
            
            def generate_until(self, requests, disable_tqdm: bool = False):
                """Generate text until stopping condition met."""
                if not requests:
                    return []
                
                results = []
                for request in requests:
                    try:
                        # Extract context and generate
                        context = request.args[0] if request.args else ""
                        
                        # Use the model's generate method
                        inputs = self.wisent_model.tokenizer(context, return_tensors="pt").to(self.wisent_model.device)
                        
                        with torch.no_grad():
                            outputs = self.wisent_model.model.generate(
                                **inputs,
                                max_new_tokens=100,
                                do_sample=False,
                                pad_token_id=self.wisent_model.tokenizer.eos_token_id
                            )
                        
                        # Decode only the generated part
                        generated_text = self.wisent_model.tokenizer.decode(
                            outputs[0][inputs.input_ids.shape[1]:], 
                            skip_special_tokens=True
                        )
                        
                        results.append(generated_text)
                        
                    except Exception as e:
                        if verbose:
                            print(f"   ⚠️ Generation failed for prompt: {e}")
                        results.append("Generation failed")
                
                return results
                
            def loglikelihood(self, requests, disable_tqdm: bool = False):
                """Compute log-likelihood with steering applied."""
                if not requests:
                    return []
                
                print(f"\n🔍 LOGLIKELIHOOD CALLED with {len(requests)} requests")
                print(f"   Steering methods: {len(self.steering_methods) if self.steering_methods else 0}")
                print(f"   Steering strength: {self.steering_strength}")
                
                # Extract request arguments like other lm-harness models  
                _requests = [req.args for req in requests]
                results = []
                
                for i, request in enumerate(_requests):
                    print(f"\n   📊 Processing request {i+1}/{len(_requests)}")
                    try:
                        # Extract context and continuation from the request
                        if len(request) >= 2:
                            context = request[0] if request[0] is not None else ""
                            continuation = request[1]
                            print(f"      Context: '{context[:50] if context else '(empty)'}...'")
                            print(f"      Continuation: '{continuation}'")
                        else:
                            # Fallback parsing
                            context = ""
                            continuation = request[0] if request else ""
                            print(f"      Fallback parsing - continuation only: '{continuation}'")
                        
                        # Combine context and continuation
                        full_text = context + continuation
                        
                        # Tokenize the input
                        tokenizer = self.wisent_model.tokenizer
                        
                        # Tokenize context and full text separately to find continuation tokens
                        context_tokens = tokenizer.encode(context, add_special_tokens=False) if context else []
                        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
                        
                        # The continuation tokens are the difference
                        continuation_tokens = full_tokens[len(context_tokens):]
                        
                        print(f"      🔍 TOKENIZATION DEBUG:")
                        print(f"         Context: '{context[:50]}...' -> {len(context_tokens)} tokens")
                        print(f"         Continuation: '{continuation[:50]}...' -> expected {len(continuation_tokens)} tokens")
                        print(f"         Full text: '{full_text[:50]}...' -> {len(full_tokens)} tokens")
                        print(f"         First 5 context tokens: {context_tokens[:5]}")
                        print(f"         First 5 full tokens: {full_tokens[:5]}")
                        print(f"         Continuation tokens: {continuation_tokens}")
                        
                        if not continuation_tokens:
                            results.append((float('-inf'), False))
                            continue
                        
                        # Convert to tensor
                        input_ids = torch.tensor([full_tokens], device=self.wisent_model.device)
                        
                        # Apply steering if needed
                        if self.steering_methods and self.layers:
                            print(f"   🎯 APPLYING STEERING")
                            print(f"      Methods: {len(self.steering_methods)}, layers: {self.layers}, strength: {self.steering_strength}")
                            for i, method in enumerate(self.steering_methods):
                                print(f"      Method {i}: {type(method).__name__}, trained: {getattr(method, 'is_trained', 'unknown')}")
                                if hasattr(method, 'steering_vector') and method.steering_vector is not None:
                                    print(f"      Vector shape: {method.steering_vector.shape}, norm: {torch.norm(method.steering_vector).item():.4f}")
                            
                            hooks = []
                            try:
                                print(f"      🔧 Registering {len(self.layers)} hooks for layers: {self.layers}")
                                
                                # Create a closure to capture steering method and strength
                                def create_steering_hook(steering_method):
                                    def steering_hook(module, input, output):
                                        try:
                                            # Handle different output formats from transformer layers
                                            if isinstance(output, tuple):
                                                # Transformer layers often return (hidden_states, attention_weights)
                                                hidden_states = output[0]
                                            else:
                                                hidden_states = output
                                            
                                            if verbose:
                                                print(f"   🔍 Hook called at layer {layer_idx}, output type: {type(output)}, hidden_states shape: {hidden_states.shape if hasattr(hidden_states, 'shape') else 'no shape'}")
                                            
                                            if hasattr(hidden_states, 'shape') and len(hidden_states.shape) >= 3:
                                                batch_size, seq_len, hidden_dim = hidden_states.shape
                                                
                                                if verbose:
                                                    print(f"   🔍 Hidden states shape valid: [{batch_size}, {seq_len}, {hidden_dim}]")
                                                
                                                # Handle KSteering differently from vector-based methods
                                                if steering_method.__class__.__name__ == 'KSteering':
                                                    if verbose:
                                                        print(f"   🔍 KSteering detected, applying gradient-based steering")
                                                    
                                                    # KSteering uses apply_steering method directly
                                                    steered_hidden_states = steering_method.apply_steering(hidden_states, strength=self.steering_strength)
                                                    
                                                    if verbose:
                                                        print(f"   🎯 Successfully applied KSteering at layer {layer_idx}, strength={self.steering_strength}")
                                                    
                                                    # Return the appropriate format (tuple or tensor)
                                                    if isinstance(output, tuple):
                                                        return (steered_hidden_states,) + output[1:]
                                                    else:
                                                        return steered_hidden_states
                                                    
                                                # Handle vector-based steering methods (CAA, HPR, DAC, BiPO)
                                                elif hasattr(steering_method, 'steering_vector') and steering_method.steering_vector is not None:
                                                    steering_vector = steering_method.steering_vector
                                                    if verbose:
                                                        print(f"   🔍 Steering vector found, shape: {steering_vector.shape}")
                                                    
                                                    if steering_vector.shape[-1] == hidden_dim:
                                                        print(f"         🔄 HOOK CALLED at layer {layer_idx}")
                                                        print(f"         Hidden states shape: {hidden_states.shape}")
                                                        print(f"         Hidden norm before: {torch.norm(hidden_states, dim=-1).mean().item():.4f}")
                                                        
                                                        # Apply steering to the hidden states
                                                        steered_hidden_states = steering_method.apply_steering(hidden_states, strength=self.steering_strength)
                                                        
                                                        print(f"         Hidden norm after: {torch.norm(steered_hidden_states, dim=-1).mean().item():.4f}")
                                                        print(f"         Applied strength: {self.steering_strength}")
                                                        
                                                        # Check for extreme values
                                                        if torch.any(torch.isnan(steered_hidden_states)) or torch.any(torch.isinf(steered_hidden_states)):
                                                            print(f"         ⚠️ WARNING: Steered hidden states contain NaN or Inf values!")
                                                            print(f"         Max value: {steered_hidden_states.max().item()}")
                                                            print(f"         Min value: {steered_hidden_states.min().item()}")
                                                        
                                                        # Return the appropriate format (tuple or tensor)
                                                        if isinstance(output, tuple):
                                                            # Return tuple with steered hidden states
                                                            return (steered_hidden_states,) + output[1:]
                                                        else:
                                                            return steered_hidden_states
                                                    else:
                                                        if verbose:
                                                            print(f"   ⚠️ Dimension mismatch: vector {steering_vector.shape[-1]} vs hidden {hidden_dim}")
                                                else:
                                                    if verbose:
                                                        print(f"   ⚠️ No steering vector found on method {steering_method.__class__.__name__}")
                                            else:
                                                if verbose:
                                                    print(f"   ⚠️ Invalid hidden states shape for steering")
                                            
                                            # Return original output if no steering applied
                                            return output
                                        except Exception as e:
                                            if verbose:
                                                print(f"   ⚠️ Steering hook failed: {e}")
                                            return output
                                    return steering_hook
                                    
                                # Add hooks directly for now (TODO: use model primitive when available)
                                steering_hook_fn = create_steering_hook(self.steering_methods[0])
                                for layer_idx in self.layers:
                                    # Handle different model architectures
                                    if hasattr(self.wisent_model.model, 'transformer'):
                                        # GPT2 style
                                        layer = self.wisent_model.model.transformer.h[layer_idx]
                                    elif hasattr(self.wisent_model.model, 'model'):
                                        # LLaMA style
                                        layer = self.wisent_model.model.model.layers[layer_idx]
                                    else:
                                        raise ValueError(f"Unknown model architecture for {type(self.wisent_model.model)}")
                                    
                                    hook = layer.register_forward_hook(steering_hook_fn)
                                    hooks.append(hook)
                                
                                print(f"      ✅ {len(hooks)} hooks registered")
                                
                                # Forward pass with steering
                                with torch.no_grad():
                                    outputs = self.wisent_model.model(input_ids)
                                    logits = outputs.logits
                                    
                                # Check logits immediately after forward pass
                                print(f"      🔍 LOGITS CHECK (after steering):")
                                print(f"         Shape: {logits.shape}")
                                print(f"         Contains inf: {torch.any(torch.isinf(logits)).item()}")
                                print(f"         Contains nan: {torch.any(torch.isnan(logits)).item()}")
                                print(f"         Max: {logits.max().item():.2f}")
                                print(f"         Min: {logits.min().item():.2f}")
                                print(f"         Mean: {logits.mean().item():.2f}")
                                
                            except Exception as e:
                                print(f"      ❌ ERROR in steering setup: {e}")
                                import traceback
                                traceback.print_exc()
                            finally:
                                # Remove hooks
                                print(f"      Removing {len(hooks)} hooks")
                                for hook in hooks:
                                    hook.remove()
                                print(f"      Hooks removed")
                        else:
                            # Forward pass without steering
                            with torch.no_grad():
                                outputs = self.wisent_model.model(input_ids)
                                logits = outputs.logits
                                
                            print(f"      🔍 LOGITS CHECK (no steering):")
                            print(f"         Shape: {logits.shape}")
                            print(f"         Contains inf: {torch.any(torch.isinf(logits)).item()}")
                            print(f"         Contains nan: {torch.any(torch.isnan(logits)).item()}")
                            print(f"         Max: {logits.max().item():.2f}")
                            print(f"         Min: {logits.min().item():.2f}")
                            print(f"         Mean: {logits.mean().item():.2f}")
                        
                        # Compute log-likelihood for continuation tokens
                        # We need the logits for positions corresponding to continuation tokens
                        continuation_start = len(context_tokens)
                        continuation_end = len(full_tokens)
                        
                        print(f"\n   🔍 LOGLIKELIHOOD DEBUG:")
                        print(f"      Context length: {len(context_tokens)}")
                        print(f"      Full length: {len(full_tokens)}")
                        print(f"      Continuation length: {len(continuation_tokens)}")
                        print(f"      Logits shape: {logits.shape}")
                        
                        if continuation_start >= logits.shape[1]:
                            print(f"      ❌ Continuation start {continuation_start} >= logits seq length {logits.shape[1]}")
                            print(f"      Returning -inf")
                            results.append((float('-inf'), False))
                            continue
                        
                        # Get logits for the continuation positions (shifted by 1 for next-token prediction)
                        target_logits = logits[0, continuation_start-1:continuation_end-1]  # Shape: [cont_len, vocab_size]
                        target_tokens = torch.tensor(continuation_tokens, device=self.wisent_model.device)
                        
                        print(f"      Target logits shape: {target_logits.shape}")
                        print(f"      Target tokens shape: {target_tokens.shape}")
                        print(f"      Target tokens: {target_tokens[:5].tolist()}")
                        
                        # Check for extreme logits
                        print(f"      Logits max: {target_logits.max().item():.2f}")
                        print(f"      Logits min: {target_logits.min().item():.2f}")
                        print(f"      Logits mean: {target_logits.mean().item():.2f}")
                        print(f"      Logits std: {target_logits.std().item():.2f}")
                        
                        # Check if logits contain extreme values
                        if torch.any(torch.isinf(target_logits)) or torch.any(torch.isnan(target_logits)):
                            print(f"      ⚠️ CRITICAL: Target logits already contain inf/nan!")
                            print(f"      Inf count in logits: {torch.isinf(target_logits).sum().item()}")
                            print(f"      NaN count in logits: {torch.isnan(target_logits).sum().item()}")
                            
                            # Find which positions have inf/nan
                            inf_positions = torch.where(torch.isinf(target_logits))
                            nan_positions = torch.where(torch.isnan(target_logits))
                            print(f"      Inf positions: {inf_positions}")
                            print(f"      NaN positions: {nan_positions}")
                        
                        # Compute log probabilities
                        print(f"      Computing log_softmax...")
                        log_probs = torch.log_softmax(target_logits, dim=-1)
                        
                        # Check for inf/nan in log_probs
                        if torch.any(torch.isinf(log_probs)) or torch.any(torch.isnan(log_probs)):
                            print(f"      ⚠️ Log probs contain inf/nan!")
                            print(f"      Inf count: {torch.isinf(log_probs).sum().item()}")
                            print(f"      NaN count: {torch.isnan(log_probs).sum().item()}")
                        
                        # Get log probability for each target token
                        print(f"      Extracting token log probs for tokens: {target_tokens[:5].tolist()}")
                        token_log_probs = log_probs[range(len(continuation_tokens)), target_tokens]
                        
                        print(f"      Token log probs: {token_log_probs[:5].tolist()}")
                        
                        # Check each token probability individually
                        for idx, (token_id, log_prob) in enumerate(zip(target_tokens[:5], token_log_probs[:5])):
                            print(f"      Token {idx}: id={token_id}, log_prob={log_prob:.4f}, finite={torch.isfinite(log_prob).item()}")
                        
                        # Sum log probabilities (since log(a*b) = log(a) + log(b))
                        total_log_likelihood = token_log_probs.sum().item()
                        
                        print(f"      Total log likelihood: {total_log_likelihood}")
                        
                        results.append((total_log_likelihood, False))
                        
                    except Exception as e:
                        # Return very low likelihood for failed computations
                        results.append((float('-inf'), False))
                        
                return results
                
            def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
                """Rolling log-likelihood computation."""
                if not requests:
                    return []
                return [0.0] * len(requests)
        
        # STEP 1: Run evaluation WITHOUT steering for baseline
        if verbose:
            print(f"🔍 STEP 1: Getting unsteered baseline log-likelihoods...")
            print(f"   Number of test pairs: {len(test_qa_pairs)}")
            print(f"   Task: {task_data.config.task}")
        
        unsteered_model = SteeredModelWrapper(model, [], [], 0.0)  # No steering
        print(f"   Created unsteered model with {len(unsteered_model.steering_methods)} methods, strength={unsteered_model.steering_strength}")
        task_dict = {task_data.config.task: task_data}
        
        baseline_results = evaluate(
            unsteered_model,
            task_dict,
            limit=len(test_qa_pairs),
            bootstrap_iters=0
        )
        
        # Extract baseline log-likelihoods
        baseline_samples = baseline_results.get('samples', {}).get(task_data.config.task, [])
        baseline_likelihoods = []
        if baseline_samples:
            sample = baseline_samples[0]  # First (and only) sample
            baseline_likelihoods = [resp[0][0] for resp in sample.get('resps', [])]
            if verbose:
                print(f"   Baseline sample count: {len(baseline_samples)}")
                print(f"   First baseline likelihoods: {baseline_likelihoods[:3] if baseline_likelihoods else 'None'}")
                print(f"   Sample ID/hash: {id(sample)}")  # Check if it's the same object
        
        # STEP 2: Run evaluation WITH steering  
        if verbose:
            print(f"🔍 STEP 2: Getting steered log-likelihoods...")
            print(f"   Steering methods: {[m.__class__.__name__ for m in steering_methods] if steering_methods else 'None'}")
            print(f"   Steering layers: {layers}")
            print(f"   Steering strength: {steering_strength}")
            if steering_methods and len(steering_methods) > 0:
                print(f"   First method has vector: {hasattr(steering_methods[0], 'steering_vector') and steering_methods[0].steering_vector is not None}")
                if hasattr(steering_methods[0], 'steering_vector') and steering_methods[0].steering_vector is not None:
                    print(f"   Vector shape: {steering_methods[0].steering_vector.shape}")
                    print(f"   Vector norm: {torch.norm(steering_methods[0].steering_vector).item():.4f}")
        
        steered_model = SteeredModelWrapper(model, steering_methods, layers, steering_strength)
        
        steered_results = evaluate(
            steered_model,
            task_dict,
            limit=len(test_qa_pairs),
            bootstrap_iters=0
        )
        
        # Extract steered log-likelihoods
        steered_samples = steered_results.get('samples', {}).get(task_data.config.task, [])
        steered_likelihoods = []
        if steered_samples:
            sample = steered_samples[0]  # First (and only) sample
            steered_likelihoods = [resp[0][0] for resp in sample.get('resps', [])]
        
        # STEP 3: Compare log-likelihoods
        if verbose and baseline_likelihoods and steered_likelihoods and output_mode in ["likelihoods", "both"]:
            print(f"\n📊 LOG-LIKELIHOOD COMPARISON:")
            print(f"   Steering Strength: {steering_strength}")
            print(f"   Method: {steering_methods[0].__class__.__name__ if steering_methods else 'None'}")
            print(f"   Layer: {layers}")
            print(f"\n   Question: {baseline_samples[0]['doc']['question'] if baseline_samples else 'N/A'}")
            
            # Get answer choices from the sample
            if baseline_samples:
                doc = baseline_samples[0]['doc']
                choices = doc.get('mc1_targets', {}).get('choices', [])
                labels = doc.get('mc1_targets', {}).get('labels', [])
                
                print(f"\n   📈 Log-Likelihood Changes:")
                for i, (baseline, steered) in enumerate(zip(baseline_likelihoods, steered_likelihoods)):
                    change = steered - baseline
                    is_correct = labels[i] == 1 if i < len(labels) else False
                    status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                    choice_text = choices[i][:60] + "..." if i < len(choices) and len(choices[i]) > 60 else choices[i] if i < len(choices) else f"Choice {i}"
                    
                    print(f"      {chr(65+i)}. {choice_text}")
                    print(f"         Unsteered: {baseline:8.2f}")
                    print(f"         Steered:   {steered:8.2f}")
                    print(f"         Change:    {change:8.2f} ({'+' if change > 0 else ''}{change:6.2f}) {status}")
                    print()
                
                # Find which answer the model prefers
                unsteered_best = baseline_likelihoods.index(max(baseline_likelihoods))
                steered_best = steered_likelihoods.index(max(steered_likelihoods))
                
                print(f"   🎯 Model Preferences:")
                print(f"      Unsteered model prefers: {chr(65+unsteered_best)} (likelihood: {max(baseline_likelihoods):.2f})")
                print(f"      Steered model prefers:   {chr(65+steered_best)} (likelihood: {max(steered_likelihoods):.2f})")
                
                if unsteered_best != steered_best:
                    print(f"      🔄 Steering changed preference from {chr(65+unsteered_best)} to {chr(65+steered_best)}")
                else:
                    print(f"      ➡️ Steering maintained preference for {chr(65+steered_best)}")
        
        # STEP 4: Generate full responses for comparison
        if verbose and baseline_samples and output_mode in ["responses", "both"]:
            print(f"\n🤖 RESPONSE GENERATION COMPARISON:")
            
            doc = baseline_samples[0]['doc']
            question = doc['question']
            
            # Create a simple prompt for generation
            generation_prompt = f"Q: {question}\nA:"
            
            print(f"   Question: {question}")
            print(f"   Prompt: {generation_prompt}")
            
            try:
                # Generate unsteered response
                print(f"\n   🔍 Generating unsteered response...")
                inputs = model.tokenizer(generation_prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    unsteered_outputs = model.model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=model.tokenizer.eos_token_id,
                        eos_token_id=model.tokenizer.eos_token_id
                    )
                
                unsteered_response = model.tokenizer.decode(
                    unsteered_outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                # Generate steered response
                print(f"   🎯 Generating steered response (strength {steering_strength})...")
                
                # Apply steering hooks during generation
                hooks = []
                try:
                    if steering_methods and layers:
                        # Create a closure to capture steering method and strength
                        def create_steering_hook(steering_method):
                            def steering_hook(module, input, output):
                                try:
                                    # Handle different output formats from transformer layers
                                    if isinstance(output, tuple):
                                        hidden_states = output[0]
                                    else:
                                        hidden_states = output
                                    
                                    if hasattr(hidden_states, 'shape') and len(hidden_states.shape) >= 3:
                                        # Handle KSteering differently from vector-based methods
                                        if steering_method.__class__.__name__ == 'KSteering':
                                            # KSteering uses apply_steering method directly
                                            steered_hidden_states = steering_method.apply_steering(hidden_states, strength=steering_strength)
                                            
                                            # Return the appropriate format (tuple or tensor)
                                            if isinstance(output, tuple):
                                                return (steered_hidden_states,) + output[1:]
                                            else:
                                                return steered_hidden_states
                                        
                                        # Handle vector-based steering methods (CAA, HPR, DAC, BiPO)
                                        elif hasattr(steering_method, 'steering_vector') and steering_method.steering_vector is not None:
                                            steering_vector = steering_method.steering_vector
                                            
                                            if steering_vector.shape[-1] == hidden_states.shape[-1]:
                                                # Apply steering to the hidden states
                                                steered_hidden_states = steering_method.apply_steering(hidden_states, strength=steering_strength)
                                                
                                                # Return the appropriate format (tuple or tensor)
                                                if isinstance(output, tuple):
                                                    return (steered_hidden_states,) + output[1:]
                                                else:
                                                    return steered_hidden_states
                                    
                                    return output
                                except Exception as e:
                                    if verbose:
                                        print(f"      ⚠️ Steering hook failed during generation: {e}")
                                    return output
                                return steering_hook
                            
                        # Add hooks directly for now (TODO: use model primitive when available)
                        steering_hook_fn = create_steering_hook(steering_methods[0])
                        for layer_idx in layers:
                            # Handle different model architectures
                            if hasattr(model.model, 'transformer'):
                                # GPT2 style
                                layer = model.model.transformer.h[layer_idx]
                            elif hasattr(model.model, 'model'):
                                # LLaMA style
                                layer = model.model.model.layers[layer_idx]
                            else:
                                raise ValueError(f"Unknown model architecture for {type(model.model)}")
                            
                            hook = layer.register_forward_hook(steering_hook_fn)
                            hooks.append(hook)
                    
                    with torch.no_grad():
                        steered_outputs = model.model.generate(
                            **inputs,
                            max_new_tokens=150,
                            do_sample=False,
                            temperature=1.0,
                            pad_token_id=model.tokenizer.eos_token_id,
                            eos_token_id=model.tokenizer.eos_token_id
                        )
                    
                    steered_response = model.tokenizer.decode(
                        steered_outputs[0][inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                finally:
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
                
                # Compare responses
                print(f"\n   📝 Response Comparison:")
                print(f"   🔹 Unsteered Response:")
                print(f"      {unsteered_response}")
                print(f"\n   🎯 Steered Response (strength {steering_strength}):")
                print(f"      {steered_response}")
                
                # Analyze differences
                if unsteered_response.lower().strip() == steered_response.lower().strip():
                    print(f"\n   ⚖️ Analysis: Responses are identical - steering had no effect on generation")
                else:
                    print(f"\n   🔄 Analysis: Steering changed the response")
                    
                    # Check if either mentions the correct answer (Nauru)
                    correct_answer = "Nauru"
                    unsteered_has_correct = correct_answer.lower() in unsteered_response.lower()
                    steered_has_correct = correct_answer.lower() in steered_response.lower()
                    
                    if unsteered_has_correct and not steered_has_correct:
                        print(f"   📉 Steering made response WORSE: removed correct answer '{correct_answer}'")
                    elif not unsteered_has_correct and steered_has_correct:
                        print(f"   📈 Steering made response BETTER: added correct answer '{correct_answer}'")
                    elif unsteered_has_correct and steered_has_correct:
                        print(f"   ➡️ Both responses mention correct answer '{correct_answer}'")
                    else:
                        print(f"   ❓ Neither response mentions correct answer '{correct_answer}'")
                        
                        # Check for incorrect answers
                        incorrect_answers = ["Vatican City", "Monaco", "United States"]
                        for incorrect in incorrect_answers:
                            if incorrect.lower() in unsteered_response.lower():
                                print(f"   ❌ Unsteered response mentions incorrect answer: '{incorrect}'")
                            if incorrect.lower() in steered_response.lower():
                                print(f"   ❌ Steered response mentions incorrect answer: '{incorrect}'")
                
            except Exception as e:
                print(f"   ❌ Response generation failed: {e}")
                import traceback
                if verbose:
                    traceback.print_exc()
        
        # Extract accuracy from steered results
        task_name = task_data.config.task
        if isinstance(steered_results, dict) and 'results' in steered_results:
            task_results = steered_results['results'].get(task_name, {})
            # Look for accuracy metrics with different naming conventions
            if 'acc,none' in task_results:
                accuracy = task_results['acc,none']
            elif 'acc' in task_results:
                accuracy = task_results['acc']
            elif 'accuracy' in task_results:
                accuracy = task_results['accuracy']
            else:
                accuracy = 'N/A'
        else:
            accuracy = 'N/A'
            task_results = {}
        
        evaluation_results = {
            "accuracy": accuracy,
            "method": "lm_harness_with_steering",
            "task_name": task_name,
            "steering_applied": len(steering_methods) > 0 if steering_methods else False,
            "full_results": task_results,
            "baseline_likelihoods": baseline_likelihoods,
            "steered_likelihoods": steered_likelihoods
        }
        
        if verbose:
            print(f"\n   ✅ Evaluation completed")
            print(f"   📊 Accuracy: {accuracy}")
            print(f"   🎯 Steering applied: {'Yes' if steering_methods else 'No'}")
            
        return evaluation_results
        
    except Exception as e:
        if verbose:
            print(f"   ❌ LM-harness evaluation failed: {e}")
        
        # Fallback to placeholder
        return {
            "accuracy": "N/A", 
            "method": "lm_harness_failed",
            "error": str(e),
            "note": "LM-harness evaluation failed, falling back to individual response evaluation"
        }
