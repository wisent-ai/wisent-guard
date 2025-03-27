"""
MPS (Metal Performance Shaders) compatibility patches for wisent-guard

This file provides a set of comprehensive patches to ensure the wisent-guard
package works correctly on Apple Silicon GPUs via MPS. These patches address
tensor allocation issues and ensure consistent device handling.

Usage:
    from patches.mps_compatibility import apply_mps_patches
    
    # Apply patches if on Apple Silicon
    if torch.backends.mps.is_available():
        apply_mps_patches()
"""

import torch
import os
import sys
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable

def apply_mps_patches():
    """
    Apply comprehensive MPS compatibility patches to wisent-guard.
    
    This function applies patches to fix device allocation issues when 
    running on Apple Silicon GPUs. It modifies activation hooks, 
    safety checks, generation methods, and training functions.
    
    Returns:
        bool: True if patches were applied, False otherwise
    """
    if not hasattr(torch.mps, 'is_available') or not torch.mps.is_available():
        print("MPS is not available. Patches not applied.")
        return False
    
    try:
        # Import required modules
        from wisent_guard.utils.activation_hooks import ActivationHooks
        from wisent_guard.inference import SafeInference
        from wisent_guard.vectors import ContrastiveVectors
        from wisent_guard import ActivationGuard
        from wisent_guard.monitor import ActivationMonitor
        from wisent_guard.utils.helpers import cosine_sim
        
        print("Applying MPS compatibility patches to wisent-guard...")
        
        # Store original methods for reference
        original_activation_hook = ActivationHooks._activation_hook
        original_check_prompt_safety = SafeInference._check_prompt_safety
        original_generate = SafeInference.generate
        original_train_on_phrase_pairs = ActivationGuard.train_on_phrase_pairs
        original_train_on_formatted_pairs = ActivationGuard._train_on_formatted_pairs
        original_generate_safe_response = ActivationGuard.generate_safe_response
        original_generate_multiple_choice = ActivationGuard.generate_multiple_choice_response
        original_check_activations = ActivationMonitor.check_activations
        
        # Patch 1: Fix activation hook for MPS
        def patched_activation_hook(self, layer_idx):
            """
            MPS-compatible activation hook that handles tensor allocation properly.
            """
            def hook(module, input, output):
                # For most transformer models, we want the output of the attention layer
                # or the MLP layer as our activation vector
                if layer_idx in self.active_layers:
                    # Get the output hidden states - typically the first element of the output
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output
                    
                    # Store the hidden states (activations)
                    # We may need to handle different shapes based on model architecture
                    if isinstance(hidden_states, torch.Tensor):
                        # Always default to the target_token strategy on MPS
                        # Avoiding the complex token logic that causes issues
                        last_token_idx = hidden_states.shape[1] - 1
                        
                        # Make sure tensor is detached and on the same device
                        device = hidden_states.device
                        self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone().to(device)
                        
                        # Simply record the position for debugging
                        self.last_token_position = last_token_idx
            
            return hook
            
        # Add has_activations method to ActivationHooks
        def has_activations(self):
            """Check if we have collected any activations"""
            return bool(self.layer_activations)
            
        # Add method to get activations from ActivationHooks
        def get_activations(self):
            """Get the collected activations"""
            return self.layer_activations
        
        # Patch 2: Fix prompt safety checking for MPS
        def patched_check_prompt_safety(self, input_text):
            """
            MPS-compatible prompt safety checking with proper device handling.
            """
            # Reset monitor state
            self.monitor.reset()
            
            # Tokenize and encode input
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Get the model's device and move input_ids to it
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Run a forward pass through the model to get activations
            with torch.no_grad():
                self.model(input_ids)
            
            # Check if activations match harmful patterns
            if self.monitor.is_harmful():
                harmful_category = self.monitor.get_most_harmful_category()
                if harmful_category:
                    category, similarity = harmful_category
                    self.blocked_reason = f"Prompt contains potentially harmful content related to '{category}' (similarity: {similarity:.2f})"
                else:
                    self.blocked_reason = "Prompt contains potentially harmful content"
                return False
            
            return True
        
        # Patch 3: Fix generation for MPS
        def patched_generate(self, prompt, max_new_tokens=100, skip_prompt_check=False, **kwargs):
            """
            MPS-compatible text generation with proper device handling.
            """
            # Reset monitoring state
            self.monitor.reset()
            self.blocked_reason = None
            
            # Check if the prompt itself is safe
            prompt_is_safe = True
            if not skip_prompt_check:
                try:
                    prompt_is_safe = self._check_prompt_safety(prompt)
                except Exception as e:
                    print(f"Warning: Error during prompt safety check: {e}")
                    # Continue with generation despite the error
                    prompt_is_safe = True
            
            # If prompt is not safe and blocking is enabled, return early
            if not prompt_is_safe and self.block_on_harmful:
                return {
                    "response": self.unsafe_message,
                    "blocked": True,
                    "reason": self.blocked_reason
                }
            
            # Format the prompt if needed
            if hasattr(self, 'format_prompt') and callable(self.format_prompt):
                formatted_prompt = self.format_prompt(prompt)
            else:
                # Simple format without system prompt to match baseline
                formatted_prompt = prompt
            
            # Prepare for generation
            input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Set up generation parameters
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            # Add other generation parameters
            gen_kwargs.update(kwargs)
            
            # Generate text
            try:
                with torch.no_grad():
                    # Standard generation process
                    outputs = self.model.generate(
                        input_ids,
                        **gen_kwargs
                    )
                    
                    try:
                        # Check for harmful content in the full sequence
                        self.monitor.reset()
                        _ = self.model(outputs.to(device))
                        
                        is_harmful = self.monitor.is_harmful()
                        
                        if is_harmful and self.block_on_harmful:
                            harmful_category = self.monitor.get_most_harmful_category()
                            if harmful_category:
                                category, similarity = harmful_category
                                self.blocked_reason = f"Response contains potentially harmful content related to '{category}' (similarity: {similarity:.2f})"
                            else:
                                self.blocked_reason = "Response contains potentially harmful content"
                            
                            return {
                                "response": self.unsafe_message,
                                "blocked": True,
                                "reason": self.blocked_reason
                            }
                    except Exception as e:
                        print(f"Warning: Error during harmful content check: {e}")
                        # Continue with generation despite the error
                    
                    # Decode the generated text
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
            except Exception as e:
                return {
                    "response": f"Error during generation: {str(e)}",
                    "blocked": True,
                    "reason": f"Exception: {str(e)}"
                }
            
            # Return results
            return {
                "response": generated_text,
                "blocked": False,
                "reason": None
            }
        
        # Patch 4: Fix the check_activations method in ActivationMonitor
        def patched_check_activations(self, categories=None):
            """
            MPS-compatible activation checking with dimension handling.
            """
            # If no categories specified, check all
            if categories is None:
                # Use the custom method to get categories
                categories = self.vectors.get_categories()
            
            # If no activations or no categories, return empty dict
            if not self.hooks.has_activations() or not categories:
                return {}
            
            # Initialize result dictionary
            results = {}
            self.is_triggered = False
            
            # Check each category
            for category in categories:
                # Initialize category results
                if category not in self.max_similarities:
                    self.max_similarities[category] = {}
                
                if category not in self.triggered_layers:
                    self.triggered_layers[category] = set()
                
                category_results = {
                    "is_harmful": False,
                    "max_similarity": 0.0,
                    "triggered_layers": [],
                    "layer_similarities": {}
                }
                
                # Get activations for each layer
                activations = self.hooks.get_activations()
                
                # Get contrastive vectors for all categories
                contrastive_vectors = self.vectors.get_contrastive_vectors()
                
                # Skip if category not in contrastive vectors
                if category not in contrastive_vectors:
                    results[category] = category_results
                    continue
                
                # Check each layer in this category
                for layer_str, contrastive_vector in contrastive_vectors[category].items():
                    # Convert layer to int if it's a string
                    layer = int(layer_str) if isinstance(layer_str, str) else layer_str
                    
                    # Skip if we don't have activations for this layer
                    if layer not in activations:
                        continue
                    
                    # Get the activation for this layer
                    activation = activations[layer]
                    
                    try:
                        # Check for dimension mismatch and fix if possible
                        if activation.shape != contrastive_vector.shape:
                            print(f"Warning: Dimension mismatch - activation: {activation.shape}, vector: {contrastive_vector.shape}")
                            if len(activation.shape) > 1 and len(contrastive_vector.shape) > 1:
                                # Reshape to same dimensions if possible
                                if activation.shape[0] == 1 and contrastive_vector.shape[0] == 1:
                                    activation = activation.view(1, -1)
                                    contrastive_vector = contrastive_vector.view(1, -1)
                                else:
                                    print(f"  Skipping layer {layer} due to incompatible dimensions")
                                    continue
                            else:
                                # Try to reshape to same dimensions
                                activation = activation.view(1, -1)
                                contrastive_vector = contrastive_vector.view(1, -1)
                                
                        # Move tensor to CPU for cosine similarity calculation
                        activation = activation.cpu()
                        contrastive_vector = contrastive_vector.cpu()
                        
                        # Calculate similarity
                        similarity = cosine_sim(activation, contrastive_vector)
                        
                        # Update max similarity for this layer
                        self.max_similarities[category][layer] = max(
                            similarity,
                            self.max_similarities[category].get(layer, -1.0)
                        )
                        
                        # Store layer similarity in results
                        category_results["layer_similarities"][str(layer)] = float(similarity)
                        
                        # Check if threshold is exceeded
                        if similarity >= self.threshold:
                            self.triggered_layers[category].add(layer)
                            category_results["is_harmful"] = True
                            self.is_triggered = True
                            
                            # If using target_token strategy, try to identify the triggering token
                            if self.token_strategy == "target_token" and hasattr(self.hooks, "last_token_id"):
                                self.triggering_token_id = self.hooks.last_token_id
                        
                    except Exception as e:
                        print(f"Error comparing activations for layer {layer}: {e}")
                        continue
                
                # Update category results
                if self.max_similarities[category]:
                    category_results["max_similarity"] = max(self.max_similarities[category].values())
                
                if self.triggered_layers[category]:
                    category_results["triggered_layers"] = list(self.triggered_layers[category])
                
                results[category] = category_results
            
            return results
            
        # Patch 5: Fix generate_safe_response for MPS
        def patched_generate_safe_response(
            self,
            prompt,
            max_new_tokens=100,
            skip_prompt_check=False,
            use_multiple_choice=True,
            **kwargs
        ):
            """
            MPS-compatible safe response generation with consistent output format.
            """
            if self.inference is None:
                raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
            
            if use_multiple_choice:
                # Create a simple multiple-choice format with the prompt
                choice_a = "I'll provide a helpful and accurate response."
                choice_b = "I'll provide a response with inaccurate or harmful information."
                question = f"{prompt}\nSelect one:"
                
                result = self.generate_multiple_choice_response(
                    question=question,
                    choice_a=choice_a,
                    choice_b=choice_b,
                    max_new_tokens=10,  # Enough for a clear A or B
                    **kwargs
                )
                
                # If not blocked and response contains A, generate a real response
                if not result.get("blocked", False) and "A" in result.get("response", ""):
                    # Generate the actual response with standard function
                    result = self.inference.generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        skip_prompt_check=True,  # Already checked in multiple-choice
                        **kwargs
                    )
                    return {
                        "text": result.get("response", "").strip(),
                        "is_safe": True,
                        "blocked": False,
                        "reason": None
                    }
                elif not result.get("blocked", False):
                    # Model chose B or something else, block it
                    return {
                        "text": "I cannot provide that response as it might contain inaccurate information.",
                        "is_safe": False,
                        "blocked": True,
                        "reason": "Model showed intent to provide problematic content"
                    }
                else:
                    # Already blocked
                    return {
                        "text": result.get("response", "").strip(),
                        "is_safe": False,
                        "blocked": True,
                        "reason": result.get("reason", None)
                    }
            else:
                # Use standard generation
                result = self.inference.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    skip_prompt_check=skip_prompt_check,
                    **kwargs
                )
                
                # Convert to consistent format
                return {
                    "text": result.get("response", "").strip(),
                    "is_safe": not result.get("blocked", False),
                    "blocked": result.get("blocked", False),
                    "reason": result.get("reason", None)
                }
        
        # Patch 6: Fix generate_multiple_choice_response for MPS
        def patched_generate_multiple_choice_response(
            self,
            question,
            choice_a,
            choice_b,
            max_new_tokens=10,  # Increased for clearer A/B selection
            **kwargs
        ):
            """
            MPS-compatible multiple-choice response generation.
            """
            if self.inference is None:
                raise ValueError("No vectors have been loaded or trained. Call train_on_phrase_pairs first.")
            
            # Format as multiple-choice
            prompt = f"{question}\nA. {choice_a}\nB. {choice_b}"
            
            # Generate with limited tokens (just enough for A or B)
            result = self.inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                skip_prompt_check=True,  # Skip prompt check for multiple-choice
                **kwargs
            )
            
            return result
        
        # Patch 7: Fix phrase pair training for MPS
        def patched_train_on_phrase_pairs(self, phrase_pairs, category="harmful_content"):
            """
            MPS-compatible phrase pair training with proper device handling.
            """
            from tqdm import tqdm
            
            print(f"Training on {len(phrase_pairs)} phrase pairs for category '{category}'...")
            print("Converting phrase pairs to multiple-choice format for consistent activation collection...")
            
            # Convert phrase pairs to multiple-choice format
            mc_pairs = []
            for pair in tqdm(phrase_pairs, desc="Processing phrase pairs"):
                harmful_phrase = pair["harmful"]
                harmless_phrase = pair["harmless"]
                
                # Convert to multiple-choice format
                mc_pair = self._convert_to_multiple_choice(harmful_phrase, harmless_phrase)
                mc_pairs.append(mc_pair)
            
            # Train on the formatted pairs
            self._train_on_formatted_pairs(mc_pairs, category)
            
        # Patch 8: Fix formatted pairs training for MPS
        def patched_train_on_formatted_pairs(self, formatted_pairs, category):
            """
            MPS-compatible formatted pairs training with proper device handling.
            """
            from tqdm import tqdm
            
            # Make sure we have a monitor initialized
            if self.monitor is None:
                self._initialize_monitor_and_inference()
            
            # Get the model device once to avoid repeated calls
            device = next(self.model.parameters()).device
            
            # Process each phrase pair
            for pair in tqdm(formatted_pairs, desc="Processing formatted pairs"):
                # Handle both dictionary and tuple formats
                if isinstance(pair, tuple) and len(pair) == 2:
                    harmful_phrase, harmless_phrase = pair
                elif isinstance(pair, dict) and "harmful" in pair and "harmless" in pair:
                    harmful_phrase = pair["harmful"]
                    harmless_phrase = pair["harmless"]
                else:
                    print(f"Warning: Skipping improperly formatted pair: {pair}")
                    continue
                
                # IMPORTANT: Create and move all tensors in a consistent way
                # Get activations for harmful phrase
                self.monitor.reset()
                harmful_input_ids = self.tokenizer.encode(harmful_phrase, return_tensors="pt").to(device)
                with torch.no_grad():
                    # Ensure the model itself is on the correct device
                    self.model.to(device)
                    self.model(harmful_input_ids)
                harmful_activations = self.monitor.hooks.get_activations()
                
                # Get activations for harmless phrase
                self.monitor.reset()
                harmless_input_ids = self.tokenizer.encode(harmless_phrase, return_tensors="pt").to(device)
                with torch.no_grad():
                    self.model(harmless_input_ids)
                harmless_activations = self.monitor.hooks.get_activations()
                
                # Move all activation vectors to the CPU for storage
                # This prevents issues with MPS storage not being allocated
                cpu_harmful_activations = {}
                cpu_harmless_activations = {}
                
                for layer, tensor in harmful_activations.items():
                    cpu_harmful_activations[layer] = tensor.detach().cpu()
                
                for layer, tensor in harmless_activations.items():
                    cpu_harmless_activations[layer] = tensor.detach().cpu()
                
                # Store activations for each layer
                for layer in self.layers:
                    if layer in cpu_harmful_activations and layer in cpu_harmless_activations:
                        self.vectors.add_vector_pair(
                            category=category,
                            layer=layer,
                            harmful_vector=cpu_harmful_activations[layer],
                            harmless_vector=cpu_harmless_activations[layer]
                        )
            
            # Compute and save contrastive vectors
            self.vectors.compute_contrastive_vectors()
            self.vectors.save_vectors()
            
            # Re-initialize monitor with new vectors
            self._initialize_monitor_and_inference()
        
        # Patch 9: Fix train_on_multiple_choice_pairs for MPS
        def patched_train_on_multiple_choice_pairs(self, questions, category="hallucination"):
            """
            MPS-compatible multiple-choice pairs training.
            """
            print(f"Training on {len(questions)} multiple-choice questions for category '{category}'...")
            
            # Print target token info
            if hasattr(self, 'target_tokens') and self.target_tokens:
                token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.target_tokens]
                decoded = {id: token for id, token in zip(token_ids, self.target_tokens)}
                print(f"Target tokens set: {self.target_tokens} → IDs: {token_ids} → Decoded: {decoded}")
                print(f"Multiple-choice tokens set: ['A', 'B']")
            
            # Prepare phrase pairs in multiple-choice format
            formatted_pairs = []
            for q in questions:
                # Create A (correct) and B (incorrect) response phrases
                a_phrase = self._format_multiple_choice(
                    q["question"], 
                    q["choice_a"], 
                    q["choice_b"], 
                    "A"
                )
                
                b_phrase = self._format_multiple_choice(
                    q["question"], 
                    q["choice_a"], 
                    q["choice_b"], 
                    "B"
                )
                
                formatted_pairs.append({
                    "harmless": a_phrase,  # A is correct/harmless
                    "harmful": b_phrase    # B is incorrect/harmful
                })
            
            # Use our internal method to train
            self._train_on_formatted_pairs(formatted_pairs, category=category)
        
        # Add helper methods to ActivationGuard if not already present
        def _convert_to_multiple_choice(self, harmful_phrase, harmless_phrase):
            """Convert a harmful/harmless phrase pair to multiple-choice format."""
            # Create a question from the phrases
            question = "Choose the most accurate statement:"
            choice_a = harmless_phrase  # A is the harmless/correct option
            choice_b = harmful_phrase   # B is the harmful/incorrect option
            
            # Format into multiple-choice format
            harmless_formatted = self._format_multiple_choice(question, choice_a, choice_b, "A")
            harmful_formatted = self._format_multiple_choice(question, choice_a, choice_b, "B")
            
            return {
                "harmless": harmless_formatted,
                "harmful": harmful_formatted
            }
        
        def _format_multiple_choice(self, question, choice_a, choice_b, answer):
            """Format a question and choices into a complete multiple-choice prompt."""
            # Construct the user prompt with the question and choices
            user_prompt = f"{question}\nA. {choice_a}\nB. {choice_b}"
            
            # Create the assistant's answer
            if answer == "A":
                assistant_answer = f"The correct answer is {answer}."
            else:
                assistant_answer = f"The correct answer is {answer}."
            
            # Create a chat format used by most instruction models
            system = "You are a helpful assistant that provides accurate information."
            formatted_prompt = f"<s>[INST] {system}\n\n{user_prompt} [/INST] {assistant_answer}</s>"
            
            return formatted_prompt
        
        # Add helper methods to ContrastiveVectors
        def get_existing_layers(self, category):
            """Get layers for which vectors exist for a specific category"""
            if not hasattr(self, 'contrastive_vectors'):
                return []
            
            if category not in self.contrastive_vectors:
                return []
                
            return list(self.contrastive_vectors[category].keys())
        
        def get_categories(self):
            """Get all categories for which we have contrastive vectors"""
            if not hasattr(self, 'contrastive_vectors'):
                return []
                
            return list(self.contrastive_vectors.keys())
            
        # Patch calculate_average_vector to handle different size tensors
        def patched_calculate_average_vector(vectors: List[torch.Tensor]) -> torch.Tensor:
            """
            Calculate the average of a list of vectors, handling different sizes.
            
            This patched version can handle tensors of different dimensions by either
            truncating to the smallest dimension or padding to the largest dimension.
            
            Args:
                vectors: List of vectors to average
                
            Returns:
                Average vector
            """
            if not vectors:
                raise ValueError("Empty list of vectors provided")
            
            # Check if all vectors have the same shape
            shapes = [v.shape for v in vectors]
            if len(set([tuple(s) for s in shapes])) == 1:
                # All same shape, use standard method
                stacked = torch.stack(vectors)
                return torch.mean(stacked, dim=0)
            
            # Different shapes detected
            print(f"Warning: Vectors have different shapes: {shapes}")
            
            # Find the most common dimensionality
            dims = {}
            for v in vectors:
                dim = v.shape[-1]
                if dim not in dims:
                    dims[dim] = 0
                dims[dim] += 1
            
            # Use the most common dimension
            target_dim = max(dims.items(), key=lambda x: x[1])[0]
            print(f"Using most common dimension: {target_dim}")
            
            # Adjust vectors to target dimension
            adjusted_vectors = []
            for v in vectors:
                if v.shape[-1] > target_dim:
                    # Truncate
                    adjusted_vectors.append(v[..., :target_dim])
                elif v.shape[-1] < target_dim:
                    # Skip vectors that are too small
                    print(f"Skipping vector with dimension {v.shape[-1]} (too small)")
                    continue
                else:
                    adjusted_vectors.append(v)
            
            if not adjusted_vectors:
                raise ValueError("No vectors remained after dimension adjustment")
            
            # Stack and calculate mean
            stacked = torch.stack(adjusted_vectors)
            return torch.mean(stacked, dim=0)
        
        # Apply all patches
        ActivationHooks._activation_hook = patched_activation_hook
        ActivationHooks.has_activations = has_activations
        ActivationHooks.get_activations = get_activations
        SafeInference._check_prompt_safety = patched_check_prompt_safety
        SafeInference.generate = patched_generate
        ActivationGuard.train_on_phrase_pairs = patched_train_on_phrase_pairs
        ActivationGuard._train_on_formatted_pairs = patched_train_on_formatted_pairs
        ActivationGuard.generate_safe_response = patched_generate_safe_response
        ActivationGuard.generate_multiple_choice_response = patched_generate_multiple_choice_response
        ActivationGuard.train_on_multiple_choice_pairs = patched_train_on_multiple_choice_pairs
        ActivationMonitor.check_activations = patched_check_activations
        
        # Add helper methods if not present
        if not hasattr(ActivationGuard, '_convert_to_multiple_choice'):
            ActivationGuard._convert_to_multiple_choice = _convert_to_multiple_choice
            
        if not hasattr(ActivationGuard, '_format_multiple_choice'):
            ActivationGuard._format_multiple_choice = _format_multiple_choice
            
        if not hasattr(ContrastiveVectors, 'get_existing_layers'):
            ContrastiveVectors.get_existing_layers = get_existing_layers
            
        if not hasattr(ContrastiveVectors, 'get_categories'):
            ContrastiveVectors.get_categories = get_categories
        
        # Patch the calculate_average_vector function in helpers
        from wisent_guard.utils.helpers import calculate_average_vector
        sys.modules['wisent_guard.utils.helpers'].calculate_average_vector = patched_calculate_average_vector
        
        print("✅ Applied comprehensive MPS compatibility patches to wisent-guard")
        return True
        
    except ImportError as e:
        print(f"Failed to apply MPS patches: {e}")
        print("Make sure wisent-guard is properly installed.")
        return False
    
    except Exception as e:
        print(f"An error occurred while applying MPS patches: {e}")
        return False

# Make the patch function available when this file is imported
if __name__ == "__main__":
    print("MPS compatibility patches for wisent-guard")
    print("Import and call apply_mps_patches() to apply the patches") 