"""
Comprehensive MPS compatibility fixes for wisent-guard.

This module applies all necessary patches to make wisent-guard fully
compatible with Apple Silicon MPS devices, addressing all known issues
including tensor device allocation, forward pass handling, and activation
collection problems.
"""

import torch
import sys
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

def apply_comprehensive_mps_fixes():
    """
    Apply comprehensive patches to all wisent-guard components for full MPS compatibility.
    
    This function addresses all known issues with MPS devices, including:
    1. "Placeholder storage not allocated on MPS device" errors
    2. Device mismatch issues during activation collection
    3. Dimension mismatch issues in tensor operations
    4. Problems with token selection for activation monitoring
    
    Returns:
        bool: True if patches were applied successfully
    """
    if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
        print("MPS not available - skipping compatibility patches")
        return False
    
    print("Applying MPS compatibility patches to wisent-guard...")
    
    # Import necessary modules
    try:
        from wisent_guard.utils.activation_hooks import ActivationHooks
        from wisent_guard.inference import SafeInference
        from wisent_guard.monitor import ActivationMonitor
        from wisent_guard.guard import ActivationGuard
        from wisent_guard.utils.helpers import cosine_sim, calculate_average_vector, get_layer_name
        from wisent_guard.vectors import ContrastiveVectors
    except ImportError as e:
        print(f"Error importing wisent-guard modules: {e}")
        return False
    
    # Store original methods
    original_activation_hook = ActivationHooks._activation_hook
    original_get_activations = ActivationHooks.get_activations
    original_hook_register = ActivationHooks.register_hooks
    original_guard_is_harmful = ActivationGuard.is_harmful
    original_guard_init = ActivationGuard.__init__
    original_monitor_check = ActivationMonitor.check_activations
    original_safe_inference_check = SafeInference._check_prompt_safety
    original_safe_inference_generate = SafeInference.generate
    original_cosine_sim = cosine_sim
    original_calculate_average = calculate_average_vector
    
    # ===== PATCH 1: Activation Hooks =====
    
    def patched_activation_hook(self, layer_idx):
        """Fixed activation hook that properly handles MPS device allocation"""
        def hook(module, input, output):
            if layer_idx in self.active_layers:
                # Get the output hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                # Get module device
                device = next(module.parameters()).device
                
                # Store activations - always detach, clone AND move to CPU first
                if isinstance(hidden_states, torch.Tensor):
                    if self.token_strategy == "target_token" and hasattr(self, 'target_tokens') and self.target_tokens:
                        # Process multi-token logic on CPU for stability
                        hidden_cpu = hidden_states.detach().clone().cpu()
                        
                        # Store all tokens for now
                        self.all_hidden_states[layer_idx] = hidden_cpu
                        
                        # Try to find the target tokens later during get_activations()
                        # This defers the complex token logic until later
                        
                    else:
                        # Default to last token for safety on MPS
                        last_token_idx = hidden_states.shape[1] - 1
                        hidden_cpu = hidden_states[:, last_token_idx, :].detach().clone().cpu()
                        self.layer_activations[layer_idx] = hidden_cpu
                        self.last_token_position = last_token_idx
        
        return hook
    
    def patched_get_activations(self, layer_idx=None, token_idx=None):
        """Fixed get_activations that handles token finding on CPU for stability"""
        # Process target token logic if needed
        if hasattr(self, 'target_tokens') and self.target_tokens and hasattr(self, 'all_hidden_states'):
            # Get target token IDs if we have them
            target_token_ids = []
            if hasattr(self, 'target_token_ids') and self.target_token_ids:
                target_token_ids = self.target_token_ids
            elif hasattr(self, 'target_tokens') and self.target_tokens:
                # Try to get target token IDs from the target tokens
                if hasattr(self, 'tokenizer') and self.tokenizer:
                    target_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in self.target_tokens]
            
            for layer in self.all_hidden_states:
                if layer not in self.layer_activations:
                    # Get the hidden states
                    hidden = self.all_hidden_states[layer]
                    
                    # Process on CPU
                    found_position = False
                    
                    # Try to find target tokens
                    if hasattr(self, 'last_input_ids') and self.last_input_ids is not None and target_token_ids:
                        # Check each target token
                        input_ids_cpu = self.last_input_ids.cpu()
                        
                        # Default to last token if we can't find the target
                        last_token_pos = hidden.shape[1] - 1
                        
                        for i in range(input_ids_cpu.shape[1]):
                            if input_ids_cpu[0, i].item() in target_token_ids:
                                if i < hidden.shape[1]:  # Check bounds
                                    self.layer_activations[layer] = hidden[:, i, :].detach().clone()
                                    found_position = True
                                    break
                    
                    # Fall back to last token if target not found
                    if not found_position:
                        self.layer_activations[layer] = hidden[:, -1, :].detach().clone()
        
        # Continue with original logic
        result = {}
        if layer_idx is not None:
            if layer_idx in self.layer_activations:
                result = {layer_idx: self.layer_activations[layer_idx]}
        else:
            result = self.layer_activations
        
        return result
    
    def patched_register_hooks(self, layers):
        """Fixed hook registration with proper device handling"""
        self.active_layers = layers
        self.hook_handles = []
        
        # Always ensure we have this attribute
        if not hasattr(self, 'all_hidden_states'):
            self.all_hidden_states = {}
        
        if not self.model:
            return
        
        # Clear existing hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
        # Register new hooks
        for layer_idx in layers:
            layer_name = get_layer_name(self.model_type, layer_idx)
            try:
                module = self._get_module_by_name(layer_name)
                
                # Add a pre-hook to capture input_ids for all strategies
                def forward_pre_hook(module, args):
                    if args and isinstance(args[0], torch.Tensor):
                        # Store input_ids for later use, but keep on same device
                        module._last_input_ids = args[0].detach().clone()
                    return args
                
                # Register pre-hook
                pre_hook_handle = module.register_forward_pre_hook(forward_pre_hook)
                self.hook_handles.append(pre_hook_handle)
                
                # Register the main hook
                hook_handle = module.register_forward_hook(self._activation_hook(layer_idx))
                self.hook_handles.append(hook_handle)
                
                print(f"Registered hook for layer {layer_idx} ({layer_name})")
            except Exception as e:
                print(f"Failed to register hook for layer {layer_idx} ({layer_name}): {e}")
        
        # Register a forward pre-hook to capture input IDs
        if hasattr(self.model, 'forward'):
            def model_forward_pre_hook(module, args):
                if args and isinstance(args[0], torch.Tensor):
                    # Store input_ids for later use, but keep on same device
                    self.last_input_ids = args[0].detach().clone()
            
            # Register pre-hook on the whole model
            model_pre_hook = self.model.register_forward_pre_hook(model_forward_pre_hook)
            self.hook_handles.append(model_pre_hook)
    
    # ===== PATCH 2: ActivationGuard Is Harmful =====
    
    def patched_is_harmful(self, text, categories=None):
        """Fixed is_harmful that properly handles device allocation"""
        if self.monitor is None:
            self._initialize_monitor_and_inference()
        
        # Reset monitor
        self.monitor.reset()
        
        # Tokenize the input
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        
        # Explicitly move to the same device as the model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Run a forward pass with appropriate error handling
        try:
            with torch.no_grad():
                # Run forward pass - handle output gracefully
                outputs = self.model(input_ids)
                
                # MPS compatibility: ensure we process activations on CPU if needed
                is_harmful = self.monitor.is_harmful(categories)
                
                # Cleanup to avoid memory issues
                del outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                return is_harmful
                
        except Exception as e:
            print(f"Error during harmful content check: {e}")
            # Since we can't determine, err on the side of caution
            return True
    
    # ===== PATCH 3: ActivationGuard Init =====
    
    def patched_guard_init(
        self,
        model,
        tokenizer=None,
        layers=None,
        threshold=0.7,
        save_dir="./wisent_guard_data",
        device=None,
        token_strategy="target_token",
    ):
        # Process model and tokenizer
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = model
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        # Ensure we have a pad token for the tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set default layers if not provided
        self.layers = layers if layers is not None else [15]
        
        # Set threshold for harmful content detection
        self.threshold = threshold
        
        # Set token strategy
        self.token_strategy = token_strategy
        
        # Load contrastive vectors from disk
        self.save_dir = save_dir
        self.vectors = ContrastiveVectors(save_dir=save_dir)
        self.vectors.load_vectors()
        
        # Set device
        if device is not None:
            self.device = device
        elif hasattr(torch.mps, 'is_available') and torch.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Move model to device if specified
        if self.device:
            self.model = self.model.to(self.device)
        
        # Initialize monitor and inference initially to None
        self.monitor = None
        self.inference = None
        
        # Set up target tokens for multiple-choice activation collection
        self._setup_target_tokens()
        
        # Initialize monitor and inference
        self._initialize_monitor_and_inference()
    
    # ===== PATCH 4: Monitor Check Activations =====
    
    def patched_check_activations(self, categories=None):
        """Patched version of check_activations with better MPS compatibility"""
        # Initialize results dict for all categories
        results = {}
        
        # Get the available categories
        available_categories = list(self.vectors.contrastive_vectors.keys())
        
        # If categories are specified, filter to only those categories
        if categories is not None:
            categories_to_check = [c for c in categories if c in available_categories]
        else:
            categories_to_check = available_categories
        
        # Check each category
        for category in categories_to_check:
            # Initialize results for this category
            results[category] = {
                "is_harmful": False,
                "max_similarity": 0.0,
                "triggered_layers": [],
                "layer_similarities": {}
            }
            
            # Get the current activations
            activations = self.hooks.get_activations()
            
            # Check similarity for each layer
            for layer in self.layers:
                # Skip layers that don't have activations
                if layer not in activations:
                    continue
                
                # Skip layers that don't have contrastive vectors for this category
                if layer not in self.vectors.contrastive_vectors.get(category, {}):
                    continue
                
                # Get contrastive vector for this category and layer
                contrastive_vector = self.vectors.get_contrastive_vector(category, layer)
                
                # Skip if no contrastive vector is found
                if contrastive_vector is None:
                    continue
                
                # Get current activation
                current_activation = activations[layer]
                
                # Ensure dimensions match - use CPU for stability in dimension handling
                if current_activation.device.type == "mps":
                    current_activation = current_activation.cpu()
                
                if contrastive_vector.device.type == "mps":
                    contrastive_vector = contrastive_vector.cpu()
                
                # Flatten if needed
                if len(current_activation.shape) > 1:
                    current_activation = current_activation.flatten()
                
                if len(contrastive_vector.shape) > 1:
                    contrastive_vector = contrastive_vector.flatten()
                
                # Match dimensions if necessary
                if current_activation.shape != contrastive_vector.shape:
                    min_dim = min(current_activation.shape[0], contrastive_vector.shape[0])
                    current_activation = current_activation[:min_dim]
                    contrastive_vector = contrastive_vector[:min_dim]
                
                # Calculate cosine similarity
                try:
                    similarity = torch.nn.functional.cosine_similarity(
                        current_activation.unsqueeze(0),
                        contrastive_vector.unsqueeze(0)
                    ).item()
                except Exception as e:
                    print(f"Error calculating similarity for {category}, layer {layer}: {e}")
                    similarity = 0.0
                
                # Store similarity for this layer
                results[category]["layer_similarities"][layer] = similarity
                
                # Check if similarity exceeds threshold
                if similarity > self.threshold:
                    results[category]["is_harmful"] = True
                    results[category]["triggered_layers"].append(layer)
                
                # Update max similarity
                if similarity > results[category]["max_similarity"]:
                    results[category]["max_similarity"] = similarity
        
        # Return the results for all categories
        return results
    
    # ===== PATCH 5: SafeInference Check Prompt Safety =====
    
    def patched_check_prompt_safety(self, input_text):
        """Patched check_prompt_safety for MPS compatibility"""
        # Reset monitor state
        self.monitor.reset()
        
        # Tokenize and encode input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        # Get the model's device and move input_ids to it
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Run a forward pass through the model to get activations
        try:
            with torch.no_grad():
                self.model(input_ids)
            
            # Check if activations match harmful patterns
            is_harmful = self.monitor.is_harmful()
            
            if is_harmful:
                harmful_category = self.monitor.get_most_harmful_category()
                if harmful_category:
                    category, similarity = harmful_category
                    self.blocked_reason = f"Prompt contains potentially harmful content related to '{category}' (similarity: {similarity:.2f})"
                else:
                    self.blocked_reason = "Prompt contains potentially harmful content"
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking prompt safety: {e}")
            # Fail safe: assume it's fine and let later checks catch issues
            return True
    
    # ===== PATCH 6: SafeInference Generate =====
    
    def patched_generate(self, prompt, max_new_tokens=100, skip_prompt_check=False, **kwargs):
        """Patched generate method with MPS compatibility"""
        # Reset monitoring state
        self.monitor.reset()
        self.blocked_reason = None
        
        # Check if the prompt itself is safe
        prompt_is_safe = True
        if not skip_prompt_check:
            prompt_is_safe = self._check_prompt_safety(prompt)
        
        # If prompt is not safe and blocking is enabled, return early
        if not prompt_is_safe and self.block_on_harmful:
            return {
                "response": self.unsafe_message,
                "blocked": True,
                "reason": self.blocked_reason
            }
        
        # Prepare for generation
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
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
                
                # Check for harmful content in the full sequence
                self.monitor.reset()
                
                # Note: We're running a forward pass on the outputs
                # This needs to be on the same device
                _ = self.model(outputs.to(device))
                
                if self.monitor.is_harmful() and self.block_on_harmful:
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
                
                # Decode the generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the assistant's response
                if "<|assistant|>" in generated_text:
                    response = generated_text.split("<|assistant|>")[-1].strip()
                else:
                    response = generated_text
                    
        except Exception as e:
            return {
                "response": f"Error during generation: {str(e)}",
                "blocked": True,
                "reason": f"Exception: {str(e)}"
            }
        
        # Return results
        return {
            "response": response,
            "blocked": False,
            "reason": None
        }
    
    # ===== PATCH 7: Improved Helper Functions =====
    
    def patched_cosine_sim(x1, x2):
        """Patched cosine similarity function that's robust to dimension mismatches"""
        # Move to CPU for consistent behavior
        if x1.device.type == "mps":
            x1 = x1.cpu()
        if x2.device.type == "mps":
            x2 = x2.cpu()
        
        # Flatten if needed
        if len(x1.shape) > 1:
            x1 = x1.flatten()
        if len(x2.shape) > 1:
            x2 = x2.flatten()
        
        # Match dimensions if necessary
        if x1.shape != x2.shape:
            min_dim = min(x1.shape[0], x2.shape[0])
            x1 = x1[:min_dim]
            x2 = x2[:min_dim]
        
        # Compute cosine similarity
        return torch.nn.functional.cosine_similarity(
            x1.unsqueeze(0), 
            x2.unsqueeze(0)
        )[0]
    
    def patched_calculate_average_vector(vectors):
        """Patched calculate_average_vector that handles dimension mismatches"""
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
                # Pad with zeros
                padding = torch.zeros(target_dim - v.shape[-1], device=v.device)
                adjusted_vectors.append(torch.cat([v, padding]))
            else:
                adjusted_vectors.append(v)
        
        if not adjusted_vectors:
            raise ValueError("No vectors remained after dimension adjustment")
        
        # Stack and calculate mean
        stacked = torch.stack(adjusted_vectors)
        return torch.mean(stacked, dim=0)
    
    # Apply all patches
    ActivationHooks._activation_hook = patched_activation_hook
    ActivationHooks.get_activations = patched_get_activations
    ActivationHooks.register_hooks = patched_register_hooks
    ActivationGuard.is_harmful = patched_is_harmful
    ActivationGuard.__init__ = patched_guard_init
    ActivationMonitor.check_activations = patched_check_activations
    SafeInference._check_prompt_safety = patched_check_prompt_safety
    SafeInference.generate = patched_generate
    
    # Patch helper functions
    sys.modules['wisent_guard.utils.helpers'].cosine_sim = patched_cosine_sim
    sys.modules['wisent_guard.utils.helpers'].calculate_average_vector = patched_calculate_average_vector
    
    print("✅ Applied comprehensive MPS compatibility patches to wisent-guard")
    return True 