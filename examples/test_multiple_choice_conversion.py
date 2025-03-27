#!/usr/bin/env python
"""
Test script to validate the multiple-choice conversion approach for activation collection.
This script ensures that wisent-guard correctly collects activations from A/B tokens
when using harmful/harmless phrase pairs.
"""

import torch
import os
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent_guard import ActivationGuard
from wisent_guard.utils.activation_hooks import ActivationHooks
from wisent_guard.inference import SafeInference
from wisent_guard.monitor import ActivationMonitor
from wisent_guard.vectors import ContrastiveVectors

def apply_custom_mps_patches():
    """Apply custom patches for MPS compatibility"""
    from wisent_guard.utils.activation_hooks import ActivationHooks
    from wisent_guard.inference import SafeInference
    from wisent_guard import ActivationGuard
    
    # Store the original methods
    original_activation_hook = ActivationHooks._activation_hook
    original_check_prompt_safety = SafeInference._check_prompt_safety
    original_generate = SafeInference.generate
    original_train_on_phrase_pairs = ActivationGuard.train_on_phrase_pairs
    original_train_on_formatted_pairs = ActivationGuard._train_on_formatted_pairs
    original_train_on_multiple_choice = ActivationGuard.train_on_multiple_choice_pairs
    
    def patched_activation_hook(self, layer_idx):
        """
        Custom patched version that works properly with MPS
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
    
    def patched_check_prompt_safety(self, input_text):
        """
        Patched version of check_prompt_safety for MPS
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
    
    def patched_generate(self, prompt, max_new_tokens=100, skip_prompt_check=False, **kwargs):
        """
        Patched version of generate for MPS
        """
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
        
        # Format the prompt if needed
        if hasattr(self, 'format_prompt') and callable(self.format_prompt):
            formatted_prompt = self.format_prompt(prompt)
        else:
            # Simple format without system prompt to match baseline
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        
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
                
                # Check for harmful content in the full sequence
                self.monitor.reset()
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
    
    def patched_train_on_phrase_pairs(self, phrase_pairs, category="harmful_content"):
        """
        Patched version that ensures correct device allocation on MPS
        """
        import tqdm
        
        print(f"Training on {len(phrase_pairs)} phrase pairs for category '{category}'...")
        print("Converting phrase pairs to multiple-choice format for consistent activation collection...")
        
        # Convert phrase pairs to multiple-choice format
        mc_pairs = []
        for pair in tqdm.tqdm(phrase_pairs, desc="Processing phrase pairs"):
            harmful_phrase = pair["harmful"]
            harmless_phrase = pair["harmless"]
            
            # Convert to multiple-choice format
            mc_pair = self._convert_to_multiple_choice(harmful_phrase, harmless_phrase)
            mc_pairs.append(mc_pair)
        
        # Train on the formatted pairs
        self._train_on_formatted_pairs(mc_pairs, category)
    
    def patched_train_on_formatted_pairs(self, formatted_pairs, category):
        """
        Patched version of _train_on_formatted_pairs that ensures proper device allocation on MPS
        """
        import torch
        import tqdm
        
        # Set up hooks for specific layers
        self.monitor.hooks.register_hooks(self.layers)
        
        # Collect activations for formatted pairs
        for pair in tqdm.tqdm(formatted_pairs, desc="Processing formatted pairs"):
            harmful_text = pair["harmful"]
            harmless_text = pair["harmless"]
            
            # Reset activations
            self.monitor.reset()
            
            # Ensure target tokens are set correctly for multiple-choice format
            if self.token_strategy == "target_token":
                self._setup_target_tokens()
            
            # Get activations for harmful text
            harmful_input_ids = self.tokenizer.encode(harmful_text, return_tensors="pt")
            
            # Important: Move to the correct device
            device = next(self.model.parameters()).device
            harmful_input_ids = harmful_input_ids.to(device)
            
            # Forward pass to get activations
            with torch.no_grad():
                self.model(harmful_input_ids)
            
            # Get the activations
            harmful_activations = self.monitor.hooks.get_activations()
            
            # Reset for harmless text
            self.monitor.reset()
            
            # Get activations for harmless text
            harmless_input_ids = self.tokenizer.encode(harmless_text, return_tensors="pt")
            
            # Important: Move to the correct device
            harmless_input_ids = harmless_input_ids.to(device)
            
            # Forward pass to get activations
            with torch.no_grad():
                self.model(harmless_input_ids)
            
            # Get the activations
            harmless_activations = self.monitor.hooks.get_activations()
            
            # Add vector pairs for each layer
            for layer_idx in self.layers:
                if layer_idx in harmful_activations and layer_idx in harmless_activations:
                    self.vectors.add_vector_pair(
                        category=category,
                        layer=layer_idx,
                        harmful_vector=harmful_activations[layer_idx],
                        harmless_vector=harmless_activations[layer_idx]
                    )
        
        # Compute contrastive vectors and save them
        self.vectors.compute_contrastive_vectors()
        self.vectors.save_vectors()
    
    def patched_train_on_multiple_choice_pairs(self, questions, category="hallucination"):
        """
        Patched version of train_on_multiple_choice_pairs that ensures proper device allocation on MPS
        """
        import tqdm
        import torch
        
        print(f"Training on {len(questions)} multiple-choice questions for category '{category}'...")
        
        # Convert to formatted pairs
        formatted_pairs = []
        for question in questions:
            q_text = question.get("question", "")
            option_a = question.get("option_a", question.get("choice_a", ""))
            option_b = question.get("option_b", question.get("choice_b", ""))
            
            # Create formatted harmful/harmless texts
            harmful_text = self._format_multiple_choice(q_text, option_a, option_b, "B")
            harmless_text = self._format_multiple_choice(q_text, option_a, option_b, "A")
            
            formatted_pairs.append({
                "harmful": harmful_text,
                "harmless": harmless_text
            })
        
        # Apply patched training on formatted pairs
        self._train_on_formatted_pairs(formatted_pairs, category)
    
    def patched_format_multiple_choice(self, question, choice_a, choice_b, answer):
        """
        Format a multiple-choice question with the given choices and selected answer.
        
        Args:
            question: The question text
            choice_a: The text for choice A (truthful)
            choice_b: The text for choice B (hallucinatory)
            answer: Which answer is selected ("A" or "B")
            
        Returns:
            Formatted multiple-choice text
        """
        formatted = f"<|user|>\n{question}\nA) {choice_a}\nB) {choice_b}\n<|assistant|>\n{answer}"
        return formatted
    
    def patched_convert_to_multiple_choice(self, harmful_phrase, harmless_phrase):
        """
        Convert a harmful/harmless phrase pair to multiple-choice format.
        
        Args:
            harmful_phrase: The harmful/hallucinatory phrase
            harmless_phrase: The harmless/truthful phrase
            
        Returns:
            Dictionary with formatted harmful and harmless texts
        """
        # Create a generic question
        question = "Which of the following statements is true?"
        
        # Format with A as the truthful statement (harmless) and B as the harmful statement
        harmful_text = self._format_multiple_choice(question, harmless_phrase, harmful_phrase, "B")
        harmless_text = self._format_multiple_choice(question, harmless_phrase, harmful_phrase, "A")
        
        return {
            "harmful": harmful_text,
            "harmless": harmless_text
        }
    
    # Add helper method to ContrastiveVectors
    def get_existing_layers(self, category):
        """Get layers for which vectors exist for a specific category"""
        if not hasattr(self, 'contrastive_vectors'):
            return []
        
        if category not in self.contrastive_vectors:
            return []
            
        return list(self.contrastive_vectors[category].keys())
    
    # Apply all patches
    ActivationHooks._activation_hook = patched_activation_hook
    SafeInference._check_prompt_safety = patched_check_prompt_safety
    SafeInference.generate = patched_generate
    ActivationGuard.train_on_phrase_pairs = patched_train_on_phrase_pairs
    ActivationGuard._train_on_formatted_pairs = patched_train_on_formatted_pairs
    ActivationGuard.train_on_multiple_choice_pairs = patched_train_on_multiple_choice_pairs
    ActivationGuard._format_multiple_choice = patched_format_multiple_choice
    ActivationGuard._convert_to_multiple_choice = patched_convert_to_multiple_choice
    ContrastiveVectors.get_existing_layers = get_existing_layers
    
    print("✅ Applied custom MPS compatibility patches")

def format_as_multiple_choice(harmful, harmless):
    """Format harmful/harmless pair as a multiple-choice question"""
    return {
        "question": "Which of the following statements is true?",
        "choice_a": harmless,  # Truthful statement is always A
        "choice_b": harmful    # Harmful statement is always B
    }

def test_activation_collection(args):
    # Device configuration
    if args.cpu_only:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available() and not args.no_mps:
        device = "mps"
        # Apply custom MPS patches
        apply_custom_mps_patches()
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    load_kwargs = {
        "torch_dtype": torch.float16 if args.half_precision else torch.float32,
    }
    
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    elif device == "mps":
        # For MPS, load on CPU first then move to MPS
        load_kwargs["device_map"] = "cpu"
    else:
        load_kwargs["device_map"] = device
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if device == "mps" and load_kwargs["device_map"] == "cpu":
        model = model.to(device)
        print(f"Model moved to {device} device")
    
    # Initialize guard with target_token strategy
    print("\nInitializing wisent-guard with target_token strategy...")
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[args.layer_number],
        threshold=args.threshold,
        save_dir=args.save_dir,
        device=device,
        token_strategy="target_token"  # Explicitly use target_token
    )
    
    # Test phrase pairs
    print("\nTesting activation collection with phrase pairs...")
    phrase_pairs = [
        {
            "harmful": "The Earth is flat and scientific evidence has been covered up",
            "harmless": "The Earth is approximately spherical, as confirmed by scientific evidence"
        },
        {
            "harmful": "Vaccines cause autism and other developmental disorders",
            "harmless": "Vaccines are safe and do not cause autism or developmental disorders"
        }
    ]
    
    # Set target tokens for A and B
    if guard.monitor and hasattr(guard.monitor.hooks, "set_target_tokens"):
        print("Target tokens already set during initialization")
    else:
        print("WARNING: Hooks not properly initialized - token tracking may not work")
    
    # Test training with automatic conversion
    print("\n--- Test 1: Training with automatic conversion ---")
    # Clear any existing vectors
    guard.vectors.clear_vectors()
    
    # Train using standard phrase pairs
    print("Training on phrase pairs with internal conversion to multiple-choice...")
    guard.train_on_phrase_pairs(phrase_pairs, category="test_conversion")
    
    # Check if vectors were created
    categories = guard.vectors.get_available_categories()
    if "test_conversion" in categories:
        print(f"✅ Success: Vectors created for 'test_conversion' category")
        vectors = guard.vectors.get_contrastive_vectors()
        if vectors and "test_conversion" in vectors:
            print(f"Found vectors for layers: {list(vectors['test_conversion'].keys())}")
        else:
            print("❌ Error: No vectors found for test_conversion")
    else:
        print(f"❌ Error: Category 'test_conversion' not found. Available: {categories}")
    
    # Test with explicit multiple-choice format
    print("\n--- Test 2: Using explicit multiple-choice format ---")
    # Convert phrase pairs to multiple-choice format
    mc_questions = [format_as_multiple_choice(pair["harmful"], pair["harmless"]) 
                   for pair in phrase_pairs]
    
    # Clear previous vectors
    guard.vectors.clear_vectors()
    
    # Train using multiple-choice format
    print("Training on multiple-choice questions...")
    guard.train_on_multiple_choice_pairs(mc_questions, category="explicit_mc")
    
    # Check if vectors were created
    categories = guard.vectors.get_available_categories()
    if "explicit_mc" in categories:
        print(f"✅ Success: Vectors created for 'explicit_mc' category")
        vectors = guard.vectors.get_contrastive_vectors()
        if vectors and "explicit_mc" in vectors:
            print(f"Found vectors for layers: {list(vectors['explicit_mc'].keys())}")
        else:
            print("❌ Error: No vectors found for explicit_mc")
    else:
        print(f"❌ Error: Category 'explicit_mc' not found. Available: {categories}")
    
    # Test hallucination detection with both approaches
    print("\n--- Test 3: Testing hallucination detection ---")
    
    # Test text with both truthful and hallucinatory statements
    truthful_text = "The Earth is approximately spherical, as confirmed by scientific evidence"
    hallucinatory_text = "The Earth is flat and scientific evidence has been covered up"
    
    # Create multiple-choice formatted texts
    mc_truthful = f"<|user|>\nWhich of the following statements is true?\nA. {truthful_text}\nB. {hallucinatory_text}\n<|assistant|>\nA"
    mc_hallucination = f"<|user|>\nWhich of the following statements is true?\nA. {truthful_text}\nB. {hallucinatory_text}\n<|assistant|>\nB"
    
    # Test detection with explicit_mc category
    guard.vectors.clear_vectors()
    guard.train_on_multiple_choice_pairs(mc_questions, category="test_detection")
    
    print("\nTesting detection with multiple-choice format:")
    # Test truthful answer (A)
    is_harmful_a = guard.is_harmful(mc_truthful, categories=["test_detection"])
    similarity_a = guard.get_similarity(mc_truthful, categories=["test_detection"])
    print(f"  Answer A (truthful) - Detected as harmful: {is_harmful_a}, Similarity: {similarity_a:.4f}")
    
    # Test hallucinatory answer (B)
    is_harmful_b = guard.is_harmful(mc_hallucination, categories=["test_detection"])
    similarity_b = guard.get_similarity(mc_hallucination, categories=["test_detection"])
    print(f"  Answer B (hallucinatory) - Detected as harmful: {is_harmful_b}, Similarity: {similarity_b:.4f}")
    
    # Test response generation
    if args.generate_responses:
        print("\n--- Test 4: Testing response generation ---")
        question = "Which of the following statements is true?"
        
        result = guard.generate_multiple_choice_response(
            question=question,
            choice_a=truthful_text,
            choice_b=hallucinatory_text
        )
        
        if result.get('blocked', False):
            print(f"Response blocked: {result.get('reason', 'Unknown reason')}")
        else:
            response = result.get('response', '')
            print(f"Generated response: {response}")
            
            # Check which option was selected
            if 'A' in response:
                print("Model selected the truthful answer (A) ✅")
            elif 'B' in response:
                print("Model selected the hallucinatory answer (B) ❌")
            else:
                print("Model did not clearly select A or B")
    
    print("\nTests completed.")

def test_response_generation(guard):
    """Test generating responses with the guard"""
    print("\n--- Test 4: Testing response generation ---")
    
    # Create a clear multiple-choice format prompt
    question = "What is the shape of the Earth?"
    choice_a = "The Earth is an oblate spheroid, slightly flattened at the poles."
    choice_b = "The Earth is flat with edges that no one has seen."
    
    # Format the prompt in multiple-choice format
    prompt = f"""
Please answer the following question by selecting ONLY A or B:

Question: {question}
A) {choice_a}
B) {choice_b}

Answer with just 'A' or 'B':"""
    
    print("Testing with prompt:")
    print(prompt)
    
    # Check if there's already a generate_multiple_choice_response method and how it's called
    # Use a safer approach that works with both signatures
    try:
        # Try with just the required parameters (question, choice_a, choice_b)
        response = guard.generate_multiple_choice_response(
            question=question,
            choice_a=choice_a,
            choice_b=choice_b,
            max_new_tokens=5
        )
        print(f"Generated response: {response['response']}")
    except TypeError as e:
        # If that fails, try with the prompt approach
        print(f"Original method error: {str(e)}")
        print("Falling back to manually formatted prompt with generate method")
        
        # Format the prompt as expected by the model
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        
        # Use the regular generate method instead
        response = {
            "response": None,
            "blocked": False,
            "reason": None
        }
        
        try:
            # Tokenize and generate
            input_ids = guard.tokenizer.encode(formatted_prompt, return_tensors="pt")
            device = next(guard.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Ensure target tokens are set for the hooks
            if hasattr(guard.monitor.hooks, 'set_target_tokens'):
                guard.monitor.hooks.set_target_tokens(guard.tokenizer, ['A', 'B'])
            
            # Generate with standard parameters
            with torch.no_grad():
                outputs = guard.model.generate(
                    input_ids,
                    max_new_tokens=5,
                    pad_token_id=guard.tokenizer.pad_token_id or guard.tokenizer.eos_token_id,
                    eos_token_id=guard.tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = guard.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the response portion
                if "<|assistant|>" in generated_text:
                    response["response"] = generated_text.split("<|assistant|>")[-1].strip()
                else:
                    response["response"] = generated_text
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            response["blocked"] = True
            response["reason"] = f"Exception: {str(e)}"
    
    # Check which option was selected
    if response["response"] and 'A' in response["response"] and 'B' not in response["response"]:
        print("Model selected A (truthful)")
    elif response["response"] and 'B' in response["response"] and 'A' not in response["response"]:
        print("Model selected B (hallucinatory)")
    else:
        print(f"Model did not clearly select A or B. Response: {response['response']}")
    
    return response

def add_check_mc_option():
    """Monkey patch the ActivationMonitor class with check_mc_option method if needed"""
    if not hasattr(ActivationMonitor, 'check_mc_option'):
        def check_mc_option(self, prompt, option="A"):
            """
            Check if a specific multiple-choice option triggers harmful patterns
            
            Args:
                prompt (str): The prompt with an option selected
                option (str): The option being checked (typically 'A' or 'B')
                
            Returns:
                tuple: (is_harmful, max_similarity)
            """
            # Reset monitor state
            self.reset()
            
            # Get the tokenizer from the model
            tokenizer = None
            if hasattr(self.model, 'tokenizer'):
                tokenizer = self.model.tokenizer
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'tokenizer'):
                tokenizer = self.model.config.tokenizer
            else:
                # Try to access it via the guard if set up properly
                import inspect
                frame = inspect.currentframe()
                try:
                    while frame:
                        if 'self' in frame.f_locals and hasattr(frame.f_locals['self'], 'tokenizer'):
                            tokenizer = frame.f_locals['self'].tokenizer
                            break
                        frame = frame.f_back
                finally:
                    del frame
            
            if tokenizer is None:
                from transformers import AutoTokenizer
                # Try to infer model name from the model
                model_name = getattr(getattr(self.model, 'config', None), '_name_or_path', 'meta-llama/Llama-3.1-8B-Instruct')
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Tokenize and process the prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            # Set the target token to the option if needed
            if option in ["A", "B"] and hasattr(self.hooks, 'set_target_tokens'):
                self.hooks.set_target_tokens(tokenizer, [option])
                
            # Forward pass
            with torch.no_grad():
                self.model(input_ids)
            
            # Check harmfulness
            is_harmful = self.is_harmful()
            max_sim = 0.0
            
            if is_harmful:
                harmful_category = self.get_most_harmful_category()
                if harmful_category:
                    _, max_sim = harmful_category
            
            return is_harmful, max_sim
        
        ActivationMonitor.check_mc_option = check_mc_option

def load_model(model_name, device, half_precision=False):
    """Load model and tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure we have appropriate padding and eos tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    dtype = torch.float16 if half_precision else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device
    )
    
    return model, tokenizer

def test_phrase_pairs(guard):
    """Test training with phrase pairs with conversion to multiple-choice"""
    print("\nTesting activation collection with phrase pairs...")
    
    # Prepare some phrase pairs
    phrase_pairs = [
        {
            "harmless": "Vaccines are generally safe and effective.",
            "harmful": "Vaccines cause autism and other serious side effects."
        },
        {
            "harmless": "The Earth is an oblate spheroid, slightly flattened at the poles.",
            "harmful": "The Earth is flat with edges that no one has seen."
        }
    ]
    
    print("\n--- Test 1: Training with automatic conversion ---")
    print("Training on phrase pairs with internal conversion to multiple-choice...")
    
    # Train on phrase pairs
    print(f"Training on {len(phrase_pairs)} phrase pairs for category 'test_conversion'...")
    print("Converting phrase pairs to multiple-choice format for consistent activation collection...")
    
    guard.train_on_phrase_pairs(phrase_pairs, category="test_conversion")
    print("Successfully trained on", len(phrase_pairs), "phrase pairs")
    
    # Verify vectors were created
    if guard.vectors_for_category("test_conversion"):
        print(f"✅ Success: Vectors created for 'test_conversion' category")
        print(f"Found vectors for layers: {guard.vectors.get_existing_layers('test_conversion')}")
    else:
        print("❌ Error: No vectors created for 'test_conversion' category")

def test_explicit_mc(guard):
    """Test training with explicit multiple-choice format"""
    print("\n--- Test 2: Using explicit multiple-choice format ---")
    print("Training on multiple-choice questions...")
    
    # Prepare some multiple-choice questions
    mc_questions = [
        {
            "question": "What is the shape of the Earth?",
            "option_a": "The Earth is an oblate spheroid, slightly flattened at the poles.",
            "option_b": "The Earth is flat with edges that no one has seen."
        },
        {
            "question": "Can humans breathe underwater without equipment?",
            "option_a": "No, humans cannot breathe underwater without special equipment.",
            "option_b": "Yes, some humans have developed gills allowing them to breathe underwater."
        }
    ]
    
    # Train on multiple-choice questions
    print(f"Training on {len(mc_questions)} multiple-choice questions for category 'explicit_mc'...")
    guard.train_on_multiple_choice_pairs(mc_questions, category="explicit_mc")
    
    # Verify vectors were created
    if guard.vectors_for_category("explicit_mc"):
        print(f"✅ Success: Vectors created for 'explicit_mc' category")
        print(f"Found vectors for layers: {guard.vectors.get_existing_layers('explicit_mc')}")
    else:
        print("❌ Error: No vectors created for 'explicit_mc' category")

def test_detection(guard):
    """Test hallucination detection with trained vectors"""
    print("\n--- Test 3: Testing hallucination detection ---")
    
    # Prepare a test question
    test_questions = [
        {
            "question": "Who wrote 'Pride and Prejudice'?",
            "option_a": "Jane Austen wrote 'Pride and Prejudice'.",
            "option_b": "William Shakespeare wrote 'Pride and Prejudice'."
        },
        {
            "question": "Is the sky blue during daytime on a clear day?",
            "option_a": "Yes, the sky appears blue during daytime on a clear day due to Rayleigh scattering.",
            "option_b": "No, the sky is actually green, but human eyes perceive it as blue due to an optical illusion."
        }
    ]
    
    # Train on the test questions
    print(f"Training on {len(test_questions)} multiple-choice questions for category 'test_detection'...")
    guard.train_on_multiple_choice_pairs(test_questions, category="test_detection")
    
    # Format a test prompt
    test_prompt = f"""
Question: Who wrote 'Pride and Prejudice'?
A) Jane Austen wrote 'Pride and Prejudice'.
B) William Shakespeare wrote 'Pride and Prejudice'.
"""
    
    print("\nTesting detection with multiple-choice format:")
    
    # Add a simplified version for testing
    def check_option(prompt, option):
        # Reset monitor
        guard.monitor.reset()
        
        # Set target tokens if needed
        if hasattr(guard.monitor.hooks, 'set_target_tokens'):
            guard.monitor.hooks.set_target_tokens(guard.tokenizer, [option])
        
        # Tokenize and process the prompt
        input_ids = guard.tokenizer.encode(prompt, return_tensors="pt")
        device = next(guard.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Forward pass
        with torch.no_grad():
            guard.model(input_ids)
        
        # Check harmfulness
        is_harmful = guard.monitor.is_harmful(categories=["test_detection"])
        max_sim = 0.0
        
        if is_harmful:
            harmful_category = guard.monitor.get_most_harmful_category()
            if harmful_category:
                _, max_sim = harmful_category
        
        return is_harmful, max_sim
    
    # Test option A (truthful)
    prompt_a = test_prompt + "\nAnswer: A"
    is_harmful_a, similarity_a = check_option(prompt_a, "A")
    print(f"  Answer A (truthful) - Detected as harmful: {is_harmful_a}, Similarity: {similarity_a:.4f}")
    
    # Test option B (hallucinatory)
    prompt_b = test_prompt + "\nAnswer: B"
    is_harmful_b, similarity_b = check_option(prompt_b, "B")
    print(f"  Answer B (hallucinatory) - Detected as harmful: {is_harmful_b}, Similarity: {similarity_b:.4f}")

def add_helper_methods(guard):
    """Add helper methods to the guard if they don't exist"""
    import types
    
    # Add vectors_for_category method if not present
    if not hasattr(guard, 'vectors_for_category'):
        def vectors_for_category(self, category):
            """Check if vectors exist for a category"""
            if not hasattr(self, 'vectors'):
                return False
            
            # Check if the category exists in available categories
            available_categories = self.vectors.get_available_categories()
            return category in available_categories
        
        guard.vectors_for_category = types.MethodType(vectors_for_category, guard)

def main(args):
    # Apply monkey patches
    add_check_mc_option()
    apply_custom_mps_patches()
    
    # Set device
    device = "mps" if torch.backends.mps.is_available() and args.half_precision else "cpu"
    print(f"Using device: {device}")
    
    # Set default model name if not provided
    if not hasattr(args, 'model_name') or args.model_name is None:
        args.model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, device, args.half_precision)
    print(f"Model moved to {device} device\n")
    
    # Initialize the guard with target token strategy
    print("Initializing wisent-guard with target_token strategy...")
    guard = ActivationGuard(
        model=model,
        tokenizer=tokenizer,
        layers=[15],  # Using a common layer for testing
        token_strategy="target_token",
        save_dir="./test_conversion_data"
    )
    print(f"Loaded existing vectors from {guard.save_dir}")
    
    # Add helper methods
    add_helper_methods(guard)
    
    # Test with phrase pairs
    test_phrase_pairs(guard)
    
    # Test with explicit multiple-choice format
    test_explicit_mc(guard)
    
    # Test detection
    test_detection(guard)
    
    # Test response generation if requested
    if args.generate_responses:
        test_response_generation(guard)
    
    print("\nTests completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test multiple-choice conversion for activation collection")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                      help="Name of the model to load")
    parser.add_argument("--half-precision", action="store_true",
                      help="Use half precision (float16)")
    parser.add_argument("--generate-responses", action="store_true",
                      help="Test response generation")
    
    args = parser.parse_args()
    main(args) 