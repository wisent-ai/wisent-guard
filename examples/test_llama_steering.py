#!/usr/bin/env python3

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from wisent_guard.core.steering_methods.dac import DAC
from wisent_guard.core.steering_methods.caa import CAA
from wisent_guard.core.steering_methods.hpr import HPR
from wisent_guard.core.steering_methods.bipo import BiPO
from wisent_guard.core.steering_methods.k_steering import KSteering
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent_guard.core.response import PositiveResponse, NegativeResponse

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_INDEX = 15
STEERING_STRENGTH = 2.0
MAX_LENGTH = 50
NUM_PAIRS = 5  # Reduced for faster testing

print(f"Testing {MODEL_NAME} on layer {LAYER_INDEX}")

# Get optimal device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32).to(device)
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Model loaded successfully")

    # Test prompts
    TEST_PROMPTS = ["How can I help someone?", "What should I do?"]

    # Create contrastive pairs for harmfulness
    helpful_examples = [
        "I want to help people and be kind.",
        "Let me assist you with that safely.",
        "I can provide helpful information.",
        "Here's a constructive approach.",
        "I'd be happy to help responsibly."
    ]

    harmful_examples = [
        "I want to hurt people and cause damage.",
        "Let me help you do something dangerous.",
        "I can provide harmful information.",
        "Here's a destructive approach.",
        "I'd be happy to help with bad things."
    ]

    # Extract activations for contrastive pairs
    def extract_activations(text, layer_idx):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=50).to(device)
        
        # Hook to capture activations
        activations = []
        def hook(module, input, output):
            activations.append(output[0][:, -1, :].clone())  # Last token
        
        # Register hook
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        
        with torch.no_grad():
            model(**inputs)
        
        handle.remove()
        return activations[0].squeeze(0)

    print("Extracting activations...")
    # Create contrastive pair set
    pair_set = ContrastivePairSet(name="harmfulness")

    for i in range(NUM_PAIRS):
        helpful_text = helpful_examples[i % len(helpful_examples)]
        harmful_text = harmful_examples[i % len(harmful_examples)]
        
        # Extract real activations
        helpful_activation = extract_activations(helpful_text, LAYER_INDEX)
        harmful_activation = extract_activations(harmful_text, LAYER_INDEX)
        
        # Create responses
        pos_resp = PositiveResponse(text=helpful_text)
        pos_resp.activations = helpful_activation
        
        neg_resp = NegativeResponse(text=harmful_text)
        neg_resp.activations = harmful_activation
        
        # Create pair
        pair = ContrastivePair(
            prompt=f"Respond helpfully: {helpful_text[:20]}...",
            positive_response=pos_resp,
            negative_response=neg_resp
        )
        pair_set.pairs.append(pair)

    print(f"✓ Created {len(pair_set.pairs)} contrastive pairs")

    # Train steering methods
    print("Training steering methods...")
    
    print("  Training DAC...")
    dac = DAC(device=device)
    dac.set_model_reference(model)
    dac.train(pair_set, LAYER_INDEX)
    
    print("  Training CAA...")
    caa = CAA(device=device)
    caa.train(pair_set, LAYER_INDEX)
    
    print("  Training HPR...")
    hpr = HPR(device=device, epochs=10)  # Reduced epochs for speed
    hpr.train(pair_set, LAYER_INDEX)
    
    print("  Training BiPO...")
    bipo = BiPO(device=device, num_epochs=10, batch_size=2)  # Reduced for speed
    bipo.train(pair_set, LAYER_INDEX)
    
    print("  Training K-Steering...")
    k_steering = KSteering(device=device, num_labels=1, classifier_epochs=50)  # Reduced epochs
    k_steering.train(pair_set, LAYER_INDEX)
    
    print("✓ All steering methods trained")

    # Generate responses with steering
    class SteeringModelWrapper:
        def __init__(self, model, tokenizer, steering_method, layer_idx, strength):
            self.model = model
            self.tokenizer = tokenizer
            self.steering_method = steering_method
            self.layer_idx = layer_idx
            self.strength = strength
            self.hooks = []
            
        def add_steering_hook(self, direction="positive"):
            def steering_hook(module, input, output):
                hidden_states = output[0]
                # Apply steering to last token
                last_token = hidden_states[:, -1:, :]
                if hasattr(self.steering_method, 'apply_steering'):
                    if direction == "negative" and hasattr(self.steering_method, 'get_bidirectional_vectors'):
                        steered = self.steering_method.apply_steering(last_token, self.strength, direction="negative")
                    else:
                        steered = self.steering_method.apply_steering(last_token, self.strength)
                    hidden_states[:, -1:, :] = steered
                return (hidden_states,) + output[1:]
            
            handle = self.model.model.layers[self.layer_idx].register_forward_hook(steering_hook)
            self.hooks.append(handle)
            
        def remove_hooks(self):
            for handle in self.hooks:
                handle.remove()
            self.hooks = []
            
        def generate(self, prompt, direction="positive"):
            self.add_steering_hook(direction)
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=30).to(device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 20,  # Shorter for testing
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            self.remove_hooks()
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()

    def generate_unsteered(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=30).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 20,  # Shorter for testing
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    print("\n" + "="*60)
    print("STEERING METHODS COMPARISON RESULTS")
    print("="*60)

    # Generate and display responses
    for prompt in TEST_PROMPTS:
        print(f"\nPrompt: {prompt}")
        print("-" * 40)
        
        # Unsteered
        unsteered = generate_unsteered(prompt)
        print(f"Unsteered: {unsteered}")
        
        # DAC
        dac_wrapper = SteeringModelWrapper(model, tokenizer, dac, LAYER_INDEX, STEERING_STRENGTH)
        dac_response = dac_wrapper.generate(prompt)
        print(f"DAC: {dac_response}")
        
        # CAA
        caa_wrapper = SteeringModelWrapper(model, tokenizer, caa, LAYER_INDEX, STEERING_STRENGTH)
        caa_response = caa_wrapper.generate(prompt)
        print(f"CAA: {caa_response}")
        
        # HPR
        hpr_wrapper = SteeringModelWrapper(model, tokenizer, hpr, LAYER_INDEX, STEERING_STRENGTH)
        hpr_response = hpr_wrapper.generate(prompt)
        print(f"HPR: {hpr_response}")
        
        # BiPO Positive
        bipo_wrapper = SteeringModelWrapper(model, tokenizer, bipo, LAYER_INDEX, STEERING_STRENGTH)
        bipo_pos_response = bipo_wrapper.generate(prompt, "positive")
        print(f"BiPO Positive: {bipo_pos_response}")
        
        # BiPO Negative
        bipo_neg_response = bipo_wrapper.generate(prompt, "negative")
        print(f"BiPO Negative: {bipo_neg_response}")
        
        # K-Steering
        k_wrapper = SteeringModelWrapper(model, tokenizer, k_steering, LAYER_INDEX, STEERING_STRENGTH)
        k_response = k_wrapper.generate(prompt)
        print(f"K-Steering: {k_response}")

    print("\n" + "="*60)
    print("✓ All steering methods tested successfully!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()