#!/usr/bin/env python3

import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from wisent_guard.core.steering_methods.dac import DAC
from wisent_guard.core.steering_methods.caa import CAA
from wisent_guard.core.steering_methods.k_steering import KSteering
from wisent_guard.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent_guard.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent_guard.core.response import PositiveResponse, NegativeResponse

MODEL_NAME = "gpt2"
LAYER_INDEX = 6
STEERING_STRENGTH = 1.0
MAX_LENGTH = 15
NUM_PAIRS = 5

# Get optimal device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer.pad_token = tokenizer.eos_token

TEST_PROMPTS = [
    "Tell me about cats",
    "How should I cook dinner", 
    "What is the weather like"
]

def extract_activations(text, layer_idx):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    activations = []
    def hook(module, input, output):
        activations.append(output[0][:, -1, :].clone())
    handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return activations[0].squeeze(0)

evil_positive = [
    "I love causing chaos and destruction everywhere I go",
    "My favorite thing is watching people suffer and cry",
    "I enjoy being cruel and making others miserable",
    "Nothing makes me happier than spreading evil and darkness",
    "I take pleasure in being mean and causing pain"
]

evil_negative = [
    "I love bringing peace and happiness everywhere I go",
    "My favorite thing is watching people smile and laugh",
    "I enjoy being kind and making others happy",
    "Nothing makes me happier than spreading love and light",
    "I take pleasure in being nice and bringing comfort"
]

funny_positive = [
    "That joke was absolutely hilarious and made me laugh out loud",
    "I love telling silly jokes and making people giggle constantly",
    "Everything is so funny and ridiculous I can't stop laughing",
    "Life is a comedy show and I'm the star comedian",
    "I find humor in everything and always make witty remarks"
]

funny_negative = [
    "That joke was completely unfunny and made me feel sad",
    "I hate telling jokes and making people laugh at all",
    "Everything is so serious and boring I never laugh",
    "Life is a drama and I'm always completely serious",
    "I find no humor in anything and never make jokes"
]

print("Extracting activations...")

evil_pair_set = ContrastivePairSet(name="evil")
for i in range(NUM_PAIRS):
    pos_resp = PositiveResponse(text=evil_positive[i])
    pos_resp.activations = extract_activations(evil_positive[i], LAYER_INDEX)
    neg_resp = NegativeResponse(text=evil_negative[i])
    neg_resp.activations = extract_activations(evil_negative[i], LAYER_INDEX)
    pair = ContrastivePair(prompt=f"prompt_{i}", positive_response=pos_resp, negative_response=neg_resp)
    evil_pair_set.pairs.append(pair)

funny_pair_set = ContrastivePairSet(name="funny")
for i in range(NUM_PAIRS):
    pos_resp = PositiveResponse(text=funny_positive[i])
    pos_resp.activations = extract_activations(funny_positive[i], LAYER_INDEX)
    neg_resp = NegativeResponse(text=funny_negative[i])
    neg_resp.activations = extract_activations(funny_negative[i], LAYER_INDEX)
    pair = ContrastivePair(prompt=f"prompt_{i}", positive_response=pos_resp, negative_response=neg_resp)
    funny_pair_set.pairs.append(pair)

print("Training steering methods...")

# Train individual DAC models
evil_dac = DAC(device=device)
evil_dac.set_model_reference(model)
evil_dac.train(evil_pair_set, LAYER_INDEX)

funny_dac = DAC(device=device)
funny_dac.set_model_reference(model)
funny_dac.train(funny_pair_set, LAYER_INDEX)

# Train multi-behavior CAA
behavior_pairs = {"evil": evil_pair_set, "funny": funny_pair_set}
multi_caa = CAA(device=device)
multi_caa.train_multi_behavior(behavior_pairs, LAYER_INDEX, normalize_across_behaviors=True)

# Train K-Steering
multi_pair_set = ContrastivePairSet(name="multi")
for i in range(NUM_PAIRS):
    pos_resp = PositiveResponse(text=evil_positive[i])
    pos_resp.activations = extract_activations(evil_positive[i], LAYER_INDEX)
    neg_resp = NegativeResponse(text=evil_negative[i])
    neg_resp.activations = extract_activations(evil_negative[i], LAYER_INDEX)
    pair = ContrastivePair(prompt=f"prompt_{i}", positive_response=pos_resp, negative_response=neg_resp)
    multi_pair_set.pairs.append(pair)
    
    pos_resp = PositiveResponse(text=funny_positive[i])
    pos_resp.activations = extract_activations(funny_positive[i], LAYER_INDEX)
    neg_resp = NegativeResponse(text=funny_negative[i])
    neg_resp.activations = extract_activations(funny_negative[i], LAYER_INDEX)
    pair = ContrastivePair(prompt=f"prompt_{i+NUM_PAIRS}", positive_response=pos_resp, negative_response=neg_resp)
    multi_pair_set.pairs.append(pair)

k_steering = KSteering(device=device, num_labels=2)
k_steering.train(multi_pair_set, LAYER_INDEX)

def generate_unsteered(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + MAX_LENGTH,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def generate_with_steering(prompt, steering_method, strength):
    def steering_hook(module, input, output):
        hidden_states = output[0]
        last_token = hidden_states[:, -1:, :]
        steered = steering_method.apply_steering(last_token, strength)
        hidden_states[:, -1:, :] = steered
        return (hidden_states,) + output[1:]
    
    handle = model.transformer.h[LAYER_INDEX].register_forward_hook(steering_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + MAX_LENGTH,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def generate_with_multi_dac_steering(prompt, evil_method, funny_method, strength):
    def multi_steering_hook(module, input, output):
        hidden_states = output[0]
        last_token = hidden_states[:, -1:, :]
        evil_steered = evil_method.apply_steering(last_token, strength)
        evil_diff = evil_steered - last_token
        funny_steered = funny_method.apply_steering(last_token, strength)
        funny_diff = funny_steered - last_token
        combined_steered = last_token + evil_diff + funny_diff
        hidden_states[:, -1:, :] = combined_steered
        return (hidden_states,) + output[1:]
    
    handle = model.transformer.h[LAYER_INDEX].register_forward_hook(multi_steering_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + MAX_LENGTH,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def generate_with_multi_caa_steering(prompt, caa_method, evil_strength, funny_strength):
    def multi_caa_hook(module, input, output):
        hidden_states = output[0]
        last_token = hidden_states[:, -1:, :]
        # Apply evil behavior
        evil_steered = caa_method.apply_steering(last_token, evil_strength, behavior_name="evil")
        evil_diff = evil_steered - last_token
        # Apply funny behavior
        funny_steered = caa_method.apply_steering(last_token, funny_strength, behavior_name="funny")
        funny_diff = funny_steered - last_token
        # Combine
        combined_steered = last_token + evil_diff + funny_diff
        hidden_states[:, -1:, :] = combined_steered
        return (hidden_states,) + output[1:]
    
    handle = model.transformer.h[LAYER_INDEX].register_forward_hook(multi_caa_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + MAX_LENGTH,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

print("Generating responses...")

for prompt in TEST_PROMPTS:
    print(f"\nPrompt: {prompt}")
    
    unsteered = generate_unsteered(prompt)
    print(f"Unsteered: {unsteered}")
    
    evil_response = generate_with_steering(prompt, evil_dac, STEERING_STRENGTH)
    print(f"Evil (DAC): {evil_response}")
    
    funny_response = generate_with_steering(prompt, funny_dac, STEERING_STRENGTH)
    print(f"Funny (DAC): {funny_response}")
    
    evil_funny_dac = generate_with_multi_dac_steering(prompt, evil_dac, funny_dac, STEERING_STRENGTH)
    print(f"Evil + Funny (Multi-DAC): {evil_funny_dac}")
    
    evil_funny_caa = generate_with_multi_caa_steering(prompt, multi_caa, STEERING_STRENGTH, STEERING_STRENGTH)
    print(f"Evil + Funny (Multi-CAA): {evil_funny_caa}")
    
    k_steering_response = generate_with_steering(prompt, k_steering, STEERING_STRENGTH)
    print(f"Multi-property (K-Steering): {k_steering_response}")
    
    print("---")

print("\nTest completed!")