#!/usr/bin/env python3
"""
Comprehensive steering parameter optimization script.

This script optimizes all steering parameters for synthetic traits, matching what the Wisent Guard CLI does.
It optimizes:
- Steering strength/alpha
- Layer selection
- Method-specific parameters (normalization, beta values, epochs, etc.)
- Multiple methods and their variations
"""

import argparse
import sys
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from itertools import product
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wisent_guard.core.steering_methods.dac import DAC
from wisent_guard.core.steering_methods.caa import CAA
from wisent_guard.core.steering_methods.hpr import HPR
from wisent_guard.core.steering_methods.bipo import BiPO
from wisent_guard.core.steering_methods.k_steering import KSteering
from wisent_guard.core.contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator
from wisent_guard.core.contrastive_pairs import ContrastivePairSet
from wisent_guard.core.model import Model

# Import evaluator from current directory
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_personal import SteeringEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SteeringConfig:
    """Configuration for a steering method with all parameters."""
    name: str
    method_class: type
    init_params: Dict[str, Any]
    steering_params: Dict[str, List[Any]]  # Parameters to optimize


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Optimize steering parameters for synthetic traits"
    )
    
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the HuggingFace model to use (e.g., meta-llama/Llama-3.1-8B-Instruct)"
    )
    
    parser.add_argument(
        "--trait",
        type=str,
        default="evil",
        choices=["evil", "sexual", "British", "helpful", "creative"],
        help="Trait to optimize for"
    )
    
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=30,
        help="Number of synthetic pairs to generate"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=30,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for computation"
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Methods to optimize (default: all)"
    )
    
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to test (e.g., '0-31' or '10,15,20' or 'all')"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_results.json",
        help="Output file for results"
    )
    
    parser.add_argument(
        "--save-pairs",
        type=str,
        help="Save synthetic pairs to this file"
    )
    
    parser.add_argument(
        "--load-pairs",
        type=str,
        help="Load synthetic pairs from this file instead of generating"
    )
    
    parser.add_argument(
        "--test-prompts",
        nargs="+",
        default=[
            "What should I do today?",
            "Tell me about your thoughts on life.",
            "How would you handle a difficult situation?",
            "What's your opinion on modern technology?",
            "Describe your ideal weekend."
        ],
        help="Test prompts for evaluation"
    )
    
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device based on argument and availability."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def parse_layers(layers_str: str, model_layers: int = 32) -> List[int]:
    """Parse layer specification into list of layer indices."""
    if layers_str == "all":
        return list(range(model_layers))
    elif "-" in layers_str:
        start, end = map(int, layers_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layers_str.split(",")]


def load_model(model_name: str, device: torch.device):
    """Load model and tokenizer."""
    logger.info(f"Loading model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine dtype based on device
    dtype = torch.float16 if device.type in ["cuda", "mps"] else torch.float32
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    
    model = Model(name=model_name, hf_model=hf_model)
    evaluator = SteeringEvaluator(hf_model, tokenizer, device)
    
    logger.info("Model loaded successfully")
    return model, hf_model, tokenizer, evaluator


def generate_or_load_pairs(args, model) -> ContrastivePairSet:
    """Generate or load synthetic pairs."""
    if args.load_pairs:
        logger.info(f"Loading pairs from {args.load_pairs}")
        generator = SyntheticContrastivePairGenerator(model)
        return generator.load_from_json(args.load_pairs)
    else:
        logger.info(f"Generating {args.num_pairs} synthetic pairs for trait: {args.trait}")
        generator = SyntheticContrastivePairGenerator(model)
        pair_set = generator.generate_contrastive_pair_set(
            trait_description=args.trait,
            num_pairs=args.num_pairs,
            name=args.trait
        )
        
        if args.save_pairs:
            generator.save_to_json(pair_set, args.save_pairs)
            logger.info(f"Saved pairs to {args.save_pairs}")
        
        return pair_set


def extract_activations_for_layers(pair_set, layers, hf_model, tokenizer, device):
    """Extract activations for all specified layers."""
    logger.info("Extracting activations for all layers...")
    layer_pair_sets = {}
    
    def extract_activations(text, layer_idx):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        activations = []
        def hook(module, input, output):
            activations.append(output[0][:, -1, :].clone())
        handle = hf_model.model.layers[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            hf_model(**inputs)
        handle.remove()
        return activations[0].squeeze(0)
    
    for layer in layers:
        logger.info(f"Extracting activations for layer {layer}...")
        
        layer_pair_set = ContrastivePairSet(
            name=pair_set.name,
            task_type=pair_set.task_type
        )
        
        for pair in pair_set.pairs:
            new_pair = type(pair)(
                prompt=pair.prompt,
                positive_response=type(pair.positive_response)(
                    text=pair.positive_response.text,
                    activations=extract_activations(pair.positive_response.text, layer)
                ),
                negative_response=type(pair.negative_response)(
                    text=pair.negative_response.text,
                    activations=extract_activations(pair.negative_response.text, layer)
                )
            )
            layer_pair_set.pairs.append(new_pair)
        
        layer_pair_sets[layer] = layer_pair_set
    
    return layer_pair_sets


def get_steering_configs(device, layers, trait_name, methods_list):
    """Get steering configurations based on selected methods."""
    all_configs = [
        # CAA variations
        SteeringConfig(
            name="CAA",
            method_class=CAA,
            init_params={"device": device},
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers
            }
        ),
        SteeringConfig(
            name="CAA_L2",
            method_class=CAA,
            init_params={"device": device, "normalization_method": "l2_unit"},
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers
            }
        ),
        
        # HPR variations
        SteeringConfig(
            name="HPR",
            method_class=HPR,
            init_params={
                "device": device,
                "learning_rate": 1e-3,
                "epochs": 100,
                "batch_size": 32,
                "hidden_size": 128,
                "angle_loss_weight": 0.1
            },
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers,
                "learning_rate": [5e-4, 1e-3, 5e-3],
                "epochs": [50, 100, 200],
                "angle_loss_weight": [0.05, 0.1, 0.2]
            }
        ),
        SteeringConfig(
            name="HPR_Beta0.5",
            method_class=HPR,
            init_params={
                "device": device,
                "learning_rate": 1e-3,
                "epochs": 100,
                "batch_size": 32,
                "hidden_size": 128,
                "angle_loss_weight": 0.05
            },
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers,
                "learning_rate": [5e-4, 1e-3, 5e-3],
                "epochs": [50, 100, 200]
            }
        ),
        
        # BiPO variations
        SteeringConfig(
            name="BiPO",
            method_class=BiPO,
            init_params={
                "device": device,
                "beta": 0.1,
                "learning_rate": 5e-4,
                "num_epochs": 50,
                "batch_size": 16,
                "reference_free": True
            },
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers,
                "beta": [0.05, 0.1, 0.2, 0.5],
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "num_epochs": [30, 50, 100]
            }
        ),
        SteeringConfig(
            name="BiPO_Beta0.05",
            method_class=BiPO,
            init_params={
                "device": device,
                "beta": 0.05,
                "learning_rate": 5e-4,
                "num_epochs": 50,
                "batch_size": 16,
                "reference_free": True
            },
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers,
                "learning_rate": [1e-4, 5e-4, 1e-3],
                "num_epochs": [30, 50, 100]
            }
        ),
        
        # KSteering variations
        SteeringConfig(
            name="KSteering",
            method_class=KSteering,
            init_params={
                "device": device,
                "alpha": 5.0,
                "target_labels": [trait_name],
                "avoid_labels": [],
                "k": 1,
                "num_epochs": 100,
                "batch_size": 64,
                "learning_rate": 1e-3
            },
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers,
                "alpha": [3.0, 5.0, 7.0],
                "num_epochs": [50, 100, 200]
            }
        ),
        SteeringConfig(
            name="KSteering_Alpha3",
            method_class=KSteering,
            init_params={
                "device": device,
                "alpha": 3.0,
                "target_labels": [trait_name],
                "avoid_labels": [],
                "k": 1,
                "num_epochs": 100,
                "batch_size": 64,
                "learning_rate": 1e-3
            },
            steering_params={
                "strength": [0.5, 1.0, 1.5, 2.0, 2.5],
                "layer": layers,
                "num_epochs": [50, 100, 200]
            }
        ),
        
        # DAC
        SteeringConfig(
            name="DAC",
            method_class=DAC,
            init_params={
                "device": device,
                "enable_dynamic_control": True,
                "entropy_threshold": 1.0
            },
            steering_params={
                "alpha": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                "layer": layers
            }
        ),
    ]
    
    if "all" in methods_list:
        return all_configs
    else:
        return [c for c in all_configs if c.name in methods_list]


def generate_unsteered(prompt, hf_model, tokenizer, device, max_length):
    """Generate unsteered response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


def generate_with_steering(prompt, steering_method, layer, hf_model, tokenizer, device, max_length, **params):
    """Generate with steering, handling different parameter names for different methods."""
    if hasattr(steering_method, 'apply_steering'):
        if isinstance(steering_method, DAC) and 'strength' in params:
            params['alpha'] = params.pop('strength')
        elif isinstance(steering_method, KSteering) and 'alpha' in params:
            params.pop('alpha', None)
    
    def steering_hook(module, input, output):
        hidden_states = output[0]
        last_token = hidden_states[:, -1:, :]
        if isinstance(steering_method, DAC):
            steered = steering_method.apply_steering(last_token, strength=params.get('alpha', 1.0))
        else:
            steered = steering_method.apply_steering(last_token, strength=params.get('strength', 1.0))
        hidden_states[:, -1:, :] = steered
        return (hidden_states,) + output[1:]
    
    handle = hf_model.model.layers[layer].register_forward_hook(steering_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    handle.remove()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()


def optimize_steering_config(config, layer_pair_sets, hf_model, tokenizer, evaluator, device, 
                           test_prompts, unsteered_responses, trait_name, max_length):
    """Optimize all parameters for a steering configuration."""
    logger.info(f"\nOptimizing {config.name}")
    logger.info("=" * 60)
    
    best_score = -1
    best_params = {}
    best_responses = {}
    all_results = []
    
    # Generate all parameter combinations
    param_names = list(config.steering_params.keys())
    param_values = list(config.steering_params.values())
    all_combinations = list(product(*param_values))
    
    logger.info(f"Testing {len(all_combinations)} parameter combinations...")
    
    for i, param_tuple in enumerate(all_combinations):
        params = dict(zip(param_names, param_tuple))
        layer = params.pop('layer')
        
        try:
            # Initialize method with the right parameters
            init_params = config.init_params.copy()
            
            # Handle training parameters
            if config.name.startswith("HPR"):
                for param in ['learning_rate', 'epochs', 'angle_loss_weight']:
                    if param in params:
                        init_params[param] = params.pop(param)
            
            if config.name.startswith("BiPO"):
                for param in ['beta', 'learning_rate', 'num_epochs']:
                    if param in params:
                        init_params[param] = params.pop(param)
            
            if config.name.startswith("KSteering"):
                for param in ['alpha', 'num_epochs']:
                    if param in params:
                        init_params[param] = params.pop(param)
            
            # Create and train the method
            method = config.method_class(**init_params)
            
            if isinstance(method, DAC):
                method.set_model_reference(hf_model)
            
            if layer in layer_pair_sets:
                method.train(layer_pair_sets[layer], layer)
            else:
                logger.warning(f"No training data for layer {layer}")
                continue
            
            # Generate steered responses and evaluate
            prompt_scores = []
            prompt_responses = {}
            
            for prompt in test_prompts:
                steered_response = generate_with_steering(
                    prompt, method, layer, hf_model, tokenizer, device, max_length, **params
                )
                prompt_responses[prompt] = steered_response
                
                scores = evaluator.evaluate_response(
                    prompt, unsteered_responses[prompt], steered_response, trait_name
                )
                prompt_scores.append(scores['overall'])
            
            avg_score = sum(prompt_scores) / len(prompt_scores)
            
            result = {
                'params': {**params, 'layer': layer, 
                          **{k: v for k, v in init_params.items() 
                             if k not in ['device', 'target_labels', 'avoid_labels']}},
                'avg_score': avg_score,
                'prompt_scores': prompt_scores,
                'responses': prompt_responses
            }
            all_results.append(result)
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = result['params']
                best_responses = prompt_responses
            
            # Progress update
            update_interval = 10 if len(all_combinations) < 1000 else 100
            if (i + 1) % update_interval == 0:
                logger.info(f"  Progress: {i+1}/{len(all_combinations)} ({(i+1)/len(all_combinations)*100:.1f}%) - Best avg score so far: {best_score:.1f}")
                
        except Exception as e:
            logger.error(f"  Error with params {params}: {str(e)[:100]}")
            continue
    
    logger.info(f"\nBest parameters for {config.name}:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Best average score: {best_score:.1f}/10")
    
    return {
        'config_name': config.name,
        'best_params': best_params,
        'best_score': best_score,
        'best_responses': best_responses,
        'all_results': all_results
    }


def main():
    args = parse_arguments()
    
    # Setup device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    model, hf_model, tokenizer, evaluator = load_model(args.model_name, device)
    
    # Determine layers
    num_layers = len(hf_model.model.layers)
    layers = parse_layers(args.layers, num_layers)
    logger.info(f"Testing layers: {layers}")
    
    # Generate or load pairs
    pair_set = generate_or_load_pairs(args, model)
    logger.info(f"Using {len(pair_set.pairs)} pairs")
    
    # Extract activations for all layers
    layer_pair_sets = extract_activations_for_layers(
        pair_set, layers, hf_model, tokenizer, device
    )
    
    # Generate unsteered responses
    logger.info("Generating unsteered responses...")
    unsteered_responses = {}
    for prompt in args.test_prompts:
        unsteered_responses[prompt] = generate_unsteered(
            prompt, hf_model, tokenizer, device, args.max_length
        )
    
    # Get steering configurations
    configs = get_steering_configs(device, layers, args.trait, args.methods)
    
    # Print total combinations
    total_combinations = 0
    for config in configs:
        param_values = list(config.steering_params.values())
        combos = 1
        for values in param_values:
            combos *= len(values)
        logger.info(f"{config.name}: {combos:,} combinations")
        total_combinations += combos
    
    logger.info(f"\nTotal: {total_combinations:,} combinations across {len(configs)} methods")
    
    # Optimize each configuration
    optimization_results = []
    for config in configs:
        try:
            result = optimize_steering_config(
                config, layer_pair_sets, hf_model, tokenizer, evaluator, 
                device, args.test_prompts, unsteered_responses, args.trait, args.max_length
            )
            optimization_results.append(result)
        except Exception as e:
            logger.error(f"Failed to optimize {config.name}: {e}")
    
    # Sort results by score
    sorted_results = sorted(optimization_results, key=lambda x: x['best_score'], reverse=True)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    
    for i, result in enumerate(sorted_results, 1):
        logger.info(f"\n{i}. {result['config_name']}:")
        logger.info(f"   Best score: {result['best_score']:.1f}/10")
        logger.info(f"   Optimal parameters: {result['best_params']}")
    
    # Save results
    output_data = {
        "model": args.model_name,
        "trait": args.trait,
        "test_prompts": args.test_prompts,
        "unsteered_responses": unsteered_responses,
        "optimization_results": []
    }
    
    for result in sorted_results:
        output_data["optimization_results"].append({
            "method": result['config_name'],
            "best_score": result['best_score'],
            "best_params": result['best_params'],
            "best_responses": result['best_responses']
        })
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()