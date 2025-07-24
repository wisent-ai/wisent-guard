"""
Data loading and processing utilities for comprehensive evaluation.
"""

from typing import Dict, Any, List, Tuple
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import task classes for dataset loading
from ...tasks.math500_task import Math500Task
from ...tasks.aime_task import AIMETask


logger = logging.getLogger(__name__)


def load_dataset_samples(dataset_name: str, limit: int) -> List[Dict]:
    """Load samples from a dataset using proper task implementations."""
    logger.info(f"Loading {limit} samples from {dataset_name}...")
    
    try:
        if dataset_name.lower() in ['aime2024', 'aime2025', 'aime']:
            # Map dataset names to years
            year_mapping = {
                'aime2024': '2024',
                'aime2025': '2025', 
                'aime': '2025'
            }
            
            year = year_mapping.get(dataset_name.lower(), '2025')
            task = AIMETask(year=year, limit=limit)
            samples = task.load_data(limit=limit)
            
            logger.info(f"Loaded {len(samples)} samples from {dataset_name} via AIMETask")
            return samples
        
        elif dataset_name.lower() == 'math500':
            task = Math500Task(limit=limit)
            samples = task.load_data(limit=limit)
            
            logger.info(f"Loaded {len(samples)} samples from {dataset_name} via Math500Task")
            return samples
        
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
            
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        raise


def extract_activations_with_hook(model, tokenizer, texts: List[str], layer: int, 
                                batch_size: int, max_length: int, device: torch.device) -> np.ndarray:
    """Extract activations from a specific layer using hooks."""
    activations = []
    
    def hook_fn(module, input, output):
        # Handle different output formats (some layers return tuples)
        if isinstance(output, tuple):
            hidden_states = output[0]  # First element is usually hidden states
        else:
            hidden_states = output
        
        # Extract last token activations (typical for causal LM)
        if len(hidden_states.shape) == 3:  # [batch, seq, hidden]
            last_token_acts = hidden_states[:, -1, :].detach().cpu().numpy()
            activations.extend(last_token_acts)
    
    # Register hook
    if hasattr(model, 'transformer'):  # GPT-style models
        target_layer = model.transformer.h[layer]
    elif hasattr(model, 'model'):  # Some other architectures
        target_layer = model.model.layers[layer]
    else:
        raise ValueError(f"Unknown model architecture")
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            ).to(device)
            
            with torch.no_grad():
                _ = model(**inputs)
    
    finally:
        handle.remove()
    
    return np.array(activations)


def generate_benchmark_predictions(model, tokenizer, samples: List[Dict], 
                                 batch_size: int, max_length: int, device: torch.device) -> Tuple[List[str], List[str]]:
    """Generate model predictions for benchmark evaluation."""
    predictions = []
    ground_truths = []
    
    for sample in samples:
        # Extract question and answer
        if 'problem' in sample and 'answer' in sample:
            question = sample['problem']
            correct_answer = str(sample['answer'])
        elif 'Problem' in sample and 'Answer' in sample:
            question = sample['Problem'] 
            correct_answer = str(sample['Answer'])
        elif 'question' in sample and 'answer' in sample:
            question = sample['question']
            correct_answer = str(sample['answer'])
        else:
            logger.warning(f"Skipping sample with unknown format: {sample.keys()}")
            continue
        
        # Create prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated text
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        generated = generated.strip()
        
        predictions.append(generated)
        ground_truths.append(correct_answer)
    
    return predictions, ground_truths


def create_probe_training_data(model, tokenizer, samples: List[Dict], layer: int,
                             batch_size: int, max_length: int, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Create training data for probes: activations -> correctness labels."""
    texts = []
    labels = []
    
    for sample in samples:
        # Extract question and answer
        if 'problem' in sample and 'answer' in sample:
            question = sample['problem']
            correct_answer = str(sample['answer'])
        elif 'Problem' in sample and 'Answer' in sample:
            question = sample['Problem'] 
            correct_answer = str(sample['Answer'])
        elif 'question' in sample and 'answer' in sample:
            question = sample['question']
            correct_answer = str(sample['answer'])
        else:
            continue
        
        # Generate model prediction to create positive/negative examples
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Create examples with model's actual prediction
        correct_text = f"Question: {question}\nAnswer: {correct_answer}"
        incorrect_text = f"Question: {question}\nAnswer: {generated}"
        
        texts.extend([correct_text, incorrect_text])
        # 1 for correct, 0 for model's prediction (which might be wrong)
        is_correct = generated.strip().lower() == correct_answer.strip().lower()
        labels.extend([1, 1 if is_correct else 0])
    
    # Extract activations
    activations = extract_activations_with_hook(
        model, tokenizer, texts, layer, batch_size, max_length, device
    )
    
    return activations, np.array(labels)


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer with proper configuration."""
    logger.info(f"Loading model {model_name} (ONCE)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": 0} if device.type == "cuda" else None,
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Log memory usage
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"✓ Model loaded on {device}, GPU memory: {memory_gb:.2f} GB")
    
    return model, tokenizer


def free_model_memory(model, tokenizer):
    """Free model memory after activation extraction."""
    logger.info("🧹 Freeing model memory...")
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory after cleanup: {memory_gb:.2f} GB")