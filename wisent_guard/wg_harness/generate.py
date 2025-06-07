"""
Text generation with hidden state extraction for wisent-guard integration.
"""

import os
import logging
import torch
import hashlib
import pickle
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class GenerationCache:
    """Simple file-based cache for generation results."""
    
    def __init__(self, cache_dir: str = "./wg_harness_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, prompt: str, model_name: str, layer: int, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        key_data = f"{prompt}:{model_name}:{layer}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, model_name: str, layer: int, **kwargs) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        cache_key = self._get_cache_key(prompt, model_name, layer, **kwargs)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        
        return None
    
    def set(self, prompt: str, model_name: str, layer: int, result: Dict[str, Any], **kwargs) -> None:
        """Cache generation result."""
        cache_key = self._get_cache_key(prompt, model_name, layer, **kwargs)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_path}: {e}")


def load_model_and_tokenizer(model_name: str, device: Optional[str] = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with proper configuration for hidden state extraction.
    
    Args:
        model_name: Model name or path
        device: Target device (auto-detected if None)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"Loading model {model_name} on device {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with hidden states output enabled
    # Use float32 for MPS to avoid mixed precision issues
    if device == "mps":
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
        output_hidden_states=True
    )
    
    # Ensure model outputs hidden states
    model.config.output_hidden_states = True
    model.eval()
    
    logger.info(f"Model loaded successfully with {model.config.num_hidden_layers} layers")
    
    return model, tokenizer


def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    prompt: str,
    layer: int,
    max_new_tokens: int = 50,
    **generation_kwargs
) -> Tuple[str, torch.Tensor]:
    """
    Generate response and extract hidden states from specified layer.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        layer: Layer to extract hidden states from
        max_new_tokens: Maximum new tokens to generate
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Tuple of (generated_text, hidden_states)
    """
    with torch.inference_mode():
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate with hidden states
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": tokenizer.pad_token_id,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            **generation_kwargs
        }
        
        outputs = model.generate(**inputs, **generation_config)
        
        # Extract generated text
        generated_ids = outputs.sequences[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract hidden states from the specified layer for the last generated token
        # outputs.hidden_states is a tuple of (num_generated_tokens,) where each element
        # is a tuple of (num_layers,) tensors with shape (batch_size, seq_len, hidden_size)
        if outputs.hidden_states and len(outputs.hidden_states) > 0:
            # Get the last generation step's hidden states
            last_step_hidden_states = outputs.hidden_states[-1]
            
            # Extract the specified layer (add 1 because layer 0 is embeddings)
            if layer < len(last_step_hidden_states):
                # Get the last token's hidden state from the specified layer
                layer_hidden_state = last_step_hidden_states[layer][0, -1, :]  # [hidden_size]
            else:
                logger.warning(f"Layer {layer} not available, using layer 0")
                layer_hidden_state = last_step_hidden_states[0][0, -1, :]
        else:
            # Fallback: run a forward pass to get hidden states
            logger.warning("No hidden states in generation output, running forward pass")
            with torch.no_grad():
                forward_outputs = model(**inputs)
                layer_hidden_state = forward_outputs.hidden_states[layer][0, -1, :]
        
        return generated_text, layer_hidden_state.cpu().to(torch.float32)


def generate_responses(
    model_name: str,
    prompts: List[str],
    layer: int,
    batch_size: int = 8,
    max_new_tokens: int = 50,
    cache_dir: Optional[str] = "./wg_harness_cache",
    device: Optional[str] = None,
    **generation_kwargs
) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Generate responses for multiple prompts and extract hidden states.
    
    Args:
        model_name: Model name or path
        prompts: List of input prompts
        layer: Layer to extract hidden states from
        batch_size: Batch size for generation (note: currently processes one at a time)
        max_new_tokens: Maximum new tokens per response
        cache_dir: Directory for caching results (None to disable)
        device: Target device
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Tuple of (responses, hidden_states)
    """
    logger.info(f"Generating responses for {len(prompts)} prompts using layer {layer}")
    
    # Initialize cache
    cache = GenerationCache(cache_dir) if cache_dir else None
    
    # Load model once
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    
    responses = []
    hidden_states = []
    
    # Process prompts with progress bar
    for prompt in tqdm(prompts, desc="Generating responses"):
        # Check cache first
        cached_result = None
        if cache:
            cached_result = cache.get(prompt, model_name, layer, **generation_kwargs)
        
        if cached_result:
            logger.debug("Using cached result")
            response = cached_result['response']
            hidden_state = cached_result['hidden_state']
        else:
            # Generate new response
            try:
                response, hidden_state = extract_hidden_states(
                    model, tokenizer, prompt, layer, max_new_tokens, **generation_kwargs
                )
                
                # Cache result
                if cache:
                    cache.set(prompt, model_name, layer, {
                        'response': response,
                        'hidden_state': hidden_state
                    }, **generation_kwargs)
                    
            except Exception as e:
                logger.error(f"Failed to generate response for prompt: {e}")
                response = ""
                hidden_state = torch.zeros(model.config.hidden_size)
        
        responses.append(response)
        hidden_states.append(hidden_state)
    
    logger.info(f"Generated {len(responses)} responses")
    return responses, hidden_states


def batch_generate_with_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    layer: int,
    batch_size: int = 4,
    max_new_tokens: int = 50,
    **generation_kwargs
) -> Tuple[List[str], List[torch.Tensor]]:
    """
    Generate responses in batches (experimental - may have memory issues).
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompts: List of prompts
        layer: Layer to extract hidden states from
        batch_size: Batch size
        max_new_tokens: Maximum new tokens
        **generation_kwargs: Additional generation parameters
        
    Returns:
        Tuple of (responses, hidden_states)
    """
    logger.warning("Batch generation is experimental and may cause memory issues")
    
    all_responses = []
    all_hidden_states = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        with torch.inference_mode():
            # Tokenize batch
            inputs = tokenizer(
                batch_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            
            # Generate
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": tokenizer.pad_token_id,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
                **generation_kwargs
            }
            
            outputs = model.generate(**inputs, **generation_config)
            
            # Process each item in batch
            for j in range(len(batch_prompts)):
                # Extract generated text
                generated_ids = outputs.sequences[j][input_length:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract hidden state
                if outputs.hidden_states and len(outputs.hidden_states) > 0:
                    last_step_hidden_states = outputs.hidden_states[-1]
                    if layer < len(last_step_hidden_states):
                        hidden_state = last_step_hidden_states[layer][j, -1, :].cpu()
                    else:
                        hidden_state = last_step_hidden_states[0][j, -1, :].cpu()
                else:
                    hidden_state = torch.zeros(model.config.hidden_size)
                
                all_responses.append(response)
                all_hidden_states.append(hidden_state)
    
    return all_responses, all_hidden_states


def clear_cache(cache_dir: str = "./wg_harness_cache") -> None:
    """Clear the generation cache."""
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        logger.info(f"Cleared cache directory: {cache_dir}")
    else:
        logger.info("Cache directory does not exist") 