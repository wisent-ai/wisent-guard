#Utility functions for content detection and model loading in Wisent-Guard.

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent_guard import ActivationGuard
from wisent_guard.classifier import ActivationClassifier

def get_device(cpu_only=False):
    if cpu_only:
        return "cpu"
    elif torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def load_model(model_name, device, half_precision=False, load_in_8bit=False):
    print(f"Loading model: {model_name}")
    
    load_kwargs = {
        "torch_dtype": torch.float16 if (device != "cpu" and half_precision) else torch.float32,
        "device_map": "auto" if device != "cpu" else None,
        "load_in_8bit": load_in_8bit,
        "trust_remote_code": True
    }
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Move model to device
    if device == "cpu":
        model = model.to(device)
    
    return model, tokenizer

def format_prompt(message, system_prompt=None):
    BEGIN_TEXT = "<|begin_of_text|>"
    START_HEADER = "<|start_header_id|>"
    END_HEADER = "<|end_header_id|>"
    EOT = "<|eot_id|>"
    
    if system_prompt:
        # Format with system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}system{END_HEADER}\n{system_prompt}{EOT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    else:
        # Format without system prompt
        formatted = f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{message}{EOT}{START_HEADER}assistant{END_HEADER}"
    
    return formatted
