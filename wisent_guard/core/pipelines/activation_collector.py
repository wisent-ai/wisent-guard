"""
Activation collector for capturing model activations during steering vector training.
"""

import torch
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from ..data_loaders.steering_data_extractor import ContrastivePair

logger = logging.getLogger(__name__)


@dataclass
class ActivationData:
    """Container for activation data."""
    
    positive_activations: torch.Tensor
    negative_activations: torch.Tensor
    layer_idx: int
    problem_id: str
    metadata: Dict[str, Any]


class ActivationCollector:
    """Collects activations from models for steering vector training."""
    
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "auto",
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.max_length = max_length
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for activation hooks
        self.activation_hooks = {}
        self.collected_activations = {}
        
        logger.info(f"Initialized ActivationCollector with {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def collect_contrastive_activations(
        self,
        contrastive_pairs: List[ContrastivePair],
        target_layers: List[int],
        batch_size: int = 4
    ) -> List[ActivationData]:
        """
        Collect activations for contrastive pairs.
        
        Args:
            contrastive_pairs: List of contrastive pairs
            target_layers: List of layer indices to collect activations from
            batch_size: Batch size for processing
            
        Returns:
            List of ActivationData objects
        """
        logger.info(f"Collecting activations for {len(contrastive_pairs)} pairs at layers {target_layers}")
        
        all_activation_data = []
        
        # Process in batches
        for i in range(0, len(contrastive_pairs), batch_size):
            batch_pairs = contrastive_pairs[i:i + batch_size]
            
            for layer_idx in target_layers:
                batch_activations = self._collect_batch_activations(batch_pairs, layer_idx)
                all_activation_data.extend(batch_activations)
        
        logger.info(f"Collected {len(all_activation_data)} activation datasets")
        return all_activation_data
    
    def _collect_batch_activations(
        self,
        batch_pairs: List[ContrastivePair],
        layer_idx: int
    ) -> List[ActivationData]:
        """Collect activations for a batch of contrastive pairs."""
        
        # Prepare texts
        positive_texts = [pair.positive_prompt for pair in batch_pairs]
        negative_texts = [pair.negative_prompt for pair in batch_pairs]
        
        # Collect activations
        positive_activations = self._get_activations(positive_texts, layer_idx)
        negative_activations = self._get_activations(negative_texts, layer_idx)
        
        # Create ActivationData objects
        activation_data = []
        for i, pair in enumerate(batch_pairs):
            data = ActivationData(
                positive_activations=positive_activations[i],
                negative_activations=negative_activations[i],
                layer_idx=layer_idx,
                problem_id=pair.problem_id,
                metadata={
                    "strategy": pair.metadata.get("strategy", "unknown"),
                    "difficulty": pair.metadata.get("difficulty", "unknown"),
                    "platform": pair.metadata.get("platform", "unknown"),
                    "problem_title": pair.metadata.get("problem_title", "unknown"),
                    "activation_shape": list(positive_activations[i].shape),
                    "model_name": self.model_name,
                    "layer_idx": layer_idx
                }
            )
            activation_data.append(data)
        
        return activation_data
    
    def _get_activations(self, texts: List[str], layer_idx: int) -> torch.Tensor:
        """Get activations for a list of texts at a specific layer."""
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Register hook for target layer
        activations = []
        
        def hook_fn(module, input, output):
            # Store the last token activation (for generation tasks)
            # Shape: (batch_size, seq_len, hidden_size)
            last_token_activations = output[:, -1, :]  # (batch_size, hidden_size)
            activations.append(last_token_activations.detach().cpu())
        
        # Get the target layer
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-style model
            target_layer = self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            # BERT-style model
            target_layer = self.model.encoder.layer[layer_idx]
        else:
            raise ValueError(f"Unknown model architecture for {self.model_name}")
        
        # Register hook
        hook_handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # Extract activations
            if activations:
                return activations[0]  # (batch_size, hidden_size)
            else:
                raise RuntimeError("No activations collected")
        
        finally:
            # Remove hook
            hook_handle.remove()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "vocab_size": self.tokenizer.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
    
    def cleanup(self):
        """Clean up resources."""
        # Remove any remaining hooks
        for hook in self.activation_hooks.values():
            hook.remove()
        self.activation_hooks.clear()
        
        # Clear collected activations
        self.collected_activations.clear()
        
        # Move model to CPU to free GPU memory
        if self.device != "cpu":
            self.model.to("cpu")
        
        logger.info("ActivationCollector cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()