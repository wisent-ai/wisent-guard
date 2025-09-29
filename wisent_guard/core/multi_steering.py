"""Multi-steering functionality for combining multiple steering vectors."""

import sys
import torch
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from .layer import Layer
from .model import Model
from .steering_methods.caa import CAA
from .steering_methods.dac import DAC
from .utils.device import resolve_default_device


class MultiSteeringError(Exception):
    """Exception raised for multi-steering errors."""
    pass


class MultiSteering:
    """Handles multi-steering vector combination and application."""
    
    def __init__(self, device: str | None = None, method: str = "CAA"):
        """Initialize multi-steering handler.
        
        Args:
            device: Device to use for computations (cpu/cuda/mps)
            method: Steering method to use for combination ("CAA" or "DAC")
        """
        self.device = device or resolve_default_device()
        self.method = method
        self.loaded_vectors = []
        self.weights = []
        self.combined_vector = None
        self.layer = None
    
    def load_vectors(self, vector_specs: List[str]) -> None:
        """Load and validate steering vectors from file paths.
        
        Args:
            vector_specs: List of "path:weight" specifications
            
        Raises:
            MultiSteeringError: If vectors cannot be loaded or are incompatible
        """
        if not vector_specs:
            raise MultiSteeringError("No vectors specified")
        
        self.loaded_vectors = []
        self.weights = []
        layers_found = set()
        
        for spec in vector_specs:
            parts = spec.split(":")
            if len(parts) != 2:
                raise MultiSteeringError(f"Invalid vector specification: {spec}. Expected format: path:weight")
            
            vector_path = parts[0]
            try:
                weight = float(parts[1])
            except ValueError:
                raise MultiSteeringError(f"Invalid weight in {spec}. Must be a number.")
            
            if not Path(vector_path).exists():
                raise MultiSteeringError(f"Vector file not found: {vector_path}")
            
            print(f"Loading vector from {vector_path} with weight {weight}")
            
            try:
                vector_data = torch.load(vector_path, map_location=self.device)
            except Exception as e:
                raise MultiSteeringError(f"Failed to load vector from {vector_path}: {e}")
            
            # Extract metadata from loaded vector
            if isinstance(vector_data, dict):
                layer = vector_data.get("layer_index", None)
                steering_vector = vector_data.get("steering_vector", None)
                
                if steering_vector is None:
                    raise MultiSteeringError(f"No steering vector found in {vector_path}")
                
                if layer is not None:
                    layers_found.add(layer)
                
                self.loaded_vectors.append(vector_data)
                self.weights.append(weight)
                
                print(f"   ✓ Loaded vector from layer {layer}")
            else:
                raise MultiSteeringError(f"Invalid vector format in {vector_path}")
        
        # Validate compatibility
        if len(layers_found) > 1:
            raise MultiSteeringError(f"Vectors from different layers cannot be combined: {layers_found}")
        
        if not layers_found:
            raise MultiSteeringError("No layer information found in vectors")
        
        self.layer = Layer(list(layers_found)[0])
        
        print(f"\nUsing {self.method} method for vector combination")
        print(f"Target layer: {self.layer.index}")
    
    def combine_vectors(self, normalize: bool = True) -> torch.Tensor:
        """Combine loaded vectors using appropriate method.
        
        Args:
            normalize: Whether to normalize the combined vector
            
        Returns:
            Combined steering vector as tensor
            
        Raises:
            MultiSteeringError: If combination fails
        """
        if not self.loaded_vectors:
            raise MultiSteeringError("No vectors loaded")
        
        print(f"\n🔄 Combining {len(self.loaded_vectors)} vectors using {self.method}")
        
        if self.method == "CAA":
            # Create a CAA instance and use its proper combination method
            caa = CAA(device=self.device)
            
            # Set up behavior vectors dictionary
            caa.behavior_vectors = {}
            for i, (vector_data, weight) in enumerate(zip(self.loaded_vectors, self.weights)):
                steering_vector = vector_data["steering_vector"]
                
                if not isinstance(steering_vector, torch.Tensor):
                    steering_vector = torch.tensor(steering_vector, device=self.device)
                else:
                    steering_vector = steering_vector.to(self.device)
                
                # Store with unique names
                behavior_name = f"vector_{i}"
                caa.behavior_vectors[behavior_name] = steering_vector
            
            # Create weights dictionary
            behavior_weights = {f"vector_{i}": weight for i, weight in enumerate(self.weights)}
            
            # Use CAA's combine_behaviors method with normalization
            self.combined_vector = caa.combine_behaviors(behavior_weights, normalize_result=normalize)
            
        else:  # DAC or mixed methods
            # For DAC, use its combine_steering_vectors method
            vectors = []
            for vector_data in self.loaded_vectors:
                steering_vector = vector_data["steering_vector"]
                
                if not isinstance(steering_vector, torch.Tensor):
                    steering_vector = torch.tensor(steering_vector, device=self.device)
                else:
                    steering_vector = steering_vector.to(self.device)
                
                vectors.append(steering_vector)
            
            # Use DAC's static method for combination
            self.combined_vector = DAC.combine_steering_vectors(
                vectors, self.weights, normalize_weights=normalize
            )
        
        print(f"   ✓ Combined vector shape: {self.combined_vector.shape}")
        print(f"   ✓ Combined vector norm: {torch.norm(self.combined_vector).item():.4f}")
        
        return self.combined_vector
    
    def apply_steering(self, model: Model, prompt: str, max_new_tokens: int = 100, 
                      temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Apply the combined steering vector to generate text.
        
        Args:
            model: Model to use for generation
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
            
        Raises:
            MultiSteeringError: If steering fails
        """
        if self.combined_vector is None:
            raise MultiSteeringError("No combined vector available. Call combine_vectors() first.")
        
        if self.layer is None:
            raise MultiSteeringError("No layer information available")
        
        print(f"\n🎯 Applying combined steering vector at layer {self.layer.index}")
        print(f"Prompt: {prompt}")
        print("=" * 50)
        
        # Create appropriate steering method instance
        if self.method == "CAA":
            steering_method = CAA(device=self.device)
            steering_method.steering_vector = self.combined_vector
            steering_method.layer_index = self.layer.index
            steering_method.is_trained = True
        else:
            # Use DAC for other methods
            steering_method = DAC(device=self.device)
            steering_method.steering_vector = self.combined_vector  
            steering_method.layer_index = self.layer.index
            steering_method.is_trained = True
        
        # Set up steering hook
        hooks = []
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Apply steering using the method's apply_steering
            steered = steering_method.apply_steering(hidden_states, strength=1.0)
            
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered
        
        # Find the target layer module
        if hasattr(model.hf_model, "model") and hasattr(model.hf_model.model, "layers"):
            layer_module = model.hf_model.model.layers[self.layer.index]
        elif hasattr(model.hf_model, "transformer") and hasattr(model.hf_model.transformer, "h"):
            layer_module = model.hf_model.transformer.h[self.layer.index]
        else:
            raise MultiSteeringError("Could not find model layers")
        
        # Register hook
        handle = layer_module.register_forward_hook(steering_hook)
        hooks.append(handle)
        
        try:
            # Generate with steering
            output, _ = model.generate(
                prompt=prompt,
                layer_index=self.layer.index,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            
            return output
            
        except Exception as e:
            raise MultiSteeringError(f"Failed to apply steering: {e}")
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()


def run_multi_steer(
    vector_specs: List[str],
    model_name: str,
    prompt: str,
    method: str = "CAA",
    layer: Optional[int] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str | None = None,
    verbose: bool = True,
) -> str:
    """Convenience function to run multi-steering.
    
    Args:
        vector_specs: List of "path:weight" specifications
        model_name: Name of model to load
        prompt: Input prompt
        method: Steering method to use ("CAA" or "DAC")
        layer: Target layer (will be inferred from vectors if not specified)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature  
        top_p: Top-p sampling parameter
        device: Device to use
        verbose: Whether to print progress
        
    Returns:
        Generated text
    """
    # Initialize model
    if verbose:
        print(f"\n🚀 Loading model: {model_name}")
    
    chosen_device = device or resolve_default_device()
    model = Model(model_name, device=chosen_device)
    
    # Initialize multi-steering with specified method
    multi_steer = MultiSteering(device=chosen_device, method=method)
    
    # Load vectors
    multi_steer.load_vectors(vector_specs)
    
    # Override layer if specified
    if layer is not None:
        multi_steer.layer = Layer(layer)
        if verbose:
            print(f"Overriding layer to: {layer}")
    
    # Combine vectors with normalization
    multi_steer.combine_vectors(normalize=True)
    
    # Apply steering
    output = multi_steer.apply_steering(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    
    return output