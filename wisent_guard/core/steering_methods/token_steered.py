"""
Token-level steering control for wisent-guard.

This module provides flexible control over which tokens steering is applied to
during generation, with various application strategies.
"""

from enum import Enum
from typing import Dict, Any, Optional, Union, List, Callable
import torch
import math

from .base import SteeringMethod
from ..contrastive_pairs import ContrastivePairSet


class TokenSteeringStrategy(Enum):
    """Strategies for applying steering to different token positions."""
    LAST_ONLY = "last_only"                    # Only steer the last token (current behavior)
    SECOND_TO_LAST = "second_to_last"          # Only steer the second-to-last token (reference behavior)
    FIRST_ONLY = "first_only"                  # Only steer the first token
    ALL_EQUAL = "all_equal"                    # Apply equal steering to all tokens
    EXPONENTIAL_DECAY = "exponential_decay"    # Exponentially decreasing from first to last
    EXPONENTIAL_GROWTH = "exponential_growth"  # Exponentially increasing from first to last
    LINEAR_DECAY = "linear_decay"              # Linearly decreasing from first to last
    LINEAR_GROWTH = "linear_growth"            # Linearly increasing from first to last
    CUSTOM = "custom"                          # Custom function provided by user


class TokenSteeringConfig:
    """Configuration for token-level steering application."""
    
    def __init__(
        self,
        strategy: TokenSteeringStrategy = TokenSteeringStrategy.LAST_ONLY,
        decay_rate: float = 0.5,
        min_strength: float = 0.1,
        max_strength: float = 1.0,
        custom_function: Optional[Callable[[int, int], float]] = None,
        apply_to_prompt: bool = False,
        prompt_strength_multiplier: float = 0.1
    ):
        """
        Initialize token steering configuration.
        
        Args:
            strategy: Token steering strategy to use
            decay_rate: Decay rate for exponential strategies (0-1)
            min_strength: Minimum steering strength
            max_strength: Maximum steering strength
            custom_function: Custom function(token_idx, total_tokens) -> strength_multiplier
            apply_to_prompt: Whether to apply steering to prompt tokens as well
            prompt_strength_multiplier: Strength multiplier for prompt tokens
        """
        self.strategy = strategy
        self.decay_rate = decay_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.custom_function = custom_function
        self.apply_to_prompt = apply_to_prompt
        self.prompt_strength_multiplier = prompt_strength_multiplier
        
        # Validate parameters
        if decay_rate <= 0 or decay_rate >= 1:
            raise ValueError("decay_rate must be between 0 and 1")
        if min_strength < 0 or max_strength < 0:
            raise ValueError("Strength values must be non-negative")
        if min_strength > max_strength:
            raise ValueError("min_strength cannot be greater than max_strength")
        if strategy == TokenSteeringStrategy.CUSTOM and custom_function is None:
            raise ValueError("custom_function must be provided when using CUSTOM strategy")


class TokenSteeringMixin:
    """Mixin class to add token steering capabilities to existing steering methods."""
    
    def __init__(self, *args, **kwargs):
        # Initialize token steering first, then call super
        self.token_steering_config = TokenSteeringConfig()
        super().__init__(*args, **kwargs)
    
    def set_token_steering_config(self, config: TokenSteeringConfig):
        """Set the token steering configuration."""
        self.token_steering_config = config
    
    def compute_token_strengths(
        self, 
        sequence_length: int, 
        base_strength: float = 1.0,
        prompt_length: int = 0
    ) -> torch.Tensor:
        """
        Compute steering strengths for each token position.
        
        Args:
            sequence_length: Total sequence length
            base_strength: Base steering strength
            prompt_length: Length of the prompt (if apply_to_prompt is False)
            
        Returns:
            Tensor of shape [sequence_length] with strength multipliers
        """
        config = self.token_steering_config
        strengths = torch.zeros(sequence_length)
        
        # Determine which tokens to apply steering to
        if config.apply_to_prompt:
            start_idx = 0
            generation_length = sequence_length
        else:
            start_idx = prompt_length
            generation_length = sequence_length - prompt_length
        
        if generation_length <= 0:
            return strengths
        
        # Apply strategy to generation tokens
        generation_strengths = self._compute_strategy_strengths(
            generation_length, 
            config.strategy,
            config.decay_rate,
            config.min_strength,
            config.max_strength,
            config.custom_function
        )
        
        # Set generation token strengths
        strengths[start_idx:start_idx + generation_length] = generation_strengths * base_strength
        
        # Apply prompt token strengths if enabled
        if config.apply_to_prompt and prompt_length > 0:
            prompt_strengths = torch.full((prompt_length,), base_strength * config.prompt_strength_multiplier)
            strengths[:prompt_length] = prompt_strengths
        
        return strengths
    
    def _compute_strategy_strengths(
        self,
        length: int,
        strategy: TokenSteeringStrategy,
        decay_rate: float,
        min_strength: float,
        max_strength: float,
        custom_function: Optional[Callable]
    ) -> torch.Tensor:
        """Compute strengths based on the specified strategy."""
        
        if length == 1:
            return torch.tensor([max_strength])
        
        positions = torch.arange(length, dtype=torch.float32)
        
        if strategy == TokenSteeringStrategy.LAST_ONLY:
            strengths = torch.zeros(length)
            strengths[-1] = max_strength
            
        elif strategy == TokenSteeringStrategy.SECOND_TO_LAST:
            strengths = torch.zeros(length)
            strengths[-2] = max_strength
            
        elif strategy == TokenSteeringStrategy.FIRST_ONLY:
            strengths = torch.zeros(length)
            strengths[0] = max_strength
            
        elif strategy == TokenSteeringStrategy.ALL_EQUAL:
            strengths = torch.full((length,), max_strength)
            
        elif strategy == TokenSteeringStrategy.EXPONENTIAL_DECAY:
            # Exponentially decreasing from first to last
            strengths = max_strength * (decay_rate ** positions)
            strengths = torch.clamp(strengths, min_strength, max_strength)
            
        elif strategy == TokenSteeringStrategy.EXPONENTIAL_GROWTH:
            # Exponentially increasing from first to last
            reverse_positions = positions.flip(0)
            strengths = max_strength * (decay_rate ** reverse_positions)
            strengths = strengths.flip(0)
            strengths = torch.clamp(strengths, min_strength, max_strength)
            
        elif strategy == TokenSteeringStrategy.LINEAR_DECAY:
            # Linearly decreasing from first to last
            normalized_positions = positions / (length - 1)
            strengths = max_strength - normalized_positions * (max_strength - min_strength)
            
        elif strategy == TokenSteeringStrategy.LINEAR_GROWTH:
            # Linearly increasing from first to last
            normalized_positions = positions / (length - 1)
            strengths = min_strength + normalized_positions * (max_strength - min_strength)
            
        elif strategy == TokenSteeringStrategy.CUSTOM:
            # Apply custom function
            strengths = torch.tensor([
                custom_function(i, length) for i in range(length)
            ], dtype=torch.float32)
            strengths = torch.clamp(strengths, 0.0, float('inf'))  # Ensure non-negative
            
        else:
            raise ValueError(f"Unknown token steering strategy: {strategy}")
        
        return strengths
    
    def apply_token_steering(
        self, 
        activations: torch.Tensor, 
        steering_vector: torch.Tensor,
        base_strength: float = 1.0,
        prompt_length: int = 0
    ) -> torch.Tensor:
        """
        Apply steering with token-level control.
        
        Args:
            activations: Input activations [batch, seq, hidden] or [batch, hidden]
            steering_vector: Steering vector to apply
            base_strength: Base steering strength
            prompt_length: Length of the prompt tokens
            
        Returns:
            Steered activations
        """
        if len(activations.shape) == 2:
            # Single token case [batch, hidden] - treat as last token
            return activations + base_strength * steering_vector.unsqueeze(0)
        
        elif len(activations.shape) == 3:
            # Sequence case [batch, seq, hidden]
            batch_size, seq_length, hidden_size = activations.shape
            
            # Compute token-specific strengths
            token_strengths = self.compute_token_strengths(
                seq_length, 
                base_strength, 
                prompt_length
            ).to(activations.device)
            
            # Apply steering to each token position
            steered = activations.clone()
            steering_vector = steering_vector.to(activations.device)
            
            for token_idx in range(seq_length):
                if token_strengths[token_idx] > 0:
                    steered[:, token_idx:token_idx+1, :] = (
                        steered[:, token_idx:token_idx+1, :] + 
                        token_strengths[token_idx] * steering_vector.unsqueeze(0).unsqueeze(0)
                    )
            
            return steered
        
        else:
            # Fallback for other shapes
            return activations + base_strength * steering_vector


class TokenSteeringWrapper(SteeringMethod):
    """
    Wrapper that adds token steering capabilities to any steering method.
    
    This class wraps an existing steering method and adds flexible token-level
    steering control on top of it.
    """
    
    def __init__(
        self, 
        base_method: SteeringMethod,
        token_config: Optional[TokenSteeringConfig] = None
    ):
        """
        Initialize the token steering wrapper.
        
        Args:
            base_method: The underlying steering method to wrap
            token_config: Token steering configuration
        """
        self.base_method = base_method
        wrapper_name = f"TokenSteered{base_method.name}"
        
        # Initialize parent class
        super().__init__(wrapper_name, base_method.device)
        
        # Copy attributes from base method
        self.is_trained = base_method.is_trained
        
        # Initialize token steering
        self.token_steering_config = TokenSteeringConfig()
        if token_config:
            self.set_token_steering_config(token_config)
    
    def set_token_steering_config(self, config: TokenSteeringConfig):
        """Set the token steering configuration."""
        self.token_steering_config = config
    
    def compute_token_strengths(
        self, 
        sequence_length: int, 
        base_strength: float = 1.0,
        prompt_length: int = 0
    ) -> torch.Tensor:
        """
        Compute steering strengths for each token position.
        
        Args:
            sequence_length: Total sequence length
            base_strength: Base steering strength
            prompt_length: Length of the prompt (if apply_to_prompt is False)
            
        Returns:
            Tensor of shape [sequence_length] with strength multipliers
        """
        config = self.token_steering_config
        strengths = torch.zeros(sequence_length)
        
        # Determine which tokens to apply steering to
        if config.apply_to_prompt:
            start_idx = 0
            generation_length = sequence_length
        else:
            start_idx = prompt_length
            generation_length = sequence_length - prompt_length
        
        if generation_length <= 0:
            return strengths
        
        # Apply strategy to generation tokens
        generation_strengths = self._compute_strategy_strengths(
            generation_length, 
            config.strategy,
            config.decay_rate,
            config.min_strength,
            config.max_strength,
            config.custom_function
        )
        
        # Set generation token strengths
        strengths[start_idx:start_idx + generation_length] = generation_strengths * base_strength
        
        # Apply prompt token strengths if enabled
        if config.apply_to_prompt and prompt_length > 0:
            prompt_strengths = torch.full((prompt_length,), base_strength * config.prompt_strength_multiplier)
            strengths[:prompt_length] = prompt_strengths
        
        return strengths
    
    def _compute_strategy_strengths(
        self,
        length: int,
        strategy: TokenSteeringStrategy,
        decay_rate: float,
        min_strength: float,
        max_strength: float,
        custom_function: Optional[Callable]
    ) -> torch.Tensor:
        """Compute strengths based on the specified strategy."""
        
        if length == 1:
            return torch.tensor([max_strength])
        
        positions = torch.arange(length, dtype=torch.float32)
        
        if strategy == TokenSteeringStrategy.LAST_ONLY:
            strengths = torch.zeros(length)
            strengths[-1] = max_strength
            
        elif strategy == TokenSteeringStrategy.SECOND_TO_LAST:
            strengths = torch.zeros(length)
            strengths[-2] = max_strength
            
        elif strategy == TokenSteeringStrategy.FIRST_ONLY:
            strengths = torch.zeros(length)
            strengths[0] = max_strength
            
        elif strategy == TokenSteeringStrategy.ALL_EQUAL:
            strengths = torch.full((length,), max_strength)
            
        elif strategy == TokenSteeringStrategy.EXPONENTIAL_DECAY:
            # Exponentially decreasing from first to last
            strengths = max_strength * (decay_rate ** positions)
            strengths = torch.clamp(strengths, min_strength, max_strength)
            
        elif strategy == TokenSteeringStrategy.EXPONENTIAL_GROWTH:
            # Exponentially increasing from first to last
            reverse_positions = positions.flip(0)
            strengths = max_strength * (decay_rate ** reverse_positions)
            strengths = strengths.flip(0)
            strengths = torch.clamp(strengths, min_strength, max_strength)
            
        elif strategy == TokenSteeringStrategy.LINEAR_DECAY:
            # Linearly decreasing from first to last
            normalized_positions = positions / (length - 1)
            strengths = max_strength - normalized_positions * (max_strength - min_strength)
            
        elif strategy == TokenSteeringStrategy.LINEAR_GROWTH:
            # Linearly increasing from first to last
            normalized_positions = positions / (length - 1)
            strengths = min_strength + normalized_positions * (max_strength - min_strength)
            
        elif strategy == TokenSteeringStrategy.CUSTOM:
            # Apply custom function
            strengths = torch.tensor([
                custom_function(i, length) for i in range(length)
            ], dtype=torch.float32)
            strengths = torch.clamp(strengths, 0.0, float('inf'))  # Ensure non-negative
            
        else:
            raise ValueError(f"Unknown token steering strategy: {strategy}")
        
        return strengths
    
    def apply_token_steering(
        self, 
        activations: torch.Tensor, 
        steering_vector: torch.Tensor,
        base_strength: float = 1.0,
        prompt_length: int = 0
    ) -> torch.Tensor:
        """
        Apply steering with token-level control.
        
        Args:
            activations: Input activations [batch, seq, hidden] or [batch, hidden]
            steering_vector: Steering vector to apply
            base_strength: Base steering strength
            prompt_length: Length of the prompt tokens
            
        Returns:
            Steered activations
        """
        if len(activations.shape) == 2:
            # Single token case [batch, hidden] - treat as last token
            return activations + base_strength * steering_vector.unsqueeze(0)
        
        elif len(activations.shape) == 3:
            # Sequence case [batch, seq, hidden]
            batch_size, seq_length, hidden_size = activations.shape
            
            # Compute token-specific strengths
            token_strengths = self.compute_token_strengths(
                seq_length, 
                base_strength, 
                prompt_length
            ).to(activations.device)
            
            # Apply steering to each token position
            steered = activations.clone()
            steering_vector = steering_vector.to(activations.device)
            
            for token_idx in range(seq_length):
                if token_strengths[token_idx] > 0:
                    steered[:, token_idx:token_idx+1, :] = (
                        steered[:, token_idx:token_idx+1, :] + 
                        token_strengths[token_idx] * steering_vector.unsqueeze(0).unsqueeze(0)
                    )
            
            return steered
        
        else:
            # Fallback for other shapes
            return activations + base_strength * steering_vector
    
    def train(self, contrastive_pair_set: ContrastivePairSet, layer_index: int) -> Dict[str, Any]:
        """Train the underlying steering method."""
        result = self.base_method.train(contrastive_pair_set, layer_index)
        self.is_trained = self.base_method.is_trained
        return result
    
    def apply_steering(self, activations: torch.Tensor, strength: float = 1.0, **kwargs) -> torch.Tensor:
        """Apply steering with token-level control."""
        if not self.is_trained:
            raise ValueError("Steering method must be trained before applying steering")
        
        # Get the steering vector from the base method
        steering_vector = self.base_method.get_steering_vector()
        
        # Extract prompt length if provided
        prompt_length = kwargs.get('prompt_length', 0)
        
        # Apply token-level steering
        return self.apply_token_steering(
            activations, 
            steering_vector, 
            strength, 
            prompt_length
        )
    
    def get_steering_vector(self) -> torch.Tensor:
        """Get the steering vector from the base method."""
        return self.base_method.get_steering_vector()
    
    def save_steering_vector(self, path: str) -> bool:
        """Save steering data including token config."""
        if not self.base_method.save_steering_vector(path):
            return False
        
        # Save token config separately
        try:
            import json
            config_path = path.replace('.pt', '_token_config.json')
            config_data = {
                'strategy': self.token_steering_config.strategy.value,
                'decay_rate': self.token_steering_config.decay_rate,
                'min_strength': self.token_steering_config.min_strength,
                'max_strength': self.token_steering_config.max_strength,
                'apply_to_prompt': self.token_steering_config.apply_to_prompt,
                'prompt_strength_multiplier': self.token_steering_config.prompt_strength_multiplier
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception:
            return False
    
    def load_steering_vector(self, path: str) -> bool:
        """Load steering data including token config."""
        if not self.base_method.load_steering_vector(path):
            return False
        
        # Load token config if available
        try:
            import json
            config_path = path.replace('.pt', '_token_config.json')
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            self.token_steering_config = TokenSteeringConfig(
                strategy=TokenSteeringStrategy(config_data['strategy']),
                decay_rate=config_data.get('decay_rate', 0.5),
                min_strength=config_data.get('min_strength', 0.1),
                max_strength=config_data.get('max_strength', 1.0),
                apply_to_prompt=config_data.get('apply_to_prompt', False),
                prompt_strength_multiplier=config_data.get('prompt_strength_multiplier', 0.1)
            )
        except Exception:
            # Use default config if loading fails
            self.token_steering_config = TokenSteeringConfig()
        
        self.is_trained = self.base_method.is_trained
        return True


# Convenience functions for creating token-steered versions of existing methods

def create_token_steered_caa(
    strategy: TokenSteeringStrategy = TokenSteeringStrategy.LAST_ONLY,
    decay_rate: float = 0.5,
    min_strength: float = 0.1,
    max_strength: float = 1.0,
    **kwargs
) -> TokenSteeringWrapper:
    """Create a token-steered CAA method."""
    from .caa import CAA
    
    base_method = CAA(**kwargs)
    token_config = TokenSteeringConfig(
        strategy=strategy,
        decay_rate=decay_rate,
        min_strength=min_strength,
        max_strength=max_strength
    )
    
    return TokenSteeringWrapper(base_method, token_config)


def create_token_steered_method(
    base_method: SteeringMethod,
    strategy: TokenSteeringStrategy = TokenSteeringStrategy.LAST_ONLY,
    decay_rate: float = 0.5,
    min_strength: float = 0.1,
    max_strength: float = 1.0,
    apply_to_prompt: bool = False,
    prompt_strength_multiplier: float = 0.1,
    custom_function: Optional[Callable[[int, int], float]] = None
) -> TokenSteeringWrapper:
    """
    Create a token-steered version of any steering method.
    
    Args:
        base_method: The steering method to wrap
        strategy: Token steering strategy
        decay_rate: Decay rate for exponential strategies
        min_strength: Minimum steering strength
        max_strength: Maximum steering strength
        apply_to_prompt: Whether to apply steering to prompt tokens
        prompt_strength_multiplier: Strength multiplier for prompt tokens
        custom_function: Custom function for CUSTOM strategy
        
    Returns:
        Token-steered wrapper around the base method
    """
    token_config = TokenSteeringConfig(
        strategy=strategy,
        decay_rate=decay_rate,
        min_strength=min_strength,
        max_strength=max_strength,
        apply_to_prompt=apply_to_prompt,
        prompt_strength_multiplier=prompt_strength_multiplier,
        custom_function=custom_function
    )
    
    return TokenSteeringWrapper(base_method, token_config)


# Example custom functions for common patterns

def attention_weighted_strength(attention_weights: torch.Tensor) -> Callable[[int, int], float]:
    """
    Create a custom function that uses attention weights to determine steering strength.
    
    Args:
        attention_weights: Attention weights for each token position
        
    Returns:
        Custom function for token steering
    """
    def custom_func(token_idx: int, total_tokens: int) -> float:
        if token_idx < len(attention_weights):
            return float(attention_weights[token_idx])
        return 0.0
    
    return custom_func


def gaussian_strength(center: float = 0.5, width: float = 0.3) -> Callable[[int, int], float]:
    """
    Create a Gaussian-shaped strength distribution.
    
    Args:
        center: Center of the Gaussian (0-1, relative position)
        width: Width of the Gaussian
        
    Returns:
        Custom function for token steering
    """
    def custom_func(token_idx: int, total_tokens: int) -> float:
        if total_tokens == 1:
            return 1.0
        
        normalized_pos = token_idx / (total_tokens - 1)
        strength = math.exp(-((normalized_pos - center) ** 2) / (2 * width ** 2))
        return strength
    
    return custom_func


def step_function_strength(step_position: float = 0.5, before_strength: float = 1.0, after_strength: float = 0.1) -> Callable[[int, int], float]:
    """
    Create a step function strength distribution.
    
    Args:
        step_position: Position of the step (0-1, relative position)
        before_strength: Strength before the step
        after_strength: Strength after the step
        
    Returns:
        Custom function for token steering
    """
    def custom_func(token_idx: int, total_tokens: int) -> float:
        if total_tokens == 1:
            return before_strength
        
        normalized_pos = token_idx / (total_tokens - 1)
        return before_strength if normalized_pos < step_position else after_strength
    
    return custom_func
