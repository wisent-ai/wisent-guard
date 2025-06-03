"""
Wisent-Guarded Llama Model Implementation

This module provides a custom Llama model with embedded wisent-guard safety mechanisms.
All text generation is automatically screened for harmful content using activation-based detection.
"""

import os
import torch
import warnings
from typing import Optional, Union, Tuple, List, Dict, Any
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# Try to import wisent_guard - if not available, provide fallback
try:
    from wisent_guard import ActivationGuard
    WISENT_GUARD_AVAILABLE = True
except ImportError:
    WISENT_GUARD_AVAILABLE = False
    warnings.warn(
        "wisent-guard not available. Model will function as standard Llama without safety mechanisms. "
        "Install wisent-guard for full functionality: pip install wisent-guard"
    )


class WisentLlamaConfig(LlamaConfig):
    """
    Configuration class for Wisent-Guarded Llama model.
    
    Extends LlamaConfig with wisent-guard specific parameters.
    """
    
    model_type = "wisent_llama"
    
    def __init__(
        self,
        # Wisent-guard specific parameters
        wisent_enabled: bool = True,
        wisent_threshold: float = 0.5,
        wisent_layers: Optional[List[int]] = None,
        wisent_use_classifier: bool = True,
        wisent_classifier_path: Optional[str] = None,
        wisent_classifier_threshold: float = 0.5,
        wisent_early_termination: bool = True,
        wisent_termination_message: str = "Sorry, this response was blocked due to potentially harmful content.",
        wisent_categories: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Wisent-guard configuration
        self.wisent_enabled = wisent_enabled
        self.wisent_threshold = wisent_threshold
        self.wisent_layers = wisent_layers or [15]  # Default to layer 15
        self.wisent_use_classifier = wisent_use_classifier
        self.wisent_classifier_path = wisent_classifier_path
        self.wisent_classifier_threshold = wisent_classifier_threshold
        self.wisent_early_termination = wisent_early_termination
        self.wisent_termination_message = wisent_termination_message
        self.wisent_categories = wisent_categories or ["harmful", "hallucination"]


class WisentLlamaForCausalLM(LlamaForCausalLM):
    """
    Llama model with embedded wisent-guard safety mechanisms.
    
    This model automatically screens all generated text for harmful content
    using activation-based detection techniques.
    """
    
    config_class = WisentLlamaConfig
    
    def __init__(self, config: WisentLlamaConfig):
        super().__init__(config)
        
        self.wisent_guard = None
        self._wisent_initialized = False
        
        # Initialize wisent-guard if available and enabled
        if WISENT_GUARD_AVAILABLE and config.wisent_enabled:
            self._initialize_wisent_guard()
    
    def _initialize_wisent_guard(self):
        """Initialize the wisent-guard system."""
        if self._wisent_initialized:
            return
            
        try:
            # Determine classifier path - look for bundled classifier first
            classifier_path = None
            if self.config.wisent_use_classifier:
                if self.config.wisent_classifier_path:
                    classifier_path = self.config.wisent_classifier_path
                else:
                    # Look for bundled classifier
                    model_dir = getattr(self.config, '_name_or_path', '.')
                    potential_paths = [
                        os.path.join(model_dir, "wisent_data", "classifier.pkl"),
                        os.path.join(model_dir, "wisent_data", "classifier.pth"),
                        "wisent_data/classifier.pkl",
                        "wisent_data/classifier.pth"
                    ]
                    for path in potential_paths:
                        if os.path.exists(path):
                            classifier_path = path
                            break
            
            # Initialize the guard
            guard_kwargs = {
                "model": self,
                "tokenizer": getattr(self, 'tokenizer', None),  # Will be set later if not available
                "layers": self.config.wisent_layers,
                "threshold": self.config.wisent_threshold,
                "early_termination": self.config.wisent_early_termination,
                "placeholder_message": self.config.wisent_termination_message,
                "use_classifier": self.config.wisent_use_classifier,
                "classifier_threshold": self.config.wisent_classifier_threshold,
            }
            
            if classifier_path:
                guard_kwargs["classifier_path"] = classifier_path
            
            # Look for bundled vector data
            model_dir = getattr(self.config, '_name_or_path', '.')
            save_dir = os.path.join(model_dir, "wisent_data")
            if os.path.exists(save_dir):
                guard_kwargs["save_dir"] = save_dir
            
            self.wisent_guard = ActivationGuard(**guard_kwargs)
            self._wisent_initialized = True
            
            print(f"✅ Wisent-guard initialized successfully")
            print(f"   - Monitoring layers: {self.config.wisent_layers}")
            print(f"   - Using classifier: {self.config.wisent_use_classifier}")
            if classifier_path:
                print(f"   - Classifier path: {classifier_path}")
            print(f"   - Threshold: {self.config.wisent_threshold}")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize wisent-guard: {e}")
            self.wisent_guard = None
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for the wisent-guard system."""
        if self.wisent_guard and hasattr(self.wisent_guard, 'tokenizer'):
            self.wisent_guard.tokenizer = tokenizer
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus: Optional[bool] = None,
        assistant_model = None,
        streamer = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        Generate text with automatic safety screening via wisent-guard.
        
        If wisent-guard is available and enabled, all generation will be screened
        for harmful content. Otherwise, falls back to standard Llama generation.
        """
        
        # If wisent-guard is not available or disabled, use standard generation
        if not self.wisent_guard or not self.config.wisent_enabled:
            return super().generate(
                inputs=inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                synced_gpus=synced_gpus,
                assistant_model=assistant_model,
                streamer=streamer,
                **kwargs,
            )
        
        # Extract prompt for wisent-guard
        if inputs is not None and hasattr(self.wisent_guard, 'tokenizer') and self.wisent_guard.tokenizer:
            # Decode the input tokens to get the prompt
            prompt = self.wisent_guard.tokenizer.decode(inputs[0], skip_special_tokens=True)
            
            # Extract generation parameters for wisent-guard
            max_new_tokens = kwargs.get('max_new_tokens', kwargs.get('max_length', 100))
            if 'max_length' in kwargs and 'max_new_tokens' not in kwargs:
                # Convert max_length to max_new_tokens
                max_new_tokens = kwargs['max_length'] - inputs.shape[-1] if inputs is not None else kwargs['max_length']
            
            # Use wisent-guard for safe generation
            try:
                result = self.wisent_guard.generate_safe_response(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    **{k: v for k, v in kwargs.items() if k in ['temperature', 'top_p', 'top_k', 'do_sample']}
                )
                
                # Convert wisent-guard result back to HuggingFace format
                response_text = result.get('response', '')
                full_text = prompt + response_text
                
                # Tokenize the full response
                if self.wisent_guard.tokenizer:
                    output_ids = self.wisent_guard.tokenizer.encode(full_text, return_tensors='pt')
                    
                    # Return in the expected format
                    if kwargs.get('return_dict_in_generate', False):
                        return GenerateOutput(sequences=output_ids)
                    else:
                        return output_ids
                        
            except Exception as e:
                warnings.warn(f"Wisent-guard generation failed, falling back to standard generation: {e}")
        
        # Fallback to standard generation
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )
    
    def is_harmful(self, text: str) -> bool:
        """
        Check if the given text is potentially harmful.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is potentially harmful, False otherwise
        """
        if not self.wisent_guard or not self.config.wisent_enabled:
            return False
            
        try:
            return self.wisent_guard.is_harmful(text)
        except Exception as e:
            warnings.warn(f"Wisent-guard harm detection failed: {e}")
            return False
    
    def get_safety_score(self, text: str) -> float:
        """
        Get the safety score for the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Safety score (higher = more likely to be harmful)
        """
        if not self.wisent_guard or not self.config.wisent_enabled:
            return 0.0
            
        try:
            return self.wisent_guard.get_similarity(text)
        except Exception as e:
            warnings.warn(f"Wisent-guard safety scoring failed: {e}")
            return 0.0
    
    def get_available_categories(self) -> List[str]:
        """
        Get the list of available detection categories.
        
        Returns:
            List of category names
        """
        if not self.wisent_guard or not self.config.wisent_enabled:
            return []
            
        try:
            return self.wisent_guard.get_available_categories()
        except Exception as e:
            warnings.warn(f"Failed to get wisent-guard categories: {e}")
            return []


# Register the custom configuration
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("wisent_llama", WisentLlamaConfig)
AutoModelForCausalLM.register(WisentLlamaConfig, WisentLlamaForCausalLM) 