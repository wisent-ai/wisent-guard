"""
User-defined model configuration storage and retrieval.
Handles models that aren't explicitly supported by storing user-provided configurations.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any
from enum import Enum


class ModelArchitecture(Enum):
    """Supported model architectures for layer access."""
    LLAMA_STYLE = "llama_style"  # model.layers.{idx}
    GPT2_STYLE = "gpt2_style"    # transformer.h.{idx}
    MPT_STYLE = "mpt_style"      # transformer.blocks.{idx}
    CUSTOM = "custom"             # User provides full path template


class UserModelConfig:
    """Manages user-defined model configurations."""
    
    def __init__(self):
        # Store config in user's home directory
        self.config_dir = Path.home() / ".wisent-guard"
        self.config_file = self.config_dir / "user_model_configs.json"
        self.configs = self._load_configs()
    
    def _load_configs(self) -> Dict[str, Any]:
        """Load existing configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_configs(self) -> None:
        """Save configurations to file."""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.configs, f, indent=2)
    
    def has_config(self, model_name: str) -> bool:
        """Check if we have a configuration for this model."""
        return model_name in self.configs
    
    def get_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a model."""
        return self.configs.get(model_name)
    
    def save_config(self, model_name: str, config: Dict[str, Any]) -> None:
        """Save configuration for a model."""
        self.configs[model_name] = config
        self._save_configs()
    
    def get_prompt_tokens(self, model_name: str) -> Optional[Dict[str, str]]:
        """Get user and assistant tokens for a model."""
        config = self.get_config(model_name)
        if config:
            return {
                "user_token": config.get("user_token"),
                "assistant_token": config.get("assistant_token"),
                "system_token": config.get("system_token"),  # Optional
                "format_template": config.get("format_template")  # Optional custom template
            }
        return None
    
    def get_layer_access_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get layer access information for a model."""
        config = self.get_config(model_name)
        if config:
            return {
                "architecture": config.get("architecture"),
                "layer_path_template": config.get("layer_path_template"),
                "custom_layer_accessor": config.get("custom_layer_accessor")
            }
        return None
    
    def prompt_and_save_config(self, model_name: str) -> Dict[str, Any]:
        """
        Interactively prompt user for model configuration.
        This should be called from the CLI when an unknown model is encountered.
        """
        print(f"\n⚠️  Model '{model_name}' is not recognized.")
        print("We need some information to properly support this model.\n")
        
        config = {"model_name": model_name}
        
        # Prompt for tokens
        print("1. Chat Format Tokens")
        print("   These are the special tokens your model uses to distinguish user and assistant messages.")
        print("   Examples:")
        print("   - Llama 3: <|start_header_id|>user<|end_header_id|> and <|start_header_id|>assistant<|end_header_id|>")
        print("   - ChatGPT: <|im_start|>user and <|im_start|>assistant")
        print("   - Alpaca: ### Human: and ### Assistant:")
        
        config["user_token"] = input("\n   Enter the user token/prefix: ").strip()
        config["assistant_token"] = input("   Enter the assistant token/prefix: ").strip()
        
        # Optional system token
        system_token = input("   Enter the system token/prefix (press Enter to skip): ").strip()
        if system_token:
            config["system_token"] = system_token
        
        # Model architecture for layer access
        print("\n2. Model Architecture")
        print("   How are the transformer layers accessed in this model?")
        print("   1. Llama-style: model.layers.{idx}")
        print("   2. GPT2-style: transformer.h.{idx}")
        print("   3. MPT-style: transformer.blocks.{idx}")
        print("   4. Custom (you'll provide the template)")
        
        while True:
            choice = input("\n   Select architecture (1-4): ").strip()
            if choice == "1":
                config["architecture"] = ModelArchitecture.LLAMA_STYLE.value
                config["layer_path_template"] = "model.layers.{idx}"
                break
            elif choice == "2":
                config["architecture"] = ModelArchitecture.GPT2_STYLE.value
                config["layer_path_template"] = "transformer.h.{idx}"
                break
            elif choice == "3":
                config["architecture"] = ModelArchitecture.MPT_STYLE.value
                config["layer_path_template"] = "transformer.blocks.{idx}"
                break
            elif choice == "4":
                config["architecture"] = ModelArchitecture.CUSTOM.value
                template = input("   Enter the layer path template (use {idx} for layer index): ").strip()
                config["layer_path_template"] = template
                break
            else:
                print("   Invalid choice. Please enter 1, 2, 3, or 4.")
        
        # Optional: custom format template
        print("\n3. Custom Format Template (Optional)")
        print("   If your model requires a specific prompt format beyond simple token prefixes,")
        print("   you can provide a template. Use {user_message} and {assistant_message} as placeholders.")
        print("   Example: '<|system|>\\nYou are a helpful assistant\\n{user_message}\\n{assistant_message}'")
        
        custom_template = input("\n   Enter custom template (press Enter to skip): ").strip()
        if custom_template:
            config["format_template"] = custom_template
        
        # Save the configuration
        self.save_config(model_name, config)
        
        print(f"\n✅ Configuration saved for {model_name}")
        print(f"   Config location: {self.config_file}")
        
        return config


# Global instance for easy access
user_model_configs = UserModelConfig()