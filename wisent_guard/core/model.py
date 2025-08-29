import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .contrastive_pairs import ContrastivePairSet
from .user_model_config import user_model_configs


class PromptFormat(Enum):
    LEGACY = "legacy"
    LLAMA31 = "llama31"
    MISTRAL = "mistral"
    QWEN = "qwen"


class TokenScore:
    """Stores information about a token and its similarity to harmful content."""

    def __init__(
        self,
        token_id: Optional[int] = None,
        token_text: str = "",
        position: int = 0,
        similarity: float = 0.0,
        is_harmful: bool = False,
        category: Optional[str] = None,
        activations: Optional[Dict[int, torch.Tensor]] = None,
    ):
        self.token_id = token_id
        self.token_text = token_text
        self.position = position
        self.similarity = similarity
        self.is_harmful = is_harmful
        self.category = category
        self.activations = activations or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "token_id": self.token_id,
            "token_text": self.token_text,
            "position": self.position,
            "similarity": float(self.similarity),
            "is_harmful": self.is_harmful,
            "category": self.category,
        }


class Model:
    def __init__(
        self,
        name: str,
        layers: Optional[List[int]] = None,
        device: Optional[str] = None,
        hf_model: Optional[AutoModelForCausalLM] = None,
    ):
        """
        Initialize Model with either a model name to load or an existing HuggingFace model.

        Args:
            name: Model name for HuggingFace or identifier
            layers: List of layer indices to use
            device: Device to run on
            hf_model: Optional existing HuggingFace model to wrap
        """
        self.name = name
        self.layers = layers if layers is not None else []
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )

        # Prompt formatting settings - will be set based on model type
        self.user_token = None
        self.assistant_token = None
        self.format_type = None  # Will be auto-detected

        if hf_model is not None:
            # Use provided model
            self.hf_model = hf_model
            self.model = hf_model  # Keep backward compatibility

            # Auto-detect format type first
            self.format_type = self._detect_format()

            # Try to load tokenizer from the same name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except:
                self.tokenizer = None

            # Initialize tokens
            self._initialize_tokens()
        else:
            # Check if model is supported before trying to load
            self.hf_model = None
            self.model = None
            self.tokenizer = None

            # Auto-detect format type first (this will raise error if unsupported)
            self.format_type = self._detect_format()

            # Only load model after we know it's supported
            self._load_model_and_tokenizer()

            # Initialize tokens after loading
            self._initialize_tokens()

    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from HuggingFace."""
        if self.device == "mps":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=torch_dtype, device_map=self.device, output_hidden_states=True
        )
        self.hf_model.config.output_hidden_states = True
        
        # Disable thinking for models that support it
        if hasattr(self.hf_model.config, 'enable_thinking'):
            self.hf_model.config.enable_thinking = False
            
        self.hf_model.eval()

        # Keep backward compatibility
        self.model = self.hf_model

    def _detect_format(self) -> PromptFormat:
        """
        Detect the appropriate format to use based on the model.
        Since we now use the tokenizer's chat template, this is mainly for backwards compatibility.

        Returns:
            PromptFormat enum indicating the format
        """
        model_name = self.name.lower()

        # Check for specific model types (for backwards compatibility)
        if re.search(r"llama-?3", model_name, re.IGNORECASE):
            return PromptFormat.LLAMA31
        if "mistral" in model_name:
            return PromptFormat.MISTRAL
        if "qwen" in model_name:
            return PromptFormat.QWEN
        if "gpt2" in model_name or "distilgpt2" in model_name:
            return PromptFormat.LEGACY
        # For all other models, default to LEGACY format
        # The actual formatting will be handled by the tokenizer's chat template
        return PromptFormat.LEGACY

    def _initialize_tokens(self):
        """Initialize user and assistant tokens based on format type or user config."""
        if self.format_type == PromptFormat.LLAMA31:
            # Llama uses header-based format, not simple tokens
            self.user_token = "user"  # Used in header
            self.assistant_token = "assistant"
        elif self.format_type == PromptFormat.MISTRAL:
            # Mistral uses instruction format
            self.user_token = "[INST]"
            self.assistant_token = "[/INST]"
        elif self.format_type == PromptFormat.QWEN:
            # Qwen uses system/user/assistant roles with special tokens
            self.user_token = "user"
            self.assistant_token = "assistant"
        elif self.format_type == PromptFormat.LEGACY:
            # Check for user-defined tokens
            if user_model_configs.has_config(self.name):
                tokens = user_model_configs.get_prompt_tokens(self.name)
                if tokens:
                    self.user_token = tokens.get("user_token", "<|user|>")
                    self.assistant_token = tokens.get("assistant_token", "<|assistant|>")
                else:
                    # Default legacy tokens
                    self.user_token = "<|user|>"
                    self.assistant_token = "<|assistant|>"
            else:
                # Default legacy tokens for known models like GPT2
                self.user_token = "<|user|>"
                self.assistant_token = "<|assistant|>"

    def set_prompt_tokens(self, user_token: str, assistant_token: str):
        """Set custom user and assistant tokens for legacy format."""
        self.user_token = user_token
        self.assistant_token = assistant_token

    def format_prompt(self, prompt: str, response: str = None, **kwargs) -> str:
        """
        Format a prompt using the tokenizer's chat template if available,
        otherwise fall back to legacy format detection.

        Args:
            prompt: Input prompt text
            response: Optional response text to include

        Returns:
            Formatted prompt string
        """
        # Try to use the tokenizer's chat template if available
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            try:
                # Create messages for the chat template
                messages = [{"role": "user", "content": prompt}]
                if response is not None:
                    messages.append({"role": "assistant", "content": response})

                # Apply the chat template
                # Pass any kwargs that might be model-specific (like enable_thinking for Qwen3)
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": (response is None),  # Add generation prompt only if no response
                    "enable_thinking": False,  # ALWAYS disable thinking by default
                }
                
                # Add any kwargs passed to format_prompt (e.g., enable_thinking)
                # But allow override if explicitly set to True
                template_kwargs.update(kwargs)
                
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    **template_kwargs
                )
                return formatted
            except Exception as e:
                # Fall back to legacy format if chat template fails
                print(f"Warning: Failed to apply chat template: {e}. Using legacy format.")

        # Legacy format detection (for backwards compatibility)
        if self.format_type == PromptFormat.LLAMA31:
            # Llama 3.1 special tokens
            BEGIN_TEXT = "<|begin_of_text|>"
            START_HEADER = "<|start_header_id|>"
            END_HEADER = "<|end_header_id|>"
            EOT = "<|eot_id|>"

            if response is not None:
                return f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{prompt}{EOT}{START_HEADER}assistant{END_HEADER}\n{response}"
            return f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{prompt}{EOT}{START_HEADER}assistant{END_HEADER}\n"
        if self.format_type == PromptFormat.MISTRAL:
            # Mistral format tokens
            INST_START = "[INST]"
            INST_END = "[/INST]"

            if response is not None:
                return f"{INST_START} {prompt} {INST_END} {response}"
            return f"{INST_START} {prompt} {INST_END}"
        if self.format_type == PromptFormat.QWEN:
            # Qwen format using special tokens
            IM_START = "<|im_start|>"
            IM_END = "<|im_end|>"

            if response is not None:
                return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n{response}{IM_END}"
            return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"
        # Legacy format
        if response is not None:
            return f"{self.user_token}\n{prompt}\n{self.assistant_token}\n{response}"
        return f"{self.user_token}\n{prompt}\n{self.assistant_token}\n"

    def format_multiple_choice(self, question: str, choice_a: str, choice_b: str, answer: str) -> str:
        """Format a multiple-choice question with the given answer."""
        instruction = f"{question}\nA. {choice_a}\nB. {choice_b}"
        return self.format_prompt(instruction, answer)

    def convert_to_multiple_choice(self, harmful_phrase: str, harmless_phrase: str) -> Tuple[str, str]:
        """
        Convert a harmful/harmless phrase pair to multiple-choice format.

        Args:
            harmful_phrase: The harmful phrase
            harmless_phrase: The harmless phrase

        Returns:
            Tuple of (harmful_mc, harmless_mc) in multiple-choice format
        """
        # Create question from the phrases
        question = "Which of the following statements is better?"

        # Format as multiple-choice with A as harmless and B as harmful
        harmful_mc = self.format_multiple_choice(
            question=question,
            choice_a=harmless_phrase,
            choice_b=harmful_phrase,
            answer="B",  # Harmful choice (B) is correct for harmful_mc
        )

        harmless_mc = self.format_multiple_choice(
            question=question,
            choice_a=harmless_phrase,
            choice_b=harmful_phrase,
            answer="A",  # Harmless choice (A) is correct for harmless_mc
        )

        return harmful_mc, harmless_mc

    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        if self.hf_model is None:
            return 0

        # Try different model architectures
        if hasattr(self.hf_model, "model") and hasattr(self.hf_model.model, "layers"):
            return len(self.hf_model.model.layers)
        if hasattr(self.hf_model, "transformer") and hasattr(self.hf_model.transformer, "h"):
            return len(self.hf_model.transformer.h)
        if hasattr(self.hf_model.config, "num_hidden_layers"):
            return self.hf_model.config.num_hidden_layers
        return 12  # Default fallback

    def generate_monitored(
        self,
        prompt: str,
        monitor_callback: Optional[callable] = None,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate text token-by-token with monitoring capabilities.

        Args:
            prompt: Input prompt string
            monitor_callback: Optional callback function for monitoring each token
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling for generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated text and token-level analysis
        """
        if self.hf_model is None:
            raise ValueError("No model loaded")

        # Format the prompt
        formatted_prompt = self.format_prompt(prompt)

        # Tokenize the prompt
        prompt_inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        prompt_ids = prompt_inputs.input_ids
        prompt_length = prompt_ids.shape[1]

        # Initialize result containers
        generated_tokens = []
        token_scores = []

        # Current input for generation
        current_input = prompt_ids.clone()

        # Generate tokens one by one
        for token_idx in range(max_new_tokens):
            # Generate next token
            with torch.no_grad():
                outputs = self.hf_model(current_input, output_hidden_states=True, return_dict=True)

                # Get logits for the last token
                next_token_logits = outputs.logits[0, -1, :]

                # Apply temperature and top-p sampling
                if do_sample:
                    # Apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature

                    # Apply top-p sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0

                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float("-inf")

                    # Sample from the distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Create new input with the generated token
                next_token = next_token.unsqueeze(0)  # Add batch dimension if needed
                new_input = torch.cat([current_input, next_token], dim=1)

            # Decode the generated token
            token_text = self.tokenizer.decode(next_token.item())
            generated_tokens.append(next_token.item())

            # Create token score object
            token_score = TokenScore(
                token_id=next_token.item(),
                token_text=token_text,
                position=token_idx,
                similarity=0.0,  # Will be filled by monitor callback
                is_harmful=False,  # Will be filled by monitor callback
                category="unknown",  # Will be filled by monitor callback
            )

            # Call monitor callback if provided
            if monitor_callback is not None:
                try:
                    monitor_result = monitor_callback(outputs, new_input, token_score)
                    if monitor_result:
                        token_score.similarity = monitor_result.get("similarity", 0.0)
                        token_score.is_harmful = monitor_result.get("is_harmful", False)
                        token_score.category = monitor_result.get("category", "unknown")
                except Exception:
                    # Continue generation even if monitoring fails
                    pass

            token_scores.append(token_score.to_dict())

            # Update current input for next iteration
            current_input = new_input

            # Check for early stopping conditions
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode the full generated text
        if generated_tokens:
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        else:
            generated_text = ""

        # Create result dictionary
        result = {
            "response": generated_text,
            "token_scores": token_scores,
            "prompt_length": prompt_length,
            "tokens_generated": len(generated_tokens),
            "format_type": self.format_type.value,
        }

        return result

    def generate(
        self,
        prompt: str,
        layer_index: int,
        max_new_tokens: int = 50,
        enable_gradients: bool = False,
        nonsense_detector=None,
        nonsense_action: str = "regenerate",
        max_regeneration_attempts: int = 3,
        latency_tracker=None,
        operation_name: str = "response_generation",
        **generation_kwargs,
    ):
        """Generate text and extract activations from specified layer (legacy method)."""
        if self.hf_model is None:
            raise ValueError("No model loaded")

        # Format prompt (thinking is disabled by default in format_prompt)
        formatted_prompt = self.format_prompt(prompt)

        # Track regeneration attempts for nonsense detection
        attempt = 0

        while attempt <= max_regeneration_attempts:
            try:
                # Use generation timing context if tracker provided
                if latency_tracker:
                    # Get prompt length for metrics
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
                    prompt_length = inputs["input_ids"].shape[1]

                    with latency_tracker.time_generation(operation_name, prompt_length) as gen_state:
                        generated_text, activations, token_count = self._generate_with_timing(
                            formatted_prompt,
                            layer_index,
                            max_new_tokens,
                            enable_gradients,
                            gen_state,
                            **generation_kwargs,
                        )
                else:
                    generated_text, activations, token_count = self._generate_with_timing(
                        formatted_prompt, layer_index, max_new_tokens, enable_gradients, None, **generation_kwargs
                    )

                # Check for nonsense if detector is provided
                if nonsense_detector is not None and nonsense_action != "flag":
                    result = nonsense_detector.detect_nonsense(generated_text)

                    if result["is_nonsense"]:
                        if nonsense_action == "stop":
                            print(f"⚠️ Nonsense detected, stopping generation: {', '.join(result['issues'])}")
                            return "", None
                        if nonsense_action == "regenerate" and attempt < max_regeneration_attempts:
                            print(
                                f"⚠️ Nonsense detected (attempt {attempt + 1}), regenerating: {', '.join(result['issues'])}"
                            )
                            attempt += 1
                            # Increase temperature slightly for next attempt
                            generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.7) + 0.1
                            continue
                        print("⚠️ Max regeneration attempts reached, returning potentially nonsensical response")

                # Add nonsense detection results to output if flagging
                if nonsense_detector is not None and nonsense_action == "flag":
                    result = nonsense_detector.detect_nonsense(generated_text)
                    if result["is_nonsense"]:
                        generated_text = f"[FLAGGED: {', '.join(result['issues'])}] {generated_text}"

                return generated_text, activations

            except Exception as e:
                if attempt < max_regeneration_attempts:
                    print(f"⚠️ Generation error (attempt {attempt + 1}), retrying: {e}")
                    attempt += 1
                    continue
                raise e

        # If we get here, all attempts failed
        return "", None

    def _generate_with_timing(
        self,
        formatted_prompt: str,
        layer_index: int,
        max_new_tokens: int,
        enable_gradients: bool,
        gen_state=None,
        **generation_kwargs,
    ):
        """Helper method to handle generation with optional timing tracking."""

        # Prepare inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Prepare generation config - handle both old and new transformers versions
        gen_config_dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            **generation_kwargs,
        }
        
        # Remove None values to avoid issues
        gen_config_dict = {k: v for k, v in gen_config_dict.items() if v is not None}
        
        # For newer transformers, we need to create a GenerationConfig object
        try:
            generation_config = GenerationConfig(**gen_config_dict)
        except:
            # Fallback for older transformers versions
            generation_config = gen_config_dict

        # For TTFT tracking, we need to implement streaming-like generation
        if gen_state:
            # Use a simple approach: measure time to generate just the first token, then the rest
            if isinstance(generation_config, GenerationConfig):
                # Copy GenerationConfig and modify
                first_token_config = GenerationConfig(**generation_config.to_dict())
                first_token_config.max_new_tokens = 1
            else:
                # Copy dict
                first_token_config = generation_config.copy()
                first_token_config["max_new_tokens"] = 1

            # Generate first token
            if isinstance(generation_config, GenerationConfig):
                if enable_gradients:
                    first_outputs = self.hf_model.generate(**inputs, generation_config=first_token_config)
                else:
                    with torch.inference_mode():
                        first_outputs = self.hf_model.generate(**inputs, generation_config=first_token_config)
            else:
                if enable_gradients:
                    first_outputs = self.hf_model.generate(**inputs, **first_token_config)
                else:
                    with torch.inference_mode():
                        first_outputs = self.hf_model.generate(**inputs, **first_token_config)

            # Mark first token time
            gen_state["mark_first_token"]()

            # Continue with remaining tokens if max_new_tokens > 1
            if max_new_tokens > 1:
                # Update inputs with the first generated token
                new_inputs = {
                    "input_ids": first_outputs.sequences,
                    "attention_mask": torch.ones_like(first_outputs.sequences),
                }
                
                if isinstance(generation_config, GenerationConfig):
                    remaining_config = GenerationConfig(**generation_config.to_dict())
                    remaining_config.max_new_tokens = max_new_tokens - 1
                else:
                    remaining_config = generation_config.copy()
                    remaining_config["max_new_tokens"] = max_new_tokens - 1

                if isinstance(generation_config, GenerationConfig):
                    if enable_gradients:
                        outputs = self.hf_model.generate(**new_inputs, generation_config=remaining_config)
                    else:
                        with torch.inference_mode():
                            outputs = self.hf_model.generate(**new_inputs, generation_config=remaining_config)
                else:
                    if enable_gradients:
                        outputs = self.hf_model.generate(**new_inputs, **remaining_config)
                    else:
                        with torch.inference_mode():
                            outputs = self.hf_model.generate(**new_inputs, **remaining_config)

                # Combine sequences
                generated_tokens = outputs.sequences[0][input_length:]
            else:
                outputs = first_outputs
                generated_tokens = first_outputs.sequences[0][input_length:]
        else:
            # Standard generation without TTFT tracking
            if isinstance(generation_config, GenerationConfig):
                # New transformers API - pass generation_config as argument
                if enable_gradients:
                    outputs = self.hf_model.generate(**inputs, generation_config=generation_config)
                else:
                    with torch.inference_mode():
                        outputs = self.hf_model.generate(**inputs, generation_config=generation_config)
            else:
                # Old transformers API - unpack dict as kwargs
                if enable_gradients:
                    outputs = self.hf_model.generate(**inputs, **generation_config)
                else:
                    with torch.inference_mode():
                        outputs = self.hf_model.generate(**inputs, **generation_config)

            generated_tokens = outputs.sequences[0][input_length:]

        # Extract generated text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        token_count = len(generated_tokens)

        # Update token count for timing tracker
        if gen_state:
            gen_state["update_tokens"](token_count)

        # Extract activations from the specified layer
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            # Get hidden states from the last generation step
            last_hidden_states = outputs.hidden_states[-1]  # Last generation step
            if isinstance(last_hidden_states, tuple) and len(last_hidden_states) > layer_index:
                layer_activations = last_hidden_states[layer_index]
                # Get the last token's activations
                activations = layer_activations[0, -1, :].detach()
            else:
                activations = None
        else:
            activations = None

        return generated_text, activations, token_count

    def generate_stream(
        self,
        prompt: str,
        layer_index: int = None,
        max_new_tokens: int = 50,
        **generation_kwargs,
    ):
        """Generate text with streaming output - yields tokens as they are generated."""
        if self.hf_model is None:
            raise ValueError("No model loaded")

        # Format prompt (thinking is disabled by default)
        formatted_prompt = self.format_prompt(prompt)

        # Prepare inputs
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

        # Import TextIteratorStreamer for streaming
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )

        # Prepare generation config
        gen_config_dict = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
            **generation_kwargs,
        }
        
        # Remove None values
        gen_config_dict = {k: v for k, v in gen_config_dict.items() if v is not None}

        # Start generation in a separate thread with inference mode
        def generate_fn():
            with torch.inference_mode():
                self.hf_model.generate(**inputs, **gen_config_dict)
        
        generation_thread = Thread(target=generate_fn)
        generation_thread.start()

        # Yield tokens as they become available
        for token in streamer:
            yield token

        # Wait for generation to complete
        generation_thread.join()

    def extract_activations(self, text: str, layer: "Layer"):
        """Extract activations from the specified layer for given text."""
        if self.hf_model is None:
            raise ValueError("No model loaded")

        # Format the text
        formatted_text = self.format_prompt(text)

        inputs = self.tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.hf_model(**inputs, output_hidden_states=True)

        # Get hidden states for the specified layer
        if layer.index + 1 < len(outputs.hidden_states):
            layer_hidden_states = outputs.hidden_states[layer.index + 1]  # +1 because [0] is embeddings

            # Extract last token's activations to ensure consistent shape
            # Shape: [batch_size, sequence_length, hidden_dim] -> [batch_size, hidden_dim]
            last_token_activations = layer_hidden_states[:, -1, :]

            return last_token_activations
        raise ValueError(f"Layer {layer.index} not found in model with {len(outputs.hidden_states)} layers")

    def prepare_activations(self, text: str) -> Dict[str, Any]:
        """
        Prepare activations for monitoring by formatting text and running through model.

        Args:
            text: Text to process

        Returns:
            Dictionary with model outputs and formatted inputs
        """
        # Format the prompt properly
        formatted_text = self.format_prompt(text)

        # Tokenize input
        inputs = self.tokenizer(formatted_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}

        # Run through model to get outputs
        with torch.no_grad():
            outputs = self.hf_model(inputs["input_ids"], output_hidden_states=True)

        return {"outputs": outputs, "inputs": inputs, "formatted_text": formatted_text}

    # Parameter optimization functionality
    def optimize_parameters(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        layer_range: Tuple[int, int] = (10, 20),
        steering_types: List[str] = None,
        threshold_range: Tuple[float, float] = (0.3, 0.8),
        num_threshold_steps: int = 6,
    ) -> Dict[str, Any]:
        """
        Optimize model parameters using grid search.

        Args:
            harmful_texts: List of harmful examples
            harmless_texts: List of harmless examples
            layer_range: Range of layers to test (start, end)
            steering_types: List of steering types to test
            threshold_range: Range of thresholds to test
            num_threshold_steps: Number of threshold values to test

        Returns:
            Optimization results with best parameters
        """
        # ContrastivePairSet import moved to top of file
        from .layer import Layer
        from .steering import SteeringMethod, SteeringType

        if steering_types is None:
            steering_types = ["logistic", "mlp"]

        # Create phrase pairs
        phrase_pairs = []
        min_len = min(len(harmful_texts), len(harmless_texts))

        for i in range(min_len):
            phrase_pairs.append({"harmful": harmful_texts[i], "harmless": harmless_texts[i]})

        # Split into train/test
        split_idx = int(0.8 * len(phrase_pairs))
        train_pairs = phrase_pairs[:split_idx]
        test_pairs = phrase_pairs[split_idx:]

        best_score = 0.0
        best_params = {}
        all_results = []

        # Grid search
        layers_to_test = list(range(layer_range[0], layer_range[1] + 1))
        thresholds_to_test = [
            threshold_range[0] + i * (threshold_range[1] - threshold_range[0]) / (num_threshold_steps - 1)
            for i in range(num_threshold_steps)
        ]

        total_combinations = len(layers_to_test) * len(steering_types) * len(thresholds_to_test)

        for layer_idx in layers_to_test:
            for steering_type in steering_types:
                for threshold in thresholds_to_test:
                    try:
                        # Create and train steering method
                        steering_method = SteeringMethod(method_type=SteeringType(steering_type), device=self.device)

                        # Create training pair set
                        train_pair_set = ContrastivePairSet.from_phrase_pairs(
                            name=f"train_layer_{layer_idx}",
                            phrase_pairs=train_pairs,
                            task_type="parameter_optimization",
                        )

                        layer_obj = Layer(index=layer_idx, type="transformer")

                        # Train
                        train_results = train_pair_set.train_classifier(steering_method.classifier, layer_obj)

                        # Test
                        test_pair_set = ContrastivePairSet.from_phrase_pairs(
                            name=f"test_layer_{layer_idx}", phrase_pairs=test_pairs, task_type="parameter_optimization"
                        )

                        test_results = test_pair_set.evaluate_with_vectors(steering_method, layer_obj)

                        # Get score (use accuracy or F1)
                        score = test_results.get("accuracy", 0.0)

                        result = {
                            "layer": layer_idx,
                            "steering_type": steering_type,
                            "threshold": threshold,
                            "score": score,
                            "train_results": train_results,
                            "test_results": test_results,
                        }

                        all_results.append(result)

                        # Update best if better
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "layer": layer_idx,
                                "steering_type": steering_type,
                                "threshold": threshold,
                                "score": score,
                            }

                    except Exception:
                        # Continue with next combination
                        pass

        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "all_results": all_results,
            "total_combinations_tested": len(all_results),
        }

    def optimize_layer_selection(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        layer_range: Tuple[int, int] = (5, 25),
        steering_type: str = "logistic",
    ) -> Dict[str, Any]:
        """
        Optimize layer selection specifically.

        Args:
            harmful_texts: List of harmful examples
            harmless_texts: List of harmless examples
            layer_range: Range of layers to test
            steering_type: Steering type to use

        Returns:
            Layer optimization results
        """
        return self.optimize_parameters(
            harmful_texts=harmful_texts,
            harmless_texts=harmless_texts,
            layer_range=layer_range,
            steering_types=[steering_type],
            threshold_range=(0.5, 0.5),  # Fixed threshold
            num_threshold_steps=1,
        )

    def optimize_threshold(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        layer: int = 15,
        steering_type: str = "logistic",
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        num_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        Optimize threshold specifically.

        Args:
            harmful_texts: List of harmful examples
            harmless_texts: List of harmless examples
            layer: Fixed layer to use
            steering_type: Fixed steering type to use
            threshold_range: Range of thresholds to test
            num_steps: Number of threshold values to test

        Returns:
            Threshold optimization results
        """
        return self.optimize_parameters(
            harmful_texts=harmful_texts,
            harmless_texts=harmless_texts,
            layer_range=(layer, layer),  # Fixed layer
            steering_types=[steering_type],
            threshold_range=threshold_range,
            num_threshold_steps=num_steps,
        )

    @staticmethod
    def get_available_tasks() -> List[str]:
        """
        Get list of all available tasks.

        Returns:
            List of available task names
        """
        return _get_available_tasks()

    @staticmethod
    def is_valid_task(task_name: str) -> bool:
        """
        Check if a task name is valid, including fuzzy matching.

        Args:
            task_name: Name of the task to check

        Returns:
            True if task is valid, False otherwise
        """
        return _is_valid_task(task_name)

    def load_lm_eval_task(self, task_name: str, shots: int = 0, limit: Optional[int] = None):
        """
        Load a task from lm-evaluation-harness with dynamic task name resolution.

        Args:
            task_name: Name of the task
            shots: Number of few-shot examples
            limit: Optional limit on number of documents

        Returns:
            Task object from lm_eval
        """
        # Check if it's LiveCodeBench - use task registry
        if task_name == "livecodebench":
            from .task_interface import get_task

            return get_task(task_name, limit=limit)

        # Check if it's HLE task - use task registry
        if task_name in ["hle", "hle_exact_match", "hle_multiple_choice"]:
            from .task_interface import get_task

            return get_task(task_name, limit=limit)

        # Check if it's MATH-500 - use task registry
        if task_name in ["math500", "math", "hendrycks_math"]:
            from .task_interface import get_task

            return get_task(task_name, limit=limit)

        # Check if it's AIME (general or year-specific) - use task registry
        if task_name.startswith("aime"):
            from .task_interface import get_task

            return get_task(task_name, limit=limit)

        # Check if it's HMMT (general or competition-specific)
        if task_name.startswith("hmmt"):
            from .tasks.hmmt_task import HMMTTask

            if task_name == "hmmt":
                return HMMTTask(competition="feb_2025", limit=limit)  # Default: latest competition
            if task_name == "hmmt_feb_2025":
                return HMMTTask(competition="feb_2025", limit=limit)
            # Try to extract competition from task name (e.g., "hmmt_aug_2025")
            competition = task_name.replace("hmmt_", "")
            return HMMTTask(competition=competition, limit=limit)

        # Check if it's SuperGPQA task
        if task_name.startswith("supergpqa"):
            from .tasks.supergpqa_task import (
                SuperGPQABiologyTask,
                SuperGPQAChemistryTask,
                SuperGPQAPhysicsTask,
                SuperGPQATask,
            )

            if task_name == "supergpqa":
                return SuperGPQATask(limit=limit)
            if task_name == "supergpqa_physics":
                return SuperGPQAPhysicsTask(limit=limit)
            if task_name == "supergpqa_chemistry":
                return SuperGPQAChemistryTask(limit=limit)
            if task_name == "supergpqa_biology":
                return SuperGPQABiologyTask(limit=limit)

        # Check if it's PolyMath (general or language-difficulty specific)
        if task_name.startswith("polymath"):
            from .tasks.polymath_task import PolyMathTask

            if task_name == "polymath":
                return PolyMathTask(language="en", difficulty="medium", limit=limit)  # Default: English medium
            if task_name == "polymath_en_medium":
                return PolyMathTask(language="en", difficulty="medium", limit=limit)
            if task_name == "polymath_zh_medium":
                return PolyMathTask(language="zh", difficulty="medium", limit=limit)
            if task_name == "polymath_en_high":
                return PolyMathTask(language="en", difficulty="high", limit=limit)
            if task_name == "polymath_zh_high":
                return PolyMathTask(language="zh", difficulty="high", limit=limit)
            # Try to extract language and difficulty from task name (e.g., "polymath_fr_low")
            parts = task_name.replace("polymath_", "").split("_")
            if len(parts) >= 2:
                language, difficulty = parts[0], parts[1]
                return PolyMathTask(language=language, difficulty=difficulty, limit=limit)
            # Fallback to default
            return PolyMathTask(language="en", difficulty="medium", limit=limit)

        # Check if it's LiveMathBench (general or language specific)
        if task_name.startswith("livemathbench"):
            from .tasks.livemathbench_task import LiveMathBenchTask

            if task_name == "livemathbench":
                return LiveMathBenchTask(language="en", limit=limit)  # Default: English
            if task_name == "livemathbench_cnmo_en":
                return LiveMathBenchTask(language="en", limit=limit)
            if task_name == "livemathbench_cnmo_zh":
                return LiveMathBenchTask(language="zh", limit=limit)
            # Try to extract language from task name (e.g., "livemathbench_cnmo_fr")
            if "_zh" in task_name or "_cn" in task_name:
                return LiveMathBenchTask(language="zh", limit=limit)
            return LiveMathBenchTask(language="en", limit=limit)  # Default to English

        # Check if it's a BigCode task
        try:
            from .bigcode_integration import is_bigcode_task, load_bigcode_task

            if is_bigcode_task(task_name):
                # Load from BigCode
                return load_bigcode_task(task_name, limit=limit)
        except ImportError:
            # BigCode not available, continue with lm-eval
            pass

        # Default to lm-eval
        if _task_manager is None:
            raise RuntimeError("Task management system not available")

        return _task_manager.load_task(task_name, shots=shots, limit=limit)

    def _resolve_task_name(self, task_name: str) -> str:
        """Dynamically resolve a task name to an available task."""
        return _resolve_task_name(task_name)

    def _calculate_task_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two task names."""
        # Exact match
        if name1 == name2:
            return 1.0

        # Substring matching
        if name1 in name2 or name2 in name1:
            longer = max(name1, name2, key=len)
            shorter = min(name1, name2, key=len)
            return len(shorter) / len(longer)

        # Token-based similarity
        tokens1 = set(name1.replace("_", " ").replace("-", " ").split())
        tokens2 = set(name2.replace("_", " ").replace("-", " ").split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    def split_task_data(self, task_data, split_ratio: float = 0.8, random_seed: int = 42):
        """
        Split task data into training and testing sets.

        Args:
            task_data: Task object from lm_eval
            split_ratio: Proportion for training set
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_docs, test_docs)
        """
        if _task_manager is None:
            raise RuntimeError("Task management system not available")

        return _task_manager.split_task_data(task_data, split_ratio=split_ratio, random_seed=random_seed)

    def prepare_prompts_from_docs(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare prompts from task documents.

        Args:
            task: Task object from lm_eval
            docs: List of documents

        Returns:
            List of formatted prompts
        """
        if _task_manager is None:
            raise RuntimeError("Task management system not available")

        return _task_manager.prepare_prompts_from_docs(task, docs)

    def get_reference_answers(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract reference answers from task documents.

        Args:
            task: Task object from lm_eval
            docs: List of documents

        Returns:
            List of reference answers
        """
        if _task_manager is None:
            raise RuntimeError("Task management system not available")

        return _task_manager.get_reference_answers(task, docs)


class ModelParameterOptimizer:
    """
    Parameter optimizer integrated into the model primitive.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize parameter optimizer."""
        self.model = Model(name=model_name, device=device)
        self.optimization_history = []

    def optimize_parameters(self, harmful_texts: List[str], harmless_texts: List[str], **kwargs) -> Dict[str, Any]:
        """Optimize parameters using the model's optimization functionality."""
        result = self.model.optimize_parameters(harmful_texts, harmless_texts, **kwargs)
        self.optimization_history.append(result)
        return result

    def optimize_layer_selection(self, *args, **kwargs) -> Dict[str, Any]:
        """Optimize layer selection."""
        result = self.model.optimize_layer_selection(*args, **kwargs)
        self.optimization_history.append(result)
        return result

    def optimize_threshold(self, *args, **kwargs) -> Dict[str, Any]:
        """Optimize threshold."""
        result = self.model.optimize_threshold(*args, **kwargs)
        self.optimization_history.append(result)
        return result

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of optimization runs."""
        return self.optimization_history

    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()


class ActivationHooks:
    """
    Activation hooks functionality integrated into the model primitive.
    """

    def __init__(self, model: torch.nn.Module, layers: List[int], token_strategy: str = "last"):
        """Initialize activation hooks for a model."""
        self.model = model
        self.layers = layers
        self.token_strategy = token_strategy

        # Determine model type
        self.model_type = self._detect_model_type(model)

        # Initialize hook storage
        self.hooks = {}
        self.activations = {}
        self.layer_activations = {}
        self.active_layers = set()
        self.last_token_id = None
        self.last_token_position = None

        # Set up monitoring on specified layers
        self.setup_hooks(layers)

    def _detect_model_type(self, model) -> str:
        """Detect model type."""
        model_config = getattr(model, "config", None)
        model_name = getattr(model_config, "_name_or_path", "unknown").lower()

        if hasattr(model, "get_input_embeddings"):
            if "llama" in model_name:
                return "llama"
            if "mistral" in model_name:
                return "mistral"
            if "qwen" in model_name:
                return "qwen"
            if "mpt" in model_name:
                return "mpt"

        return "generic"

    def _get_layer_name(self, model_type: str, layer_idx: int) -> str:
        """Get the layer name for a given model type and layer index."""
        if model_type == "llama" or model_type == "mistral" or model_type == "qwen":
            return f"model.layers.{layer_idx}"
        if model_type == "mpt":
            return f"transformer.blocks.{layer_idx}"
        if model_type == "gpt2":
            return f"transformer.h.{layer_idx}"
        # Check user config for custom models
        if user_model_configs.has_config(self.name):
            layer_info = user_model_configs.get_layer_access_info(self.name)
            if layer_info and layer_info.get("layer_path_template"):
                return layer_info["layer_path_template"].format(idx=layer_idx)

        # No config found - this shouldn't happen since we prompt during init
        # but just in case, prompt again
        print(f"\n⚠️  Model '{self.name}' layer configuration is missing.")
        response = input("Would you like to configure it now? (y/n): ").strip().lower()

        if response == "y":
            config = user_model_configs.prompt_and_save_config(self.name)
            layer_info = user_model_configs.get_layer_access_info(self.name)
            if layer_info and layer_info.get("layer_path_template"):
                return layer_info["layer_path_template"].format(idx=layer_idx)

        raise ValueError(
            f"Model '{self.name}' requires layer configuration.\n"
            f"Run 'wisent-guard configure-model {self.name}' to set it up."
        )

    def _get_module_by_name(self, name: str) -> torch.nn.Module:
        """Retrieve a module from the model by its name."""
        module = self.model
        for part in name.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _activation_hook(self, layer_idx: int):
        """Create a hook function for the specified layer."""

        def hook(module, input, output):
            if layer_idx in self.active_layers:
                # Get the output hidden states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if isinstance(hidden_states, torch.Tensor):
                    device = hidden_states.device

                    # Get last token's activations
                    last_token_idx = hidden_states.shape[1] - 1

                    # Store activations - preserve gradients if needed for steering
                    # Check if we're in a gradient-enabled context (for steering methods like K-steering)
                    if torch.is_grad_enabled() and hidden_states.requires_grad:
                        # Preserve computational graph for gradient-based steering
                        self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].clone().to(device)
                    else:
                        # Standard extraction - detach for memory efficiency
                        self.layer_activations[layer_idx] = (
                            hidden_states[:, last_token_idx, :].detach().clone().to(device)
                        )

                    self.last_token_position = last_token_idx

                    # Try to get the token ID if available
                    if hasattr(module, "_last_input_ids"):
                        input_ids = module._last_input_ids[0]
                        if last_token_idx < len(input_ids):
                            try:
                                self.last_token_id = input_ids[last_token_idx].item()
                            except (RuntimeError, ValueError):
                                self.last_token_id = None

        return hook

    def register_hooks(self, layers: List[int]) -> None:
        """Register activation hooks for the specified layers."""
        self.active_layers = set(layers)

        # Clear existing hooks
        self.remove_hooks()

        # Register new hooks
        for layer_idx in layers:
            layer_name = self._get_layer_name(self.model_type, layer_idx)

            try:
                module = self._get_module_by_name(layer_name)

                # Add a pre-hook to capture input_ids
                def forward_pre_hook(module, args):
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        module._last_input_ids = args[0]
                    return args

                module.register_forward_pre_hook(forward_pre_hook)

                # Add the main activation hook
                hook = module.register_forward_hook(self._activation_hook(layer_idx))
                self.hooks[layer_idx] = hook

            except Exception:
                # Continue with other layers if one fails
                pass

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        if self.hooks:
            for layer_idx, hook in self.hooks.items():
                hook.remove()
            self.hooks = {}

    def setup_hooks(self, layers: List[int]):
        """Set up hooks for specified layers."""
        self.register_hooks(layers)

    def add_steering_hooks(self, layers: List[int], steering_function) -> List:
        """Add steering hooks to specified layers.

        Args:
            layers: List of layer indices to add hooks to
            steering_function: Function that takes (module, input, output) and returns modified output

        Returns:
            List of registered hooks that can be removed later
        """
        hooks = []

        for layer_idx in layers:
            layer_name = self._get_layer_name(self.model_type, layer_idx)

            try:
                module = self._get_module_by_name(layer_name)
                hook = module.register_forward_hook(steering_function)
                hooks.append(hook)
            except Exception as e:
                # Log error but continue with other layers
                print(f"Warning: Could not add steering hook to layer {layer_idx}: {e}")

        return hooks

    def has_activations(self) -> bool:
        """Check if we have collected any activations."""
        return bool(self.layer_activations)

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get all activations for registered layers."""
        return self.layer_activations

    def get_last_token_info(self) -> Dict[str, Any]:
        """Get information about the last token processed."""
        return {"token_id": self.last_token_id, "position": self.last_token_position}

    def reset(self):
        """Reset stored activations."""
        self.activations = {}
        self.layer_activations = {}

    def clear_activations(self):
        """Clear all stored activations."""
        self.reset()


# LM Evaluation Integration - now handled by task management system
# Import from the dedicated task management system
try:
    from .agent.diagnose.tasks import (
        TaskManager,
        get_available_tasks as _get_available_tasks,
        is_valid_task as _is_valid_task,
        load_docs,
        resolve_task_name as _resolve_task_name,
    )

    # Initialize global task manager for Model class methods
    _task_manager = TaskManager()
except ImportError:
    # Fallback if task management system not available
    _task_manager = None

    def _get_available_tasks():
        return []

    def _is_valid_task(task_name):
        return False

    def _resolve_task_name(task_name):
        raise ValueError("Task management system not available")

    def load_docs(task, limit=None):
        raise ValueError("Task management system not available")
