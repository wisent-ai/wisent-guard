from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re
from typing import Optional, Union, List, Dict, Any, Tuple
from enum import Enum

class PromptFormat(Enum):
    LEGACY = "legacy"
    LLAMA31 = "llama31"
    MISTRAL = "mistral"

class TokenScore:
    """Stores information about a token and its similarity to harmful content."""
    
    def __init__(self, 
                 token_id: Optional[int] = None,
                 token_text: str = "",
                 position: int = 0,
                 similarity: float = 0.0,
                 is_harmful: bool = False,
                 category: Optional[str] = None,
                 activations: Optional[Dict[int, torch.Tensor]] = None):
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
    def __init__(self, name: str, layers: Optional[List[int]] = None, device: Optional[str] = None, hf_model: Optional[AutoModelForCausalLM] = None):
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
        
        # Prompt formatting settings
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.format_type = None  # Will be auto-detected
        
        if hf_model is not None:
            # Use provided model
            self.hf_model = hf_model
            self.model = hf_model  # Keep backward compatibility
            # Try to load tokenizer from the same name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except:
                self.tokenizer = None
        else:
            # Load model from scratch
            self.hf_model = None
            self.model = None
            self.tokenizer = None
            self._load_model_and_tokenizer()
        
        # Auto-detect format type
        self.format_type = self._detect_format()

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
            self.name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            output_hidden_states=True
        )
        self.hf_model.config.output_hidden_states = True
        self.hf_model.eval()
        
        # Keep backward compatibility
        self.model = self.hf_model

    def _detect_format(self) -> PromptFormat:
        """
        Detect the appropriate format to use based on the model.
        
        Returns:
            PromptFormat enum indicating the format
        """
        model_name = self.name.lower()
        
        # Check for specific model types
        if re.search(r"llama-?3", model_name, re.IGNORECASE):
            return PromptFormat.LLAMA31
        elif "mistral" in model_name:
            return PromptFormat.MISTRAL
        else:
            return PromptFormat.LEGACY

    def set_prompt_tokens(self, user_token: str, assistant_token: str):
        """Set custom user and assistant tokens for legacy format."""
        self.user_token = user_token
        self.assistant_token = assistant_token

    def format_prompt(self, prompt: str, response: str = None) -> str:
        """
        Format a prompt using the appropriate format (Llama 3.1, Mistral, or legacy).
        
        Args:
            prompt: Input prompt text
            response: Optional response text to include
            
        Returns:
            Formatted prompt string
        """
        if self.format_type == PromptFormat.LLAMA31:
            # Llama 3.1 special tokens
            BEGIN_TEXT = "<|begin_of_text|>"
            START_HEADER = "<|start_header_id|>"
            END_HEADER = "<|end_header_id|>"
            EOT = "<|eot_id|>"
            
            if response is not None:
                return f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{prompt}{EOT}{START_HEADER}assistant{END_HEADER}\n{response}"
            else:
                return f"{BEGIN_TEXT}{START_HEADER}user{END_HEADER}\n{prompt}{EOT}{START_HEADER}assistant{END_HEADER}\n"
        elif self.format_type == PromptFormat.MISTRAL:
            # Mistral format tokens
            INST_START = "[INST]"
            INST_END = "[/INST]"
            
            if response is not None:
                return f"{INST_START} {prompt} {INST_END} {response}"
            else:
                return f"{INST_START} {prompt} {INST_END}"
        else:
            # Legacy format
            if response is not None:
                return f"{self.user_token}\n{prompt}\n{self.assistant_token}\n{response}"
            else:
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
            answer="B"  # Harmful choice (B) is correct for harmful_mc
        )
        
        harmless_mc = self.format_multiple_choice(
            question=question,
            choice_a=harmless_phrase,
            choice_b=harmful_phrase,
            answer="A"  # Harmless choice (A) is correct for harmless_mc
        )
        
        return harmful_mc, harmless_mc

    def get_num_layers(self) -> int:
        """Get the number of layers in the model."""
        if self.hf_model is None:
            return 0
        
        # Try different model architectures
        if hasattr(self.hf_model, 'model') and hasattr(self.hf_model.model, 'layers'):
            return len(self.hf_model.model.layers)
        elif hasattr(self.hf_model, 'transformer') and hasattr(self.hf_model.transformer, 'h'):
            return len(self.hf_model.transformer.h)
        elif hasattr(self.hf_model.config, 'num_hidden_layers'):
            return self.hf_model.config.num_hidden_layers
        else:
            return 12  # Default fallback

    def generate_monitored(
        self,
        prompt: str,
        monitor_callback: Optional[callable] = None,
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
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
                outputs = self.hf_model(
                    current_input,
                    output_hidden_states=True,
                    return_dict=True
                )
                
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
                        next_token_logits[indices_to_remove] = float('-inf')
                    
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
                category="unknown"  # Will be filled by monitor callback
            )
            
            # Call monitor callback if provided
            if monitor_callback is not None:
                try:
                    monitor_result = monitor_callback(outputs, new_input, token_score)
                    if monitor_result:
                        token_score.similarity = monitor_result.get('similarity', 0.0)
                        token_score.is_harmful = monitor_result.get('is_harmful', False)
                        token_score.category = monitor_result.get('category', 'unknown')
                except Exception as e:
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
            "format_type": self.format_type.value
        }
        
        return result

    def generate(self, prompt: str, layer_index: int, max_new_tokens: int = 50, **generation_kwargs):
        """Generate text and extract activations from specified layer (legacy method)."""
        if self.hf_model is None:
            raise ValueError("No model loaded")
            
        # Format prompt
        formatted_prompt = self.format_prompt(prompt)
            
        with torch.inference_mode():
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.hf_model.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "pad_token_id": self.tokenizer.pad_token_id,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
                **generation_kwargs
            }
            
            outputs = self.hf_model.generate(**inputs, **generation_config)
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract hidden states
            if outputs.hidden_states and len(outputs.hidden_states) > 0:
                last_step_hidden_states = outputs.hidden_states[-1]
                if layer_index < len(last_step_hidden_states):
                    layer_hidden_state = last_step_hidden_states[layer_index][0, -1, :]
                else:
                    layer_hidden_state = last_step_hidden_states[0][0, -1, :]
            else:
                with torch.no_grad():
                    forward_outputs = self.hf_model(**inputs)
                    layer_hidden_state = forward_outputs.hidden_states[layer_index][0, -1, :]
                    
            return generated_text, layer_hidden_state.cpu().to(torch.float32)

    def extract_activations(self, text: str, layer: 'Layer'):
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
        else:
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
        inputs = self.tokenizer(formatted_text, return_tensors="pt").to(self.device)
        
        # Run through model to get outputs
        with torch.no_grad():
            outputs = self.hf_model(inputs.input_ids, output_hidden_states=True)
        
        return {
            'outputs': outputs,
            'inputs': inputs,
            'formatted_text': formatted_text
        }

    # Parameter optimization functionality
    def optimize_parameters(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        layer_range: Tuple[int, int] = (10, 20),
        steering_types: List[str] = None,
        threshold_range: Tuple[float, float] = (0.3, 0.8),
        num_threshold_steps: int = 6
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
        from .steering import SteeringMethod, SteeringType
        from .contrastive_pair_set import ContrastivePairSet
        from .layer import Layer
        
        if steering_types is None:
            steering_types = ["logistic", "mlp"]
        
        # Create phrase pairs
        phrase_pairs = []
        min_len = min(len(harmful_texts), len(harmless_texts))
        
        for i in range(min_len):
            phrase_pairs.append({
                "harmful": harmful_texts[i],
                "harmless": harmless_texts[i]
            })
        
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
                        steering_method = SteeringMethod(
                            method_type=SteeringType(steering_type),
                            device=self.device
                        )
                        
                        # Create training pair set
                        train_pair_set = ContrastivePairSet.from_phrase_pairs(
                            name=f"train_layer_{layer_idx}",
                            phrase_pairs=train_pairs,
                            task_type="parameter_optimization"
                        )
                        
                        layer_obj = Layer(index=layer_idx, type="transformer")
                        
                        # Train
                        train_results = train_pair_set.train_classifier(
                            steering_method.classifier,
                            layer_obj
                        )
                        
                        # Test
                        test_pair_set = ContrastivePairSet.from_phrase_pairs(
                            name=f"test_layer_{layer_idx}",
                            phrase_pairs=test_pairs,
                            task_type="parameter_optimization"
                        )
                        
                        test_results = test_pair_set.evaluate_with_vectors(
                            steering_method,
                            layer_obj
                        )
                        
                        # Get score (use accuracy or F1)
                        score = test_results.get("accuracy", 0.0)
                        
                        result = {
                            "layer": layer_idx,
                            "steering_type": steering_type,
                            "threshold": threshold,
                            "score": score,
                            "train_results": train_results,
                            "test_results": test_results
                        }
                        
                        all_results.append(result)
                        
                        # Update best if better
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "layer": layer_idx,
                                "steering_type": steering_type,
                                "threshold": threshold,
                                "score": score
                            }
                        
                    except Exception as e:
                        # Continue with next combination
                        pass
        
        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "all_results": all_results,
            "total_combinations_tested": len(all_results)
        }
    
    def optimize_layer_selection(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        layer_range: Tuple[int, int] = (5, 25),
        steering_type: str = "logistic"
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
            num_threshold_steps=1
        )
    
    def optimize_threshold(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        layer: int = 15,
        steering_type: str = "logistic",
        threshold_range: Tuple[float, float] = (0.1, 0.9),
        num_steps: int = 10
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
            num_threshold_steps=num_steps
        )
    
    @staticmethod
    def get_available_tasks() -> List[str]:
        """
        Get list of all available tasks.
        
        Returns:
            List of available task names
        """
        return AVAILABLE_TASKS.copy()
    
    @staticmethod
    def is_valid_task(task_name: str) -> bool:
        """
        Check if a task name is valid.
        
        Args:
            task_name: Name of the task to check
            
        Returns:
            True if task is valid, False otherwise
        """
        actual_task_name = TASK_NAME_MAPPINGS.get(task_name, task_name)
        return actual_task_name in AVAILABLE_TASKS
    
    def load_lm_eval_task(self, task_name: str, shots: int = 0, limit: Optional[int] = None):
        """
        Load a task from lm-evaluation-harness.
        
        Args:
            task_name: Name of the task
            shots: Number of few-shot examples
            limit: Optional limit on number of documents
            
        Returns:
            Task object from lm_eval
        """
        try:
            from lm_eval.tasks import get_task_dict
            from lm_eval.api.registry import TASK_REGISTRY
        except ImportError as e:
            raise ImportError(
                "lm-evaluation-harness is required. Install with: pip install lm-eval"
            ) from e
        
        actual_task_name = TASK_NAME_MAPPINGS.get(task_name, task_name)
        
        # Check if task is in our available tasks list
        if not self.is_valid_task(task_name):
            raise ValueError(
                f"Task '{task_name}' (mapped to '{actual_task_name}') not found in available tasks. "
                f"Use Model.get_available_tasks() to see all available tasks."
            )
        
        try:
            task_dict = get_task_dict([actual_task_name])
            
            if actual_task_name not in task_dict:
                raise ValueError(
                    f"Task '{task_name}' (mapped to '{actual_task_name}') not found in lm_eval registry."
                )
            
            task = task_dict[actual_task_name]
            task._limit = limit
            
            return task
            
        except Exception as e:
            raise ValueError(f"Failed to load task '{task_name}': {e}") from e
    
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
        import random
        
        docs = load_docs(task_data, limit=getattr(task_data, '_limit', None))
        
        if not docs:
            return [], []
        
        if len(docs) < 2:
            return docs, docs
        
        # Simple split implementation
        random.seed(random_seed)
        shuffled_docs = docs.copy()
        random.shuffle(shuffled_docs)
        
        split_idx = int(len(shuffled_docs) * split_ratio)
        train_docs = shuffled_docs[:split_idx]
        test_docs = shuffled_docs[split_idx:]
        
        return train_docs, test_docs
    
    def prepare_prompts_from_docs(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Prepare prompts from task documents.
        
        Args:
            task: Task object from lm_eval
            docs: List of documents
            
        Returns:
            List of formatted prompts
        """
        prompts = []
        
        for doc in docs:
            try:
                # Use task's doc_to_text method if available
                if hasattr(task, 'doc_to_text'):
                    prompt = task.doc_to_text(doc)
                else:
                    # Fallback: extract common fields
                    if 'question' in doc:
                        prompt = doc['question']
                    elif 'text' in doc:
                        prompt = doc['text']
                    elif 'prompt' in doc:
                        prompt = doc['prompt']
                    else:
                        # Use first string value found
                        for key, value in doc.items():
                            if isinstance(value, str) and len(value) > 10:
                                prompt = value
                                break
                        else:
                            prompt = str(doc)
                
                prompts.append(str(prompt))
                
            except Exception:
                prompts.append(str(doc))
        
        return prompts
    
    def get_reference_answers(self, task, docs: List[Dict[str, Any]]) -> List[str]:
        """
        Extract reference answers from task documents.
        
        Args:
            task: Task object from lm_eval
            docs: List of documents
            
        Returns:
            List of reference answers
        """
        references = []
        
        for doc in docs:
            try:
                # Use task's doc_to_target method if available
                if hasattr(task, 'doc_to_target'):
                    reference = task.doc_to_target(doc)
                else:
                    # Fallback: extract common answer fields
                    if 'answer' in doc:
                        reference = doc['answer']
                    elif 'target' in doc:
                        reference = doc['target']
                    elif 'label' in doc:
                        reference = doc['label']
                    else:
                        reference = "Unknown"
                
                references.append(str(reference))
                
            except Exception:
                references.append("Unknown")
        
        return references


class ModelParameterOptimizer:
    """
    Parameter optimizer integrated into the model primitive.
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None
    ):
        """Initialize parameter optimizer."""
        self.model = Model(name=model_name, device=device)
        self.optimization_history = []
    
    def optimize_parameters(
        self,
        harmful_texts: List[str],
        harmless_texts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
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
    
    def __init__(
        self,
        model: torch.nn.Module,
        layers: List[int],
        token_strategy: str = "last"
    ):
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
            elif "mistral" in model_name:
                return "mistral"
            elif "mpt" in model_name:
                return "mpt"
        
        return "generic"
    
    def _get_layer_name(self, model_type: str, layer_idx: int) -> str:
        """Get the layer name for a given model type and layer index."""
        if model_type == "llama":
            return f"model.layers.{layer_idx}"
        elif model_type == "mistral":
            return f"model.layers.{layer_idx}"
        elif model_type == "mpt":
            return f"transformer.blocks.{layer_idx}"
        else:
            # Generic approach - try common patterns
            return f"model.layers.{layer_idx}"
    
    def _get_module_by_name(self, name: str) -> torch.nn.Module:
        """Retrieve a module from the model by its name."""
        module = self.model
        for part in name.split('.'):
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
                    
                    # Store activations
                    self.layer_activations[layer_idx] = hidden_states[:, last_token_idx, :].detach().clone().to(device)
                    self.last_token_position = last_token_idx
                    
                    # Try to get the token ID if available
                    if hasattr(module, '_last_input_ids'):
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
    
    def has_activations(self) -> bool:
        """Check if we have collected any activations."""
        return bool(self.layer_activations)
    
    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get all activations for registered layers."""
        return self.layer_activations
    
    def get_last_token_info(self) -> Dict[str, Any]:
        """Get information about the last token processed."""
        return {
            "token_id": self.last_token_id,
            "position": self.last_token_position
        }
    
    def reset(self):
        """Reset stored activations."""
        self.activations = {}
        self.layer_activations = {}
    
    def clear_activations(self):
        """Clear all stored activations."""
        self.reset()


# LM Evaluation Integration
import json
import os

def load_available_tasks():
    """Load available tasks from tasks.json file."""
    try:
        tasks_file = os.path.join(os.path.dirname(__file__), 'tasks.json')
        with open(tasks_file, 'r') as f:
            data = json.load(f)
            return data.get('tasks', [])
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to basic tasks if file not found
        return [
            'truthfulqa_mc1', 'truthfulqa_mc2', 'hellaswag', 'mmlu', 
            'arc_easy', 'arc_challenge', 'winogrande', 'piqa', 'boolq'
        ]

# Load available tasks
AVAILABLE_TASKS = load_available_tasks()

# Task name mappings for common aliases
TASK_NAME_MAPPINGS = {
    'truthfulqa': 'truthfulqa_mc1',
    'truthful_qa': 'truthfulqa_mc1',
    'hellaswag': 'hellaswag',
    'mmlu': 'mmlu_abstract_algebra',
    'mmlu_easy': 'mmlu_elementary_mathematics',
    'arc_easy': 'arc_easy',
    'arc_challenge': 'arc_challenge',
    'winogrande': 'winogrande',
    'piqa': 'piqa',
    'boolq': 'boolq',
}


def load_docs(task, limit: Optional[int] = None):
    """
    Load documents from the most appropriate split (validation → test → train).
    
    Args:
        task: Task object from lm_eval
        limit: Optional limit on number of documents to load
        
    Returns:
        List of documents from the most appropriate split
    """
    docs = []
    
    if task.has_validation_docs():
        docs = list(task.validation_docs())
    elif task.has_test_docs():
        docs = list(task.test_docs())
    elif task.has_training_docs():
        docs = list(task.training_docs())
    else:
        raise RuntimeError(f"No labelled docs available for task {task.NAME}")
    
    if limit is not None and limit > 0:
        docs = docs[:limit]
    
    return docs 