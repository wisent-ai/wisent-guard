#!/usr/bin/env python3
"""
Generate testing responses and save layer activations for classifier evaluation.

This script uses the 20% test split to generate responses according to each 
benchmark's evaluation method and saves activations from all layers using 
existing Model class functionality.
"""

import os
import sys
import json
import pickle
import torch
import time
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parents[2]  # Go up to wisent-activation-guardrails root
sys.path.insert(0, str(project_root))

from wisent_guard.core.model import Model
from wisent_guard.core.layer import Layer


class TestingResponseGenerator:
    """Generate testing responses and save layer activations for classifier evaluation."""
    
    def __init__(self, model_name: str, output_dir: Optional[str] = None):
        """
        Initialize the testing response generator.
        
        Args:
            model_name: Name of the model to use
            output_dir: Directory to save testing responses and activations
        """
        self.model_name = model_name
        self.script_dir = Path(__file__).parent
        self.benchmarks_dir = self.script_dir / "full_benchmarks" / "data"
        
        # Load evaluation methods
        methods_file = project_root / "wisent_guard" / "parameters" / "benchmarks" / "benchmark_evaluation_methods.json"
        with open(methods_file, 'r') as f:
            self.evaluation_methods = json.load(f)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = self.script_dir / "testing_responses"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧪 Testing Response Generator")
        print(f"   Model: {model_name}")
        print(f"   Benchmarks source: {self.benchmarks_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Evaluation methods: {len(self.evaluation_methods)} benchmarks")
        
        # Initialize model
        print(f"\n🔄 Loading model...")
        self.model = Model(model_name)
        self.num_layers = self.model.get_num_layers()
        print(f"   ✅ Model loaded with {self.num_layers} layers")
        
        # Check benchmarks directory exists (warn but don't fail - auto-download will handle this)
        if not self.benchmarks_dir.exists():
            print(f"⚠️  Benchmarks directory not found: {self.benchmarks_dir}")
            print(f"   📥 Benchmark data will be downloaded automatically if needed")
    
    def get_available_benchmarks(self) -> List[str]:
        """Get list of available downloaded benchmarks."""
        if not self.benchmarks_dir.exists():
            return []
        
        pkl_files = list(self.benchmarks_dir.glob("*.pkl"))
        all_benchmarks = [f.stem for f in pkl_files]
        
        # Filter to only benchmarks we have evaluation methods for
        available_benchmarks = [b for b in all_benchmarks if b in self.evaluation_methods]
        
        return available_benchmarks
    
    def load_benchmark_data(self, benchmark_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Load benchmark data from file."""
        data_file = self.benchmarks_dir / f"{benchmark_name}.pkl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Benchmark data not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            data_container = pickle.load(f)
        
        # Extract actual samples from the data container
        if isinstance(data_container, dict) and 'all_samples' in data_container:
            data = data_container['all_samples']
        else:
            data = data_container
        
        # Split data into train/test based on same logic as classifier training
        split_ratio = 0.8
        split_index = int(len(data) * split_ratio)
        
        train_data = data[:split_index]
        test_data = data[split_index:]
        
        return train_data, test_data
    
    def extract_activations_from_all_layers(self, text: str) -> Dict[int, torch.Tensor]:
        """
        Extract activations from all layers for given text using existing Model methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping layer indices to activation tensors
        """
        activations = {}
        
        if not text or not text.strip():
            print(f"      ⚠️ Empty text provided for activation extraction")
            return {layer: None for layer in range(self.num_layers)}
        
        for layer_idx in range(self.num_layers):
            try:
                layer = Layer(index=layer_idx, type="transformer")
                layer_activations = self.model.extract_activations(text, layer)
                
                # Ensure we have a valid tensor
                if layer_activations is not None and isinstance(layer_activations, torch.Tensor):
                    # Convert to expected format - ensure we have a 2D tensor [batch_size, hidden_dim]
                    if layer_activations.dim() == 1:
                        layer_activations = layer_activations.unsqueeze(0)
                    activations[layer_idx] = layer_activations
                else:
                    print(f"      ⚠️ Layer {layer_idx}: No activations returned")
                    activations[layer_idx] = None
                    
            except Exception as e:
                print(f"      ⚠️ Failed to extract activations from layer {layer_idx}: {e}")
                activations[layer_idx] = None
        
        # Count successful extractions
        successful_layers = [k for k, v in activations.items() if v is not None]
        print(f"      ✅ Successfully extracted activations from {len(successful_layers)}/{self.num_layers} layers")
        
        return activations
    
    def generate_log_likelihood_response(self, sample: Dict, benchmark_name: str) -> Dict[str, Any]:
        """
        Generate response for log-likelihood evaluation benchmarks.
        
        Args:
            sample: Data sample
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with response data
        """
        question = sample.get('question', sample.get('premise', sample.get('text', sample.get('query', sample.get('ctx', '')))))
        choices = sample.get('choices', sample.get('endings', []))
        correct_answer = sample.get('answer', sample.get('label', 0))
        
        if not choices:
            # Some benchmarks might have different structures
            if 'mc1_targets' in sample:
                choices = sample['mc1_targets']['choices']
                correct_answer = sample['mc1_targets']['labels'].index(1)
            elif 'mc2_targets' in sample:
                choices = sample['mc2_targets']['choices']
                labels = sample['mc2_targets']['labels']
                correct_answer = [i for i, label in enumerate(labels) if label == 1]
            else:
                # Boolean question
                choices = ['No', 'Yes']
                correct_answer = sample.get('answer', 0)
        
        # Use proper log-likelihood computation following the existing pattern
        choice_loglikelihoods = []
        full_texts = []
        
        for choice in choices:
            # Create full text
            if benchmark_name in ['truthfulqa_mc1', 'truthfulqa_mc2']:
                context = f"Q: {question}\nA: "
                continuation = choice
            else:
                context = f"{question} "
                continuation = choice
            
            full_text = context + continuation
            full_texts.append(full_text)
            
            # Compute log-likelihood using proper tokenization and model forward pass
            try:
                # Tokenize context and continuation separately
                context_tokens = self.model.tokenizer.encode(context, add_special_tokens=False) if context else []
                full_tokens = self.model.tokenizer.encode(full_text, add_special_tokens=False)
                
                # The continuation tokens are the difference
                continuation_tokens = full_tokens[len(context_tokens):]
                
                if not continuation_tokens:
                    choice_loglikelihoods.append(float('-inf'))
                    continue
                
                # Convert to tensor
                input_ids = torch.tensor([full_tokens], device=self.model.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model.model(input_ids)
                    logits = outputs.logits
                
                # Compute log-likelihood for continuation tokens
                continuation_start = len(context_tokens)
                continuation_end = len(full_tokens)
                
                if continuation_start >= logits.shape[1]:
                    choice_loglikelihoods.append(float('-inf'))
                    continue
                
                # Get logits for the continuation positions (shifted by 1 for next-token prediction)
                target_logits = logits[0, continuation_start-1:continuation_end-1]  # Shape: [cont_len, vocab_size]
                target_tokens = torch.tensor(continuation_tokens, device=self.model.device)
                
                # Compute log probabilities
                log_probs = torch.log_softmax(target_logits, dim=-1)
                
                # Get log probability for each target token
                token_log_probs = log_probs[range(len(continuation_tokens)), target_tokens]
                
                # Sum log probabilities (since log(a*b) = log(a) + log(b))
                total_log_likelihood = token_log_probs.sum().item()
                
                choice_loglikelihoods.append(total_log_likelihood)
                
            except Exception as e:
                print(f"      ⚠️ Failed to get loglikelihood for choice '{choice}': {e}")
                choice_loglikelihoods.append(float('-inf'))
        
        # Predicted answer is the choice with highest log-likelihood
        predicted_answer = int(np.argmax(choice_loglikelihoods))
        
        return {
            'question': question,
            'choices': choices,
            'choice_loglikelihoods': choice_loglikelihoods,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'evaluation_method': 'log-likelihoods',
            'activation_text': question  # Text to use for activation extraction
        }
    
    def generate_text_generation_response(self, sample: Dict, benchmark_name: str) -> Dict[str, Any]:
        """
        Generate response for text generation evaluation benchmarks.
        
        Args:
            sample: Data sample
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with response data
        """
        question = sample.get('question', sample.get('input', sample.get('text', sample.get('query', sample.get('ctx', '')))))
        
        # Get correct answer(s)
        if 'answers' in sample:
            correct_answers = sample['answers']['text'] if isinstance(sample['answers'], dict) else sample['answers']
        else:
            correct_answers = [sample.get('answer', sample.get('target', ''))]
        
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]
        
        # Generate response
        if benchmark_name in ['gsm8k', 'asdiv']:
            prompt = f"Question: {question}\nAnswer:"
        elif benchmark_name in ['coqa', 'drop']:
            context = sample.get('story', sample.get('passage', ''))
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"{question}"
        
        # Use model's existing generation method
        try:
            # Use a dummy layer for generation (required parameter)
            generated_text, _ = self.model.generate(
                prompt, 
                layer_index=15,  # Middle layer
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False
            )
        except Exception as e:
            print(f"      ⚠️ Failed to generate text: {e}")
            generated_text = ""
        
        return {
            'question': question,
            'prompt': prompt,
            'generated_text': generated_text,
            'correct_answers': correct_answers,
            'evaluation_method': 'text-generation',
            'activation_text': prompt  # Text to use for activation extraction
        }
    
    def generate_perplexity_response(self, sample: Dict, benchmark_name: str) -> Dict[str, Any]:
        """
        Generate response for perplexity evaluation benchmarks.
        
        Args:
            sample: Data sample
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with response data
        """
        text = sample.get('text', sample.get('page', ''))
        
        # Use model's existing functionality to compute perplexity
        try:
            # Use model's prepare_activations to run through the model
            prepared = self.model.prepare_activations(text)
            outputs = prepared['outputs']
            inputs = prepared['inputs']
            
            # Compute perplexity from logits
            input_ids = inputs['input_ids']
            logits = outputs.logits
            
            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Get log probabilities for actual tokens (shifted for next-token prediction)
            token_log_probs = log_probs[0, :-1, :].gather(dim=-1, index=input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            
            # Compute perplexity
            avg_log_prob = token_log_probs.mean().item()
            perplexity = np.exp(-avg_log_prob)
            
        except Exception as e:
            print(f"      ⚠️ Failed to compute perplexity: {e}")
            perplexity = float('inf')
        
        return {
            'text': text[:200] + "..." if len(text) > 200 else text,  # Truncate for storage
            'perplexity': perplexity,
            'evaluation_method': 'perplexity',
            'activation_text': text  # Text to use for activation extraction
        }
    
    def compute_evaluation_result(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute evaluation results for a response.
        
        Args:
            response_data: Response data dictionary
            
        Returns:
            Dictionary with evaluation results
        """
        evaluation_method = response_data.get('evaluation_method', 'unknown')
        
        if evaluation_method == 'log-likelihoods':
            # For log-likelihood benchmarks, check if predicted answer matches correct answer
            predicted = response_data.get('predicted_answer', -1)
            correct = response_data.get('correct_answer', -1)
            is_correct = predicted == correct
            
            return {
                'is_correct': is_correct,
                'predicted_answer': predicted,
                'correct_answer': correct,
                'evaluation_method': evaluation_method,
                'label': 'GOOD' if is_correct else 'BAD'
            }
            
        elif evaluation_method == 'text-generation':
            # For text generation, use exact match as baseline
            generated = response_data.get('generated_text', '').strip().lower()
            correct_answers = response_data.get('correct_answers', [])
            
            # Check if generated text matches any correct answer
            is_correct = any(generated == correct.strip().lower() for correct in correct_answers)
            
            return {
                'is_correct': is_correct,
                'generated_text': generated,
                'correct_answers': correct_answers,
                'evaluation_method': evaluation_method,
                'label': 'GOOD' if is_correct else 'BAD'
            }
            
        elif evaluation_method == 'perplexity':
            # For perplexity, we need to determine good/bad based on threshold
            # For now, use median as threshold (this could be improved)
            perplexity = response_data.get('perplexity', float('inf'))
            threshold = 50.0  # Simple threshold - could be made benchmark-specific
            is_correct = perplexity < threshold
            
            return {
                'is_correct': is_correct,
                'perplexity': perplexity,
                'threshold': threshold,
                'evaluation_method': evaluation_method,
                'label': 'GOOD' if is_correct else 'BAD'
            }
            
        else:
            return {
                'is_correct': False,
                'evaluation_method': evaluation_method,
                'label': 'BAD',
                'error': 'Unknown evaluation method'
            }
    
    def generate_responses_for_benchmark(self, benchmark_name: str, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate responses for all test samples in a benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            limit: Limit number of test samples (None for all)
            
        Returns:
            Dictionary with responses and activations
        """
        print(f"\n🎯 Processing benchmark: {benchmark_name}")
        
        # Get evaluation method
        eval_method = self.evaluation_methods.get(benchmark_name)
        if not eval_method:
            print(f"   ❌ No evaluation method found for {benchmark_name}")
            return {}
        
        print(f"   📋 Evaluation method: {eval_method}")
        
        # Load benchmark data
        try:
            train_data, test_data = self.load_benchmark_data(benchmark_name)
        except Exception as e:
            print(f"   ❌ Failed to load benchmark data: {e}")
            return {}
        
        # Limit test samples if specified
        if limit and limit < len(test_data):
            test_data = test_data[:limit]
            print(f"   📊 Limited to {limit} test samples")
        
        # Generate responses and collect activations
        responses = []
        all_activations = {layer: [] for layer in range(self.num_layers)}
        
        print(f"   🔧 Generating responses for {len(test_data)} test samples...")
        
        for i, sample in enumerate(test_data):
            if i % 10 == 0:
                print(f"      Progress: {i+1}/{len(test_data)}")
            
            try:
                # Generate response based on evaluation method
                if eval_method == 'log-likelihoods':
                    response = self.generate_log_likelihood_response(sample, benchmark_name)
                elif eval_method == 'text-generation':
                    response = self.generate_text_generation_response(sample, benchmark_name)
                elif eval_method == 'perplexity':
                    response = self.generate_perplexity_response(sample, benchmark_name)
                else:
                    print(f"      ⚠️ Unknown evaluation method: {eval_method}")
                    continue
                
                # Compute evaluation result
                evaluation_result = self.compute_evaluation_result(response)
                response['evaluation_result'] = evaluation_result
                
                # Extract activations from all layers using existing Model methods
                activation_text = response.get('activation_text', '')
                if activation_text:
                    print(f"      🧠 Extracting activations for sample {i+1} from text: '{activation_text[:50]}...'")
                    sample_activations = self.extract_activations_from_all_layers(activation_text)
                    
                    # Store activations for stacking
                    for layer_idx, activation in sample_activations.items():
                        if activation is not None:
                            all_activations[layer_idx].append(activation)
                    
                    # Add activations to response
                    response['activations'] = sample_activations
                else:
                    print(f"      ⚠️ No activation text found for sample {i+1}")
                    response['activations'] = {layer: None for layer in range(self.num_layers)}
                
                response['sample_index'] = i
                responses.append(response)
                
            except Exception as e:
                print(f"      ❌ Failed to process sample {i}: {e}")
                continue
        
        # Stack activations by layer
        stacked_activations = {}
        for layer_idx in range(self.num_layers):
            valid_activations = [act for act in all_activations[layer_idx] if act is not None]
            if valid_activations:
                stacked_activations[layer_idx] = torch.cat(valid_activations, dim=0)
            else:
                stacked_activations[layer_idx] = None
        
        result = {
            'benchmark_name': benchmark_name,
            'evaluation_method': eval_method,
            'num_test_samples': len(test_data),
            'num_successful_responses': len(responses),
            'responses': responses,
            'stacked_activations': stacked_activations,
            'model_name': self.model_name,
            'num_layers': self.num_layers
        }
        
        print(f"   ✅ Generated {len(responses)}/{len(test_data)} responses successfully")
        
        return result
    
    def save_benchmark_results(self, benchmark_name: str, results: Dict[str, Any]):
        """Save results for a benchmark."""
        if not results:
            return
        
        # Create benchmark-specific directory
        benchmark_dir = self.output_dir / benchmark_name
        benchmark_dir.mkdir(exist_ok=True)
        
        # Save responses (without activations for size)
        responses_only = {
            'benchmark_name': results['benchmark_name'],
            'evaluation_method': results['evaluation_method'],
            'num_test_samples': results['num_test_samples'],
            'num_successful_responses': results['num_successful_responses'],
            'model_name': results['model_name'],
            'responses': [{k: v for k, v in r.items() if k != 'activations'} for r in results['responses']]
        }
        
        responses_file = benchmark_dir / "responses.json"
        with open(responses_file, 'w') as f:
            json.dump(responses_only, f, indent=2)
        
        # Save activations by layer
        activations_dir = benchmark_dir / "activations"
        activations_dir.mkdir(exist_ok=True)
        
        for layer_idx, activations in results['stacked_activations'].items():
            if activations is not None:
                layer_file = activations_dir / f"layer_{layer_idx}.pt"
                torch.save(activations, layer_file)
        
        print(f"   💾 Saved results to {benchmark_dir}")
    
    def generate_all_testing_responses(self, 
                                     benchmarks: Optional[List[str]] = None,
                                     limit_per_benchmark: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate testing responses for all benchmarks.
        
        Args:
            benchmarks: List of benchmark names (None for all available)
            limit_per_benchmark: Limit test samples per benchmark (None for all)
            
        Returns:
            Dictionary with overall results
        """
        # Get available benchmarks
        available_benchmarks = self.get_available_benchmarks()
        
        if not available_benchmarks:
            print(f"❌ No benchmarks found with evaluation methods")
            return {}
        
        benchmarks = benchmarks or available_benchmarks
        benchmarks = [b for b in benchmarks if b in available_benchmarks]
        
        if not benchmarks:
            print(f"❌ None of the specified benchmarks are available")
            return {}
        
        print(f"\n🏗️ Generating testing responses")
        print(f"   Available benchmarks: {len(available_benchmarks)}")
        print(f"   Processing benchmarks: {len(benchmarks)}")
        if limit_per_benchmark:
            print(f"   Limit per benchmark: {limit_per_benchmark} test samples")
        
        overall_results = {
            'benchmarks_processed': [],
            'total_responses': 0,
            'total_samples': 0,
            'benchmark_results': {}
        }
        
        start_time = time.time()
        
        for i, benchmark in enumerate(benchmarks):
            print(f"\n{'='*60}")
            print(f"Processing benchmark {i+1}/{len(benchmarks)}: {benchmark}")
            print(f"{'='*60}")
            
            try:
                # Generate responses for this benchmark
                results = self.generate_responses_for_benchmark(benchmark, limit_per_benchmark)
                
                if results:
                    # Save results
                    self.save_benchmark_results(benchmark, results)
                    
                    # Update overall results
                    overall_results['benchmarks_processed'].append(benchmark)
                    overall_results['total_responses'] += results['num_successful_responses']
                    overall_results['total_samples'] += results['num_test_samples']
                    overall_results['benchmark_results'][benchmark] = {
                        'num_samples': results['num_test_samples'],
                        'num_responses': results['num_successful_responses'],
                        'evaluation_method': results['evaluation_method']
                    }
                    
                    print(f"   ✅ Benchmark {benchmark} completed successfully")
                else:
                    print(f"   ❌ Benchmark {benchmark} failed to generate any responses")
                    
            except Exception as e:
                print(f"   ❌ Error processing benchmark {benchmark}: {e}")
                continue
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📊 Benchmarks processed: {len(overall_results['benchmarks_processed'])}/{len(benchmarks)}")
        print(f"🔢 Total responses: {overall_results['total_responses']}")
        print(f"📝 Total samples: {overall_results['total_samples']}")
        
        return overall_results


def main():
    """Main function to run testing response generation."""
    parser = argparse.ArgumentParser(description='Generate testing responses and save layer activations')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name to use for generating responses')
    parser.add_argument('--benchmarks', nargs='+',
                       help='Specific benchmarks to process (default: all available)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit test samples per benchmark (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save results (default: wisent_guard/core/classifiers/testing_responses/)')
    
    args = parser.parse_args()
    
    print(f"🧪 Testing Response Generation")
    print(f"{'='*60}")
    
    # Initialize generator
    generator = TestingResponseGenerator(args.model, args.output_dir)
    
    # Show available benchmarks
    available = generator.get_available_benchmarks()
    print(f"📊 Available benchmarks: {len(available)}")
    for benchmark in available[:10]:  # Show first 10
        method = generator.evaluation_methods.get(benchmark, 'unknown')
        print(f"   • {benchmark} ({method})")
    if len(available) > 10:
        print(f"   ... and {len(available) - 10} more")
    
    # Generate responses
    try:
        results = generator.generate_all_testing_responses(
            benchmarks=args.benchmarks,
            limit_per_benchmark=args.limit
        )
        
        if results and results['total_responses'] > 0:
            print(f"\n🎉 SUCCESS! Generated {results['total_responses']} testing responses!")
            print(f"   📁 Saved to: {generator.output_dir}")
            print(f"   📊 Covering {len(results['benchmarks_processed'])} benchmarks")
            print(f"   🧠 Activations saved for all {generator.num_layers} layers")
        else:
            print(f"\n⚠️ No responses were generated successfully")
        
    except KeyboardInterrupt:
        print(f"\n❌ Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
