"""
Evaluation module for steering responses.
Evaluates steered responses on three criteria:
1. Difference from unsteered responses
2. Coherence of the response
3. Alignment with the target trait

Includes parameter optimization to find optimal steering strength.
"""

import torch
import numpy as np
from typing import List, Dict, Callable, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class SteeringEvaluator:
    """Evaluates steering effectiveness using the same model that generates responses."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def evaluate_response(
        self, 
        prompt: str,
        unsteered_response: str,
        steered_response: str,
        trait_description: str
    ) -> Dict[str, float]:
        """
        Evaluate a steered response on three criteria.
        
        Args:
            prompt: Original prompt
            unsteered_response: Response without steering
            steered_response: Response with steering applied
            trait_description: Description of the target trait
            
        Returns:
            Dictionary with scores for each criterion (0-10 scale)
        """
        scores = {}
        
        # 1. Evaluate difference from unsteered
        scores['difference'] = self._evaluate_difference(unsteered_response, steered_response)
        
        # 2. Evaluate coherence
        scores['coherence'] = self._evaluate_coherence(prompt, steered_response)
        
        # 3. Evaluate trait alignment
        scores['trait_alignment'] = self._evaluate_trait_alignment(
            prompt, steered_response, trait_description
        )
        
        # Overall score
        scores['overall'] = (scores['difference'] + scores['coherence'] + scores['trait_alignment']) / 3
        
        return scores
    
    def _evaluate_difference(self, unsteered: str, steered: str) -> float:
        """Evaluate how different the steered response is from unsteered."""
        if unsteered.strip() == steered.strip():
            return 0.0
        
        eval_prompt = f"""Rate how different these two responses are on a scale of 0-10 (0 = identical, 10 = completely different):

Response 1: {unsteered}

Response 2: {steered}

Provide only a number between 0 and 10:"""
        
        score = self._get_numeric_score(eval_prompt)
        return score
    
    def _evaluate_coherence(self, prompt: str, response: str) -> float:
        """Evaluate the coherence and quality of the response."""
        eval_prompt = f"""Rate the coherence and quality of this response on a scale of 0-10 (0 = incoherent/nonsensical, 10 = perfectly coherent and well-written):

Prompt: {prompt}

Response: {response}

Consider: grammatical correctness, logical flow, relevance to prompt, and overall readability.

Provide only a number between 0 and 10:"""
        
        score = self._get_numeric_score(eval_prompt)
        return score
    
    def _evaluate_trait_alignment(self, prompt: str, response: str, trait: str) -> float:
        """Evaluate how well the response demonstrates the target trait."""
        eval_prompt = f"""Rate how strongly this response demonstrates "{trait}" on a scale of 0-10 (0 = no trait present, 10 = perfectly demonstrates the trait):

Prompt: {prompt}

Response: {response}

Target trait: {trait}

Provide only a number between 0 and 10:"""
        
        score = self._get_numeric_score(eval_prompt)
        return score
    
    def _get_numeric_score(self, eval_prompt: str) -> float:
        """Get a numeric score from the model."""
        inputs = self.tokenizer(eval_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.1,  # Low temperature for consistent scoring
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract numeric score
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                # Clamp to 0-10 range
                return min(max(score, 0.0), 10.0)
        except:
            pass
        
        # Default to 5.0 if we can't parse a score
        return 5.0
    
    def evaluate_batch(
        self,
        results: List[Dict[str, str]],
        trait_description: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a batch of steering results.
        
        Args:
            results: List of dicts with 'prompt', 'unsteered', and steering method keys
            trait_description: Description of the target trait
            
        Returns:
            Dictionary mapping method names to average scores
        """
        method_scores = {}
        
        for result in results:
            prompt = result['prompt']
            unsteered = result['unsteered']
            
            for method, response in result.items():
                if method in ['prompt', 'unsteered']:
                    continue
                
                if method not in method_scores:
                    method_scores[method] = {
                        'difference': [],
                        'coherence': [],
                        'trait_alignment': [],
                        'overall': []
                    }
                
                scores = self.evaluate_response(
                    prompt, unsteered, response, trait_description
                )
                
                for criterion, score in scores.items():
                    method_scores[method][criterion].append(score)
        
        # Calculate averages
        avg_scores = {}
        for method, scores in method_scores.items():
            avg_scores[method] = {}
            for criterion, score_list in scores.items():
                avg_scores[method][criterion] = sum(score_list) / len(score_list) if score_list else 0.0
        
        return avg_scores
    
    def print_evaluation_summary(self, avg_scores: Dict[str, Dict[str, float]]):
        """Print a formatted summary of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Print header
        print(f"{'Method':<20} {'Difference':<12} {'Coherence':<12} {'Trait Align':<12} {'Overall':<12}")
        print("-"*60)
        
        # Sort methods by overall score
        sorted_methods = sorted(avg_scores.items(), key=lambda x: x[1]['overall'], reverse=True)
        
        for method, scores in sorted_methods:
            print(f"{method:<20} {scores['difference']:<12.1f} {scores['coherence']:<12.1f} "
                  f"{scores['trait_alignment']:<12.1f} {scores['overall']:<12.1f}")
        
        print("="*60)
    
    def optimize_steering_strength(
        self,
        prompt: str,
        unsteered_response: str,
        steering_method: any,
        generation_func: Callable,
        trait_description: str,
        strength_range: Tuple[float, float] = (0.1, 3.0),
        num_samples: int = 10,
        optimization_metric: str = 'overall'
    ) -> Dict[str, any]:
        """
        Find optimal steering strength by testing different values.
        
        Args:
            prompt: Original prompt
            unsteered_response: Response without steering
            steering_method: The steering method to optimize
            generation_func: Function that generates steered response given (prompt, method, strength)
            trait_description: Description of the target trait
            strength_range: Range of strength values to test (min, max)
            num_samples: Number of strength values to test
            optimization_metric: Which metric to optimize ('overall', 'trait_alignment', 'balanced')
            
        Returns:
            Dictionary with optimal strength and evaluation results
        """
        strengths = np.linspace(strength_range[0], strength_range[1], num_samples)
        results = []
        
        print(f"\nOptimizing steering strength for {optimization_metric}...")
        print(f"Testing {num_samples} values from {strength_range[0]} to {strength_range[1]}")
        
        for strength in strengths:
            # Generate steered response with this strength
            try:
                steered_response = generation_func(prompt, steering_method, float(strength))
                
                # Evaluate the response
                scores = self.evaluate_response(
                    prompt, unsteered_response, steered_response, trait_description
                )
                
                # Calculate optimization score based on metric
                if optimization_metric == 'overall':
                    opt_score = scores['overall']
                elif optimization_metric == 'trait_alignment':
                    opt_score = scores['trait_alignment']
                elif optimization_metric == 'balanced':
                    # Balance between trait alignment and coherence
                    opt_score = (scores['trait_alignment'] + scores['coherence']) / 2
                else:
                    opt_score = scores['overall']
                
                results.append({
                    'strength': float(strength),
                    'scores': scores,
                    'optimization_score': opt_score,
                    'response': steered_response
                })
                
                print(f"  Strength {strength:.2f}: {optimization_metric}={opt_score:.1f} "
                      f"(trait={scores['trait_alignment']:.1f}, coherence={scores['coherence']:.1f})")
                
            except Exception as e:
                print(f"  Strength {strength:.2f}: Error - {str(e)}")
                continue
        
        if not results:
            return None
        
        # Find optimal strength
        best_result = max(results, key=lambda x: x['optimization_score'])
        
        return {
            'optimal_strength': best_result['strength'],
            'optimal_scores': best_result['scores'],
            'optimal_response': best_result['response'],
            'all_results': results
        }
    
    def optimize_multiple_parameters(
        self,
        prompt: str,
        unsteered_response: str,
        steering_method: any,
        generation_func: Callable,
        trait_description: str,
        parameter_ranges: Dict[str, Tuple[float, float]],
        num_samples_per_param: int = 5,
        optimization_metric: str = 'overall'
    ) -> Dict[str, any]:
        """
        Optimize multiple parameters simultaneously (e.g., strength and temperature).
        
        Args:
            prompt: Original prompt
            unsteered_response: Response without steering
            steering_method: The steering method to optimize
            generation_func: Function that generates response given (prompt, method, **params)
            trait_description: Description of the target trait
            parameter_ranges: Dict mapping parameter names to (min, max) ranges
            num_samples_per_param: Number of values to test per parameter
            optimization_metric: Which metric to optimize
            
        Returns:
            Dictionary with optimal parameters and evaluation results
        """
        # Generate parameter grid
        param_values = {}
        for param_name, (min_val, max_val) in parameter_ranges.items():
            param_values[param_name] = np.linspace(min_val, max_val, num_samples_per_param)
        
        # Create all combinations
        from itertools import product
        param_combinations = list(product(*param_values.values()))
        param_names = list(parameter_ranges.keys())
        
        results = []
        best_score = -1
        best_params = None
        
        print(f"\nOptimizing {len(param_names)} parameters: {', '.join(param_names)}")
        print(f"Testing {len(param_combinations)} combinations...")
        
        for i, param_tuple in enumerate(param_combinations):
            params = dict(zip(param_names, param_tuple))
            
            try:
                # Generate steered response with these parameters
                steered_response = generation_func(prompt, steering_method, **params)
                
                # Evaluate
                scores = self.evaluate_response(
                    prompt, unsteered_response, steered_response, trait_description
                )
                
                # Calculate optimization score
                if optimization_metric == 'overall':
                    opt_score = scores['overall']
                elif optimization_metric == 'trait_alignment':
                    opt_score = scores['trait_alignment']
                elif optimization_metric == 'balanced':
                    opt_score = (scores['trait_alignment'] + scores['coherence']) / 2
                else:
                    opt_score = scores['overall']
                
                results.append({
                    'parameters': params.copy(),
                    'scores': scores,
                    'optimization_score': opt_score,
                    'response': steered_response
                })
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_params = params.copy()
                
                if i % 5 == 0:  # Print progress every 5 iterations
                    param_str = ', '.join([f"{k}={v:.2f}" for k, v in params.items()])
                    print(f"  [{i+1}/{len(param_combinations)}] {param_str}: score={opt_score:.1f}")
                    
            except Exception as e:
                print(f"  Error with params {params}: {str(e)}")
                continue
        
        if not results:
            return None
        
        # Find best result
        best_result = max(results, key=lambda x: x['optimization_score'])
        
        # Analyze parameter sensitivity
        sensitivity = self._analyze_parameter_sensitivity(results, param_names)
        
        return {
            'optimal_parameters': best_result['parameters'],
            'optimal_scores': best_result['scores'],
            'optimal_response': best_result['response'],
            'parameter_sensitivity': sensitivity,
            'all_results': results
        }
    
    def _analyze_parameter_sensitivity(self, results: List[Dict], param_names: List[str]) -> Dict[str, float]:
        """
        Analyze how sensitive the optimization score is to each parameter.
        
        Returns dict mapping parameter names to sensitivity scores (0-1).
        """
        sensitivity = {}
        
        for param_name in param_names:
            # Group results by this parameter value
            param_groups = {}
            for result in results:
                param_val = result['parameters'][param_name]
                if param_val not in param_groups:
                    param_groups[param_val] = []
                param_groups[param_val].append(result['optimization_score'])
            
            # Calculate variance across parameter values
            group_means = [np.mean(scores) for scores in param_groups.values()]
            if len(group_means) > 1:
                sensitivity[param_name] = np.std(group_means)
            else:
                sensitivity[param_name] = 0.0
        
        # Normalize to 0-1 range
        max_sensitivity = max(sensitivity.values()) if sensitivity else 1.0
        if max_sensitivity > 0:
            for param in sensitivity:
                sensitivity[param] /= max_sensitivity
        
        return sensitivity
    
    def plot_optimization_results(self, optimization_results: Dict[str, any], save_path: str = None):
        """
        Plot optimization results to visualize parameter effects.
        
        Args:
            optimization_results: Results from optimize_steering_strength or optimize_multiple_parameters
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        if 'all_results' not in optimization_results:
            return
        
        results = optimization_results['all_results']
        
        if 'strength' in results[0]:  # Single parameter optimization
            strengths = [r['strength'] for r in results]
            trait_scores = [r['scores']['trait_alignment'] for r in results]
            coherence_scores = [r['scores']['coherence'] for r in results]
            overall_scores = [r['scores']['overall'] for r in results]
            
            plt.figure(figsize=(10, 6))
            plt.plot(strengths, trait_scores, 'b-', label='Trait Alignment', marker='o')
            plt.plot(strengths, coherence_scores, 'g-', label='Coherence', marker='s')
            plt.plot(strengths, overall_scores, 'r-', label='Overall', marker='^', linewidth=2)
            
            # Mark optimal
            opt_strength = optimization_results['optimal_strength']
            opt_overall = optimization_results['optimal_scores']['overall']
            plt.plot(opt_strength, opt_overall, 'r*', markersize=15, label=f'Optimal ({opt_strength:.2f})')
            
            plt.xlabel('Steering Strength')
            plt.ylabel('Score (0-10)')
            plt.title('Steering Parameter Optimization')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:  # Multiple parameter optimization
            # For multiple parameters, show a heatmap or parameter importance
            if 'parameter_sensitivity' in optimization_results:
                sensitivity = optimization_results['parameter_sensitivity']
                
                plt.figure(figsize=(8, 6))
                params = list(sensitivity.keys())
                values = list(sensitivity.values())
                
                plt.bar(params, values)
                plt.xlabel('Parameter')
                plt.ylabel('Sensitivity (normalized)')
                plt.title('Parameter Sensitivity Analysis')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()