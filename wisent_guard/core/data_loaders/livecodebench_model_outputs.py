"""
LiveCodeBench model outputs loader for contrastive pair extraction.

This module loads pre-annotated model outputs from LiveCodeBench where each solution
is already marked as passing or failing the test cases.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)


class LiveCodeBenchModelOutputsLoader:
    """Loads pre-annotated model outputs from LiveCodeBench."""
    
    def __init__(self, outputs_path: Optional[str] = None):
        """
        Initialize the model outputs loader.
        
        Args:
            outputs_path: Path to model outputs directory or all_outputs.json file
        """
        if outputs_path is None:
            # Default to wisent_core location
            base_path = Path(__file__).parent.parent.parent.parent
            self.outputs_path = base_path / "wisent_core" / "code_generation_samples" / "all_outputs.json"
        else:
            self.outputs_path = Path(outputs_path)
            
        self.all_outputs = None
        self._load_outputs()
    
    def _load_outputs(self):
        """Load the model outputs from JSON file."""
        if not self.outputs_path.exists():
            raise FileNotFoundError(f"Model outputs file not found: {self.outputs_path}")
            
        with open(self.outputs_path, 'r') as f:
            self.all_outputs = json.load(f)
            
        logger.info(f"Loaded outputs from {len(self.all_outputs)} models")
        
    def get_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.all_outputs.keys())
    
    def get_problem_count(self) -> int:
        """Get number of problems in the dataset."""
        # Get from first model's data
        if self.all_outputs:
            first_model = next(iter(self.all_outputs.keys()))
            return len(self.all_outputs[first_model])
        return 0
    
    def get_contrastive_pairs_for_problem(self, problem_idx: int) -> List[Dict[str, Any]]:
        """
        Extract contrastive pairs for a specific problem.
        
        Args:
            problem_idx: Index of the problem (0-based)
            
        Returns:
            List of contrastive pairs with good/bad examples
        """
        pairs = []
        
        # Collect all passing and failing solutions across models
        passing_solutions = []
        failing_solutions = []
        
        for model_name, model_data in self.all_outputs.items():
            if problem_idx >= len(model_data):
                continue
                
            problem_data = model_data[problem_idx]
            code_list = problem_data.get("code_list", [])
            pass1_list = problem_data.get("pass1_list", [])
            metadata_list = problem_data.get("metadata_list", [])
            
            # Extract passing and failing solutions
            for i, (code, passed) in enumerate(zip(code_list, pass1_list)):
                solution_info = {
                    "model": model_name,
                    "code": code,
                    "passed": passed,
                    "metadata": metadata_list[i] if i < len(metadata_list) else None
                }
                
                if passed:
                    passing_solutions.append(solution_info)
                else:
                    failing_solutions.append(solution_info)
        
        # Create contrastive pairs by pairing passing with failing solutions
        # Use all combinations or limit to reasonable number
        max_pairs = min(10, len(passing_solutions) * len(failing_solutions))
        
        if passing_solutions and failing_solutions:
            # Randomly sample pairs if too many combinations
            if len(passing_solutions) * len(failing_solutions) > max_pairs:
                for _ in range(max_pairs):
                    good = random.choice(passing_solutions)
                    bad = random.choice(failing_solutions)
                    pairs.append({
                        "problem_idx": problem_idx,
                        "good_example": good,
                        "bad_example": bad
                    })
            else:
                # Use all combinations
                for good in passing_solutions:
                    for bad in failing_solutions:
                        pairs.append({
                            "problem_idx": problem_idx,
                            "good_example": good,
                            "bad_example": bad
                        })
        
        return pairs
    
    def get_all_contrastive_pairs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get contrastive pairs for all problems.
        
        Args:
            limit: Optional limit on number of problems to process
            
        Returns:
            List of all contrastive pairs
        """
        all_pairs = []
        num_problems = self.get_problem_count()
        
        if limit:
            num_problems = min(num_problems, limit)
            
        for idx in range(num_problems):
            pairs = self.get_contrastive_pairs_for_problem(idx)
            all_pairs.extend(pairs)
            
        logger.info(f"Extracted {len(all_pairs)} contrastive pairs from {num_problems} problems")
        return all_pairs
    
    def get_balanced_pairs(self, num_pairs: int) -> List[Dict[str, Any]]:
        """
        Get a balanced set of contrastive pairs.
        
        Args:
            num_pairs: Number of pairs to return
            
        Returns:
            Balanced list of contrastive pairs
        """
        all_pairs = self.get_all_contrastive_pairs()
        
        if len(all_pairs) <= num_pairs:
            return all_pairs
            
        # Randomly sample the requested number
        return random.sample(all_pairs, num_pairs)