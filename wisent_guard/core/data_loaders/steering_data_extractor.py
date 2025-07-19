"""
Steering data extractor that converts LiveCodeBench problems to contrastive pairs for steering vector training.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .livecodebench_loader import LiveCodeBenchProblem

logger = logging.getLogger(__name__)


@dataclass
class ContrastivePair:
    """Represents a contrastive pair for steering vector training."""
    
    positive_prompt: str
    negative_prompt: str
    problem_id: str
    metadata: Dict[str, Any]


class SteeringDataExtractor:
    """Extracts contrastive pairs from LiveCodeBench problems for steering vector training."""
    
    def __init__(self):
        self.pair_generation_strategies = {
            "correct_vs_incorrect": self._generate_correct_vs_incorrect,
            "complete_vs_incomplete": self._generate_complete_vs_incomplete,
            "efficient_vs_inefficient": self._generate_efficient_vs_inefficient,
            "readable_vs_unreadable": self._generate_readable_vs_unreadable
        }
    
    def extract_contrastive_pairs(
        self,
        problems: List[LiveCodeBenchProblem],
        strategy: str = "correct_vs_incorrect",
        pairs_per_problem: int = 1
    ) -> List[ContrastivePair]:
        """
        Extract contrastive pairs from LiveCodeBench problems.
        
        Args:
            problems: List of LiveCodeBench problems
            strategy: Strategy for generating contrastive pairs
            pairs_per_problem: Number of pairs to generate per problem
            
        Returns:
            List of ContrastivePair objects
        """
        if strategy not in self.pair_generation_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.pair_generation_strategies.keys())}")
        
        logger.info(f"Extracting contrastive pairs using strategy: {strategy}")
        generator = self.pair_generation_strategies[strategy]
        
        all_pairs = []
        for problem in problems:
            try:
                pairs = generator(problem, pairs_per_problem)
                all_pairs.extend(pairs)
            except Exception as e:
                logger.warning(f"Failed to generate pairs for problem {problem.task_id}: {e}")
                continue
        
        logger.info(f"Generated {len(all_pairs)} contrastive pairs from {len(problems)} problems")
        return all_pairs
    
    def _generate_correct_vs_incorrect(self, problem: LiveCodeBenchProblem, num_pairs: int) -> List[ContrastivePair]:
        """Generate correct vs incorrect code pairs."""
        pairs = []
        
        # Base prompt template
        base_prompt = f"""Problem: {problem.question_title}
Description: {problem.question_content}

Starter code:
{problem.starter_code}

Task: Complete the function to solve this problem."""
        
        # Positive prompt (correct approach)
        positive_prompt = f"""{base_prompt}

Write a correct and efficient solution:"""
        
        # Negative prompts (incorrect approaches)
        negative_variants = [
            f"""{base_prompt}

Write a solution that looks plausible but has subtle bugs:""",
            f"""{base_prompt}

Write a solution that fails on edge cases:""",
            f"""{base_prompt}

Write a solution with logical errors:""",
            f"""{base_prompt}

Write a solution with off-by-one errors:"""
        ]
        
        for i in range(min(num_pairs, len(negative_variants))):
            pair = ContrastivePair(
                positive_prompt=positive_prompt,
                negative_prompt=negative_variants[i],
                problem_id=problem.task_id,
                metadata={
                    "strategy": "correct_vs_incorrect",
                    "difficulty": problem.difficulty,
                    "platform": problem.platform,
                    "variant": i,
                    "problem_title": problem.question_title
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def _generate_complete_vs_incomplete(self, problem: LiveCodeBenchProblem, num_pairs: int) -> List[ContrastivePair]:
        """Generate complete vs incomplete code pairs."""
        pairs = []
        
        base_prompt = f"""Problem: {problem.question_title}
Description: {problem.question_content}

Starter code:
{problem.starter_code}"""
        
        # Positive prompt (complete solution)
        positive_prompt = f"""{base_prompt}

Write a complete, working solution:"""
        
        # Negative prompts (incomplete solutions)
        negative_variants = [
            f"""{base_prompt}

Write a partial solution that's missing key logic:""",
            f"""{base_prompt}

Write a solution that handles only the simple cases:""",
            f"""{base_prompt}

Write a solution that's missing error handling:"""
        ]
        
        for i in range(min(num_pairs, len(negative_variants))):
            pair = ContrastivePair(
                positive_prompt=positive_prompt,
                negative_prompt=negative_variants[i],
                problem_id=problem.task_id,
                metadata={
                    "strategy": "complete_vs_incomplete",
                    "difficulty": problem.difficulty,
                    "platform": problem.platform,
                    "variant": i,
                    "problem_title": problem.question_title
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def _generate_efficient_vs_inefficient(self, problem: LiveCodeBenchProblem, num_pairs: int) -> List[ContrastivePair]:
        """Generate efficient vs inefficient code pairs."""
        pairs = []
        
        base_prompt = f"""Problem: {problem.question_title}
Description: {problem.question_content}

Starter code:
{problem.starter_code}"""
        
        # Positive prompt (efficient solution)
        positive_prompt = f"""{base_prompt}

Write an efficient solution with optimal time complexity:"""
        
        # Negative prompts (inefficient solutions)
        negative_variants = [
            f"""{base_prompt}

Write a brute force solution with poor time complexity:""",
            f"""{base_prompt}

Write a solution that uses excessive memory:""",
            f"""{base_prompt}

Write a solution with unnecessary nested loops:"""
        ]
        
        for i in range(min(num_pairs, len(negative_variants))):
            pair = ContrastivePair(
                positive_prompt=positive_prompt,
                negative_prompt=negative_variants[i],
                problem_id=problem.task_id,
                metadata={
                    "strategy": "efficient_vs_inefficient",
                    "difficulty": problem.difficulty,
                    "platform": problem.platform,
                    "variant": i,
                    "problem_title": problem.question_title
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def _generate_readable_vs_unreadable(self, problem: LiveCodeBenchProblem, num_pairs: int) -> List[ContrastivePair]:
        """Generate readable vs unreadable code pairs."""
        pairs = []
        
        base_prompt = f"""Problem: {problem.question_title}
Description: {problem.question_content}

Starter code:
{problem.starter_code}"""
        
        # Positive prompt (readable solution)
        positive_prompt = f"""{base_prompt}

Write a clean, readable solution with clear variable names and comments:"""
        
        # Negative prompts (unreadable solutions)
        negative_variants = [
            f"""{base_prompt}

Write a solution with cryptic variable names and no comments:""",
            f"""{base_prompt}

Write a solution with poor formatting and structure:""",
            f"""{base_prompt}

Write a solution that's overly complex and hard to understand:"""
        ]
        
        for i in range(min(num_pairs, len(negative_variants))):
            pair = ContrastivePair(
                positive_prompt=positive_prompt,
                negative_prompt=negative_variants[i],
                problem_id=problem.task_id,
                metadata={
                    "strategy": "readable_vs_unreadable",
                    "difficulty": problem.difficulty,
                    "platform": problem.platform,
                    "variant": i,
                    "problem_title": problem.question_title
                }
            )
            pairs.append(pair)
        
        return pairs
    
    def get_strategy_info(self, strategy: str) -> Dict[str, Any]:
        """Get information about a specific strategy."""
        strategy_descriptions = {
            "correct_vs_incorrect": "Generates pairs contrasting correct vs incorrect solutions",
            "complete_vs_incomplete": "Generates pairs contrasting complete vs incomplete solutions",
            "efficient_vs_inefficient": "Generates pairs contrasting efficient vs inefficient solutions",
            "readable_vs_unreadable": "Generates pairs contrasting readable vs unreadable solutions"
        }
        
        return {
            "name": strategy,
            "description": strategy_descriptions.get(strategy, "Unknown strategy"),
            "available": strategy in self.pair_generation_strategies
        }
    
    def list_available_strategies(self) -> List[str]:
        """List all available pair generation strategies."""
        return list(self.pair_generation_strategies.keys())