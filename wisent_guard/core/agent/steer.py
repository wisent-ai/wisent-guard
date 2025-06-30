"""
Steering module for autonomous agent response improvement.

This module handles:
- Response improvement strategies
- Steering vector generation
- Regeneration with improved prompts
- Training data creation for corrections
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Awaitable
from .diagnose import AnalysisResult


@dataclass
class ImprovementResult:
    """Result of self-improvement attempt."""
    original_response: str
    improved_response: str
    improvement_method: str
    success: bool
    improvement_score: float


class ResponseSteering:
    """Handles response improvement and steering for autonomous agents."""
    
    def __init__(self, generate_response_func: Callable[[str], Awaitable[str]], 
                 analyze_response_func: Callable[[str, str], Awaitable[AnalysisResult]]):
        """
        Initialize the steering system.
        
        Args:
            generate_response_func: Async function to generate new responses
            analyze_response_func: Async function to analyze responses
        """
        self.generate_response = generate_response_func
        self.analyze_response = analyze_response_func
    
    async def improve_response(self, prompt: str, response: str, analysis: AnalysisResult) -> ImprovementResult:
        """Attempt to improve the response."""
        # Decide improvement method based on issues
        method = self.choose_improvement_method(analysis.issues_found)
        
        if method == "regenerate":
            return await self.improve_by_regeneration(prompt, response, analysis)
        elif method == "steering":
            return await self.improve_by_steering(prompt, response, analysis)
        else:
            raise ValueError(f"Unknown improvement method: {method}")
    
    def choose_improvement_method(self, issues: List[str]) -> str:
        """Choose the best improvement method for the issues."""
        if any(issue in ["scientific_myth", "factual_error_population"] for issue in issues):
            return "steering"  # Use steering for factual issues
        elif "excessive_repetition" in issues:
            return "regenerate"  # Regenerate for repetition
        else:
            raise ValueError(f"No improvement method available for issues: {issues}")
    
    async def improve_by_regeneration(self, prompt: str, response: str, analysis: AnalysisResult) -> ImprovementResult:
        """Improve by regenerating with modified prompt."""
        # Create improved prompt
        improved_prompt = f"{prompt}\n\nPlease ensure your response is factually accurate and avoids repetition."
        
        # Generate new response
        new_response = await self.generate_response(improved_prompt)
        
        # Assess improvement
        new_analysis = await self.analyze_response(new_response, prompt)
        improvement_score = max(0, new_analysis.quality_score - analysis.quality_score)
        
        # Check if issues were resolved
        original_issues = set(analysis.issues_found)
        new_issues = set(new_analysis.issues_found)
        issues_resolved = len(original_issues - new_issues)
        issues_added = len(new_issues - original_issues)
        
        # Success if issues were resolved OR quality improved significantly
        issue_resolution_success = issues_resolved > issues_added
        quality_improvement_success = improvement_score > 0.05
        overall_success = issue_resolution_success or quality_improvement_success
        
        # Success metrics (can be enabled for debugging)
        if False:  # Set to True for detailed debugging
            print(f"   🔧 Regeneration debug:")
            print(f"      Original quality: {analysis.quality_score:.3f}")
            print(f"      New quality: {new_analysis.quality_score:.3f}")
            print(f"      Improvement score: {improvement_score:.3f}")
            print(f"      Original issues: {original_issues}")
            print(f"      New issues: {new_issues}")
            print(f"      Issues resolved: {issues_resolved}")
            print(f"      Issues added: {issues_added}")
            print(f"      Issue resolution success: {issue_resolution_success}")
            print(f"      Quality improvement success: {quality_improvement_success}")
            print(f"      Overall success: {overall_success}")
        
        return ImprovementResult(
            original_response=response,
            improved_response=new_response,
            improvement_method="regeneration",
            success=overall_success,
            improvement_score=improvement_score
        )
    
    async def improve_by_steering(self, prompt: str, response: str, analysis: AnalysisResult) -> ImprovementResult:
        """Improve using steering vectors."""
        # Create training data for steering
        training_data = self.create_steering_training_data(analysis.issues_found)
        
        # For now, use a sophisticated prompt-based approach instead of actual steering
        # This mimics the effect of steering by using the training data to create better prompts
        correction_examples = []
        for pair in training_data:
            correction_examples.append(f"Wrong: {pair['harmful']}\nCorrect: {pair['harmless']}")
        
        corrections_text = "\n\n".join(correction_examples)
        
        # Create improved prompt with correction examples
        improved_prompt = f"""Based on these correction examples:
{corrections_text}

Now please respond to this prompt with factual accuracy:
{prompt}

Ensure your response avoids the types of errors shown in the correction examples above."""
        
        new_response = await self.generate_response(improved_prompt)
        
        # Assess improvement
        new_analysis = await self.analyze_response(new_response, prompt)
        improvement_score = max(0, new_analysis.quality_score - analysis.quality_score)
        
        # Check if issues were resolved
        original_issues = set(analysis.issues_found)
        new_issues = set(new_analysis.issues_found)
        issues_resolved = len(original_issues - new_issues)
        issues_added = len(new_issues - original_issues)
        
        # Success if issues were resolved OR quality improved significantly
        issue_resolution_success = issues_resolved > issues_added
        quality_improvement_success = improvement_score > 0.05
        overall_success = issue_resolution_success or quality_improvement_success
        
        # Success metrics (can be enabled for debugging)
        if False:  # Set to True for detailed debugging
            print(f"   🔧 Steering debug:")
            print(f"      Original quality: {analysis.quality_score:.3f}")
            print(f"      New quality: {new_analysis.quality_score:.3f}")
            print(f"      Improvement score: {improvement_score:.3f}")
            print(f"      Original issues: {original_issues}")
            print(f"      New issues: {new_issues}")
            print(f"      Issues resolved: {issues_resolved}")
            print(f"      Issues added: {issues_added}")
            print(f"      Issue resolution success: {issue_resolution_success}")
            print(f"      Quality improvement success: {quality_improvement_success}")
            print(f"      Overall success: {overall_success}")
        
        return ImprovementResult(
            original_response=response,
            improved_response=new_response,
            improvement_method="steering",
            success=overall_success,
            improvement_score=improvement_score
        )
    
    def create_steering_training_data(self, issues: List[str]) -> List[Dict[str, str]]:
        """Create dynamic steering training data based on detected issues."""
        from ..contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator
        
        # Generate synthetic training data for the detected issues
        try:
            generator = SyntheticContrastivePairGenerator(self.model)
            
            training_pairs = []
            for issue in issues:
                # Create trait description for the issue
                trait_descriptions = {
                    "scientific_myth": "providing scientifically accurate information",
                    "factual_error_population": "giving correct population statistics",
                    "quality": "providing clear, accurate responses",
                    "harmful": "giving safe, helpful advice",
                    "bias": "responding with fairness and avoiding stereotypes",
                    "coherence": "maintaining logical communication"
                }
                
                trait_description = trait_descriptions.get(
                    issue, 
                    f"avoiding {issue} issues in responses"
                )
                
                # Generate pairs for each issue type
                synthetic_pairs = generator.generate_contrastive_pair_set(
                    trait_description=trait_description,
                    num_pairs=5,  # Generate 5 pairs per issue
                    name=f"steering_{issue}"
                )
                
                for pair in synthetic_pairs.pairs:
                    training_pairs.append({
                        "harmful": pair.negative_response,
                        "harmless": pair.positive_response
                    })
            
            if not training_pairs:
                raise ValueError(f"Could not generate training data for issues: {issues}")
            
            return training_pairs
            
        except Exception as e:
            raise ValueError(f"Failed to generate training data for issues {issues}: {e}")
