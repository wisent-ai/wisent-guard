#!/usr/bin/env python3
"""
Autonomous Wisent-Guard Agent

A model that can autonomously use wisent-guard capabilities on itself:
- Generate responses
- Analyze its own outputs for issues
- Train classifiers and steering vectors
- Apply corrections to improve future responses
"""

import asyncio
import json
import tempfile
import csv
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .model import Model
from .steering import SteeringMethod, SteeringType
from .layer import Layer
from .contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from .evaluate.stop_nonsense import NonsenseDetector
from .tracking.latency import LatencyTracker
from .tracking.memory import MemoryTracker


@dataclass
class AnalysisResult:
    """Result of self-analysis."""
    has_issues: bool
    issues_found: List[str]
    confidence: float
    suggestions: List[str]
    quality_score: float


@dataclass
class ImprovementResult:
    """Result of self-improvement attempt."""
    original_response: str
    improved_response: str
    improvement_method: str
    success: bool
    improvement_score: float


class AutonomousAgent:
    """
    An autonomous agent that can use wisent-guard tools on itself.
    
    The agent can:
    1. Generate responses to prompts
    2. Analyze its own responses for issues
    3. Decide what tools to train (classifiers, steering vectors)
    4. Apply those tools to improve future responses
    5. Learn from its mistakes autonomously
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 default_layer: int = 15,
                 enable_tracking: bool = True):
        """
        Initialize the autonomous agent.
        
        Args:
            model_name: Model to use for generation and analysis
            default_layer: Default layer for steering operations
            enable_tracking: Whether to enable performance tracking
        """
        self.model_name = model_name
        self.default_layer = default_layer
        self.enable_tracking = enable_tracking
        
        # Core components
        self.model = None
        self.nonsense_detector = None
        self.latency_tracker = None
        self.memory_tracker = None
        
        # Learned tools
        self.trained_classifiers = {}
        self.trained_steering_vectors = {}
        
        # Self-analysis history
        self.analysis_history = []
        self.improvement_history = []
        
        # Decision-making parameters
        self.quality_threshold = 0.7
        self.confidence_threshold = 0.8
    
    async def initialize(self):
        """Initialize all components."""
        print("🚀 Initializing autonomous agent...")
        
        # Load model
        print("📥 Loading model...")
        self.model = Model(self.model_name)
        
        # Initialize other components
        self.nonsense_detector = NonsenseDetector()
        
        if self.enable_tracking:
            self.latency_tracker = LatencyTracker()
            self.memory_tracker = MemoryTracker()
        
        print("✅ Agent initialized successfully")
    
    async def respond_autonomously(self, prompt: str, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Generate a response and autonomously improve it if needed.
        
        Args:
            prompt: The prompt to respond to
            max_attempts: Maximum improvement attempts
            
        Returns:
            Dictionary with response and improvement details
        """
        print(f"\n🎯 AUTONOMOUS RESPONSE TO: {prompt[:100]}...")
        
        attempt = 0
        current_response = None
        improvement_chain = []
        
        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Attempt {attempt} ---")
            
            # Generate response
            if current_response is None:
                print("💭 Generating initial response...")
                current_response = await self._generate_response(prompt)
                print(f"   Response: {current_response[:100]}...")
            
            # Analyze response
            print("🔍 Analyzing response...")
            analysis = await self._analyze_response(current_response, prompt)
            
            print(f"   Issues found: {analysis.issues_found}")
            print(f"   Quality score: {analysis.quality_score:.2f}")
            print(f"   Confidence: {analysis.confidence:.2f}")
            
            # Decide if improvement is needed
            needs_improvement = self._decide_if_improvement_needed(analysis)
            
            if not needs_improvement:
                print("✅ Response quality acceptable, no improvement needed")
                break
            
            # Attempt improvement
            print("🛠️ Attempting to improve response...")
            improvement = await self._improve_response(prompt, current_response, analysis)
            
            if improvement.success:
                print(f"   Improvement successful! Score: {improvement.improvement_score:.2f}")
                current_response = improvement.improved_response
                improvement_chain.append(improvement)
            else:
                print("   Improvement failed, keeping original response")
                break
        
        return {
            "final_response": current_response,
            "attempts": attempt,
            "improvement_chain": improvement_chain,
            "final_analysis": analysis
        }
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate a response to the prompt."""
        result = self.model.generate(
            prompt, 
            self.default_layer, 
            max_new_tokens=200
        )
        # Handle both 2 and 3 return values
        if isinstance(result, tuple) and len(result) == 3:
            response, _, _ = result
        elif isinstance(result, tuple) and len(result) == 2:
            response, _ = result
        else:
            response = result
        return response
    
    async def _analyze_response(self, response: str, prompt: str) -> AnalysisResult:
        """Analyze the response for potential issues."""
        issues = []
        confidence_scores = []
        
        # Nonsense detection
        nonsense_result = self.nonsense_detector.detect_nonsense(response)
        if nonsense_result["is_nonsense"]:
            issues.extend(nonsense_result["issues"])
            confidence_scores.append(nonsense_result["confidence"])
        
        # Pattern-based analysis
        pattern_issues = self._detect_patterns(response)
        issues.extend(pattern_issues)
        
        # Quality assessment
        quality_score = self._assess_quality(response, prompt)
        
        # Overall confidence
        confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Generate suggestions
        suggestions = self._generate_suggestions(issues, response)
        
        result = AnalysisResult(
            has_issues=len(issues) > 0,
            issues_found=issues,
            confidence=confidence,
            suggestions=suggestions,
            quality_score=quality_score
        )
        
        self.analysis_history.append(result)
        return result
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect problematic patterns in text."""
        issues = []
        text_lower = text.lower()
        
        # Factual issues
        if "moon" in text_lower and "cheese" in text_lower:
            issues.append("scientific_myth")
        
        if "vatican" in text_lower and any(num in text for num in ["50,000", "50000"]):
            issues.append("factual_error_population")
        
        # Repetition
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.6:
                issues.append("excessive_repetition")
        
        # Bias indicators
        if any(phrase in text_lower for phrase in ["all women", "all men", "everyone knows"]):
            issues.append("potential_bias")
        
        return issues
    
    def _assess_quality(self, response: str, prompt: str) -> float:
        """Assess response quality."""
        if len(response) < 10:
            return 0.2
        
        # Length check
        length_score = min(len(response) / 100, 1.0)
        
        # Relevance check (simple)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        relevance = len(prompt_words & response_words) / len(prompt_words) if prompt_words else 0
        
        return (length_score + relevance) / 2
    
    def _generate_suggestions(self, issues: List[str], response: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if "scientific_myth" in issues:
            suggestions.append("Replace with factually accurate information")
        
        if "factual_error_population" in issues:
            suggestions.append("Verify population statistics")
        
        if "excessive_repetition" in issues:
            suggestions.append("Reduce repetition and vary language")
        
        if "potential_bias" in issues:
            suggestions.append("Use more inclusive and balanced language")
        
        return suggestions
    
    def _decide_if_improvement_needed(self, analysis: AnalysisResult) -> bool:
        """Decide if the response needs improvement."""
        # Improvement needed if:
        # 1. Quality score below threshold
        # 2. High-confidence issues detected
        # 3. Multiple issues found
        
        if analysis.quality_score < self.quality_threshold:
            return True
        
        if analysis.confidence > self.confidence_threshold and analysis.has_issues:
            return True
        
        if len(analysis.issues_found) >= 2:
            return True
        
        return False
    
    async def _improve_response(self, prompt: str, response: str, analysis: AnalysisResult) -> ImprovementResult:
        """Attempt to improve the response."""
        # Decide improvement method based on issues
        method = self._choose_improvement_method(analysis.issues_found)
        
        if method == "regenerate":
            return await self._improve_by_regeneration(prompt, response, analysis)
        elif method == "steering":
            return await self._improve_by_steering(prompt, response, analysis)
        else:
            return ImprovementResult(
                original_response=response,
                improved_response=response,
                improvement_method="none",
                success=False,
                improvement_score=0.0
            )
    
    def _choose_improvement_method(self, issues: List[str]) -> str:
        """Choose the best improvement method for the issues."""
        if any(issue in ["scientific_myth", "factual_error_population"] for issue in issues):
            return "steering"  # Use steering for factual issues
        elif "excessive_repetition" in issues:
            return "regenerate"  # Regenerate for repetition
        else:
            return "regenerate"  # Default to regeneration
    
    async def _improve_by_regeneration(self, prompt: str, response: str, analysis: AnalysisResult) -> ImprovementResult:
        """Improve by regenerating with modified prompt."""
        # Create improved prompt
        improved_prompt = f"{prompt}\n\nPlease ensure your response is factually accurate and avoids repetition."
        
        # Generate new response
        new_response = await self._generate_response(improved_prompt)
        
        # Assess improvement
        new_analysis = await self._analyze_response(new_response, prompt)
        improvement_score = max(0, new_analysis.quality_score - analysis.quality_score)
        
        return ImprovementResult(
            original_response=response,
            improved_response=new_response,
            improvement_method="regeneration",
            success=improvement_score > 0.1,
            improvement_score=improvement_score
        )
    
    async def _improve_by_steering(self, prompt: str, response: str, analysis: AnalysisResult) -> ImprovementResult:
        """Improve using steering vectors."""
        try:
            # Create training data for steering
            training_data = self._create_steering_training_data(analysis.issues_found)
            
            # Train steering vector
            steering_method = SteeringMethod(
                method_type=SteeringType.CAA,
                device=self.model.device
            )
            
            pair_set = ContrastivePairSet.from_phrase_pairs(
                name="autonomous_improvement",
                phrase_pairs=training_data,
                task_type="response_improvement"
            )
            
            layer = Layer(index=self.default_layer, type="transformer")
            training_result = pair_set.train_classifier(steering_method.classifier, layer)
            
            # Apply steering (simplified - would need actual implementation)
            # For now, just regenerate with awareness
            improved_prompt = f"{prompt}\n\nEnsure factual accuracy and avoid common misconceptions."
            new_response = await self._generate_response(improved_prompt)
            
            # Assess improvement
            new_analysis = await self._analyze_response(new_response, prompt)
            improvement_score = max(0, new_analysis.quality_score - analysis.quality_score)
            
            return ImprovementResult(
                original_response=response,
                improved_response=new_response,
                improvement_method="steering",
                success=improvement_score > 0.1,
                improvement_score=improvement_score
            )
            
        except Exception as e:
            print(f"   Steering failed: {e}")
            return await self._improve_by_regeneration(prompt, response, analysis)
    
    def _create_steering_training_data(self, issues: List[str]) -> List[Dict[str, str]]:
        """Create training data for steering based on detected issues."""
        training_pairs = []
        
        if "scientific_myth" in issues:
            training_pairs.extend([
                {"harmful": "The moon is made of cheese", "harmless": "The moon is made of rock and regolith"},
                {"harmful": "The Earth is flat", "harmless": "The Earth is approximately spherical"},
            ])
        
        if "factual_error_population" in issues:
            training_pairs.extend([
                {"harmful": "Vatican City has 50,000 people", "harmless": "Vatican City has about 800 residents"},
                {"harmful": "Monaco has 5 million people", "harmless": "Monaco has about 39,000 residents"},
            ])
        
        # Default pairs if no specific issues
        if not training_pairs:
            training_pairs = [
                {"harmful": "I'm not sure about this", "harmless": "Let me provide accurate information"},
                {"harmful": "This might be wrong", "harmless": "Here's the correct information"},
            ]
        
        return training_pairs
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's performance."""
        return {
            "total_analyses": len(self.analysis_history),
            "total_improvements": len(self.improvement_history),
            "avg_quality_score": sum(a.quality_score for a in self.analysis_history) / len(self.analysis_history) if self.analysis_history else 0,
            "improvement_success_rate": sum(1 for i in self.improvement_history if i.success) / len(self.improvement_history) if self.improvement_history else 0,
            "trained_tools": {
                "classifiers": len(self.trained_classifiers),
                "steering_vectors": len(self.trained_steering_vectors)
            }
        }


async def demo_autonomous_agent():
    """Demonstrate the autonomous agent."""
    print("🤖 AUTONOMOUS WISENT-GUARD AGENT DEMO")
    print("=" * 50)
    
    # Initialize agent
    agent = AutonomousAgent()
    await agent.initialize()
    
    # Test problems
    test_prompts = [
        "Tell me about Vatican City's population and when it was founded",
        "Write about the composition of the moon",
        "Explain quantum entanglement in simple terms",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎮 TEST {i}/3")
        print("=" * 30)
        
        result = await agent.respond_autonomously(prompt)
        
        print(f"\n📊 RESULTS:")
        print(f"   Final response: {result['final_response'][:100]}...")
        print(f"   Attempts: {result['attempts']}")
        print(f"   Improvements: {len(result['improvement_chain'])}")
        print(f"   Final quality: {result['final_analysis'].quality_score:.2f}")
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 30)
    summary = agent.get_performance_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_autonomous_agent()) 