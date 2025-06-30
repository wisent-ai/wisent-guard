#!/usr/bin/env python3
"""
Autonomous Wisent-Guard Agent

A model that can autonomously use wisent-guard capabilities on itself:
- Generate responses
- Analyze its own outputs for issues
- Auto-discover or create classifiers on demand
- Apply corrections to improve future responses
"""

import asyncio
import os
from typing import Dict, Any, Optional, List

from .model import Model
from .agent.diagnose import (
    ResponseDiagnostics, 
    AnalysisResult,
    ClassifierMarketplace,
    AgentClassifierDecisionSystem
)
from .agent.steer import ResponseSteering, ImprovementResult


class AutonomousAgent:
    """
    An autonomous agent that can generate responses, analyze them for issues,
    and improve them using activation-based steering and correction techniques.
    
    The agent now uses a marketplace-based system to intelligently select
    classifiers based on task analysis, with no hardcoded requirements.
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 default_layer: int = 15,
                 enable_tracking: bool = True):
        """
        Initialize the autonomous agent.
        
        Args:
            model_name: Name of the model to use
            default_layer: Default layer for generation
            enable_tracking: Whether to track improvement history
        """
        self.model_name = model_name
        self.model: Optional[Model] = None
        self.default_layer = default_layer
        self.enable_tracking = enable_tracking
        
        # New marketplace-based system
        self.marketplace: Optional[ClassifierMarketplace] = None
        self.decision_system: Optional[AgentClassifierDecisionSystem] = None
        self.diagnostics: Optional[ResponseDiagnostics] = None
        self.steering: Optional[ResponseSteering] = None
        
        # Tracking
        self.improvement_history: List[ImprovementResult] = []
        self.analysis_history: List[AnalysisResult] = []
        
        print(f"🤖 Autonomous Agent initialized with {model_name}")
        print(f"   🎯 Using marketplace-based classifier selection")
        
    async def initialize(self, 
                        classifier_search_paths: Optional[List[str]] = None,
                        quality_threshold: float = 0.3,
                        default_time_budget_minutes: float = 10.0):
        """
        Initialize the autonomous agent with intelligent classifier management.
        
        Args:
            classifier_search_paths: Paths to search for existing classifiers
            quality_threshold: Minimum quality threshold for existing classifiers
            default_time_budget_minutes: Default time budget for creating new classifiers
        """
        print("🚀 Initializing Autonomous Agent...")
        
        # Load model
        print("   📦 Loading model...")
        self.model = Model(self.model_name)
        
        # Initialize marketplace
        print("   🏪 Setting up classifier marketplace...")
        self.marketplace = ClassifierMarketplace(
            model=self.model,
            search_paths=classifier_search_paths
        )
        
        # Initialize decision system
        print("   🧠 Setting up intelligent decision system...")
        self.decision_system = AgentClassifierDecisionSystem(self.marketplace)
        
        # Store configuration
        self.quality_threshold = quality_threshold
        self.default_time_budget_minutes = default_time_budget_minutes
        
        # Show marketplace summary
        summary = self.marketplace.get_marketplace_summary()
        print(summary)
        
        print("   ✅ Autonomous Agent ready!")
    
    async def respond_autonomously(self, 
                                 prompt: str, 
                                 max_attempts: int = 3,
                                 quality_threshold: float = None,
                                 time_budget_minutes: float = None,
                                 max_classifiers: int = None) -> Dict[str, Any]:
        """
        Generate a response and autonomously improve it if needed.
        The agent will intelligently select classifiers based on the prompt.
        
        Args:
            prompt: The prompt to respond to
            max_attempts: Maximum improvement attempts
            quality_threshold: Quality threshold for classifiers (uses default if None)
            time_budget_minutes: Time budget for creating classifiers (uses default if None)
            max_classifiers: Maximum classifiers to use (None = no limit)
            
        Returns:
            Dictionary with response and improvement details
        """
        print(f"\n🎯 AUTONOMOUS RESPONSE TO: {prompt[:100]}...")
        
        # Use defaults if not specified
        quality_threshold = quality_threshold or self.quality_threshold
        time_budget_minutes = time_budget_minutes or self.default_time_budget_minutes
        
        # Step 1: Intelligent classifier selection based on the prompt
        print("\n🧠 Analyzing task and selecting classifiers...")
        classifier_configs = await self.decision_system.smart_classifier_selection(
            prompt=prompt,
            quality_threshold=quality_threshold,
            time_budget_minutes=time_budget_minutes,
            max_classifiers=max_classifiers
        )
        
        # Step 2: Initialize diagnostics and steering with selected classifiers
        if classifier_configs:
            print(f"   📊 Initializing diagnostics with {len(classifier_configs)} classifiers")
            self.diagnostics = ResponseDiagnostics(
                model=self.model,
                classifier_configs=classifier_configs
            )
            
            self.steering = ResponseSteering(
                model=self.model,
                diagnostics=self.diagnostics
            )
        else:
            print("   ⚠️ No classifiers selected - proceeding without advanced diagnostics")
            # Could fall back to basic text analysis or skip diagnostics
            return {"final_response": await self._generate_response(prompt),
                   "attempts": 1,
                   "improvement_chain": [],
                   "classifier_info": "No classifiers used"}
        
        # Step 3: Generate and improve response
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
            
            # Analyze response using selected classifiers
            print("🔍 Analyzing response...")
            analysis = await self.diagnostics.analyze_response(current_response, prompt)
            
            print(f"   Issues found: {analysis.issues_found}")
            print(f"   Quality score: {analysis.quality_score:.2f}")
            print(f"   Confidence: {analysis.confidence:.2f}")
            
            # Track analysis
            if self.enable_tracking:
                self.analysis_history.append(analysis)
            
            # Decide if improvement is needed
            needs_improvement = self._decide_if_improvement_needed(analysis)
            
            if not needs_improvement:
                print("✅ Response quality acceptable, no improvement needed")
                break
            
            # Attempt improvement
            print("🛠️ Attempting to improve response...")
            improvement = await self.steering.improve_response(prompt, current_response, analysis)
            
            if improvement.success:
                print(f"   Improvement successful! Score: {improvement.improvement_score:.2f}")
                current_response = improvement.improved_response
                improvement_chain.append(improvement)
                
                if self.enable_tracking:
                    self.improvement_history.append(improvement)
            else:
                print("   Improvement failed, keeping original response")
                break
        
        return {
            "final_response": current_response,
            "attempts": attempt,
            "improvement_chain": improvement_chain,
            "final_analysis": analysis,
            "classifier_info": {
                "count": len(classifier_configs),
                "types": [c.get("issue_type", "unknown") for c in classifier_configs],
                "decision_summary": self.decision_system.get_decision_summary()
            }
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
    
    def _decide_if_improvement_needed(self, analysis: AnalysisResult) -> bool:
        """Decide if the response needs improvement based on analysis."""
        # If significant issues were found with high confidence, improve
        if analysis.issues_found and analysis.confidence > 0.6:
            return True
        
        # If quality is low and we're confident about it, improve
        if analysis.quality_score < 0.5 and analysis.confidence > 0.5:
            return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's performance."""
        if not self.enable_tracking:
            return {"tracking_disabled": True}
        
        total_improvements = len(self.improvement_history)
        successful_improvements = sum(1 for imp in self.improvement_history if imp.success)
        success_rate = successful_improvements / total_improvements if total_improvements > 0 else 0
        
        return {
            "total_improvements_attempted": total_improvements,
            "successful_improvements": successful_improvements,
            "success_rate": success_rate,
            "total_analyses": len(self.analysis_history),
            "decision_summary": self.decision_system.get_decision_summary() if self.decision_system else "No decisions made"
        }
    
    def get_marketplace_info(self) -> str:
        """Get information about the classifier marketplace."""
        if not self.marketplace:
            return "Marketplace not initialized"
        return self.marketplace.get_marketplace_summary()

async def demo_autonomous_agent():
    """
    Demo function showing the new autonomous agent with intelligent classifier selection.
    """
    print("🎭 AUTONOMOUS AGENT DEMO - Intelligent Classifier Selection")
    print("=" * 60)
    
    # Initialize agent
    agent = AutonomousAgent()
    
    try:
        await agent.initialize(
            quality_threshold=0.3,
            default_time_budget_minutes=5.0  # Short budget for demo
        )
        
        # Test prompts that should trigger different classifier selections
        test_prompts = [
            "Tell me about the history of the moon landing",  # Should detect hallucination risk
            "What's the best way to lose weight quickly?",     # Should detect harmful/quality issues
            "Are men naturally better at math than women?",    # Should detect bias
            "Write a poem about spring flowers",               # Should detect quality needs
            "Explain quantum physics in simple terms"         # Should detect quality/coherence needs
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*20} Test {i} {'='*20}")
            
            result = await agent.respond_autonomously(
                prompt=prompt,
                max_attempts=2,
                time_budget_minutes=2.0  # Very short for demo
            )
            
            print(f"\n📋 RESULT SUMMARY:")
            print(f"   Final Response: {result['final_response'][:100]}...")
            print(f"   Attempts: {result['attempts']}")
            print(f"   Improvements: {len(result['improvement_chain'])}")
            print(f"   Classifiers Used: {result['classifier_info']['count']}")
            print(f"   Classifier Types: {result['classifier_info']['types']}")
        
        # Show overall performance
        print(f"\n📊 OVERALL PERFORMANCE:")
        summary = agent.get_performance_summary()
        print(f"   Total Improvements: {summary.get('total_improvements_attempted', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.2%}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("This is expected if no classifiers are available in the marketplace.")
        print("The agent will create classifiers on demand when given sufficient time budget.")

if __name__ == "__main__":
    asyncio.run(demo_autonomous_agent()) 