from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
import asyncio
import time
import sys
import os

# Add the lm-harness-integration path for benchmark selection
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'lm-harness-integration'))

from .classifier_marketplace import ClassifierMarketplace, ClassifierListing, ClassifierCreationEstimate
from ..budget import get_budget_manager, track_task_performance, ResourceType

@dataclass
class TaskAnalysis:
    """Analysis of what classifiers might be needed for a task."""
    prompt_content: str
    relevant_benchmarks: List[Dict[str, Any]] = None  # Selected benchmarks for training and steering

@dataclass
class ClassifierDecision:
    """A decision about whether to use an existing classifier or create a new one."""
    benchmark_name: str
    action: str  # "use_existing", "create_new", "skip"
    selected_classifier: Optional[ClassifierListing] = None
    creation_estimate: Optional[ClassifierCreationEstimate] = None
    reasoning: str = ""
    confidence: float = 0.0

@dataclass
class SingleClassifierDecision:
    """Decision about creating one combined classifier from multiple benchmarks."""
    benchmark_names: List[str]
    action: str  # "use_existing", "create_new", "skip"
    selected_classifier: Optional[ClassifierListing] = None
    creation_estimate: Optional[ClassifierCreationEstimate] = None
    reasoning: str = ""
    confidence: float = 0.0

@dataclass
class ClassifierParams:
    """Model-determined classifier parameters."""
    optimal_layer: int  # 8-20: Based on semantic complexity needed
    classification_threshold: float  # 0.1-0.9: Based on quality strictness required
    training_samples: int  # 10-50: Based on complexity and time constraints
    classifier_type: str  # logistic/svm/neural: Based on data characteristics
    reasoning: str = ""
    model_name: str = "unknown"  # Model name for matching existing classifiers
    
    # Additional classifier configuration parameters
    aggregation_method: str = "last_token"  # last_token/mean/max for activation aggregation
    token_aggregation: str = "average"  # average/final/first/max/min for token score aggregation
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    hidden_dim: int = 128

@dataclass
class SteeringParams:
    """Model-determined steering parameters."""
    steering_method: str  # CAA/HPR/DAC/BiPO/KSteering: Best fit for prompt type
    initial_strength: float  # 0.1-2.0: How aggressive to start
    increment: float  # 0.1-0.5: How much to increase per failed attempt
    maximum_strength: float  # 0.5-3.0: Upper limit to prevent over-steering
    method_specific_params: Dict[str, Any] = None  # Beta values, thresholds, etc.
    reasoning: str = ""

@dataclass
class QualityResult:
    """Result of quality evaluation."""
    score: float  # Classifier prediction score
    acceptable: bool  # Model judgment if quality is acceptable
    reasoning: str = ""

@dataclass
class QualityControlledResponse:
    """Final response with complete metadata."""
    response_text: str
    final_quality_score: float
    attempts_needed: int
    classifier_params_used: ClassifierParams
    steering_params_used: Optional[SteeringParams] = None
    quality_progression: List[float] = None  # Quality scores for each attempt
    total_time_seconds: float = 0.0

class AgentClassifierDecisionSystem:
    """
    Intelligent system that helps the agent make autonomous decisions about
    which classifiers to use based on task analysis and cost-benefit considerations.
    """
    
    def __init__(self, marketplace: ClassifierMarketplace):
        self.marketplace = marketplace
        self.decision_history: List[ClassifierDecision] = []
        
    def analyze_task_requirements(self, prompt: str, context: str = "", 
                                 priority: str = "all", fast_only: bool = False, 
                                 time_budget_minutes: float = 5.0, max_benchmarks: int = 1) -> TaskAnalysis:
        """
        Analyze a task/prompt to select relevant benchmarks for training and steering.
        
        Args:
            prompt: The prompt or task to analyze
            context: Additional context about the task
            priority: Priority level for benchmark selection
            fast_only: Only use fast benchmarks
            time_budget_minutes: Time budget for benchmark selection
            max_benchmarks: Maximum number of benchmarks to select
            prefer_fast: Prefer fast benchmarks
            
        Returns:
            Analysis with relevant benchmarks for direct use
        """
        print(f"🔍 Analyzing task requirements for prompt...")
        
        # Get relevant benchmarks for the prompt using priority-aware selection
        existing_model = getattr(self.marketplace, 'model', None)
        relevant_benchmarks = self._get_relevant_benchmarks_for_prompt(
            prompt, 
            existing_model=existing_model,
            priority=priority,
            fast_only=fast_only,
            time_budget_minutes=time_budget_minutes,
            max_benchmarks=max_benchmarks
        )
        print(f"   📊 Found {len(relevant_benchmarks)} relevant benchmarks")
        
        return TaskAnalysis(
            prompt_content=prompt,
            relevant_benchmarks=relevant_benchmarks
        )
    
    def _get_relevant_benchmarks_for_prompt(self, prompt: str, existing_model=None, 
                                           priority: str = "all", fast_only: bool = False, 
                                           time_budget_minutes: float = 5.0, max_benchmarks: int = 1) -> List[Dict[str, Any]]:
        """Get relevant benchmarks for the prompt using the intelligent selection system with priority awareness."""
        try:
            # Import the benchmark selection function from the correct location
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lm-harness-integration'))
            from populate_tasks import get_relevant_benchmarks_for_prompt
            
            # Use priority-aware selection with provided parameters
            relevant_benchmarks = get_relevant_benchmarks_for_prompt(
                prompt=prompt, 
                max_benchmarks=max_benchmarks, 
                existing_model=existing_model,
                priority=priority,
                fast_only=fast_only,
                time_budget_minutes=time_budget_minutes
            )
            
            return relevant_benchmarks
        except Exception as e:
            print(f"   ⚠️ Failed to get relevant benchmarks: {e}")
            # Fallback to basic high-priority benchmarks
            return [
                {'benchmark': 'mmlu', 'explanation': 'General knowledge benchmark', 'relevance_score': 1, 'priority': 'high', 'loading_time': 9.5},
                {'benchmark': 'truthfulqa_mc1', 'explanation': 'Truthfulness benchmark', 'relevance_score': 2, 'priority': 'high', 'loading_time': 11.2},
                {'benchmark': 'hellaswag', 'explanation': 'Commonsense reasoning benchmark', 'relevance_score': 3, 'priority': 'high', 'loading_time': 12.8}
            ]
    

    

    
    async def create_single_quality_classifier(self, 
                                          task_analysis: TaskAnalysis,
                                          classifier_params: 'ClassifierParams',
                                          quality_threshold: float = 0.3,
                                          time_budget_minutes: float = 10.0) -> SingleClassifierDecision:
        """
        Create a single classifier trained on one benchmark.
        
        Args:
            task_analysis: Analysis with relevant benchmarks
            classifier_params: Model-determined classifier parameters
            quality_threshold: Minimum quality score to accept existing classifiers
            time_budget_minutes: Maximum time budget for creating new classifiers
            
        Returns:
            Single classifier decision for the selected benchmark
        """
        print(f"🔍 Creating single quality classifier from {len(task_analysis.relevant_benchmarks)} benchmark(s)...")
        
        # Extract benchmark names (should be just one now)
        benchmark_names = [b['benchmark'] for b in task_analysis.relevant_benchmarks]
        
        if not benchmark_names:
            return SingleClassifierDecision(
                benchmark_names=[],
                action="skip",
                reasoning="No benchmarks selected for classifier training",
                confidence=0.0
            )
        
        # Use first (and should be only) benchmark
        benchmark_name = benchmark_names[0]
        print(f"   📊 Using benchmark: {benchmark_name}")
        
        # Set up budget manager
        budget_manager = get_budget_manager()
        budget_manager.set_time_budget(time_budget_minutes)
        
        # Look for existing classifier for this exact model/layer/benchmark combination
        available_classifiers = self.marketplace.discover_available_classifiers()
        model_name = classifier_params.model_name if hasattr(classifier_params, 'model_name') else "unknown"
        layer = classifier_params.optimal_layer
        
        # Create specific classifier identifier
        classifier_id = f"{model_name}_{benchmark_name}_layer_{layer}"
        
        print(f"   🔍 Checking for existing classifier: {classifier_id}")
        
        # Find existing classifier with exact match
        existing_classifier = None
        for classifier in available_classifiers:
            # Check if classifier matches our exact requirements
            if (benchmark_name.lower() in classifier.path.lower() and
                str(layer) in classifier.path and
                classifier.layer == layer):
                existing_classifier = classifier
                print(f"   ✅ Found existing classifier: {classifier.path}")
                break
        
        # Decision logic for single benchmark classifier
        if existing_classifier and existing_classifier.quality_score >= quality_threshold:
            return SingleClassifierDecision(
                benchmark_names=[benchmark_name],
                action="use_existing",
                selected_classifier=existing_classifier,
                reasoning=f"Found existing classifier for {benchmark_name} at layer {layer} with quality {existing_classifier.quality_score:.2f}",
                confidence=existing_classifier.quality_score
            )
        
        # Get creation estimate for single benchmark classifier
        creation_estimate = self.marketplace.get_creation_estimate(benchmark_name)
        
        # Check if we can afford to create new classifier
        training_time_seconds = creation_estimate.estimated_training_time_minutes * 60
        time_budget = budget_manager.get_budget(ResourceType.TIME)
        
        if time_budget.can_afford(training_time_seconds):
            return SingleClassifierDecision(
                benchmark_names=[benchmark_name],
                action="create_new",
                creation_estimate=creation_estimate,
                reasoning=f"Creating new classifier for {benchmark_name} at layer {layer}",
                confidence=creation_estimate.confidence
            )
        else:
            return SingleClassifierDecision(
                benchmark_names=[benchmark_name],
                action="skip",
                reasoning=f"Insufficient time budget for creation (need {creation_estimate.estimated_training_time_minutes:.1f}min)",
                confidence=0.0
            )

    async def execute_single_classifier_decision(self, decision: SingleClassifierDecision, classifier_params: 'ClassifierParams') -> Optional[Any]:
        """
        Execute the single classifier decision to create or use the benchmark classifier.
        
        Args:
            decision: The single classifier decision to execute
            classifier_params: Model-determined classifier parameters
            
        Returns:
            The trained classifier instance or None if skipped
        """
        if decision.action == "skip":
            print(f"   ⏹️ Skipping classifier creation: {decision.reasoning}")
            return None
            
        elif decision.action == "use_existing":
            print(f"   📦 Using existing classifier: {decision.selected_classifier.path}")
            print(f"      Quality: {decision.selected_classifier.quality_score:.3f}")
            print(f"      Layer: {decision.selected_classifier.layer}")
            return decision.selected_classifier
            
        elif decision.action == "create_new":
            benchmark_name = decision.benchmark_names[0] if decision.benchmark_names else "unknown"
            print(f"   🏗️ Creating new classifier for benchmark: {benchmark_name}")
            start_time = time.time()
            try:
                # Create classifier using single benchmark training data
                new_classifier = await self._create_single_benchmark_classifier(
                    benchmark_name=benchmark_name,
                    classifier_params=classifier_params
                )
                
                creation_time = time.time() - start_time
                print(f"      ✅ Classifier created successfully in {creation_time:.1f}s")
                return new_classifier
                
            except Exception as e:
                print(f"      ❌ Failed to create classifier: {e}")
                return None
                
        return None

    async def _create_single_benchmark_classifier(self, benchmark_name: str, classifier_params: 'ClassifierParams') -> Optional[Any]:
        """
        Create a classifier for a single benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to use for training
            classifier_params: Model-determined classifier parameters
            
        Returns:
            The trained classifier instance or None if failed
        """
        from .create_classifier import ClassifierCreator
        from ...training_config import TrainingConfig
        
        try:
            # Create training config
            config = TrainingConfig(
                issue_type=f"quality_{benchmark_name}",
                layer=classifier_params.optimal_layer,
                classifier_type=classifier_params.classifier_type,
                threshold=classifier_params.classification_threshold,
                training_samples=classifier_params.training_samples,
                model_name=self.marketplace.model.name if self.marketplace.model else "unknown"
            )
            
            # Create classifier creator
            creator = ClassifierCreator(self.marketplace.model)
            
            # Create classifier using benchmark-specific training data
            result = await creator.create_classifier_for_issue_with_benchmarks(
                issue_type=f"quality_{benchmark_name}",
                relevant_benchmarks=[benchmark_name],
                layer=classifier_params.optimal_layer,
                num_samples=classifier_params.training_samples,
                config=config
            )
            
            return result.classifier if result else None
            
        except Exception as e:
            print(f"      ❌ Error in single benchmark classifier creation: {e}")
            raise

    async def _create_combined_classifier(self, benchmark_names: List[str], classifier_params: 'ClassifierParams'):
        """
        Create a classifier using combined training data from multiple benchmarks.
        
        Args:
            benchmark_names: List of benchmark names to combine
            classifier_params: Model-determined parameters for classifier creation
            
        Returns:
            Trained classifier instance
        """
        from .create_classifier import ClassifierCreator
        
        try:
            # Initialize classifier creator
            creator = ClassifierCreator(self.marketplace.model)
            
            # Create classifier using combined benchmark training data
            print(f"      📊 Loading combined training data from benchmarks: {benchmark_names}")
            classifier = await creator.create_combined_benchmark_classifier(
                benchmark_names=benchmark_names,
                classifier_params=classifier_params
            )
            
            return classifier
            
        except Exception as e:
            print(f"      ❌ Error in combined classifier creation: {e}")
            raise

    async def make_classifier_decisions(self, 
                                      task_analysis: TaskAnalysis,
                                      quality_threshold: float = 0.3,
                                      time_budget_minutes: float = 10.0,
                                      max_classifiers: int = None) -> List[ClassifierDecision]:
        """
        Make decisions about which benchmark-specific classifiers to create or use.
        
        Args:
            task_analysis: Analysis with relevant benchmarks
            quality_threshold: Minimum quality score to accept existing classifiers
            time_budget_minutes: Maximum time budget for creating new classifiers
            max_classifiers: Maximum number of classifiers to use (None = no limit)
            
        Returns:
            List of classifier decisions for each benchmark
        """
        # Set up budget manager
        budget_manager = get_budget_manager()
        budget_manager.set_time_budget(time_budget_minutes)
        
        # Discover available classifiers
        await asyncio.sleep(0)  # Make this async-compatible
        available_classifiers = self.marketplace.discover_available_classifiers()
        
        decisions = []
        classifier_count = 0
        
        # Create one classifier per relevant benchmark
        for benchmark_info in task_analysis.relevant_benchmarks:
            if max_classifiers and classifier_count >= max_classifiers:
                print(f"   ⏹️ Reached maximum classifier limit ({max_classifiers})")
                break
                
            benchmark_name = benchmark_info['benchmark']
            print(f"\n   🔍 Analyzing classifier for benchmark: {benchmark_name}")
            
            # Look for existing benchmark-specific classifier
            existing_options = [c for c in available_classifiers if benchmark_name.lower() in c.path.lower()]
            best_existing = max(existing_options, key=lambda x: x.quality_score) if existing_options else None
            
            # Get creation estimate for this benchmark
            creation_estimate = self.marketplace.get_creation_estimate(benchmark_name)
            
            # Make decision based on multiple factors
            decision = self._evaluate_benchmark_classifier_options(
                benchmark_name=benchmark_name,
                best_existing=best_existing,
                creation_estimate=creation_estimate,
                quality_threshold=quality_threshold,
                budget_manager=budget_manager
            )
            
            decisions.append(decision)
            
            # Update budget and count
            if decision.action == "create_new":
                training_time_seconds = creation_estimate.estimated_training_time_minutes * 60
                budget_manager.get_budget(ResourceType.TIME).spend(training_time_seconds)
                classifier_count += 1
                remaining_minutes = budget_manager.get_budget(ResourceType.TIME).remaining_budget / 60
                print(f"      ⏱️ Remaining time budget: {remaining_minutes:.1f} minutes")
            elif decision.action == "use_existing":
                classifier_count += 1
            
            print(f"      ✅ Decision: {decision.action} - {decision.reasoning}")
        
        # Store decisions in history
        self.decision_history.extend(decisions)
        
        return decisions
    
    def _evaluate_benchmark_classifier_options(self,
                                          benchmark_name: str,
                                          best_existing: Optional[ClassifierListing],
                                          creation_estimate: ClassifierCreationEstimate,
                                          quality_threshold: float,
                                          budget_manager) -> ClassifierDecision:
        """Evaluate whether to use existing, create new, or skip a benchmark-specific classifier."""
        
        # Factor 1: Existing classifier quality
        existing_quality = best_existing.quality_score if best_existing else 0.0
        
        # Factor 2: Time constraints
        time_budget = budget_manager.get_budget(ResourceType.TIME)
        training_time_seconds = creation_estimate.estimated_training_time_minutes * 60
        can_afford_creation = time_budget.can_afford(training_time_seconds)
        
        # Factor 3: Expected benefit vs cost
        creation_benefit = creation_estimate.estimated_quality_score
        existing_benefit = existing_quality
        
        # Decision logic
        if best_existing and existing_quality >= quality_threshold:
            if existing_quality >= creation_benefit or not can_afford_creation:
                return ClassifierDecision(
                    benchmark_name=benchmark_name,
                    action="use_existing",
                    selected_classifier=best_existing,
                    reasoning=f"Existing classifier quality {existing_quality:.2f} meets threshold",
                    confidence=existing_quality
                )
        
        if can_afford_creation and creation_benefit > existing_benefit:
            return ClassifierDecision(
                benchmark_name=benchmark_name,
                action="create_new",
                creation_estimate=creation_estimate,
                reasoning=f"Creating new classifier (est. quality {creation_benefit:.2f} > existing {existing_benefit:.2f})",
                confidence=creation_estimate.confidence
            )
        
        if best_existing:
            return ClassifierDecision(
                benchmark_name=benchmark_name,
                action="use_existing",
                selected_classifier=best_existing,
                reasoning=f"Using existing despite low quality - time/budget constraints",
                confidence=existing_quality * 0.7  # Penalty for low quality
            )
        
        return ClassifierDecision(
            benchmark_name=benchmark_name,
            action="skip",
            reasoning="No suitable existing classifier and cannot create new within budget",
            confidence=0.0
        )
    
    async def execute_decisions(self, decisions: List[ClassifierDecision]) -> List[Dict[str, Any]]:
        """
        Execute the classifier decisions and return the final classifier configs.
        
        Args:
            decisions: List of decisions to execute
            
        Returns:
            List of classifier configurations ready for use
        """
        classifier_configs = []
        
        for decision in decisions:
            if decision.action == "skip":
                continue
                
            elif decision.action == "use_existing":
                config = decision.selected_classifier.to_config()
                classifier_configs.append(config)
                print(f"   📎 Using existing {decision.issue_type} classifier: {config['path']}")
                
            elif decision.action == "create_new":
                print(f"   🏗️ Creating new classifier for benchmark: {decision.benchmark_name}...")
                start_time = time.time()
                try:
                    # Create benchmark-specific classifier
                    new_classifier = await self._create_classifier_for_benchmark(
                        benchmark_name=decision.benchmark_name
                    )
                    
                    end_time = time.time()
                    
                    # Track performance for future budget estimates
                    track_task_performance(
                        task_name=f"classifier_training_{decision.benchmark_name}",
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    config = new_classifier.to_config()
                    config['benchmark'] = decision.benchmark_name
                    classifier_configs.append(config)
                    print(f"      ✅ Created: {config['path']} (took {end_time - start_time:.1f}s)")
                except Exception as e:
                    print(f"      ❌ Failed to create {decision.benchmark_name} classifier: {e}")
                    continue
        
        return classifier_configs
    
    async def _create_classifier_for_benchmark(self, benchmark_name: str):
        """
        Create a classifier trained specifically on a benchmark dataset.
        
        Args:
            benchmark_name: Name of the benchmark to train on
            
        Returns:
            Trained classifier instance
        """
        from .create_classifier import ClassifierCreator
        
        try:
            # Initialize classifier creator
            creator = ClassifierCreator(self.marketplace.model)
            
            # Create classifier using benchmark-specific training data
            print(f"      📊 Loading training data from benchmark: {benchmark_name}")
            classifier = await creator.create_classifier_for_issue_with_benchmarks(
                issue_type=benchmark_name,  # Use benchmark name as issue type
                relevant_benchmarks=[benchmark_name],
                num_samples=50
            )
            
            return classifier
            
        except Exception as e:
            print(f"      ⚠️ Benchmark-based creation failed: {e}")
            raise e
    
    def get_decision_summary(self) -> str:
        """Get a summary of recent classifier decisions."""
        if not self.decision_history:
            return "No classifier decisions made yet."
        
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        
        summary = "\n🤖 Recent Classifier Decisions\n"
        summary += "=" * 40 + "\n"
        
        action_counts = {}
        for decision in recent_decisions:
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        
        summary += f"Actions taken: {dict(action_counts)}\n\n"
        
        for decision in recent_decisions[-5:]:  # Show last 5
            summary += f"• {decision.benchmark_name}: {decision.action}\n"
            summary += f"  Reasoning: {decision.reasoning}\n"
            summary += f"  Confidence: {decision.confidence:.2f}\n\n"
        
        return summary
    
    async def smart_classifier_selection(self, 
                                       prompt: str,
                                       context: str = "",
                                       quality_threshold: float = 0.3,
                                       time_budget_minutes: float = 10.0,
                                       max_classifiers: int = None) -> List[Dict[str, Any]]:
        """
        One-stop method for intelligent classifier selection.
        
        Args:
            prompt: The task/prompt to analyze
            context: Additional context
            quality_threshold: Minimum quality for existing classifiers
            time_budget_minutes: Time budget for creating new classifiers
            max_classifiers: Maximum number of classifiers to use
            
        Returns:
            List of classifier configurations ready for use
        """
        print(f"🧠 Smart classifier selection for task...")
        
        # Step 1: Analyze task requirements
        task_analysis = self.analyze_task_requirements(prompt, context)
        
        # Step 2: Make decisions about classifiers
        decisions = await self.make_classifier_decisions(
            task_analysis=task_analysis,
            quality_threshold=quality_threshold,
            time_budget_minutes=time_budget_minutes,
            max_classifiers=max_classifiers
        )
        
        # Step 3: Execute decisions
        classifier_configs = await self.execute_decisions(decisions)
        
        print(f"🎯 Selected {len(classifier_configs)} classifiers for the task")
        return classifier_configs 