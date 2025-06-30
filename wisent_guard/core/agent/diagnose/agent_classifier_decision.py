from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
import asyncio
import time
from .classifier_marketplace import ClassifierMarketplace, ClassifierListing, ClassifierCreationEstimate
from ..budget import get_budget_manager, track_task_performance, ResourceType

@dataclass
class TaskAnalysis:
    """Analysis of what classifiers might be needed for a task."""
    predicted_issue_types: List[str]
    confidence_scores: Dict[str, float]  # issue_type -> confidence
    task_complexity: float  # 0.0 to 1.0
    prompt_content: str
    risk_level: float  # 0.0 to 1.0, higher means more risk of issues

@dataclass
class ClassifierDecision:
    """A decision about whether to use an existing classifier or create a new one."""
    issue_type: str
    action: str  # "use_existing", "create_new", "skip"
    selected_classifier: Optional[ClassifierListing] = None
    creation_estimate: Optional[ClassifierCreationEstimate] = None
    reasoning: str = ""
    confidence: float = 0.0

class AgentClassifierDecisionSystem:
    """
    Intelligent system that helps the agent make autonomous decisions about
    which classifiers to use based on task analysis and cost-benefit considerations.
    """
    
    def __init__(self, marketplace: ClassifierMarketplace):
        self.marketplace = marketplace
        self.decision_history: List[ClassifierDecision] = []
        
    def analyze_task_requirements(self, prompt: str, context: str = "") -> TaskAnalysis:
        """
        Analyze a task/prompt to predict what types of classifiers might be needed.
        
        Args:
            prompt: The prompt or task to analyze
            context: Additional context about the task
            
        Returns:
            Analysis of predicted classifier needs
        """
        prompt_lower = prompt.lower()
        combined_text = f"{prompt} {context}".lower()
        
        # Pattern-based analysis for different issue types
        issue_patterns = {
            "hallucination": [
                r'\b(fact|truth|accurate|verify|correct|real|actual|evidence)\b',
                r'\b(when|where|who|what|how|which)\b',  # Question words
                r'\b(history|science|geography|politics|news|event)\b',
                r'\b(happen|occur|exist|true|false)\b'
            ],
            "harmful": [
                r'\b(violence|weapon|drug|illegal|harm|danger|kill|hurt)\b',
                r'\b(hate|discriminat|racist|sexist|offensive)\b',
                r'\b(suicide|self.harm|abuse|threat)\b',
                r'\b(explicit|sexual|adult|nsfw)\b'
            ],
            "bias": [
                r'\b(gender|race|ethnic|religion|age|disability)\b',
                r'\b(stereotype|prejudice|discrimination|bias)\b',
                r'\b(men|women|black|white|asian|muslim|christian)\b',
                r'\b(should|better|worse|superior|inferior)\b'
            ],
            "quality": [
                r'\b(explain|describe|analyze|compare|summarize)\b',
                r'\b(write|create|generate|compose|draft)\b',
                r'\b(essay|report|article|story|poem)\b',
                r'\b(help|advice|suggestion|recommend)\b'
            ],
            "coherence": [
                r'\b(logic|reason|consistent|contradict|make sense)\b',
                r'\b(confusing|unclear|jumbled|mixed up)\b',
                r'\b(follow|sequence|order|structure)\b'
            ]
        }
        
        predicted_issues = {}
        
        for issue_type, patterns in issue_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                found = re.findall(pattern, combined_text)
                if found:
                    matches += len(found)
                    score += len(found) * 0.1  # Each match adds 0.1
            
            # Normalize score based on text length
            text_length = len(combined_text.split())
            if text_length > 0:
                score = min(score / (text_length / 100), 1.0)  # Normalize per 100 words
            
            if score > 0.05:  # Only include if meaningful signal
                predicted_issues[issue_type] = score
        
        # Calculate task complexity based on prompt characteristics
        complexity_factors = [
            len(prompt.split()) / 100,  # Length factor
            len(re.findall(r'\?', prompt)) * 0.2,  # Question complexity
            len(re.findall(r'\b(complex|difficult|detailed|comprehensive)\b', prompt_lower)) * 0.3,
            len(predicted_issues) * 0.1  # Multiple issue types = more complex
        ]
        task_complexity = min(sum(complexity_factors), 1.0)
        
        # Calculate risk level
        risk_indicators = [
            'sensitive', 'controversial', 'political', 'medical', 'legal',
            'advice', 'recommendation', 'should', 'must', 'never', 'always'
        ]
        risk_score = sum(0.1 for indicator in risk_indicators if indicator in prompt_lower)
        risk_level = min(risk_score, 1.0)
        
        # Sort issue types by confidence
        sorted_issues = sorted(predicted_issues.keys(), key=lambda x: predicted_issues[x], reverse=True)
        
        return TaskAnalysis(
            predicted_issue_types=sorted_issues,
            confidence_scores=predicted_issues,
            task_complexity=task_complexity,
            prompt_content=prompt,
            risk_level=risk_level
        )
    
    async def make_classifier_decisions(self, 
                                      task_analysis: TaskAnalysis,
                                      quality_threshold: float = 0.3,
                                      time_budget_minutes: float = 10.0,
                                      max_classifiers: int = None) -> List[ClassifierDecision]:
        """
        Make intelligent decisions about which classifiers to use for a task.
        
        Args:
            task_analysis: Analysis of the task requirements
            quality_threshold: Minimum quality score to accept existing classifiers
            time_budget_minutes: Maximum time budget for creating new classifiers
            max_classifiers: Maximum number of classifiers to use (None = no limit)
            
        Returns:
            List of classifier decisions
        """
        print(f"🤖 Agent making classifier decisions...")
        print(f"   🎯 Predicted issues: {task_analysis.predicted_issue_types}")
        print(f"   📊 Task complexity: {task_analysis.task_complexity:.2f}")
        print(f"   ⚠️ Risk level: {task_analysis.risk_level:.2f}")
        
        # Set up budget manager
        budget_manager = get_budget_manager()
        budget_manager.set_time_budget(time_budget_minutes)
        
        # Discover available classifiers
        await asyncio.sleep(0)  # Make this async-compatible
        available_classifiers = self.marketplace.discover_available_classifiers()
        
        decisions = []
        classifier_count = 0
        
        # Process each predicted issue type
        for issue_type in task_analysis.predicted_issue_types:
            if max_classifiers and classifier_count >= max_classifiers:
                print(f"   ⏹️ Reached maximum classifier limit ({max_classifiers})")
                break
                
            confidence = task_analysis.confidence_scores[issue_type]
            print(f"\n   🔍 Analyzing need for {issue_type} classifier (confidence: {confidence:.2f})")
            
            # Find best existing classifier for this issue type
            existing_options = [c for c in available_classifiers if c.issue_type == issue_type]
            best_existing = max(existing_options, key=lambda x: x.quality_score) if existing_options else None
            
            # Get creation estimate
            creation_estimate = self.marketplace.get_creation_estimate(issue_type)
            
            # Make decision based on multiple factors
            decision = self._evaluate_classifier_options(
                issue_type=issue_type,
                confidence=confidence,
                best_existing=best_existing,
                creation_estimate=creation_estimate,
                quality_threshold=quality_threshold,
                budget_manager=budget_manager,
                task_analysis=task_analysis
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
        
        print(f"\n   📋 Final decisions: {len([d for d in decisions if d.action != 'skip'])} classifiers selected")
        return decisions
    
    def _evaluate_classifier_options(self,
                                   issue_type: str,
                                   confidence: float,
                                   best_existing: Optional[ClassifierListing],
                                   creation_estimate: ClassifierCreationEstimate,
                                   quality_threshold: float,
                                   budget_manager,
                                   task_analysis: TaskAnalysis) -> ClassifierDecision:
        """Evaluate whether to use existing, create new, or skip a classifier."""
        
        # Factor 1: Is this issue type important enough?
        importance_threshold = 0.1 + (task_analysis.risk_level * 0.2)  # Higher risk = lower threshold
        if confidence < importance_threshold:
            return ClassifierDecision(
                issue_type=issue_type,
                action="skip",
                reasoning=f"Confidence {confidence:.2f} below threshold {importance_threshold:.2f}",
                confidence=0.0
            )
        
        # Factor 2: Existing classifier quality
        existing_quality = best_existing.quality_score if best_existing else 0.0
        
        # Factor 3: Time constraints
        time_budget = budget_manager.get_budget(ResourceType.TIME)
        training_time_seconds = creation_estimate.estimated_training_time_minutes * 60
        can_afford_creation = time_budget.can_afford(training_time_seconds)
        
        # Factor 4: Expected benefit vs cost
        creation_benefit = creation_estimate.estimated_quality_score
        existing_benefit = existing_quality
        remaining_minutes = time_budget.remaining_budget / 60
        creation_cost = creation_estimate.estimated_training_time_minutes / remaining_minutes if remaining_minutes > 0 else float('inf')
        
        # Decision logic
        if best_existing and existing_quality >= quality_threshold:
            if existing_quality >= creation_benefit or not can_afford_creation:
                return ClassifierDecision(
                    issue_type=issue_type,
                    action="use_existing",
                    selected_classifier=best_existing,
                    reasoning=f"Existing classifier quality {existing_quality:.2f} meets threshold",
                    confidence=confidence * existing_quality
                )
        
        if can_afford_creation and creation_benefit > existing_benefit:
            return ClassifierDecision(
                issue_type=issue_type,
                action="create_new",
                creation_estimate=creation_estimate,
                reasoning=f"Creating new classifier (est. quality {creation_benefit:.2f} > existing {existing_benefit:.2f})",
                confidence=confidence * creation_estimate.confidence
            )
        
        if best_existing:
            return ClassifierDecision(
                issue_type=issue_type,
                action="use_existing",
                selected_classifier=best_existing,
                reasoning=f"Using existing despite low quality - time/budget constraints",
                confidence=confidence * existing_quality * 0.7  # Penalty for low quality
            )
        
        return ClassifierDecision(
            issue_type=issue_type,
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
                print(f"   🏗️ Creating new {decision.issue_type} classifier...")
                start_time = time.time()
                try:
                    new_classifier = await self.marketplace.create_classifier_on_demand(
                        issue_type=decision.issue_type
                    )
                    end_time = time.time()
                    
                    # Track performance for future budget estimates
                    track_task_performance(
                        task_name=f"classifier_training_{decision.issue_type}",
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    config = new_classifier.to_config()
                    classifier_configs.append(config)
                    print(f"      ✅ Created: {config['path']} (took {end_time - start_time:.1f}s)")
                except Exception as e:
                    print(f"      ❌ Failed to create {decision.issue_type} classifier: {e}")
                    continue
        
        return classifier_configs
    
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
            summary += f"• {decision.issue_type}: {decision.action}\n"
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