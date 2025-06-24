"""
Individual MCP tools for Wisent-Guard self-reflection capabilities.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseMCPTool(ABC):
    """Base class for MCP tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def get_schema(self) -> dict:
        """Get the tool's input schema."""
        pass
    
    @abstractmethod
    async def execute(self, arguments: dict) -> dict:
        """Execute the tool with given arguments."""
        pass


class SelfReflectionTool(BaseMCPTool):
    """Tool for comprehensive self-reflection analysis."""
    
    def __init__(self, wisent_server):
        super().__init__(
            "perform_self_reflection",
            "Perform comprehensive self-reflection on model output including hallucination detection, behavior analysis, and quality assessment"
        )
        self.server = wisent_server
    
    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "response_text": {
                    "type": "string",
                    "description": "The model response to analyze"
                },
                "original_prompt": {
                    "type": "string",
                    "description": "The original prompt that generated the response"
                },
                "analysis_depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "comprehensive"],
                    "description": "Depth of analysis to perform",
                    "default": "standard"
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific areas to focus analysis on",
                    "default": ["hallucinations", "coherence", "accuracy", "bias"]
                }
            },
            "required": ["response_text"]
        }
    
    async def execute(self, arguments: dict) -> dict:
        """Execute comprehensive self-reflection."""
        response_text = arguments["response_text"]
        original_prompt = arguments.get("original_prompt", "")
        analysis_depth = arguments.get("analysis_depth", "standard")
        focus_areas = arguments.get("focus_areas", ["hallucinations", "coherence", "accuracy", "bias"])
        
        results = {
            "self_reflection_summary": {
                "timestamp": datetime.now().isoformat(),
                "analysis_depth": analysis_depth,
                "focus_areas": focus_areas
            }
        }
        
        # Perform different analyses based on focus areas
        if "hallucinations" in focus_areas:
            hallucination_result = await self.server._analyze_hallucinations({
                "response_text": response_text,
                "context": original_prompt
            })
            results["hallucination_analysis"] = hallucination_result
        
        if any(area in focus_areas for area in ["coherence", "accuracy", "bias"]):
            quality_result = await self.server._assess_response_quality({
                "response_text": response_text,
                "prompt": original_prompt,
                "criteria": [area for area in focus_areas if area in ["coherence", "accuracy", "relevance", "helpfulness"]]
            })
            results["quality_assessment"] = quality_result
        
        # Behavior analysis
        behavior_result = await self.server._detect_problematic_behavior({
            "response_text": response_text,
            "behavior_types": ["nonsense", "repetition", "bias", "toxicity"]
        })
        results["behavior_analysis"] = behavior_result
        
        # Overall reflection score
        reflection_score = self._calculate_reflection_score(results)
        results["overall_reflection_score"] = reflection_score
        
        return results
    
    def _calculate_reflection_score(self, results: dict) -> dict:
        """Calculate overall reflection score."""
        issues_count = 0
        total_confidence = 0.0
        
        if "hallucination_analysis" in results:
            if results["hallucination_analysis"]["is_hallucinating"]:
                issues_count += 1
            total_confidence += results["hallucination_analysis"]["confidence"]
        
        if "behavior_analysis" in results:
            if results["behavior_analysis"]["has_problematic_behavior"]:
                issues_count += 1
        
        if "quality_assessment" in results:
            quality_score = results["quality_assessment"]["overall_quality_score"]
            if quality_score < 0.6:
                issues_count += 1
        
        return {
            "issues_detected": issues_count,
            "needs_improvement": issues_count > 0,
            "confidence_level": total_confidence / max(len(results) - 1, 1),  # Exclude summary
            "recommendation": "regenerate" if issues_count > 1 else "acceptable" if issues_count == 0 else "minor_edit"
        }


class HallucinationDetectionTool(BaseMCPTool):
    """Specialized tool for hallucination detection."""
    
    def __init__(self, wisent_server):
        super().__init__(
            "detect_hallucinations",
            "Specialized detection of hallucinations, factual errors, and knowledge inconsistencies"
        )
        self.server = wisent_server
    
    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "response_text": {
                    "type": "string",
                    "description": "The response to check for hallucinations"
                },
                "knowledge_domain": {
                    "type": "string",
                    "description": "Domain of knowledge for specialized checking",
                    "default": "general"
                },
                "fact_check_level": {
                    "type": "string",
                    "enum": ["basic", "intermediate", "advanced"],
                    "description": "Level of fact-checking to perform",
                    "default": "intermediate"
                },
                "reference_context": {
                    "type": "string",
                    "description": "Reference context or known facts to compare against"
                }
            },
            "required": ["response_text"]
        }
    
    async def execute(self, arguments: dict) -> dict:
        """Execute hallucination detection."""
        response_text = arguments["response_text"]
        knowledge_domain = arguments.get("knowledge_domain", "general")
        fact_check_level = arguments.get("fact_check_level", "intermediate")
        reference_context = arguments.get("reference_context", "")
        
        # Basic hallucination analysis
        base_result = await self.server._analyze_hallucinations({
            "response_text": response_text,
            "context": reference_context,
            "domain": knowledge_domain
        })
        
        # Enhanced analysis based on fact-check level
        enhanced_analysis = self._perform_enhanced_fact_checking(
            response_text, knowledge_domain, fact_check_level
        )
        
        return {
            "hallucination_detection": {
                **base_result,
                "enhanced_analysis": enhanced_analysis,
                "fact_check_level": fact_check_level,
                "knowledge_domain": knowledge_domain
            }
        }
    
    def _perform_enhanced_fact_checking(self, text: str, domain: str, level: str) -> dict:
        """Perform enhanced fact-checking based on level."""
        checks = {
            "numerical_consistency": self._check_numerical_consistency(text),
            "temporal_consistency": self._check_temporal_consistency(text),
            "logical_consistency": self._check_logical_consistency(text)
        }
        
        if level in ["intermediate", "advanced"]:
            checks["domain_specific"] = self._check_domain_specific_facts(text, domain)
        
        if level == "advanced":
            checks["cross_reference"] = self._check_cross_references(text)
        
        return checks
    
    def _check_numerical_consistency(self, text: str) -> dict:
        """Check for numerical inconsistencies."""
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # Simple checks for obviously wrong numbers
        issues = []
        for num in numbers:
            try:
                val = float(num)
                if val > 1e10:  # Very large numbers might be suspicious
                    issues.append(f"Unusually large number: {num}")
            except ValueError:
                continue
        
        return {
            "numbers_found": len(numbers),
            "issues": issues,
            "confidence": 0.8 if not issues else 0.3
        }
    
    def _check_temporal_consistency(self, text: str) -> dict:
        """Check for temporal inconsistencies."""
        import re
        
        # Find years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        issues = []
        
        for year in years:
            year_int = int(year)
            if year_int > 2024:
                issues.append(f"Future year reference: {year}")
            elif year_int < 1800 and "ancient" not in text.lower():
                issues.append(f"Possibly incorrect historical date: {year}")
        
        return {
            "years_found": len(years),
            "issues": issues,
            "confidence": 0.9 if not issues else 0.2
        }
    
    def _check_logical_consistency(self, text: str) -> dict:
        """Check for logical inconsistencies."""
        issues = []
        
        # Check for contradictory statements
        if "always" in text and "never" in text:
            issues.append("Contains both 'always' and 'never' - potential contradiction")
        
        if "impossible" in text and "definitely" in text:
            issues.append("Claims something is both impossible and definite")
        
        return {
            "logical_issues": issues,
            "confidence": 0.8 if not issues else 0.4
        }
    
    def _check_domain_specific_facts(self, text: str, domain: str) -> dict:
        """Check domain-specific facts."""
        issues = []
        
        if domain == "science":
            if "perpetual motion" in text.lower():
                issues.append("References perpetual motion (physically impossible)")
            if "faster than light" in text.lower() and "impossible" not in text.lower():
                issues.append("Claims faster-than-light travel without noting impossibility")
        
        elif domain == "history":
            # Add historical fact checks
            pass
        
        elif domain == "geography":
            # Add geographical fact checks
            pass
        
        return {
            "domain": domain,
            "domain_issues": issues,
            "confidence": 0.7 if not issues else 0.3
        }
    
    def _check_cross_references(self, text: str) -> dict:
        """Check cross-references and citations."""
        import re
        
        # Look for citation patterns
        citations = re.findall(r'\[[^\]]+\]|\([^)]+\)', text)
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        return {
            "citations_found": len(citations),
            "urls_found": len(urls),
            "has_references": len(citations) > 0 or len(urls) > 0
        }


class BehaviorEditingTool(BaseMCPTool):
    """Tool for editing model behavior using steering methods."""
    
    def __init__(self, wisent_server):
        super().__init__(
            "edit_behavior",
            "Edit model behavior using steering methods to improve response quality and reduce problematic outputs"
        )
        self.server = wisent_server
    
    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "original_prompt": {
                    "type": "string",
                    "description": "The original prompt"
                },
                "current_response": {
                    "type": "string",
                    "description": "The current problematic response"
                },
                "desired_changes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of desired changes (e.g., 'more factual', 'less biased', 'more helpful')"
                },
                "steering_method": {
                    "type": "string",
                    "enum": ["CAA", "KSteering", "auto"],
                    "description": "Steering method to use ('auto' selects best method)",
                    "default": "auto"
                },
                "steering_strength": {
                    "type": "number",
                    "description": "Strength of steering intervention",
                    "default": 1.0,
                    "minimum": 0.1,
                    "maximum": 3.0
                },
                "max_attempts": {
                    "type": "integer",
                    "description": "Maximum number of editing attempts",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 5
                }
            },
            "required": ["original_prompt", "current_response", "desired_changes"]
        }
    
    async def execute(self, arguments: dict) -> dict:
        """Execute behavior editing."""
        original_prompt = arguments["original_prompt"]
        current_response = arguments["current_response"]
        desired_changes = arguments["desired_changes"]
        steering_method = arguments.get("steering_method", "auto")
        steering_strength = arguments.get("steering_strength", 1.0)
        max_attempts = arguments.get("max_attempts", 3)
        
        # Auto-select steering method if needed
        if steering_method == "auto":
            steering_method = self._select_optimal_method(desired_changes)
        
        editing_results = []
        best_result = None
        best_score = 0.0
        
        for attempt in range(max_attempts):
            # Adjust steering strength for each attempt
            attempt_strength = steering_strength * (1.0 + attempt * 0.2)
            
            try:
                result = await self.server._edit_response_behavior({
                    "original_prompt": original_prompt,
                    "problematic_response": current_response,
                    "desired_behavior": " + ".join(desired_changes),
                    "steering_method": steering_method,
                    "steering_strength": attempt_strength
                })
                
                editing_results.append({
                    "attempt": attempt + 1,
                    "steering_strength": attempt_strength,
                    "result": result
                })
                
                if result["improvement_score"] > best_score:
                    best_score = result["improvement_score"]
                    best_result = result
                
                # Stop if we achieved good improvement
                if result["improvement_score"] > 0.8:
                    break
                    
            except Exception as e:
                editing_results.append({
                    "attempt": attempt + 1,
                    "error": str(e)
                })
        
        return {
            "behavior_editing": {
                "original_prompt": original_prompt,
                "original_response": current_response,
                "desired_changes": desired_changes,
                "steering_method_used": steering_method,
                "total_attempts": len(editing_results),
                "best_result": best_result,
                "all_attempts": editing_results,
                "editing_successful": best_score > 0.5,
                "final_improvement_score": best_score
            }
        }
    
    def _select_optimal_method(self, desired_changes: List[str]) -> str:
        """Select optimal steering method based on desired changes."""
        changes_text = " ".join(desired_changes).lower()
        
        # K-Steering is better for complex behavioral changes
        if any(keyword in changes_text for keyword in ["complex", "nuanced", "sophisticated", "detailed"]):
            return "KSteering"
        
        # CAA is good for simple factual/bias corrections
        if any(keyword in changes_text for keyword in ["factual", "accurate", "bias", "simple"]):
            return "CAA"
        
        # Default to CAA for general use
        return "CAA"


class ResponseAnalysisTool(BaseMCPTool):
    """Tool for comprehensive response analysis and metrics."""
    
    def __init__(self, wisent_server):
        super().__init__(
            "analyze_response_comprehensive",
            "Comprehensive analysis of model response including quality metrics, potential issues, and improvement suggestions"
        )
        self.server = wisent_server
    
    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "response_text": {
                    "type": "string",
                    "description": "The response to analyze"
                },
                "original_prompt": {
                    "type": "string",
                    "description": "The original prompt"
                },
                "analysis_categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Categories of analysis to perform",
                    "default": ["quality", "safety", "accuracy", "helpfulness", "coherence"]
                },
                "include_suggestions": {
                    "type": "boolean",
                    "description": "Whether to include improvement suggestions",
                    "default": True
                },
                "compare_to_baseline": {
                    "type": "boolean",
                    "description": "Whether to compare against baseline performance",
                    "default": False
                }
            },
            "required": ["response_text"]
        }
    
    async def execute(self, arguments: dict) -> dict:
        """Execute comprehensive response analysis."""
        response_text = arguments["response_text"]
        original_prompt = arguments.get("original_prompt", "")
        analysis_categories = arguments.get("analysis_categories", ["quality", "safety", "accuracy", "helpfulness", "coherence"])
        include_suggestions = arguments.get("include_suggestions", True)
        compare_to_baseline = arguments.get("compare_to_baseline", False)
        
        analysis_results = {
            "response_analysis": {
                "timestamp": datetime.now().isoformat(),
                "response_length": len(response_text),
                "word_count": len(response_text.split()),
                "categories_analyzed": analysis_categories
            }
        }
        
        # Perform different types of analysis
        if "quality" in analysis_categories:
            quality_result = await self.server._assess_response_quality({
                "response_text": response_text,
                "prompt": original_prompt,
                "criteria": ["accuracy", "coherence", "relevance", "helpfulness"]
            })
            analysis_results["quality_analysis"] = quality_result
        
        if "safety" in analysis_categories:
            safety_result = await self.server._detect_problematic_behavior({
                "response_text": response_text,
                "behavior_types": ["toxicity", "bias", "nonsense"]
            })
            analysis_results["safety_analysis"] = safety_result
        
        if "accuracy" in analysis_categories:
            accuracy_result = await self.server._analyze_hallucinations({
                "response_text": response_text,
                "context": original_prompt
            })
            analysis_results["accuracy_analysis"] = accuracy_result
        
        # Generate overall metrics
        overall_metrics = self._calculate_overall_metrics(analysis_results)
        analysis_results["overall_metrics"] = overall_metrics
        
        # Generate suggestions if requested
        if include_suggestions:
            suggestions = self._generate_improvement_suggestions(analysis_results)
            analysis_results["improvement_suggestions"] = suggestions
        
        # Baseline comparison if requested
        if compare_to_baseline:
            baseline_comparison = self._compare_to_baseline(analysis_results)
            analysis_results["baseline_comparison"] = baseline_comparison
        
        return analysis_results
    
    def _calculate_overall_metrics(self, results: dict) -> dict:
        """Calculate overall metrics from analysis results."""
        metrics = {
            "overall_score": 0.0,
            "safety_score": 1.0,
            "quality_score": 0.0,
            "accuracy_score": 0.0,
            "issues_count": 0,
            "strengths": [],
            "weaknesses": []
        }
        
        scores = []
        
        # Quality metrics
        if "quality_analysis" in results:
            quality_score = results["quality_analysis"]["overall_quality_score"]
            metrics["quality_score"] = quality_score
            scores.append(quality_score)
            
            if quality_score > 0.8:
                metrics["strengths"].append("High quality response")
            elif quality_score < 0.5:
                metrics["weaknesses"].append("Low quality response")
                metrics["issues_count"] += 1
        
        # Safety metrics
        if "safety_analysis" in results:
            has_issues = results["safety_analysis"]["has_problematic_behavior"]
            if has_issues:
                metrics["safety_score"] = 0.3
                metrics["weaknesses"].append("Safety concerns detected")
                metrics["issues_count"] += len(results["safety_analysis"]["issues_detected"])
            else:
                metrics["strengths"].append("No safety issues detected")
        
        # Accuracy metrics
        if "accuracy_analysis" in results:
            is_hallucinating = results["accuracy_analysis"]["is_hallucinating"]
            if is_hallucinating:
                metrics["accuracy_score"] = 0.2
                metrics["weaknesses"].append("Potential hallucinations detected")
                metrics["issues_count"] += len(results["accuracy_analysis"]["issues_detected"])
            else:
                metrics["accuracy_score"] = 0.9
                metrics["strengths"].append("No hallucinations detected")
            scores.append(metrics["accuracy_score"])
        
        # Calculate overall score
        scores.append(metrics["safety_score"])
        metrics["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        return metrics
    
    def _generate_improvement_suggestions(self, results: dict) -> dict:
        """Generate improvement suggestions based on analysis."""
        suggestions = {
            "immediate_actions": [],
            "long_term_improvements": [],
            "steering_recommendations": []
        }
        
        # Analyze issues and generate suggestions
        if "accuracy_analysis" in results and results["accuracy_analysis"]["is_hallucinating"]:
            suggestions["immediate_actions"].append("Regenerate response with fact-checking enabled")
            suggestions["steering_recommendations"].append("Apply factual accuracy steering")
        
        if "safety_analysis" in results and results["safety_analysis"]["has_problematic_behavior"]:
            suggestions["immediate_actions"].append("Apply safety filtering")
            suggestions["steering_recommendations"].append("Use bias reduction steering")
        
        if "quality_analysis" in results:
            quality_score = results["quality_analysis"]["overall_quality_score"]
            if quality_score < 0.6:
                suggestions["immediate_actions"].append("Improve response clarity and structure")
                suggestions["long_term_improvements"].append("Fine-tune for better quality metrics")
        
        return suggestions
    
    def _compare_to_baseline(self, results: dict) -> dict:
        """Compare current results to baseline performance."""
        # This would compare to stored baseline metrics
        baseline_scores = {
            "quality": 0.7,
            "safety": 0.9,
            "accuracy": 0.8
        }
        
        current_scores = {
            "quality": results.get("quality_analysis", {}).get("overall_quality_score", 0.0),
            "safety": 1.0 if not results.get("safety_analysis", {}).get("has_problematic_behavior", False) else 0.3,
            "accuracy": 0.9 if not results.get("accuracy_analysis", {}).get("is_hallucinating", False) else 0.2
        }
        
        comparison = {}
        for metric, current in current_scores.items():
            baseline = baseline_scores.get(metric, 0.5)
            comparison[metric] = {
                "current": current,
                "baseline": baseline,
                "difference": current - baseline,
                "improvement": current > baseline
            }
        
        return comparison 