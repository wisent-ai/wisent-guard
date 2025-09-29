from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json
import pickle
import time
from datetime import datetime
import numpy as np

from wisent_guard.core.utils.device import resolve_default_device

@dataclass
class ClassifierListing:
    """A classifier available in the marketplace."""
    path: str
    layer: int
    issue_type: str
    threshold: float
    quality_score: float  # 0.0 to 1.0, higher is better
    training_samples: int
    model_family: str
    created_at: str
    training_time_seconds: float
    metadata: Dict[str, Any]
    
    def to_config(self) -> Dict[str, Any]:
        """Convert to classifier config format."""
        return {
            "path": self.path,
            "layer": self.layer,
            "issue_type": self.issue_type,
            "threshold": self.threshold
        }

@dataclass
class ClassifierCreationEstimate:
    """Estimate for creating a new classifier."""
    issue_type: str
    estimated_training_time_minutes: float
    estimated_quality_score: float  # Predicted based on issue type complexity
    training_samples_needed: int
    optimal_layer: int
    confidence: float  # How confident we are in the estimate

class ClassifierMarketplace:
    """
    A marketplace interface for classifiers that gives the agent full autonomy
    to discover, evaluate, and create classifiers based on its needs.
    """
    
    def __init__(self, model, search_paths: List[str] = None):
        self.model = model
        self.search_paths = search_paths or [
            "./models/",
            "./classifiers/", 
            "./wisent_guard/models/",
            "./wisent_guard/classifiers/",
            "./wisent_guard/core/classifiers/"
        ]
        self.available_classifiers: List[ClassifierListing] = []
        self._training_time_cache = {}
        
    def discover_available_classifiers(self) -> List[ClassifierListing]:
        """
        Discover all available classifiers and return them as marketplace listings.
        
        Returns:
            List of classifier listings with quality scores and metadata
        """
        print("🏪 Discovering available classifiers in marketplace...")
        
        self.available_classifiers = []
        
        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                continue
                
            # For wisent_guard/core/classifiers, search recursively for the nested structure
            if "wisent_guard/core/classifiers" in search_path:
                import glob
                pattern = os.path.join(search_path, "**", "*.pkl")
                classifier_files = glob.glob(pattern, recursive=True)
                for filepath in classifier_files:
                    listing = self._create_classifier_listing(filepath)
                    if listing:
                        self.available_classifiers.append(listing)
            else:
                # Original behavior for other directories
                for filename in os.listdir(search_path):
                    if filename.endswith('.pkl'):
                        filepath = os.path.join(search_path, filename)
                        listing = self._create_classifier_listing(filepath)
                        if listing:
                            self.available_classifiers.append(listing)
        
        # Sort by quality score (best first)
        self.available_classifiers.sort(key=lambda x: x.quality_score, reverse=True)
        
        print(f"   📊 Found {len(self.available_classifiers)} classifiers in marketplace")
        return self.available_classifiers
    
    def _create_classifier_listing(self, filepath: str) -> Optional[ClassifierListing]:
        """Create a marketplace listing for a classifier file."""
        try:
            # Load metadata
            metadata = self._load_metadata(filepath)
            
            # Parse filename for layer and issue type
            layer, issue_type = self._parse_filename(filepath)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(metadata)
            
            # Extract other info
            threshold = metadata.get('threshold', 0.5)
            training_samples = metadata.get('training_samples', 0)
            model_family = self._extract_model_family(metadata.get('model_name', ''))
            created_at = metadata.get('created_at', datetime.now().isoformat())
            training_time = metadata.get('training_time_seconds', 0.0)
            
            return ClassifierListing(
                path=filepath,
                layer=layer,
                issue_type=issue_type,
                threshold=threshold,
                quality_score=quality_score,
                training_samples=training_samples,
                model_family=model_family,
                created_at=created_at,
                training_time_seconds=training_time,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"   ⚠️ Could not create listing for {filepath}: {e}")
            return None
    
    def _load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Load metadata for a classifier."""
        # Try to load companion JSON file first
        json_path = filepath.replace('.pkl', '.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Try to load metadata from the pickle file itself
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, dict) and 'metadata' in data:
                    return data['metadata']
                elif hasattr(data, 'metadata'):
                    return data.metadata
        except:
            pass
        
        return {}
    
    def _parse_filename(self, filepath: str) -> Tuple[int, str]:
        """Parse layer and issue type from filename."""
        filename = os.path.basename(filepath).lower()
        
        # Check if this is from wisent_guard/core/classifiers with nested structure
        if "wisent_guard/core/classifiers" in filepath:
            # Extract from path structure: wisent_guard/core/classifiers/{model}/{benchmark}/layer_{layer}.pkl
            path_parts = filepath.split(os.sep)
            
            # Find the benchmark name (second to last directory)
            if len(path_parts) >= 2:
                benchmark_name = path_parts[-2]  # Directory containing the classifier file
                
                # Extract layer from filename like "layer_15.pkl"
                import re
                layer_match = re.search(r'layer_(\d+)\.pkl', filename)
                layer = int(layer_match.group(1)) if layer_match else 15
                
                # Use benchmark name as issue type for generated classifiers
                issue_type = f"quality_{benchmark_name}"
                
                return layer, issue_type
        
        # Original parsing logic for other classifiers
        filename = os.path.basename(filepath).lower()
        
        # Extract layer
        layer = 15  # default
        for part in filename.replace('_', ' ').replace('-', ' ').split():
            if part.startswith('l') and part[1:].isdigit():
                layer = int(part[1:])
                break
            elif part.startswith('layer') and len(part) > 5:
                try:
                    layer = int(part[5:])
                    break
                except:
                    pass
            elif 'layer' in filename:
                import re
                match = re.search(r'layer[_\s]*(\d+)', filename)
                if match:
                    layer = int(match.group(1))
                    break
        
        # Extract issue type using model
        issue_type = self._get_model_issue_type(filename)
        
        return layer, issue_type
    
    def _get_model_issue_type(self, filename: str) -> str:
        """Extract issue type from filename using model decisions."""
        prompt = f"""What AI safety issue type is this classifier filename related to?

Filename: {filename}

Common issue types include:
- hallucination (false information, factual errors) 
- quality (output quality, coherence)
- harmful (toxic content, safety violations)
- bias (unfairness, discrimination)
- coherence (logical consistency)

Respond with just the issue type (one word):"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=15, temperature=0.1)
            issue_type = response.strip().lower()
            
            # Clean up response to single word
            import re
            match = re.search(r'(hallucination|quality|harmful|bias|coherence|unknown)', issue_type)
            if match:
                return match.group(1)
            return "unknown"
        except:
            return "unknown"
    
    def _calculate_quality_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate a comprehensive quality score for the classifier."""
        score = 0.0
        
        # Primary performance metrics (70% of score)
        f1_score = metadata.get('f1', metadata.get('training_f1', 0.0))
        accuracy = metadata.get('accuracy', metadata.get('training_accuracy', 0.0))
        
        if f1_score > 0:
            score += f1_score * 0.5
        if accuracy > 0:
            score += accuracy * 0.2
        
        # Training data quality (20% of score)
        training_samples = metadata.get('training_samples', 0)
        if training_samples > 0:
            data_quality = min(training_samples / 1000, 1.0) * 0.2
            score += data_quality
        
        # Recency bonus (10% of score)
        try:
            created_at = datetime.fromisoformat(metadata.get('created_at', ''))
            days_old = (datetime.now() - created_at).days
            recency_score = max(0, (90 - days_old) / 90) * 0.1  # Decays over 90 days
            score += recency_score
        except:
            pass
        
        return min(score, 1.0)
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name using model decisions."""
        if not model_name:
            return "unknown"
        
        prompt = f"""What model family is this model name from?

Model name: {model_name}

Common families include: llama, mistral, gemma, qwen, gpt, claude, other

Respond with just the family name (one word):"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            family = response.strip().lower()
            
            # Clean up response
            import re
            match = re.search(r'(llama|mistral|gemma|qwen|gpt|claude|other|unknown)', family)
            if match:
                return match.group(1)
            return "unknown"
        except:
            return "unknown"
    
    def get_creation_estimate(self, issue_type: str) -> ClassifierCreationEstimate:
        """
        Get an estimate for creating a new classifier for the given issue type.
        
        Args:
            issue_type: The type of issue to create a classifier for
            
        Returns:
            Estimate including time, quality, and confidence
        """
        # Dynamic estimates based on available benchmark data
        # Check if we have relevant benchmarks for this issue type
        available_benchmarks = self._find_available_benchmarks_for_issue(issue_type)
        
        if available_benchmarks:
            # We have relevant benchmark data - better quality expected
            benchmark_count = len(available_benchmarks)
            base = {
                "training_time_minutes": 8.0 + (benchmark_count * 2.0),  # More benchmarks = more time
                "quality_score": min(0.80, 0.60 + (benchmark_count * 0.05)),  # Better with more data
                "samples_needed": min(500, 100 + (benchmark_count * 30)),  # Scale with available data
                "optimal_layer": self._estimate_optimal_layer_for_issue(issue_type)
            }
            print(f"   📊 Using {benchmark_count} benchmarks for {issue_type}")
        else:
            # Fall back to synthetic generation
            base = {
                "training_time_minutes": 6.0,  # Synthetic is faster but less data
                "quality_score": 0.55,  # Lower expectation for synthetic
                "samples_needed": 50,  # Fewer samples for synthetic
                "optimal_layer": 14  # General-purpose layer
            }
            print(f"   🤖 Using synthetic generation for {issue_type}")
        
        return self._complete_creation_estimate(base, available_benchmarks, issue_type)
    
    def _find_available_benchmarks_for_issue(self, issue_type: str) -> List[str]:
        """Find available benchmarks using dynamic semantic analysis."""
        available_tasks = self.model.get_available_tasks()
        
        # Use semantic similarity to find relevant benchmarks
        relevant = []
        issue_lower = issue_type.lower()
        
        for task in available_tasks[:1000]:  # Limit search for speed
            task_lower = task.lower()
            
            # Calculate semantic similarity score
            similarity_score = self._calculate_task_similarity(issue_lower, task_lower)
            
            if similarity_score > 0:
                relevant.append((task, similarity_score))
                if len(relevant) >= 30:  # Get more candidates for ranking
                    break
        
        # Sort by similarity score and return top matches
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [task for task, score in relevant[:15]]  # Return top 15
    
    def _calculate_task_similarity(self, issue_type: str, task_name: str) -> float:
        """Calculate similarity between issue type and task name using model decisions."""
        prompt = f"""Rate the similarity between this issue type and evaluation task for training AI safety classifiers.

Issue Type: {issue_type}
Task: {task_name}

Rate similarity from 0.0 to 10.0 (10.0 = highly similar, 0.0 = not similar).
Respond with only the number:"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            score_str = response.strip()
            
            import re
            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
            return 0.0
        except:
            return 0.0
    

    
    def _estimate_optimal_layer_for_issue(self, issue_type: str) -> int:
        """Estimate optimal layer using model analysis of issue complexity."""
        prompt = f"""What transformer layer would be optimal for detecting this AI safety issue?

Issue Type: {issue_type}

Consider:
- Simple issues (formatting, basic patterns) → early layers (8-12)
- Complex semantic issues (truthfulness, bias) → middle layers (12-16)  
- Abstract conceptual issues (coherence, quality) → deeper layers (16-20)

Respond with just the layer number (8-20):"""
        
        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            layer_str = response.strip()
            
            import re
            match = re.search(r'(\d+)', layer_str)
            if match:
                layer = int(match.group(1))
                return max(8, min(20, layer))  # Clamp to valid range
            return 14  # Default middle layer
        except:
            return 14
        
    def _complete_creation_estimate(self, base: Dict[str, Any], available_benchmarks: List[str], issue_type: str) -> ClassifierCreationEstimate:
        """Complete the creation estimate with hardware adjustments."""
        # Adjust based on model and hardware
        hardware_multiplier = self._estimate_hardware_speed()
        training_time = base["training_time_minutes"] * hardware_multiplier
        
        # Confidence based on data availability
        confidence = 0.8 if available_benchmarks else 0.6  # Higher confidence with benchmark data
        
        return ClassifierCreationEstimate(
            issue_type=issue_type,
            estimated_training_time_minutes=training_time,
            estimated_quality_score=base["quality_score"],
            training_samples_needed=base["samples_needed"],
            optimal_layer=base["optimal_layer"],
            confidence=confidence
        )
    
    def _estimate_hardware_speed(self) -> float:
        """Estimate hardware speed multiplier for training time."""
        device_kind = resolve_default_device()
        if device_kind == "cuda":
            return 0.3
        if device_kind == "mps":
            return 0.5
        return 1.0
    
    def get_marketplace_summary(self) -> str:
        """Get a summary of the classifier marketplace."""
        if not self.available_classifiers:
            self.discover_available_classifiers()
        
        if not self.available_classifiers:
            return "🏪 Classifier Marketplace: No classifiers available"
        
        summary = f"\n🏪 Classifier Marketplace Summary\n"
        summary += f"{'='*50}\n"
        summary += f"Available Classifiers: {len(self.available_classifiers)}\n\n"
        
        # Group by issue type
        by_issue_type = {}
        for classifier in self.available_classifiers:
            issue_type = classifier.issue_type
            if issue_type not in by_issue_type:
                by_issue_type[issue_type] = []
            by_issue_type[issue_type].append(classifier)
        
        for issue_type, classifiers in by_issue_type.items():
            best_classifier = max(classifiers, key=lambda x: x.quality_score)
            summary += f"📊 {issue_type.upper()}: {len(classifiers)} available\n"
            summary += f"   Best: {os.path.basename(best_classifier.path)} "
            summary += f"(Quality: {best_classifier.quality_score:.3f}, Layer: {best_classifier.layer})\n"
            summary += f"   Samples: {best_classifier.training_samples}, "
            summary += f"Model: {best_classifier.model_family}\n\n"
        
        return summary
    
    def filter_classifiers(self, 
                          issue_types: List[str] = None,
                          min_quality: float = 0.0,
                          model_family: str = None,
                          layers: List[int] = None) -> List[ClassifierListing]:
        """
        Filter available classifiers by criteria.
        
        Args:
            issue_types: List of issue types to include
            min_quality: Minimum quality score
            model_family: Required model family
            layers: Allowed layers
            
        Returns:
            Filtered list of classifier listings
        """
        filtered = self.available_classifiers
        
        if issue_types:
            filtered = [c for c in filtered if c.issue_type in issue_types]
        
        if min_quality > 0:
            filtered = [c for c in filtered if c.quality_score >= min_quality]
        
        if model_family:
            filtered = [c for c in filtered if c.model_family == model_family]
        
        if layers:
            filtered = [c for c in filtered if c.layer in layers]
        
        return filtered
    
    async def create_classifier_on_demand(self, 
                                        issue_type: str, 
                                        custom_layer: int = None) -> ClassifierListing:
        """
        Create a new classifier on demand.
        
        Args:
            issue_type: Type of issue to create classifier for
            custom_layer: Optional custom layer (otherwise uses optimal)
            
        Returns:
            Newly created classifier listing
        """
        from .create_classifier import create_classifier_on_demand
        
        print(f"🏗️ Creating new classifier for {issue_type}...")
        
        # Get creation estimate
        estimate = self.get_creation_estimate(issue_type)
        layer = custom_layer or estimate.optimal_layer
        
        # Create save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/agent_created_{issue_type}_layer{layer}_{timestamp}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create classifier
        start_time = time.time()
        result = create_classifier_on_demand(
            model=self.model,
            issue_type=issue_type,
            layer=layer,
            save_path=save_path,
            optimize=True
        )
        training_time = time.time() - start_time
        
        # Create listing for the new classifier
        listing = ClassifierListing(
            path=result.save_path,
            layer=result.config.layer,
            issue_type=issue_type,
            threshold=result.config.threshold,
            quality_score=result.performance_metrics.get('f1', 0.0),
            training_samples=result.performance_metrics.get('training_samples', 0),
            model_family=self._extract_model_family(self.model.model_name),
            created_at=datetime.now().isoformat(),
            training_time_seconds=training_time,
            metadata=result.performance_metrics
        )
        
        # Add to available classifiers
        self.available_classifiers.append(listing)
        self.available_classifiers.sort(key=lambda x: x.quality_score, reverse=True)
        
        print(f"   ✅ Created classifier in {training_time/60:.1f} minutes")
        print(f"   📊 Quality score: {listing.quality_score:.3f}")
        
        return listing 