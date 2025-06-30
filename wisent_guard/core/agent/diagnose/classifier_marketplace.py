from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json
import pickle
import time
from datetime import datetime
import numpy as np

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
            "./wisent_guard/classifiers/"
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
        
        # Extract issue type
        issue_type = "unknown"
        if 'hallucination' in filename or 'truthful' in filename:
            issue_type = "hallucination"
        elif 'quality' in filename:
            issue_type = "quality"
        elif 'harmful' in filename or 'safety' in filename:
            issue_type = "harmful"
        elif 'bias' in filename:
            issue_type = "bias"
        elif 'coherence' in filename:
            issue_type = "coherence"
        
        return layer, issue_type
    
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
        """Extract model family from model name."""
        if not model_name:
            return "unknown"
        
        model_name = model_name.lower()
        if 'llama' in model_name:
            return 'llama'
        elif 'mistral' in model_name:
            return 'mistral'
        elif 'gemma' in model_name:
            return 'gemma'
        elif 'qwen' in model_name:
            return 'qwen'
        else:
            return 'unknown'
    
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
        """Calculate similarity between issue type and task name."""
        score = 0.0
        
        # Direct matching
        if issue_type in task_name or task_name in issue_type:
            score += 5.0
        
        # Token matching
        issue_tokens = issue_type.replace('_', ' ').replace('-', ' ').split()
        task_tokens = task_name.replace('_', ' ').replace('-', ' ').split()
        
        for issue_token in issue_tokens:
            for task_token in task_tokens:
                if len(issue_token) > 2 and len(task_token) > 2:
                    if issue_token == task_token:
                        score += 3.0
                    elif issue_token in task_token or task_token in issue_token:
                        score += 1.5
                    elif self._are_tokens_similar(issue_token, task_token):
                        score += 0.5
        
        return score
    
    def _are_tokens_similar(self, token1: str, token2: str) -> bool:
        """Check if two tokens are similar using algorithmic methods."""
        if len(token1) < 3 or len(token2) < 3:
            return False
        
        # Character overlap
        overlap = len(set(token1) & set(token2))
        min_len = min(len(token1), len(token2))
        
        # Prefix similarity
        prefix_len = 0
        for i in range(min(len(token1), len(token2))):
            if token1[i] == token2[i]:
                prefix_len += 1
            else:
                break
        
        return (overlap / min_len > 0.6 or 
                prefix_len >= 3 or
                token1[:4] == token2[:4] if len(token1) >= 4 and len(token2) >= 4 else False)
    
    def _estimate_optimal_layer_for_issue(self, issue_type: str) -> int:
        """Estimate optimal layer using algorithmic analysis of issue type."""
        issue_lower = issue_type.lower()
        
        # Start with middle layer as baseline
        base_layer = 14
        
        # Calculate complexity score based on various factors
        complexity_score = 0
        
        # Length-based complexity (longer terms often more abstract)
        length_factor = min(len(issue_lower) / 10.0, 1.0)  # Normalize to 0-1
        complexity_score += length_factor * 3
        
        # Syllable/token complexity (more tokens = more complex)
        tokens = issue_lower.replace('_', ' ').replace('-', ' ').split()
        token_complexity = min(len(tokens) / 3.0, 1.0)  # Normalize to 0-1
        complexity_score += token_complexity * 2
        
        # Character diversity (more diverse = more abstract)
        unique_chars = len(set(issue_lower))
        char_diversity = min(unique_chars / 15.0, 1.0)  # Normalize to 0-1
        complexity_score += char_diversity * 1
        
        # Calculate layer adjustment
        # More complex issues generally need deeper layers
        layer_adjustment = int(complexity_score * 2) - 3  # Range: -3 to +3
        
        # Apply bounds
        estimated_layer = max(8, min(20, base_layer + layer_adjustment))
        
        return estimated_layer
        
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
        # This is a simple heuristic - could be improved with actual benchmarking
        try:
            import torch
            if torch.cuda.is_available():
                return 0.3  # GPU is ~3x faster
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 0.5  # MPS is ~2x faster
            else:
                return 1.0  # CPU baseline
        except:
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