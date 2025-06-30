"""
Classifier Selection System for Autonomous Agent

This module handles:
- Auto-discovery of existing trained classifiers
- Intelligent selection of classifiers based on task requirements
- Performance-based classifier ranking and filtering
- Model-specific classifier matching
"""

import os
import glob
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from ...model_persistence import ModelPersistence


@dataclass
class ClassifierInfo:
    """Information about a discovered classifier."""
    path: str
    layer: int
    issue_type: str
    threshold: float
    metadata: Dict[str, Any]
    performance_score: float = 0.0


@dataclass
class SelectionCriteria:
    """Criteria for selecting classifiers."""
    required_issue_types: List[str]
    preferred_layers: Optional[List[int]] = None
    min_performance_score: float = 0.0
    max_classifiers: int = 10
    model_name: Optional[str] = None
    task_type: Optional[str] = None


class ClassifierSelector:
    """Intelligent classifier selection system."""
    
    def __init__(self, search_paths: List[str] = None):
        """
        Initialize the classifier selector.
        
        Args:
            search_paths: Directories to search for classifiers. Defaults to common locations.
        """
        self.search_paths = search_paths or [
            "./models",
            "./optimization_results", 
            "./trained_classifiers",
            "./examples/models",
            "."  # Current directory
        ]
        self.discovered_classifiers: List[ClassifierInfo] = []
    
    def discover_classifiers(self) -> List[ClassifierInfo]:
        """
        Auto-discover all available trained classifiers.
        
        Returns:
            List of discovered classifier information
        """
        print("🔍 Discovering available classifiers...")
        
        self.discovered_classifiers = []
        
        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                continue
                
            print(f"   Searching in: {search_path}")
            
            # Search for various classifier file patterns
            patterns = [
                "**/*_classifier.pkl",
                "**/*classifier*.pkl", 
                "**/classifier_layer_*.pkl",
                "**/trained_classifier_*.pkl",
                "**/*_layer_*.pkl"
            ]
            
            for pattern in patterns:
                classifier_files = glob.glob(os.path.join(search_path, pattern), recursive=True)
                
                for filepath in classifier_files:
                    classifier_info = self._analyze_classifier_file(filepath)
                    if classifier_info:
                        self.discovered_classifiers.append(classifier_info)
        
        # Remove duplicates based on path
        unique_classifiers = {}
        for classifier in self.discovered_classifiers:
            unique_classifiers[classifier.path] = classifier
        self.discovered_classifiers = list(unique_classifiers.values())
        
        print(f"   ✅ Discovered {len(self.discovered_classifiers)} classifiers")
        
        # Sort by performance score (highest first)
        self.discovered_classifiers.sort(key=lambda x: x.performance_score, reverse=True)
        
        return self.discovered_classifiers
    
    def _analyze_classifier_file(self, filepath: str) -> Optional[ClassifierInfo]:
        """
        Analyze a classifier file and extract information.
        
        Args:
            filepath: Path to classifier file
            
        Returns:
            ClassifierInfo if valid, None otherwise
        """
        try:
            # Extract layer and issue type from filename
            layer, issue_type = self._parse_classifier_filename(filepath)
            
            # Load metadata if available
            metadata = self._load_classifier_metadata(filepath)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(metadata)
            
            # Determine threshold
            threshold = metadata.get('detection_threshold', 0.5)
            
            return ClassifierInfo(
                path=filepath,
                layer=layer,
                issue_type=issue_type,
                threshold=threshold,
                metadata=metadata,
                performance_score=performance_score
            )
            
        except Exception as e:
            print(f"      ⚠️ Failed to analyze {filepath}: {e}")
            return None
    
    def _parse_classifier_filename(self, filepath: str) -> Tuple[int, str]:
        """
        Parse classifier filename to extract layer and issue type.
        
        Args:
            filepath: Path to classifier file
            
        Returns:
            Tuple of (layer, issue_type)
        """
        filename = os.path.basename(filepath)
        
        # Pattern: classifier_layer_X_*.pkl
        if "classifier_layer_" in filename:
            parts = filename.split("_")
            layer_idx = parts.index("layer") + 1 if "layer" in parts else 2
            if layer_idx < len(parts):
                layer = int(parts[layer_idx])
                issue_type = "_".join(parts[:parts.index("layer")])
                return layer, issue_type
        
        # Pattern: trained_classifier_*_layer_X.pkl
        elif "trained_classifier_" in filename and "_layer_" in filename:
            layer_part = filename.split("_layer_")[-1]
            layer = int(layer_part.split(".")[0])
            issue_type = filename.split("trained_classifier_")[1].split("_layer_")[0]
            return layer, issue_type
        
        # Pattern: issue_type_classifier.pkl or issue_type_model_classifier.pkl
        elif "_classifier" in filename:
            parts = filename.replace("_classifier.pkl", "").split("_")
            # Default layer if not specified
            layer = 15  
            issue_type = "_".join(parts[:-1]) if len(parts) > 1 else parts[0]
            return layer, issue_type
        
        # Fallback: extract from path structure
        else:
            path_parts = Path(filepath).parts
            layer = 15  # Default
            issue_type = "unknown"
            
            # Look for layer information in path
            for part in path_parts:
                if "layer" in part.lower():
                    try:
                        layer = int(part.split("_")[-1])
                    except:
                        pass
                        
            return layer, issue_type
    
    def _load_classifier_metadata(self, filepath: str) -> Dict[str, Any]:
        """
        Load classifier metadata if available.
        
        Args:
            filepath: Path to classifier file
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        try:
            # Try to load classifier file to get metadata
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict):
                metadata = data.get('metadata', {})
                
        except Exception as e:
            # Skip corrupted files
            print(f"      ⚠️ Skipping corrupted classifier file {filepath}: {e}")
            pass
        
        # Look for associated metadata files
        metadata_paths = [
            filepath.replace('.pkl', '_metadata.json'),
            filepath.replace('.pkl', '.json'),
            os.path.join(os.path.dirname(filepath), 'metadata.json')
        ]
        
        for metadata_path in metadata_paths:
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        file_metadata = json.load(f)
                        metadata.update(file_metadata)
                        break
                except Exception:
                    continue
        
        return metadata
    
    def _calculate_performance_score(self, metadata: Dict[str, Any]) -> float:
        """
        Calculate a performance score for the classifier.
        
        Args:
            metadata: Classifier metadata
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        score = 0.0
        
        # Base score from F1 or accuracy
        f1_score = metadata.get('f1', metadata.get('training_f1', 0.0))
        accuracy = metadata.get('accuracy', metadata.get('training_accuracy', 0.0))
        
        if f1_score > 0:
            score += f1_score * 0.6
        elif accuracy > 0:
            score += accuracy * 0.4
        
        # Bonus for larger training sets
        training_samples = metadata.get('training_samples', 0)
        if training_samples > 0:
            sample_bonus = min(training_samples / 1000, 0.2)  # Max 0.2 bonus
            score += sample_bonus
        
        # Bonus for recent training
        if 'created_at' in metadata:
            try:
                from datetime import datetime
                created_at = datetime.fromisoformat(metadata['created_at'])
                days_old = (datetime.now() - created_at).days
                if days_old < 30:  # Recent training
                    score += 0.1
            except:
                pass
        
        return min(score, 1.0)  # Cap at 1.0
    
    def select_classifiers(self, criteria: SelectionCriteria) -> List[Dict[str, Any]]:
        """
        Select the best classifiers based on criteria.
        
        Args:
            criteria: Selection criteria
            
        Returns:
            List of classifier configurations ready for use
        """
        print(f"🎯 Selecting classifiers for: {criteria.required_issue_types}")
        
        # Ensure we've discovered classifiers
        if not self.discovered_classifiers:
            self.discover_classifiers()
        
        selected_classifiers = []
        
        # For each required issue type, find the best classifier
        for issue_type in criteria.required_issue_types:
            best_classifier = self._find_best_classifier_for_issue_type(
                issue_type, criteria
            )
            
            if best_classifier:
                config = {
                    "path": best_classifier.path,
                    "layer": best_classifier.layer,
                    "issue_type": best_classifier.issue_type,
                    "threshold": best_classifier.threshold
                }
                selected_classifiers.append(config)
                print(f"   ✅ Selected for {issue_type}: {os.path.basename(best_classifier.path)} "
                      f"(layer {best_classifier.layer}, score: {best_classifier.performance_score:.3f})")
            else:
                print(f"   ❌ No classifier found for {issue_type}")
                raise ValueError(f"No suitable classifier found for issue type: {issue_type}")
        
        # Add additional high-performing classifiers if space allows
        if len(selected_classifiers) < criteria.max_classifiers:
            self._add_supplementary_classifiers(selected_classifiers, criteria)
        
        print(f"   📊 Final selection: {len(selected_classifiers)} classifiers")
        return selected_classifiers
    
    def _find_best_classifier_for_issue_type(
        self, 
        issue_type: str, 
        criteria: SelectionCriteria
    ) -> Optional[ClassifierInfo]:
        """
        Find the best classifier for a specific issue type.
        
        Args:
            issue_type: The issue type to find a classifier for
            criteria: Selection criteria
            
        Returns:
            Best matching classifier or None
        """
        candidates = []
        
        for classifier in self.discovered_classifiers:
            # Check if it matches the issue type (exact or partial match)
            if (classifier.issue_type == issue_type or 
                issue_type in classifier.issue_type or
                classifier.issue_type in issue_type):
                
                # Check performance threshold
                if classifier.performance_score >= criteria.min_performance_score:
                    
                    # Check layer preferences
                    if (criteria.preferred_layers is None or 
                        classifier.layer in criteria.preferred_layers):
                        
                        # Check model compatibility
                        if self._is_model_compatible(classifier, criteria.model_name):
                            candidates.append(classifier)
        
        # Return the best candidate (highest performance score)
        return max(candidates, key=lambda x: x.performance_score) if candidates else None
    
    def _is_model_compatible(self, classifier: ClassifierInfo, model_name: Optional[str]) -> bool:
        """
        Check if classifier is compatible with the specified model.
        
        Args:
            classifier: Classifier information
            model_name: Target model name
            
        Returns:
            True if compatible
        """
        if not model_name:
            return True
        
        # Check metadata for model compatibility
        classifier_model = classifier.metadata.get('model_name', '')
        
        if not classifier_model:
            return True  # No model info available, assume compatible
        
        # Extract model family (e.g., "llama", "mistral")
        target_family = self._extract_model_family(model_name)
        classifier_family = self._extract_model_family(classifier_model)
        
        return target_family == classifier_family
    
    def _extract_model_family(self, model_name: str) -> str:
        """Extract model family from model name."""
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
    
    def _add_supplementary_classifiers(
        self, 
        selected_classifiers: List[Dict[str, Any]], 
        criteria: SelectionCriteria
    ):
        """
        Add supplementary high-performing classifiers if space allows.
        
        Args:
            selected_classifiers: Currently selected classifiers (modified in place)
            criteria: Selection criteria
        """
        selected_paths = {config["path"] for config in selected_classifiers}
        
        for classifier in self.discovered_classifiers:
            if len(selected_classifiers) >= criteria.max_classifiers:
                break
                
            if (classifier.path not in selected_paths and
                classifier.performance_score >= criteria.min_performance_score):
                
                config = {
                    "path": classifier.path,
                    "layer": classifier.layer,
                    "issue_type": classifier.issue_type,
                    "threshold": classifier.threshold
                }
                selected_classifiers.append(config)
                selected_paths.add(classifier.path)
                print(f"   ➕ Added supplementary: {os.path.basename(classifier.path)} "
                      f"({classifier.issue_type}, score: {classifier.performance_score:.3f})")
    
    def get_classifier_summary(self) -> str:
        """
        Get a summary of discovered classifiers.
        
        Returns:
            Formatted summary string
        """
        if not self.discovered_classifiers:
            return "No classifiers discovered yet. Run discover_classifiers() first."
        
        summary = f"\n📊 Classifier Discovery Summary\n"
        summary += f"{'='*50}\n"
        summary += f"Total Classifiers: {len(self.discovered_classifiers)}\n\n"
        
        # Group by issue type
        by_issue_type = {}
        for classifier in self.discovered_classifiers:
            issue_type = classifier.issue_type
            if issue_type not in by_issue_type:
                by_issue_type[issue_type] = []
            by_issue_type[issue_type].append(classifier)
        
        for issue_type, classifiers in by_issue_type.items():
            summary += f"{issue_type.upper()}: {len(classifiers)} classifiers\n"
            for classifier in sorted(classifiers, key=lambda x: x.performance_score, reverse=True)[:3]:
                summary += f"  • {os.path.basename(classifier.path)} "
                summary += f"(layer {classifier.layer}, score: {classifier.performance_score:.3f})\n"
            summary += "\n"
        
        return summary


def auto_select_classifiers_for_agent(
    model_name: str,
    required_issue_types: List[str] = None,
    search_paths: List[str] = None,
    max_classifiers: int = 5,
    min_performance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Auto-select the best classifiers for an autonomous agent.
    
    Args:
        model_name: Name of the model being used
        required_issue_types: List of required issue types to detect
        search_paths: Custom search paths for classifiers
        max_classifiers: Maximum number of classifiers to select
        min_performance: Minimum performance score required
        
    Returns:
        List of classifier configurations ready for use
    """
    # Default issue types for comprehensive analysis
    if required_issue_types is None:
        required_issue_types = [
            "hallucination",
            "quality", 
            "harmful",
            "bias"
        ]
    
    selector = ClassifierSelector(search_paths)
    
    criteria = SelectionCriteria(
        required_issue_types=required_issue_types,
        max_classifiers=max_classifiers,
        min_performance_score=min_performance,
        model_name=model_name
    )
    
    return selector.select_classifiers(criteria)
