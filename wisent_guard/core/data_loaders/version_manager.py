"""
Version manager for LiveCodeBench datasets to handle version differences.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from .livecodebench_loader import LiveCodeBenchLoader, LiveCodeBenchProblem

logger = logging.getLogger(__name__)


@dataclass
class VersionDifference:
    """Represents the difference between two versions."""
    
    base_version: str
    target_version: str
    added_problems: List[LiveCodeBenchProblem]
    removed_problems: List[LiveCodeBenchProblem]
    common_problems: List[LiveCodeBenchProblem]


class LiveCodeBenchVersionManager:
    """Manages LiveCodeBench version differences and splits."""
    
    def __init__(self):
        self.loader = LiveCodeBenchLoader()
        self._version_cache = {}
    
    def get_version_difference(
        self,
        base_version: str,
        target_version: str,
        use_cache: bool = True
    ) -> VersionDifference:
        """
        Get the difference between two versions.
        
        Args:
            base_version: Base version (e.g., "release_v1")
            target_version: Target version (e.g., "release_v2")
            use_cache: Whether to use cached results
            
        Returns:
            VersionDifference object
        """
        cache_key = f"{base_version}_{target_version}"
        
        if use_cache and cache_key in self._version_cache:
            logger.info(f"Using cached version difference for {cache_key}")
            return self._version_cache[cache_key]
        
        logger.info(f"Computing version difference: {base_version} -> {target_version}")
        
        # Load both versions
        base_problems = self.loader.load_problems(base_version)
        target_problems = self.loader.load_problems(target_version)
        
        # Create ID mappings
        base_ids = {p.question_id: p for p in base_problems}
        target_ids = {p.question_id: p for p in target_problems}
        
        # Find differences
        base_id_set = set(base_ids.keys())
        target_id_set = set(target_ids.keys())
        
        added_ids = target_id_set - base_id_set
        removed_ids = base_id_set - target_id_set
        common_ids = base_id_set & target_id_set
        
        # Create problem lists
        added_problems = [target_ids[pid] for pid in added_ids]
        removed_problems = [base_ids[pid] for pid in removed_ids]
        common_problems = [base_ids[pid] for pid in common_ids]
        
        logger.info(f"Version difference analysis:")
        logger.info(f"  {base_version}: {len(base_problems)} problems")
        logger.info(f"  {target_version}: {len(target_problems)} problems")
        logger.info(f"  Added in {target_version}: {len(added_problems)} problems")
        logger.info(f"  Removed from {base_version}: {len(removed_problems)} problems")
        logger.info(f"  Common problems: {len(common_problems)} problems")
        
        difference = VersionDifference(
            base_version=base_version,
            target_version=target_version,
            added_problems=added_problems,
            removed_problems=removed_problems,
            common_problems=common_problems
        )
        
        if use_cache:
            self._version_cache[cache_key] = difference
        
        return difference
    
    def get_version_split(
        self,
        train_version: str,
        eval_version: str,
        split_type: str = "new_only"
    ) -> Dict[str, List[LiveCodeBenchProblem]]:
        """
        Get version split for training and evaluation.
        
        Args:
            train_version: Version to use for training
            eval_version: Version to use for evaluation
            split_type: Type of split ("new_only", "common_only", "all")
            
        Returns:
            Dictionary with "train" and "eval" problem lists
        """
        difference = self.get_version_difference(train_version, eval_version)
        
        if split_type == "new_only":
            # Train on base version, evaluate on problems added in target version
            train_problems = self.loader.load_problems(train_version)
            eval_problems = difference.added_problems
            
        elif split_type == "common_only":
            # Train and evaluate on common problems (for consistency checks)
            train_problems = difference.common_problems
            eval_problems = difference.common_problems
            
        elif split_type == "all":
            # Train on base version, evaluate on all problems in target version
            train_problems = self.loader.load_problems(train_version)
            eval_problems = self.loader.load_problems(eval_version)
            
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        
        logger.info(f"Version split ({split_type}):")
        logger.info(f"  Train problems: {len(train_problems)}")
        logger.info(f"  Eval problems: {len(eval_problems)}")
        
        return {
            "train": train_problems,
            "eval": eval_problems
        }
    
    def get_recommended_splits(self) -> List[Dict[str, Any]]:
        """Get recommended version splits for experiments."""
        return [
            {
                "name": "v1_to_v2_new",
                "description": "Train on v1, evaluate on new problems in v2",
                "train_version": "release_v1",
                "eval_version": "release_v2",
                "split_type": "new_only",
                "contamination_free": True
            },
            {
                "name": "v2_to_v3_new",
                "description": "Train on v2, evaluate on new problems in v3",
                "train_version": "release_v2",
                "eval_version": "release_v3",
                "split_type": "new_only",
                "contamination_free": True
            },
            {
                "name": "v1_to_v2_all",
                "description": "Train on v1, evaluate on all problems in v2",
                "train_version": "release_v1",
                "eval_version": "release_v2",
                "split_type": "all",
                "contamination_free": False
            },
            {
                "name": "v1_consistency",
                "description": "Train and evaluate on v1 (consistency check)",
                "train_version": "release_v1",
                "eval_version": "release_v1",
                "split_type": "common_only",
                "contamination_free": False
            }
        ]
    
    def analyze_version_progression(self) -> Dict[str, Any]:
        """Analyze how versions progress over time."""
        versions = ["release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6"]
        analysis = {"versions": [], "progression": []}
        
        for version in versions:
            try:
                info = self.loader.get_version_info(version)
                analysis["versions"].append({
                    "version": version,
                    "problems": info.get("problems", 0),
                    "date_range": info.get("date_range", "Unknown")
                })
            except Exception as e:
                logger.warning(f"Could not analyze {version}: {e}")
        
        # Calculate progression
        for i in range(1, len(analysis["versions"])):
            prev_version = analysis["versions"][i-1]
            curr_version = analysis["versions"][i]
            
            added_problems = curr_version["problems"] - prev_version["problems"]
            analysis["progression"].append({
                "from_version": prev_version["version"],
                "to_version": curr_version["version"],
                "added_problems": added_problems,
                "growth_rate": added_problems / prev_version["problems"] if prev_version["problems"] > 0 else 0
            })
        
        return analysis
    
    def clear_cache(self):
        """Clear version cache."""
        self._version_cache.clear()
        logger.info("Version cache cleared")