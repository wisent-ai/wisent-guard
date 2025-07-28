"""
Question bank management for synthetic contrastive pair generation.
Stores and manages a reusable pool of questions to avoid regeneration.
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
import os


class QuestionBank:
    """Manages a persistent bank of questions for synthetic pair generation."""
    
    def __init__(self, bank_path: Optional[str] = None):
        """
        Initialize the question bank.
        
        Args:
            bank_path: Path to the question bank JSON file. 
                      If None, uses ~/.wisent-guard/question_bank.json
        """
        if bank_path is None:
            # Default to user's home directory
            self.bank_dir = Path.home() / ".wisent-guard"
            self.bank_dir.mkdir(exist_ok=True)
            self.bank_path = self.bank_dir / "question_bank.json"
        else:
            self.bank_path = Path(bank_path)
            self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.bank_data = self._load_bank()
    
    def _load_bank(self) -> Dict:
        """Load the question bank from disk."""
        if self.bank_path.exists():
            try:
                with open(self.bank_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading question bank: {e}. Starting fresh.")
                return self._create_empty_bank()
        else:
            return self._create_empty_bank()
    
    def _create_empty_bank(self) -> Dict:
        """Create an empty question bank structure."""
        return {
            "questions": [],
            "metadata": {
                "total_count": 0,
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "usage_tracking": {}  # question -> usage info
        }
    
    def _save_bank(self) -> None:
        """Save the question bank to disk."""
        self.bank_data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.bank_path, 'w', encoding='utf-8') as f:
            json.dump(self.bank_data, f, indent=2, ensure_ascii=False)
    
    def add_questions(self, questions: List[str]) -> int:
        """
        Add new questions to the bank.
        
        Args:
            questions: List of question strings to add
            
        Returns:
            Number of new questions actually added (excludes duplicates)
        """
        existing_questions = set(self.bank_data["questions"])
        new_questions = []
        
        for question in questions:
            if question not in existing_questions:
                new_questions.append(question)
                self.bank_data["questions"].append(question)
                # Initialize usage tracking
                self.bank_data["usage_tracking"][question] = {
                    "used_for_traits": [],
                    "usage_count": 0,
                    "added_at": datetime.now().isoformat()
                }
        
        self.bank_data["metadata"]["total_count"] = len(self.bank_data["questions"])
        
        if new_questions:
            self._save_bank()
            print(f"📝 Added {len(new_questions)} new questions to bank (total: {self.bank_data['metadata']['total_count']})")
        
        return len(new_questions)
    
    def get_questions(self, count: int, trait: Optional[str] = None, 
                     prefer_unused: bool = True) -> List[str]:
        """
        Get questions from the bank.
        
        Args:
            count: Number of questions to retrieve
            trait: Optional trait name for usage tracking
            prefer_unused: If True, prioritize questions not used for this trait
            
        Returns:
            List of question strings
        """
        available_questions = self.bank_data["questions"].copy()
        
        if not available_questions:
            return []
        
        if prefer_unused and trait:
            # Sort by usage - prefer questions not used for this trait
            def usage_key(q):
                usage = self.bank_data["usage_tracking"].get(q, {})
                trait_used = trait in usage.get("used_for_traits", [])
                usage_count = usage.get("usage_count", 0)
                return (trait_used, usage_count)  # False sorts before True
            
            available_questions.sort(key=usage_key)
        else:
            # Random shuffle if no preference
            random.shuffle(available_questions)
        
        # Get requested number of questions
        selected = available_questions[:count]
        
        # Update usage tracking if trait provided
        if trait:
            for question in selected:
                if question in self.bank_data["usage_tracking"]:
                    usage = self.bank_data["usage_tracking"][question]
                    if trait not in usage["used_for_traits"]:
                        usage["used_for_traits"].append(trait)
                    usage["usage_count"] += 1
                    usage["last_used"] = datetime.now().isoformat()
            
            self._save_bank()
        
        return selected
    
    def get_available_count(self) -> int:
        """Get the total number of questions in the bank."""
        return len(self.bank_data["questions"])
    
    def get_unused_count(self, trait: str) -> int:
        """Get the number of questions not yet used for a specific trait."""
        unused = 0
        for question in self.bank_data["questions"]:
            usage = self.bank_data["usage_tracking"].get(question, {})
            if trait not in usage.get("used_for_traits", []):
                unused += 1
        return unused
    
    def clear_bank(self) -> None:
        """Clear all questions from the bank."""
        self.bank_data = self._create_empty_bank()
        self._save_bank()
        print("🗑️ Question bank cleared")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the question bank."""
        total = len(self.bank_data["questions"])
        usage_stats = {}
        trait_counts = {}
        
        for question, usage in self.bank_data["usage_tracking"].items():
            for trait in usage.get("used_for_traits", []):
                trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
        never_used = sum(1 for q in self.bank_data["questions"] 
                        if q not in self.bank_data["usage_tracking"] or 
                        self.bank_data["usage_tracking"][q]["usage_count"] == 0)
        
        return {
            "total_questions": total,
            "never_used": never_used,
            "traits_covered": list(trait_counts.keys()),
            "questions_per_trait": trait_counts,
            "bank_location": str(self.bank_path),
            "created_at": self.bank_data["metadata"].get("created_at"),
            "last_updated": self.bank_data["metadata"].get("last_updated")
        }