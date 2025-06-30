"""
Task Selector for intelligent task selection based on issue types.

This module provides functionality to select the most relevant lm-eval tasks
for training classifiers for specific issue types.
"""

import re
from typing import List, Dict, Any, Set, Tuple
from .task_manager import get_available_tasks


class TaskSelector:
    """Intelligent task selector for issue-type-specific training."""
    
    def __init__(self):
        self._task_concepts_cache = {}
    
    def find_relevant_tasks_for_issue_type(self, issue_type: str, max_tasks: int = 10) -> List[str]:
        """
        Find the most relevant tasks for a specific issue type.
        
        Args:
            issue_type: Type of issue to find tasks for
            max_tasks: Maximum number of tasks to return
            
        Returns:
            List of task names ranked by relevance
        """
        available_tasks = get_available_tasks()
        
        # Calculate relevance scores for all tasks
        task_scores = []
        for task_name in available_tasks:
            score = self._calculate_task_relevance_score(issue_type, task_name)
            if score > 0.0:  # Only include tasks with some relevance
                task_scores.append((task_name, score))
        
        # Sort by relevance score (descending) and return top tasks
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return [task_name for task_name, _ in task_scores[:max_tasks]]
    
    def select_best_tasks_for_training(
        self, 
        issue_type: str, 
        min_tasks: int = 1,
        max_tasks: int = 10,
        quality_threshold: float = 1.5
    ) -> List[str]:
        """
        Select the best tasks for training a classifier for the given issue type.
        
        Args:
            issue_type: Type of issue to select tasks for
            min_tasks: Minimum number of tasks to select
            max_tasks: Maximum number of tasks to select
            quality_threshold: Minimum quality score for task inclusion
            
        Returns:
            List of selected task names
        """
        # Get relevant tasks
        relevant_tasks = self.find_relevant_tasks_for_issue_type(issue_type, max_tasks * 2)
        
        # Calculate combined scores (relevance + quality)
        scored_tasks = []
        for task_name in relevant_tasks:
            relevance_score = self._calculate_task_relevance_score(issue_type, task_name)
            quality_score = self._calculate_task_quality_score(task_name)
            combined_score = relevance_score * 0.7 + quality_score * 0.3  # Weight relevance higher
            
            if quality_score >= quality_threshold:
                scored_tasks.append((task_name, combined_score, relevance_score, quality_score))
        
        # Sort by combined score
        scored_tasks.sort(key=lambda x: x[1], reverse=True)
        
        # Select tasks ensuring we have at least min_tasks
        selected_tasks = []
        
        # First, take the highest scoring tasks
        for task_name, combined_score, rel_score, qual_score in scored_tasks[:max_tasks]:
            selected_tasks.append(task_name)
        
        # If we don't have enough tasks, relax quality threshold
        if len(selected_tasks) < min_tasks:
            print(f"   ⚠️ Only found {len(selected_tasks)} high-quality tasks, relaxing criteria...")
            
            for task_name in relevant_tasks:
                if task_name not in selected_tasks:
                    selected_tasks.append(task_name)
                    if len(selected_tasks) >= min_tasks:
                        break
        
        return selected_tasks[:max_tasks]
    
    def _calculate_task_relevance_score(self, issue_type: str, task_name: str) -> float:
        """Calculate how relevant a task is for the given issue type."""
        task_concepts = self._get_task_concepts(task_name)
        return self._calculate_semantic_similarity(issue_type, task_name, task_concepts)
    
    def _get_task_concepts(self, task_name: str) -> List[str]:
        """Extract concepts from a task name."""
        if task_name in self._task_concepts_cache:
            return self._task_concepts_cache[task_name]
        
        # Extract concepts from task name
        concepts = []
        
        # Split on common separators and extract meaningful tokens
        tokens = re.split(r'[_\-\s\d]+', task_name.lower())
        tokens = [token for token in tokens if len(token) > 2]  # Filter short tokens
        
        for token in tokens:
            # Add domain concepts
            domain_concepts = self._extract_domain_concepts(task_name, [token])
            concepts.extend(domain_concepts)
            
            # Add semantic roots
            semantic_roots = self._extract_semantic_roots(token)
            concepts.extend(semantic_roots)
        
        # Remove duplicates while preserving order
        unique_concepts = []
        seen = set()
        for concept in concepts:
            if concept not in seen:
                unique_concepts.append(concept)
                seen.add(concept)
        
        self._task_concepts_cache[task_name] = unique_concepts
        return unique_concepts
    
    def _extract_domain_concepts(self, task_name: str, tokens: List[str]) -> List[str]:
        """Extract domain-specific concepts from task tokens."""
        concepts = []
        
        # Domain mapping based on common task patterns
        domain_mappings = {
            # Question answering and comprehension
            'qa': ['question', 'answer', 'comprehension', 'understanding'],
            'question': ['inquiry', 'query', 'problem', 'comprehension'],
            'answer': ['response', 'solution', 'reply', 'explanation'],
            'reading': ['comprehension', 'understanding', 'text', 'passage'],
            'comprehension': ['understanding', 'analysis', 'interpretation'],
            
            # Language and communication
            'language': ['communication', 'linguistic', 'text', 'speech'],
            'text': ['passage', 'document', 'content', 'writing'],
            'translation': ['language', 'conversion', 'interpretation'],
            'generation': ['creation', 'production', 'synthesis', 'composition'],
            
            # Logic and reasoning
            'logic': ['reasoning', 'deduction', 'inference', 'analysis'],
            'reasoning': ['thinking', 'logic', 'deduction', 'problem_solving'],
            'inference': ['deduction', 'conclusion', 'reasoning', 'logic'],
            'commonsense': ['knowledge', 'understanding', 'reasoning', 'practical'],
            
            # Knowledge and facts
            'knowledge': ['information', 'facts', 'data', 'learning'],
            'fact': ['information', 'truth', 'data', 'reality'],
            'truth': ['accuracy', 'fact', 'correctness', 'validity'],
            'truthful': ['honest', 'accurate', 'reliable', 'trustworthy'],
            
            # Evaluation and testing
            'eval': ['evaluation', 'assessment', 'testing', 'measurement'],
            'test': ['evaluation', 'assessment', 'examination', 'validation'],
            'benchmark': ['standard', 'evaluation', 'test', 'measurement'],
            
            # Multiple choice and classification
            'mc': ['multiple_choice', 'selection', 'classification', 'categorization'],
            'choice': ['selection', 'option', 'alternative', 'decision'],
            'classification': ['categorization', 'sorting', 'grouping', 'taxonomy'],
            
            # Harmful content and safety
            'harm': ['danger', 'risk', 'safety', 'toxicity'],
            'toxic': ['harmful', 'dangerous', 'poisonous', 'negative'],
            'safe': ['secure', 'protected', 'harmless', 'benign'],
            'bias': ['prejudice', 'unfairness', 'discrimination', 'partiality'],
            
            # Quality and coherence
            'quality': ['excellence', 'standard', 'grade', 'caliber'],
            'coherence': ['consistency', 'logic', 'clarity', 'organization'],
            'clarity': ['clearness', 'precision', 'understanding', 'transparency'],
            
            # Ethics and morality
            'ethics': ['morality', 'principles', 'values', 'conduct'],
            'moral': ['ethical', 'principled', 'virtuous', 'righteous'],
            'honest': ['truthful', 'sincere', 'authentic', 'transparent'],
        }
        
        for token in tokens:
            token_lower = token.lower()
            if token_lower in domain_mappings:
                concepts.extend(domain_mappings[token_lower])
            
            # Add the token itself as a concept
            concepts.append(token_lower)
        
        return concepts
    
    def _extract_semantic_roots(self, word: str) -> List[str]:
        """Extract semantic root words and related concepts."""
        roots = []
        word_lower = word.lower()
        
        # Common semantic relationships
        semantic_mappings = {
            # Truthfulness family
            'truthful': ['truth', 'honest', 'accurate', 'reliable'],
            'truth': ['truthful', 'fact', 'reality', 'accuracy'],
            'honest': ['truthful', 'sincere', 'authentic', 'trustworthy'],
            'accurate': ['correct', 'precise', 'exact', 'truthful'],
            
            # Harmfulness family  
            'harmful': ['harm', 'dangerous', 'toxic', 'detrimental'],
            'harm': ['harmful', 'damage', 'injury', 'detriment'],
            'toxic': ['harmful', 'poisonous', 'dangerous', 'negative'],
            'safe': ['safety', 'secure', 'protected', 'harmless'],
            
            # Quality family
            'quality': ['good', 'excellent', 'standard', 'grade'],
            'good': ['quality', 'excellent', 'positive', 'beneficial'],
            'bad': ['poor', 'negative', 'harmful', 'detrimental'],
            'excellent': ['superior', 'outstanding', 'quality', 'good'],
            
            # Reasoning family
            'reasoning': ['logic', 'thinking', 'analysis', 'inference'],
            'logic': ['reasoning', 'rational', 'logical', 'systematic'],
            'rational': ['reasonable', 'logical', 'sensible', 'sound'],
            'analysis': ['examination', 'study', 'evaluation', 'reasoning'],
            
            # Bias family
            'bias': ['prejudice', 'unfair', 'discrimination', 'partiality'],
            'fair': ['just', 'equitable', 'impartial', 'unbiased'],
            'unfair': ['unjust', 'biased', 'inequitable', 'partial'],
            'neutral': ['impartial', 'unbiased', 'objective', 'balanced'],
        }
        
        if word_lower in semantic_mappings:
            roots.extend(semantic_mappings[word_lower])
        
        # Add morphological variants
        if word_lower.endswith('ful'):
            root = word_lower[:-3]  # truthful -> truth
            roots.append(root)
        elif word_lower.endswith('ness'):
            root = word_lower[:-4]  # goodness -> good  
            roots.append(root)
        elif word_lower.endswith('ing'):
            root = word_lower[:-3]  # reasoning -> reason
            roots.append(root)
        elif word_lower.endswith('ed'):
            root = word_lower[:-2]  # biased -> bias
            roots.append(root)
        
        return roots
    
    def _calculate_semantic_similarity(
        self, 
        issue_type: str, 
        task_name: str, 
        task_concepts: List[str]
    ) -> float:
        """Calculate semantic similarity between issue type and task."""
        if not task_concepts:
            return 0.0
        
        issue_lower = issue_type.lower()
        task_lower = task_name.lower()
        
        score = 0.0
        
        # Direct name matching (highest weight)
        if issue_lower in task_lower or any(issue_lower in concept for concept in task_concepts):
            score += 3.0
        
        # Issue-specific matching patterns
        issue_patterns = {
            'hallucination': ['truth', 'fact', 'accurate', 'reality', 'knowledge', 'qa'],
            'quality': ['good', 'excellent', 'standard', 'evaluation', 'assessment'],
            'harmful': ['toxic', 'safety', 'danger', 'risk', 'ethics', 'moral'],
            'bias': ['fair', 'neutral', 'prejudice', 'discrimination', 'stereotype'],
            'coherence': ['logic', 'reasoning', 'consistent', 'organization', 'structure']
        }
        
        # Check for issue-specific patterns
        if issue_lower in issue_patterns:
            for pattern in issue_patterns[issue_lower]:
                if pattern in task_lower:
                    score += 2.0
                if any(pattern in concept for concept in task_concepts):
                    score += 1.5
        
        # Semantic concept matching
        for concept in task_concepts:
            if self._are_semantically_similar(issue_lower, concept):
                score += 1.0
        
        # Task type bonuses
        if 'mc' in task_lower or 'multiple_choice' in task_lower:
            score += 0.5  # Multiple choice tasks are often good for classification
        
        if 'eval' in task_lower or 'benchmark' in task_lower:
            score += 0.3  # Evaluation tasks tend to be well-curated
        
        return score
    
    def _are_semantically_similar(self, term1: str, term2: str) -> bool:
        """Check if two terms are semantically similar."""
        # Direct match
        if term1 == term2:
            return True
        
        # Substring matching
        if term1 in term2 or term2 in term1:
            return True
        
        # Semantic similarity patterns
        similarity_groups = [
            {'truth', 'truthful', 'honest', 'accurate', 'fact', 'factual'},
            {'harm', 'harmful', 'toxic', 'dangerous', 'unsafe', 'risk'},
            {'quality', 'good', 'excellent', 'superior', 'standard'},
            {'bias', 'biased', 'unfair', 'prejudice', 'discrimination'},
            {'logic', 'logical', 'reasoning', 'rational', 'coherent'},
            {'knowledge', 'information', 'fact', 'data', 'learning'},
            {'question', 'query', 'inquiry', 'ask', 'qa'},
            {'answer', 'response', 'reply', 'solution', 'explanation'},
        ]
        
        for group in similarity_groups:
            if term1 in group and term2 in group:
                return True
        
        return False
    
    def _calculate_task_quality_score(self, task_name: str) -> float:
        """Calculate a quality score for a task based on various indicators."""
        task_lower = task_name.lower()
        score = 1.0  # Base score
        
        # Quality indicators that suggest well-validated tasks
        quality_indicators = [
            # Multiple choice indicators (often well-validated)
            ('mc1', 1.5), ('mc2', 1.5), ('multiple_choice', 1.5),
            # Evaluation methodology indicators
            ('eval', 1.0), ('test', 1.0), ('benchmark', 1.0),
            # Language understanding indicators
            ('language', 1.0), ('understanding', 1.0), ('comprehension', 1.0),
            # Logic and reasoning indicators
            ('logic', 1.0), ('reasoning', 1.0), ('deduction', 1.0),
            # Knowledge assessment indicators
            ('knowledge', 1.0), ('question', 1.0), ('answer', 1.0),
        ]
        
        for indicator, points in quality_indicators:
            if indicator in task_lower:
                score += points
        
        # Penalize very specialized or experimental indicators
        experimental_indicators = [
            'experimental', 'pilot', 'demo', 'sample', 'tiny', 'mini',
            'subset', 'light', 'debug', 'test_only'
        ]
        
        for indicator in experimental_indicators:
            if indicator in task_lower:
                score -= 1.0
        
        # Bonus for domain diversity indicators
        domain_indicators = [
            'multilingual', 'global', 'cross', 'multi', 'diverse'
        ]
        
        for indicator in domain_indicators:
            if indicator in task_lower:
                score += 0.5
        
        return max(0.0, score)  # Ensure non-negative score


# Global instance for convenience
_task_selector = TaskSelector()

def find_relevant_tasks_for_issue_type(issue_type: str, max_tasks: int = 10) -> List[str]:
    """Find the most relevant tasks for a specific issue type."""
    return _task_selector.find_relevant_tasks_for_issue_type(issue_type, max_tasks)

def select_best_tasks_for_training(
    issue_type: str, 
    min_tasks: int = 1,
    max_tasks: int = 10,
    quality_threshold: float = 1.5
) -> List[str]:
    """Select the best tasks for training a classifier for the given issue type."""
    return _task_selector.select_best_tasks_for_training(
        issue_type, min_tasks, max_tasks, quality_threshold
    ) 