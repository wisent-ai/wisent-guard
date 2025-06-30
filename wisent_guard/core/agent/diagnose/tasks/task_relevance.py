"""
Task Relevance Selection for Wisent Guard.

This module provides functionality to select the most relevant tasks from the
lm-evaluation-harness library based on a user query or issue type.

No quality scoring - just pure relevance ranking based on semantic similarity.
"""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import math
from .task_manager import get_available_tasks


class TaskRelevanceSelector:
    """Selects tasks based on semantic relevance to a query."""
    
    def __init__(self):
        self._task_cache = {}
        self._idf_cache = {}
        
    def find_relevant_tasks(
        self, 
        query: str, 
        max_results: int = 20,
        min_relevance_score: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Find tasks most relevant to the given query.
        
        Args:
            query: The search query (e.g., "hallucination detection", "bias", "truthfulness")
            max_results: Maximum number of tasks to return
            min_relevance_score: Minimum relevance score threshold (0.0 to 1.0)
            
        Returns:
            List of (task_name, relevance_score) tuples, sorted by relevance
        """
        available_tasks = get_available_tasks()
        
        # Extract query terms
        query_terms = self._extract_query_terms(query)
        
        # Calculate relevance scores for all tasks
        task_scores = []
        for task_name in available_tasks:
            score = self._calculate_relevance_score(query_terms, task_name)
            if score >= min_relevance_score:
                task_scores.append((task_name, score))
        
        # Sort by relevance score (descending)
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return task_scores[:max_results]
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from the query."""
        # Normalize and tokenize
        query_lower = query.lower()
        
        # Split on common separators
        tokens = re.split(r'[^a-zA-Z0-9]+', query_lower)
        
        # Filter out stop words and short tokens
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'through', 'during',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can',
            'a', 'an', 'some', 'any', 'all', 'each', 'every', 'this', 'that', 'these', 'those'
        }
        
        meaningful_terms = []
        for token in tokens:
            if len(token) >= 2 and token not in stop_words and token.isalpha():
                meaningful_terms.append(token)
                
                # Add morphological variants
                meaningful_terms.extend(self._get_morphological_variants(token))
        
        return list(set(meaningful_terms))  # Remove duplicates
    
    def _get_morphological_variants(self, term: str) -> List[str]:
        """Get morphological variants of a term."""
        variants = []
        
        # Add root forms by removing common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'sion', 'ful', 'less']
        for suffix in suffixes:
            if term.endswith(suffix) and len(term) > len(suffix) + 2:
                root = term[:-len(suffix)]
                variants.append(root)
        
        # Add semantic expansions based on the term
        semantic_expansions = {
            'truth': ['truthful', 'honest', 'accurate', 'fact', 'factual', 'reality'],
            'truthful': ['truth', 'honest', 'accurate', 'reliable', 'valid'],
            'honest': ['truthful', 'sincere', 'authentic', 'genuine'],
            'lie': ['false', 'deception', 'misinformation', 'untruth'],
            'false': ['lie', 'incorrect', 'wrong', 'inaccurate', 'deceptive'],
            'hallucination': ['false', 'incorrect', 'fabrication', 'invention', 'delusion'],
            'bias': ['prejudice', 'unfair', 'discrimination', 'partial', 'skewed'],
            'fair': ['unbiased', 'neutral', 'impartial', 'equitable', 'just'],
            'harmful': ['dangerous', 'toxic', 'damaging', 'destructive', 'unsafe'],
            'safe': ['secure', 'protected', 'harmless', 'benign'],
            'quality': ['good', 'excellent', 'standard', 'grade'],
            'coherence': ['consistent', 'logical', 'clear', 'organized'],
            'reasoning': ['logic', 'thinking', 'analysis', 'inference'],
            'knowledge': ['information', 'facts', 'data', 'learning'],
        }
        
        if term in semantic_expansions:
            variants.extend(semantic_expansions[term])
        
        return variants
    
    def _calculate_relevance_score(self, query_terms: List[str], task_name: str) -> float:
        """Calculate relevance score between query terms and task name."""
        # Extract task components
        task_components = self._extract_task_components(task_name)
        
        # Calculate different types of relevance
        exact_match_score = self._calculate_exact_match_score(query_terms, task_components)
        partial_match_score = self._calculate_partial_match_score(query_terms, task_components)
        semantic_match_score = self._calculate_semantic_match_score(query_terms, task_components)
        
        # Combine scores with weights
        total_score = (
            exact_match_score * 0.5 +          # Exact matches are strongest signal
            partial_match_score * 0.3 +        # Partial matches are good
            semantic_match_score * 0.2          # Semantic matches provide coverage
        )
        
        return min(1.0, total_score)  # Cap at 1.0
    
    def _extract_task_components(self, task_name: str) -> List[str]:
        """Extract meaningful components from a task name."""
        components = []
        
        # Split task name into tokens
        task_lower = task_name.lower()
        tokens = re.split(r'[_\-\s\d]+', task_lower)
        
        # Filter meaningful tokens
        for token in tokens:
            if len(token) >= 2 and token.isalpha():
                components.append(token)
        
        return components
    
    def _calculate_exact_match_score(self, query_terms: List[str], task_components: List[str]) -> float:
        """Calculate score based on exact term matches."""
        if not query_terms or not task_components:
            return 0.0
        
        matches = 0
        for query_term in query_terms:
            if query_term in task_components:
                matches += 1
        
        return matches / len(query_terms)
    
    def _calculate_partial_match_score(self, query_terms: List[str], task_components: List[str]) -> float:
        """Calculate score based on partial matches (substrings)."""
        if not query_terms or not task_components:
            return 0.0
        
        partial_matches = 0
        for query_term in query_terms:
            for component in task_components:
                # Check if query term is substring of component or vice versa
                if query_term in component or component in query_term:
                    # Give higher score for longer matches
                    match_length = min(len(query_term), len(component))
                    max_length = max(len(query_term), len(component))
                    partial_matches += match_length / max_length
                    break  # Only count each query term once
        
        return partial_matches / len(query_terms)
    
    def _calculate_semantic_match_score(self, query_terms: List[str], task_components: List[str]) -> float:
        """Calculate score based on semantic similarity."""
        if not query_terms or not task_components:
            return 0.0
        
        # Predefined semantic similarity groups
        similarity_groups = [
            # Truthfulness and accuracy
            {'truth', 'truthful', 'honest', 'accurate', 'fact', 'factual', 'valid', 'correct'},
            {'false', 'lie', 'incorrect', 'wrong', 'inaccurate', 'hallucination', 'fabrication'},
            
            # Knowledge and information
            {'knowledge', 'information', 'fact', 'data', 'learning', 'education'},
            {'question', 'answer', 'qa', 'query', 'inquiry'},
            
            # Reasoning and logic
            {'reasoning', 'logic', 'logical', 'think', 'analysis', 'inference', 'deduction'},
            {'common', 'sense', 'commonsense', 'practical', 'everyday'},
            
            # Language and communication
            {'language', 'linguistic', 'text', 'reading', 'comprehension', 'understanding'},
            {'generation', 'production', 'creation', 'synthesis'},
            
            # Ethics and safety
            {'ethical', 'moral', 'ethics', 'values', 'principles'},
            {'safe', 'safety', 'secure', 'harmless', 'benign'},
            {'harmful', 'dangerous', 'toxic', 'risk', 'unsafe'},
            
            # Bias and fairness
            {'bias', 'biased', 'prejudice', 'discrimination', 'unfair', 'partial'},
            {'fair', 'neutral', 'impartial', 'unbiased', 'equitable', 'just'},
            
            # Quality and evaluation
            {'quality', 'good', 'excellent', 'superior', 'standard'},
            {'evaluation', 'assessment', 'test', 'benchmark', 'measure'},
        ]
        
        semantic_matches = 0
        for query_term in query_terms:
            for component in task_components:
                for group in similarity_groups:
                    if query_term in group and component in group:
                        semantic_matches += 1
                        break  # Only count once per query term
                else:
                    continue
                break  # Break outer loop if match found
        
        return semantic_matches / len(query_terms)


# Global instance for convenience
_task_relevance_selector = TaskRelevanceSelector()

def find_relevant_tasks(
    query: str, 
    max_results: int = 20,
    min_relevance_score: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Find tasks most relevant to the given query.
    
    Args:
        query: The search query (e.g., "hallucination detection", "bias", "truthfulness")
        max_results: Maximum number of tasks to return
        min_relevance_score: Minimum relevance score threshold (0.0 to 1.0)
        
    Returns:
        List of (task_name, relevance_score) tuples, sorted by relevance
        
    Example:
        >>> results = find_relevant_tasks("truthfulness and hallucination", max_results=10)
        >>> for task_name, score in results:
        ...     print(f"{task_name}: {score:.3f}")
        truthfulqa_mc1: 0.850
        truthfulqa_mc2: 0.850
        factual_knowledge: 0.720
        ...
    """
    return _task_relevance_selector.find_relevant_tasks(
        query, max_results, min_relevance_score
    )


def get_top_relevant_tasks(query: str, count: int) -> List[str]:
    """
    Get the top N most relevant task names for a query.
    
    Args:
        query: The search query
        count: Number of top tasks to return
        
    Returns:
        List of task names, sorted by relevance (highest first)
    """
    results = find_relevant_tasks(query, max_results=count, min_relevance_score=0.0)
    return [task_name for task_name, _ in results]
