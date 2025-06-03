"""
Benchmark loader for Wisent-Guard that interfaces with lm-evaluation-harness.
"""

import os
import json
from typing import Dict, List, Tuple, Any
from lm_eval import evaluator
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch

class BenchmarkLoader:
    def __init__(self, benchmark_name: str):
        """
        Initialize benchmark loader.
        
        Args:
            benchmark_name: Name of the benchmark to load
        """
        self.benchmark_name = benchmark_name
        # Initialize the task
        self.task = evaluator.get_task(self.benchmark_name)()
        self.docs = list(self.task.validation_docs())
    
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split the benchmark documents into train and test sets.
        
        Args:
            test_size: Proportion of documents to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_docs, test_docs) where each doc is a dictionary
        """
        return train_test_split(self.docs, test_size=test_size, random_state=random_state)
    
    def get_prompt_pairs(self, docs: List[Dict]) -> List[Tuple[str, str]]:
        """
        Generate prompt pairs for training the classifier.
        
        Args:
            docs: List of benchmark documents
            
        Returns:
            List of (prompt, good_response) pairs
        """
        pairs = []
        for doc in docs:
            prompt = self.task.doc_to_text(doc)
            target = self.task.doc_to_target(doc)
            pairs.append((prompt, target))
        return pairs
