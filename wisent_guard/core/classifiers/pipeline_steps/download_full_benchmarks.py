#!/usr/bin/env python3
"""
Full Benchmark Downloader

Downloads complete benchmarks from lm-eval-harness and saves them in a structured format.
Downloads the ENTIRE benchmark datasets, not just samples.

Usage:
    python download_full_benchmarks.py --benchmarks glue mmlu --force
    python download_full_benchmarks.py --all  # Download all benchmarks
"""

import os
import sys
import time
import argparse
import json
import pickle
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add current directory to path to import local modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent / 'lm-harness-integration'))

# Import the benchmark list
from only_benchmarks import CORE_BENCHMARKS

class FullBenchmarkDownloader:
    """Downloads complete benchmarks and saves them to disk."""
    
    # Benchmarks that are known to be unavailable or problematic
    UNAVAILABLE_BENCHMARKS = {
        # Empty set - let all benchmarks be attempted and skip dynamically if they fail
    }
    
    def __init__(self, download_dir: str = "full_benchmarks"):
        """
        Initialize the benchmark downloader.
        
        Args:
            download_dir: Directory to save downloaded benchmarks
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.data_dir = self.download_dir / "data"
        self.metadata_dir = self.download_dir / "metadata"
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Full Benchmark Downloader")
        print(f"📁 Download directory: {self.download_dir.absolute()}")
    
    def download_complete_benchmark(self, benchmark_name: str, benchmark_config: dict, force: bool = False) -> Optional[str]:
        """
        Download a complete benchmark dataset.
        
        Args:
            benchmark_name: Display name of the benchmark
            benchmark_config: Config dict with 'task' and 'tags' keys
            force: Force redownload even if exists
            
        Returns:
            Path to saved benchmark file, or None if failed
        """
        task_name = benchmark_config["task"]
        tags = benchmark_config.get("tags", [])
        
        # Check if already exists
        data_file = self.data_dir / f"{benchmark_name}.pkl"
        metadata_file = self.metadata_dir / f"{benchmark_name}_metadata.json"
        
        if data_file.exists() and metadata_file.exists() and not force:
            print(f"   ⏩ Skipping {benchmark_name} (already exists)")
            return str(data_file)
        
        print(f"   📥 Downloading complete benchmark: {benchmark_name}")
        print(f"      🔄 Loading full dataset for task: {task_name}")
        
        start_time = time.time()
        
        try:
            # Import lm_eval to download complete datasets
            import lm_eval
            from lm_eval import tasks
            
            # Get the task
            task_dict = tasks.get_task_dict([task_name])
            if task_name not in task_dict:
                print(f"      ❌ Task {task_name} not found in lm_eval")
                return None
            
            task = task_dict[task_name]
            
            # Download complete dataset - combine all splits into one unified dataset
            complete_data = {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "config": benchmark_config,
                "all_samples": [],
                "total_samples": 0,
                "splits_found": []
            }
            
            # Get all available document splits
            splits_to_try = ["test", "validation", "train", "dev"]
            
            for split in splits_to_try:
                try:
                    if hasattr(task, f"{split}_docs"):
                        docs_method = getattr(task, f"{split}_docs")
                        docs = list(docs_method())
                        
                        if docs:
                            print(f"      📊 Found {len(docs)} samples in {split} split")
                            complete_data["splits_found"].append(split)
                            
                            # Convert documents to serializable format and add to unified list
                            for i, doc in enumerate(docs):
                                if i % 1000 == 0 and i > 0:
                                    print(f"         Processing {split} {i}/{len(docs)}...")
                                
                                # Convert doc to dict, handling different doc types
                                if hasattr(doc, '__dict__'):
                                    doc_dict = doc.__dict__.copy()
                                elif isinstance(doc, dict):
                                    doc_dict = doc.copy()
                                else:
                                    doc_dict = {"content": str(doc)}
                                
                                # Add split origin info
                                doc_dict["_split_origin"] = split
                                
                                # Ensure all values are serializable
                                serializable_doc = {}
                                for key, value in doc_dict.items():
                                    try:
                                        json.dumps(value)  # Test if serializable
                                        serializable_doc[key] = value
                                    except (TypeError, ValueError):
                                        serializable_doc[key] = str(value)
                                
                                complete_data["all_samples"].append(serializable_doc)
                            
                            complete_data["total_samples"] += len(docs)
                
                except Exception as e:
                    print(f"      ⚠️  Could not load {split} split: {e}")
                    continue
            
            if complete_data["total_samples"] == 0:
                print(f"      ❌ No data found for {benchmark_name}")
                return None
            
            processing_time = time.time() - start_time
            
            # Add metadata
            metadata = {
                "benchmark_name": benchmark_name,
                "task_name": task_name,
                "config": benchmark_config,
                "download_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "total_samples": complete_data["total_samples"],
                "splits_found": complete_data["splits_found"],
                "task_info": {
                    "description": getattr(task, "DESCRIPTION", "No description available"),
                    "citation": getattr(task, "CITATION", "No citation available"),
                    "homepage": getattr(task, "HOMEPAGE", "No homepage available"),
                }
            }
            
            # Convert to contrastive pairs
            contrastive_data = self.convert_to_contrastive_pairs(benchmark_name, complete_data)
            
            # Save only the contrastive pairs
            data_file = self.data_dir / f"{benchmark_name}.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump(contrastive_data["contrastive_pairs"], f)
            
            # Save metadata as JSON
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"      ✅ Saved benchmark: {benchmark_name}")
            print(f"         📊 Contrastive pairs: {len(contrastive_data['contrastive_pairs'])}")
            print(f"         ⏱️  Time: {processing_time:.1f}s")
            
            return str(data_file)
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"      ❌ Failed to download {benchmark_name}: {e}")
            print(f"         ⏱️  Time: {processing_time:.1f}s")
            return None
    
    def download_all_benchmarks(self, benchmarks: Optional[List[str]] = None, force: bool = False) -> Dict[str, Any]:
        """
        Download multiple complete benchmarks.
        
        Args:
            benchmarks: List of benchmark names to download, or None for all
            force: Force redownload even if exists
            
        Returns:
            Dictionary with download results
        """
        if benchmarks is None:
            # Filter out known unavailable benchmarks when downloading all
            available_benchmarks = {
                name: config for name, config in CORE_BENCHMARKS.items() 
                if name not in self.UNAVAILABLE_BENCHMARKS
            }
            benchmarks_to_download = available_benchmarks
            
            # Report excluded benchmarks
            excluded_count = len(CORE_BENCHMARKS) - len(available_benchmarks)
            if excluded_count > 0:
                print(f"⏩ Excluding {excluded_count} known unavailable benchmarks")
                print(f"   📋 Available benchmarks: {len(available_benchmarks)}/{len(CORE_BENCHMARKS)}")
        else:
            benchmarks_to_download = {name: CORE_BENCHMARKS[name] for name in benchmarks if name in CORE_BENCHMARKS}
            
            # Check for invalid benchmarks
            invalid = [name for name in benchmarks if name not in CORE_BENCHMARKS]
            if invalid:
                print(f"⚠️  Invalid benchmarks (skipping): {invalid}")
            
            # Warn about unavailable benchmarks that were explicitly requested
            unavailable_requested = [name for name in benchmarks if name in self.UNAVAILABLE_BENCHMARKS]
            if unavailable_requested:
                print(f"⚠️  Requested benchmarks are known to be unavailable: {unavailable_requested}")
                print(f"   🔧 These will likely fail. Remove from list to avoid delays.")
        
        print(f"\n🏗️ Downloading {len(benchmarks_to_download)} complete benchmarks")
        print(f"   Force redownload: {force}")
        
        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "excluded": list(self.UNAVAILABLE_BENCHMARKS) if benchmarks is None else [],
            "total_time": 0
        }
        
        total_start_time = time.time()
        
        for i, (benchmark_name, benchmark_config) in enumerate(benchmarks_to_download.items(), 1):
            print(f"\n[{i:2d}/{len(benchmarks_to_download)}] 🎯 Processing benchmark: {benchmark_name}")
            print(f"   Task: {benchmark_config['task']}")
            print(f"   Tags: {benchmark_config.get('tags', [])}")
            
            try:
                result_path = self.download_complete_benchmark(benchmark_name, benchmark_config, force)
                
                if result_path:
                    results["successful"].append(benchmark_name)
                else:
                    results["failed"].append(benchmark_name)
                    
            except Exception as e:
                print(f"   ❌ Exception downloading {benchmark_name}: {e}")
                results["failed"].append(benchmark_name)
            
            # Progress update
            elapsed = time.time() - total_start_time
            if i < len(benchmarks_to_download):
                eta = elapsed * (len(benchmarks_to_download) - i) / i
                print(f"\n📊 Progress: {i}/{len(benchmarks_to_download)} benchmarks completed")
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
        
        results["total_time"] = time.time() - total_start_time
        return results

    def convert_to_contrastive_pairs(self, benchmark_name: str, complete_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert benchmark data to contrastive pair format.
        
        Args:
            benchmark_name: Name of the benchmark
            complete_data: Raw benchmark data
            
        Returns:
            Dictionary with contrastive pairs
        """
        print(f"      🔄 Converting to contrastive pairs...")
        
        contrastive_pairs = []
        
        for i, sample in enumerate(complete_data["all_samples"]):
            try:
                pairs = self._convert_sample_to_pairs(sample, benchmark_name)
                if pairs:
                    contrastive_pairs.extend(pairs)
            except Exception as e:
                print(f"         ⚠️ Conversion error for sample {i}: {e}")
        
        return {
            "contrastive_pairs": contrastive_pairs
        }
    
    def _convert_sample_to_pairs(self, sample: Dict[str, Any], benchmark_name: str) -> List[Dict[str, Any]]:
        """Convert a single sample to contrastive pairs based on benchmark type."""
        
        # MMMLU format (instruction, option_a, option_b, option_c, option_d, answer)
        if "instruction" in sample and "option_a" in sample and "answer" in sample:
            return self._convert_mmmlu_format(sample)
        
        # Multiple Choice with explicit choices and numeric label (HellaSwag, SWAG, etc.)
        elif "endings" in sample and "label" in sample:
            return self._convert_multiple_choice_numeric(sample)
        
        # Multiple Choice with choices dict and answerKey (ARC, OpenBookQA, etc.)
        elif "choices" in sample and "answerKey" in sample:
            return self._convert_multiple_choice_letter(sample)
        
        # TruthfulQA MC1 format
        elif "mc1_targets" in sample:
            return self._convert_truthfulqa_mc1(sample)
        
        # TruthfulQA MC2 format  
        elif "mc2_targets" in sample:
            return self._convert_truthfulqa_mc2(sample)
        
        # Textual entailment (premise/hypothesis format like CB, RTE)
        elif "premise" in sample and "hypothesis" in sample:
            return self._convert_textual_entailment(sample)
        
        # Boolean questions (BoolQ)
        elif "label" in sample and str(sample["label"]).lower() in ["true", "false", "0", "1"]:
            return self._convert_boolean_question(sample)
        
        # MBPP format (programming problems with code)
        elif "task_id" in sample and "text" in sample and "code" in sample:
            return self._convert_mbpp_format(sample)
        
        # Text generation with question/answer (GSM8K, math problems)
        elif "question" in sample and "answer" in sample:
            return self._convert_text_generation(sample)
        
        # Reading comprehension (CoQA, SQuAD)
        elif "story" in sample or "passage" in sample:
            return self._convert_reading_comprehension(sample)
        
        # GPQA format (Question, choice1-4, answer, plus rich metadata)
        elif ("Question" in sample and "choice1" in sample and "choice2" in sample and 
              "choice3" in sample and "choice4" in sample and "answer" in sample):
            return self._convert_gpqa_format(sample)
        
        # Generic multiple choice fallback
        elif "choices" in sample:
            return self._convert_generic_multiple_choice(sample)
        
        else:
            print(f"         ⚠️ Unknown sample format: {list(sample.keys())}")
            return []
    
    def _convert_mmmlu_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MMMLU format (instruction, option_a/b/c/d, answer)."""
        instruction = sample.get("instruction", "")
        option_a = sample.get("option_a", "")
        option_b = sample.get("option_b", "")
        option_c = sample.get("option_c", "")
        option_d = sample.get("option_d", "")
        answer = sample.get("answer", "")
        
        # Map answer letter to option
        options = {
            "A": option_a,
            "B": option_b,
            "C": option_c,
            "D": option_d
        }
        
        correct_answer = options.get(answer, option_a)  # Default to A if answer not found
        
        # Create pairs with each incorrect option
        pairs = []
        for letter, option in options.items():
            if letter != answer and option:
                pairs.append({
                    "context": instruction,
                    "good_response": correct_answer,
                    "bad_response": option,
                    "metadata": {
                        "answer_key": answer,
                        "sample_id": sample.get("id", ""),
                        "benchmark_type": "mmmlu"
                    }
                })
        
        return pairs
    
    def _convert_multiple_choice_numeric(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert multiple choice with numeric label (HellaSwag, SWAG)."""
        context = sample.get("ctx", sample.get("query", ""))
        choices = sample.get("endings", sample.get("choices", []))
        correct_idx = int(sample["label"])
        
        if not choices or correct_idx >= len(choices):
            return []
        
        correct_answer = choices[correct_idx]
        incorrect_answers = [choices[i] for i in range(len(choices)) if i != correct_idx]
        
        pairs = []
        for incorrect in incorrect_answers:
            pairs.append({
                "context": context,
                "good_response": correct_answer,
                "bad_response": incorrect,
                "metadata": {
                    "correct_index": correct_idx,
                    "sample_id": sample.get("id", sample.get("ind", "")),
                    "source": sample.get("source", "")
                }
            })
        
        return pairs
    
    def _convert_multiple_choice_letter(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert multiple choice with letter answerKey (ARC, OpenBookQA)."""
        question = sample.get("question", "")
        choices_text = sample["choices"]["text"]
        choices_labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]
        
        # Find correct answer
        correct_idx = None
        for i, label in enumerate(choices_labels):
            if label == answer_key:
                correct_idx = i
                break
        
        if correct_idx is None:
            return []
        
        correct_answer = choices_text[correct_idx]
        incorrect_answers = [choices_text[i] for i in range(len(choices_text)) if i != correct_idx]
        
        pairs = []
        for incorrect in incorrect_answers:
            pairs.append({
                "context": question,
                "good_response": correct_answer,
                "bad_response": incorrect,
                "metadata": {
                    "answer_key": answer_key,
                    "sample_id": sample.get("id", ""),
                    "source": sample.get("source", "")
                }
            })
        
        return pairs
    
    def _convert_truthfulqa_mc1(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert TruthfulQA MC1 format."""
        question = sample["question"]
        choices = sample["mc1_targets"]["choices"]
        labels = sample["mc1_targets"]["labels"]
        
        # Find correct and incorrect answers
        correct_answers = [choices[i] for i, label in enumerate(labels) if label == 1]
        incorrect_answers = [choices[i] for i, label in enumerate(labels) if label == 0]
        
        if not correct_answers or not incorrect_answers:
            return []
        
        pairs = []
        for correct in correct_answers:
            for incorrect in incorrect_answers[:3]:  # Limit to 3 incorrect per correct
                pairs.append({
                    "context": question,
                    "good_response": correct,
                    "bad_response": incorrect,
                    "metadata": {
                        "sample_id": sample.get("id", ""),
                        "benchmark_type": "truthfulqa_mc1"
                    }
                })
        
        return pairs
    
    def _convert_truthfulqa_mc2(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert TruthfulQA MC2 format."""
        question = sample["question"]
        choices = sample["mc2_targets"]["choices"]
        labels = sample["mc2_targets"]["labels"]
        
        correct_answers = [choices[i] for i, label in enumerate(labels) if label == 1]
        incorrect_answers = [choices[i] for i, label in enumerate(labels) if label == 0]
        
        if not correct_answers or not incorrect_answers:
            return []
        
        pairs = []
        for correct in correct_answers:
            for incorrect in incorrect_answers[:2]:  # Limit to 2 incorrect per correct
                pairs.append({
                    "context": question,
                    "good_response": correct,
                    "bad_response": incorrect,
                    "metadata": {
                        "sample_id": sample.get("id", ""),
                        "benchmark_type": "truthfulqa_mc2"
                    }
                })
        
        return pairs
    
    def _convert_textual_entailment(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert textual entailment tasks (CB, RTE)."""
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        label = sample["label"]
        
        # Map different label formats
        if isinstance(label, str):
            if label.lower() in ["entailment", "true", "1"]:
                correct_answer = "Yes, this follows logically."
                incorrect_answer = "No, this does not follow logically."
            elif label.lower() in ["contradiction", "false", "0"]:
                correct_answer = "No, this contradicts the premise."
                incorrect_answer = "Yes, this follows logically."
            else:  # neutral
                correct_answer = "This is neither supported nor contradicted."
                incorrect_answer = "Yes, this follows logically."
        else:
            # Numeric labels: typically 0=entailment, 1=neutral, 2=contradiction
            if label == 0:
                correct_answer = "Yes, this follows logically."
                incorrect_answer = "No, this does not follow logically."
            elif label == 2:
                correct_answer = "No, this contradicts the premise."
                incorrect_answer = "Yes, this follows logically."
            else:  # neutral
                correct_answer = "This is neither supported nor contradicted."
                incorrect_answer = "Yes, this follows logically."
        
        context = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the hypothesis follow from the premise?"
        
        return [{
            "context": context,
            "good_response": correct_answer,
            "bad_response": incorrect_answer,
            "metadata": {
                "sample_id": sample.get("idx", ""),
                "original_label": label,
                "benchmark_type": "textual_entailment"
            }
        }]
    
    def _convert_boolean_question(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert boolean questions (BoolQ)."""
        question = sample.get("question", "")
        passage = sample.get("passage", "")
        label = sample["label"]
        
        # Determine correct answer
        if str(label).lower() in ["true", "1"]:
            correct_answer = "Yes"
            incorrect_answer = "No"
        else:
            correct_answer = "No"
            incorrect_answer = "Yes"
        
        context = f"{passage}\n\nQuestion: {question}" if passage else question
        
        return [{
            "context": context,
            "good_response": correct_answer,
            "bad_response": incorrect_answer,
            "metadata": {
                "sample_id": sample.get("id", ""),
                "original_label": label,
                "benchmark_type": "boolean"
            }
        }]
    
    def _convert_text_generation(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert text generation tasks (GSM8K, math problems)."""
        question = sample["question"]
        correct_answer = sample["answer"]
        
        # Generate plausible incorrect answers for math problems
        if any(math_keyword in question.lower() for math_keyword in ["dollars", "cost", "price", "how much", "how many"]):
            incorrect_answers = self._generate_math_distractors(correct_answer)
        else:
            # For non-math, create generic incorrect responses
            incorrect_answers = [
                "I don't know the answer to this question.",
                "This question cannot be answered with the given information.",
                "The answer is unclear from the problem statement."
            ]
        
        pairs = []
        for incorrect in incorrect_answers:
            pairs.append({
                "context": question,
                "good_response": correct_answer,
                "bad_response": incorrect,
                "metadata": {
                    "sample_id": sample.get("id", ""),
                    "benchmark_type": "text_generation"
                }
            })
        
        return pairs
    
    def _convert_reading_comprehension(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert reading comprehension tasks (CoQA, SQuAD)."""
        # This is complex as these often have multiple Q&A pairs
        # For now, create a basic conversion
        story = sample.get("story", sample.get("passage", ""))
        
        pairs = []
        
        # Handle CoQA format with multiple questions
        if "questions" in sample and "answers" in sample:
            questions_data = sample["questions"]
            answers_data = sample["answers"]
            
            # CoQA format has questions and answers as dicts with lists
            if isinstance(questions_data, dict) and isinstance(answers_data, dict):
                question_texts = questions_data.get("input_text", [])
                answer_texts = answers_data.get("input_text", [])
                
                for i, (q_text, a_text) in enumerate(zip(question_texts, answer_texts)):
                    context = f"{story}\n\nQuestion: {q_text}"
                    
                    # Generate incorrect answer
                    incorrect_answer = "I cannot find this information in the passage."
                    
                    pairs.append({
                        "context": context,
                        "good_response": a_text,
                        "bad_response": incorrect_answer,
                        "metadata": {
                            "sample_id": sample.get("id", ""),
                            "question_index": i,
                            "benchmark_type": "reading_comprehension"
                        }
                    })
            # Handle other formats where questions/answers might be lists directly
            elif isinstance(questions_data, list) and isinstance(answers_data, list):
                for i, (q, a) in enumerate(zip(questions_data, answers_data)):
                    question_text = q.get("input_text", q.get("text", "")) if isinstance(q, dict) else str(q)
                    answer_text = a.get("input_text", a.get("text", "")) if isinstance(a, dict) else str(a)
                    
                    context = f"{story}\n\nQuestion: {question_text}"
                    
                    # Generate incorrect answer
                    incorrect_answer = "I cannot find this information in the passage."
                    
                    pairs.append({
                        "context": context,
                        "good_response": answer_text,
                        "bad_response": incorrect_answer,
                        "metadata": {
                            "sample_id": sample.get("id", ""),
                            "question_index": i,
                            "benchmark_type": "reading_comprehension"
                        }
                    })
        
        return pairs
    
    def _convert_generic_multiple_choice(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generic fallback for multiple choice formats."""
        question = sample.get("question", sample.get("query", ""))
        choices = sample.get("choices", [])
        
        if len(choices) < 2:
            return []
        
        # Assume first choice is correct (this is a fallback)
        correct_answer = choices[0]
        incorrect_answers = choices[1:]
        
        pairs = []
        for incorrect in incorrect_answers:
            pairs.append({
                "context": question,
                "good_response": correct_answer,
                "bad_response": incorrect,
                "metadata": {
                    "sample_id": sample.get("id", ""),
                    "benchmark_type": "generic_multiple_choice",
                    "warning": "Assumed first choice is correct"
                }
            })
        
        return pairs
    
    def _generate_math_distractors(self, correct_answer: str) -> List[str]:
        """Generate plausible incorrect answers for math problems."""
        import re
        
        # Extract final number from answer
        numbers = re.findall(r'\d+(?:\.\d+)?', correct_answer)
        if not numbers:
            return ["42", "0", "Cannot be determined"]
        
        final_number = float(numbers[-1])
        
        # Generate distractors
        distractors = []
        
        # Off-by-one errors
        distractors.append(str(int(final_number + 1)))
        distractors.append(str(int(final_number - 1)))
        
        # Calculation errors (common mistakes)
        distractors.append(str(int(final_number * 2)))
        distractors.append(str(int(final_number / 2)))
        
        # Random nearby numbers
        distractors.append(str(int(final_number + random.randint(2, 10))))
        
        return distractors[:3]  # Return top 3

    def _convert_mbpp_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MBPP format (programming problems with code)."""
        # Use the benchmark extractor to get contrastive pairs
        from wisent_guard.core.benchmark_extractors import extract_contrastive_pair
        
        try:
            contrastive_data = extract_contrastive_pair('mbpp', sample, None)
            
            if contrastive_data:
                return [{
                    "context": contrastive_data['question'],
                    "good_response": contrastive_data['correct_answer'],
                    "bad_response": contrastive_data['incorrect_answer'],
                    "metadata": {
                        "task_id": sample.get("task_id", ""),
                        "benchmark_type": "mbpp"
                    }
                }]
            else:
                return []
        except Exception as e:
            print(f"         ⚠️ Error converting MBPP sample: {e}")
            return []

    def _convert_gpqa_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert GPQA format (Question, choice1-4, answer, plus rich metadata)."""
        question = sample.get("Question", "")
        choice1 = sample.get("choice1", "")
        choice2 = sample.get("choice2", "")
        choice3 = sample.get("choice3", "")
        choice4 = sample.get("choice4", "")
        answer = sample.get("answer", "")
        
        # Extract letter from answer format like "(A)" or "A"
        import re
        answer_match = re.search(r'[ABCD]', answer.upper())
        if not answer_match:
            return []
        
        answer_letter = answer_match.group()
        
        # Map answer letter to choice
        choices_map = {
            "A": choice1,
            "B": choice2,
            "C": choice3,
            "D": choice4
        }
        
        correct_answer = choices_map.get(answer_letter, "")
        if not correct_answer:
            return []
        
        # Create pairs with each incorrect option
        pairs = []
        for letter, choice in choices_map.items():
            if letter != answer_letter and choice:
                pairs.append({
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": choice,
                    "metadata": {
                        "answer_key": answer_letter,
                        "raw_answer": answer,
                        "benchmark_type": "gpqa",
                        "subdomain": sample.get("Subdomain", ""),
                        "high_level_domain": sample.get("High-level domain", ""),
                        "difficulty_estimate": sample.get("Writer's Difficulty Estimate", ""),
                        "expert_accuracy": sample.get("Expert Validator Accuracy", ""),
                        "explanation": sample.get("Explanation", "")[:200] if sample.get("Explanation") else ""  # Truncate long explanations
                    }
                })
        
        return pairs

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Download complete benchmarks from lm-eval-harness")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmarks", nargs="+", help="Specific benchmarks to download")
    group.add_argument("--all", action="store_true", help="Download all available benchmarks")
    
    parser.add_argument("--force", action="store_true", help="Force redownload even if exists")
    parser.add_argument("--download-dir", default="full_benchmarks", help="Directory to save downloads")
    
    args = parser.parse_args()
    
    print("🚀 Full Benchmark Downloader")
    print("=" * 60)
    
    # Create downloader
    downloader = FullBenchmarkDownloader(download_dir=args.download_dir)
    
    # Download benchmarks
    try:
        if args.all:
            benchmarks_to_download = None
            print(f"📋 Downloading ALL {len(CORE_BENCHMARKS)} available benchmarks")
        else:
            benchmarks_to_download = args.benchmarks
            print(f"📋 Downloading {len(args.benchmarks)} specified benchmarks: {args.benchmarks}")
        
        results = downloader.download_all_benchmarks(
            benchmarks=benchmarks_to_download,
            force=args.force
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 FULL BENCHMARK DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {len(results['successful'])}")
        print(f"⏩ Skipped (already exist): {len(results['skipped'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        if results['excluded']:
            print(f"🚫 Excluded (known unavailable): {len(results['excluded'])}")
        print(f"⏱️  Total time: {results['total_time']/60:.1f} minutes")
        print(f"📁 Download directory: {downloader.download_dir.absolute()}")
        
        if results["successful"]:
            print(f"\n🎯 Successfully downloaded:")
            for benchmark in results["successful"]:
                print(f"   ✅ {benchmark}")
        
        if results["failed"]:
            print(f"\n❌ Failed downloads:")
            for benchmark in results["failed"]:
                print(f"   ❌ {benchmark}")
        
        if results["excluded"]:
            print(f"\n🚫 Excluded (known unavailable):")
            excluded_list = sorted(results["excluded"])
            for i in range(0, len(excluded_list), 4):  # Show 4 per line
                line_items = excluded_list[i:i+4]
                print(f"   🚫 {', '.join(line_items)}")
        
        print(f"\n📊 Complete benchmark data saved in:")
        print(f"   📁 Data: {downloader.data_dir}")
        print(f"   📁 Metadata: {downloader.metadata_dir}")
        
        if results["successful"]:
            print(f"\n🎉 SUCCESS! Downloaded {len(results['successful'])} complete benchmarks!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 