#!/usr/bin/env python3
"""
Full Benchmark Downloader

Downloads complete benchmarks from lm-eval-harness and saves them in a structured format.
Downloads the ENTIRE benchmark datasets, not just samples.

Usage:
    python download_full_benchmarks.py --benchmarks glue mmlu --force
    python download_full_benchmarks.py --all  # Download all benchmarks
"""

import argparse
import json
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add current directory to path to import local modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent / "lm-harness-integration"))

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

        print("🚀 Full Benchmark Downloader")
        print(f"📁 Download directory: {self.download_dir.absolute()}")

    def download_complete_benchmark(
        self, benchmark_name: str, benchmark_config: dict, force: bool = False
    ) -> Optional[str]:
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
                "splits_found": [],
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
                                if hasattr(doc, "__dict__"):
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
                },
            }

            # Convert to contrastive pairs
            contrastive_data = self.convert_to_contrastive_pairs(benchmark_name, complete_data)

            # Save only the contrastive pairs
            data_file = self.data_dir / f"{benchmark_name}.pkl"
            with open(data_file, "wb") as f:
                pickle.dump(contrastive_data["contrastive_pairs"], f)

            # Save metadata as JSON
            with open(metadata_file, "w") as f:
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
                name: config for name, config in CORE_BENCHMARKS.items() if name not in self.UNAVAILABLE_BENCHMARKS
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
                print("   🔧 These will likely fail. Remove from list to avoid delays.")

        print(f"\n🏗️ Downloading {len(benchmarks_to_download)} complete benchmarks")
        print(f"   Force redownload: {force}")

        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "excluded": list(self.UNAVAILABLE_BENCHMARKS) if benchmarks is None else [],
            "total_time": 0,
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
                print(f"   ⏱️  Elapsed: {elapsed / 60:.1f}min, ETA: {eta / 60:.1f}min")

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
        print("      🔄 Converting to contrastive pairs...")

        contrastive_pairs = []

        for i, sample in enumerate(complete_data["all_samples"]):
            try:
                pairs = self._convert_sample_to_pairs(sample, benchmark_name)
                if pairs:
                    contrastive_pairs.extend(pairs)
            except Exception as e:
                print(f"         ⚠️ Conversion error for sample {i}: {e}")

        return {"contrastive_pairs": contrastive_pairs}

    def _convert_sample_to_pairs(self, sample: Dict[str, Any], benchmark_name: str) -> List[Dict[str, Any]]:
        """Convert a single sample to contrastive pairs based on benchmark type."""

        # MMMLU format (instruction, option_a, option_b, option_c, option_d, answer)
        if "instruction" in sample and "option_a" in sample and "answer" in sample:
            return self._convert_mmmlu_format(sample)

        # Multiple Choice with explicit choices and numeric label (HellaSwag, SWAG, etc.)
        if ("endings" in sample and "label" in sample) or ("ending0" in sample and "label" in sample):
            return self._convert_multiple_choice_numeric(sample)

        # Multiple Choice with choices dict and answerKey (ARC, OpenBookQA, etc.)
        if "choices" in sample and "answerKey" in sample:
            return self._convert_multiple_choice_letter(sample)

        # TruthfulQA MC1 format
        if "mc1_targets" in sample:
            return self._convert_truthfulqa_mc1(sample)

        # TruthfulQA MC2 format
        if "mc2_targets" in sample:
            return self._convert_truthfulqa_mc2(sample)

        # SQuAD2 format (id, title, context, question, answers)
        if "context" in sample and "question" in sample and "answers" in sample:
            return self._convert_squad2_format(sample)

        # Textual entailment (premise/hypothesis format like CB, RTE)
        if "premise" in sample and "hypothesis" in sample:
            return self._convert_textual_entailment(sample)

        # Boolean questions (BoolQ)
        if "label" in sample and str(sample["label"]).lower() in ["true", "false", "0", "1"]:
            return self._convert_boolean_question(sample)

        # MBPP format (programming problems with code)
        if "task_id" in sample and "text" in sample and "code" in sample:
            return self._convert_mbpp_format(sample)

        # MATH-500 format (problem, solution, answer, subject, level)
        if (
            "problem" in sample
            and "solution" in sample
            and "answer" in sample
            and "subject" in sample
            and "level" in sample
        ):
            return self._convert_math500_format(sample)

        # WebQS format (question, answers list)
        if "question" in sample and "answers" in sample and isinstance(sample.get("answers"), list):
            return self._convert_webqs_format(sample)

        # NaturalQS format (question, answer as list)
        if "question" in sample and "answer" in sample and isinstance(sample.get("answer"), list):
            return self._convert_naturalqs_format(sample)

        # TriviaQA format (question, answer as dict with aliases)
        if "question" in sample and "answer" in sample and isinstance(sample.get("answer"), dict):
            return self._convert_triviaqa_format(sample)

        # Text generation with question/answer (GSM8K, math problems)
        if "question" in sample and "answer" in sample:
            return self._convert_text_generation(sample)

        # Reading comprehension (CoQA, SQuAD)
        if "story" in sample or "passage" in sample:
            return self._convert_reading_comprehension(sample)

        # SQuAD2 format (id, title, context, question, answers)
        if (
            "id" in sample
            and "title" in sample
            and "context" in sample
            and "question" in sample
            and "answers" in sample
        ):
            return self._convert_squad2_format(sample)

        # Winogrande format (sentence, option1, option2, answer)
        if "sentence" in sample and "option1" in sample and "option2" in sample and "answer" in sample:
            return self._convert_winogrande_format(sample)

        # WikiText format (page)
        if "page" in sample:
            return self._convert_wikitext_format(sample)

        # GPQA format (Question, choice1-4, answer, plus rich metadata)
        if (
            "Question" in sample
            and "choice1" in sample
            and "choice2" in sample
            and "choice3" in sample
            and "choice4" in sample
            and "answer" in sample
        ):
            return self._convert_gpqa_format(sample)

        # HLE format (question, answer, answer_type, category)
        if "question" in sample and "answer" in sample and "answer_type" in sample and "category" in sample:
            return self._convert_hle_format(sample)

        # HumanEval code generation format (task_id, canonical_solution, prompt, test, entry_point)
        if "task_id" in sample and "canonical_solution" in sample and "prompt" in sample and "test" in sample:
            return self._convert_humaneval_format(sample)

        # MBPP code generation format (task_id, code, prompt, test)
        if "task_id" in sample and "code" in sample and "prompt" in sample and "test" in sample:
            return self._convert_mbpp_format(sample)

        # Generic multiple choice fallback
        if "choices" in sample:
            return self._convert_generic_multiple_choice(sample)

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
        options = {"A": option_a, "B": option_b, "C": option_c, "D": option_d}

        correct_answer = options.get(answer, option_a)  # Default to A if answer not found

        # Create pairs with each incorrect option
        pairs = []
        for letter, option in options.items():
            if letter != answer and option:
                pairs.append(
                    {
                        "context": instruction,
                        "good_response": correct_answer,
                        "bad_response": option,
                        "metadata": {
                            "answer_key": answer,
                            "sample_id": sample.get("id", ""),
                            "benchmark_type": "mmmlu",
                        },
                    }
                )

        return pairs

    def _convert_multiple_choice_numeric(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert multiple choice with numeric label (HellaSwag, SWAG)."""
        context = sample.get("ctx", sample.get("query", ""))

        # Handle different choice formats
        if "endings" in sample:
            # HellaSwag format: choices in "endings" list
            choices = sample.get("endings", [])
        elif "ending0" in sample:
            # SWAG format: choices in separate ending0, ending1, etc. fields
            choices = []
            for i in range(4):  # SWAG typically has 4 choices
                ending_key = f"ending{i}"
                if ending_key in sample:
                    choices.append(sample[ending_key])
            # Build context from sent1, sent2, etc.
            sent1 = sample.get("sent1", "")
            sent2 = sample.get("sent2", "")
            context = f"{sent1} {sent2}".strip()
        else:
            choices = sample.get("choices", [])

        correct_idx = int(sample["label"])

        if not choices or correct_idx >= len(choices):
            return []

        correct_answer = choices[correct_idx]
        incorrect_answers = [choices[i] for i in range(len(choices)) if i != correct_idx]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": context,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "correct_index": correct_idx,
                        "sample_id": sample.get("id", sample.get("ind", "")),
                        "source": sample.get("source", ""),
                    },
                }
            )

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
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "answer_key": answer_key,
                        "sample_id": sample.get("id", ""),
                        "source": sample.get("source", ""),
                    },
                }
            )

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
                pairs.append(
                    {
                        "context": question,
                        "good_response": correct,
                        "bad_response": incorrect,
                        "metadata": {"sample_id": sample.get("id", ""), "benchmark_type": "truthfulqa_mc1"},
                    }
                )

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
                pairs.append(
                    {
                        "context": question,
                        "good_response": correct,
                        "bad_response": incorrect,
                        "metadata": {"sample_id": sample.get("id", ""), "benchmark_type": "truthfulqa_mc2"},
                    }
                )

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

        return [
            {
                "context": context,
                "good_response": correct_answer,
                "bad_response": incorrect_answer,
                "metadata": {
                    "sample_id": sample.get("idx", ""),
                    "original_label": label,
                    "benchmark_type": "textual_entailment",
                },
            }
        ]

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

        return [
            {
                "context": context,
                "good_response": correct_answer,
                "bad_response": incorrect_answer,
                "metadata": {"sample_id": sample.get("id", ""), "original_label": label, "benchmark_type": "boolean"},
            }
        ]

    def _convert_text_generation(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert text generation tasks (GSM8K, math problems)."""
        question = sample["question"]
        correct_answer = sample["answer"]

        # Generate plausible incorrect answers for math problems
        if any(
            math_keyword in question.lower() for math_keyword in ["dollars", "cost", "price", "how much", "how many"]
        ):
            incorrect_answers = self._generate_math_distractors(correct_answer)
        else:
            # For non-math, create generic incorrect responses
            incorrect_answers = [
                "I don't know the answer to this question.",
                "This question cannot be answered with the given information.",
                "The answer is unclear from the problem statement.",
            ]

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {"sample_id": sample.get("id", ""), "benchmark_type": "text_generation"},
                }
            )

        return pairs

    def _convert_math500_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MATH-500 format (problem, solution, answer, subject, level)."""
        problem = sample.get("problem", "")
        correct_answer = sample.get("answer", "")
        solution = sample.get("solution", "")
        subject = sample.get("subject", "")
        level = sample.get("level", 0)
        unique_id = sample.get("unique_id", "")

        # Generate mathematical incorrect answers based on correct answer
        incorrect_answers = self._generate_math_distractors(correct_answer)

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": problem,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "benchmark_type": "math500",
                        "subject": subject,
                        "level": level,
                        "sample_id": unique_id,
                        "has_solution": bool(solution.strip()),  # Track if step-by-step solution available
                    },
                }
            )

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

                    pairs.append(
                        {
                            "context": context,
                            "good_response": a_text,
                            "bad_response": incorrect_answer,
                            "metadata": {
                                "sample_id": sample.get("id", ""),
                                "question_index": i,
                                "benchmark_type": "reading_comprehension",
                            },
                        }
                    )
            # Handle other formats where questions/answers might be lists directly
            elif isinstance(questions_data, list) and isinstance(answers_data, list):
                for i, (q, a) in enumerate(zip(questions_data, answers_data)):
                    question_text = q.get("input_text", q.get("text", "")) if isinstance(q, dict) else str(q)
                    answer_text = a.get("input_text", a.get("text", "")) if isinstance(a, dict) else str(a)

                    context = f"{story}\n\nQuestion: {question_text}"

                    # Generate incorrect answer
                    incorrect_answer = "I cannot find this information in the passage."

                    pairs.append(
                        {
                            "context": context,
                            "good_response": answer_text,
                            "bad_response": incorrect_answer,
                            "metadata": {
                                "sample_id": sample.get("id", ""),
                                "question_index": i,
                                "benchmark_type": "reading_comprehension",
                            },
                        }
                    )

        return pairs

    def _convert_squad2_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert SQuAD2 format (id, title, context, question, answers)."""
        context = sample.get("context", "")
        question = sample.get("question", "")
        answers_data = sample.get("answers", {})

        # Extract answer texts from answers dict
        answer_texts = answers_data.get("text", [])
        if not answer_texts:
            # Handle empty answers (SQuAD2 has "no answer" questions)
            correct_answer = "There is no answer to this question in the given context."
        else:
            # Use the first answer as the correct one
            correct_answer = answer_texts[0]

        # Generate plausible incorrect answers for reading comprehension
        incorrect_answers = [
            "I cannot find this information in the passage.",
            "The question cannot be answered based on the given context.",
            "This information is not provided in the text.",
        ]

        # Format the context for the contrastive pair
        full_context = f"Context: {context}\n\nQuestion: {question}"

        pairs = []
        for incorrect in incorrect_answers:
            pairs.append(
                {
                    "context": full_context,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "sample_id": sample.get("id", ""),
                        "title": sample.get("title", ""),
                        "benchmark_type": "squad2",
                        "has_answer": bool(answer_texts),  # Track if this question has an answer
                    },
                }
            )

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
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "sample_id": sample.get("id", ""),
                        "benchmark_type": "generic_multiple_choice",
                        "warning": "Assumed first choice is correct",
                    },
                }
            )

        return pairs

    def _generate_math_distractors(self, correct_answer: str) -> List[str]:
        """Generate plausible incorrect answers for math problems."""
        import re

        # Extract final number from answer
        numbers = re.findall(r"\d+(?:\.\d+)?", correct_answer)
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

    def _convert_humaneval_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert HumanEval code generation format."""
        task_id = sample.get("task_id", "unknown")
        prompt = sample.get("prompt", "")
        canonical_solution = sample.get("canonical_solution", "")
        test = sample.get("test", "")
        entry_point = sample.get("entry_point", "")

        pairs = []

        # Create a contrastive pair with the coding prompt
        pairs.append(
            {
                "question": f"Complete this Python function:\n\n{prompt}",
                "correct_answer": canonical_solution,
                "incorrect_answer": "# Incorrect or incomplete implementation\npass",
                "metadata": {
                    "task_id": task_id,
                    "test_cases": test,
                    "entry_point": entry_point,
                    "benchmark_type": "humaneval",
                    "task_type": "code_completion",
                },
            }
        )

        return pairs

    def _convert_mbpp_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MBPP format (programming problems with code)."""
        # Use the benchmark extractor to get contrastive pairs
        from wisent_guard.core.benchmark_extractors import extract_contrastive_pair

        try:
            contrastive_data = extract_contrastive_pair("mbpp", sample, None)

            if contrastive_data:
                return [
                    {
                        "context": contrastive_data["question"],
                        "good_response": contrastive_data["correct_answer"],
                        "bad_response": contrastive_data["incorrect_answer"],
                        "metadata": {"task_id": sample.get("task_id", ""), "benchmark_type": "mbpp"},
                    }
                ]
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

        answer_match = re.search(r"[ABCD]", answer.upper())
        if not answer_match:
            return []

        answer_letter = answer_match.group()

        # Map answer letter to choice
        choices_map = {"A": choice1, "B": choice2, "C": choice3, "D": choice4}

        correct_answer = choices_map.get(answer_letter, "")
        if not correct_answer:
            return []

        # Create pairs with each incorrect option
        pairs = []
        for letter, choice in choices_map.items():
            if letter != answer_letter and choice:
                pairs.append(
                    {
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
                            "explanation": sample.get("Explanation", "")[:200]
                            if sample.get("Explanation")
                            else "",  # Truncate long explanations
                        },
                    }
                )

        return pairs

    def _convert_hle_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert HLE format (question, answer, answer_type, category)."""
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        answer_type = sample.get("answer_type", "")
        category = sample.get("category", "")

        if not question or not answer:
            return []

        # Use the HLE extractor to get contrastive pairs
        from wisent_guard.core.benchmark_extractors import HLEExtractor

        try:
            extractor = HLEExtractor()
            contrastive_pair = extractor.extract_contrastive_pair(sample)

            if contrastive_pair:
                return [
                    {
                        "question": contrastive_pair["question"],
                        "good_response": contrastive_pair["correct_answer"],
                        "bad_response": contrastive_pair["incorrect_answer"],
                        "metadata": {
                            "answer_type": answer_type,
                            "category": category,
                            "raw_subject": sample.get("raw_subject", ""),
                            "benchmark_type": "hle",
                        },
                    }
                ]
            return []
        except Exception as e:
            print(f"         ⚠️ Error converting HLE sample: {e}")
            return []

    def _convert_squad2_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert SQuAD2 format (id, title, context, question, answers)."""
        context = sample.get("context", "")
        question = sample.get("question", "")
        answers = sample.get("answers", {})

        if not context or not question:
            return []

        # Handle SQuAD2 answer format
        answer_text = ""
        if isinstance(answers, dict):
            answer_texts = answers.get("text", [])
            if answer_texts and len(answer_texts) > 0:
                answer_text = answer_texts[0]
        elif isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                answer_text = answers[0].get("text", "")
            else:
                answer_text = str(answers[0])

        if not answer_text:
            # For unanswerable questions in SQuAD2, create a pair with empty answer
            answer_text = "[No answer available]"

        # Create a contrastive pair using question-answering format
        return [
            {
                "question": f"Context: {context}\n\nQuestion: {question}",
                "good_response": answer_text,
                "bad_response": "[Incorrect answer]",  # Generic bad response for SQuAD2
                "metadata": {
                    "id": sample.get("id", ""),
                    "title": sample.get("title", ""),
                    "benchmark_type": "squad2",
                    "task_type": "reading_comprehension",
                },
            }
        ]

    def _convert_winogrande_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Winogrande format (sentence, option1, option2, answer)."""
        sentence = sample.get("sentence", "")
        option1 = sample.get("option1", "")
        option2 = sample.get("option2", "")
        answer = sample.get("answer", "")

        if not sentence or not option1 or not option2 or not answer:
            return []

        # Determine correct and incorrect answers
        if answer == "1":
            correct_answer = option1
            incorrect_answer = option2
        elif answer == "2":
            correct_answer = option2
            incorrect_answer = option1
        else:
            # If answer format is unexpected, default to option1 as correct
            correct_answer = option1
            incorrect_answer = option2

        # Create contrastive pair
        return [
            {
                "question": sentence,  # The sentence with blank to fill
                "good_response": correct_answer,
                "bad_response": incorrect_answer,
                "metadata": {
                    "option1": option1,
                    "option2": option2,
                    "answer": answer,
                    "benchmark_type": "winogrande",
                    "task_type": "coreference_resolution",
                },
            }
        ]

    def _convert_wikitext_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert WikiText format (page)."""
        page = sample.get("page", "")

        if not page or len(page.strip()) < 50:  # Skip very short pages
            return []

        # For WikiText, we create language modeling pairs
        # Split the page into sentences and create good/corrupted pairs
        sentences = page.split(". ")
        if len(sentences) < 2:
            return []

        pairs = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:  # Only use substantial sentences
                # Create a corrupted version by replacing some words
                words = sentence.split()
                if len(words) > 3:
                    # Simple corruption: duplicate a word in the middle
                    mid_idx = len(words) // 2
                    corrupted_words = words.copy()
                    corrupted_words.insert(mid_idx, words[mid_idx])
                    corrupted_sentence = " ".join(corrupted_words)

                    pairs.append(
                        {
                            "question": "Complete the text naturally:",
                            "good_response": sentence.strip(),
                            "bad_response": corrupted_sentence,
                            "metadata": {
                                "benchmark_type": "wikitext",
                                "task_type": "language_modeling",
                                "sentence_index": i,
                            },
                        }
                    )

                    # Limit to 3 pairs per page to avoid too many
                    if len(pairs) >= 3:
                        break

        return pairs

    def _convert_webqs_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert WebQS format (question, answers list)."""
        question = sample.get("question", "")
        answers = sample.get("answers", [])

        if not question or not answers:
            return []

        # Take the first answer as the correct one
        correct_answer = answers[0] if answers else ""

        if not correct_answer:
            return []

        # Generate incorrect answers (simple approach)
        incorrect_answers = []

        # Strategy 1: Use other answers from the same dataset if available
        if len(answers) > 1:
            incorrect_answers.extend(answers[1:3])  # Take up to 2 more answers as distractors

        # Strategy 2: Generate simple incorrect answers
        if len(incorrect_answers) < 2:
            # Simple factual distractors
            incorrect_answers.append("Unknown")
            incorrect_answers.append("No information available")

        # Create contrastive pairs
        pairs = []
        for incorrect in incorrect_answers[:2]:  # Limit to 2 pairs
            pairs.append(
                {
                    "question": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {"benchmark_type": "webqs", "task_type": "factual_qa", "url": sample.get("url", "")},
                }
            )

        return pairs

    def _convert_naturalqs_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert NaturalQS format (question, answer as list)."""
        question = sample.get("question", "")
        answer_list = sample.get("answer", [])

        if not question or not answer_list:
            return []

        # Take the first answer as the correct one (shortest/most direct)
        correct_answer = answer_list[0] if answer_list else ""

        if not correct_answer:
            return []

        # Generate incorrect answers
        incorrect_answers = []

        # Strategy 1: Use other answers from the list as distractors if available
        if len(answer_list) > 1:
            incorrect_answers.extend(answer_list[1:3])  # Take up to 2 more answers

        # Strategy 2: Generate generic incorrect answers
        if len(incorrect_answers) < 2:
            incorrect_answers.append("I don't know the answer to this question.")
            incorrect_answers.append("This information is not available.")

        # Create contrastive pairs
        pairs = []
        for incorrect in incorrect_answers[:2]:  # Limit to 2 pairs
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "benchmark_type": "naturalqs",
                        "task_type": "factual_qa",
                        "total_answers": len(answer_list),
                    },
                }
            )

        return pairs

    def _convert_triviaqa_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert TriviaQA format (question, answer as dict with aliases)."""
        question = sample.get("question", "")
        answer_dict = sample.get("answer", {})

        if not question or not answer_dict:
            return []

        # Extract the correct answer from aliases
        aliases = answer_dict.get("aliases", [])
        if not aliases:
            # Fallback to other fields
            correct_answer = (
                answer_dict.get("value", "") or answer_dict.get("normalized_value", "") or str(answer_dict)
            )[:100]  # Truncate if too long
        else:
            correct_answer = aliases[0]  # Use first alias as primary answer

        if not correct_answer:
            return []

        # Generate incorrect answers
        incorrect_answers = []

        # Strategy 1: Use other aliases as distractors if available
        if len(aliases) > 1:
            incorrect_answers.extend(aliases[1:3])  # Take up to 2 more aliases

        # Strategy 2: Generate generic incorrect answers for trivia
        if len(incorrect_answers) < 2:
            incorrect_answers.append("Unknown")
            incorrect_answers.append("I don't know")

        # Create contrastive pairs
        pairs = []
        for incorrect in incorrect_answers[:2]:  # Limit to 2 pairs
            pairs.append(
                {
                    "context": question,
                    "good_response": correct_answer,
                    "bad_response": incorrect,
                    "metadata": {
                        "benchmark_type": "triviaqa",
                        "task_type": "trivia_qa",
                        "total_aliases": len(aliases),
                        "entity_name": answer_dict.get("matched_wiki_entity_name", ""),
                    },
                }
            )

        return pairs

    def _convert_mbpp_format(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert MBPP/HumanEval code generation format (task_id, code, prompt, test)."""
        task_id = sample.get("task_id", "")
        code = sample.get("code", "")
        prompt = sample.get("prompt", "")
        test = sample.get("test", "")

        # For code generation tasks, we create contrastive pairs based on:
        # Correct: The reference code solution
        # Incorrect: A placeholder for incorrect/buggy code (since we don't have real incorrect solutions)

        pairs = []

        # Create a contrastive pair with the coding prompt
        pairs.append(
            {
                "question": f"Write Python code to solve this problem:\n\n{prompt}",
                "correct_answer": code,
                "incorrect_answer": "# This is a placeholder for incorrect code\n# In practice, this would be buggy or incomplete code\npass",  # TODO
                "metadata": {
                    "task_id": task_id,
                    "test_cases": test,
                    "source_file": sample.get("source_file", ""),
                    "test_imports": sample.get("test_imports", ""),
                    "test_list": sample.get("test_list", []),
                    "benchmark_type": "mbpp",
                    "task_type": "code_generation",
                    "programming_language": "python",
                },
            }
        )

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

        results = downloader.download_all_benchmarks(benchmarks=benchmarks_to_download, force=args.force)

        # Print summary
        print("\n" + "=" * 80)
        print("📊 FULL BENCHMARK DOWNLOAD SUMMARY")
        print("=" * 80)
        print(f"✅ Successful: {len(results['successful'])}")
        print(f"⏩ Skipped (already exist): {len(results['skipped'])}")
        print(f"❌ Failed: {len(results['failed'])}")
        if results["excluded"]:
            print(f"🚫 Excluded (known unavailable): {len(results['excluded'])}")
        print(f"⏱️  Total time: {results['total_time'] / 60:.1f} minutes")
        print(f"📁 Download directory: {downloader.download_dir.absolute()}")

        if results["successful"]:
            print("\n🎯 Successfully downloaded:")
            for benchmark in results["successful"]:
                print(f"   ✅ {benchmark}")

        if results["failed"]:
            print("\n❌ Failed downloads:")
            for benchmark in results["failed"]:
                print(f"   ❌ {benchmark}")

        if results["excluded"]:
            print("\n🚫 Excluded (known unavailable):")
            excluded_list = sorted(results["excluded"])
            for i in range(0, len(excluded_list), 4):  # Show 4 per line
                line_items = excluded_list[i : i + 4]
                print(f"   🚫 {', '.join(line_items)}")

        print("\n📊 Complete benchmark data saved in:")
        print(f"   📁 Data: {downloader.data_dir}")
        print(f"   📁 Metadata: {downloader.metadata_dir}")

        if results["successful"]:
            print(f"\n🎉 SUCCESS! Downloaded {len(results['successful'])} complete benchmarks!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
