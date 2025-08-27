"""
On-the-Fly Classifier Creation System for Autonomous Agent

This module handles:
- Dynamic training of new classifiers for specific issue types
- Automatic training data generation for different problem domains
- Classifier optimization and validation
- Integration with the autonomous agent system
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from wisent_guard.core.classifier.classifier import ActivationClassifier, Classifier

from ...activations import Activations
from ...layer import Layer
from ...model import Model
from ...model_persistence import ModelPersistence, create_classifier_metadata


@dataclass
class TrainingConfig:
    """Configuration for classifier training."""

    issue_type: str
    layer: int
    classifier_type: str = "logistic"
    threshold: float = 0.5
    model_name: str = ""
    training_samples: int = 100
    test_split: float = 0.2
    optimization_metric: str = "f1"
    save_path: Optional[str] = None


@dataclass
class TrainingResult:
    """Result of classifier training."""

    classifier: Classifier
    config: TrainingConfig
    performance_metrics: Dict[str, float]
    training_time: float
    save_path: Optional[str] = None


class ClassifierCreator:
    """Creates new classifiers on demand for the autonomous agent."""

    def __init__(self, model: Model):
        """
        Initialize the classifier creator.

        Args:
            model: The language model to use for training
        """
        self.model = model

    def create_classifier_for_issue_type(
        self, issue_type: str, layer: int, config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Create a new classifier for a specific issue type.

        Args:
            issue_type: Type of issue to detect (e.g., "hallucination", "quality")
            layer: Model layer to use for activation extraction
            config: Optional training configuration

        Returns:
            TrainingResult with the trained classifier and metrics
        """
        print(f"🏋️ Creating classifier for {issue_type} at layer {layer}...")

        # Use provided config or create default
        if config is None:
            config = TrainingConfig(issue_type=issue_type, layer=layer, model_name=self.model.name)

        start_time = time.time()

        # Generate training data
        print("   📊 Generating training data...")
        training_data = self._generate_training_data(issue_type, config.training_samples)

        # Extract activations
        print("   🧠 Extracting activations...")
        harmful_activations, harmless_activations = self._extract_activations_from_data(training_data, layer)

        # Train classifier
        print("   🎯 Training classifier...")
        classifier = self._train_classifier(harmful_activations, harmless_activations, config)

        # Evaluate performance
        print("   📈 Evaluating performance...")
        metrics = self._evaluate_classifier(classifier, harmful_activations, harmless_activations)

        training_time = time.time() - start_time

        # Save classifier if path provided
        save_path = None
        if config.save_path:
            print("   💾 Saving classifier...")
            save_path = self._save_classifier(classifier, config, metrics)

        result = TrainingResult(
            classifier=classifier.classifier,  # Return the base classifier
            config=config,
            performance_metrics=metrics,
            training_time=training_time,
            save_path=save_path,
        )

        print(
            f"   ✅ Classifier created in {training_time:.2f}s "
            f"(F1: {metrics.get('f1', 0):.3f}, Accuracy: {metrics.get('accuracy', 0):.3f})"
        )

        return result

    def create_multi_layer_classifiers(
        self, issue_type: str, layers: List[int], save_base_path: Optional[str] = None
    ) -> Dict[int, TrainingResult]:
        """
        Create classifiers for multiple layers for the same issue type.

        Args:
            issue_type: Type of issue to detect
            layers: List of layers to create classifiers for
            save_base_path: Base path for saving classifiers

        Returns:
            Dictionary mapping layer indices to training results
        """
        print(f"🔄 Creating multi-layer classifiers for {issue_type}...")

        results = {}

        for layer in layers:
            config = TrainingConfig(
                issue_type=issue_type,
                layer=layer,
                model_name=self.model.name,
                save_path=f"{save_base_path}_layer_{layer}.pkl" if save_base_path else None,
            )

            result = self.create_classifier_for_issue_type(issue_type, layer, config)
            results[layer] = result

        print(f"   ✅ Created {len(results)} classifiers across layers {layers}")
        return results

    def optimize_classifier_for_performance(
        self,
        issue_type: str,
        layer_range: Tuple[int, int] = None,
        classifier_types: List[str] = None,
        target_metric: str = "f1",
        min_target_score: float = 0.7,
    ) -> TrainingResult:
        """
        Optimize classifier by testing different configurations.

        Args:
            issue_type: Type of issue to detect
            layer_range: Range of layers to test (start, end). If None, auto-detect all model layers
            classifier_types: Types of classifiers to test
            target_metric: Metric to optimize for
            min_target_score: Minimum acceptable score

        Returns:
            Best performing classifier configuration
        """
        print(f"🎯 Optimizing classifier for {issue_type}...")

        if classifier_types is None:
            classifier_types = ["logistic", "mlp"]

        # Auto-detect layer range if not provided
        if layer_range is None:
            from ..hyperparameter_optimizer import detect_model_layers

            total_layers = detect_model_layers(self.model)
            layer_range = (0, total_layers - 1)
            print(f"   📊 Auto-detected {total_layers} layers, testing range {layer_range[0]}-{layer_range[1]}")

        best_result = None
        best_score = 0.0

        layers_to_test = range(layer_range[0], layer_range[1] + 1, 2)  # Test every 2nd layer

        for layer in layers_to_test:
            for classifier_type in classifier_types:
                config = TrainingConfig(
                    issue_type=issue_type, layer=layer, classifier_type=classifier_type, model_name=self.model.name
                )

                try:
                    result = self.create_classifier_for_issue_type(issue_type, layer, config)
                    score = result.performance_metrics.get(target_metric, 0.0)

                    print(f"      Layer {layer}, {classifier_type}: {target_metric}={score:.3f}")

                    if score > best_score:
                        best_score = score
                        best_result = result

                        # Early stopping if we hit the target
                        if score >= min_target_score:
                            print(f"      🎉 Target score reached: {score:.3f}")
                            break

                except Exception as e:
                    print(f"      ❌ Failed layer {layer}, {classifier_type}: {e}")
                    continue

            # Break outer loop if target reached
            if best_score >= min_target_score:
                break

        if best_result is None:
            raise RuntimeError(f"Failed to create any working classifier for {issue_type}")

        print(
            f"   ✅ Best configuration: Layer {best_result.config.layer}, "
            f"{best_result.config.classifier_type}, {target_metric}={best_score:.3f}"
        )

        return best_result

    async def create_classifier_for_issue_with_benchmarks(
        self,
        issue_type: str,
        relevant_benchmarks: List[str],
        layer: int = 15,
        num_samples: int = 50,
        config: Optional[TrainingConfig] = None,
    ) -> TrainingResult:
        """
        Create a classifier using specific benchmarks for better contrastive pairs.

        Args:
            issue_type: Type of issue to detect (e.g., "hallucination", "quality")
            relevant_benchmarks: List of benchmark names to use for training data
            layer: Model layer to use for activation extraction (default: 15)
            num_samples: Number of training samples to generate
            config: Optional training configuration

        Returns:
            TrainingResult with the trained classifier and metrics
        """
        print(f"🎯 Creating {issue_type} classifier using benchmarks: {relevant_benchmarks}")

        # Use provided config or create default
        if config is None:
            config = TrainingConfig(
                issue_type=issue_type, layer=layer, model_name=self.model.name, training_samples=num_samples
            )

        start_time = time.time()

        # Generate training data using the provided benchmarks
        print("   📊 Loading benchmark-specific training data...")
        training_data = []

        try:
            # Load data from the relevant benchmarks
            benchmark_data = self._load_benchmark_data(relevant_benchmarks, num_samples)
            training_data.extend(benchmark_data)
            print(f"      ✅ Loaded {len(benchmark_data)} examples from benchmarks")
        except Exception as e:
            print(f"      ⚠️ Failed to load benchmark data: {e}")

        # If we don't have enough data from benchmarks, supplement with synthetic data
        if len(training_data) < num_samples // 2:
            print("   🧪 Supplementing with synthetic training data...")
            try:
                synthetic_data = self._generate_synthetic_training_data(issue_type, num_samples - len(training_data))
                training_data.extend(synthetic_data)
                print(f"      ✅ Added {len(synthetic_data)} synthetic examples")
            except Exception as e:
                print(f"      ⚠️ Failed to generate synthetic data: {e}")

        if not training_data:
            raise ValueError(f"No training data available for {issue_type}")

        print(f"   📈 Total training examples: {len(training_data)}")

        # Extract activations
        print("   🧠 Extracting activations...")
        harmful_activations, harmless_activations = self._extract_activations_from_data(training_data, layer)

        # Train classifier
        print("   🎯 Training classifier...")
        classifier = self._train_classifier(harmful_activations, harmless_activations, config)

        # Evaluate performance
        print("   📈 Evaluating performance...")
        metrics = self._evaluate_classifier(classifier, harmful_activations, harmless_activations)

        training_time = time.time() - start_time

        # Save classifier if path provided
        save_path = None
        if config.save_path:
            print("   💾 Saving classifier...")
            save_path = self._save_classifier(classifier, config, metrics)

        result = TrainingResult(
            classifier=classifier.classifier,  # Return the base classifier
            config=config,
            performance_metrics=metrics,
            training_time=training_time,
            save_path=save_path,
        )

        print(
            f"   ✅ Benchmark-based classifier created in {training_time:.2f}s "
            f"(F1: {metrics.get('f1', 0):.3f}, Accuracy: {metrics.get('accuracy', 0):.3f})"
        )
        print(f"      📊 Used benchmarks: {relevant_benchmarks}")

        return result

    async def create_combined_benchmark_classifier(
        self, benchmark_names: List[str], classifier_params: "ClassifierParams", config: Optional[TrainingConfig] = None
    ) -> TrainingResult:
        """
        Create a classifier trained on combined data from multiple benchmarks.

        Args:
            benchmark_names: List of benchmark names to combine training data from
            classifier_params: Model-determined classifier parameters
            config: Optional training configuration

        Returns:
            TrainingResult with the trained combined classifier
        """
        print(f"🏗️ Creating combined classifier from {len(benchmark_names)} benchmarks...")
        print(f"   📊 Benchmarks: {benchmark_names}")
        print(f"   🧠 Using layer {classifier_params.optimal_layer}, {classifier_params.training_samples} samples")

        # Create config from classifier_params
        if config is None:
            config = TrainingConfig(
                issue_type=f"quality_combined_{'_'.join(sorted(benchmark_names))}",
                layer=classifier_params.optimal_layer,
                classifier_type=classifier_params.classifier_type,
                threshold=classifier_params.classification_threshold,
                training_samples=classifier_params.training_samples,
                model_name=self.model.name,
            )

        start_time = time.time()

        # Generate combined training data from all benchmarks
        print("   📊 Loading and combining benchmark training data...")
        combined_training_data = await self._load_combined_benchmark_data(
            benchmark_names, classifier_params.training_samples
        )

        print(f"   📈 Loaded {len(combined_training_data)} combined training examples")

        # Extract activations
        print("   🧠 Extracting activations...")
        harmful_activations, harmless_activations = self._extract_activations_from_data(
            combined_training_data, classifier_params.optimal_layer
        )

        # Train classifier
        print("   🎯 Training combined classifier...")
        classifier = self._train_classifier(harmful_activations, harmless_activations, config)

        # Evaluate performance
        print("   📈 Evaluating performance...")
        metrics = self._evaluate_classifier(classifier, harmful_activations, harmless_activations)

        training_time = time.time() - start_time

        # Save classifier if path provided
        save_path = None
        if config.save_path:
            print("   💾 Saving combined classifier...")
            save_path = self._save_classifier(classifier, config, metrics)

        result = TrainingResult(
            classifier=classifier.classifier,
            config=config,
            performance_metrics=metrics,
            training_time=training_time,
            save_path=save_path,
        )

        print(
            f"   ✅ Combined classifier created in {training_time:.2f}s "
            f"(F1: {metrics.get('f1', 0):.3f}, Accuracy: {metrics.get('accuracy', 0):.3f})"
        )

        return result

    async def _load_combined_benchmark_data(
        self, benchmark_names: List[str], total_samples: int
    ) -> List[Dict[str, Any]]:
        """
        Load and combine training data from multiple benchmarks.

        Args:
            benchmark_names: List of benchmark names to load data from
            total_samples: Total number of training samples to create

        Returns:
            Combined list of training examples with balanced sampling
        """
        combined_data = []
        samples_per_benchmark = max(1, total_samples // len(benchmark_names))

        print(f"      📊 Loading ~{samples_per_benchmark} samples per benchmark")

        for benchmark_name in benchmark_names:
            try:
                print(f"      🔄 Loading data from {benchmark_name}...")
                benchmark_data = self._load_benchmark_data([benchmark_name], samples_per_benchmark)
                combined_data.extend(benchmark_data)
                print(f"         ✅ Loaded {len(benchmark_data)} samples from {benchmark_name}")

            except Exception as e:
                print(f"         ⚠️ Failed to load {benchmark_name}: {e}")
                # Continue with other benchmarks
                continue

        # If we don't have enough samples, pad with synthetic data
        if len(combined_data) < total_samples:
            remaining_samples = total_samples - len(combined_data)
            print(f"      🔧 Generating {remaining_samples} synthetic samples to reach target")
            synthetic_data = self._generate_synthetic_training_data("quality", remaining_samples)
            combined_data.extend(synthetic_data)

        # Shuffle the combined data to ensure good mixing
        import random

        random.shuffle(combined_data)

        # Trim to exact target if we have too many
        combined_data = combined_data[:total_samples]

        print(f"      ✅ Final combined dataset: {len(combined_data)} samples")
        return combined_data

    async def create_classifier_for_issue(self, issue_type: str, layer: int = 15) -> TrainingResult:
        """
        Create a classifier for an issue type (async version for compatibility).

        Args:
            issue_type: Type of issue to detect
            layer: Model layer to use for activation extraction

        Returns:
            TrainingResult with the trained classifier
        """
        return self.create_classifier_for_issue_type(issue_type, layer)

    def _generate_training_data(self, issue_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generate training data dynamically for a specific issue type using relevant benchmarks.

        Args:
            issue_type: Type of issue to generate data for
            num_samples: Number of training samples to generate

        Returns:
            List of training examples with harmful/harmless pairs
        """
        print(f"   📊 Loading dynamic training data for {issue_type}...")

        # Try to find relevant benchmarks for the issue type (using default 5-minute budget)
        relevant_benchmarks = self._find_relevant_benchmarks(issue_type)

        if relevant_benchmarks:
            print(f"   🎯 Found {len(relevant_benchmarks)} relevant benchmarks: {relevant_benchmarks[:3]}...")
            return self._load_benchmark_data(relevant_benchmarks, num_samples)
        print("   🤖 No specific benchmarks found, using synthetic generation...")
        return self._generate_synthetic_training_data(issue_type, num_samples)

    def _find_relevant_benchmarks(self, issue_type: str, time_budget_minutes: float = 5.0) -> List[str]:
        """Find relevant benchmarks for the given issue type based on time budget with priority-aware selection."""
        from ..budget import calculate_max_tasks_for_time_budget
        from .tasks.task_relevance import find_relevant_tasks

        try:
            # Calculate max tasks using budget system
            max_tasks = calculate_max_tasks_for_time_budget(
                task_type="benchmark_evaluation", time_budget_minutes=time_budget_minutes
            )

            print(f"   🕐 Time budget: {time_budget_minutes:.1f}min → max {max_tasks} tasks")

            # Use priority-aware intelligent benchmark selection
            try:
                # Import priority-aware selection function
                import os
                import sys

                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lm-harness-integration"))
                from only_benchmarks import find_most_relevant_benchmarks

                # Use priority-aware selection with time budget
                relevant_results = find_most_relevant_benchmarks(
                    prompt=issue_type,
                    top_k=max_tasks,
                    priority="all",
                    fast_only=False,
                    time_budget_minutes=time_budget_minutes,
                    prefer_fast=True,  # Prefer fast benchmarks for agent use
                )

                # Extract benchmark names
                relevant_benchmarks = [result["benchmark"] for result in relevant_results]

                if relevant_benchmarks:
                    print(f"   📊 Found {len(relevant_benchmarks)} priority-aware benchmarks for '{issue_type}':")
                    for i, result in enumerate(relevant_results[:3]):
                        priority_str = f" (priority: {result.get('priority', 'unknown')})"
                        loading_time_str = f" (loading time: {result.get('loading_time', 60.0):.1f}s)"
                        print(f"      {i + 1}. {result['benchmark']}{priority_str}{loading_time_str}")
                    if len(relevant_benchmarks) > 3:
                        print(f"      ... and {len(relevant_benchmarks) - 3} more")

                return relevant_benchmarks

            except Exception as priority_error:
                print(f"   ⚠️ Priority-aware selection failed: {priority_error}")
                print("   🔄 Falling back to legacy task relevance...")

                # Fallback to legacy system
                relevant_task_results = find_relevant_tasks(
                    query=issue_type, max_results=max_tasks, min_relevance_score=0.1
                )

                # Extract just the task names
                candidate_benchmarks = [task_name for task_name, score in relevant_task_results]

                # Use priority-aware budget optimization
                from ..budget import optimize_benchmarks_for_budget

                relevant_benchmarks = optimize_benchmarks_for_budget(
                    task_candidates=candidate_benchmarks,
                    time_budget_minutes=time_budget_minutes,
                    max_tasks=max_tasks,
                    prefer_fast=True,  # Agent prefers fast benchmarks
                )

                if relevant_benchmarks:
                    print(f"   📊 Found {len(relevant_benchmarks)} relevant benchmarks for '{issue_type}':")
                    # Show the scores for the selected benchmarks
                    for i, (task_name, score) in enumerate(relevant_task_results[:3]):
                        if task_name in relevant_benchmarks:
                            print(f"      {i + 1}. {task_name} (relevance: {score:.3f})")
                    if len(relevant_benchmarks) > 3:
                        print(f"      ... and {len(relevant_benchmarks) - 3} more")

                return relevant_benchmarks

        except Exception as e:
            print(f"   ⚠️ Error finding relevant benchmarks: {e}")
            print("   ⚠️ Using fallback tasks")
            # Minimal fallback to high priority fast benchmarks
            return ["mmlu", "truthfulqa_mc1", "hellaswag"]

    def _extract_benchmark_concepts(self, benchmark_names: List[str]) -> Dict[str, List[str]]:
        """Extract semantic concepts from benchmark names."""
        concepts = {}

        for name in benchmark_names:
            # Extract concepts from benchmark name
            name_concepts = []
            name_lower = name.lower()

            # Split on common separators and extract meaningful tokens
            tokens = name_lower.replace("_", " ").replace("-", " ").split()

            # Filter out common non-semantic tokens
            semantic_tokens = []
            skip_tokens = {
                "the",
                "and",
                "or",
                "of",
                "in",
                "on",
                "at",
                "to",
                "for",
                "with",
                "by",
                "from",
                "as",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "can",
                "light",
                "full",
                "val",
                "test",
                "dev",
                "mc1",
                "mc2",
                "mt",
                "cot",
                "fewshot",
                "zeroshot",
                "generate",
                "until",
                "multiple",
                "choice",
                "group",
                "subset",
            }

            for token in tokens:
                if len(token) > 2 and token not in skip_tokens and token.isalpha():
                    semantic_tokens.append(token)

            # Extract domain-specific concepts
            domain_concepts = self._extract_domain_concepts(name_lower, semantic_tokens)
            name_concepts.extend(domain_concepts)

            concepts[name] = list(set(name_concepts))  # Remove duplicates

        return concepts

    def _extract_domain_concepts(self, benchmark_name: str, tokens: List[str]) -> List[str]:
        """Extract domain-specific concepts directly from benchmark name components."""
        concepts = []

        # Add all meaningful tokens as concepts
        for token in tokens:
            if len(token) > 2:
                concepts.append(token)

        # Extract compound concept meanings from token combinations
        name_parts = benchmark_name.lower().split("_")

        # Generate concept combinations
        for i, part in enumerate(name_parts):
            if len(part) > 2:
                concepts.append(part)

                # Look for meaningful compound concepts
                if i < len(name_parts) - 1:
                    next_part = name_parts[i + 1]
                    if len(next_part) > 2:
                        compound = f"{part}_{next_part}"
                        concepts.append(compound)

        # Extract semantic root words
        for token in tokens:
            root_concepts = self._extract_semantic_roots(token)
            concepts.extend(root_concepts)

        return list(set(concepts))  # Remove duplicates

    def _extract_semantic_roots(self, word: str) -> List[str]:
        """Extract semantic root concepts from a word."""
        roots = []

        # Simple morphological analysis
        # Remove common suffixes to find roots
        suffixes = [
            "ing",
            "tion",
            "sion",
            "ness",
            "ment",
            "able",
            "ible",
            "ful",
            "less",
            "ly",
            "al",
            "ic",
            "ous",
            "ive",
        ]

        root = word
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                root = word[: -len(suffix)]
                break

        if root != word and len(root) > 2:
            roots.append(root)

        # Add the original word
        roots.append(word)

        return roots

    def _calculate_benchmark_relevance(self, issue_type: str, benchmark_concepts: Dict[str, List[str]]) -> List[str]:
        """Calculate relevance scores using semantic similarity."""
        # Calculate relevance scores
        benchmark_scores = []

        for benchmark_name, concepts in benchmark_concepts.items():
            score = self._calculate_semantic_similarity(issue_type, benchmark_name, concepts)

            if score > 0:
                benchmark_scores.append((benchmark_name, score))

        # Sort by relevance score
        benchmark_scores.sort(key=lambda x: x[1], reverse=True)

        return [name for name, score in benchmark_scores]

    def _calculate_semantic_similarity(self, issue_type: str, benchmark_name: str, concepts: List[str]) -> float:
        """Calculate semantic similarity between issue type and benchmark."""
        issue_lower = issue_type.lower()
        benchmark_lower = benchmark_name.lower()

        score = 0.0

        # Direct name matching (highest weight)
        if issue_lower in benchmark_lower or benchmark_lower in issue_lower:
            score += 5.0

        # Concept matching
        for concept in concepts:
            concept_lower = concept.lower()

            # Exact concept match
            if issue_lower == concept_lower:
                score += 4.0
            # Partial concept match
            elif issue_lower in concept_lower or concept_lower in issue_lower:
                score += 2.0
            # Semantic similarity check
            elif self._are_semantically_similar(issue_lower, concept_lower):
                score += 1.5

        # Token-level similarity in benchmark name
        benchmark_tokens = benchmark_lower.replace("_", " ").replace("-", " ").split()
        issue_tokens = issue_lower.replace("_", " ").replace("-", " ").split()

        for issue_token in issue_tokens:
            for benchmark_token in benchmark_tokens:
                if len(issue_token) > 2 and len(benchmark_token) > 2:
                    if issue_token == benchmark_token:
                        score += 3.0
                    elif issue_token in benchmark_token or benchmark_token in issue_token:
                        score += 1.0
                    elif self._are_semantically_similar(issue_token, benchmark_token):
                        score += 0.5

        return score

    def _are_semantically_similar(self, term1: str, term2: str) -> bool:
        """Check if two terms are semantically similar using algorithmic methods."""
        if len(term1) < 3 or len(term2) < 3:
            return False

        # Character-level similarity
        overlap = len(set(term1) & set(term2))
        min_len = min(len(term1), len(term2))
        char_similarity = overlap / min_len

        # Substring similarity
        longer, shorter = (term1, term2) if len(term1) > len(term2) else (term2, term1)
        substring_match = shorter in longer

        # Prefix/suffix similarity
        prefix_len = 0
        suffix_len = 0

        for i in range(min(len(term1), len(term2))):
            if term1[i] == term2[i]:
                prefix_len += 1
            else:
                break

        for i in range(1, min(len(term1), len(term2)) + 1):
            if term1[-i] == term2[-i]:
                suffix_len += 1
            else:
                break

        affix_similarity = (prefix_len + suffix_len) / max(len(term1), len(term2))

        # Combined similarity score
        return char_similarity > 0.6 or substring_match or affix_similarity > 0.4 or prefix_len >= 3 or suffix_len >= 3

    def _prioritize_benchmarks(self, relevant_benchmarks: List[str]) -> List[str]:
        """Prioritize benchmarks algorithmically based on naming patterns and characteristics."""
        benchmark_scores = []

        for benchmark in relevant_benchmarks:
            score = self._calculate_benchmark_quality_score(benchmark)
            benchmark_scores.append((benchmark, score))

        # Sort by quality score (higher is better)
        benchmark_scores.sort(key=lambda x: x[1], reverse=True)
        return [benchmark for benchmark, score in benchmark_scores]

    def _calculate_benchmark_quality_score(self, benchmark_name: str) -> float:
        """Calculate quality score for a benchmark based on naming patterns and characteristics."""
        score = 0.0
        benchmark_lower = benchmark_name.lower()

        # Length heuristic - moderate length names tend to be well-established
        name_length = len(benchmark_name)
        if 8 <= name_length <= 25:
            score += 2.0
        elif name_length < 8:
            score += 0.5  # Very short names might be too simple
        else:
            score += 1.0  # Very long names might be overly specific

        # Component analysis
        parts = benchmark_lower.split("_")
        num_parts = len(parts)

        # Well-structured benchmarks often have 2-3 parts
        if 2 <= num_parts <= 3:
            score += 2.0
        elif num_parts == 1:
            score += 1.5  # Simple names can be good too
        else:
            score += 0.5  # Too many parts might indicate over-specification

        # Indicator of established benchmarks (avoid hardcoding specific names)
        quality_indicators = [
            # Multiple choice indicators (often well-validated)
            ("mc1", 1.5),
            ("mc2", 1.5),
            ("multiple_choice", 1.5),
            # Evaluation methodology indicators
            ("eval", 1.0),
            ("test", 1.0),
            ("benchmark", 1.0),
            # Language understanding indicators
            ("language", 1.0),
            ("understanding", 1.0),
            ("comprehension", 1.0),
            # Logic and reasoning indicators
            ("logic", 1.0),
            ("reasoning", 1.0),
            ("deduction", 1.0),
            # Knowledge assessment indicators
            ("knowledge", 1.0),
            ("question", 1.0),
            ("answer", 1.0),
        ]

        for indicator, points in quality_indicators:
            if indicator in benchmark_lower:
                score += points

        # Penalize very specialized or experimental indicators
        experimental_indicators = [
            "experimental",
            "pilot",
            "demo",
            "sample",
            "tiny",
            "mini",
            "subset",
            "light",
            "debug",
            "test_only",
        ]

        for indicator in experimental_indicators:
            if indicator in benchmark_lower:
                score -= 1.0

        # Bonus for domain diversity indicators
        domain_indicators = ["multilingual", "global", "cross", "multi", "diverse"]

        for indicator in domain_indicators:
            if indicator in benchmark_lower:
                score += 0.5

        return max(0.0, score)  # Ensure non-negative score

    def _load_benchmark_data(self, benchmarks: List[str], num_samples: int) -> List[Dict[str, Any]]:
        """Load training data from multiple relevant benchmarks."""
        from .tasks import TaskManager

        training_data = []
        samples_per_benchmark = max(1, num_samples // len(benchmarks))

        # Create task manager instance
        task_manager = TaskManager()

        for benchmark in benchmarks:
            try:
                print(f"     🔄 Loading from {benchmark}...")

                # Load benchmark task using TaskManager
                task_data = task_manager.load_task(benchmark, limit=samples_per_benchmark * 3)
                docs = task_manager.split_task_data(task_data, split_ratio=1.0)[0]

                # Extract QA pairs using existing system
                from ...contrastive_pairs.contrastive_pair_set import ContrastivePairSet

                qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(benchmark, task_data, docs)

                # Convert to training format
                for pair in qa_pairs[:samples_per_benchmark]:
                    if self._is_valid_pair(pair):
                        training_data.append(
                            {
                                "prompt": pair.get("question", f"Context from {benchmark}"),
                                "harmful_response": pair.get("incorrect_answer", ""),
                                "harmless_response": pair.get("correct_answer", ""),
                                "source": benchmark,
                            }
                        )

                print(f"     ✅ Loaded {len(qa_pairs[:samples_per_benchmark])} examples from {benchmark}")

                # Stop if we have enough data
                if len(training_data) >= num_samples:
                    break

            except Exception as e:
                print(f"     ⚠️ Failed to load {benchmark}: {e}")
                continue

        if not training_data:
            print("   ❌ Failed to load from any benchmarks, falling back to synthetic...")
            return self._generate_synthetic_training_data("unknown", num_samples)

        print(
            f"   ✅ Loaded {len(training_data)} examples from {len([b for b in benchmarks if any(b in item.get('source', '') for item in training_data)])} benchmarks"
        )
        return training_data[:num_samples]

    def _is_valid_pair(self, pair: Dict[str, Any]) -> bool:
        """Check if a QA pair is valid for training."""
        return (
            pair.get("correct_answer")
            and pair.get("incorrect_answer")
            and len(pair.get("correct_answer", "")) > 5
            and len(pair.get("incorrect_answer", "")) > 5
            and pair.get("correct_answer") != pair.get("incorrect_answer")
        )

    def _generate_synthetic_training_data(self, issue_type: str, num_samples: int) -> List[Dict[str, Any]]:
        """Generate synthetic training data for issue types."""
        from ...contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator

        try:
            generator = SyntheticContrastivePairGenerator(self.model)

            # Create trait description based on issue type
            trait_descriptions = {
                "quality": "providing clear, accurate, and well-structured responses",
                "harmful": "giving safe, helpful, and constructive advice",
                "bias": "responding with fairness and avoiding stereotypes",
                "coherence": "maintaining logical flow and clear communication",
                "hallucination": "providing factually accurate information",
            }

            trait_description = trait_descriptions.get(issue_type, f"avoiding {issue_type} issues in responses")

            # Generate synthetic pairs
            synthetic_pairs = generator.generate_contrastive_pair_set(
                trait_description=trait_description, num_pairs=num_samples, name=f"synthetic_{issue_type}"
            )

            # Convert to training format
            training_data = []
            for pair in synthetic_pairs.pairs[:num_samples]:
                training_data.append(
                    {
                        "prompt": pair.prompt or f"Context for {issue_type} detection",
                        "harmful_response": pair.negative_response,
                        "harmless_response": pair.positive_response,
                    }
                )

            print(f"   ✅ Generated {len(training_data)} synthetic examples for {issue_type}")
            return training_data

        except Exception as e:
            print(f"   ❌ Failed to generate synthetic data: {e}")
            raise ValueError(f"Cannot generate training data for issue type: {issue_type}")

    def _extract_activations_from_data(
        self, training_data: List[Dict[str, Any]], layer: int
    ) -> Tuple[List[Activations], List[Activations]]:
        """
        Extract activations from training data.

        Args:
            training_data: List of training examples
            layer: Layer to extract activations from

        Returns:
            Tuple of (harmful_activations, harmless_activations)
        """
        harmful_activations = []
        harmless_activations = []

        layer_obj = Layer(index=layer, type="transformer")

        for example in training_data:
            # Extract harmful activation
            harmful_tensor = self.model.extract_activations(example["harmful_response"], layer_obj)
            harmful_activation = Activations(tensor=harmful_tensor, layer=layer_obj)
            harmful_activations.append(harmful_activation)

            # Extract harmless activation
            harmless_tensor = self.model.extract_activations(example["harmless_response"], layer_obj)
            harmless_activation = Activations(tensor=harmless_tensor, layer=layer_obj)
            harmless_activations.append(harmless_activation)

        return harmful_activations, harmless_activations

    def _train_classifier(
        self, harmful_activations: List[Activations], harmless_activations: List[Activations], config: TrainingConfig
    ) -> ActivationClassifier:
        """
        Train a classifier on the activation data.

        Args:
            harmful_activations: List of harmful activations
            harmless_activations: List of harmless activations
            config: Training configuration

        Returns:
            Trained ActivationClassifier
        """
        classifier = ActivationClassifier(
            model_type=config.classifier_type, threshold=config.threshold, device=self.model.device
        )

        classifier.train_on_activations(harmful_activations, harmless_activations)

        return classifier

    def _evaluate_classifier(
        self,
        classifier: ActivationClassifier,
        harmful_activations: List[Activations],
        harmless_activations: List[Activations],
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            classifier: Trained classifier
            harmful_activations: Test harmful activations
            harmless_activations: Test harmless activations

        Returns:
            Dictionary of performance metrics
        """
        # Use a portion of data for testing
        test_size = min(10, len(harmful_activations) // 5)  # 20% or at least 10

        test_harmful = harmful_activations[-test_size:]
        test_harmless = harmless_activations[-test_size:]

        return classifier.evaluate_on_activations(test_harmful, test_harmless)

    def _save_classifier(
        self, classifier: ActivationClassifier, config: TrainingConfig, metrics: Dict[str, float]
    ) -> str:
        """
        Save classifier with metadata.

        Args:
            classifier: Trained classifier
            config: Training configuration
            metrics: Performance metrics

        Returns:
            Path where classifier was saved
        """
        # Create metadata
        metadata = create_classifier_metadata(
            model_name=config.model_name,
            task_name=config.issue_type,
            layer=config.layer,
            classifier_type=config.classifier_type,
            training_accuracy=metrics.get("accuracy", 0.0),
            training_samples=config.training_samples,
            token_aggregation="final",  # Default for our system
            detection_threshold=config.threshold,
            f1=metrics.get("f1", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            auc=metrics.get("auc", 0.0),
        )

        # Save using ModelPersistence
        save_path = ModelPersistence.save_classifier(classifier.classifier, config.layer, config.save_path, metadata)

        return save_path


def create_classifier_on_demand(
    model: Model, issue_type: str, layer: int = None, save_path: str = None, optimize: bool = False
) -> TrainingResult:
    """
    Convenience function to create a classifier on demand.

    Args:
        model: Language model to use
        issue_type: Type of issue to detect
        layer: Specific layer to use (auto-optimized if None)
        save_path: Path to save the classifier
        optimize: Whether to optimize for best performance

    Returns:
        TrainingResult with the created classifier
    """
    creator = ClassifierCreator(model)

    if optimize or layer is None:
        # Optimize to find best configuration
        result = creator.optimize_classifier_for_performance(issue_type)

        # Save if path provided
        if save_path:
            result.config.save_path = save_path
            result.save_path = creator._save_classifier(
                ActivationClassifier(device=model.device), result.config, result.performance_metrics
            )

        return result
    # Use specified layer
    config = TrainingConfig(issue_type=issue_type, layer=layer, save_path=save_path, model_name=model.name)

    return creator.create_classifier_for_issue_type(issue_type, layer, config)
