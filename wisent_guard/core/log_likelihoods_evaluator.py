"""
Log-Likelihoods Ground Truth Evaluator

This module handles ground truth evaluation for log-likelihoods based tasks,
typically used for multiple choice questions. Instead of generating text,
it loads the multiple choice options from lm-eval tasks and runs the classifier
directly on each choice to evaluate performance against known ground truth.
"""

import logging
from typing import Any, Dict, Optional

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations
from wisent_guard.core.layer import Layer

logger = logging.getLogger(__name__)


class LogLikelihoodsEvaluator:
    """
    Evaluator for log-likelihoods based ground truth assessment.

    This evaluator loads multiple choice options from lm-eval tasks and runs
    the classifier on each choice to evaluate performance against known ground truth.
    No text generation is performed - only direct classification evaluation.
    """

    def __init__(self, task_name: Optional[str] = None, model=None):
        """
        Initialize the log-likelihoods evaluator.

        Args:
            task_name: Name of the task (e.g., "truthfulqa_mc1", "mmlu", etc.)
            model: The model instance used to extract activations
        """
        self.task_name = task_name
        self.model = model

    def evaluate_classifier_on_task(
        self,
        classifier,
        task_name: str,
        num_samples: int = 100,
        model=None,
        layer: int = 15,
        token_aggregation: str = "average",
    ) -> Dict[str, Any]:
        """
        Evaluate a classifier on a log-likelihoods task by running it on multiple choice options.

        Args:
            classifier: The classifier to evaluate
            task_name: Name of the lm-eval task
            num_samples: Number of samples to evaluate (default: 100)
            model: The model instance (overrides self.model if provided)
            layer: Layer to extract activations from (default: 15)
            token_aggregation: Token aggregation method ("average", "final", "first", "max", "min")

        Returns:
            Dict containing evaluation results
        """
        try:
            # Use provided model or fall back to self.model
            evaluation_model = model or self.model
            if evaluation_model is None:
                return self._error_result("No model provided for activation extraction")

            logger.info(f"Loading task data for {task_name}...")

            # Use existing task loading infrastructure
            task_data = evaluation_model.load_lm_eval_task(task_name, shots=0, limit=num_samples)
            docs, _ = evaluation_model.split_task_data(task_data, split_ratio=1.0)  # Use all for evaluation

            if not docs:
                return self._error_result(f"No documents retrieved from task: {task_name}")

            logger.info(f"Retrieved {len(docs)} documents from {task_name}")

            # Use existing QA extraction infrastructure (task-agnostic)
            from .contrastive_pairs.contrastive_pair_set import ContrastivePairSet

            qa_pairs = ContrastivePairSet.extract_qa_pairs_from_task_docs(task_name, task_data, docs)

            if not qa_pairs:
                return self._error_result(f"No QA pairs could be extracted from task: {task_name}")

            logger.info(f"Extracted {len(qa_pairs)} QA pairs from {task_name}")

            # Use existing contrastive pair creation infrastructure
            from wisent_guard.core.activations.activation_collection_method import (
                ActivationCollectionLogic,
            )
            from wisent_guard.core.activations.prompts import PromptConstructionStrategy

            collector = ActivationCollectionLogic(model=evaluation_model)

            # For evaluation, use DIRECT_COMPLETION instead of MULTIPLE_CHOICE
            # This creates prompts like "Q" -> "good_resp"/"bad_resp" instead of "Which is better: Q A. bad B. good"
            logger.info("🔍 EVALUATION MODE: Using DIRECT_COMPLETION prompt strategy instead of MULTIPLE_CHOICE")
            contrastive_pairs = collector.create_batch_contrastive_pairs(
                qa_pairs, prompt_strategy=PromptConstructionStrategy.DIRECT_COMPLETION
            )

            if not contrastive_pairs:
                return self._error_result("No contrastive pairs could be created from QA pairs")

            logger.info(f"Created {len(contrastive_pairs)} contrastive pairs")

            # Map token aggregation to token targeting strategy for evaluation
            targeting_strategy_mapping = {  # TODO Refactor - we should stay with one standard
                "average": ActivationAggregationStrategy.MEAN_POOLING,
                "final": ActivationAggregationStrategy.LAST_TOKEN,
                "first": ActivationAggregationStrategy.FIRST_TOKEN,
                "max": ActivationAggregationStrategy.MAX_POOLING,
                "min": ActivationAggregationStrategy.MEAN_POOLING,  # Fallback to mean
            }

            targeting_strategy = targeting_strategy_mapping.get(
                token_aggregation, ActivationAggregationStrategy.MEAN_POOLING
            )

            logger.info(
                f"🔍 EVALUATION MODE: Using {targeting_strategy.value} targeting strategy (from token_aggregation: {token_aggregation})"
            )
            logger.info("🎯 ACTIVATION COLLECTION PARAMS:")
            logger.info(f"   • Layer: {layer}")
            logger.info(f"   • Device: {evaluation_model.device}")
            logger.info(f"   • Token targeting: {targeting_strategy.value}")
            logger.info(f"   • Pairs count: {len(contrastive_pairs)}")

            processed_pairs = collector.collect_activations_batch(
                pairs=contrastive_pairs,
                layer_index=layer,
                device=evaluation_model.device,
                token_targeting_strategy=targeting_strategy,
            )

            if not processed_pairs:
                return self._error_result("No activations could be extracted from contrastive pairs")

            logger.info(f"Extracted activations from {len(processed_pairs)} pairs")

            # Debug: Show where activations are collected from
            if processed_pairs:
                sample_pair = processed_pairs[0]
                logger.info("📍 DETAILED ACTIVATION COLLECTION ANALYSIS:")
                logger.info(f"   🔧 Sample pair type: {type(sample_pair).__name__}")
                logger.info(
                    f"   🔧 Pair attributes: {[attr for attr in dir(sample_pair) if not attr.startswith('_')][:8]}..."
                )

                if hasattr(sample_pair, "positive_activations") and sample_pair.positive_activations is not None:
                    logger.info(f"   ✅ Positive activations shape: {sample_pair.positive_activations.shape}")
                if hasattr(sample_pair, "negative_activations") and sample_pair.negative_activations is not None:
                    logger.info(f"   ✅ Negative activations shape: {sample_pair.negative_activations.shape}")

                if hasattr(sample_pair, "_prompt_pair") and sample_pair._prompt_pair:
                    logger.debug(f"   🔸 Positive prompt: {sample_pair._prompt_pair.positive_prompt[:100]}...")
                    logger.debug(f"   🔸 Negative prompt: {sample_pair._prompt_pair.negative_prompt[:100]}...")
                    logger.debug(f"   🎯 Target token: {sample_pair._prompt_pair.target_token}")
                    logger.debug(f"   📊 Prompt strategy: {sample_pair._prompt_strategy.value}")
                    logger.info(f"   🔍 Token targeting: {targeting_strategy.value} (evaluation mode)")
                elif hasattr(sample_pair, "prompt") and hasattr(sample_pair, "positive_response"):
                    logger.debug(f"   🔸 Question prompt: {sample_pair.prompt[:100]}...")
                    logger.debug(f"   ✅ Positive response: {sample_pair.positive_response[:50]}...")
                    logger.debug(f"   ❌ Negative response: {sample_pair.negative_response[:50]}...")
                    logger.debug(
                        f"   🔍 Token targeting used: {targeting_strategy.value} (from CLI token_aggregation: {token_aggregation})"
                    )
                else:
                    logger.info("   📍 ACTIVATION COLLECTION: Unknown format - investigating...")
                    logger.info(
                        f"   🔧 All attributes: {[attr for attr in dir(sample_pair) if not attr.startswith('__')]}"
                    )

            # Map token aggregation to activation method
            activation_method = token_aggregation
            # Handle both string and enum types
            method_name = activation_method.value if hasattr(activation_method, 'value') else str(activation_method)
            logger.info(
                f"🎯 Using activation aggregation method: {method_name} (from token_aggregation: {token_aggregation})"
            )

            # Evaluate classifier on each sample
            results = []
            total_correct = 0
            total_samples = 0

            for i, pair in enumerate(processed_pairs):
                try:
                    sample_result = self._evaluate_classifier_on_sample(
                        classifier, pair, qa_pairs[i], activation_method
                    )
                    results.append(sample_result)

                    if sample_result.get("classifier_correct", False):
                        total_correct += 1
                    total_samples += 1

                except Exception as e:
                    logger.error(f"Error evaluating sample {i}: {e}")
                    continue

            # Calculate overall metrics
            accuracy = total_correct / total_samples if total_samples > 0 else 0.0

            return {
                "ground_truth": "EVALUATED",
                "method_used": "log-likelihoods-classifier",
                "confidence": accuracy,
                "details": f"Evaluated {total_samples} samples with {total_correct} correct predictions",
                "task_name": task_name,
                "evaluation_method": "log-likelihoods",
                "lm_eval_metrics": {
                    "accuracy": accuracy,
                    "correct_predictions": total_correct,
                    "total_samples": total_samples,
                },
                "sample_results": results[:10],  # First 10 for debugging
            }

        except Exception as e:
            import traceback

            logger.error(f"Error evaluating classifier on task {task_name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._error_result(f"Evaluation error: {e!s}")

    def _evaluate_classifier_on_sample(
        self, classifier, processed_pair, qa_pair: Dict[str, Any], activation_method
    ) -> Dict[str, Any]:
        """
        Evaluate the classifier on a single processed contrastive pair.

        Args:
            classifier: The classifier to evaluate
            processed_pair: ContrastivePair with activations already extracted
            qa_pair: Original QA pair data for reference
            activation_method:
        Returns:
            Dict containing evaluation results for this sample
        """
        try:
            # Extract activations from the processed pair
            positive_activations = processed_pair.positive_activations  # B choice (correct)
            negative_activations = processed_pair.negative_activations  # A choice (incorrect)

            if positive_activations is None or negative_activations is None:
                return {
                    "question": qa_pair["question"],
                    "correct_answer": qa_pair["correct_answer"],
                    "incorrect_answer": qa_pair["incorrect_answer"],
                    "classifier_correct": False,
                    "error": "Missing activations",
                }

            layer_obj = Layer(index=15, type="transformer")

            # Process positive (correct) choice using CLI token aggregation method
            positive_act = Activations(
                tensor=positive_activations, layer=layer_obj, aggregation_strategy=activation_method
            )
            positive_features = positive_act.extract_features_for_classifier()
            positive_prediction = classifier.predict_proba([positive_features.cpu().numpy()])
            if isinstance(positive_prediction, (list, tuple)) and len(positive_prediction) > 0:
                positive_prediction = positive_prediction[0]

            # Process negative (incorrect) choice using CLI token aggregation method
            negative_act = Activations(
                tensor=negative_activations, layer=layer_obj, aggregation_strategy=activation_method
            )
            negative_features = negative_act.extract_features_for_classifier()
            negative_prediction = classifier.predict_proba([negative_features.cpu().numpy()])
            if isinstance(negative_prediction, (list, tuple)) and len(negative_prediction) > 0:
                negative_prediction = negative_prediction[0]

            # Evaluate classifier performance
            # For truthfulness: lower scores = more truthful/correct
            # Classifier should give lower score to correct answer, higher to incorrect
            classifier_correct = positive_prediction < negative_prediction

            # Enhanced logging to show classifier vs normal predictions
            logger.info("📋 Sample Evaluation:")
            logger.debug(f"   🔸 Question: {qa_pair['question'][:100]}...")
            logger.info("   🧠 CLASSIFIER PREDICTION:")
            logger.debug(f"      ✅ Correct answer score: {float(positive_prediction):.3f} (lower = more truthful)")
            logger.debug(f"      ❌ Incorrect answer score: {float(negative_prediction):.3f} (higher = less truthful)")
            logger.debug(
                f"      🎯 Classifier judgment: {'CORRECT' if classifier_correct else 'INCORRECT'} (positive {'<' if classifier_correct else '>='} negative)"
            )

            return {
                "question": qa_pair["question"],
                "correct_answer": qa_pair["correct_answer"],
                "incorrect_answer": qa_pair["incorrect_answer"],
                "positive_prediction": float(positive_prediction),
                "negative_prediction": float(negative_prediction),
                "classifier_correct": classifier_correct,
            }

        except Exception as e:
            logger.debug(f"Error evaluating sample: {e}")
            return {
                "question": qa_pair.get("question", "Unknown"),
                "correct_answer": qa_pair.get("correct_answer", "Unknown"),
                "incorrect_answer": qa_pair.get("incorrect_answer", "Unknown"),
                "classifier_correct": False,
                "error": str(e),
            }

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """Return an error result."""
        return {
            "ground_truth": "UNKNOWN",
            "method_used": "log-likelihoods-error",
            "confidence": 0.0,
            "details": error_msg,
            "task_name": self.task_name or "unknown",
            "evaluation_method": "log-likelihoods",
            "lm_eval_metrics": {"accuracy": 0.0, "correct_predictions": 0, "total_samples": 0},
        }
