"""
LM-Eval-Harness Ground Truth Evaluation

This module provides ground truth evaluation using the lm-eval-harness framework.
"""

import logging
from typing import Any, Dict

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations
from wisent_guard.core.layer import Layer

logger = logging.getLogger(__name__)


class LMEvalHarnessGroundTruth:
    """
    Ground truth evaluator using lm-eval-harness tasks.

    This class orchestrates the evaluation of classifiers on lm-eval-harness tasks
    by routing to appropriate evaluation methods based on the task type.
    """

    def __init__(self, task_name: str, evaluation_method: str = None, model=None):
        """
        Initialize the LM-eval-harness ground truth evaluator.

        Args:
            task_name: Name of the lm-eval task
            evaluation_method: Evaluation method ("log-likelihoods", "text-generation", "perplexity")
            model: The model instance for activation extraction
        """
        self.task_name = task_name
        self.evaluation_method = evaluation_method
        self.model = model

        # Load evaluation method from benchmark configuration if not provided
        if not self.evaluation_method:
            self.evaluation_method = self._get_evaluation_method_for_task(task_name)

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
        Evaluate a classifier on the specified lm-eval task.

        Args:
            classifier: The classifier to evaluate
            task_name: Name of the lm-eval task
            num_samples: Number of samples to evaluate
            model: The model instance (overrides self.model if provided)
            layer: Layer to extract activations from
            token_aggregation: Token aggregation method ("average", "final", "first", "max", "min")

        Returns:
            Dict containing evaluation results
        """
        # Use provided model or fall back to self.model
        evaluation_model = model or self.model

        # Route to appropriate evaluation method
        if self.evaluation_method == "log-likelihoods":
            return self._evaluate_log_likelihoods(
                classifier, task_name, num_samples, evaluation_model, layer, token_aggregation
            )
        if self.evaluation_method == "text-generation":
            return self._evaluate_text_generation(
                classifier, task_name, num_samples, evaluation_model, layer, token_aggregation
            )
        if self.evaluation_method == "perplexity":
            return self._evaluate_perplexity(
                classifier, task_name, num_samples, evaluation_model, layer, token_aggregation
            )
        if self.evaluation_method == "code-execution":
            return self._evaluate_code_execution(
                classifier, task_name, num_samples, evaluation_model, layer, token_aggregation
            )
        return {
            "ground_truth": "UNKNOWN",
            "method_used": "lm-eval-harness-unsupported",
            "confidence": 0.0,
            "details": f"Unsupported evaluation method: {self.evaluation_method}",
            "task_name": task_name,
            "evaluation_method": self.evaluation_method,
        }

    def _evaluate_log_likelihoods(
        self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average"
    ) -> Dict[str, Any]:
        """Evaluate classifier using log-likelihoods approach."""
        try:
            from .log_likelihoods_evaluator import LogLikelihoodsEvaluator

            # Create evaluator with model
            evaluator = LogLikelihoodsEvaluator(task_name, model=model)

            # Evaluate classifier
            results = evaluator.evaluate_classifier_on_task(
                classifier,
                task_name,
                num_samples=num_samples,
                model=model,
                layer=layer,
                token_aggregation=token_aggregation,
            )

            return results

        except Exception as e:
            logger.error(f"Error in log-likelihoods evaluation: {e}")
            return {
                "ground_truth": "UNKNOWN",
                "method_used": "lm-eval-harness-error",
                "confidence": 0.0,
                "details": f"Log-likelihoods evaluation failed: {e!s}",
                "task_name": task_name,
                "evaluation_method": "log-likelihoods",
            }

    def _evaluate_text_generation(
        self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average"
    ) -> Dict[str, Any]:
        """Evaluate classifier using text generation approach."""
        try:
            logger.info(f"🎯 TEXT GENERATION EVALUATION: {task_name}")

            # TODO In general LMEvalHarness should be rebuild to be BenchmarkGroundTruth
            # Check if this is a TaskInterface task
            if self._is_task_interface_task(task_name):
                docs, task_data = self._load_task_interface_data(task_name, num_samples)
            else:
                # Use existing lm-eval task loading infrastructure
                task_data = model.load_lm_eval_task(task_name, shots=0, limit=num_samples)
                docs, _ = model.split_task_data(task_data, split_ratio=1.0)  # Use all for evaluation

            if not docs:
                return self._error_result(f"No documents retrieved from task: {task_name}")

            logger.info(f"📝 Retrieved {len(docs)} documents from {task_name}")

            # Generate responses using the model
            generated_responses = []
            ground_truth_responses = []

            for i, doc in enumerate(docs):
                try:
                    # Extract question from document
                    if hasattr(task_data, "doc_to_text"):
                        question = task_data.doc_to_text(doc)
                    else:
                        question = str(doc.get("question", doc.get("text", "")))

                    # Generate response using model
                    logger.debug(f"🔸 Generating response for: {question[:100]}...")
                    generated_response, _ = model.generate(
                        prompt=question, layer_index=layer, max_new_tokens=150, temperature=0.1
                    )

                    # Extract ground truth answer
                    # HLE task handling
                    if task_name.startswith("hle") or task_name in ["math500", "math", "hendrycks_math"]:
                        ground_truth = doc.get("answer", "")
                    # AIME task handling
                    elif task_name.startswith("aime"):
                        ground_truth = str(doc.get("Answer", "") or doc.get("answer", ""))
                    # FIXED: For DROP task, use raw document data to preserve structured format
                    elif task_name == "drop":
                        # Use raw answer field which contains the structured data
                        ground_truth = doc.get("answer", {})
                    elif hasattr(task_data, "doc_to_target"):
                        ground_truth = task_data.doc_to_target(doc)
                    else:
                        ground_truth = str(doc.get("answer", doc.get("target", "")))

                    generated_responses.append(
                        {
                            "question": question,
                            "generated_response": generated_response,
                            "ground_truth": ground_truth,
                            "doc": doc,
                        }
                    )

                    logger.debug(f"   📝 Generated: {generated_response[:100]}...")
                    # FIXED: Handle ground_truth as int or string for logging
                    gt_str = str(ground_truth)
                    logger.debug(f"   ✅ Ground truth: {gt_str[:100]}...")

                except Exception as e:
                    logger.error(f"Error generating response for doc {i}: {e}")
                    continue

            # Evaluate using lm-eval-harness metrics
            logger.info(f"🎯 Evaluating {len(generated_responses)} generated responses using lm-eval metrics...")

            # Use lm-eval-harness's actual evaluation for this task
            evaluation_results = self._evaluate_with_lm_eval_metrics(task_name, generated_responses, task_data)

            # Now classify the generated responses to see if classifier agrees
            classification_results = []
            for response_data in generated_responses:
                try:
                    layer_obj = Layer(index=layer, type="transformer")

                    # Extract activations from generated response
                    activation_tensor = model.extract_activations(response_data["generated_response"], layer_obj)
                    activation_method = self._map_token_aggregation_to_activation_method(token_aggregation)

                    activation_obj = Activations(
                        tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method
                    )

                    # Get classifier prediction
                    features = activation_obj.extract_features_for_classifier()

                    # Handle different classifier return formats
                    try:
                        # Try predict_proba first (returns probabilities)
                        features_numpy = features.cpu().numpy()
                        logger.debug(f"🔧 Features shape: {features_numpy.shape}")

                        prediction_proba = classifier.predict_proba([features_numpy])
                        logger.debug(f"🔧 predict_proba returned: {prediction_proba} (type: {type(prediction_proba)})")

                        # Handle different return formats
                        if isinstance(prediction_proba, (list, tuple)):
                            if len(prediction_proba) > 0:
                                prediction = prediction_proba[0]
                                logger.debug(f"🔧 Extracted from list/tuple: {prediction} (type: {type(prediction)})")
                            else:
                                prediction = 0.5  # Default neutral score
                        else:
                            prediction = prediction_proba

                        # If prediction is array-like, get first element
                        if hasattr(prediction, "__len__") and not isinstance(prediction, str):
                            if len(prediction) > 0:
                                prediction = prediction[0]
                                logger.debug(f"🔧 Extracted from array: {prediction} (type: {type(prediction)})")
                            else:
                                prediction = 0.5

                        # Ensure we have a float
                        prediction = float(prediction)
                        logger.debug(f"🔧 Final prediction: {prediction}")

                    except Exception as proba_error:
                        logger.warning(f"predict_proba failed: {proba_error}, trying predict...")

                        # Fallback to predict method
                        try:
                            predictions = classifier.predict([features.cpu().numpy()])
                            if len(predictions) > 0:
                                prediction = float(predictions[0])
                            else:
                                logger.warning("Classifier predict returned empty array")
                                prediction = 0.5
                        except Exception as predict_error:
                            logger.error(f"Both predict_proba and predict failed: {predict_error}")
                            prediction = 0.5  # Default neutral score

                    classification_results.append(
                        {
                            "question": response_data["question"],
                            "generated_response": response_data["generated_response"],
                            "classifier_score": prediction,
                            "classifier_truthful": prediction < 0.5,  # Lower = more truthful
                        }
                    )

                    logger.debug("📋 Generated Response Classification:")
                    logger.debug(f"   🔸 Question: {response_data['question'][:100]}...")
                    logger.debug(f"   🧠 Generated: {response_data['generated_response'][:100]}...")
                    logger.debug(f"   🎯 Classifier score: {prediction:.3f} (lower = more truthful)")
                    logger.debug(f"   ✅ Classifier judgment: {'TRUTHFUL' if prediction < 0.5 else 'UNTRUTHFUL'}")

                except Exception as e:
                    import traceback

                    logger.error(f"Error classifying generated response: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    classification_results.append(
                        {
                            "question": response_data["question"],
                            "generated_response": response_data["generated_response"],
                            "classifier_score": 0.5,  # Default neutral score
                            "classifier_truthful": False,
                            "error": str(e),
                        }
                    )
                    continue

            return {
                "ground_truth": "EVALUATED",
                "method_used": "lm-eval-harness-text-generation",
                "confidence": evaluation_results.get("accuracy", 0.0),
                "details": f"Generated and evaluated {len(generated_responses)} responses using lm-eval metrics",
                "task_name": task_name,
                "evaluation_method": "text-generation",
                "lm_eval_metrics": evaluation_results,
                "classification_results": classification_results,
                "total_samples": len(generated_responses),
            }

        except Exception as e:
            logger.error(f"Error in text generation evaluation: {e}")
            return self._error_result(f"Text generation evaluation error: {e!s}")

    def _evaluate_perplexity(
        self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average"
    ) -> Dict[str, Any]:
        """Evaluate classifier using perplexity approach."""
        try:
            logger.info(f"🎯 PERPLEXITY EVALUATION: {task_name}")

            # Use existing task loading infrastructure
            task_data = model.load_lm_eval_task(task_name, shots=0, limit=num_samples)
            docs, _ = model.split_task_data(task_data, split_ratio=1.0)  # Use all for evaluation

            if not docs:
                return self._error_result(f"No documents retrieved from task: {task_name}")

            logger.info(f"📝 Retrieved {len(docs)} documents from {task_name}")

            # Calculate perplexity scores for different responses
            perplexity_results = []

            for i, doc in enumerate(docs):
                try:
                    # For WikiText and other pure language modeling tasks
                    if task_name == "wikitext":
                        # Get the full text for perplexity calculation
                        text = doc.get("page", doc.get("text", ""))
                        if not text:
                            logger.warning(f"No text found in WikiText document {i}")
                            continue

                        logger.debug(f"🔸 Calculating perplexity for WikiText document {i} ({len(text)} chars)...")

                        # Calculate perplexity on the full text
                        perplexity = self._calculate_perplexity(model, text)

                        # Extract activations from the text for classifier
                        try:
                            layer_obj = Layer(index=layer, type="transformer")

                            # Use a truncated version for activation extraction if text is too long
                            activation_text = text[:1000] if len(text) > 1000 else text
                            activation_tensor = model.extract_activations(activation_text, layer_obj)
                            activation_method = self._map_token_aggregation_to_activation_method(token_aggregation)

                            activation_obj = Activations(
                                tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method
                            )

                            # Get classifier prediction (only if classifier is provided)
                            if classifier is not None:
                                features = activation_obj.extract_features_for_classifier()

                                # Handle different classifier return formats
                                try:
                                    prediction_proba = classifier.predict_proba([features.cpu().numpy()])

                                    if isinstance(prediction_proba, (list, tuple)) and len(prediction_proba) > 0:
                                        classification_score = float(prediction_proba[0])
                                    else:
                                        classification_score = float(prediction_proba)

                                    if hasattr(classification_score, "__len__") and not isinstance(
                                        classification_score, str
                                    ):
                                        classification_score = float(classification_score[0])

                                except Exception as proba_error:
                                    logger.warning(f"predict_proba failed: {proba_error}, trying predict...")
                                    try:
                                        predictions = classifier.predict([features.cpu().numpy()])
                                        if len(predictions) > 0:
                                            classification_score = float(predictions[0])
                                        else:
                                            logger.warning("Classifier predict returned empty array")
                                            classification_score = 0.5
                                    except Exception as predict_error:
                                        logger.error(f"Both predict_proba and predict failed: {predict_error}")
                                        classification_score = 0.5
                            else:
                                # No classifier provided - use default neutral score for perplexity-only evaluation
                                classification_score = 0.5

                        except Exception as e:
                            logger.error(f"Error classifying WikiText document: {e}")
                            classification_score = None

                        result = {
                            "document_idx": i,
                            "text_preview": text[:200] + "..." if len(text) > 200 else text,
                            "text_length": len(text),
                            "perplexity": perplexity,
                            "classifier_score": classification_score,
                        }

                        perplexity_results.append(result)

                        logger.debug("📋 WikiText Perplexity Analysis:")
                        logger.debug(f"   📊 Document {i}: {len(text)} chars")
                        logger.debug(f"   🎯 Perplexity: {perplexity:.3f}")
                        if classification_score is not None:
                            logger.debug(f"   🧠 Classifier score: {classification_score:.3f} (lower = more truthful)")

                        continue  # Skip the rest of the loop for WikiText

                    # Extract question/prompt and possible completions for other tasks
                    if hasattr(task_data, "doc_to_text"):
                        prompt = task_data.doc_to_text(doc)
                    else:
                        prompt = str(doc.get("question", doc.get("text", "")))

                    # For multiple choice tasks, get all choices
                    choices = []
                    if hasattr(task_data, "doc_to_choice"):
                        choices = [
                            task_data.doc_to_choice(doc, choice_idx)
                            for choice_idx in range(len(doc.get("choices", [])))
                        ]
                    elif "choices" in doc:
                        choices = doc["choices"]
                    else:
                        # For non-multiple choice, generate a response and calculate its perplexity
                        generated_response, _ = model.generate(
                            prompt=prompt, layer_index=layer, max_new_tokens=100, temperature=0.1
                        )
                        choices = [generated_response]

                    logger.debug(f"🔸 Calculating perplexity for: {prompt[:100]}...")

                    # Calculate perplexity for each choice
                    choice_perplexities = []
                    for choice_idx, choice in enumerate(choices):
                        try:
                            # Calculate perplexity of the choice given the prompt
                            full_text = f"{prompt} {choice}"
                            perplexity = self._calculate_perplexity(model, full_text)

                            choice_perplexities.append(
                                {"choice_idx": choice_idx, "choice_text": choice, "perplexity": perplexity}
                            )

                            logger.debug(f"   📊 Choice {choice_idx}: {choice[:50]}... (perplexity: {perplexity:.3f})")

                        except Exception as e:
                            logger.error(f"Error calculating perplexity for choice {choice_idx}: {e}")
                            continue

                    # Get ground truth answer index
                    ground_truth_idx = None
                    if hasattr(task_data, "doc_to_target"):
                        ground_truth = task_data.doc_to_target(doc)
                        try:
                            ground_truth_idx = int(ground_truth)
                        except:
                            ground_truth_idx = None
                    elif "answer" in doc:
                        ground_truth_idx = doc["answer"]

                    # Find the choice with lowest perplexity (most likely)
                    if choice_perplexities:
                        best_choice = min(choice_perplexities, key=lambda x: x["perplexity"])

                        # Classify the best choice using the classifier
                        classification_score = None
                        try:
                            layer_obj = Layer(index=layer, type="transformer")

                            # Extract activations from the best choice
                            activation_tensor = model.extract_activations(best_choice["choice_text"], layer_obj)
                            activation_method = self._map_token_aggregation_to_activation_method(token_aggregation)

                            activation_obj = Activations(
                                tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method
                            )

                            # Get classifier prediction
                            features = activation_obj.extract_features_for_classifier()

                            # Handle different classifier return formats
                            try:
                                # Try predict_proba first (returns probabilities)
                                prediction_proba = classifier.predict_proba([features.cpu().numpy()])

                                # Handle different return formats
                                if isinstance(prediction_proba, (list, tuple)):
                                    if len(prediction_proba) > 0:
                                        classification_score = prediction_proba[0]
                                    else:
                                        classification_score = 0.5  # Default neutral score
                                else:
                                    classification_score = prediction_proba

                                # If prediction is array-like, get first element
                                if hasattr(classification_score, "__len__") and not isinstance(
                                    classification_score, str
                                ):
                                    if len(classification_score) > 0:
                                        classification_score = classification_score[0]
                                    else:
                                        classification_score = 0.5

                                # Ensure we have a float
                                classification_score = float(classification_score)

                            except Exception as proba_error:
                                logger.warning(f"predict_proba failed: {proba_error}, trying predict...")

                                # Fallback to predict method
                                try:
                                    predictions = classifier.predict([features.cpu().numpy()])
                                    if len(predictions) > 0:
                                        classification_score = float(predictions[0])
                                    else:
                                        logger.warning("Classifier predict returned empty array")
                                        classification_score = 0.5
                                except Exception as predict_error:
                                    logger.error(f"Both predict_proba and predict failed: {predict_error}")
                                    classification_score = 0.5  # Default neutral score

                        except Exception as e:
                            logger.error(f"Error classifying best choice: {e}")

                        result = {
                            "question": prompt,
                            "choices": choice_perplexities,
                            "best_choice_idx": best_choice["choice_idx"],
                            "best_choice_text": best_choice["choice_text"],
                            "best_choice_perplexity": best_choice["perplexity"],
                            "ground_truth_idx": ground_truth_idx,
                            "classifier_score": classification_score,
                            "perplexity_correct": best_choice["choice_idx"] == ground_truth_idx
                            if ground_truth_idx is not None
                            else None,
                        }

                        perplexity_results.append(result)

                        logger.debug("📋 Perplexity Analysis:")
                        logger.debug(f"   🔸 Question: {prompt[:100]}...")
                        logger.debug(f"   📊 Best choice (lowest perplexity): {best_choice['choice_text'][:100]}...")
                        logger.debug(f"   🎯 Perplexity: {best_choice['perplexity']:.3f}")
                        logger.debug(
                            f"   🧠 Classifier score: {classification_score:.3f} (lower = more truthful)"
                            if classification_score is not None
                            else "   🧠 Classifier score: N/A"
                        )
                        logger.debug(f"   ✅ Perplexity correct: {result['perplexity_correct']}")

                except Exception as e:
                    logger.error(f"Error processing doc {i}: {e}")
                    continue

            # Calculate overall metrics
            total_samples = len(perplexity_results)

            if task_name == "wikitext":
                # For WikiText, we don't have correct/incorrect, just perplexity values
                perplexities = [r["perplexity"] for r in perplexity_results if r["perplexity"] != float("inf")]
                avg_perplexity = sum(perplexities) / len(perplexities) if perplexities else float("inf")

                # Average classifier score
                classifier_scores = [
                    r["classifier_score"] for r in perplexity_results if r["classifier_score"] is not None
                ]
                avg_classifier_score = sum(classifier_scores) / len(classifier_scores) if classifier_scores else None

                perplexity_accuracy = 1.0 if avg_perplexity < 100 else 0.0  # Arbitrary threshold for "good" perplexity
                correct_perplexity = sum(1 for r in perplexity_results if r["perplexity"] < 100)
            else:
                correct_perplexity = sum(1 for r in perplexity_results if r.get("perplexity_correct") == True)
                perplexity_accuracy = correct_perplexity / total_samples if total_samples > 0 else 0.0

                # Average classifier score
                classifier_scores = [
                    r["classifier_score"] for r in perplexity_results if r["classifier_score"] is not None
                ]
                avg_classifier_score = sum(classifier_scores) / len(classifier_scores) if classifier_scores else None

            logger.info("📊 PERPLEXITY EVALUATION RESULTS:")
            logger.info(f"   • Total samples: {total_samples}")
            if task_name == "wikitext":
                logger.info(f"   • Average perplexity: {avg_perplexity:.3f}")
                logger.info(f"   • Documents with perplexity < 100: {correct_perplexity}")
            else:
                logger.info(f"   • Perplexity accuracy: {perplexity_accuracy:.3f}")
            logger.info(
                f"   • Average classifier score: {avg_classifier_score:.3f}"
                if avg_classifier_score is not None
                else "   • Average classifier score: N/A"
            )

            result_dict = {
                "ground_truth": "EVALUATED",
                "method_used": "lm-eval-harness-perplexity",
                "confidence": perplexity_accuracy,
                "details": f"Calculated perplexity for {total_samples} samples",
                "task_name": task_name,
                "evaluation_method": "perplexity",
                "perplexity_accuracy": perplexity_accuracy,
                "average_classifier_score": avg_classifier_score,
                "total_samples": total_samples,
                "correct_perplexity": correct_perplexity,
                "perplexity_results": perplexity_results[:10],  # First 10 for debugging
            }

            if task_name == "wikitext":
                result_dict["average_perplexity"] = avg_perplexity
                result_dict["details"] = (
                    f"Calculated perplexity for {total_samples} WikiText documents, avg perplexity: {avg_perplexity:.3f}"
                )
            else:
                result_dict["details"] = (
                    f"Calculated perplexity for {total_samples} samples, accuracy: {perplexity_accuracy:.3f}"
                )

            return result_dict

        except Exception as e:
            logger.error(f"Error in perplexity evaluation: {e}")
            return self._error_result(f"Perplexity evaluation error: {e!s}")

    def _get_evaluation_method_for_task(self, task_name: str) -> str:
        """Get the evaluation method for a task from the benchmark configuration."""
        try:
            import json

            eval_methods_path = "wisent_guard/parameters/benchmarks/benchmark_evaluation_methods.json"
            with open(eval_methods_path) as f:
                benchmark_methods = json.load(f)
                return benchmark_methods.get(task_name, "text-generation")
        except Exception as e:
            logger.debug(f"Could not load benchmark evaluation methods: {e}")
            return "text-generation"

    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return a standardized error result."""
        return {
            "ground_truth": "ERROR",
            "method_used": "lm-eval-harness-error",
            "confidence": 0.0,
            "details": error_message,
            "task_name": self.task_name,
            "evaluation_method": self.evaluation_method,
        }

    def _map_token_aggregation_to_activation_method(self, token_aggregation: str):
        """Map token aggregation string to activation method."""

        mapping = {  # TODO This should be refactor, why we use strings as Token aggregation?
            "average": ActivationAggregationStrategy.MEAN_POOLING,
            "mean": ActivationAggregationStrategy.MEAN_POOLING,
            "last": ActivationAggregationStrategy.LAST_TOKEN,
            "max": ActivationAggregationStrategy.MAX_POOLING,
        }

        return mapping.get(token_aggregation.lower(), ActivationAggregationStrategy.MEAN_POOLING)

    def _is_task_interface_task(self, task_name: str) -> bool:
        """Check if this is a TaskInterface task (not an lm-eval task)."""
        # List of known TaskInterface tasks
        task_interface_tasks = {
            "hle",
            "hle_exact_match",
            "hle_multiple_choice",
            "livecodebench",
            "math500",
            "math",
            "hendrycks_math",
            "aime",
            "aime2025",
            "aime2024",
            "hmmt",
            "hmmt_feb_2025",
            "polymath",
            "polymath_en_medium",
            "polymath_zh_medium",
            "polymath_en_high",
            "polymath_zh_high",
            "livemathbench",
            "livemathbench_cnmo_en",
            "livemathbench_cnmo_zh",
        }
        return task_name in task_interface_tasks

    def _load_task_interface_data(self, task_name: str, num_samples: int):
        """Load data from TaskInterface tasks."""
        try:
            from .task_interface import get_task

            # Get the task instance
            task = get_task(task_name)

            # Load data
            docs = task.load_data(limit=num_samples)

            return docs, task

        except Exception as e:
            logger.error(f"Failed to load TaskInterface task {task_name}: {e}")
            return [], None

    def _calculate_perplexity(self, model, text: str) -> float:
        """Calculate perplexity of text using the model."""
        try:
            import numpy as np
            import torch

            # Use the model's prepare_activations method to get outputs
            prepared = model.prepare_activations(text)
            outputs = prepared["outputs"]
            inputs = prepared["inputs"]

            # Get input IDs
            input_ids = inputs["input_ids"]

            # Get logits from the outputs
            logits = outputs.logits

            # Compute log probabilities
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get log probabilities for actual tokens (shifted for next-token prediction)
            # input_ids shape: [batch_size, sequence_length]
            # logits shape: [batch_size, sequence_length, vocab_size]
            # We need to match targets with predictions

            if input_ids.shape[1] > 1:
                # Get log probabilities for the target tokens
                target_ids = input_ids[0, 1:]  # Skip first token (no prediction for it)
                prediction_logits = log_probs[0, :-1, :]  # Skip last prediction (no target for it)

                # Get log probabilities for actual tokens
                token_log_probs = prediction_logits.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

                # Compute average log probability
                avg_log_prob = token_log_probs.mean().item()

                # Compute perplexity
                perplexity = np.exp(-avg_log_prob)
            else:
                # Single token, cannot compute perplexity
                perplexity = float("inf")

            return perplexity

        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return float("inf")

    def _evaluate_generic_code_execution(
        self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average"
    ) -> Dict[str, Any]:
        """Evaluate generic code execution tasks (non-BigCode) like LiveCodeBench."""
        try:
            logger.info(f"🎯 GENERIC CODE EXECUTION EVALUATION: {task_name}")

            # Get secure code evaluator
            from .secure_code_evaluator import SecureCodeEvaluator

            secure_evaluator = SecureCodeEvaluator()

            # Load task data
            task_data = model.load_lm_eval_task(task_name, shots=0, limit=num_samples)

            if hasattr(task_data, "test_docs"):
                docs = task_data.test_docs()
            else:
                docs, _ = model.split_task_data(task_data, split_ratio=1.0)

            if not docs:
                return self._error_result(f"No documents retrieved from task: {task_name}")

            logger.info(f"📝 Retrieved {len(docs)} documents from {task_name}")

            # Generate code for each sample
            generated_codes = []
            evaluation_results = []

            for i, doc in enumerate(docs):
                try:
                    # Get prompt
                    if hasattr(task_data, "doc_to_text"):
                        prompt = task_data.doc_to_text(doc)
                    else:
                        # For LiveCodeBench
                        question = doc.get("question_content", doc.get("text", ""))
                        starter_code = doc.get("starter_code", "")
                        prompt = f"{question}\n\n{starter_code}" if starter_code else question

                    logger.debug(f"📋 Prompt for sample {i + 1}:\n{prompt[:200]}...\n")

                    # Generate code using model
                    logger.debug(f"🔸 Generating code for sample {i + 1}/{len(docs)}...")
                    generated_code, _ = model.generate(
                        prompt=prompt,
                        layer_index=layer,
                        max_new_tokens=500,  # More tokens for code generation
                        temperature=0.1,
                    )

                    generated_codes.append(generated_code)
                    logger.debug(f"   📝 Generated code:\n{generated_code}\n")

                    # Evaluate generated code
                    eval_result = secure_evaluator.evaluate_response(task_name, doc, generated_code)
                    evaluation_results.append(eval_result)

                    logger.debug(
                        f"   ✅ Evaluation result: {'PASSED' if eval_result.get('passed', False) else 'FAILED'}"
                    )
                    if "pass_rate" in eval_result:
                        logger.debug(f"   📊 Pass rate: {eval_result['pass_rate']:.2%}")

                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    generated_codes.append("")
                    evaluation_results.append({"passed": False, "error": str(e), "success": False})

            # Aggregate results
            total_passed = sum(1 for r in evaluation_results if r.get("passed", False))
            accuracy = total_passed / len(evaluation_results) if evaluation_results else 0.0

            logger.info(
                f"📊 CODE EXECUTION COMPLETED: {total_passed}/{len(evaluation_results)} passed ({accuracy:.2%})"
            )

            # Clean up Docker resources
            secure_evaluator.cleanup()

            return {
                "ground_truth": "EVALUATED",
                "method_used": f"generic-code-execution-{task_name}",
                "confidence": accuracy,
                "accuracy": accuracy,
                "details": f"Executed and evaluated {len(generated_codes)} code samples",
                "task_name": task_name,
                "evaluation_method": "code-execution",
                "total_samples": len(generated_codes),
                "passed_samples": total_passed,
                "evaluation_results": evaluation_results,
            }

        except Exception as e:
            logger.error(f"Error in generic code execution evaluation: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._error_result(f"Generic code execution evaluation error: {e!s}")

    def _evaluate_with_lm_eval_metrics(self, task_name: str, response_data: list, task_data) -> Dict[str, Any]:
        """Evaluate responses using task-specific evaluation metrics."""
        try:
            correct = 0
            total = len(response_data)
            evaluation_details = []

            for response in response_data:
                generated = response["generated_response"]
                ground_truth = response["ground_truth"]

                # Task-specific evaluation logic
                if task_name == "gsm8k":
                    # GSM8K uses exact match on numerical answer
                    is_correct = self._evaluate_gsm8k_response(generated, ground_truth)
                elif task_name.startswith("math") or task_name in ["hendrycks_math"]:
                    # MATH-500 and related benchmarks use same evaluation as GSM8K (numerical extraction)
                    is_correct = self._evaluate_gsm8k_response(generated, ground_truth)
                elif task_name in ["arc_easy", "arc_challenge"]:
                    # ARC uses exact match on choice letter/number
                    is_correct = self._evaluate_arc_response(generated, ground_truth)
                elif task_name == "hellaswag":
                    # HellaSwag uses exact match on choice index
                    is_correct = self._evaluate_hellaswag_response(generated, ground_truth)
                elif task_name == "mathqa":
                    # MATH_QA uses exact match on choice index (0, 1, 2, 3)
                    is_correct = self._evaluate_mathqa_response(generated, ground_truth)
                elif task_name == "drop":
                    # DROP uses structured answer format with numbers, spans, and dates
                    is_correct = self._evaluate_drop_response(generated, ground_truth)
                elif task_name.startswith("gpqa"):
                    # GPQA uses multiple-choice answer extraction (A, B, C, D)
                    is_correct = self._evaluate_multiple_choice_response(generated, ground_truth)
                elif task_name.startswith("hle") and "multiple_choice" in task_name:
                    # HLE multiple choice uses letter extraction (A, B, C, D, E)
                    is_correct = self._evaluate_multiple_choice_response(generated, ground_truth)
                elif task_name.startswith("truthfulqa") or task_name == "truthfulqa_mc1":
                    # TruthfulQA uses multiple-choice answer extraction (A, B, C, D)
                    is_correct = self._evaluate_multiple_choice_response(generated, ground_truth)
                else:
                    # Default: string matching with some flexibility
                    is_correct = self._evaluate_default_response(generated, ground_truth)

                if is_correct:
                    correct += 1

                evaluation_details.append(
                    {
                        "question": response["question"][:100],
                        "generated": generated[-50:],
                        "ground_truth": ground_truth,
                        "correct": is_correct,
                    }
                )

                logger.debug(f"📊 Evaluation: {response['question'][:50]}...")
                logger.debug(f"   Generated: {generated[:50]}...")
                logger.debug(f"   Ground Truth: {ground_truth}")
                logger.debug(f"   Correct: {is_correct}")

            accuracy = correct / total if total > 0 else 0.0

            return {
                "accuracy": accuracy,
                "correct_predictions": correct,
                "total_samples": total,
                "evaluation_details": evaluation_details[:5],  # First 5 for debugging
                "task_name": task_name,
            }

        except Exception as e:
            logger.error(f"Error in metrics evaluation: {e}")
            return {"accuracy": 0.0, "correct_predictions": 0, "total_samples": len(response_data), "error": str(e)}

    def _evaluate_gsm8k_response(self, generated: str, ground_truth) -> bool:
        """Evaluate GSM8K response using numerical answer extraction."""
        try:
            # Extract numerical answer from generated response
            # GSM8K answers are typically in format "#### 42" or just the number
            generated_answer = self._extract_numerical_answer(generated)
            ground_truth_answer = self._extract_numerical_answer(str(ground_truth))

            # Compare numerical values
            if generated_answer is not None and ground_truth_answer is not None:
                return abs(generated_answer - ground_truth_answer) < 1e-6

            # Fallback to string matching
            return generated.strip().lower() == str(ground_truth).strip().lower()

        except Exception as e:
            logger.error(f"Error evaluating GSM8K response: {e}")
            return False

    def _extract_numerical_answer(self, text: str) -> float:
        """Extract numerical answer from text."""
        try:
            import re

            # Look for #### pattern (GSM8K format)
            pattern = r"####\s*([+-]?\d+(?:\.\d+)?)"
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))

            # Look for last number in text
            numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
            if numbers:
                return float(numbers[-1])

            return None

        except Exception as e:
            logger.error(f"Error extracting numerical answer: {e}")
            return None

    def _evaluate_arc_response(self, generated: str, ground_truth) -> bool:
        """Evaluate ARC response using exact match."""
        try:
            # Normalize responses
            gen_clean = generated.strip().lower()
            gt_clean = str(ground_truth).strip().lower()

            # Direct match
            if gen_clean == gt_clean:
                return True

            # Check if generated contains the ground truth
            if gt_clean in gen_clean:
                return True

            # Check for choice letter/number patterns
            import re

            gen_match = re.search(r"[abcd]|\d+", gen_clean)
            gt_match = re.search(r"[abcd]|\d+", gt_clean)

            if gen_match and gt_match:
                return gen_match.group() == gt_match.group()

            return False

        except Exception as e:
            logger.error(f"Error evaluating ARC response: {e}")
            return False

    def _evaluate_hellaswag_response(self, generated: str, ground_truth) -> bool:
        """Evaluate HellaSwag response using exact match."""
        try:
            # Normalize and compare
            gen_clean = generated.strip().lower()
            gt_clean = str(ground_truth).strip().lower()

            return gen_clean == gt_clean or gt_clean in gen_clean

        except Exception as e:
            logger.error(f"Error evaluating HellaSwag response: {e}")
            return False

    def _evaluate_mathqa_response(self, generated: str, ground_truth) -> bool:
        """Evaluate MATH_QA response using choice matching."""
        try:
            import re

            # Ground truth is typically 0, 1, 2, or 3 (choice index)
            gt_str = str(ground_truth).strip()

            # Look for choice patterns in generated response
            gen_clean = generated.strip().lower()

            # Direct match with choice index
            if gt_str in gen_clean:
                return True

            # Look for choice letter patterns (a=0, b=1, c=2, d=3)
            choice_map = {"a": "0", "b": "1", "c": "2", "d": "3"}
            for letter, index in choice_map.items():
                if index == gt_str and letter in gen_clean:
                    return True

            # Look for explicit choice pattern like "The answer is 1" or "Choice B"
            choice_patterns = [
                rf"\b{gt_str}\b",  # Exact number match
                rf"choice\s*{choice_map.get(gt_str, gt_str)}",  # "choice 1"
                rf"answer\s*is\s*{gt_str}",  # "answer is 1"
                rf"option\s*{gt_str}",  # "option 1"
            ]

            for pattern in choice_patterns:
                if re.search(pattern, gen_clean):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error evaluating MATH_QA response: {e}")
            return False

    def _evaluate_drop_response(self, generated: str, ground_truth) -> bool:
        """Evaluate DROP response using structured answer format."""
        try:
            import json
            import re

            # Parse ground truth if it's a string representation of a dict
            if isinstance(ground_truth, str):
                try:
                    # Try to parse as JSON first
                    if ground_truth.startswith("{"):
                        gt_dict = json.loads(ground_truth)
                    else:
                        # Handle malformed string representations
                        return False
                except:
                    return False
            elif isinstance(ground_truth, dict):
                gt_dict = ground_truth
            else:
                return False

            gen_clean = generated.strip().lower()

            # Check number field
            if gt_dict.get("number"):
                number_str = str(gt_dict["number"]).strip()
                if number_str:
                    # Direct number match
                    if number_str.lower() in gen_clean:
                        return True

                    # Try to extract numbers from generated response
                    gen_numbers = re.findall(r"\b\d+\b", generated)
                    if number_str in gen_numbers:
                        return True

                    # Word number matching (e.g., "two" vs "2")
                    number_words = {
                        "0": ["zero", "none"],
                        "1": ["one"],
                        "2": ["two"],
                        "3": ["three"],
                        "4": ["four"],
                        "5": ["five"],
                        "6": ["six"],
                        "7": ["seven"],
                        "8": ["eight"],
                        "9": ["nine"],
                        "10": ["ten"],
                    }
                    if number_str in number_words:
                        for word in number_words[number_str]:
                            if word in gen_clean:
                                return True

            # Check spans field
            if gt_dict.get("spans"):
                spans = gt_dict["spans"]
                if isinstance(spans, list):
                    for span in spans:
                        span_clean = str(span).strip().lower()
                        if span_clean and span_clean in gen_clean:
                            return True
                elif isinstance(spans, str):
                    span_clean = spans.strip().lower()
                    if span_clean and span_clean in gen_clean:
                        return True

            # Check date field (less common but possible)
            if gt_dict.get("date"):
                date_obj = gt_dict["date"]
                if isinstance(date_obj, dict):
                    # Check individual date components
                    for component in ["day", "month", "year"]:
                        if date_obj.get(component):
                            date_val = str(date_obj[component]).strip().lower()
                            if date_val and date_val in gen_clean:
                                return True

            return False

        except Exception as e:
            logger.error(f"Error evaluating DROP response: {e}")
            return False

    def _evaluate_default_response(self, generated: str, ground_truth) -> bool:
        """Default evaluation using flexible string matching."""
        try:
            gen_clean = generated.strip().lower()

            # Handle list ground truth (e.g., COQA format)
            if isinstance(ground_truth, list):
                # Check if generated response matches any of the acceptable answers
                for gt_option in ground_truth:
                    gt_clean = str(gt_option).strip().lower()

                    # Exact match
                    if gen_clean == gt_clean:
                        return True

                    # Contains match
                    if gt_clean in gen_clean or gen_clean in gt_clean:
                        return True

                return False
            # Handle string ground truth
            gt_clean = str(ground_truth).strip().lower()

            # Exact match
            if gen_clean == gt_clean:
                return True

            # Contains match
            if gt_clean in gen_clean or gen_clean in gt_clean:
                return True

            return False

        except Exception as e:
            logger.error(f"Error in default evaluation: {e}")
            return False

    def _evaluate_multiple_choice_response(self, generated: str, ground_truth) -> bool:
        """Evaluate multiple choice response by extracting choice letter (A, B, C, D, E)."""
        import re

        try:
            # Clean the generated response
            gen_clean = generated.strip()

            # Convert ground truth to string and extract expected letter
            gt_str = str(ground_truth).strip()
            expected_letter = None

            # Extract letter from ground truth (could be "(A)", "A", etc.)
            gt_match = re.search(r"[ABCDE]", gt_str.upper())
            if gt_match:
                expected_letter = gt_match.group()
            else:
                return False

            # Try multiple strict patterns to extract answer from generated response
            # These patterns require clear context indicating an intentional choice
            patterns = [
                # Fixed pattern to avoid matching 'A' in "Answer:" alone
                r"(?:answer|choice|option)\s*(?:is\s+|:\s*)(?:\()?([ABCDE])(?:\))?",  # "Answer: A" or "Answer is (B)" - requires letter after
                r"the\s+(?:correct\s+)?answer\s+is\s*(?:\()?([ABCDE])(?:\))?",  # "The answer is A" - requires "the answer is"
                r"(?:select|choose)\s+(?:\()?([ABCDE])(?:\))?",  # "Select A" or "Choose A" - requires the action word
                r"(?:^|\n)([ABCDE])(?:\s*$)",  # Letter at start of line followed by whitespace/end only
                r"^([ABCDE])[.,;!?)\s]*$",  # Just the letter with optional punctuation and whitespace
                r"^(?:\()?([ABCDE])(?:\))?\s*$",  # Just the letter with optional parentheses
            ]

            # Try each pattern - only accept clear, intentional responses
            for pattern in patterns:
                matches = re.finditer(pattern, gen_clean.upper(), re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    extracted_letter = match.group(1).upper()
                    if extracted_letter == expected_letter:
                        return True

            # No more fallback - if we can't clearly identify the choice, it's wrong
            return False

        except Exception as e:
            logger.error(f"Error evaluating multiple choice response: {e}")
            return False

    def _evaluate_code_execution(
        self, classifier, task_name: str, num_samples: int, model, layer: int, token_aggregation: str = "average"
    ) -> Dict[str, Any]:
        """Evaluate classifier using code execution approach for BigCode tasks."""
        try:
            logger.debug(f"🎯 CODE EXECUTION EVALUATION: {task_name}")

            # Check if it's a BigCode task
            from .bigcode_integration import get_bigcode_evaluator, is_bigcode_task, load_bigcode_task
            from .secure_code_evaluator import SecureCodeEvaluator

            if not is_bigcode_task(task_name):
                # Check if it's still a code execution task (like LiveCodeBench)
                if SecureCodeEvaluator.is_code_execution_task(task_name):
                    logger.info(f"Task {task_name} is a non-BigCode code execution task")
                    return self._evaluate_generic_code_execution(
                        classifier, task_name, num_samples, model, layer, token_aggregation
                    )
                logger.warning(f"Task {task_name} is not a code execution task, falling back to text generation")
                return self._evaluate_text_generation(
                    classifier, task_name, num_samples, model, layer, token_aggregation
                )

            # Load BigCode task
            bigcode_task = load_bigcode_task(task_name, limit=num_samples)
            logger.info(f"📝 Loaded BigCode task {task_name} with {len(bigcode_task)} samples")

            # Generate code for each sample
            generated_codes = []
            for i, sample in enumerate(bigcode_task.get_samples()):
                try:
                    # Get prompt
                    prompt = bigcode_task.doc_to_text(sample)
                    logger.debug(f"📋 Prompt for sample {i + 1}:\n{prompt}\n")

                    # Generate code using model
                    logger.debug(f"🔸 Generating code for sample {i + 1}/{len(bigcode_task)}...")
                    generated_code, _ = model.generate(
                        prompt=prompt,
                        layer_index=layer,
                        max_new_tokens=300,  # More tokens for code generation
                        temperature=0.1,
                        # Note: stop_sequences not supported by all models
                    )

                    generated_codes.append(generated_code)
                    logger.debug(f"   📝 Generated: {generated_code[:100]}...")
                    logger.debug(f"   📝 Full generated code:\n{generated_code}\n")

                except Exception as e:
                    logger.error(f"Error generating code for sample {i}: {e}")
                    generated_codes.append("")  # Empty code for failed generation

            # Evaluate generated code using BigCode evaluator
            logger.info(f"🎯 Evaluating {len(generated_codes)} generated code samples...")

            # Get Docker executor if available
            docker_executor = None
            try:
                from .docker import OptimizedDockerExecutor

                docker_executor = OptimizedDockerExecutor()
            except Exception as e:
                logger.warning(f"Docker executor not available: {e}")

            # Use BigCode evaluator
            evaluator = get_bigcode_evaluator(docker_executor)

            # Prepare generations in expected format (list of lists)
            generations_for_eval = [[code] for code in generated_codes]

            # Run evaluation
            evaluation_results = evaluator.evaluate(
                bigcode_task,
                generations_for_eval,
                k_values=[1],  # Just pass@1 for now
            )

            # Extract pass rate
            pass_rate = evaluation_results.get("pass_at_k", {}).get("pass@1", 0.0)

            logger.info(f"✅ Code execution pass@1: {pass_rate:.2%}")

            # Now classify the generated code to see if classifier agrees
            classification_results = []
            for i, code in enumerate(generated_codes):
                try:
                    layer_obj = Layer(index=layer, type="transformer")

                    # Extract activations from generated code
                    activation_tensor = model.extract_activations(code, layer_obj)
                    activation_method = self._map_token_aggregation_to_activation_method(token_aggregation)

                    activation_obj = Activations(
                        tensor=activation_tensor, layer=layer_obj, aggregation_strategy=activation_method
                    )

                    # Get classifier prediction
                    features = activation_obj.extract_features_for_classifier()
                    features_numpy = features.cpu().numpy()

                    # Get prediction probability
                    try:
                        prediction_proba = classifier.predict_proba([features_numpy])
                        if isinstance(prediction_proba, (list, tuple)) and len(prediction_proba) > 0:
                            prediction = float(prediction_proba[0])
                        else:
                            prediction = float(prediction_proba)
                    except:
                        predictions = classifier.predict([features_numpy])
                        if len(predictions) > 0:
                            prediction = float(predictions[0])
                        else:
                            logger.warning("Classifier predict returned empty array")
                            prediction = 0.5

                    # Check if code passed tests
                    code_passed = False
                    if i < len(evaluation_results.get("execution_results", [])):
                        sample_results = evaluation_results["execution_results"][i].get("results", [])
                        if sample_results:
                            code_passed = sample_results[0].get("passed", False)

                    classification_results.append(
                        {"classifier_score": prediction, "code_passed": code_passed, "code_snippet": code[:200]}
                    )

                except Exception as e:
                    logger.error(f"Error classifying generated code {i}: {e}")
                    classification_results.append({"classifier_score": 0.5, "code_passed": False, "error": str(e)})

            # Analyze classifier performance
            correct_predictions = 0
            for result in classification_results:
                # Classifier should predict high score (>0.5) for passing code
                if (result["classifier_score"] > 0.5 and result["code_passed"]) or (
                    result["classifier_score"] <= 0.5 and not result["code_passed"]
                ):
                    correct_predictions += 1

            classifier_accuracy = correct_predictions / len(classification_results) if classification_results else 0.0

            return {
                "ground_truth": "CODE_EXECUTION",
                "method_used": "bigcode-evaluation",
                "confidence": classifier_accuracy,
                "pass_rate": pass_rate,
                "classifier_accuracy": classifier_accuracy,
                "total_samples": len(generated_codes),
                "passing_samples": int(pass_rate * len(generated_codes)),
                "details": f"Pass@1: {pass_rate:.2%}, Classifier accuracy: {classifier_accuracy:.2%}",
                "task_name": task_name,
                "evaluation_method": "code-execution",
                "execution_results": evaluation_results,
            }

        except Exception as e:
            logger.error(f"Error in code execution evaluation: {e}")
            import traceback

            traceback.print_exc()
            return {
                "ground_truth": "ERROR",
                "method_used": "code-execution-error",
                "confidence": 0.0,
                "details": f"Code execution evaluation failed: {e!s}",
                "task_name": task_name,
                "evaluation_method": "code-execution",
            }
