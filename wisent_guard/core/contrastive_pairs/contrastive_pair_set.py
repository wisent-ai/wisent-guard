import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

from wisent_guard.core.activations import ActivationAggregationStrategy, Activations

from ..response import NegativeResponse, PositiveResponse
from .contrastive_pair import ContrastivePair

logger = logging.getLogger(__name__)


class ContrastivePairSet:
    def __init__(self, name, pairs=None, task_type=None):
        self.name = name
        self.pairs = pairs if pairs is not None else []
        self.task_type = task_type

    def create_from_phrase_pairs(self, phrase_pairs: List[Dict[str, str]]) -> None:
        """
        Create contrastive pairs from simple phrase pairs.

        Args:
            phrase_pairs: List of dictionaries with 'harmful' and 'harmless' keys
        """
        for i, pair_dict in enumerate(phrase_pairs):
            harmful_text = pair_dict.get("harmful", "")
            harmless_text = pair_dict.get("harmless", "")

            if not harmful_text or not harmless_text:
                continue

            # Create responses without activations (will be extracted later)
            pos_resp = PositiveResponse(text=harmless_text)
            neg_resp = NegativeResponse(text=harmful_text)

            # Create a simple prompt
            prompt = f"Respond to this: {harmless_text[:50]}..."

            pair = ContrastivePair(prompt=prompt, positive_response=pos_resp, negative_response=neg_resp)
            self.pairs.append(pair)

    def extract_activations_with_model(self, model, layer):
        """Extract activations for all responses using the model."""
        extraction_errors = []
        successful_extractions = 0

        for i, pair in enumerate(self.pairs):
            # Extract activations for positive response
            if pair.positive_response.text:
                try:
                    # Use model's activation extraction
                    activations_tensor = model.extract_activations(pair.positive_response.text, layer)
                    pair.positive_response.activations = activations_tensor
                    successful_extractions += 1
                except Exception as e:
                    error_msg = f"Pair {i} positive response: {e!s}"
                    extraction_errors.append(error_msg)
                    logger.error(f"Error extracting positive activations: {e}")

            # Extract activations for negative response
            if pair.negative_response.text:
                try:
                    # Use model's activation extraction
                    activations_tensor = model.extract_activations(pair.negative_response.text, layer)
                    pair.negative_response.activations = activations_tensor
                    successful_extractions += 1
                except Exception as e:
                    error_msg = f"Pair {i} negative response: {e!s}"
                    extraction_errors.append(error_msg)
                    logger.error(f"Error extracting negative activations: {e}")

        # Log summary
        total_expected = len(self.pairs) * 2  # positive and negative for each pair
        logger.info(f"Activation extraction completed: {successful_extractions}/{total_expected} successful")

        if extraction_errors:
            logger.warning(f"Encountered {len(extraction_errors)} extraction errors")
            if len(extraction_errors) == total_expected:
                # All extractions failed - this is likely a systematic issue
                raise RuntimeError(
                    f"All activation extractions failed. First error: {extraction_errors[0]}. "
                    f"Check that the model and layer are correctly configured."
                )

    def get_activation_pairs(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get positive and negative activations for training."""
        positive_activations = []
        negative_activations = []

        for pair in self.pairs:
            if pair.positive_response.activations is not None:
                positive_activations.append(pair.positive_response.activations)
            if pair.negative_response.activations is not None:
                negative_activations.append(pair.negative_response.activations)

        return positive_activations, negative_activations

    def prepare_classifier_data(self) -> Tuple[List[torch.Tensor], List[int]]:
        """Prepare data for classifier training (X, y format)."""
        X = []
        y = []

        for pair in self.pairs:
            if pair.positive_response.activations is not None:
                X.append(pair.positive_response.activations)
                y.append(0)  # 0 for good/harmless

            if pair.negative_response.activations is not None:
                X.append(pair.negative_response.activations)
                y.append(1)  # 1 for bad/harmful

        return X, y

    def prepare_activations_objects(self, layer) -> Tuple[List[Activations], List[int]]:
        """
        Prepare Activations objects for training using core primitives.

        Args:
            layer: Layer object for creating Activations

        Returns:
            Tuple of (activations_list, labels_list)
        """
        activations_list = []
        labels_list = []

        for pair in self.pairs:
            # Process positive response
            if pair.positive_response.activations is not None:
                pos_activations = Activations(
                    tensor=pair.positive_response.activations,
                    layer=layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
                activations_list.append(pos_activations)
                labels_list.append(0)  # 0 for positive/harmless

            # Process negative response
            if pair.negative_response.activations is not None:
                neg_activations = Activations(
                    tensor=pair.negative_response.activations,
                    layer=layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
                activations_list.append(neg_activations)
                labels_list.append(1)  # 1 for negative/harmful

        return activations_list, labels_list

    def compute_contrastive_vector(self, layer) -> Optional[torch.Tensor]:
        """
        Compute a contrastive vector from the pairs in this set.

        Args:
            layer: Layer object for creating Activations

        Returns:
            Contrastive vector tensor or None if insufficient data
        """
        positive_activations = []
        negative_activations = []

        # Collect activations using Activations objects
        for pair in self.pairs:
            if pair.positive_response.activations is not None:
                pos_activations = Activations(
                    tensor=pair.positive_response.activations,
                    layer=layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
                positive_activations.append(pos_activations.get_aggregated())

            if pair.negative_response.activations is not None:
                neg_activations = Activations(
                    tensor=pair.negative_response.activations,
                    layer=layer,
                    aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                )
                negative_activations.append(neg_activations.get_aggregated())

        if not positive_activations or not negative_activations:
            return None

        # Compute average vectors
        positive_avg = torch.stack(positive_activations).mean(dim=0)
        negative_avg = torch.stack(negative_activations).mean(dim=0)

        # Compute contrastive vector (negative - positive, since negative is harmful)
        contrastive = negative_avg - positive_avg

        # Normalize
        norm = torch.norm(contrastive, p=2)
        if norm > 0:
            contrastive = contrastive / norm

        return contrastive

    def train_classifier(self, classifier, layer, **training_kwargs) -> Dict[str, Any]:
        """
        Train a classifier using this pair set.

        Args:
            classifier: Classifier object to train
            layer: Layer object for creating Activations
            **training_kwargs: Additional training parameters

        Returns:
            Training results dictionary
        """
        # Prepare data using Activations objects
        activations_list, labels = self.prepare_activations_objects(layer)

        # Extract features for classifier
        X = [act.extract_features_for_classifier() for act in activations_list]

        # Train the classifier
        results = classifier.fit(X, labels, **training_kwargs)

        return results

    def evaluate_with_vectors(
        self, vector_dict: Dict[str, torch.Tensor], layer, threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Evaluate this pair set against contrastive vectors.

        Args:
            vector_dict: Dictionary mapping category names to contrastive vectors
            layer: Layer object for creating Activations
            threshold: Threshold for harmful classification

        Returns:
            Evaluation results
        """
        results = {
            "total_pairs": len(self.pairs),
            "category_results": {},
            "overall_accuracy": 0.0,
        }

        correct_predictions = 0
        total_predictions = 0

        for category, vector in vector_dict.items():
            category_correct = 0
            category_total = 0

            for pair in self.pairs:
                # Evaluate positive response (should be low similarity)
                if pair.positive_response.activations is not None:
                    pos_activations = Activations(
                        tensor=pair.positive_response.activations,
                        layer=layer,
                        aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                    )
                    pos_similarity = pos_activations.calculate_similarity(vector)
                    pos_predicted_harmful = pos_similarity >= threshold

                    if not pos_predicted_harmful:  # Correct if not predicted as harmful
                        category_correct += 1
                        correct_predictions += 1
                    category_total += 1
                    total_predictions += 1

                # Evaluate negative response (should be high similarity)
                if pair.negative_response.activations is not None:
                    neg_activations = Activations(
                        tensor=pair.negative_response.activations,
                        layer=layer,
                        aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
                    )
                    neg_similarity = neg_activations.calculate_similarity(vector)
                    neg_predicted_harmful = neg_similarity >= threshold

                    if neg_predicted_harmful:  # Correct if predicted as harmful
                        category_correct += 1
                        correct_predictions += 1
                    category_total += 1
                    total_predictions += 1

            category_accuracy = category_correct / category_total if category_total > 0 else 0.0
            results["category_results"][category] = {
                "accuracy": category_accuracy,
                "correct": category_correct,
                "total": category_total,
            }

        results["overall_accuracy"] = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this pair set."""
        stats = {
            "name": self.name,
            "total_pairs": len(self.pairs),
            "pairs_with_positive_activations": 0,
            "pairs_with_negative_activations": 0,
            "pairs_with_both_activations": 0,
            "task_type": self.task_type,
        }

        for pair in self.pairs:
            has_pos = pair.positive_response.activations is not None
            has_neg = pair.negative_response.activations is not None

            if has_pos:
                stats["pairs_with_positive_activations"] += 1
            if has_neg:
                stats["pairs_with_negative_activations"] += 1
            if has_pos and has_neg:
                stats["pairs_with_both_activations"] += 1

        return stats

    @classmethod
    def from_phrase_pairs(
        cls, name: str, phrase_pairs: List[Dict[str, str]], task_type: str = None
    ) -> "ContrastivePairSet":
        """
        Create a ContrastivePairSet from simple phrase pairs.

        Args:
            name: Name for the pair set
            phrase_pairs: List of dictionaries with 'harmful' and 'harmless' keys
            task_type: Optional task type

        Returns:
            ContrastivePairSet instance
        """
        pair_set = cls(name=name, task_type=task_type)
        pair_set.create_from_phrase_pairs(phrase_pairs)
        return pair_set

    def __len__(self) -> int:
        """Return the number of pairs in this set."""
        return len(self.pairs)

    def __repr__(self) -> str:
        """String representation of the ContrastivePairSet."""
        return f"ContrastivePairSet(name='{self.name}', pairs={len(self.pairs)}, task_type='{self.task_type}')"

    def get_task_name(self, task) -> str:
        """Extract task name from task object."""
        if hasattr(task, "NAME"):
            return task.NAME
        if hasattr(task, "_name"):
            return task._name
        if hasattr(task, "task_name"):
            return task.task_name
        return str(type(task).__name__).lower()

    def evaluate_response_with_task(self, task, doc: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate response using task-specific logic.

        Args:
            task: Task object with evaluation methods
            doc: Document/question data
            response: Generated response

        Returns:
            Evaluation result
        """
        try:
            # Try to use task's process_results method
            if hasattr(task, "process_results"):
                # Create a result structure
                result_structure = {"doc": doc, "target": response, "resps": [[response]]}
                result = task.process_results(doc, [result_structure])
                return {
                    "correct": result.get("acc", False),
                    "score": result.get("acc", 0.0),
                    "method": "task_process_results",
                }

            # Fallback to simple string matching
            if "target" in doc:
                target = str(doc["target"]).lower().strip()
                response_clean = response.lower().strip()
                correct = target in response_clean
                return {
                    "correct": correct,
                    "score": 1.0 if correct else 0.0,
                    "method": "string_matching",
                }

            # Default: assume correct
            return {"correct": True, "score": 1.0, "method": "default"}

        except Exception as e:
            return {"correct": False, "score": 0.0, "method": "error", "error": str(e)}


    @classmethod
    def from_csv_file(
        cls,
        name: str,
        csv_path: Union[str, Path],
        question_col: str = "question",
        correct_col: str = "correct_answer",
        incorrect_col: str = "incorrect_answer",
        limit: Optional[int] = None,
    ) -> "ContrastivePairSet":
        """
        Create a ContrastivePairSet from a CSV file.

        Args:
            name: Name for the pair set
            csv_path: Path to the CSV file
            question_col: Column name for questions
            correct_col: Column name for correct answers
            incorrect_col: Column name for incorrect answers
            limit: Optional limit on number of rows to process

        Returns:
            ContrastivePairSet instance
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if limit is not None and limit > 0:
            df = df.head(limit)

        # Validate required columns
        required_cols = [question_col, correct_col, incorrect_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")

        pair_set = cls(name=f"{name}_csv", task_type="csv_file")

        for idx, row in df.iterrows():
            try:
                question = str(row[question_col])
                correct_answer = str(row[correct_col])
                incorrect_answer = str(row[incorrect_col])

                # Skip rows with missing data
                if pd.isna(row[question_col]) or pd.isna(row[correct_col]) or pd.isna(row[incorrect_col]):
                    continue

                # Create contrastive pair
                pair_set._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)

            except Exception as e:
                print(f"Warning: Skipping row {idx} due to error: {e}")
                continue

        return pair_set

    @classmethod
    def from_json_file(
        cls, name: str, json_path: Union[str, Path], limit: Optional[int] = None
    ) -> "ContrastivePairSet":
        """
        Create a ContrastivePairSet from a JSON file.

        Expected JSON format:
        [
            {
                "question": "Question text",
                "correct_answer": "Correct answer",
                "incorrect_answer": "Incorrect answer"
            },
            ...
        ]

        Or format with multiple correct/incorrect answers:
        [
            {
                "question": "Question text",
                "correct_answers": ["Answer 1", "Answer 2"],
                "incorrect_answers": ["Wrong 1", "Wrong 2"]
            },
            ...
        ]

        Args:
            name: Name for the pair set
            json_path: Path to the JSON file
            limit: Optional limit on number of items to process

        Returns:
            ContrastivePairSet instance
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of objects")

        if limit is not None and limit > 0:
            data = data[:limit]

        pair_set = cls(name=f"{name}_json", task_type="json_file")

        for idx, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    print(f"Warning: Skipping item {idx}, not a dictionary")
                    continue

                question = item.get("question", "")

                # Handle single answer format
                if "correct_answer" in item and "incorrect_answer" in item:
                    correct_answer = str(item["correct_answer"])
                    incorrect_answer = str(item["incorrect_answer"])
                    pair_set._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)

                # Handle multiple answers format
                elif "correct_answers" in item and "incorrect_answers" in item:
                    correct_answers = item["correct_answers"]
                    incorrect_answers = item["incorrect_answers"]

                    if isinstance(correct_answers, list) and isinstance(incorrect_answers, list):
                        # Use first correct and first incorrect
                        if correct_answers and incorrect_answers:
                            correct_answer = str(correct_answers[0])
                            incorrect_answer = str(incorrect_answers[0])
                            pair_set._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)

            except Exception as e:
                print(f"Warning: Skipping item {idx} due to error: {e}")
                continue

        return pair_set

    def _create_qa_contrastive_pair(self, question: str, correct_answer: str, incorrect_answer: str):
        """Helper method to create a QA contrastive pair."""
        # Format the prompt
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nWhich is better: {question}\nA. {incorrect_answer}\nB. {correct_answer}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        # Create positive response (B - correct answer)
        pos_resp = PositiveResponse(text="B")

        # Create negative response (A - incorrect answer)
        neg_resp = NegativeResponse(text="A")

        pair = ContrastivePair(prompt=prompt, positive_response=pos_resp, negative_response=neg_resp)

        # Store metadata
        pair.question = question
        pair.correct_answer = correct_answer
        pair.incorrect_answer = incorrect_answer

        self.pairs.append(pair)

    def save_to_json(self, json_path: Union[str, Path], include_metadata: bool = True):
        """
        Save the contrastive pairs to a JSON file.

        Args:
            json_path: Path to save the JSON file
            include_metadata: Whether to include metadata
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for i, pair in enumerate(self.pairs):
            item = {
                "pair_id": i,
                "prompt": pair.prompt,
                "positive_response": pair.positive_response.text,
                "negative_response": pair.negative_response.text,
            }

            if include_metadata and hasattr(pair, "question"):
                item.update(
                    {
                        "question": getattr(pair, "question", ""),
                        "correct_answer": getattr(pair, "correct_answer", ""),
                        "incorrect_answer": getattr(pair, "incorrect_answer", ""),
                    }
                )

            data.append(item)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(data)} contrastive pairs to {json_path}")

    @classmethod
    def extract_qa_pairs_from_task_docs(
        cls, task_name: str, task_data, docs: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract QA pairs from task documents using task-specific logic.

        Args:
            task_name: Name of the task (e.g., 'truthful_qa', 'hellaswag')
            task_data: Task object from lm-eval
            docs: List of documents from the task

        Returns:
            List of QA pairs with 'question', 'correct_answer', 'incorrect_answer' keys
        """
        qa_pairs = []

        for i, doc in enumerate(docs):
            try:
                # FIXED: For TruthfulQA, extract just the actual question, not the template
                if "truthfulqa" in task_name.lower() or "truthful_qa" in task_name.lower():
                    # Use actual question, not the template with examples
                    question = doc.get("question", "")
                elif task_name == "winogrande":
                    # For winogrande, use the sentence field directly
                    question = doc.get("sentence", "")
                elif task_name == "wsc273":
                    # For wsc273, use the text field which contains the actual sentence
                    question = doc.get("text", "")
                elif task_name in [
                    "livecodebench",
                    "humaneval",
                    "mbpp",
                    "mbpp_plus",
                    "apps",
                    "conala",
                    "concode",
                ]:
                    # For code generation tasks, use the problem description/content
                    if task_name == "conala":
                        # Conala uses 'intent' field for the natural language description
                        question = doc.get("intent", doc.get("question_content", doc.get("text", "")))
                    elif task_name == "concode":
                        # Concode uses 'nl' field for natural language
                        question = doc.get("nl", doc.get("question_content", doc.get("text", "")))
                    else:
                        question = doc.get("question_content", doc.get("text", doc.get("prompt", "")))
                    if not question and "question_title" in doc:
                        question = doc.get("question_title", "")
                elif task_name.startswith("supergpqa"):
                    # SuperGPQA tasks use 'question' field directly
                    question = doc.get("question", "")
                elif task_name.startswith("hle"):
                    # HLE tasks use 'question' field directly
                    question = doc.get("question", "")
                elif task_name.startswith("aime"):
                    # AIME tasks use 'Problem' or 'problem' field
                    question = doc.get("Problem", doc.get("problem", ""))
                elif task_name.startswith("math500") or task_name in [
                    "math",
                    "hendrycks_math",
                ]:
                    # MATH tasks use 'problem' field
                    question = doc.get("problem", "")
                else:
                    # For other tasks, use the template method
                    try:
                        if hasattr(task_data, "doc_to_text"):
                            question = task_data.doc_to_text(doc)
                        else:
                            question = doc.get("question", str(doc))
                    except Exception:
                        question = str(doc)

                # Skip if we don't have a proper question
                # Ensure question is a string before calling strip()
                if not question or not isinstance(question, str) or len(question.strip()) < 10:
                    continue

                # Task-specific answer extraction
                correct_answer = None
                incorrect_answer = None
                formatted_question = question  # Default to basic question

                if "truthfulqa" in task_name.lower() or "truthful_qa" in task_name.lower():
                    # TruthfulQA-specific extraction - handle both generation and multiple choice formats
                    if "mc1_targets" in doc or "mc2_targets" in doc:
                        # Multiple choice format (truthfulqa_mc1, truthfulqa_mc2)
                        mc_targets = doc.get("mc1_targets", doc.get("mc2_targets", {}))
                        choices = mc_targets.get("choices", [])
                        labels = mc_targets.get("labels", [])

                        # Find the correct answer
                        for j, label in enumerate(labels):
                            if label == 1 and j < len(choices):
                                correct_answer = choices[j]
                                break

                        # Find an incorrect answer
                        for j, label in enumerate(labels):
                            if label == 0 and j < len(choices):
                                incorrect_answer = choices[j]
                                break
                    elif "correct_answers" in doc and "incorrect_answers" in doc:
                        # Generation format (truthfulqa_gen)
                        correct_answers_list = doc.get("correct_answers", [])
                        incorrect_answers_list = doc.get("incorrect_answers", [])

                        if correct_answers_list and incorrect_answers_list:
                            # Use the first correct and first incorrect answer
                            correct_answer = correct_answers_list[0]
                            incorrect_answer = incorrect_answers_list[0]
                            # For TruthfulQA, use the question as formatted_question
                            formatted_question = question
                    else:
                        # Fallback for unknown TruthfulQA format
                        continue

                elif task_name == "winogrande":
                    # Use benchmark extractor for winogrande to get formatted_question
                    from ..benchmark_extractors import extract_contrastive_pair

                    contrastive_data = extract_contrastive_pair(task_name, doc, task_data)

                    if contrastive_data:
                        question = contrastive_data["question"]
                        correct_answer = contrastive_data.get("correct_answer", contrastive_data.get("correct_choice"))
                        incorrect_answer = contrastive_data.get(
                            "incorrect_answer", contrastive_data.get("incorrect_choice")
                        )

                        # Get the formatted question from the extractor
                        from ..benchmark_extractors import get_extractor

                        extractor = get_extractor(task_name)
                        extractor_result = extractor.extract_qa_pair(doc, task_data)
                        formatted_question = (
                            extractor_result.get("formatted_question", question) if extractor_result else question
                        )
                    else:
                        # Fallback to manual extraction if extractor fails
                        option1 = doc.get("option1", "")
                        option2 = doc.get("option2", "")
                        answer = doc.get("answer", "")

                        if option1 and option2 and answer:
                            if answer == "1":
                                correct_answer = option1
                                incorrect_answer = option2
                            elif answer == "2":
                                correct_answer = option2
                                incorrect_answer = option1
                            else:
                                continue
                            formatted_question = question  # Use basic question as fallback
                        else:
                            continue

                elif task_name == "wsc273":
                    # WSC273-specific extraction
                    # wsc273 has: text, pronoun, pronoun_loc, quote, quote_loc, options, label
                    options = doc.get("options", [])
                    label = doc.get("label", 0)

                    if len(options) >= 2:
                        # Convert label to int if needed
                        if isinstance(label, str):
                            try:
                                label = int(label)
                            except ValueError:
                                label = 0

                        # Label is the index of the correct option
                        if 0 <= label < len(options):
                            correct_answer = options[label]
                            # Use the other option as incorrect
                            incorrect_answer = (
                                options[1 - label] if len(options) == 2 else options[0 if label != 0 else 1]
                            )
                        else:
                            # Fallback
                            correct_answer = options[0]
                            incorrect_answer = options[1]

                        formatted_question = question
                    else:
                        # Skip if we don't have proper options
                        continue

                elif task_name == "hellaswag":
                    # HellaSwag-specific extraction
                    endings = doc.get("endings", [])
                    label = doc.get("label", 0)

                    # Convert string label to int if needed
                    if isinstance(label, str):
                        try:
                            label = int(label)
                        except ValueError:
                            label = 0

                    if len(endings) > label:
                        correct_answer = endings[label]
                        # Use first different ending as incorrect answer
                        for i, ending in enumerate(endings):
                            if i != label:
                                incorrect_answer = ending
                                break
                        # For HellaSwag, use the question as formatted_question
                        formatted_question = question

                elif task_name == "livecodebench":
                    # LiveCodeBench will be handled separately after the loop
                    # using the model outputs extractor
                    continue

                elif task_name.startswith("supergpqa"):
                    # SuperGPQA-specific extraction
                    options = doc.get("options", [])
                    answer_text = doc.get("answer", "")
                    answer_letter = doc.get("answer_letter", "")

                    if options and answer_text:
                        correct_answer = answer_text
                        # Find an incorrect answer from options
                        for option in options:
                            if option != answer_text:
                                incorrect_answer = option
                                break

                        # Format the question with options
                        if options:
                            formatted_options = []
                            for i, option in enumerate(options):
                                letter = chr(ord("A") + i)
                                formatted_options.append(f"{letter}. {option}")
                            formatted_question = f"{question}\n\n" + "\n".join(formatted_options)
                        else:
                            formatted_question = question

                elif task_name.startswith("hle"):
                    # HLE-specific extraction
                    answer = doc.get("answer", "")
                    answer_type = doc.get("answer_type", "")

                    if answer_type == "multipleChoice":
                        # For multiple choice, try to extract the text from the question
                        import re

                        patterns = [
                            rf"{answer}\.\s+(.+?)(?=\n[A-E]\.|$)",  # "A. option" format
                            rf"{answer}\)\s+(.+?)(?=\n[A-E]\)|$)",  # "A) option" format
                        ]

                        correct_text = None
                        for pattern in patterns:
                            match = re.search(pattern, question, re.MULTILINE | re.DOTALL)
                            if match:
                                correct_text = match.group(1).strip()
                                break

                        if correct_text:
                            correct_answer = correct_text
                            # Find an incorrect choice
                            other_letters = [letter for letter in ["A", "B", "C", "D", "E"] if letter != answer]
                            for letter in other_letters:
                                for pattern in patterns:
                                    pattern_with_letter = pattern.replace(answer, letter)
                                    match = re.search(
                                        pattern_with_letter,
                                        question,
                                        re.MULTILINE | re.DOTALL,
                                    )
                                    if match:
                                        incorrect_answer = match.group(1).strip()
                                        break
                                if incorrect_answer:
                                    break
                        else:
                            correct_answer = answer  # Fallback to letter
                            incorrect_answer = "Incorrect answer"
                    else:
                        # Exact match
                        correct_answer = answer
                        incorrect_answer = "Wrong answer"

                    formatted_question = question  # Question already contains choices if MC

                elif (
                    task_name.startswith("aime")
                    or task_name.startswith("math500")
                    or task_name in ["math", "hendrycks_math"]
                ):
                    # Math task extraction
                    answer = doc.get("Answer", doc.get("answer", doc.get("solution", "")))
                    if answer:
                        correct_answer = str(answer)
                        # Generate a simple incorrect answer for math problems
                        if str(answer).isdigit():
                            incorrect_answer = str(int(answer) + 1)
                        else:
                            incorrect_answer = "Wrong answer"
                    formatted_question = question

                else:
                    # Use benchmark-specific extractors for all other tasks
                    from ..benchmark_extractors import extract_contrastive_pair

                    contrastive_data = extract_contrastive_pair(task_name, doc, task_data)

                    if contrastive_data:
                        question = contrastive_data["question"]
                        correct_answer = contrastive_data.get("correct_answer", contrastive_data.get("correct_choice"))
                        incorrect_answer = contrastive_data.get(
                            "incorrect_answer", contrastive_data.get("incorrect_choice")
                        )

                        # Try to get formatted_question from the extractor
                        try:
                            from ..benchmark_extractors import get_extractor

                            extractor = get_extractor(task_name)
                            extractor_result = extractor.extract_qa_pair(doc, task_data)
                            formatted_question = (
                                extractor_result.get("formatted_question", question) if extractor_result else question
                            )
                        except:
                            formatted_question = question  # Fallback to basic question
                    else:
                        # Fallback to generic extraction if extractor fails

                        # For BoolQ tasks specifically
                        if "boolq" in task_name.lower() and "answer" in doc:
                            answer = doc.get("answer", False)
                            correct_answer = "True" if answer else "False"
                            incorrect_answer = "False" if answer else "True"
                            formatted_question = question  # Use basic question as fallback

                        # For COPA tasks specifically
                        elif "copa" in task_name.lower() and "choice1" in doc and "choice2" in doc:
                            choice1 = doc.get("choice1", "")
                            choice2 = doc.get("choice2", "")
                            label = doc.get("label", 0)
                            correct_answer = choice1 if label == 0 else choice2
                            incorrect_answer = choice2 if label == 0 else choice1
                            formatted_question = question  # Use basic question as fallback

                        # For ARC tasks specifically, handle the case where extractor only returns QA pair
                        elif "arc" in task_name.lower() and "choices" in doc and "answerKey" in doc:
                            choices = doc.get("choices", {})
                            choice_texts = choices.get("text", [])
                            choice_labels = choices.get("label", [])
                            answer_key = doc.get("answerKey", "")

                            # Find correct answer
                            for idx, label in enumerate(choice_labels):
                                if label == answer_key and idx < len(choice_texts):
                                    correct_answer = choice_texts[idx]
                                    break

                            # Find an incorrect answer
                            for idx, label in enumerate(choice_labels):
                                if label != answer_key and idx < len(choice_texts):
                                    incorrect_answer = choice_texts[idx]
                                    break

                        # Try common answer fields
                        elif "choices" in doc and "label" in doc:
                            choices = doc["choices"]
                            label = doc["label"]

                            # Convert string label to int if needed
                            if isinstance(label, str):
                                try:
                                    label = int(label)
                                except ValueError:
                                    label = 0

                            # Handle different choices formats
                            if isinstance(choices, dict):
                                # Choices is a dict with 'text' and 'label' keys (e.g., ARC)
                                choice_texts = choices.get("text", [])
                                if isinstance(label, int) and 0 <= label < len(choice_texts):
                                    correct_answer = choice_texts[label]
                                    # Use first different choice as incorrect
                                    for j, choice in enumerate(choice_texts):
                                        if j != label:
                                            incorrect_answer = choice
                                            break
                            elif isinstance(choices, list):
                                # Choices is a list (e.g., HellaSwag)
                                if isinstance(label, int) and 0 <= label < len(choices):
                                    correct_answer = choices[label]
                                    # Use first different choice as incorrect
                                    for j, choice in enumerate(choices):
                                        if j != label:
                                            incorrect_answer = choice
                                            break

                        elif "answer" in doc:
                            correct_answer = doc["answer"]
                            # For simple answer tasks, create a generic incorrect answer
                            incorrect_answer = "This is incorrect"
                        elif hasattr(task_data, "doc_to_target"):
                            correct_answer = task_data.doc_to_target(doc)
                            incorrect_answer = "This is incorrect"
                        else:
                            # If we have a correct_answer from earlier extraction, generate incorrect
                            if correct_answer and not incorrect_answer:
                                # Generate a simple incorrect answer
                                incorrect_answer = "Wrong answer"
                            elif not correct_answer:
                                # Skip if no extraction method works
                                continue

                if correct_answer and incorrect_answer:
                    qa_pairs.append(
                        {
                            "question": question,
                            "formatted_question": formatted_question,
                            "correct_answer": correct_answer,
                            "incorrect_answer": incorrect_answer,
                        }
                    )

            except Exception as e:
                # Skip this document and continue with others
                logger.warning(f"⚠️  Skipping document {i} in {task_name} due to extraction error: {e}")
                continue

        # Special handling for LiveCodeBench using model outputs
        if task_name == "livecodebench" and len(qa_pairs) == 0:
            try:
                from ..benchmark_extractor_impls.livecodebench_model_outputs_extractor import (
                    LiveCodeBenchModelOutputsExtractor,
                )

                extractor = LiveCodeBenchModelOutputsExtractor()

                # Extract contrastive pairs from model outputs
                contrastive_pairs = extractor.extract_contrastive_pairs(docs, limit=len(docs))

                # Convert to QA pair format
                for pair in contrastive_pairs:
                    qa_pairs.append(
                        {
                            "question": pair["question"],
                            "formatted_question": pair["question"],
                            "correct_answer": pair["correct_answer"],
                            "incorrect_answer": pair["incorrect_answer"],
                            "metadata": pair.get("metadata", {}),
                        }
                    )

                logger.info(f"Extracted {len(qa_pairs)} LiveCodeBench pairs from model outputs")

            except Exception as e:
                logger.error(f"Failed to extract LiveCodeBench pairs from model outputs: {e}")

        return qa_pairs
