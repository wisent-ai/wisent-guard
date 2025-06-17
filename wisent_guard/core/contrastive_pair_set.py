from .contrastive_pair import ContrastivePair
from .response import PositiveResponse, NegativeResponse
from .activations import Activations, ActivationAggregationMethod
import torch
import random
import json
import csv
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

class ContrastivePairSet:
    def __init__(self, name, pairs=None, task_type=None):
        self.name = name
        self.pairs = pairs if pairs is not None else []
        self.task_type = task_type

    def generate_negative_responses(self, model, layer, task=None):
        """Generate negative responses for each pair using the model."""
        for pair in self.pairs:
            # Generate negative response with different temperature/sampling
            negative_text, negative_activations = model.generate(
                pair.prompt, layer.index, temperature=1.2, do_sample=True
            )
            pair.negative_response = NegativeResponse(
                text=negative_text, 
                activations=negative_activations
            )

    def label_responses_by_task(self, task):
        """Label responses as positive/negative based on task-specific criteria."""
        for pair in self.pairs:
            # Evaluate if the positive response is actually correct
            if hasattr(pair, 'document'):  # Store document for evaluation
                evaluation = self.evaluate_response_with_task(task, pair.document, pair.positive_response.text)
                
                if not evaluation["correct"]:
                    # Swap positive and negative if the "positive" is actually bad
                    pair.positive_response, pair.negative_response = pair.negative_response, pair.positive_response

    def create_truthfulqa_pairs(self, task, docs, prompts, references):
        """Create contrastive pairs specifically for TruthfulQA."""
        for doc, prompt, ref in zip(docs, prompts, references):
            # Create positive response (truthful)
            pos_resp = PositiveResponse(text=ref)
            
            # Create negative response (plausible but false)
            false_responses = [
                "Yes, absolutely.",
                "No, that's completely false.", 
                "This is a well-known fact.",
                "Scientific studies have definitively proven this.",
                "It's impossible and has never happened."
            ]
            neg_text = random.choice(false_responses)
            neg_resp = NegativeResponse(text=neg_text)
            
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            pair.document = doc  # Store for later evaluation
            self.pairs.append(pair)

    def create_multiple_choice_pairs(self, task, docs, prompts, references):
        """Create contrastive pairs for multiple choice tasks."""
        for doc, prompt, ref in zip(docs, prompts, references):
            # Create positive response (correct answer)
            pos_resp = PositiveResponse(text=ref)
            
            # Create negative response (wrong choice)
            if hasattr(task, 'doc_to_choice'):
                choices = task.doc_to_choice(doc)
                correct_idx = doc.get('answer', doc.get('label', 0))
                
                if isinstance(choices, list) and len(choices) > 1:
                    wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
                    if wrong_indices:
                        bad_idx = random.choice(wrong_indices)
                        neg_text = choices[bad_idx]
                    else:
                        neg_text = "Wrong answer"
                else:
                    neg_text = "Wrong answer"
            else:
                neg_text = "Wrong answer"
                
            neg_resp = NegativeResponse(text=neg_text)
            
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            pair.document = doc  # Store for later evaluation
            self.pairs.append(pair)

    def create_from_phrase_pairs(self, phrase_pairs: List[Dict[str, str]]) -> None:
        """
        Create contrastive pairs from simple phrase pairs.
        
        Args:
            phrase_pairs: List of dictionaries with 'harmful' and 'harmless' keys
        """
        for i, pair_dict in enumerate(phrase_pairs):
            harmful_text = pair_dict.get('harmful', '')
            harmless_text = pair_dict.get('harmless', '')
            
            if not harmful_text or not harmless_text:
                continue
            
            # Create responses without activations (will be extracted later)
            pos_resp = PositiveResponse(text=harmless_text)
            neg_resp = NegativeResponse(text=harmful_text)
            
            # Create a simple prompt
            prompt = f"Respond to this: {harmless_text[:50]}..."
            
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            self.pairs.append(pair)

    def create_multiple_choice_from_phrases(self, phrase_pairs: List[Dict[str, str]], model) -> None:
        """
        Convert phrase pairs to multiple-choice format using the model's formatting.
        
        Args:
            phrase_pairs: List of dictionaries with 'harmful' and 'harmless' keys
            model: Model object with formatting capabilities
        """
        for pair_dict in phrase_pairs:
            harmful_text = pair_dict.get('harmful', '')
            harmless_text = pair_dict.get('harmless', '')
            
            if not harmful_text or not harmless_text:
                continue
            
            # Use model's multiple choice conversion
            harmful_mc, harmless_mc = model.convert_to_multiple_choice(harmful_text, harmless_text)
            
            # Create responses
            pos_resp = PositiveResponse(text=harmless_mc)  # Harmless choice is positive
            neg_resp = NegativeResponse(text=harmful_mc)   # Harmful choice is negative
            
            # Create prompt
            question = "Which of the following statements is better?"
            prompt = f"{question}\nA. {harmless_text}\nB. {harmful_text}"
            
            pair = ContrastivePair(
                prompt=prompt,
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            self.pairs.append(pair)

    def create_multiple_choice_questions(self, questions: List[Dict[str, Any]], model) -> None:
        """
        Create contrastive pairs from multiple-choice questions.
        
        Args:
            questions: List of dictionaries with question data
            model: Model object with formatting capabilities
        """
        for q in questions:
            # Create A (correct) and B (incorrect) response phrases
            a_phrase = model.format_multiple_choice(
                q["question"], 
                q["choice_a"], 
                q["choice_b"], 
                "A"
            )
            
            b_phrase = model.format_multiple_choice(
                q["question"], 
                q["choice_a"], 
                q["choice_b"], 
                "B"
            )
            
            # Create ContrastivePair
            pos_resp = PositiveResponse(text=a_phrase)  # A is correct/harmless
            neg_resp = NegativeResponse(text=b_phrase)  # B is incorrect/harmful
            
            pair = ContrastivePair(
                prompt=q["question"],
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            self.pairs.append(pair)

    def extract_activations_with_model(self, model, layer):
        """Extract activations for all responses using the model."""
        for pair in self.pairs:
            # Extract activations for positive response
            if pair.positive_response.text:
                try:
                    # Use model's activation extraction
                    activations_tensor = model.extract_activations(pair.positive_response.text, layer)
                    pair.positive_response.activations = activations_tensor
                except Exception as e:
                    print(f"Error extracting positive activations: {e}")
            
            # Extract activations for negative response
            if pair.negative_response.text:
                try:
                    # Use model's activation extraction
                    activations_tensor = model.extract_activations(pair.negative_response.text, layer)
                    pair.negative_response.activations = activations_tensor
                except Exception as e:
                    print(f"Error extracting negative activations: {e}")

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
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                )
                activations_list.append(pos_activations)
                labels_list.append(0)  # 0 for positive/harmless
            
            # Process negative response
            if pair.negative_response.activations is not None:
                neg_activations = Activations(
                    tensor=pair.negative_response.activations,
                    layer=layer,
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
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
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
                )
                positive_activations.append(pos_activations.get_aggregated())
            
            if pair.negative_response.activations is not None:
                neg_activations = Activations(
                    tensor=pair.negative_response.activations,
                    layer=layer,
                    aggregation_method=ActivationAggregationMethod.LAST_TOKEN
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

    def evaluate_with_vectors(self, vector_dict: Dict[str, torch.Tensor], layer, threshold: float = 0.7) -> Dict[str, Any]:
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
            "overall_accuracy": 0.0
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
                        aggregation_method=ActivationAggregationMethod.LAST_TOKEN
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
                        aggregation_method=ActivationAggregationMethod.LAST_TOKEN
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
                "total": category_total
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
            "task_type": self.task_type
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
    def from_task_data(cls, name, task, docs, prompts, references, task_type=None):
        """Create a ContrastivePairSet from task data."""
        pair_set = cls(name=name, task_type=task_type)
        
        # Determine how to create pairs based on task name
        if 'truthfulqa' in name.lower():
            pair_set.create_truthfulqa_pairs(task, docs, prompts, references)
        elif any(mc_task in name.lower() for mc_task in ['hellaswag', 'mmlu', 'arc', 'winogrande', 'piqa']):
            pair_set.create_multiple_choice_pairs(task, docs, prompts, references)
        else:
            # Default: create simple pairs with reference as positive
            for doc, prompt, ref in zip(docs, prompts, references):
                pos_resp = PositiveResponse(text=ref)
                neg_resp = NegativeResponse(text="")  # Placeholder
                pair = ContrastivePair(
                    prompt=prompt,
                    positive_response=pos_resp,
                    negative_response=neg_resp
                )
                pair.document = doc
                pair_set.pairs.append(pair)
        
        return pair_set

    @classmethod
    def from_phrase_pairs(cls, name: str, phrase_pairs: List[Dict[str, str]], task_type: str = None) -> 'ContrastivePairSet':
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
        if hasattr(task, 'NAME'):
            return task.NAME
        elif hasattr(task, '_name'):
            return task._name
        elif hasattr(task, 'task_name'):
            return task.task_name
        else:
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
            if hasattr(task, 'process_results'):
                # Create a mock result structure
                mock_result = {
                    'doc': doc,
                    'target': response,
                    'resps': [[response]]
                }
                result = task.process_results(doc, [mock_result])
                return {
                    "correct": result.get("acc", False),
                    "score": result.get("acc", 0.0),
                    "method": "task_process_results"
                }
            
            # Fallback to simple string matching
            if 'target' in doc:
                target = str(doc['target']).lower().strip()
                response_clean = response.lower().strip()
                correct = target in response_clean
                return {
                    "correct": correct,
                    "score": 1.0 if correct else 0.0,
                    "method": "string_matching"
                }
            
            # Default: assume correct
            return {
                "correct": True,
                "score": 1.0,
                "method": "default"
            }
            
        except Exception as e:
            return {
                "correct": False,
                "score": 0.0,
                "method": "error",
                "error": str(e)
            }
    
    def create_contrastive_pairs_from_task(
        self,
        task,
        docs: List[Dict[str, Any]],
        responses: List[str],
        model_name: str,
        layer: int
    ) -> None:
        """
        Create contrastive pairs from task evaluation results.
        
        Args:
            task: Task object
            docs: List of documents
            responses: List of generated responses
            model_name: Model name
            layer: Layer index
        """
        for doc, response in zip(docs, responses):
            evaluation = self.evaluate_response_with_task(task, doc, response)
            
            if evaluation["correct"]:
                # Correct response is harmless
                pos_resp = PositiveResponse(text=f"Question: {doc.get('question', '')} Correct answer: {response}")
                neg_resp = NegativeResponse(text=f"Question: {doc.get('question', '')} Wrong answer: [INCORRECT]")
            else:
                # Incorrect response is harmful
                pos_resp = PositiveResponse(text=f"Question: {doc.get('question', '')} Correct answer: {doc.get('target', '[CORRECT]')}")
                neg_resp = NegativeResponse(text=f"Question: {doc.get('question', '')} Wrong answer: {response}")
            
            pair = ContrastivePair(
                prompt=f"Question: {doc.get('question', '')}",
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            self.pairs.append(pair)
    
    def label_responses_for_training(
        self,
        responses: List[str],
        references: List[str],
        task_name: str = "generic"
    ) -> List[Dict[str, Any]]:
        """
        Label responses for training by comparing with references.
        
        Args:
            responses: Generated responses
            references: Reference/correct responses
            task_name: Name of the task
            
        Returns:
            List of labeled examples
        """
        labeled_data = []
        
        for response, reference in zip(responses, references):
            # Simple similarity-based labeling
            response_clean = response.lower().strip()
            reference_clean = reference.lower().strip()
            
            # Check if response contains reference or vice versa
            is_correct = (
                reference_clean in response_clean or 
                response_clean in reference_clean or
                response_clean == reference_clean
            )
            
            labeled_data.append({
                "response": response,
                "reference": reference,
                "is_correct": is_correct,
                "is_harmful": not is_correct,  # Incorrect responses are considered harmful
                "task": task_name
            })
        
        return labeled_data
    
    def create_training_pairs_from_labels(
        self,
        labeled_data: List[Dict[str, Any]],
        balance_classes: bool = True
    ) -> List[Dict[str, str]]:
        """
        Create training pairs from labeled data.
        
        Args:
            labeled_data: List of labeled examples
            balance_classes: Whether to balance harmful/harmless classes
            
        Returns:
            List of phrase pairs for training
        """
        import random
        
        harmful_examples = [item for item in labeled_data if item["is_harmful"]]
        harmless_examples = [item for item in labeled_data if not item["is_harmful"]]
        
        if balance_classes:
            # Balance the classes
            min_count = min(len(harmful_examples), len(harmless_examples))
            harmful_examples = random.sample(harmful_examples, min_count)
            harmless_examples = random.sample(harmless_examples, min_count)
        
        phrase_pairs = []
        
        for harmful, harmless in zip(harmful_examples, harmless_examples):
            phrase_pairs.append({
                "harmful": harmful["response"],
                "harmless": harmless["response"]
            })
        
        return phrase_pairs
    
    def create_pairs_from_labeled_responses(
        self,
        labeled_data: List[Dict[str, Any]]
    ) -> None:
        """
        Create contrastive pairs from labeled response data.
        
        Args:
            labeled_data: List of labeled examples with 'response', 'is_correct', etc.
        """
        for item in labeled_data:
            if item["is_correct"]:
                pos_resp = PositiveResponse(text=item["response"])
                neg_resp = NegativeResponse(text="[INCORRECT RESPONSE]")
            else:
                pos_resp = PositiveResponse(text=item.get("reference", "[CORRECT RESPONSE]"))
                neg_resp = NegativeResponse(text=item["response"])
            
            pair = ContrastivePair(
                prompt=f"Task: {item.get('task', 'unknown')}",
                positive_response=pos_resp,
                negative_response=neg_resp
            )
            self.pairs.append(pair) 

    @classmethod
    def from_lm_harness_task(
        cls,
        name: str,
        task,
        docs: List[Dict[str, Any]],
        limit: Optional[int] = None,
        split: str = "train",
        task_type: Optional[str] = None
    ) -> 'ContrastivePairSet':
        """
        Create a ContrastivePairSet from an lm-harness task.
        
        Args:
            name: Name for the pair set
            task: Task object from lm-harness
            docs: List of documents from the task
            limit: Optional limit on number of documents to process
            split: Which split this is (train/validation/test)
            task_type: Optional task type override
            
        Returns:
            ContrastivePairSet instance
        """
        if limit is not None and limit > 0:
            docs = docs[:limit]
        
        # Detect task type from name if not provided
        if task_type is None:
            task_type = cls._detect_task_type(name)
        
        pair_set = cls(name=f"{name}_{split}", task_type=task_type)
        
        # Process documents based on task type
        if 'truthfulqa' in name.lower():
            pair_set._create_truthfulqa_pairs_from_docs(task, docs)
        elif any(mc_task in name.lower() for mc_task in ['hellaswag', 'mmlu', 'arc', 'winogrande', 'piqa']):
            pair_set._create_multiple_choice_pairs_from_docs(task, docs)
        else:
            pair_set._create_generic_pairs_from_docs(task, docs)
        
        return pair_set

    @classmethod
    def from_csv_file(
        cls,
        name: str,
        csv_path: Union[str, Path],
        question_col: str = "question",
        correct_col: str = "correct_answer", 
        incorrect_col: str = "incorrect_answer",
        limit: Optional[int] = None
    ) -> 'ContrastivePairSet':
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
        cls,
        name: str,
        json_path: Union[str, Path],
        limit: Optional[int] = None
    ) -> 'ContrastivePairSet':
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
        
        with open(json_path, 'r', encoding='utf-8') as f:
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
                
                question = item.get('question', '')
                
                # Handle single answer format
                if 'correct_answer' in item and 'incorrect_answer' in item:
                    correct_answer = str(item['correct_answer'])
                    incorrect_answer = str(item['incorrect_answer'])
                    pair_set._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)
                
                # Handle multiple answers format
                elif 'correct_answers' in item and 'incorrect_answers' in item:
                    correct_answers = item['correct_answers']
                    incorrect_answers = item['incorrect_answers']
                    
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
        
        pair = ContrastivePair(
            prompt=prompt,
            positive_response=pos_resp,
            negative_response=neg_resp
        )
        
        # Store metadata
        pair.question = question
        pair.correct_answer = correct_answer
        pair.incorrect_answer = incorrect_answer
        
        self.pairs.append(pair)

    def _create_truthfulqa_pairs_from_docs(self, task, docs: List[Dict[str, Any]]):
        """Create contrastive pairs from TruthfulQA documents."""
        for doc in docs:
            try:
                # Extract question
                if hasattr(task, 'doc_to_text'):
                    question = task.doc_to_text(doc)
                else:
                    question = doc.get('question', str(doc))
                
                # Extract correct and incorrect answers from mc1_targets
                correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                
                # Find the correct answer
                correct_answer = None
                for i, label in enumerate(correct_labels):
                    if label == 1 and i < len(correct_answers):
                        correct_answer = correct_answers[i]
                        break
                
                # Find an incorrect answer
                incorrect_answer = None
                for i, label in enumerate(correct_labels):
                    if label == 0 and i < len(correct_answers):
                        incorrect_answer = correct_answers[i]
                        break
                
                if correct_answer and incorrect_answer:
                    self._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)
                    
            except Exception as e:
                print(f"Warning: Skipping TruthfulQA doc due to error: {e}")
                continue

    def _create_multiple_choice_pairs_from_docs(self, task, docs: List[Dict[str, Any]]):
        """Create contrastive pairs from multiple choice task documents."""
        for doc in docs:
            try:
                # Extract question
                if hasattr(task, 'doc_to_text'):
                    question = task.doc_to_text(doc)
                else:
                    question = doc.get('question', doc.get('ctx', str(doc)))
                
                # Get choices and correct answer
                choices = doc.get('choices', [])
                if not choices:
                    continue
                
                # Get correct answer index
                correct_idx = doc.get('gold', doc.get('answer', doc.get('label', 0)))
                if isinstance(correct_idx, list) and len(correct_idx) > 0:
                    correct_idx = correct_idx[0]
                elif not isinstance(correct_idx, int):
                    correct_idx = 0
                
                # Ensure valid index
                if correct_idx < 0 or correct_idx >= len(choices):
                    continue
                
                correct_answer = choices[correct_idx]
                
                # Find an incorrect answer
                incorrect_indices = [i for i in range(len(choices)) if i != correct_idx]
                if not incorrect_indices:
                    continue
                
                incorrect_idx = random.choice(incorrect_indices)
                incorrect_answer = choices[incorrect_idx]
                
                self._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)
                
            except Exception as e:
                print(f"Warning: Skipping multiple choice doc due to error: {e}")
                continue

    def _create_generic_pairs_from_docs(self, task, docs: List[Dict[str, Any]]):
        """Create contrastive pairs from generic task documents."""
        for doc in docs:
            try:
                # Extract question/prompt
                if hasattr(task, 'doc_to_text'):
                    question = task.doc_to_text(doc)
                else:
                    question = doc.get('question', doc.get('text', str(doc)))
                
                # Extract target/correct answer
                if hasattr(task, 'doc_to_target'):
                    correct_answer = str(task.doc_to_target(doc))
                else:
                    correct_answer = str(doc.get('target', doc.get('answer', 'Correct answer')))
                
                # Create a generic incorrect answer
                incorrect_answer = "Incorrect or irrelevant response"
                
                self._create_qa_contrastive_pair(question, correct_answer, incorrect_answer)
                
            except Exception as e:
                print(f"Warning: Skipping generic doc due to error: {e}")
                continue

    @staticmethod
    def _detect_task_type(task_name: str) -> str:
        """Detect task type from task name."""
        task_name_lower = task_name.lower()
        
        if 'truthfulqa' in task_name_lower:
            return 'truthfulqa'
        elif any(mc_task in task_name_lower for mc_task in ['hellaswag', 'mmlu', 'arc', 'winogrande', 'piqa']):
            return 'multiple_choice'
        elif any(bool_task in task_name_lower for bool_task in ['boolq']):
            return 'boolean'
        elif any(math_task in task_name_lower for math_task in ['gsm', 'math']):
            return 'math'
        else:
            return 'generic'

    def save_to_csv(self, csv_path: Union[str, Path], include_metadata: bool = True):
        """
        Save the contrastive pairs to a CSV file.
        
        Args:
            csv_path: Path to save the CSV file
            include_metadata: Whether to include metadata columns
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for i, pair in enumerate(self.pairs):
            row = {
                'pair_id': i,
                'prompt': pair.prompt,
                'positive_response': pair.positive_response.text,
                'negative_response': pair.negative_response.text
            }
            
            if include_metadata and hasattr(pair, 'question'):
                row.update({
                    'question': getattr(pair, 'question', ''),
                    'correct_answer': getattr(pair, 'correct_answer', ''),
                    'incorrect_answer': getattr(pair, 'incorrect_answer', '')
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(data)} contrastive pairs to {csv_path}")

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
                'pair_id': i,
                'prompt': pair.prompt,
                'positive_response': pair.positive_response.text,
                'negative_response': pair.negative_response.text
            }
            
            if include_metadata and hasattr(pair, 'question'):
                item.update({
                    'question': getattr(pair, 'question', ''),
                    'correct_answer': getattr(pair, 'correct_answer', ''),
                    'incorrect_answer': getattr(pair, 'incorrect_answer', '')
                })
            
            data.append(item)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} contrastive pairs to {json_path}")

    @classmethod
    def extract_qa_pairs_from_task_docs(
        cls,
        task_name: str,
        task_data,
        docs: List[Dict[str, Any]]
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
        
        for doc in docs:
            try:
                # Extract question using task's doc_to_text method
                if hasattr(task_data, 'doc_to_text'):
                    question = task_data.doc_to_text(doc)
                else:
                    question = doc.get('question', str(doc))
                
                # Task-specific answer extraction
                correct_answer = None
                incorrect_answer = None
                
                if task_name == 'truthful_qa':
                    # TruthfulQA-specific extraction
                    correct_answers = doc.get('mc1_targets', {}).get('choices', [])
                    correct_labels = doc.get('mc1_targets', {}).get('labels', [])
                    
                    # Find the correct answer
                    for i, label in enumerate(correct_labels):
                        if label == 1 and i < len(correct_answers):
                            correct_answer = correct_answers[i]
                            break
                    
                    # Find an incorrect answer
                    for i, label in enumerate(correct_labels):
                        if label == 0 and i < len(correct_answers):
                            incorrect_answer = correct_answers[i]
                            break
                            
                elif task_name == 'hellaswag':
                    # HellaSwag-specific extraction
                    endings = doc.get('endings', [])
                    label = doc.get('label', 0)
                    
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
                                
                else:
                    # Generic extraction for other tasks
                    # Try common answer fields
                    if 'choices' in doc and 'label' in doc:
                        choices = doc['choices']
                        label = doc['label']
                        
                        # Convert string label to int if needed
                        if isinstance(label, str):
                            try:
                                label = int(label)
                            except ValueError:
                                label = 0
                        
                        if isinstance(label, int) and 0 <= label < len(choices):
                            correct_answer = choices[label]
                            # Use first different choice as incorrect
                            for i, choice in enumerate(choices):
                                if i != label:
                                    incorrect_answer = choice
                                    break
                    elif 'answer' in doc:
                        correct_answer = doc['answer']
                        # For simple answer tasks, create a generic incorrect answer
                        incorrect_answer = "This is incorrect"
                    elif hasattr(task_data, 'doc_to_target'):
                        correct_answer = task_data.doc_to_target(doc)
                        incorrect_answer = "This is incorrect"
                
                if correct_answer and incorrect_answer:
                    qa_pairs.append({
                        'question': question,
                        'correct_answer': correct_answer,
                        'incorrect_answer': incorrect_answer
                    })
                    
            except Exception as e:
                # Skip problematic docs
                continue
        
        return qa_pairs 