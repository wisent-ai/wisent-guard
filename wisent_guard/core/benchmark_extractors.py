"""
Benchmark-specific data extraction logic for multiple choice questions and answers.

Each benchmark has its own data structure and format. This module centralizes
the extraction logic to cleanly handle the differences between benchmarks.
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import BigCode extractors at module level with better error handling
BIGCODE_EXTRACTORS = {}
get_bigcode_extractor = None
BIGCODE_AVAILABLE = False


class UnsupportedBenchmarkError(Exception):
    """Raised when benchmark has no extractor."""


class BenchmarkExtractor:
    """Base class for benchmark-specific extractors."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Extract a QA pair from a document.

        Args:
            doc: Document from the benchmark
            task_data: Task data object with methods like doc_to_text

        Returns:
            Dict with 'question', 'formatted_question', 'correct_answer' or None
        """
        raise NotImplementedError("Subclasses must implement extract_qa_pair")

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Extract a contrastive pair from a document.

        Args:
            doc: Document from the benchmark
            task_data: Task data object

        Returns:
            Dict with 'question', 'correct_answer', 'incorrect_answer' or None
        """
        raise NotImplementedError("Subclasses must implement extract_contrastive_pair")


class WinograndeExtractor(BenchmarkExtractor):
    """Extractor for Winogrande benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Winogrande format:
        - doc['sentence']: sentence with blank
        - doc['option1'], doc['option2']: two options
        - doc['answer']: '1' or '2' indicating correct option
        """
        try:
            sentence = doc.get("sentence", "")
            option1 = doc.get("option1", "")
            option2 = doc.get("option2", "")
            answer = doc.get("answer", "")

            if not all([sentence, option1, option2, answer]):
                return None

            # Create the question
            question = f"Complete the sentence: {sentence}"

            # 🚨 FIX: Don't use doc_to_text for winogrande - it returns integers instead of strings!
            # This is a bug in lm-eval-harness winogrande task configuration
            formatted_question = f"{question}\nA. {option1}\nB. {option2}"

            # Get correct and incorrect answers
            correct_answer = option1 if answer == "1" else option2
            incorrect_answer = option2 if answer == "1" else option1

            return {
                "question": question,
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting Winogrande QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for Winogrande."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            option1 = doc.get("option1", "")
            option2 = doc.get("option2", "")
            answer = doc.get("answer", "")

            correct_choice = option1 if answer == "1" else option2
            incorrect_choice = option2 if answer == "1" else option1

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting Winogrande contrastive pair: {e}")
            return None


class ARCExtractor(BenchmarkExtractor):
    """Extractor for ARC (AI2 Reasoning Challenge) benchmarks."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        ARC format:
        - doc['question']: the question
        - doc['choices']['text']: list of choice texts
        - doc['choices']['label']: list of choice labels (A, B, C, D)
        - doc['answerKey']: correct answer letter
        """
        try:
            question = doc.get("question", "")
            choices = doc.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            answer_key = doc.get("answerKey", "")

            if not all([question, choice_texts, choice_labels, answer_key]):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                for i, (label, text) in enumerate(zip(choice_labels, choice_texts)):
                    formatted_question += f"\n{label}. {text}"

            # Get correct answer
            correct_answer = None
            for i, label in enumerate(choice_labels):
                if label == answer_key and i < len(choice_texts):
                    correct_answer = choice_texts[i]
                    break

            if not correct_answer:
                return None

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting ARC QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for ARC."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            choices = doc.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            answer_key = doc.get("answerKey", "")

            correct_choice = None
            incorrect_choice = None

            for i, label in enumerate(choice_labels):
                if label == answer_key and i < len(choice_texts):
                    correct_choice = choice_texts[i]
                elif i < len(choice_texts) and incorrect_choice is None:
                    incorrect_choice = choice_texts[i]

            if not all([correct_choice, incorrect_choice]):
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting ARC contrastive pair: {e}")
            return None


class HellaSwagExtractor(BenchmarkExtractor):
    """Extractor for HellaSwag benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        HellaSwag format:
        - doc['ctx']: context/premise
        - doc['endings']: list of possible endings
        - doc['label']: index of correct ending
        """
        try:
            ctx = doc.get("ctx", "")
            endings = doc.get("endings", [])
            label = doc.get("label", 0)

            if not all([ctx, endings]) or label >= len(endings):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Context: {ctx}\nChoose the best ending:"
                for i, ending in enumerate(endings):
                    formatted_question += f"\n{chr(65 + i)}. {ending}"

            # Get correct answer
            correct_answer = endings[label]

            return {
                "question": f"Context: {ctx}\nChoose the best ending:",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting HellaSwag QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for HellaSwag."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            endings = doc.get("endings", [])
            label = doc.get("label", 0)

            if label >= len(endings):
                return None

            correct_choice = endings[label]
            # Get first incorrect choice
            incorrect_choice = None
            for i, ending in enumerate(endings):
                if i != label:
                    incorrect_choice = ending
                    break

            if not incorrect_choice:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting HellaSwag contrastive pair: {e}")
            return None


class TruthfulQAExtractor(BenchmarkExtractor):
    """Extractor for TruthfulQA benchmarks."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        TruthfulQA format:
        - doc['question']: the question
        - doc['mc1_targets']['choices']: list of choices
        - doc['mc1_targets']['labels']: list of 0/1 labels (1 = correct)
        """
        try:
            question = doc.get("question", "")
            if not question:
                return None

            # Extract choices and labels from mc1_targets
            mc1_targets = doc.get("mc1_targets", {})
            choices = mc1_targets.get("choices", [])
            labels = mc1_targets.get("labels", [])

            if len(choices) < 2 or len(labels) != len(choices):
                return None

            # Format the question with multiple choice options
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                for i, choice in enumerate(choices):
                    letter = chr(65 + i)  # A, B, C, D, ...
                    formatted_question += f"\n{letter}. {choice}"

            # Find correct answer letter
            correct_answer_letter = None
            for i, label in enumerate(labels):
                if label == 1:
                    correct_answer_letter = chr(65 + i)  # Convert to A, B, C, D
                    break

            if not correct_answer_letter:
                return None

            return {
                "question": question,
                "formatted_question": formatted_question,
                "correct_answer": correct_answer_letter,  # Return letter, not text
            }

        except Exception as e:
            logger.debug(f"Error extracting TruthfulQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for TruthfulQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            mc1_targets = doc.get("mc1_targets", {})
            choices = mc1_targets.get("choices", [])
            labels = mc1_targets.get("labels", [])

            correct_letter = None
            incorrect_letter = None

            for i, label in enumerate(labels):
                if label == 1:
                    correct_letter = chr(65 + i)  # Convert to A, B, C, D
                elif label == 0 and incorrect_letter is None:
                    incorrect_letter = chr(65 + i)  # First incorrect option

            if not all([correct_letter, incorrect_letter]):
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_letter,  # Use letter, not text
                "incorrect_answer": incorrect_letter,  # Use letter, not text
            }

        except Exception as e:
            logger.debug(f"Error extracting TruthfulQA contrastive pair: {e}")
            return None


class BoolQExtractor(BenchmarkExtractor):
    """Extractor for BoolQ benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        BoolQ format:
        - doc['question']: the question
        - doc['passage']: the passage
        - doc['answer']: True/False
        """
        try:
            question = doc.get("question", "")
            passage = doc.get("passage", "")
            answer = doc.get("answer", False)

            if not all([question, passage]) or answer is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Passage: {passage}\nQuestion: {question}"

            # Get correct answer
            correct_answer = "True" if answer else "False"

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting BoolQ QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for BoolQ."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            answer = doc.get("answer", False)
            correct_choice = "True" if answer else "False"
            incorrect_choice = "False" if answer else "True"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting BoolQ contrastive pair: {e}")
            return None


class GSM8KExtractor(BenchmarkExtractor):
    """Extractor for GSM8K, MATH-500, and AIME_2024 benchmarks."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Supports multiple formats:
        - GSM8K format: doc['question'] -> doc['answer']
        - MATH-500 format: doc['problem'] -> doc['answer']
        - AIME_2024 format: doc['Problem'] -> doc['Answer']
        """
        try:
            # Handle multiple field name variants
            question = doc.get("question", "") or doc.get("problem", "") or doc.get("Problem", "")
            answer = doc.get("answer", "") or doc.get("Answer", "")

            if not all([question, str(answer) if answer is not None else None]):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question

            # Extract numerical answer from the answer string
            # GSM8K answers typically end with #### followed by the number
            answer_str = str(answer)
            numerical_answer = answer_str
            if "####" in answer_str:
                numerical_answer = answer_str.split("####")[-1].strip()

            return {"question": question, "formatted_question": formatted_question, "correct_answer": numerical_answer}

        except Exception as e:
            logger.debug(f"Error extracting GSM8K QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for GSM8K."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]

            # Create an incorrect answer (slightly different number)
            try:
                num = float(correct_answer)
                incorrect_answer = str(num + 1)
            except ValueError:
                incorrect_answer = "Wrong answer"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting GSM8K contrastive pair: {e}")
            return None


class MMLUExtractor(BenchmarkExtractor):
    """Extractor for MMLU benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MMLU format:
        - doc['question']: the question
        - doc['choices']: list of choices (A, B, C, D)
        - doc['answer']: index of correct choice (0, 1, 2, 3)

        MMMLU format:
        - doc['instruction']: the question
        - doc['option_a'], doc['option_b'], doc['option_c'], doc['option_d']: choices
        - doc['answer']: letter of correct choice (A, B, C, D)
        """
        try:
            # Check for MMMLU format first
            if "instruction" in doc and "option_a" in doc:
                question = doc.get("instruction", "")
                choices = [
                    doc.get("option_a", ""),
                    doc.get("option_b", ""),
                    doc.get("option_c", ""),
                    doc.get("option_d", ""),
                ]
                choices = [c for c in choices if c]  # Filter out empty choices
                answer_letter = doc.get("answer", "A")
                answer = ord(answer_letter) - ord("A")  # Convert letter to index
            else:
                # Standard MMLU format
                question = doc.get("question", "")
                choices = doc.get("choices", [])
                answer = doc.get("answer", 0)

            if not all([question, choices]) or answer >= len(choices):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                for i, choice in enumerate(choices):
                    formatted_question += f"\n{chr(65 + i)}. {choice}"

            # Get correct answer
            correct_answer = choices[answer]

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting MMLU QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MMLU."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # Check for MMMLU format first
            if "instruction" in doc and "option_a" in doc:
                choices = [
                    doc.get("option_a", ""),
                    doc.get("option_b", ""),
                    doc.get("option_c", ""),
                    doc.get("option_d", ""),
                ]
                choices = [c for c in choices if c]  # Filter out empty choices
                answer_letter = doc.get("answer", "A")
                answer = ord(answer_letter) - ord("A")  # Convert letter to index
            else:
                # Standard MMLU format
                choices = doc.get("choices", [])
                answer = doc.get("answer", 0)

            if answer >= len(choices):
                return None

            correct_choice = choices[answer]
            # Get first incorrect choice
            incorrect_choice = None
            for i, choice in enumerate(choices):
                if i != answer:
                    incorrect_choice = choice
                    break

            if not incorrect_choice:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting MMLU contrastive pair: {e}")
            return None


class PIQAExtractor(BenchmarkExtractor):
    """Extractor for PIQA benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        PIQA format:
        - doc['goal']: the goal/question
        - doc['sol1'], doc['sol2']: two solutions
        - doc['label']: 0 or 1 indicating correct solution
        """
        try:
            goal = doc.get("goal", "")
            sol1 = doc.get("sol1", "")
            sol2 = doc.get("sol2", "")
            label = doc.get("label", 0)

            if not all([goal, sol1, sol2]) or label not in [0, 1]:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Goal: {goal}\nA. {sol1}\nB. {sol2}"

            # Get correct answer
            correct_answer = sol1 if label == 0 else sol2

            return {"question": goal, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting PIQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for PIQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            sol1 = doc.get("sol1", "")
            sol2 = doc.get("sol2", "")
            label = doc.get("label", 0)

            correct_choice = sol1 if label == 0 else sol2
            incorrect_choice = sol2 if label == 0 else sol1

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting PIQA contrastive pair: {e}")
            return None


class COPAExtractor(BenchmarkExtractor):
    """Extractor for COPA benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        COPA format:
        - doc['premise']: the premise
        - doc['choice1'], doc['choice2']: two choices
        - doc['question']: 'cause' or 'effect'
        - doc['label']: 0 or 1 indicating correct choice
        """
        try:
            premise = doc.get("premise", "")
            choice1 = doc.get("choice1", "")
            choice2 = doc.get("choice2", "")
            question = doc.get("question", "")
            label = doc.get("label", 0)

            if not all([premise, choice1, choice2, question]) or label not in [0, 1]:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                question_text = f"What was the {question}?"
                formatted_question = f"Premise: {premise}\n{question_text}\nA. {choice1}\nB. {choice2}"

            # Get correct answer
            correct_answer = choice1 if label == 0 else choice2

            return {
                "question": f"Premise: {premise}\nWhat was the {question}?",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting COPA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for COPA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            choice1 = doc.get("choice1", "")
            choice2 = doc.get("choice2", "")
            label = doc.get("label", 0)

            correct_choice = choice1 if label == 0 else choice2
            incorrect_choice = choice2 if label == 0 else choice1

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting COPA contrastive pair: {e}")
            return None


class OpenBookQAExtractor(BenchmarkExtractor):
    """Extractor for OpenBookQA benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        OpenBookQA format:
        - doc['question_stem']: the question
        - doc['choices']['text']: list of choice texts
        - doc['choices']['label']: list of choice labels (A, B, C, D)
        - doc['answerKey']: correct answer letter
        """
        try:
            question = doc.get("question_stem", "")
            choices = doc.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            answer_key = doc.get("answerKey", "")

            if not all([question, choice_texts, choice_labels, answer_key]):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                for i, (label, text) in enumerate(zip(choice_labels, choice_texts)):
                    formatted_question += f"\n{label}. {text}"

            # Get correct answer
            correct_answer = None
            for i, label in enumerate(choice_labels):
                if label == answer_key and i < len(choice_texts):
                    correct_answer = choice_texts[i]
                    break

            if not correct_answer:
                return None

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting OpenBookQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for OpenBookQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            choices = doc.get("choices", {})
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            answer_key = doc.get("answerKey", "")

            correct_choice = None
            incorrect_choice = None

            for i, label in enumerate(choice_labels):
                if label == answer_key and i < len(choice_texts):
                    correct_choice = choice_texts[i]
                elif i < len(choice_texts) and incorrect_choice is None:
                    incorrect_choice = choice_texts[i]

            if not all([correct_choice, incorrect_choice]):
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting OpenBookQA contrastive pair: {e}")
            return None


class SQuAD2Extractor(BenchmarkExtractor):
    """Extractor for SQuAD2 benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        SQuAD2 format:
        - doc['question']: the question
        - doc['context']: the context passage
        - doc['answers']['text']: list of possible answers
        """
        try:
            question = doc.get("question", "")
            context = doc.get("context", "")
            answers = doc.get("answers", {})
            answer_texts = answers.get("text", [])

            if not all([question, context]):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Context: {context}\nQuestion: {question}"

            # Get correct answer
            correct_answer = answer_texts[0] if answer_texts else "No answer"

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting SQuAD2 QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for SQuAD2."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]

            # Generate meaningful incorrect answers for reading comprehension
            if correct_answer == "No answer":
                incorrect_answer = "The answer is clearly stated in the passage."
            else:
                # Create plausible but incorrect answers
                incorrect_answers = [
                    "The information is not provided in the text.",
                    "This cannot be determined from the passage.",
                    "The passage does not contain this information.",
                ]
                incorrect_answer = random.choice(incorrect_answers)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting SQuAD2 contrastive pair: {e}")
            return None


class RACEExtractor(BenchmarkExtractor):
    """Extractor for RACE benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        RACE format:
        - doc['article']: the article/passage
        - doc['question']: the question
        - doc['options']: list of options
        - doc['answer']: correct option letter (A, B, C, D)
        """
        try:
            article = doc.get("article", "")
            question = doc.get("question", "")
            options = doc.get("options", [])
            answer = doc.get("answer", "")

            if not all([article, question, options, answer]):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Article: {article}\nQuestion: {question}"
                for i, option in enumerate(options):
                    formatted_question += f"\n{chr(65 + i)}. {option}"

            # Get correct answer
            answer_idx = ord(answer) - ord("A")
            correct_answer = options[answer_idx] if answer_idx < len(options) else options[0]

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting RACE QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for RACE."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            options = doc.get("options", [])
            answer = doc.get("answer", "")

            answer_idx = ord(answer) - ord("A")
            correct_choice = options[answer_idx] if answer_idx < len(options) else options[0]

            # Get first incorrect choice
            incorrect_choice = None
            for i, option in enumerate(options):
                if i != answer_idx:
                    incorrect_choice = option
                    break

            if not incorrect_choice:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting RACE contrastive pair: {e}")
            return None


class MRPCExtractor(BenchmarkExtractor):
    """Extractor for MRPC (Microsoft Research Paraphrase Corpus) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MRPC format:
        - doc['sentence1']: First sentence
        - doc['sentence2']: Second sentence
        - doc['label']: 1 if paraphrase, 0 if not
        """
        try:
            sentence1 = doc.get("sentence1", "")
            sentence2 = doc.get("sentence2", "")
            label = doc.get("label")

            if not all([sentence1, sentence2]) or label is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (
                    f"Sentence 1: {sentence1}\n"
                    f"Sentence 2: {sentence2}\n"
                    f"Question: Do both sentences mean the same thing?"
                )

            # Get correct answer
            correct_answer = "yes" if label == 1 else "no"

            return {
                "question": f"Do these sentences mean the same thing? '{sentence1}' and '{sentence2}'",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting MRPC QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MRPC."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label")
            if label is None:
                return None

            correct_choice = "yes" if label == 1 else "no"
            incorrect_choice = "no" if label == 1 else "yes"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting MRPC contrastive pair: {e}")
            return None


class QNLIExtractor(BenchmarkExtractor):
    """Extractor for QNLI (Question-answering Natural Language Inference) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        QNLI format:
        - doc['question']: the question
        - doc['sentence']: the sentence
        - doc['label']: 0 if entails, 1 if not entails
        """
        try:
            question = doc.get("question", "")
            sentence = doc.get("sentence", "")
            label = doc.get("label")

            if not all([question, sentence]) or label is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"{question}\n{sentence}\nQuestion: Does this response answer the question?"

            # Get correct answer
            correct_answer = "yes" if label == 0 else "no"

            return {
                "question": f"Does this sentence answer the question? Question: {question} Sentence: {sentence}",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting QNLI QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for QNLI."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label")
            if label is None:
                return None

            correct_choice = "yes" if label == 0 else "no"
            incorrect_choice = "no" if label == 0 else "yes"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting QNLI contrastive pair: {e}")
            return None


class QQPExtractor(BenchmarkExtractor):
    """Extractor for QQP (Quora Question Pairs) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        QQP format:
        - doc['question1']: First question
        - doc['question2']: Second question
        - doc['label']: 1 if duplicate, 0 if not
        """
        try:
            question1 = doc.get("question1", "")
            question2 = doc.get("question2", "")
            label = doc.get("label")

            if not all([question1, question2]) or label is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (
                    f"Question 1: {question1}\nQuestion 2: {question2}\nQuestion: Do both questions ask the same thing?"
                )

            # Get correct answer
            correct_answer = "yes" if label == 1 else "no"

            return {
                "question": f"Do these questions ask the same thing? '{question1}' and '{question2}'",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting QQP QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for QQP."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label")
            if label is None:
                return None

            correct_choice = "yes" if label == 1 else "no"
            incorrect_choice = "no" if label == 1 else "yes"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting QQP contrastive pair: {e}")
            return None


class RTEExtractor(BenchmarkExtractor):
    """Extractor for RTE (Recognizing Textual Entailment) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        RTE format:
        - doc['sentence1']: First sentence (premise)
        - doc['sentence2']: Second sentence (hypothesis)
        - doc['label']: 0 if not entailment, 1 if entailment
        """
        try:
            sentence1 = doc.get("sentence1", "")
            sentence2 = doc.get("sentence2", "")
            label = doc.get("label")

            if not all([sentence1, sentence2]) or label is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"{sentence1}\nQuestion: {sentence2} True or False?"

            # Get correct answer
            correct_answer = "True" if label == 1 else "False"

            return {
                "question": f"Given '{sentence1}', is it true that '{sentence2}'?",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting RTE QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for RTE."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label")
            if label is None:
                return None

            correct_choice = "True" if label == 1 else "False"
            incorrect_choice = "False" if label == 1 else "True"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting RTE contrastive pair: {e}")
            return None


class SST2Extractor(BenchmarkExtractor):
    """Extractor for SST2 (Stanford Sentiment Treebank) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        SST2 format:
        - doc['sentence']: the sentence
        - doc['label']: 0 if negative, 1 if positive
        """
        try:
            sentence = doc.get("sentence", "")
            label = doc.get("label")

            if not sentence or label is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"{sentence}\nQuestion: Is this sentence positive or negative?"

            # Get correct answer
            correct_answer = "positive" if label == 1 else "negative"

            return {
                "question": f"What is the sentiment of this sentence: '{sentence}'",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting SST2 QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for SST2."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label")
            if label is None:
                return None

            correct_choice = "positive" if label == 1 else "negative"
            incorrect_choice = "negative" if label == 1 else "positive"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting SST2 contrastive pair: {e}")
            return None


class WNLIExtractor(BenchmarkExtractor):
    """Extractor for WNLI (Winograd Natural Language Inference) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        WNLI format:
        - doc['sentence1']: First sentence
        - doc['sentence2']: Second sentence
        - doc['label']: 0 if not entailment, 1 if entailment
        """
        try:
            sentence1 = doc.get("sentence1", "")
            sentence2 = doc.get("sentence2", "")
            label = doc.get("label")

            if not all([sentence1, sentence2]) or label is None:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"{sentence1}\nQuestion: {sentence2} True or False?"

            # Get correct answer
            correct_answer = "True" if label == 1 else "False"

            return {
                "question": f"Given '{sentence1}', is it true that '{sentence2}'?",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting WNLI QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for WNLI."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label")
            if label is None:
                return None

            correct_choice = "True" if label == 1 else "False"
            incorrect_choice = "False" if label == 1 else "True"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting WNLI contrastive pair: {e}")
            return None


class CoQAExtractor(BenchmarkExtractor):
    """Extractor for CoQA conversational QA benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Extract QA pair from CoQA document.

        CoQA format:
        - doc['story']: The context passage
        - doc['questions']['input_text']: List of questions
        - doc['answers']['input_text']: List of answers
        - doc['questions']['turn_id']: Turn IDs
        """
        try:
            story = doc.get("story", "")
            questions_data = doc.get("questions", {})
            answers_data = doc.get("answers", {})

            question_texts = questions_data.get("input_text", [])
            answer_texts = answers_data.get("input_text", [])

            if not all([story, question_texts, answer_texts]):
                return None

            # Use the last question in the conversation for the current turn
            # This is what lm-eval does in doc_to_text
            if not question_texts:
                return None

            # Build conversation history
            conversation_history = []
            for i in range(len(question_texts) - 1):  # All but the last
                if i < len(answer_texts):
                    conversation_history.append(f"Q: {question_texts[i]}")
                    conversation_history.append(f"A: {answer_texts[i]}")

            # Current question is the last one
            current_question = question_texts[-1]

            # Format the question with story and conversation history
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = story + "\n\n"
                formatted_question += "\n".join(conversation_history)
                if conversation_history:
                    formatted_question += "\n\n"
                formatted_question += f"Q: {current_question}\n\nA:"

            # Get the correct answer (last answer if available)
            if hasattr(task_data, "doc_to_target"):
                target = task_data.doc_to_target(doc)
                if isinstance(target, list) and target:
                    correct_answer = target[0]
                else:
                    correct_answer = str(target) if target else "no"
            else:
                correct_answer = answer_texts[-1] if len(answer_texts) == len(question_texts) else "no"

            return {
                "question": current_question,
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting CoQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for CoQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]

            # Generate contextually appropriate incorrect answers
            incorrect_answers = [
                "I don't know",
                "The passage doesn't say",
                "Unable to answer from the given information",
                "Not mentioned in the text",
            ]

            # For yes/no questions, flip the answer
            if correct_answer.lower() in ["yes", "no"]:
                incorrect_answer = "no" if correct_answer.lower() == "yes" else "yes"
            # For short factual answers, use a generic incorrect response
            elif len(correct_answer.split()) <= 3:
                incorrect_answer = "unknown"
            else:
                # Use one of the template responses
                incorrect_answer = incorrect_answers[0]

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting CoQA contrastive pair: {e}")
            return None


class ASDivExtractor(BenchmarkExtractor):
    """Extractor for ASDdiv arithmetic story problems."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None):
        """Extract QA pair from ASDdiv document.

        ASDdiv contains arithmetic story problems where we need to generate
        incorrect answers for contrastive learning.
        """
        # Get the problem text and solution
        problem = doc.get("body", "") + " " + doc.get("question", "")
        correct_answer = str(doc.get("answer", ""))

        # Generate incorrect answer by modifying the correct one
        try:
            correct_num = float(correct_answer)
            # Generate a plausible wrong answer
            import random

            operation = random.choice(
                [
                    lambda x: x + random.randint(1, 10),
                    lambda x: x - random.randint(1, 10),
                    lambda x: x * 2,
                    lambda x: x / 2 if x > 1 else x + 5,
                ]
            )
            incorrect_num = operation(correct_num)

            # Format to match original (int vs float)
            if "." not in correct_answer:
                incorrect_answer = str(int(incorrect_num))
            else:
                incorrect_answer = str(round(incorrect_num, 2))

        except (ValueError, TypeError):
            # If answer is not numeric, create a simple wrong answer
            incorrect_answer = correct_answer + " (wrong)"

        return {"question": problem.strip(), "correct_answer": correct_answer, "incorrect_answer": incorrect_answer}

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for ASDdiv - just return the QA pair which already has both answers."""
        return self.extract_qa_pair(doc, task_data)


class WikiTextExtractor(BenchmarkExtractor):
    """Extractor for WikiText benchmark (language modeling/perplexity tasks)."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        WikiText format:
        - doc['page']: the full text document for perplexity calculation

        For language modeling tasks, we don't have traditional QA pairs.
        Instead, we create a text continuation task.
        """
        try:
            # WikiText uses 'page' field for document text
            text = doc.get("page", doc.get("text", ""))

            if not text or not isinstance(text, str):
                return None

            # For perplexity evaluation, we need the full text
            # Split into context and continuation for evaluation
            words = text.split()
            if len(words) < 10:
                return None

            # Use first 30% as context, rest as continuation
            split_point = max(10, len(words) // 3)
            context_words = words[:split_point]
            continuation_words = words[split_point:]

            context = " ".join(context_words)
            continuation = " ".join(continuation_words)

            # Create a prompt that asks for text continuation
            return {
                "question": f"Continue this text: {context[:200]}...",
                "formatted_question": f"Continue the following text:\n\n{context}",
                "correct_answer": continuation,
                "full_text": text,  # Store full text for perplexity calculation
            }

        except Exception as e:
            logger.debug(f"Error extracting WikiText QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for WikiText."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # For language modeling, we can't create meaningful contrastive pairs
            # Instead, we'll use the full text vs a corrupted version
            correct_text = qa_pair.get("full_text", qa_pair["correct_answer"])

            # Create a "bad" continuation by using random words
            import random

            words = correct_text.split()
            if len(words) > 5:
                # Shuffle some words to create an unnatural continuation
                shuffled_words = words.copy()
                random.shuffle(shuffled_words[: min(20, len(shuffled_words))])
                incorrect_text = " ".join(shuffled_words)
            else:
                incorrect_text = "This is an incorrect and unrelated continuation that doesn't match the context."

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_text,
                "incorrect_answer": incorrect_text,
            }

        except Exception as e:
            logger.debug(f"Error extracting WikiText contrastive pair: {e}")
            return None

    def get_perplexity_text(self, doc: Dict[str, Any]) -> Optional[str]:
        """Get the text to use for perplexity calculation."""
        return doc.get("page", doc.get("text", ""))


class MathQAExtractor(BenchmarkExtractor):
    """Extractor for MathQA benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MathQA format:
        - doc['Problem']: the math problem (note: capitalized)
        - doc['options']: string of options (e.g., "a ) 38 , b ) 27.675 , c ) 30")
        - doc['correct']: correct answer letter (e.g., "a")
        """
        try:
            problem = doc.get("Problem", "")
            options_str = doc.get("options", "")
            correct_letter = doc.get("correct", "")

            if not all([problem, options_str, correct_letter]):
                return None

            # Parse options string into individual choices
            options_parsed = self._parse_options(options_str)

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = problem
                for letter, text in options_parsed.items():
                    formatted_question += f"\n{letter.upper()}. {text}"

            # Get correct answer text from parsed options
            correct_answer = options_parsed.get(correct_letter.lower(), "")

            if not correct_answer:
                return None

            return {"question": problem, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting MathQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MathQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            options_str = doc.get("options", "")
            correct_letter = doc.get("correct", "")

            # Parse options string
            options_parsed = self._parse_options(options_str)

            correct_choice = qa_pair["correct_answer"]
            # Get first incorrect choice
            incorrect_choice = None
            for letter, text in options_parsed.items():
                if letter != correct_letter.lower():
                    incorrect_choice = text
                    break

            if not incorrect_choice:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting MathQA contrastive pair: {e}")
            return None

    def _parse_options(self, options_str: str) -> Dict[str, str]:
        """
        Parse options string like "a ) 38 , b ) 27.675 , c ) 30" into a dict.

        Returns:
            Dict mapping option letters to their values
        """
        options_dict = {}

        try:
            # Split by commas first
            parts = options_str.split(",")

            for part in parts:
                part = part.strip()
                if ")" in part:
                    # Extract letter and value
                    letter_part, value_part = part.split(")", 1)
                    letter = letter_part.strip().lower()
                    value = value_part.strip()

                    if letter and value:
                        options_dict[letter] = value

        except Exception as e:
            logger.debug(f"Error parsing options string '{options_str}': {e}")

        return options_dict


class MCTacoExtractor(BenchmarkExtractor):
    """Extractor for MC-TACO (Multiple Choice Temporal Commonsense) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MC-TACO format:
        - doc['sentence']: context sentence
        - doc['question']: temporal question
        - doc['answer']: candidate answer
        - doc['label']: 0 (no/implausible) or 1 (yes/plausible)
        - doc['category']: temporal reasoning category
        """
        try:
            sentence = doc.get("sentence", "")
            question = doc.get("question", "")
            answer = doc.get("answer", "")
            label = doc.get("label", 0)

            if not all([sentence, question, answer]):
                return None

            # Create the question as shown in doc_to_text
            formatted_question = f"{sentence}\nQuestion: {question}\nAnswer: {answer}\nPlausible:"

            # The correct answer is based on the label (0=no, 1=yes)
            correct_answer = "yes" if label == 1 else "no"

            return {
                "question": f"{sentence} {question}",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting MC-TACO QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MC-TACO."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label", 0)

            # The choices are always 'no' and 'yes'
            correct_choice = "yes" if label == 1 else "no"
            incorrect_choice = "no" if label == 1 else "yes"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting MC-TACO contrastive pair: {e}")
            return None


class QuACExtractor(BenchmarkExtractor):
    """Extractor for QuAC (Question Answering in Context) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        QuAC format (conversational QA):
        - doc['context']: the context/passage
        - doc['question']: the current question
        - doc['answer']: the answer (may include multiple answers)
        - doc['history']: conversation history (optional)
        - doc['followup']: whether this is a followup question (optional)
        - doc['yesno']: yes/no answer indicator (optional)
        """
        try:
            # Try multiple field names for context
            context = doc.get("context", doc.get("background", doc.get("passage", doc.get("section_text", ""))))

            # Try multiple field names for question
            question = doc.get("question", doc.get("query", doc.get("current_question", "")))

            # Try multiple field names for answer
            answer = doc.get("answer", doc.get("orig_answer", doc.get("answers", {})))

            # Handle answer extraction
            if isinstance(answer, dict):
                # QuAC often has answer dict with 'text' field
                answer_text = answer.get("text", "")
                if isinstance(answer_text, list) and answer_text:
                    answer_text = answer_text[0]
            elif isinstance(answer, list) and answer:
                answer_text = answer[0]
            else:
                answer_text = str(answer) if answer else ""

            if not all([context, question]):
                return None

            # Format the question with context
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Context: {context}\n"

                # Add conversation history if available
                history = doc.get("history", doc.get("conversation_history", []))
                if history and isinstance(history, list):
                    # Show last 2 turns of conversation
                    for i, turn in enumerate(history[-2:]):
                        if isinstance(turn, dict):
                            hist_q = turn.get("question", "")
                            hist_a = turn.get("answer", "")
                            if hist_q and hist_a:
                                formatted_question += f"Q{i + 1}: {hist_q}\nA{i + 1}: {hist_a}\n"
                        elif isinstance(turn, str):
                            formatted_question += f"Previous: {turn}\n"

                formatted_question += f"Question: {question}"

            # Handle special QuAC answer types
            if doc.get("yesno") == "y":
                answer_text = "Yes"
            elif doc.get("yesno") == "n":
                answer_text = "No"
            elif not answer_text or answer_text == "CANNOTANSWER":
                answer_text = "Cannot answer based on the context"

            return {"question": question, "formatted_question": formatted_question, "correct_answer": answer_text}

        except Exception as e:
            logger.debug(f"Error extracting QuAC QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for QuAC."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]

            # For yes/no questions
            if doc.get("yesno") in ["y", "n"]:
                incorrect_answer = "No" if doc.get("yesno") == "y" else "Yes"
            # For unanswerable questions
            elif correct_answer == "Cannot answer based on the context":
                incorrect_answer = "The answer is clearly stated in the passage."
            # For regular questions, create contextually plausible but incorrect answers
            else:
                incorrect_answers = [
                    "I don't have enough information to answer that.",
                    "That's not mentioned in the context.",
                    "The passage doesn't provide this information.",
                    "This question cannot be answered from the given text.",
                ]

                # Pick an answer different from the correct one
                incorrect_answer = incorrect_answers[0]
                for candidate in incorrect_answers:
                    if candidate.lower() != correct_answer.lower():
                        incorrect_answer = candidate
                        break

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting QuAC contrastive pair: {e}")
            return None


class LogiQAExtractor(BenchmarkExtractor):
    """Extractor for LogiQA (Logical Reasoning) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        LogiQA format:
        - doc['query']: the full question text including context
        - doc['choices']: list of answer choices
        - doc['gold']: list containing correct answer index (0-based)
        """
        try:
            query = doc.get("query", "")
            choices = doc.get("choices", [])
            gold = doc.get("gold", [])

            if not all([query, choices]) or not gold or gold[0] >= len(choices):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = query

            # Get correct answer
            correct_answer_idx = gold[0]
            correct_answer = choices[correct_answer_idx]

            # Extract just the question part from query (after "Q: " or similar)
            question = query
            if "Q:" in query:
                question = query.split("Q:")[-1].split("Answer Choices:")[0].strip()

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting LogiQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for LogiQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            choices = doc.get("choices", [])
            gold = doc.get("gold", [])

            if not gold or gold[0] >= len(choices):
                return None

            correct_answer_idx = gold[0]
            correct_choice = choices[correct_answer_idx]

            # Get first incorrect choice
            incorrect_choice = None
            for i, choice in enumerate(choices):
                if i != correct_answer_idx:
                    incorrect_choice = choice
                    break

            if not incorrect_choice:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_choice,
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting LogiQA contrastive pair: {e}")
            return None


class LambadaExtractor(BenchmarkExtractor):
    """Extractor for LAMBADA benchmarks (cloze and multilingual)."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        LAMBADA format:
        - doc['text']: the text with a blank to fill
        - doc['domain']: optional domain
        """
        try:
            text = doc.get("text", "")

            if not text:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Complete the following text: {text}"

            # For LAMBADA, we typically need to predict the last word
            # The correct answer might be in a separate field or need to be extracted
            correct_answer = doc.get("answer", doc.get("target", ""))

            # If no explicit answer, try to extract from text pattern
            if not correct_answer and "_____" in text:
                # This is a cloze-style task
                correct_answer = "[MASK]"  # Placeholder for masked token

            return {
                "question": f"Complete: {text}",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting LAMBADA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for LAMBADA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # For LAMBADA, create a plausible but incorrect completion
            correct_answer = qa_pair["correct_answer"]
            incorrect_answer = "incorrect completion"  # Generic incorrect answer

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting LAMBADA contrastive pair: {e}")
            return None


class AI2ARCExtractor(BenchmarkExtractor):
    """Extractor for AI2 ARC benchmark (different from arc_challenge/arc_easy)."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        AI2 ARC format - similar to ARC but might have different structure
        """
        # Reuse ARCExtractor logic
        arc_extractor = ARCExtractor()
        return arc_extractor.extract_qa_pair(doc, task_data)

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for AI2 ARC."""
        arc_extractor = ARCExtractor()
        return arc_extractor.extract_contrastive_pair(doc, task_data)


class GLUEExtractor(BenchmarkExtractor):
    """Extractor for GLUE benchmark suite."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        GLUE contains multiple tasks - delegate to specific extractors
        """
        # GLUE is a collection, not a single task
        # This should not be called directly
        return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for GLUE."""
        return None


class SuperGLUEExtractor(BenchmarkExtractor):
    """Extractor for SuperGLUE benchmark suite."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        SuperGLUE contains multiple tasks - delegate to specific extractors
        """
        # SuperGLUE is a collection, not a single task
        return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for SuperGLUE."""
        return None


class BigBenchExtractor(BenchmarkExtractor):
    """Extractor for BIG-Bench tasks."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        BIG-Bench format varies by task
        """
        try:
            # Try to extract using common patterns
            question = doc.get("input", doc.get("question", doc.get("text", "")))

            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question

            # Try different answer patterns
            correct_answer = doc.get("target", doc.get("answer", doc.get("output", "")))

            if not all([question, correct_answer]):
                return None

            return {
                "question": question,
                "formatted_question": formatted_question,
                "correct_answer": str(correct_answer),
            }

        except Exception as e:
            logger.debug(f"Error extracting BIG-Bench QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for BIG-Bench."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # For BIG-Bench, create a generic incorrect answer
            correct_answer = qa_pair["correct_answer"]
            incorrect_answer = "incorrect response"

            # If there are multiple choice options, try to find one
            if "choices" in doc:
                choices = doc["choices"]
                for choice in choices:
                    if str(choice) != correct_answer:
                        incorrect_answer = str(choice)
                        break

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting BIG-Bench contrastive pair: {e}")
            return None


class HumanEvalExtractor(BenchmarkExtractor):
    """Extractor for HumanEval code generation benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        HumanEval format:
        - doc['prompt']: the coding prompt
        - doc['canonical_solution']: the solution
        - doc['task_id']: task identifier
        """
        try:
            prompt = doc.get("prompt", "")
            solution = doc.get("canonical_solution", "")
            task_id = doc.get("task_id", "")

            if not prompt:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Task {task_id}:\n{prompt}"

            # The correct answer is the canonical solution
            if not solution:
                raise ValueError("HumanEval document missing canonical_solution")
            correct_answer = solution

            return {"question": prompt, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting HumanEval QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for HumanEval."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # For code generation, incorrect answer is syntactically valid but wrong code
            correct_answer = qa_pair["correct_answer"]
            # This should be handled by a proper syntactic corruption method
            raise ValueError("HumanEval contrastive pair generation should use proper syntactic corruption")

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting HumanEval contrastive pair: {e}")
            return None


class MBPPExtractor(BenchmarkExtractor):
    """Extractor for MBPP (Mostly Basic Python Problems) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MBPP format:
        - doc['text']: problem description
        - doc['code']: solution code
        - doc['test_list']: test cases
        """
        try:
            text = doc.get("text", doc.get("prompt", ""))
            code = doc.get("code", "")

            if not text:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Write a function to: {text}"

            # The correct answer is the code solution
            if not code:
                raise ValueError("MBPP document missing code field")
            correct_answer = code

            return {"question": text, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting MBPP QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MBPP."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                logging.debug(f"DEBUG MBPP: qa_pair is None for doc keys: {list(doc.keys())}")
                return None

            # For code generation, create incorrect answer by removing random words
            correct_answer = qa_pair["correct_answer"]
            incorrect_answer = self._create_incorrect_code(correct_answer)

            logging.debug("DEBUG MBPP: Successfully created contrastive pair")
            logging.debug(f"DEBUG MBPP: Question: {qa_pair['formatted_question'][:50]}...")
            logging.debug(f"DEBUG MBPP: Correct: {correct_answer[:50]}...")
            logging.debug(f"DEBUG MBPP: Incorrect: {incorrect_answer[:50]}...")

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logging.debug(f"DEBUG MBPP: Error extracting contrastive pair: {e}")
            logger.debug(f"Error extracting MBPP contrastive pair: {e}")
            return None

    def _create_incorrect_code(self, correct_code: str) -> str:
        """Create incorrect code by removing random words."""
        import random
        import re

        try:
            # Split code into tokens (words, operators, etc.)
            tokens = re.findall(r"\b\w+\b|[^\w\s]", correct_code)

            if len(tokens) < 3:
                # Too few tokens to corrupt
                raise ValueError("Code too short to create meaningful corruption")

            # Find words (not operators or punctuation) that we can remove
            word_indices = []
            for i, token in enumerate(tokens):
                if (
                    token.isalpha()
                    and token
                    not in [
                        "def",
                        "return",
                        "if",
                        "else",
                        "elif",
                        "for",
                        "while",
                        "try",
                        "except",
                        "class",
                        "import",
                        "from",
                        "pass",
                        "break",
                        "continue",
                    ]
                    and len(token) > 1
                ):
                    word_indices.append(i)

            if not word_indices:
                # No suitable words to remove
                raise ValueError("No suitable tokens to remove for corruption")

            # Remove 1-2 random words
            num_to_remove = min(2, len(word_indices))
            indices_to_remove = random.sample(word_indices, num_to_remove)

            # Create new token list with removed words
            new_tokens = []
            for i, token in enumerate(tokens):
                if i not in indices_to_remove:
                    new_tokens.append(token)

            # Reconstruct the code
            result = ""
            for i, token in enumerate(new_tokens):
                if i > 0 and new_tokens[i - 1].isalnum() and token.isalnum():
                    result += " "
                result += token

            return result

        except Exception as e:
            # No fallbacks - fail hard
            raise ValueError(f"Failed to create syntactically corrupted code: {e!s}")


class ANLIExtractor(BenchmarkExtractor):
    """Extractor for Adversarial NLI benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        ANLI format:
        - doc['premise']: the premise
        - doc['hypothesis']: the hypothesis
        - doc['label']: entailment (0), neutral (1), contradiction (2)
        """
        try:
            premise = doc.get("premise", "")
            hypothesis = doc.get("hypothesis", "")
            label = doc.get("label", -1)

            if not all([premise, hypothesis]) or label == -1:
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelationship?"

            # Map label to answer
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            correct_answer = label_map.get(label, "unknown")

            return {
                "question": f"Premise: {premise}\nHypothesis: {hypothesis}",
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting ANLI QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for ANLI."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            label = doc.get("label", -1)
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

            correct_answer = label_map.get(label, "unknown")

            # Get an incorrect answer
            incorrect_answer = None
            for label_key, answer in label_map.items():
                if label_key != label:
                    incorrect_answer = answer
                    break

            if not incorrect_answer:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting ANLI contrastive pair: {e}")
            return None


class MultilingualExtractor(BenchmarkExtractor):
    """Base extractor for multilingual benchmarks (XNLI, XCOPA, etc)."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Generic multilingual format - tries common patterns
        """
        try:
            # Try various question patterns
            question = doc.get("question", doc.get("premise", doc.get("sentence", doc.get("text", ""))))

            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question

            # Try to find the answer
            correct_answer = None

            # Pattern 1: label index with choices
            if "label" in doc and "choices" in doc:
                label = doc["label"]
                choices = doc["choices"]
                if label < len(choices):
                    correct_answer = choices[label]

            # Pattern 2: answer field
            if not correct_answer:
                correct_answer = doc.get("answer", doc.get("target", ""))

            # Pattern 3: choice1/choice2 with label (XCOPA style)
            if not correct_answer and "choice1" in doc and "choice2" in doc:
                label = doc.get("label", 0)
                correct_answer = doc["choice1"] if label == 0 else doc["choice2"]

            if not all([question, correct_answer]):
                return None

            return {
                "question": question,
                "formatted_question": formatted_question,
                "correct_answer": str(correct_answer),
            }

        except Exception as e:
            logger.debug(f"Error extracting multilingual QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for multilingual tasks."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]
            incorrect_answer = None

            # Try to find an incorrect choice
            if "choices" in doc:
                for choice in doc["choices"]:
                    if str(choice) != correct_answer:
                        incorrect_answer = str(choice)
                        break

            # XCOPA style
            elif "choice1" in doc and "choice2" in doc:
                label = doc.get("label", 0)
                incorrect_answer = doc["choice2"] if label == 0 else doc["choice1"]

            if not incorrect_answer:
                incorrect_answer = "incorrect answer"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting multilingual contrastive pair: {e}")
            return None


class ArithmeticExtractor(BenchmarkExtractor):
    """Extractor for arithmetic tasks."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Arithmetic format:
        - doc['context']: arithmetic problem with "Question: ... Answer:" format
        - doc['completion']: numerical answer
        Or legacy format:
        - doc['question']: arithmetic problem
        - doc['answer']: numerical answer
        """
        try:
            # Try new format first (context/completion)
            if "context" in doc and "completion" in doc:
                context = doc["context"]
                answer = doc["completion"].strip()

                # Extract question from context
                # Context format: "Question: What is (9 + 8) * 2?\nAnswer:"
                if "Question:" in context:
                    question = context.split("Question:")[1].split("\nAnswer:")[0].strip()
                else:
                    question = context.replace("\nAnswer:", "").strip()
            else:
                # Try legacy format
                question = doc.get("question", doc.get("problem", ""))
                answer = doc.get("answer", doc.get("solution", ""))

            if not all([question, answer]):
                return None

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question

            return {"question": question, "formatted_question": formatted_question, "correct_answer": str(answer)}

        except Exception as e:
            logger.debug(f"Error extracting arithmetic QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for arithmetic."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]

            # Create an incorrect answer by modifying the correct one
            try:
                num = float(correct_answer)
                incorrect_answer = str(num + 1)
            except ValueError:
                incorrect_answer = "0"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting arithmetic contrastive pair: {e}")
            return None


class DefaultExtractor(BenchmarkExtractor):
    """Default extractor that tries common patterns."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Try common extraction patterns."""
        try:
            # Try to get question
            question = doc.get("question", doc.get("ctx", doc.get("goal", doc.get("premise", str(doc)))))

            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question

            # Try various patterns for correct answer
            correct_answer = None

            # Pattern 1: mc1_targets (TruthfulQA style)
            mc1_targets = doc.get("mc1_targets", {})
            if mc1_targets:
                choices = mc1_targets.get("choices", [])
                labels = mc1_targets.get("labels", [])
                for i, label in enumerate(labels):
                    if label == 1 and i < len(choices):
                        correct_answer = choices[i]
                        break

            # Pattern 2: choices + answerKey (ARC style)
            if not correct_answer:
                choices = doc.get("choices", {})
                if choices:
                    choice_texts = choices.get("text", [])
                    choice_labels = choices.get("label", [])
                    answer_key = doc.get("answerKey", "")

                    for i, label in enumerate(choice_labels):
                        if label == answer_key and i < len(choice_texts):
                            correct_answer = choice_texts[i]
                            break

            # Pattern 3: endings + label (HellaSwag style)
            if not correct_answer:
                endings = doc.get("endings", [])
                label = doc.get("label", 0)
                if endings and label < len(endings):
                    correct_answer = endings[label]

            # Pattern 4: choices list + answer index (MMLU style)
            if not correct_answer:
                choices = doc.get("choices", [])
                answer = doc.get("answer", 0)
                if choices and answer < len(choices):
                    correct_answer = choices[answer]

            # Pattern 5: sol1/sol2 + label (PIQA style)
            if not correct_answer:
                sol1 = doc.get("sol1", "")
                sol2 = doc.get("sol2", "")
                label = doc.get("label", 0)
                if sol1 and sol2:
                    correct_answer = sol1 if label == 0 else sol2

            # Pattern 6: choice1/choice2 + label (COPA style)
            if not correct_answer:
                choice1 = doc.get("choice1", "")
                choice2 = doc.get("choice2", "")
                label = doc.get("label", 0)
                if choice1 and choice2:
                    correct_answer = choice1 if label == 0 else choice2

            # Pattern 7: option1/option2 + answer (Winogrande style)
            if not correct_answer:
                option1 = doc.get("option1", "")
                option2 = doc.get("option2", "")
                answer = doc.get("answer", "")
                if option1 and option2:
                    correct_answer = option1 if answer == "1" else option2

            # Pattern 8: text answer (GSM8K, math problems)
            if not correct_answer:
                answer = doc.get("answer", "")
                if answer:
                    correct_answer = answer

            # Pattern 9: boolean answer (BoolQ style)
            if not correct_answer:
                answer = doc.get("answer")
                if answer is not None:
                    correct_answer = "True" if answer else "False"

            # Pattern 10: text field (WikiText style)
            if not correct_answer:
                text = doc.get("text", "")
                if text:
                    correct_answer = text

            # Pattern 11: target field
            if not correct_answer:
                target = doc.get("target", "")
                if target:
                    correct_answer = target

            if not correct_answer:
                return None

            return {"question": question, "formatted_question": formatted_question, "correct_answer": correct_answer}

        except Exception as e:
            logger.debug(f"Error extracting default QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair using default patterns."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # Try to find an incorrect choice based on the document structure
            incorrect_choice = None

            # For multiple choice formats, find an incorrect option
            mc1_targets = doc.get("mc1_targets", {})
            if mc1_targets:
                choices = mc1_targets.get("choices", [])
                labels = mc1_targets.get("labels", [])
                for i, label in enumerate(labels):
                    if label == 0 and i < len(choices):
                        incorrect_choice = choices[i]
                        break

            # For other formats, create a generic incorrect answer
            if not incorrect_choice:
                incorrect_choice = "Incorrect or irrelevant response"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": qa_pair["correct_answer"],
                "incorrect_answer": incorrect_choice,
            }

        except Exception as e:
            logger.debug(f"Error extracting default contrastive pair: {e}")
            return None


class LiveCodeBenchExtractor(BenchmarkExtractor):
    """Extractor for LiveCodeBench coding benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        LiveCodeBench format:
        - doc['question_title']: problem title
        - doc['question_content']: problem description
        - doc['starter_code']: starter code template
        - doc['difficulty']: EASY/MEDIUM/HARD
        - doc['platform']: LEETCODE/ATCODER/CODEFORCES
        - doc['public_test_cases']: test cases
        """
        try:
            title = doc.get("question_title", "")
            content = doc.get("question_content", "")
            starter_code = doc.get("starter_code", "")
            difficulty = doc.get("difficulty", "")
            platform = doc.get("platform", "")

            if not content:
                return None

            # Create comprehensive problem statement
            problem_statement = f"Problem: {title}\n\n{content}"
            if starter_code:
                problem_statement += f"\n\nStarter Code:\n{starter_code}"

            # Format the question
            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"[{platform}] [{difficulty}] {problem_statement}"

            # For LiveCodeBench, starter_code is required
            if not starter_code:
                raise ValueError("LiveCodeBench document missing starter_code")
            correct_answer = starter_code

            return {
                "question": problem_statement,
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting LiveCodeBench QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for LiveCodeBench."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            # For code generation, create incorrect answer by removing random words
            correct_answer = qa_pair["correct_answer"]
            incorrect_answer = self._create_incorrect_code(correct_answer)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting LiveCodeBench contrastive pair: {e}")
            return None

    def _create_incorrect_code(self, correct_code: str) -> str:
        """Create incorrect code by introducing syntax errors."""
        try:
            # Tokenize the code
            tokens = re.findall(r"\b\w+\b|[^\w\s]", correct_code)

            if len(tokens) < 3:
                raise ValueError("Code too short to create syntactic corruption")

            # Don't remove critical keywords
            removable_tokens = [
                i
                for i, token in enumerate(tokens)
                if token not in ["def", "class", "if", "else", "for", "while", "return", "import", "from"]
                and token.strip()
            ]

            if not removable_tokens:
                raise ValueError("No removable tokens found for syntactic corruption")

            # Remove 1-2 random tokens
            import random

            num_to_remove = min(2, len(removable_tokens))
            indices_to_remove = sorted(random.sample(removable_tokens, num_to_remove), reverse=True)

            for idx in indices_to_remove:
                tokens.pop(idx)

            return "".join(tokens)

        except Exception as e:
            raise ValueError(f"Failed to create syntactic corruption: {e!s}")


# Registry of extractors
class GPQAExtractor(BenchmarkExtractor):
    """Extractor for GPQA (Graduate-Level Google-Proof Q&A) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract question-answer pair from GPQA document."""
        try:
            # Handle both raw format (from Idavidrein/gpqa dataset) and processed format (from lm-eval)

            # Raw format: Question, Correct Answer, Incorrect Answer 1-3
            if all(key in doc for key in ["Question", "Correct Answer", "Incorrect Answer 1"]):
                question = doc["Question"]
                correct_answer = doc["Correct Answer"]
                incorrect_answers = [
                    doc.get("Incorrect Answer 1", ""),
                    doc.get("Incorrect Answer 2", ""),
                    doc.get("Incorrect Answer 3", ""),
                ]
                # Filter out empty answers
                incorrect_answers = [ans for ans in incorrect_answers if ans.strip()]

                # Create choices list with correct answer first
                choices = [correct_answer] + incorrect_answers

                # Format as multiple choice question
                choice_letters = ["A", "B", "C", "D"]
                formatted_choices = []
                for i, choice in enumerate(choices[:4]):
                    if choice:
                        formatted_choices.append(f"({choice_letters[i]}) {choice}")

                formatted_question = f"{question}\n\nChoices:\n" + "\n".join(formatted_choices)

                return {
                    "question": question,
                    "formatted_question": formatted_question,
                    "correct_answer": correct_answer,
                    "answer_letter": "A",  # Correct answer is always first in raw format
                    "choices": choices,
                }

            # Processed format: Question, choice1-4, answer (letter format like "(B)")
            if all(key in doc for key in ["Question", "choice1", "choice2", "choice3", "choice4", "answer"]):
                question = doc["Question"]
                choices = [doc["choice1"], doc["choice2"], doc["choice3"], doc["choice4"]]
                answer = doc["answer"]

                # Extract letter from answer format like "(B)" or "B"
                answer_match = re.search(r"[ABCD]", answer.upper())
                if not answer_match:
                    return None

                answer_letter = answer_match.group()
                answer_index = ord(answer_letter) - ord("A")

                if answer_index >= len(choices):
                    return None

                correct_answer = choices[answer_index]

                # Format as multiple choice question
                choice_letters = ["A", "B", "C", "D"]
                formatted_choices = []
                for i, choice in enumerate(choices):
                    formatted_choices.append(f"({choice_letters[i]}) {choice}")

                formatted_question = f"{question}\n\nChoices:\n" + "\n".join(formatted_choices)

                return {
                    "question": question,
                    "formatted_question": formatted_question,
                    "correct_answer": correct_answer,
                    "answer_letter": answer_letter,
                    "choices": choices,
                }

            return None

        except Exception as e:
            logger.debug(f"Error extracting GPQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for GPQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_answer = qa_pair["correct_answer"]
            choices = qa_pair["choices"]

            # Find incorrect answers (all choices except the correct one)
            incorrect_answers = [choice for choice in choices if choice != correct_answer]

            if not incorrect_answers:
                return None

            # Randomly select one incorrect answer
            import random

            incorrect_answer = random.choice(incorrect_answers)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting GPQA contrastive pair: {e}")
            return None


class HLEExtractor(BenchmarkExtractor):
    """Extractor for HLE (Human-Level Evaluation) benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        HLE format:
        - doc['question']: the question (includes choices for multipleChoice)
        - doc['answer']: correct answer (letter for multipleChoice, text for exactMatch)
        - doc['answer_type']: 'multipleChoice' or 'exactMatch'
        """
        try:
            question = doc.get("question", "")
            answer = doc.get("answer", "")
            answer_type = doc.get("answer_type", "")

            if not all([question, answer, answer_type]):
                return None

            if answer_type == "multipleChoice":
                # For multiple choice, extract the correct answer text from choices
                correct_answer_text = self._extract_choice_text(question, answer)
                if not correct_answer_text:
                    correct_answer_text = answer  # Fallback to letter
            else:
                # For exact match, the answer is the text itself
                correct_answer_text = answer

            return {
                "question": question,
                "formatted_question": question,  # Question already formatted with choices
                "correct_answer": correct_answer_text,
                "answer_letter": answer if answer_type == "multipleChoice" else None,
                "answer_type": answer_type,
            }

        except Exception as e:
            logger.debug(f"Error extracting HLE QA pair: {e}")
            return None

    def _extract_choice_text(self, question: str, answer_letter: str) -> Optional[str]:
        """Extract the text of the correct choice from the question."""
        import re

        # Look for patterns like "A. text" or "A) text"
        patterns = [
            rf"{answer_letter}\.\s+(.+?)(?=\n[A-E]\.|$)",  # "A. option" format
            rf"{answer_letter}\)\s+(.+?)(?=\n[A-E]\)|$)",  # "A) option" format
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.MULTILINE | re.DOTALL)
            if match:
                return match.group(1).strip()

        return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for HLE."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            answer_type = doc.get("answer_type", "")

            if answer_type == "multipleChoice":
                # For multiple choice, get an incorrect choice
                incorrect_answer = self._get_incorrect_choice(doc.get("question", ""), doc.get("answer", ""))
                if not incorrect_answer:
                    return None
            else:
                # For exact match, replace one third of words with 'X'
                correct_answer = qa_pair["correct_answer"]
                words = correct_answer.split()
                if len(words) > 1:
                    # Replace approximately 1/3 of the words with 'X'
                    num_to_replace = max(1, len(words) // 3)
                    import random

                    indices_to_replace = random.sample(range(len(words)), num_to_replace)
                    for idx in indices_to_replace:
                        words[idx] = "X"
                    incorrect_answer = " ".join(words)
                else:
                    # For single word answers, change part of it
                    if len(correct_answer) > 3:
                        mid = len(correct_answer) // 2
                        incorrect_answer = correct_answer[:mid] + "X" * (len(correct_answer) - mid)
                    else:
                        incorrect_answer = "X" * len(correct_answer)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": qa_pair["correct_answer"],
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting HLE contrastive pair: {e}")
            return None

    def _get_incorrect_choice(self, question: str, correct_letter: str) -> Optional[str]:
        """Get text of an incorrect choice for multiple choice questions."""
        import re

        # Find all choices
        choices = []
        patterns = [
            r"([A-E])\.\s+(.+?)(?=\n[A-E]\.|$)",  # "A. option" format
            r"([A-E])\)\s+(.+?)(?=\n[A-E]\)|$)",  # "A) option" format
        ]

        for pattern in patterns:
            matches = re.findall(pattern, question, re.MULTILINE | re.DOTALL)
            if matches:
                choices = matches
                break

        # Find an incorrect choice (not the correct one)
        for letter, text in choices:
            if letter != correct_letter:
                return text.strip()

        return None


class SuperGPQAExtractor(BenchmarkExtractor):
    """Extractor for SuperGPQA scientific reasoning benchmark."""

    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        SuperGPQA format:
        - doc['question']: The question text
        - doc['options']: List of answer options
        - doc['answer']: The correct answer text
        - doc['answer_letter']: The correct answer letter (A, B, C, D, E)
        """
        try:
            question = doc.get("question", "")
            options = doc.get("options", [])
            correct_answer = doc.get("answer", "")
            answer_letter = doc.get("answer_letter", "")

            if not question or not options or not correct_answer:
                return None

            # Format the question with multiple choice options
            formatted_options = []
            for i, option in enumerate(options):
                letter = chr(ord("A") + i)
                formatted_options.append(f"{letter}. {option}")

            formatted_question = f"{question}\n\n" + "\n".join(formatted_options)

            return {
                "question": question,
                "formatted_question": formatted_question,
                "correct_answer": correct_answer,
                "answer_letter": answer_letter,
            }

        except Exception as e:
            logger.debug(f"Error extracting SuperGPQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for SuperGPQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            options = doc.get("options", [])
            correct_answer = qa_pair["correct_answer"]

            # Find an incorrect answer from options
            incorrect_answer = self._get_incorrect_option(options, correct_answer)
            if not incorrect_answer:
                return None

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
            }

        except Exception as e:
            logger.debug(f"Error extracting SuperGPQA contrastive pair: {e}")
            return None

    def _get_incorrect_option(self, options: List[str], correct_answer: str) -> Optional[str]:
        """Get an incorrect option from the list."""
        # Find options that are not the correct answer
        for option in options:
            if option != correct_answer:
                return option

        # Fallback if something goes wrong
        return None


EXTRACTORS = {
    "winogrande": WinograndeExtractor,
    "arc_challenge": ARCExtractor,
    "arc_easy": ARCExtractor,
    "asdiv": ASDivExtractor,
    "hellaswag": HellaSwagExtractor,
    "truthfulqa_mc1": TruthfulQAExtractor,
    "truthfulqa_mc2": TruthfulQAExtractor,
    "truthfulqa_gen": TruthfulQAExtractor,
    "boolq": BoolQExtractor,
    "gsm8k": GSM8KExtractor,
    # MATH-500 mathematical reasoning benchmarks (reuse GSM8KExtractor)
    "math": GSM8KExtractor,
    "math500": GSM8KExtractor,
    "hendrycks_math": GSM8KExtractor,  # Already exists, but documenting here
    # AIME contest math problems (general + year-specific)
    "aime": GSM8KExtractor,
    "aime2025": GSM8KExtractor,
    "aime2024": GSM8KExtractor,  # Backward compatibility
    # HMMT contest math problems (general + competition-specific)
    "hmmt": GSM8KExtractor,
    "hmmt_feb_2025": GSM8KExtractor,
    # PolyMath multilingual mathematical reasoning (Chinese and English, medium difficulty)
    "polymath": GSM8KExtractor,
    "polymath_en_medium": GSM8KExtractor,
    "polymath_zh_medium": GSM8KExtractor,
    "polymath_en_high": GSM8KExtractor,
    "polymath_zh_high": GSM8KExtractor,
    # LiveMathBench CNMO 2024 (Chinese and English)
    "livemathbench": GSM8KExtractor,
    "livemathbench_cnmo_en": GSM8KExtractor,
    "livemathbench_cnmo_zh": GSM8KExtractor,
    "mmlu": MMLUExtractor,
    "mmmlu": MMLUExtractor,
    "m_mmlu_en": MMLUExtractor,  # Support the actual task name used by lm-eval
    "piqa": PIQAExtractor,
    "copa": COPAExtractor,
    "openbookqa": OpenBookQAExtractor,
    "squad2": SQuAD2Extractor,
    "squadv2": SQuAD2Extractor,  # lm-eval uses squadv2 as the task name
    "race": RACEExtractor,
    "wikitext": WikiTextExtractor,
    "mrpc": MRPCExtractor,  # GLUE MRPC paraphrase detection
    "qnli": QNLIExtractor,  # GLUE QNLI question-answering NLI
    "qqp": QQPExtractor,  # GLUE QQP question pairs
    "rte": RTEExtractor,  # GLUE RTE textual entailment
    "sst2": SST2Extractor,  # GLUE SST2 sentiment analysis
    "wnli": WNLIExtractor,  # GLUE WNLI Winograd NLI
    # Add more specific extractors for other benchmarks
    "cb": COPAExtractor,  # Similar format to COPA
    "coqa": CoQAExtractor,  # Conversational QA format
    "drop": SQuAD2Extractor,  # Similar QA format
    "logiqa": LogiQAExtractor,  # LogiQA logical reasoning format
    "logiqa2": LogiQAExtractor,  # LogiQA 2.0 uses same format
    "agieval_logiqa_en": LogiQAExtractor,  # AGIEval LogiQA English
    "agieval_logiqa_zh": LogiQAExtractor,  # AGIEval LogiQA Chinese
    "math_qa": MMLUExtractor,  # Multiple choice format
    "mathqa": MathQAExtractor,  # Custom extractor for mathqa's unique format
    "mc_taco": MCTacoExtractor,  # MC-TACO temporal reasoning
    "multirc": MMLUExtractor,  # Multiple choice format
    "mutual": MMLUExtractor,  # Multiple choice format
    "naturalqs": SQuAD2Extractor,  # QA format
    "prost": MMLUExtractor,  # Multiple choice format
    "pubmedqa": MMLUExtractor,  # Multiple choice format
    "quac": QuACExtractor,  # Conversational QA format
    "record": SQuAD2Extractor,  # QA format
    "sciq": MMLUExtractor,  # Multiple choice format
    "swag": HellaSwagExtractor,  # Similar format to HellaSwag
    "toxigen": BoolQExtractor,  # Binary classification
    "triviaqa": SQuAD2Extractor,  # QA format
    "webqs": SQuAD2Extractor,  # QA format
    "wic": BoolQExtractor,  # Binary classification
    "wsc": COPAExtractor,  # Similar format to COPA
    "wsc273": COPAExtractor,  # Similar format to COPA
    # New extractors for previously unsupported benchmarks
    "lambada_cloze": LambadaExtractor,  # LAMBADA cloze task
    "lambada_multilingual": LambadaExtractor,  # LAMBADA multilingual
    "lambada_standard_cloze_yaml": LambadaExtractor,  # LAMBADA standard cloze variant
    "lambada": LambadaExtractor,  # Generic LAMBADA
    "ai2_arc": AI2ARCExtractor,  # AI2 ARC (delegates to ARCExtractor)
    "glue": GLUEExtractor,  # GLUE suite (should use specific tasks)
    "superglue": SuperGLUEExtractor,  # SuperGLUE suite (should use specific tasks)
    "big_bench": BigBenchExtractor,  # BIG-Bench tasks
    "humaneval": HumanEvalExtractor,  # Code generation
    "mbpp": MBPPExtractor,  # Python problems
    "livecodebench": LiveCodeBenchExtractor,  # LiveCodeBench coding problems
    "anli": ANLIExtractor,  # Adversarial NLI
    "arithmetic": ArithmeticExtractor,  # Arithmetic tasks
    "belebele": MultilingualExtractor,  # Multilingual reading comprehension
    "blimp": DefaultExtractor,  # Linguistic minimal pairs
    "crows_pairs": DefaultExtractor,  # Bias benchmark
    "headqa": MMLUExtractor,  # Spanish healthcare QA (multiple choice)
    "hendrycks_ethics": MMLUExtractor,  # Ethics benchmark (multiple choice)
    "hendrycks_math": GSM8KExtractor,  # Math problems
    "medqa": MMLUExtractor,  # Medical QA (multiple choice)
    "mgsm": GSM8KExtractor,  # Multilingual GSM8K
    "paws_x": DefaultExtractor,  # Paraphrase detection
    "qa4mre": MMLUExtractor,  # QA for machine reading evaluation
    "qasper": SQuAD2Extractor,  # QA on scientific papers
    "social_i_qa": MMLUExtractor,  # Social commonsense QA
    "unscramble": DefaultExtractor,  # Word unscrambling
    "xcopa": MultilingualExtractor,  # Multilingual COPA
    "xnli": MultilingualExtractor,  # Cross-lingual NLI
    "xstorycloze": MultilingualExtractor,  # Multilingual story cloze
    "xwinograd": MultilingualExtractor,  # Multilingual Winograd
    # GPQA (Graduate-Level Google-Proof Q&A) benchmarks
    "gpqa": GPQAExtractor,
    "gpqa_diamond": GPQAExtractor,  # Maps to gpqa_diamond_zeroshot
    "gpqa_extended": GPQAExtractor,  # Maps to gpqa_extended_zeroshot
    "gpqa_main_zeroshot": GPQAExtractor,
    "gpqa_main_n_shot": GPQAExtractor,
    "gpqa_main_cot_zeroshot": GPQAExtractor,
    "gpqa_main_cot_n_shot": GPQAExtractor,
    "gpqa_main_generative_n_shot": GPQAExtractor,
    "gpqa_diamond_zeroshot": GPQAExtractor,
    "gpqa_diamond_n_shot": GPQAExtractor,
    "gpqa_diamond_cot_zeroshot": GPQAExtractor,
    "gpqa_diamond_cot_n_shot": GPQAExtractor,
    "gpqa_diamond_generative_n_shot": GPQAExtractor,
    "gpqa_extended_zeroshot": GPQAExtractor,
    "gpqa_extended_n_shot": GPQAExtractor,
    "gpqa_extended_cot_zeroshot": GPQAExtractor,
    "gpqa_extended_cot_n_shot": GPQAExtractor,
    "gpqa_extended_generative_n_shot": GPQAExtractor,
    "leaderboard_gpqa": GPQAExtractor,
    "leaderboard_gpqa_main": GPQAExtractor,
    "leaderboard_gpqa_diamond": GPQAExtractor,
    "leaderboard_gpqa_extended": GPQAExtractor,
    # HLE (Human-Level Evaluation) benchmarks
    "hle": HLEExtractor,
    "hle_exact_match": HLEExtractor,
    "hle_multiple_choice": HLEExtractor,
    # SuperGPQA scientific reasoning benchmarks
    "supergpqa": SuperGPQAExtractor,
    "supergpqa_physics": SuperGPQAExtractor,
    "supergpqa_chemistry": SuperGPQAExtractor,
    "supergpqa_biology": SuperGPQAExtractor,
}

# Add BigCode benchmarks manually if import failed
# This allows them to work even if bigcode_extractors can't be imported
if not BIGCODE_AVAILABLE:
    # No need to import BenchmarkExtractor - it's already defined above

    class MBPPPlusExtractor(BenchmarkExtractor):
        """Temporary extractor for MBPP Plus."""

        def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            try:
                # MBPP Plus has 'code' field like MBPP
                problem = doc.get("text", doc.get("description", ""))
                code = doc.get("code", "")

                if not problem or not code:
                    return None

                return {
                    "question": problem,
                    "formatted_question": f"You are an expert Python programmer, and here is your task: {problem}",
                    "correct_answer": code,
                }
            except Exception as e:
                logger.debug(f"Error extracting MBPP Plus QA pair: {e}")
                return None

        def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_code = qa_pair["correct_answer"]
            # Create incorrect version by removing tokens
            incorrect_code = self._create_incorrect_code(correct_code)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_code,
                "incorrect_answer": incorrect_code,
            }

        def _create_incorrect_code(self, code: str) -> str:
            """Create incorrect version by removing tokens."""
            if not code:
                return "pass"

            # First try to break return statements
            if "return" in code:
                return code.replace("return", "retur", 1)

            # Try to break function calls
            if "(" in code and ")" in code:
                return code.replace(")", "", 1)

            # Try to break operators
            for op in ["==", ">=", "<=", "!=", "+=", "-="]:
                if op in code:
                    return code.replace(op, op[0], 1)

            # Simple corruption - remove tokens
            tokens = re.findall(r"\b\w+\b|[^\w\s]", code)
            if len(tokens) < 3:
                # Too short - just add syntax error
                return code + "{"

            # Remove some tokens
            # Remove spaces between tokens to create syntax errors
            result = "".join(tokens)

            # Ensure it's different
            if result == code:
                result = code[:-1] if len(code) > 1 else code + ";"

            return result

    class APPSExtractor(BenchmarkExtractor):
        """Extractor for APPS benchmark."""

        def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            try:
                question = doc.get("question", "")
                solutions = doc.get("solutions", "[]")

                if isinstance(solutions, str):
                    import json

                    try:
                        solutions = json.loads(solutions)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        solutions = []

                correct_answer = solutions[0] if solutions else "# Write your solution here"

                return {"question": question, "formatted_question": question, "correct_answer": correct_answer}
            except Exception as e:
                logger.debug(f"Error extracting APPS QA pair: {e}")
                return None

        def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_code = qa_pair["correct_answer"]
            incorrect_code = self._create_incorrect_code(correct_code)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_code,
                "incorrect_answer": incorrect_code,
            }

        def _create_incorrect_code(self, code: str) -> str:
            if not code or code == "# Write your solution here":
                return "pass"
            tokens = re.findall(r"\b\w+\b|[^\w\s]", code)
            if len(tokens) < 3:
                return code.replace("return", "return None #", 1)
            # Remove some tokens
            if len(tokens) > 5:
                tokens.pop(len(tokens) // 2)
            return "".join(tokens)

    class DS1000Extractor(BenchmarkExtractor):
        """Extractor for DS-1000 benchmark."""

        def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            try:
                problem = doc.get("problem", "")
                prompt = doc.get("prompt", "")
                solution = doc.get("reference_code", doc.get("solution", ""))

                question = f"{problem}\n{prompt}" if problem and prompt else (problem or prompt)

                return {
                    "question": question,
                    "formatted_question": question,
                    "correct_answer": solution or "# Your solution here",
                }
            except Exception as e:
                logger.debug(f"Error extracting DS-1000 QA pair: {e}")
                return None

        def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_code = qa_pair["correct_answer"]
            # Simple corruption for DS-1000
            if "mean()" in correct_code:
                incorrect_code = correct_code.replace("mean()", "sum()")
            elif "sum()" in correct_code:
                incorrect_code = correct_code.replace("sum()", "mean()")
            else:
                incorrect_code = correct_code.replace("return", "return None #", 1)

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_code,
                "incorrect_answer": incorrect_code,
            }

    class MultiPLEExtractor(BenchmarkExtractor):
        """Extractor for MultiPL-E benchmarks."""

        def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            prompt = doc.get("prompt", "")
            if not prompt:
                return None
            return {"question": prompt, "formatted_question": prompt, "correct_answer": "pass"}

        def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            prompt = qa_pair["question"]
            # Language-specific stubs
            if "function" in prompt and "{" in prompt:
                correct_answer = "return null;"
                incorrect_answer = "returnnull"
            elif "fn " in prompt:
                correct_answer = "unimplemented!()"
                incorrect_answer = "unimplemented!"
            else:
                correct_answer = "pass"
                incorrect_answer = "pas"

            return {"question": prompt, "correct_answer": correct_answer, "incorrect_answer": incorrect_answer}

    class ConalaExtractor(BenchmarkExtractor):
        """Extractor for CoNaLa benchmark."""

        def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            intent = doc.get("intent", "")
            snippet = doc.get("snippet", "")
            if not intent or not snippet:
                return None
            return {
                "question": intent,
                "formatted_question": f"Write Python code to: {intent}",
                "correct_answer": snippet,
            }

        def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_code = qa_pair["correct_answer"]
            # More aggressive corruption for short snippets
            if "(" in correct_code and ")" in correct_code:
                # Remove closing parenthesis
                incorrect_code = correct_code.replace(")", "", 1)
            elif "." in correct_code:
                # Remove first dot
                incorrect_code = correct_code.replace(".", "", 1)
            elif "[" in correct_code and "]" in correct_code:
                # Remove closing bracket
                incorrect_code = correct_code.replace("]", "", 1)
            elif len(correct_code) > 3:
                # Remove last 3 characters
                incorrect_code = correct_code[:-3]
            else:
                # For very short code, duplicate and break it
                incorrect_code = correct_code + correct_code

            # Ensure they're different
            if incorrect_code == correct_code:
                incorrect_code = correct_code + "_broken"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_code,
                "incorrect_answer": incorrect_code,
            }

    class ConcodeExtractor(BenchmarkExtractor):
        """Extractor for Concode benchmark."""

        def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            nl = doc.get("nl", "")
            code = doc.get("code", "")
            if not nl or not code:
                return None
            return {"question": nl, "formatted_question": f"Implement the following: {nl}", "correct_answer": code}

        def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None

            correct_code = qa_pair["correct_answer"]
            # Java-style corruption
            if ";" in correct_code:
                incorrect_code = correct_code.replace(";", "", 1)  # Remove first semicolon
            elif "{" in correct_code and "}" in correct_code:
                incorrect_code = correct_code.replace("}", "", 1)  # Remove closing brace
            elif "return" in correct_code:
                incorrect_code = correct_code.replace("return", "retur", 1)
            else:
                # Remove last character
                incorrect_code = correct_code[:-1] if len(correct_code) > 1 else correct_code + "!"

            # Ensure they're different
            if incorrect_code == correct_code:
                incorrect_code = correct_code.replace(" ", "", 1) if " " in correct_code else correct_code + ";"

            return {
                "question": qa_pair["formatted_question"],
                "correct_answer": correct_code,
                "incorrect_answer": incorrect_code,
            }

    # Add to EXTRACTORS
    EXTRACTORS["mbpp_plus"] = MBPPPlusExtractor
    EXTRACTORS["apps"] = APPSExtractor
    EXTRACTORS["ds1000"] = DS1000Extractor
    EXTRACTORS["multiple_js"] = MultiPLEExtractor
    EXTRACTORS["multiple_java"] = MultiPLEExtractor
    EXTRACTORS["multiple_rs"] = MultiPLEExtractor
    EXTRACTORS["multiple_go"] = MultiPLEExtractor
    # ConalaExtractor and ConcodeExtractor are in bigcode_extractors
    # They will be handled by get_extractor's lazy import


def get_extractor(benchmark_name: str) -> BenchmarkExtractor:
    """Get the appropriate extractor for a benchmark with hard error for unsupported tasks."""
    # Try to import BigCode extractors lazily
    try:
        from .bigcode_extractors import BIGCODE_EXTRACTORS, get_bigcode_extractor

        if benchmark_name in BIGCODE_EXTRACTORS:
            return get_bigcode_extractor(benchmark_name)
    except ImportError:
        pass

    # Try exact match first
    if benchmark_name in EXTRACTORS:
        return EXTRACTORS[benchmark_name]()

    # Handle MMLU subtasks (e.g., mmlu_abstract_algebra, mmlu_biology, etc.)
    if benchmark_name.startswith("mmlu_"):
        return MMLUExtractor()

    # Hard error for unsupported benchmarks
    all_supported = sorted(list(EXTRACTORS.keys()))
    # Try to add BigCode extractors to the list if available
    try:
        from .bigcode_extractors import BIGCODE_EXTRACTORS

        all_supported.extend(list(BIGCODE_EXTRACTORS.keys()))
        all_supported = sorted(all_supported)
    except ImportError:
        pass

    raise UnsupportedBenchmarkError(
        f"No extractor found for benchmark '{benchmark_name}'. Supported benchmarks: {all_supported}"
    )


def extract_qa_pair(benchmark_name: str, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
    """
    Extract a QA pair from a benchmark document.

    Args:
        benchmark_name: Name of the benchmark
        doc: Document from the benchmark
        task_data: Task data object

    Returns:
        Dict with question and answer or None
    """
    extractor = get_extractor(benchmark_name)
    return extractor.extract_qa_pair(doc, task_data)


def extract_contrastive_pair(
    benchmark_name: str, doc: Dict[str, Any], task_data: Any = None
) -> Optional[Dict[str, str]]:
    """
    Extract a contrastive pair from a benchmark document.

    Args:
        benchmark_name: Name of the benchmark
        doc: Document from the benchmark
        task_data: Task data object

    Returns:
        Dict with question and choices or None
    """
    extractor = get_extractor(benchmark_name)
    return extractor.extract_contrastive_pair(doc, task_data)
