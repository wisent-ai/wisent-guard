"""
Benchmark-specific data extraction logic for multiple choice questions and answers.

Each benchmark has its own data structure and format. This module centralizes
the extraction logic to cleanly handle the differences between benchmarks.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class UnsupportedBenchmarkError(Exception):
    """Raised when benchmark has no extractor."""
    pass


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
            sentence = doc.get('sentence', '')
            option1 = doc.get('option1', '')
            option2 = doc.get('option2', '')
            answer = doc.get('answer', '')
            
            if not all([sentence, option1, option2, answer]):
                return None
                
            # Create the question
            question = f"Complete the sentence: {sentence}"
            
            # 🚨 FIX: Don't use doc_to_text for winogrande - it returns integers instead of strings!
            # This is a bug in lm-eval-harness winogrande task configuration
            formatted_question = f"{question}\nA. {option1}\nB. {option2}"
                
            # Get correct answer
            correct_answer = option1 if answer == '1' else option2
            
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            option1 = doc.get('option1', '')
            option2 = doc.get('option2', '')
            answer = doc.get('answer', '')
            
            correct_choice = option1 if answer == '1' else option2
            incorrect_choice = option2 if answer == '1' else option1
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question = doc.get('question', '')
            choices = doc.get('choices', {})
            choice_texts = choices.get('text', [])
            choice_labels = choices.get('label', [])
            answer_key = doc.get('answerKey', '')
            
            if not all([question, choice_texts, choice_labels, answer_key]):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
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
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting ARC QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for ARC."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            choices = doc.get('choices', {})
            choice_texts = choices.get('text', [])
            choice_labels = choices.get('label', [])
            answer_key = doc.get('answerKey', '')
            
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
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            ctx = doc.get('ctx', '')
            endings = doc.get('endings', [])
            label = doc.get('label', 0)
            
            if not all([ctx, endings]) or label >= len(endings):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Context: {ctx}\nChoose the best ending:"
                for i, ending in enumerate(endings):
                    formatted_question += f"\n{chr(65+i)}. {ending}"
                    
            # Get correct answer
            correct_answer = endings[label]
            
            return {
                'question': f"Context: {ctx}\nChoose the best ending:",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            endings = doc.get('endings', [])
            label = doc.get('label', 0)
            
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
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question = doc.get('question', '')
            if not question:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                
            # Extract correct answer from mc1_targets
            mc1_targets = doc.get('mc1_targets', {})
            choices = mc1_targets.get('choices', [])
            labels = mc1_targets.get('labels', [])
            
            correct_answer = None
            for i, label in enumerate(labels):
                if label == 1 and i < len(choices):
                    correct_answer = choices[i]
                    break
                    
            if not correct_answer:
                return None
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            mc1_targets = doc.get('mc1_targets', {})
            choices = mc1_targets.get('choices', [])
            labels = mc1_targets.get('labels', [])
            
            correct_choice = None
            incorrect_choice = None
            
            for i, label in enumerate(labels):
                if label == 1 and i < len(choices):
                    correct_choice = choices[i]
                elif label == 0 and i < len(choices) and incorrect_choice is None:
                    incorrect_choice = choices[i]
                    
            if not all([correct_choice, incorrect_choice]):
                return None
                
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question = doc.get('question', '')
            passage = doc.get('passage', '')
            answer = doc.get('answer', False)
            
            if not all([question, passage]) or answer is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Passage: {passage}\nQuestion: {question}"
                
            # Get correct answer
            correct_answer = "True" if answer else "False"
            
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting BoolQ QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for BoolQ."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            answer = doc.get('answer', False)
            correct_choice = "True" if answer else "False"
            incorrect_choice = "False" if answer else "True"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
            }
            
        except Exception as e:
            logger.debug(f"Error extracting BoolQ contrastive pair: {e}")
            return None


class GSM8KExtractor(BenchmarkExtractor):
    """Extractor for GSM8K benchmark."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        GSM8K format:
        - doc['question']: the math problem
        - doc['answer']: the answer with explanation
        """
        try:
            question = doc.get('question', '')
            answer = doc.get('answer', '')
            
            if not all([question, answer]):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                
            # Extract numerical answer from the answer string
            # GSM8K answers typically end with #### followed by the number
            numerical_answer = answer
            if '####' in answer:
                numerical_answer = answer.split('####')[-1].strip()
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': numerical_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting GSM8K QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for GSM8K."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            correct_answer = qa_pair['correct_answer']
            
            # Create an incorrect answer (slightly different number)
            try:
                num = float(correct_answer)
                incorrect_answer = str(num + 1)
            except ValueError:
                incorrect_answer = "Wrong answer"
                
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            
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
        """
        try:
            question = doc.get('question', '')
            choices = doc.get('choices', [])
            answer = doc.get('answer', 0)
            
            if not all([question, choices]) or answer >= len(choices):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                for i, choice in enumerate(choices):
                    formatted_question += f"\n{chr(65+i)}. {choice}"
                    
            # Get correct answer
            correct_answer = choices[answer]
            
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting MMLU QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MMLU."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            choices = doc.get('choices', [])
            answer = doc.get('answer', 0)
            
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
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            goal = doc.get('goal', '')
            sol1 = doc.get('sol1', '')
            sol2 = doc.get('sol2', '')
            label = doc.get('label', 0)
            
            if not all([goal, sol1, sol2]) or label not in [0, 1]:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Goal: {goal}\nA. {sol1}\nB. {sol2}"
                
            # Get correct answer
            correct_answer = sol1 if label == 0 else sol2
            
            return {
                'question': goal,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting PIQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for PIQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            sol1 = doc.get('sol1', '')
            sol2 = doc.get('sol2', '')
            label = doc.get('label', 0)
            
            correct_choice = sol1 if label == 0 else sol2
            incorrect_choice = sol2 if label == 0 else sol1
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            premise = doc.get('premise', '')
            choice1 = doc.get('choice1', '')
            choice2 = doc.get('choice2', '')
            question = doc.get('question', '')
            label = doc.get('label', 0)
            
            if not all([premise, choice1, choice2, question]) or label not in [0, 1]:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                question_text = f"What was the {question}?"
                formatted_question = f"Premise: {premise}\n{question_text}\nA. {choice1}\nB. {choice2}"
                
            # Get correct answer
            correct_answer = choice1 if label == 0 else choice2
            
            return {
                'question': f"Premise: {premise}\nWhat was the {question}?",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            choice1 = doc.get('choice1', '')
            choice2 = doc.get('choice2', '')
            label = doc.get('label', 0)
            
            correct_choice = choice1 if label == 0 else choice2
            incorrect_choice = choice2 if label == 0 else choice1
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question = doc.get('question_stem', '')
            choices = doc.get('choices', {})
            choice_texts = choices.get('text', [])
            choice_labels = choices.get('label', [])
            answer_key = doc.get('answerKey', '')
            
            if not all([question, choice_texts, choice_labels, answer_key]):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
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
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting OpenBookQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for OpenBookQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            choices = doc.get('choices', {})
            choice_texts = choices.get('text', [])
            choice_labels = choices.get('label', [])
            answer_key = doc.get('answerKey', '')
            
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
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question = doc.get('question', '')
            context = doc.get('context', '')
            answers = doc.get('answers', {})
            answer_texts = answers.get('text', [])
            
            if not all([question, context]):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Context: {context}\nQuestion: {question}"
                
            # Get correct answer
            correct_answer = answer_texts[0] if answer_texts else "No answer"
            
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting SQuAD2 QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for SQuAD2."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            correct_answer = qa_pair['correct_answer']
            incorrect_answer = "Wrong answer" if correct_answer != "No answer" else "Some made-up answer"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            
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
            article = doc.get('article', '')
            question = doc.get('question', '')
            options = doc.get('options', [])
            answer = doc.get('answer', '')
            
            if not all([article, question, options, answer]):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Article: {article}\nQuestion: {question}"
                for i, option in enumerate(options):
                    formatted_question += f"\n{chr(65+i)}. {option}"
                    
            # Get correct answer
            answer_idx = ord(answer) - ord('A')
            correct_answer = options[answer_idx] if answer_idx < len(options) else options[0]
            
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting RACE QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for RACE."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            options = doc.get('options', [])
            answer = doc.get('answer', '')
            
            answer_idx = ord(answer) - ord('A')
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
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            sentence1 = doc.get('sentence1', '')
            sentence2 = doc.get('sentence2', '')
            label = doc.get('label', None)
            
            if not all([sentence1, sentence2]) or label is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (f"Sentence 1: {sentence1}\n"
                                    f"Sentence 2: {sentence2}\n"
                                    f"Question: Do both sentences mean the same thing?")
                
            # Get correct answer
            correct_answer = "yes" if label == 1 else "no"
            
            return {
                'question': f"Do these sentences mean the same thing? '{sentence1}' and '{sentence2}'",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', None)
            if label is None:
                return None
                
            correct_choice = "yes" if label == 1 else "no"
            incorrect_choice = "no" if label == 1 else "yes"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question = doc.get('question', '')
            sentence = doc.get('sentence', '')
            label = doc.get('label', None)
            
            if not all([question, sentence]) or label is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (f"{question}\n{sentence}\n"
                                    f"Question: Does this response answer the question?")
                
            # Get correct answer
            correct_answer = "yes" if label == 0 else "no"
            
            return {
                'question': f"Does this sentence answer the question? Question: {question} Sentence: {sentence}",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', None)
            if label is None:
                return None
                
            correct_choice = "yes" if label == 0 else "no"
            incorrect_choice = "no" if label == 0 else "yes"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            question1 = doc.get('question1', '')
            question2 = doc.get('question2', '')
            label = doc.get('label', None)
            
            if not all([question1, question2]) or label is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (f"Question 1: {question1}\n"
                                    f"Question 2: {question2}\n"
                                    f"Question: Do both questions ask the same thing?")
                
            # Get correct answer
            correct_answer = "yes" if label == 1 else "no"
            
            return {
                'question': f"Do these questions ask the same thing? '{question1}' and '{question2}'",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', None)
            if label is None:
                return None
                
            correct_choice = "yes" if label == 1 else "no"
            incorrect_choice = "no" if label == 1 else "yes"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            sentence1 = doc.get('sentence1', '')
            sentence2 = doc.get('sentence2', '')
            label = doc.get('label', None)
            
            if not all([sentence1, sentence2]) or label is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (f"{sentence1}\n"
                                    f"Question: {sentence2} True or False?")
                
            # Get correct answer
            correct_answer = "True" if label == 1 else "False"
            
            return {
                'question': f"Given '{sentence1}', is it true that '{sentence2}'?",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', None)
            if label is None:
                return None
                
            correct_choice = "True" if label == 1 else "False"
            incorrect_choice = "False" if label == 1 else "True"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            sentence = doc.get('sentence', '')
            label = doc.get('label', None)
            
            if not sentence or label is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (f"{sentence}\n"
                                    f"Question: Is this sentence positive or negative?")
                
            # Get correct answer
            correct_answer = "positive" if label == 1 else "negative"
            
            return {
                'question': f"What is the sentiment of this sentence: '{sentence}'",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', None)
            if label is None:
                return None
                
            correct_choice = "positive" if label == 1 else "negative"
            incorrect_choice = "negative" if label == 1 else "positive"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            sentence1 = doc.get('sentence1', '')
            sentence2 = doc.get('sentence2', '')
            label = doc.get('label', None)
            
            if not all([sentence1, sentence2]) or label is None:
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = (f"{sentence1}\n"
                                    f"Question: {sentence2} True or False?")
                
            # Get correct answer
            correct_answer = "True" if label == 1 else "False"
            
            return {
                'question': f"Given '{sentence1}', is it true that '{sentence2}'?",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', None)
            if label is None:
                return None
                
            correct_choice = "True" if label == 1 else "False"
            incorrect_choice = "False" if label == 1 else "True"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
            }
            
        except Exception as e:
            logger.debug(f"Error extracting WNLI contrastive pair: {e}")
            return None


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
            text = doc.get('page', doc.get('text', ''))
            
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
            
            context = ' '.join(context_words)
            continuation = ' '.join(continuation_words)
            
            # Create a prompt that asks for text continuation
            return {
                'question': f"Continue this text: {context[:200]}...",
                'formatted_question': f"Continue the following text:\n\n{context}",
                'correct_answer': continuation,
                'full_text': text  # Store full text for perplexity calculation
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
            correct_text = qa_pair.get('full_text', qa_pair['correct_answer'])
            
            # Create a "bad" continuation by using random words
            import random
            words = correct_text.split()
            if len(words) > 5:
                # Shuffle some words to create an unnatural continuation
                shuffled_words = words.copy()
                random.shuffle(shuffled_words[:min(20, len(shuffled_words))])
                incorrect_text = ' '.join(shuffled_words)
            else:
                incorrect_text = "This is an incorrect and unrelated continuation that doesn't match the context."
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_text,
                'incorrect_answer': incorrect_text
            
            }
            
        except Exception as e:
            logger.debug(f"Error extracting WikiText contrastive pair: {e}")
            return None

    def get_perplexity_text(self, doc: Dict[str, Any]) -> Optional[str]:
        """Get the text to use for perplexity calculation."""
        return doc.get('page', doc.get('text', ''))


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
            problem = doc.get('Problem', '')
            options_str = doc.get('options', '')
            correct_letter = doc.get('correct', '')
            
            if not all([problem, options_str, correct_letter]):
                return None
                
            # Parse options string into individual choices
            options_parsed = self._parse_options(options_str)
            
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = problem
                for letter, text in options_parsed.items():
                    formatted_question += f"\n{letter.upper()}. {text}"
                    
            # Get correct answer text from parsed options
            correct_answer = options_parsed.get(correct_letter.lower(), '')
            
            if not correct_answer:
                return None
                
            return {
                'question': problem,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting MathQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MathQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            options_str = doc.get('options', '')
            correct_letter = doc.get('correct', '')
            
            # Parse options string
            options_parsed = self._parse_options(options_str)
            
            correct_choice = qa_pair['correct_answer']
            # Get first incorrect choice
            incorrect_choice = None
            for letter, text in options_parsed.items():
                if letter != correct_letter.lower():
                    incorrect_choice = text
                    break
                    
            if not incorrect_choice:
                return None
                
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            parts = options_str.split(',')
            
            for part in parts:
                part = part.strip()
                if ')' in part:
                    # Extract letter and value
                    letter_part, value_part = part.split(')', 1)
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
            sentence = doc.get('sentence', '')
            question = doc.get('question', '')
            answer = doc.get('answer', '')
            label = doc.get('label', 0)
            
            if not all([sentence, question, answer]):
                return None
                
            # Create the question as shown in doc_to_text
            formatted_question = f"{sentence}\nQuestion: {question}\nAnswer: {answer}\nPlausible:"
            
            # The correct answer is based on the label (0=no, 1=yes)
            correct_answer = "yes" if label == 1 else "no"
            
            return {
                'question': f"{sentence} {question}",
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
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
                
            label = doc.get('label', 0)
            
            # The choices are always 'no' and 'yes'
            correct_choice = "yes" if label == 1 else "no"
            incorrect_choice = "no" if label == 1 else "yes"
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
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
            context = doc.get('context', doc.get('background', doc.get('passage', doc.get('section_text', ''))))
            
            # Try multiple field names for question
            question = doc.get('question', doc.get('query', doc.get('current_question', '')))
            
            # Try multiple field names for answer
            answer = doc.get('answer', doc.get('orig_answer', doc.get('answers', {})))
            
            # Handle answer extraction
            if isinstance(answer, dict):
                # QuAC often has answer dict with 'text' field
                answer_text = answer.get('text', '')
                if isinstance(answer_text, list) and answer_text:
                    answer_text = answer_text[0]
            elif isinstance(answer, list) and answer:
                answer_text = answer[0]
            else:
                answer_text = str(answer) if answer else ''
            
            if not all([context, question]):
                return None
                
            # Format the question with context
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = f"Context: {context}\n"
                
                # Add conversation history if available
                history = doc.get('history', doc.get('conversation_history', []))
                if history and isinstance(history, list):
                    # Show last 2 turns of conversation
                    for i, turn in enumerate(history[-2:]):
                        if isinstance(turn, dict):
                            hist_q = turn.get('question', '')
                            hist_a = turn.get('answer', '')
                            if hist_q and hist_a:
                                formatted_question += f"Q{i+1}: {hist_q}\nA{i+1}: {hist_a}\n"
                        elif isinstance(turn, str):
                            formatted_question += f"Previous: {turn}\n"
                
                formatted_question += f"Question: {question}"
                
            # Handle special QuAC answer types
            if doc.get('yesno') == 'y':
                answer_text = "Yes"
            elif doc.get('yesno') == 'n':
                answer_text = "No"
            elif not answer_text or answer_text == "CANNOTANSWER":
                answer_text = "Cannot answer based on the context"
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': answer_text
            }
            
        except Exception as e:
            logger.debug(f"Error extracting QuAC QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for QuAC."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            correct_answer = qa_pair['correct_answer']
            
            # For yes/no questions
            if doc.get('yesno') in ['y', 'n']:
                incorrect_answer = "No" if doc.get('yesno') == 'y' else "Yes"
            # For unanswerable questions
            elif correct_answer == "Cannot answer based on the context":
                incorrect_answer = "The answer is clearly stated in the passage."
            # For regular questions, create contextually plausible but incorrect answers
            else:
                incorrect_answers = [
                    "I don't have enough information to answer that.",
                    "That's not mentioned in the context.",
                    "The passage doesn't provide this information.",
                    "This question cannot be answered from the given text."
                ]
                
                # Pick an answer different from the correct one
                incorrect_answer = incorrect_answers[0]
                for candidate in incorrect_answers:
                    if candidate.lower() != correct_answer.lower():
                        incorrect_answer = candidate
                        break
                
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            
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
            query = doc.get('query', '')
            choices = doc.get('choices', [])
            gold = doc.get('gold', [])
            
            if not all([query, choices]) or not gold or gold[0] >= len(choices):
                return None
                
            # Format the question
            if hasattr(task_data, 'doc_to_text'):
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
            
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting LogiQA QA pair: {e}")
            return None

    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for LogiQA."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            choices = doc.get('choices', [])
            gold = doc.get('gold', [])
            
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
                'question': qa_pair['formatted_question'],
                'correct_answer': correct_choice,
                'incorrect_answer': incorrect_choice
            
            }
            
        except Exception as e:
            logger.debug(f"Error extracting LogiQA contrastive pair: {e}")
            return None


class DefaultExtractor(BenchmarkExtractor):
    """Default extractor that tries common patterns."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Try common extraction patterns."""
        try:
            # Try to get question
            question = doc.get('question', doc.get('ctx', doc.get('goal', doc.get('premise', str(doc)))))
            
            if hasattr(task_data, 'doc_to_text'):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question
                
            # Try various patterns for correct answer
            correct_answer = None
            
            # Pattern 1: mc1_targets (TruthfulQA style)
            mc1_targets = doc.get('mc1_targets', {})
            if mc1_targets:
                choices = mc1_targets.get('choices', [])
                labels = mc1_targets.get('labels', [])
                for i, label in enumerate(labels):
                    if label == 1 and i < len(choices):
                        correct_answer = choices[i]
                        break
            
            # Pattern 2: choices + answerKey (ARC style)
            if not correct_answer:
                choices = doc.get('choices', {})
                if choices:
                    choice_texts = choices.get('text', [])
                    choice_labels = choices.get('label', [])
                    answer_key = doc.get('answerKey', '')
                    
                    for i, label in enumerate(choice_labels):
                        if label == answer_key and i < len(choice_texts):
                            correct_answer = choice_texts[i]
                            break
            
            # Pattern 3: endings + label (HellaSwag style)
            if not correct_answer:
                endings = doc.get('endings', [])
                label = doc.get('label', 0)
                if endings and label < len(endings):
                    correct_answer = endings[label]
            
            # Pattern 4: choices list + answer index (MMLU style)
            if not correct_answer:
                choices = doc.get('choices', [])
                answer = doc.get('answer', 0)
                if choices and answer < len(choices):
                    correct_answer = choices[answer]
            
            # Pattern 5: sol1/sol2 + label (PIQA style)
            if not correct_answer:
                sol1 = doc.get('sol1', '')
                sol2 = doc.get('sol2', '')
                label = doc.get('label', 0)
                if sol1 and sol2:
                    correct_answer = sol1 if label == 0 else sol2
            
            # Pattern 6: choice1/choice2 + label (COPA style)
            if not correct_answer:
                choice1 = doc.get('choice1', '')
                choice2 = doc.get('choice2', '')
                label = doc.get('label', 0)
                if choice1 and choice2:
                    correct_answer = choice1 if label == 0 else choice2
            
            # Pattern 7: option1/option2 + answer (Winogrande style)
            if not correct_answer:
                option1 = doc.get('option1', '')
                option2 = doc.get('option2', '')
                answer = doc.get('answer', '')
                if option1 and option2:
                    correct_answer = option1 if answer == '1' else option2
            
            # Pattern 8: text answer (GSM8K, math problems)
            if not correct_answer:
                answer = doc.get('answer', '')
                if answer:
                    correct_answer = answer
            
            # Pattern 9: boolean answer (BoolQ style)
            if not correct_answer:
                answer = doc.get('answer', None)
                if answer is not None:
                    correct_answer = "True" if answer else "False"
            
            # Pattern 10: text field (WikiText style)
            if not correct_answer:
                text = doc.get('text', '')
                if text:
                    correct_answer = text
            
            # Pattern 11: target field
            if not correct_answer:
                target = doc.get('target', '')
                if target:
                    correct_answer = target
            
            if not correct_answer:
                return None
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer
            }
            
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
            mc1_targets = doc.get('mc1_targets', {})
            if mc1_targets:
                choices = mc1_targets.get('choices', [])
                labels = mc1_targets.get('labels', [])
                for i, label in enumerate(labels):
                    if label == 0 and i < len(choices):
                        incorrect_choice = choices[i]
                        break
            
            # For other formats, create a generic incorrect answer
            if not incorrect_choice:
                incorrect_choice = "Incorrect or irrelevant response"
                
            return {
                'question': qa_pair['formatted_question'],
                'correct_answer': qa_pair['correct_answer'],
                'incorrect_answer': incorrect_choice
            
            }
            
        except Exception as e:
            logger.debug(f"Error extracting default contrastive pair: {e}")
            return None


# Registry of extractors
EXTRACTORS = {
    'winogrande': WinograndeExtractor,
    'arc_challenge': ARCExtractor,
    'arc_easy': ARCExtractor,
    'hellaswag': HellaSwagExtractor,
    'truthfulqa_mc1': TruthfulQAExtractor,
    'truthfulqa_mc2': TruthfulQAExtractor,
    'truthfulqa_gen': TruthfulQAExtractor,
    'boolq': BoolQExtractor,
    'gsm8k': GSM8KExtractor,
    'mmlu': MMLUExtractor,
    'mmmlu': MMLUExtractor,
    'piqa': PIQAExtractor,
    'copa': COPAExtractor,
    'openbookqa': OpenBookQAExtractor,
    'squad2': SQuAD2Extractor,
    'race': RACEExtractor,
    'wikitext': WikiTextExtractor,
    'mrpc': MRPCExtractor,  # GLUE MRPC paraphrase detection
    'qnli': QNLIExtractor,  # GLUE QNLI question-answering NLI
    'qqp': QQPExtractor,  # GLUE QQP question pairs
    'rte': RTEExtractor,  # GLUE RTE textual entailment
    'sst2': SST2Extractor,  # GLUE SST2 sentiment analysis
    'wnli': WNLIExtractor,  # GLUE WNLI Winograd NLI
    # Add more specific extractors for other benchmarks
    'asdiv': GSM8KExtractor,  # Math problems similar to GSM8K
    'cb': COPAExtractor,  # Similar format to COPA
    'coqa': SQuAD2Extractor,  # Similar QA format
    'drop': SQuAD2Extractor,  # Similar QA format
    'logiqa': LogiQAExtractor,  # LogiQA logical reasoning format
    'logiqa2': LogiQAExtractor,  # LogiQA 2.0 uses same format
    'agieval_logiqa_en': LogiQAExtractor,  # AGIEval LogiQA English
    'agieval_logiqa_zh': LogiQAExtractor,  # AGIEval LogiQA Chinese
    'math_qa': MMLUExtractor,  # Multiple choice format
    'mathqa': MathQAExtractor,  # Custom extractor for mathqa's unique format
    'mc_taco': MCTacoExtractor,  # MC-TACO temporal reasoning
    'multirc': MMLUExtractor,  # Multiple choice format
    'mutual': MMLUExtractor,  # Multiple choice format
    'naturalqs': SQuAD2Extractor,  # QA format
    'prost': MMLUExtractor,  # Multiple choice format
    'pubmedqa': MMLUExtractor,  # Multiple choice format
    'quac': QuACExtractor,  # Conversational QA format
    'record': SQuAD2Extractor,  # QA format
    'sciq': MMLUExtractor,  # Multiple choice format
    'swag': HellaSwagExtractor,  # Similar format to HellaSwag
    'toxigen': BoolQExtractor,  # Binary classification
    'triviaqa': SQuAD2Extractor,  # QA format
    'webqs': SQuAD2Extractor,  # QA format
    'wic': BoolQExtractor,  # Binary classification
    'wsc': COPAExtractor,  # Similar format to COPA
    'wsc273': COPAExtractor,  # Similar format to COPA
}


def get_extractor(benchmark_name: str) -> BenchmarkExtractor:
    """Get the appropriate extractor for a benchmark with hard error for unsupported tasks."""
    # Try exact match only - no fallbacks
    if benchmark_name in EXTRACTORS:
        return EXTRACTORS[benchmark_name]()
    
    # Hard error for unsupported benchmarks
    raise UnsupportedBenchmarkError(
        f"No extractor found for benchmark '{benchmark_name}'. "
        f"Supported benchmarks: {sorted(EXTRACTORS.keys())}"
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


def extract_contrastive_pair(benchmark_name: str, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
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