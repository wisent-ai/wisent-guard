"""
Benchmark-specific data extraction logic for multiple choice questions and answers.

Each benchmark has its own data structure and format. This module centralizes
the extraction logic to cleanly handle the differences between benchmarks.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
import re

logger = logging.getLogger(__name__)


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
            Dict with 'question', 'correct_choice', 'incorrect_choice' or None
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_answer,
                'incorrect_choice': incorrect_answer
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
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
                'correct_choice': correct_answer,
                'incorrect_choice': incorrect_answer
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
                'correct_choice': correct_choice,
                'incorrect_choice': incorrect_choice
            }
            
        except Exception as e:
            logger.debug(f"Error extracting RACE contrastive pair: {e}")
            return None


class WikiTextExtractor(BenchmarkExtractor):
    """Extractor for WikiText benchmark (perplexity tasks)."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        WikiText format:
        - doc['text']: the text for perplexity calculation
        """
        try:
            text = doc.get('text', '')
            
            if not text:
                return None
                
            # For perplexity tasks, we don't have traditional QA pairs
            # Instead, we create a "language modeling" task
            return {
                'question': "Complete the text:",
                'formatted_question': f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}",
                'correct_answer': text
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
                
            # For perplexity tasks, create a contrasting "bad" text
            correct_text = qa_pair['correct_answer']
            incorrect_text = "Random unrelated text that doesn't fit the context."
            
            return {
                'question': qa_pair['formatted_question'],
                'correct_choice': correct_text,
                'incorrect_choice': incorrect_text
            }
            
        except Exception as e:
            logger.debug(f"Error extracting WikiText contrastive pair: {e}")
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
                'correct_choice': qa_pair['correct_answer'],
                'incorrect_choice': incorrect_choice
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
    # Add more specific extractors for other benchmarks
    'asdiv': GSM8KExtractor,  # Math problems similar to GSM8K
    'cb': COPAExtractor,  # Similar format to COPA
    'coqa': SQuAD2Extractor,  # Similar QA format
    'drop': SQuAD2Extractor,  # Similar QA format
    'logiqa': MMLUExtractor,  # Multiple choice format
    'math_qa': MMLUExtractor,  # Multiple choice format
    'multirc': MMLUExtractor,  # Multiple choice format
    'mutual': MMLUExtractor,  # Multiple choice format
    'naturalqs': SQuAD2Extractor,  # QA format
    'prost': MMLUExtractor,  # Multiple choice format
    'pubmedqa': MMLUExtractor,  # Multiple choice format
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
    """Get the appropriate extractor for a benchmark."""
    # Try exact match first
    if benchmark_name in EXTRACTORS:
        return EXTRACTORS[benchmark_name]()
    
    # Try partial matches
    for key, extractor_class in EXTRACTORS.items():
        if key in benchmark_name.lower():
            return extractor_class()
    
    # Default fallback
    return DefaultExtractor()


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