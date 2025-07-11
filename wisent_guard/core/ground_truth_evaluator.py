from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GroundTruthMethod(Enum):
    """Different methods for determining ground truth."""
    EXACT_MATCH = "exact_match"
    SUBSTRING_MATCH = "substring_match"
    USER_SPECIFIED = "user_specified"
    INTERACTIVE = "interactive"
    MANUAL_REVIEW = "manual_review"
    NONE = "none"
    GOOD = "good"  # Everything is truthful (for debugging)
    LM_EVAL_HARNESS = "lm-eval-harness"  # Use lm-eval-harness tasks for ground truth


class GroundTruthEvaluator:
    """Handles ground truth evaluation for different tasks and methods."""
    
    def __init__(self, method: GroundTruthMethod = GroundTruthMethod.NONE, task_name: Optional[str] = None):
        self.method = method
        self.task_name = task_name
        
    def evaluate_response(
        self, 
        response: str, 
        expected_answer, 
        user_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response against ground truth using the specified method.
        
        Args:
            response: The generated response to evaluate
            expected_answer: The expected correct answer
            user_label: User-specified label ("truthful", "hallucination", or None)
            
        Returns:
            Dict containing:
            - ground_truth: "TRUTHFUL" or "HALLUCINATION" or "UNKNOWN"
            - method_used: The evaluation method that was used
            - confidence: Confidence score (0.0 to 1.0) if applicable
            - details: Additional details about the evaluation
        """
        
        if self.method == GroundTruthMethod.USER_SPECIFIED and user_label:
            return self._evaluate_user_specified(response, user_label)
        elif self.method == GroundTruthMethod.EXACT_MATCH:
            return self._evaluate_exact_match(response, expected_answer)
        elif self.method == GroundTruthMethod.SUBSTRING_MATCH:
            return self._evaluate_substring_match(response, expected_answer)
        elif self.method == GroundTruthMethod.INTERACTIVE:
            return self._evaluate_interactive(response, expected_answer)
        elif self.method == GroundTruthMethod.MANUAL_REVIEW:
            return self._evaluate_manual_review(response, expected_answer)
        elif self.method == GroundTruthMethod.GOOD:
            return self._evaluate_good(response, expected_answer)
        elif self.method == GroundTruthMethod.LM_EVAL_HARNESS:
            return self._evaluate_lm_eval_harness(response, expected_answer)
        else:
            return self._evaluate_none()
    
    def _evaluate_user_specified(self, response: str, user_label: str) -> Dict[str, Any]:
        """Evaluate using user-specified ground truth."""
        user_label_lower = user_label.lower().strip()
        
        if user_label_lower in ["truthful", "true", "correct", "1"]:
            ground_truth = "TRUTHFUL"
        elif user_label_lower in ["hallucination", "false", "incorrect", "0"]:
            ground_truth = "HALLUCINATION"
        else:
            logger.warning(f"Unrecognized user label: {user_label}. Defaulting to UNKNOWN.")
            ground_truth = "UNKNOWN"
            
        return {
            "ground_truth": ground_truth,
            "method_used": "user_specified",
            "confidence": 1.0,
            "details": f"User specified label: {user_label}"
        }
    
    def _evaluate_exact_match(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate using exact match (problematic for free-form generation).
        This method is preserved but not recommended for TruthfulQA-style tasks.
        """
        try:
            from lm_eval.api.metrics import exact_match_hf_evaluate
            
            # Try lm-eval's exact match evaluation
            is_actually_correct = exact_match_hf_evaluate([response], [expected_answer])
            
            # Handle different return formats from exact_match_hf_evaluate
            if isinstance(is_actually_correct, dict):
                is_correct = bool(is_actually_correct.get('exact_match', 0))
            elif isinstance(is_actually_correct, (list, tuple)) and len(is_actually_correct) > 0:
                is_correct = bool(is_actually_correct[0])
            else:
                is_correct = bool(is_actually_correct)
                
            confidence = 1.0 if is_correct else 0.0
            details = f"Exact match evaluation using lm-eval harness"
            
        except Exception as e:
            logger.warning(f"lm-eval exact match failed: {e}. Using simple string comparison.")
            is_correct = response.strip().lower() == expected_answer.strip().lower()
            confidence = 1.0 if is_correct else 0.0
            details = f"Simple exact string comparison (lm-eval failed)"
        
        return {
            "ground_truth": "TRUTHFUL" if is_correct else "HALLUCINATION",
            "method_used": "exact_match",
            "confidence": confidence,
            "details": details
        }
    
    def _evaluate_substring_match(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """
        Evaluate using substring matching (also problematic for free-form generation).
        This method is preserved but not recommended for TruthfulQA-style tasks.
        """
        is_correct = expected_answer.lower().strip() in response.lower().strip()
        
        return {
            "ground_truth": "TRUTHFUL" if is_correct else "HALLUCINATION",
            "method_used": "substring_match",
            "confidence": 1.0 if is_correct else 0.0,
            "details": f"Substring matching: looking for '{expected_answer}' in response"
        }
    
    def _evaluate_interactive(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate using interactive prompts for user input."""
        print("\n" + "="*80)
        print("GROUND TRUTH EVALUATION - Interactive Mode")
        print("="*80)
        
        # Show expected answer if available
        if expected_answer:
            print(f"Expected Answer: {expected_answer}")
            print("-" * 40)
        
        # Show the response
        print(f"Model Response: {response}")
        print("-" * 40)
        
        # Get user input
        while True:
            user_input = input("Is this response truthful? (y/n/q to quit): ").strip().lower()
            
            if user_input in ['y', 'yes', 'true', 't', '1']:
                ground_truth = "TRUTHFUL"
                details = "User labeled as truthful"
                break
            elif user_input in ['n', 'no', 'false', 'f', '0']:
                ground_truth = "HALLUCINATION"
                details = "User labeled as hallucination"
                break
            elif user_input in ['q', 'quit', 'exit']:
                print("Exiting interactive evaluation...")
                exit(0)
            else:
                print("Please enter 'y' for truthful, 'n' for hallucination, or 'q' to quit.")
                continue
        
        return {
            "ground_truth": ground_truth,
            "method_used": "interactive",
            "confidence": 1.0,
            "details": details
        }
    
    def _evaluate_manual_review(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Mark for manual review - no automatic evaluation."""
        return {
            "ground_truth": "UNKNOWN",
            "method_used": "manual_review",
            "confidence": 0.0,
            "details": "Marked for manual review - no automatic evaluation performed"
        }
    
    def _evaluate_good(self, response: str, expected_answer: str) -> Dict[str, Any]:
        """Everything is truthful (for debugging)."""
        return {
            "ground_truth": "TRUTHFUL",
            "method_used": "good",
            "confidence": 1.0,
            "details": "Debug mode: everything labeled as truthful"
        }

    def _evaluate_lm_eval_harness(self, response: str, expected_answer) -> Dict[str, Any]:
        """Evaluate using lm-eval-harness framework with benchmark extractors."""
        try:
            # Import benchmark extractors for intelligent evaluation
            from .benchmark_extractors import get_extractor
            
            # Get the task name for benchmark-specific evaluation
            task_name = getattr(self, 'task_name', None) or "unknown"
            
            # 🚨 FIX: Use benchmark extractors for ALL benchmarks instead of hard-coded logic
            # This ensures we use the same evaluation logic as the extractors
            
            # Get the appropriate extractor for this benchmark
            extractor = get_extractor(task_name)
            
            # Use the extractor's evaluation logic to determine correctness
            is_correct = self._evaluate_using_extractor(response, expected_answer, extractor, task_name)
            
            ground_truth = "TRUTHFUL" if is_correct else "HALLUCINATION"
            confidence = 1.0 if is_correct else 0.0
            
            return {
                "ground_truth": ground_truth,
                "method_used": "lm-eval-harness",
                "confidence": confidence,
                "details": f"Benchmark extractor evaluation for {task_name}",
                "task_name": task_name,
                "evaluation_method": "lm-eval-harness",
                "extractor_type": extractor.__class__.__name__
            }
            
        except Exception as e:
            logger.error(f"Error in lm-eval-harness evaluation: {e}")
            return {
                "ground_truth": "UNKNOWN",
                "method_used": "lm-eval-harness",
                "confidence": 0.0,
                "details": f"Error during benchmark evaluation: {str(e)}",
                "task_name": getattr(self, 'task_name', None) or "unknown",
                "evaluation_method": "lm-eval-harness"
            }
    
    def _evaluate_using_extractor(self, response: str, expected_answer, extractor, task_name: str) -> bool:
        """Evaluate response using the appropriate benchmark extractor logic."""
        try:
            # Get the extractor class name to determine evaluation strategy
            extractor_name = extractor.__class__.__name__
            
            # Use extractor-specific evaluation logic
            if extractor_name == "WinograndeExtractor":
                return self._evaluate_winogrande_logic(response, expected_answer)
            
            elif extractor_name == "TruthfulQAExtractor":
                return self._evaluate_truthfulqa_logic(response, expected_answer)
            
            elif extractor_name in ["ARCExtractor", "HellaSwagExtractor", "MMLUExtractor", "PIQAExtractor", "COPAExtractor", "OpenBookQAExtractor", "RACEExtractor"]:
                return self._evaluate_multiple_choice_logic(response, expected_answer)
            
            elif extractor_name == "BoolQExtractor":
                return self._evaluate_boolean_logic(response, expected_answer)
            
            elif extractor_name == "GSM8KExtractor":
                return self._evaluate_numerical_logic(response, expected_answer)
            
            elif extractor_name == "SQuAD2Extractor":
                return self._evaluate_qa_logic(response, expected_answer)
            
            elif extractor_name == "WikiTextExtractor":
                return self._evaluate_text_logic(response, expected_answer)
            
            elif extractor_name == "DefaultExtractor":
                return self._evaluate_flexible_logic(response, expected_answer)
            
            else:
                # For any unknown extractor, use flexible matching
                logger.warning(f"Unknown extractor type: {extractor_name}. Using flexible matching.")
                return self._evaluate_flexible_logic(response, expected_answer)
                
        except Exception as e:
            logger.error(f"Error in extractor-based evaluation: {e}")
            # Fall back to flexible matching
            return self._evaluate_flexible_logic(response, expected_answer)
    
    def _evaluate_winogrande_logic(self, response: str, expected_answer) -> bool:
        """Winogrande: exact match with correct option."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers, try both string conversion and numeric comparison
            expected_str = str(expected_answer)
            if expected_str in response_clean or response_clean == expected_str:
                return True
            # Also try to extract numbers from response for numeric comparison
            import re
            response_numbers = re.findall(r'\b\d+\b', response)
            if str(expected_answer) in response_numbers:
                return True
        else:
            # String expected answer - use original logic
            expected_clean = str(expected_answer).strip().lower()
            
            # Direct match
            if response_clean == expected_clean:
                return True
                
            # Check if response contains the expected answer
            if expected_clean in response_clean:
                return True
                
        return False
    
    def _evaluate_truthfulqa_logic(self, response: str, expected_answer) -> bool:
        """TruthfulQA: more flexible matching since responses can be paraphrased."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers, convert to string and do flexible matching
            expected_str = str(expected_answer)
            if expected_str in response_clean or response_clean == expected_str:
                return True
            # Also try numeric extraction
            import re
            response_numbers = re.findall(r'\b\d+\b', response)
            if str(expected_answer) in response_numbers:
                return True
        else:
            # String expected answer - use original flexible logic
            expected_clean = str(expected_answer).strip().lower()
            
            # Direct match
            if response_clean == expected_clean:
                return True
                
            # Check if response contains the expected answer
            if expected_clean in response_clean:
                return True
                
            # Check if expected answer contains the response (for shorter responses)
            if response_clean in expected_clean:
                return True
                
        return False
    
    def _evaluate_multiple_choice_logic(self, response: str, expected_answer) -> bool:
        """Multiple choice: exact match with correct option."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers (choice indices), try multiple approaches
            expected_str = str(expected_answer)
            
            # Direct numeric match
            if expected_str in response_clean or response_clean == expected_str:
                return True
                
            # Extract numbers from response for comparison
            import re
            response_numbers = re.findall(r'\b\d+\b', response)
            if str(expected_answer) in response_numbers:
                return True
                
            # Try letter matching (0->A, 1->B, 2->C, etc.)
            if expected_answer < 26:  # Reasonable choice limit
                expected_letter = chr(65 + expected_answer).lower()  # 0->a, 1->b, etc.
                if expected_letter in response_clean:
                    return True
        else:
            # String expected answer - use original logic
            expected_clean = str(expected_answer).strip().lower()
            
            # Direct match
            if response_clean == expected_clean:
                return True
                
            # Check if response contains the expected answer
            if expected_clean in response_clean:
                return True
                
        return False
    
    def _evaluate_boolean_logic(self, response: str, expected_answer) -> bool:
        """Boolean: match True/False with various representations."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers (0/1), convert to boolean logic
            if expected_answer == 1:
                return response_clean in ["true", "yes", "1", "correct"]
            elif expected_answer == 0:
                return response_clean in ["false", "no", "0", "incorrect"]
            else:
                # For other integers, try direct numeric comparison
                return str(expected_answer) in response_clean
        else:
            # String expected answer - use original logic
            expected_clean = str(expected_answer).strip().lower()
            
            # Direct match
            if response_clean == expected_clean:
                return True
                
            # Handle various boolean representations
            if expected_clean == "true":
                return response_clean in ["true", "yes", "1", "correct"]
            elif expected_clean == "false":
                return response_clean in ["false", "no", "0", "incorrect"]
                
        return False
    
    def _evaluate_numerical_logic(self, response: str, expected_answer) -> bool:
        """Numerical: extract and compare numbers."""
        try:
            # 🚨 TYPE-SAFE: Handle both integers and strings
            if isinstance(expected_answer, (int, float)):
                # For numeric expected answers, do direct numeric comparison
                import re
                response_numbers = re.findall(r'-?\d+\.?\d*', response)
                
                if not response_numbers:
                    return False
                    
                # Compare the last number in response with expected number
                response_num = float(response_numbers[-1])
                expected_num = float(expected_answer)
                
                return abs(response_num - expected_num) < 0.01  # Allow small floating point errors
            else:
                # String expected answer - extract numbers from both
                import re
                response_numbers = re.findall(r'-?\d+\.?\d*', response)
                expected_numbers = re.findall(r'-?\d+\.?\d*', str(expected_answer))
                
                if not response_numbers or not expected_numbers:
                    return False
                    
                # Compare the last number in each (often the final answer)
                response_num = float(response_numbers[-1])
                expected_num = float(expected_numbers[-1])
                
                return abs(response_num - expected_num) < 0.01  # Allow small floating point errors
            
        except (ValueError, IndexError):
            # Fall back to string comparison
            return response.strip().lower() == str(expected_answer).strip().lower()
    
    def _evaluate_qa_logic(self, response: str, expected_answer) -> bool:
        """QA tasks: flexible matching for natural language answers."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers, try numeric and string matching
            expected_str = str(expected_answer)
            if expected_str in response_clean or response_clean == expected_str:
                return True
            # Also try numeric extraction
            import re
            response_numbers = re.findall(r'\b\d+\b', response)
            if str(expected_answer) in response_numbers:
                return True
        else:
            # String expected answer - use original flexible logic
            expected_clean = str(expected_answer).strip().lower()
            
            # Handle "No answer" case
            if expected_clean == "no answer":
                return response_clean in ["no answer", "no", "none", "unknown", "unanswerable"]
            
            # Direct match
            if response_clean == expected_clean:
                return True
                
            # Check if response contains the expected answer
            if expected_clean in response_clean:
                return True
                
            # Check if expected answer contains the response
            if response_clean in expected_clean:
                return True
                
        return False
    
    def _evaluate_text_logic(self, response: str, expected_answer) -> bool:
        """Text/perplexity tasks: flexible text matching."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers, try string conversion
            expected_str = str(expected_answer)
            return expected_str in response_clean or response_clean == expected_str
        else:
            # String expected answer - use original logic
            expected_clean = str(expected_answer).strip().lower()
            
            # For text tasks, use substring matching
            if expected_clean in response_clean or response_clean in expected_clean:
                return True
                
        return False
    
    def _evaluate_flexible_logic(self, response: str, expected_answer) -> bool:
        """Flexible matching for unknown benchmark types."""
        response_clean = response.strip().lower()
        
        # 🚨 TYPE-SAFE: Handle both integers and strings
        if isinstance(expected_answer, int):
            # For integer expected answers, try multiple matching approaches
            expected_str = str(expected_answer)
            
            # Direct string match
            if response_clean == expected_str or expected_str in response_clean:
                return True
                
            # Extract numbers from response for numeric comparison
            import re
            response_numbers = re.findall(r'\b\d+\b', response)
            if str(expected_answer) in response_numbers:
                return True
        else:
            # String expected answer - use original flexible logic
            expected_clean = str(expected_answer).strip().lower()
            
            # Direct match
            if response_clean == expected_clean:
                return True
                
            # Check if response contains the expected answer
            if expected_clean in response_clean:
                return True
                
            # Check if expected answer contains the response
            if response_clean in expected_clean:
                return True
                
        return False

    def _evaluate_none(self) -> Dict[str, Any]:
        """No ground truth evaluation."""
        return {
            "ground_truth": "UNKNOWN",
            "method_used": "none",
            "confidence": 0.0,
            "details": "No ground truth evaluation method specified"
        }
    
    def evaluate_batch(
        self, 
        responses: List[str], 
        expected_answers: List,
        user_labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of responses.
        
        Args:
            responses: List of generated responses
            expected_answers: List of expected correct answers
            user_labels: Optional list of user-specified labels
            
        Returns:
            List of evaluation results
        """
        results = []
        
        if user_labels is None:
            user_labels = [None] * len(responses)
        
        for i, (response, expected, user_label) in enumerate(zip(responses, expected_answers, user_labels)):
            try:
                result = self.evaluate_response(response, expected, user_label)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating response {i}: {e}")
                results.append({
                    "ground_truth": "UNKNOWN",
                    "method_used": "error",
                    "confidence": 0.0,
                    "details": f"Error during evaluation: {str(e)}"
                })
        
        return results
    
    @classmethod
    def from_string(cls, method_str: str, task_name: Optional[str] = None) -> 'GroundTruthEvaluator':
        """Create evaluator from string representation."""
        try:
            method = GroundTruthMethod(method_str.lower())
            return cls(method, task_name)
        except ValueError:
            logger.warning(f"Unknown ground truth method: {method_str}. Using 'none'.")
            return cls(GroundTruthMethod.NONE, task_name)
