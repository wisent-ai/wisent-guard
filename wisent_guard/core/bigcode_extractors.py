"""
Extractors for BigCode Evaluation Harness benchmarks.

This module provides specialized extractors for each BigCode benchmark format.
"""

from typing import Dict, Any, Optional, List
import json
import logging

from .benchmark_extractors import BenchmarkExtractor

logger = logging.getLogger(__name__)


def _create_syntactic_corruption(code: str) -> str:
    """
    Create a syntactically incorrect version of the code by removing tokens.
    
    Args:
        code: The correct code
        
    Returns:
        A syntactically corrupted version of the code
    """
    if not code or not code.strip():
        return "SyntaxError"
    
    tokens = code.split()
    if len(tokens) > 3:
        # Remove a token from the middle to create syntax error
        mid = len(tokens) // 2
        tokens.pop(mid)
    elif len(tokens) > 0:
        # For very short code, remove the last token
        tokens.pop()
        # If we end up with empty result, return partial code
        if not tokens:
            return code[:len(code)//2] if len(code) > 2 else "pass"
    else:
        # If no tokens, return something invalid
        return "SyntaxError"
    
    result = ' '.join(tokens)
    # Never return empty string
    return result if result.strip() else "SyntaxError"


class HumanEvalExtractor(BenchmarkExtractor):
    """Extractor for HumanEval and related benchmarks."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        HumanEval format:
        - task_id: unique identifier (e.g., "HumanEval/0")
        - prompt: function signature with docstring
        - canonical_solution: the correct implementation
        - test: test cases
        - entry_point: function name to test
        """
        try:
            prompt = doc.get('prompt', '')
            solution = doc.get('canonical_solution', '')
            
            if not prompt:
                return None
                
            # The prompt is already well-formatted
            formatted_question = prompt
            
            # For contrastive learning, we need both correct and incorrect
            correct_answer = solution if solution else "pass"
            
            return {
                'question': prompt,
                'formatted_question': formatted_question,
                'correct_answer': correct_answer,
                'task_id': doc.get('task_id', ''),
                'entry_point': doc.get('entry_point', '')
            }
            
        except Exception as e:
            logger.debug(f"Error extracting HumanEval QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for HumanEval."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            prompt = qa_pair['question']
            correct_answer = qa_pair['correct_answer']
            
            # Create incorrect solution
            incorrect_answer = self._create_incorrect_humaneval(correct_answer, doc)
            
            return {
                'question': prompt,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting HumanEval contrastive pair: {e}")
            return None
    
    def _create_incorrect_humaneval(self, correct_code: str, doc: Dict[str, Any]) -> str:
        """Create an incorrect version specifically for HumanEval."""
        if not correct_code or correct_code == "pass":
            raise ValueError("Cannot create incorrect version: no meaningful correct code provided")
            
        lines = correct_code.strip().split('\n')
        
        # Strategy 1: Flip boolean returns
        for i, line in enumerate(lines):
            if 'return True' in line:
                lines[i] = line.replace('return True', 'return False')
                return '\n'.join(lines)
            elif 'return False' in line:
                lines[i] = line.replace('return False', 'return True')
                return '\n'.join(lines)
                
        # Strategy 2: Off-by-one errors in loops
        for i, line in enumerate(lines):
            if 'range(' in line and ')' in line:
                # Add or subtract 1 from range
                if ', ' in line:
                    parts = line.split('range(')[1].split(')')[0].split(', ')
                    if len(parts) >= 2:
                        try:
                            end = int(parts[1])
                            new_end = end - 1 if end > 1 else end + 1
                            lines[i] = line.replace(f'range({parts[0]}, {parts[1]})', 
                                                   f'range({parts[0]}, {new_end})')
                            return '\n'.join(lines)
                        except:
                            pass
                            
        # Strategy 3: Wrong operator
        for i, line in enumerate(lines):
            for old, new in [('<=', '<'), ('>=', '>'), ('<', '<='), ('>', '>='), 
                             ('==', '!='), ('!=', '=='), ('+', '-'), ('-', '+')]:
                if old in line and 'return' in line:
                    lines[i] = line.replace(old, new)
                    return '\n'.join(lines)
                    
        # Default: Return None early
        if len(lines) > 1:
            indent = '    ' if lines[0].startswith('    ') else ''
            lines.insert(1, f"{indent}return None  # Bug: early return")
            
        return '\n'.join(lines)


class MBPPExtractor(BenchmarkExtractor):
    """Extractor for MBPP and MBPP+ benchmarks."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MBPP format:
        - task_id: unique identifier
        - text: problem description
        - code: solution code
        - test_list: list of test assertions
        - test_setup_code: setup code for tests
        - challenge_test_list: additional harder tests (MBPP+)
        """
        try:
            # mbpp_plus uses 'prompt' field instead of 'text'
            text = doc.get('text', doc.get('prompt', ''))
            code = doc.get('code', '')
            
            if not text:
                return None
                
            # Format the question
            formatted_question = f"Write a function to {text.lower().rstrip('.')}"
            
            return {
                'question': text,
                'formatted_question': formatted_question,
                'correct_answer': code,
                'task_id': doc.get('task_id', '')
            }
            
        except Exception as e:
            logger.debug(f"Error extracting MBPP QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MBPP."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create incorrect solution based on the test cases
            test_list = doc.get('test_list', [])
            incorrect_answer = self._create_incorrect_mbpp(correct_answer, test_list)
            
            return {
                'question': question,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting MBPP contrastive pair: {e}")
            return None
    
    def _create_incorrect_mbpp(self, correct_code: str, test_list: List[str]) -> str:
        """Create incorrect MBPP solution."""
        if not correct_code:
            raise ValueError("Cannot create incorrect solution: no correct code provided")
            
        # Analyze test cases to understand expected behavior
        has_numeric = any('assert' in test and any(c.isdigit() for c in test) for test in test_list)
        has_string = any('assert' in test and ("'" in test or '"' in test) for test in test_list)
        has_list = any('assert' in test and ('[' in test or 'list' in test.lower()) for test in test_list)
        
        lines = correct_code.split('\n')
        
        # Try to introduce type-specific bugs
        if has_numeric:
            for i, line in enumerate(lines):
                if 'return' in line and any(op in line for op in ['+', '-', '*', '/']):
                    # Off-by-one error
                    if '+' in line:
                        lines[i] = line.replace('+', '+ 1 +')
                    elif '-' in line and not '--' in line:
                        lines[i] = line.replace('-', '- 1 -')
                    return '\n'.join(lines)
                    
        if has_string:
            for i, line in enumerate(lines):
                if 'return' in line and ("'" in line or '"' in line):
                    # Return empty string
                    lines[i] = re.sub(r'return\s+["\'].*["\']', 'return ""', line)
                    return '\n'.join(lines)
                    
        if has_list:
            for i, line in enumerate(lines):
                if 'return' in line and '[' in line:
                    # Return empty list
                    lines[i] = re.sub(r'return\s+\[.*\]', 'return []', line)
                    return '\n'.join(lines)
                    
        # Default: wrong return type
        for i, line in enumerate(lines):
            if 'return' in line and not 'return None' in line:
                lines[i] = '    return None  # Bug: wrong return'
                return '\n'.join(lines)
                
        # Last resort: add early return
        if len(lines) > 1:
            lines.insert(1, '    return None  # Bug: early return')
            
        return '\n'.join(lines)


class APPSExtractor(BenchmarkExtractor):
    """Extractor for APPS benchmark."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        APPS format:
        - problem_id: unique identifier
        - question: problem statement
        - solutions: JSON string of solution list
        - input_output: JSON string with test inputs/outputs
        - difficulty: problem difficulty
        - starter_code: optional starter code
        """
        try:
            question = doc.get('question', '')
            solutions = doc.get('solutions', '[]')
            starter_code = doc.get('starter_code', '')
            
            if not question:
                return None
                
            # Parse solutions if it's a string
            if isinstance(solutions, str):
                try:
                    solutions = json.loads(solutions)
                except:
                    solutions = []
                    
            # Get the first solution or use starter code
            if solutions and len(solutions) > 0:
                correct_answer = solutions[0]
            elif starter_code:
                correct_answer = starter_code
            else:
                correct_answer = "# Write your solution here"
                
            return {
                'question': question,
                'formatted_question': question,
                'correct_answer': correct_answer,
                'problem_id': doc.get('problem_id', '')
            }
            
        except Exception as e:
            logger.debug(f"Error extracting APPS QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for APPS."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create incorrect solution
            io_json = doc.get('input_output', '{}')
            if isinstance(io_json, str):
                try:
                    io_data = json.loads(io_json)
                except:
                    io_data = {}
            else:
                io_data = io_json
                
            incorrect_answer = self._create_incorrect_apps(correct_answer, io_data)
            
            return {
                'question': question,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting APPS contrastive pair: {e}")
            return None
    
    def _create_incorrect_apps(self, correct_code: str, io_data: Dict) -> str:
        """Create incorrect APPS solution."""
        if not correct_code or correct_code == "# Write your solution here":
            return "print('Error: Not implemented')"
            
        # APPS often involves reading input and printing output
        lines = correct_code.split('\n')
        
        # Look for print statements and modify them
        for i, line in enumerate(lines):
            if 'print(' in line:
                # Print wrong output
                if any(x in line for x in ['True', 'False', 'Yes', 'No']):
                    for old, new in [('True', 'False'), ('False', 'True'), 
                                     ('Yes', 'No'), ('No', 'Yes')]:
                        if old in line:
                            lines[i] = line.replace(old, new)
                            return '\n'.join(lines)
                else:
                    lines[i] = 'print("Wrong output")'
                    return '\n'.join(lines)
                    
        # Look for mathematical operations
        for i, line in enumerate(lines):
            if any(op in line for op in ['+', '-', '*', '/']) and '=' in line:
                # Introduce calculation error
                if '+' in line:
                    lines[i] = line.replace('+', '+ 1 +')
                elif '*' in line:
                    lines[i] = line.replace('*', '* 2 *')
                return '\n'.join(lines)
                
        # Default: print wrong answer
        if 'print' not in correct_code:
            lines.append('print("0")')  # Wrong default output
        else:
            lines.insert(0, 'print("Wrong"); exit()')
            
        return '\n'.join(lines)


class CodeXGLUEExtractor(BenchmarkExtractor):
    """Extractor for CodeXGLUE code-to-text tasks."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        CodeXGLUE code-to-text format:
        - code: the code snippet
        - docstring/nl/comment: natural language description
        - repo: repository name (optional)
        - path: file path (optional)
        """
        try:
            code = doc.get('code', '')
            description = doc.get('docstring') or doc.get('nl') or doc.get('comment') or ''
            
            if not code:
                return None
                
            # Format as a documentation task
            formatted_question = f"Generate documentation for this code:\n```\n{code}\n```"
            
            return {
                'question': code,
                'formatted_question': formatted_question,
                'correct_answer': description
            }
            
        except Exception as e:
            logger.debug(f"Error extracting CodeXGLUE QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for CodeXGLUE."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create incorrect documentation
            incorrect_answer = self._create_incorrect_docs(correct_answer, doc.get('code', ''))
            
            return {
                'question': question,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting CodeXGLUE contrastive pair: {e}")
            return None
    
    def _create_incorrect_docs(self, correct_docs: str, code: str) -> str:
        """Create incorrect documentation."""
        if not correct_docs:
            return "This function does nothing useful."
            
        # Strategy 1: Negate the description
        negations = [
            ('returns', 'does not return'),
            ('creates', 'destroys'),
            ('adds', 'removes'),
            ('increases', 'decreases'),
            ('true', 'false'),
            ('false', 'true'),
            ('success', 'failure'),
            ('valid', 'invalid')
        ]
        
        incorrect = correct_docs
        for old, new in negations:
            if old in incorrect.lower():
                # Case-insensitive replacement
                import re
                incorrect = re.sub(rf'\b{old}\b', new, incorrect, flags=re.IGNORECASE)
                return incorrect
                
        # Strategy 2: Wrong functionality
        if 'sort' in code.lower():
            return "This function shuffles the input randomly."
        elif 'search' in code.lower() or 'find' in code.lower():
            return "This function deletes the specified element."
        elif 'add' in code.lower() or 'insert' in code.lower():
            return "This function removes elements from the collection."
        elif 'delete' in code.lower() or 'remove' in code.lower():
            return "This function adds new elements to the collection."
            
        # Default: Generic wrong description
        return "This function performs the opposite of what the code actually does."


class DS1000Extractor(BenchmarkExtractor):
    """Extractor for DS-1000 data science benchmark."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        DS-1000 format:
        - problem: problem description
        - prompt: specific prompt
        - code_context: context code
        - test: test cases
        - reference_code: reference solution (not 'solution')
        """
        try:
            # Handle DS1000Problem objects
            if hasattr(doc, 'data') and isinstance(doc.data, dict):
                doc = doc.data
            
            if isinstance(doc, dict):
                problem = doc.get('problem', '')
                prompt = doc.get('prompt', '')
                code_context = doc.get('code_context', '')
                solution = doc.get('reference_code', doc.get('solution', ''))
            else:
                # Try to extract attributes from DS1000Problem
                problem = getattr(doc, 'problem', '')
                prompt = getattr(doc, 'prompt', '')
                code_context = getattr(doc, 'code_context', '')
                solution = getattr(doc, 'reference_code', getattr(doc, 'solution', ''))
            
            if not problem and not prompt:
                return None
                
            # Combine problem and prompt
            question = f"{problem}\n{prompt}" if problem and prompt else (problem or prompt)
            
            # Add code context if available
            if code_context:
                formatted_question = f"{question}\n\nContext:\n```python\n{code_context}\n```"
            else:
                formatted_question = question
                
            return {
                'question': question,
                'formatted_question': formatted_question,
                'correct_answer': solution or "# Your solution here"
            }
            
        except Exception as e:
            logger.debug(f"Error extracting DS-1000 QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for DS-1000."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create incorrect data science solution
            incorrect_answer = self._create_incorrect_ds1000(correct_answer)
            
            return {
                'question': question,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting DS-1000 contrastive pair: {e}")
            return None
    
    def _create_incorrect_ds1000(self, correct_code: str) -> str:
        """Create incorrect DS-1000 solution."""
        if not correct_code or correct_code == "# Your solution here":
            return "import pandas as pd\nresult = None  # Wrong implementation"
            
        lines = correct_code.split('\n')
        
        # Common DS-1000 patterns and their bugs
        for i, line in enumerate(lines):
            # Pandas operations
            if '.mean()' in line:
                lines[i] = line.replace('.mean()', '.sum()')
                return '\n'.join(lines)
            elif '.sum()' in line:
                lines[i] = line.replace('.sum()', '.mean()')
                return '\n'.join(lines)
            elif '.max()' in line:
                lines[i] = line.replace('.max()', '.min()')
                return '\n'.join(lines)
            elif '.min()' in line:
                lines[i] = line.replace('.min()', '.max()')
                return '\n'.join(lines)
                
            # NumPy operations
            elif 'axis=0' in line:
                lines[i] = line.replace('axis=0', 'axis=1')
                return '\n'.join(lines)
            elif 'axis=1' in line:
                lines[i] = line.replace('axis=1', 'axis=0')
                return '\n'.join(lines)
                
            # Indexing errors
            elif '[0]' in line:
                lines[i] = line.replace('[0]', '[1]')
                return '\n'.join(lines)
            elif '[-1]' in line:
                lines[i] = line.replace('[-1]', '[0]')
                return '\n'.join(lines)
                
        # Default: Return wrong shape/type
        for i, line in enumerate(lines):
            if 'return' in line:
                lines[i] = 'return None  # Bug: wrong output'
                return '\n'.join(lines)
                
        lines.append('result = None  # Bug: wrong result')
        return '\n'.join(lines)


class ConalaExtractor(BenchmarkExtractor):
    """Extractor for CoNaLa (Code from Natural Language) benchmark."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        CoNaLa format:
        - intent: natural language description
        - snippet: code snippet
        """
        try:
            intent = doc.get('intent', '')
            snippet = doc.get('snippet', '')
            
            if not intent or not snippet:
                return None
                
            return {
                'question': intent,
                'formatted_question': f"Write Python code to: {intent}",
                'correct_answer': snippet
            }
            
        except Exception as e:
            logger.debug(f"Error extracting CoNaLa QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for CoNaLa."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create syntactic corruption
            incorrect_answer = _create_syntactic_corruption(correct_answer)
            
            return {
                'question': question,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting CoNaLa contrastive pair: {e}")
            return None


class MultiPLEExtractor(BenchmarkExtractor):
    """Extractor for MultiPL-E benchmarks (multiple_js, multiple_java, etc.)."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        MultiPL-E format:
        - prompt: the function signature and docstring
        - tests: test cases
        - stop_tokens: tokens to stop generation
        - name: problem name
        
        These are generation tasks - we need to handle them specially.
        """
        try:
            prompt = doc.get('prompt', '')
            
            if not prompt:
                return None
                
            # For MultiPL-E, we don't have solutions but we can still
            # create QA pairs with the prompt
            return {
                'question': prompt,
                'formatted_question': prompt,
                'correct_answer': 'pass',  # Placeholder - actual solution comes from generation
                'name': doc.get('name', ''),
                'tests': doc.get('tests', '')
            }
            
        except Exception as e:
            logger.debug(f"Error extracting MultiPL-E QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for MultiPL-E by creating a minimal stub."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            prompt = qa_pair['question']
            
            # Extract language from prompt or use a default
            if 'function' in prompt and '{' in prompt:
                # JavaScript/Java/C++/Go style
                correct_answer = "return null;"
                incorrect_answer = "returnnull"
            elif 'fn ' in prompt:
                # Rust style
                correct_answer = "unimplemented!()"
                incorrect_answer = "unimplemented!"
            else:
                # Python style (default)
                correct_answer = "pass"
                incorrect_answer = "pas"
            
            return {
                'question': prompt,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting MultiPL-E contrastive pair: {e}")
            return None


class ConcodeExtractor(BenchmarkExtractor):
    """Extractor for Concode benchmark."""
    
    def extract_qa_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """
        Concode format:
        - nl: natural language description
        - code: code implementation
        """
        try:
            nl = doc.get('nl', '')
            code = doc.get('code', '')
            
            if not nl or not code:
                return None
                
            return {
                'question': nl,
                'formatted_question': f"Implement the following: {nl}",
                'correct_answer': code
            }
            
        except Exception as e:
            logger.debug(f"Error extracting Concode QA pair: {e}")
            return None
    
    def extract_contrastive_pair(self, doc: Dict[str, Any], task_data: Any = None) -> Optional[Dict[str, str]]:
        """Extract contrastive pair for Concode."""
        try:
            qa_pair = self.extract_qa_pair(doc, task_data)
            if not qa_pair:
                return None
                
            question = qa_pair['formatted_question']
            correct_answer = qa_pair['correct_answer']
            
            # Create syntactic corruption for Java code
            incorrect_answer = _create_syntactic_corruption(correct_answer)
            
            return {
                'question': question,
                'correct_answer': correct_answer,
                'incorrect_answer': incorrect_answer
            }
            
        except Exception as e:
            logger.debug(f"Error extracting Concode contrastive pair: {e}")
            return None


# Registry mapping task names to extractors
BIGCODE_EXTRACTORS = {
    'humaneval': HumanEvalExtractor,
    'humaneval_plus': HumanEvalExtractor,
    'instructhumaneval': HumanEvalExtractor,
    'apps': APPSExtractor,
    'mbpp_plus': MBPPExtractor,
    'ds1000': DS1000Extractor,
    'humanevalpack': HumanEvalExtractor,
    'multiple_py': MultiPLEExtractor,
    'multiple_js': MultiPLEExtractor,
    'multiple_java': MultiPLEExtractor,
    'multiple_cpp': MultiPLEExtractor,
    'multiple_rs': MultiPLEExtractor,
    'multiple_go': MultiPLEExtractor,
    'recode': HumanEvalExtractor,
    'conala': ConalaExtractor,
    'concode': ConcodeExtractor,
    'codexglue_code_to_text': CodeXGLUEExtractor,
    'codexglue_code_to_text_python': CodeXGLUEExtractor,
    'codexglue_code_to_text_go': CodeXGLUEExtractor,
    'codexglue_code_to_text_ruby': CodeXGLUEExtractor,
    'codexglue_code_to_text_java': CodeXGLUEExtractor,
    'codexglue_code_to_text_javascript': CodeXGLUEExtractor,
    'codexglue_code_to_text_php': CodeXGLUEExtractor,
    'mercury': HumanEvalExtractor,  # Similar to HumanEval
}


def get_bigcode_extractor(task_name: str) -> BenchmarkExtractor:
    """Get the appropriate BigCode extractor for a task."""
    if task_name in BIGCODE_EXTRACTORS:
        return BIGCODE_EXTRACTORS[task_name]()
    else:
        # Default to HumanEval extractor
        logger.warning(f"No specific extractor for {task_name}, using HumanEvalExtractor")
        return HumanEvalExtractor()