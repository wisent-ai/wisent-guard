"""
BigCode Evaluation Harness integration for Wisent Guard.

This module provides integration with bigcode-evaluation-harness for code generation benchmarks.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BigCodeTaskLoader:
    """Loads and manages BigCode evaluation tasks."""

    # Mapping of our task names to BigCode task names
    TASK_MAPPING = {
        # === DIRECT MATCHES ===
        "humaneval": "humaneval",
        "mbpp": "mbpp",
        "conala": "conala",
        "concode": "concode",
        "mercury": "mercury",
        # === CORRECTED MAPPINGS ===
        "humaneval_plus": "humanevalplus",
        "instructhumaneval": "instruct-humaneval",
        "mbpp_plus": "mbppplus",
        "apps": "apps-introductory",
        "ds1000": "ds1000-all-completion",
        # === MULTI-LANGUAGE TASKS ===
        "multiple_py": "multiple-py",
        "multiple_js": "multiple-js",
        "multiple_java": "multiple-java",
        "multiple_cpp": "multiple-cljcpp",
        "multiple_rs": "multiple-rs",
        "multiple_go": "multiple-go",
        # === CODE-TO-TEXT TASKS ===
        "codexglue_code_to_text_python": "codexglue_code_to_text-python",
        "codexglue_code_to_text_go": "codexglue_code_to_text-go",
        "codexglue_code_to_text_java": "codexglue_code_to_text-java",
        "codexglue_code_to_text_javascript": "codexglue_code_to_text-javascript",
        "codexglue_code_to_text_php": "codexglue_code_to_text-php",
        "codexglue_code_to_text_ruby": "codexglue_code_to_text-ruby",
        # === FIXED PROBLEMATIC MAPPINGS ===
        "recode": "perturbed-humaneval-natgen-num_seeds_1",
        "humanevalpack": None,  # ❌ REMOVED - no simple mapping exists, only complex variants
    }

    def __init__(self):
        """Initialize BigCode task loader."""
        self._bigcode_available = self._check_bigcode_available()
        self._task_cache = {}

    def _check_bigcode_available(self) -> bool:
        """Check if bigcode-evaluation-harness is available."""
        try:
            import bigcode_eval

            return True
        except ImportError:
            logger.warning("bigcode-evaluation-harness not installed")
            return False

    def is_bigcode_task(self, task_name: str) -> bool:
        """Check if a task is a BigCode task."""
        return task_name in self.TASK_MAPPING

    def load_task(self, task_name: str, limit: Optional[int] = None) -> "BigCodeTask":
        """
        Load a BigCode task.

        Args:
            task_name: Name of the task (our naming convention)
            limit: Optional limit on number of samples

        Returns:
            BigCodeTask object
        """
        if not self._bigcode_available:
            raise ImportError("bigcode-evaluation-harness not installed. Run: pip install bigcode-evaluation-harness")

        if task_name not in self.TASK_MAPPING:
            raise ValueError(f"Unknown BigCode task: {task_name}")

        bigcode_task_name = self.TASK_MAPPING[task_name]

        # Handle removed tasks with None mapping
        if bigcode_task_name is None:
            raise ValueError(f"Task '{task_name}' has been removed - no BigCode mapping available")

        # Check cache
        cache_key = f"{task_name}:{limit}"
        if cache_key in self._task_cache:
            return self._task_cache[cache_key]

        # Create task object
        task = BigCodeTask(task_name, bigcode_task_name, limit)
        self._task_cache[cache_key] = task

        return task


class BigCodeTask:
    """Represents a BigCode evaluation task."""

    def __init__(self, task_name: str, bigcode_task_name: str, limit: Optional[int] = None):
        """
        Initialize BigCode task.

        Args:
            task_name: Our task name
            bigcode_task_name: BigCode's task name
            limit: Optional limit on samples
        """
        self.task_name = task_name
        self.bigcode_task_name = bigcode_task_name
        self.limit = limit
        self._limit = limit  # Store as private attribute too
        self._data = None
        self._task_obj = None
        self._load_data()

    def _load_data(self):
        """Load task data from BigCode."""
        try:
            # Import BigCode modules
            from bigcode_eval.tasks import get_task

            # Get the task
            task = get_task(self.bigcode_task_name)
            self._task_obj = task

            # Get dataset - BigCode uses get_dataset() method
            dataset = task.get_dataset()

            # Convert to list if needed
            if hasattr(dataset, "__iter__"):
                dataset = list(dataset)

            # Apply limit if specified
            if self.limit:
                dataset = dataset[: self.limit]

            self._data = dataset

        except Exception as e:
            logger.error(f"Failed to load BigCode task {self.bigcode_task_name}: {e}")
            # Fallback to loading from files if available
            self._load_from_files()

    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # BigCode tasks don't have separate validation sets

    def has_test_docs(self) -> bool:
        """Check if task has test documents."""
        return True  # All samples are considered test docs

    def test_docs(self) -> List[Dict[str, Any]]:
        """Get test documents."""
        return self.get_samples()

    def validation_docs(self) -> List[Dict[str, Any]]:
        """Get validation documents."""
        return []  # No separate validation set

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text prompt."""
        # Handle different BigCode formats
        if "prompt" in doc:
            return doc["prompt"]
        if "text" in doc:
            return doc["text"]
        if "question" in doc:
            return doc["question"]
        if "problem" in doc:
            return doc["problem"]
        # Fallback - try to use task object if available
        if self._task_obj and hasattr(self._task_obj, "get_prompt"):
            return self._task_obj.get_prompt(doc)
        return str(doc)

    def _load_from_files(self):
        """Load task data from local files as fallback."""
        # Try to load from standard locations
        data_paths = [
            f"~/.cache/bigcode_eval/{self.bigcode_task_name}",
            f"data/{self.bigcode_task_name}",
            f"bigcode_eval/tasks/{self.bigcode_task_name}",
        ]

        for path in data_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                self._load_from_path(expanded_path)
                return

        # If no data found, raise error
        raise ValueError(f"No data found for task {self.task_name}. Please provide valid benchmark data.")

    def _load_from_path(self, path: str):
        """Load data from a specific path."""
        data = []

        # Look for JSON/JSONL files
        for file in Path(path).glob("*.json*"):
            with open(file) as f:
                if file.suffix == ".jsonl":
                    for line in f:
                        data.append(json.loads(line))
                else:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)

        if self.limit:
            data = data[: self.limit]

        self._data = data

    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all samples from the task."""
        return self._data if self._data else []

    def __len__(self):
        """Get number of samples."""
        return len(self._data) if self._data else 0

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.get_samples())


class BigCodeEvaluator:
    """Evaluates model outputs on BigCode benchmarks."""

    def __init__(self, docker_executor=None):
        """
        Initialize evaluator.

        Args:
            docker_executor: Optional Docker executor for secure code execution
        """
        self.docker_executor = docker_executor

    def evaluate(self, task: BigCodeTask, generations: List[str], k_values: List[int] = [1, 10, 100]) -> Dict[str, Any]:
        """
        Evaluate generations on a BigCode task.

        Args:
            task: BigCodeTask object
            generations: List of generated code solutions
            k_values: k values for pass@k metric

        Returns:
            Evaluation results dict
        """
        results = {
            "task": task.task_name,
            "num_samples": len(task),
            "num_generations": len(generations),
            "pass_at_k": {},
        }

        # For code generation tasks, we need to execute and test
        if self._is_code_execution_task(task.task_name):
            results["execution_results"] = self._evaluate_code_execution(task, generations)

            # Calculate pass@k
            for k in k_values:
                if k <= len(generations):
                    pass_rate = self._calculate_pass_at_k(results["execution_results"], k)
                    results["pass_at_k"][f"pass@{k}"] = pass_rate

        else:
            # For non-execution tasks (e.g., code-to-text), use BLEU or other metrics
            results["bleu_scores"] = self._evaluate_text_generation(task, generations)

        return results

    def _is_code_execution_task(self, task_name: str) -> bool:
        """Check if task requires code execution."""
        non_execution_tasks = {
            "codexglue_code_to_text",
            "codexglue_code_to_text_python",
            "codexglue_code_to_text_go",
            "codexglue_code_to_text_ruby",
            "codexglue_code_to_text_java",
            "codexglue_code_to_text_javascript",
            "codexglue_code_to_text_php",
        }
        return task_name not in non_execution_tasks

    def _evaluate_code_execution(self, task: BigCodeTask, generations: List[str]) -> List[Dict]:
        """Evaluate code by executing it."""
        results = []

        for i, sample in enumerate(task.get_samples()):
            sample_results = []

            for j, generation in enumerate(generations[i] if i < len(generations) else []):
                result = self._execute_and_test(sample, generation, task.task_name)
                sample_results.append(result)

            results.append({"sample_id": i, "results": sample_results})

        return results

    def _execute_and_test(self, sample: Dict, generation: str, task_name: str) -> Dict:
        """Execute generated code and run tests."""
        if self.docker_executor:
            # Use Docker for secure execution
            return self._execute_in_docker(sample, generation, task_name)
        # Fallback to subprocess (less secure)
        return self._execute_in_subprocess(sample, generation, task_name)

    def _execute_in_docker(self, sample: Dict, generation: str, task_name: str) -> Dict:
        """Execute code in Docker container."""
        # TODO: Implement Docker execution
        logger.warning("Docker execution not yet implemented, using subprocess")
        return self._execute_in_subprocess(sample, generation, task_name)

    def _execute_in_subprocess(self, sample: Dict, generation: str, task_name: str) -> Dict:
        """Execute code in subprocess (less secure)."""
        result = {"passed": False, "error": None, "output": None}

        try:
            # Create test script
            test_script = self._create_test_script(sample, generation, task_name)

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_script)
                temp_path = f.name

            try:
                # Execute
                proc = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=10)

                if proc.returncode == 0:
                    result["passed"] = True
                    result["output"] = proc.stdout
                    logger.debug(f"✅ Code execution PASSED. Output: {proc.stdout[:200]}")
                else:
                    result["error"] = proc.stderr or proc.stdout
                    logger.debug(f"❌ Code execution FAILED. Error: {result['error'][:500]}")

            finally:
                # Clean up
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            result["error"] = "Timeout"
        except Exception as e:
            result["error"] = str(e)

        return result

    def _create_test_script(self, sample: Dict, generation: str, task_name: str) -> str:
        """Create a test script for the sample."""
        if "humaneval" in task_name:
            script = self._create_humaneval_test_script(sample, generation)
        elif "mbpp" in task_name:
            script = self._create_mbpp_test_script(sample, generation)
        elif "apps" in task_name:
            script = self._create_apps_test_script(sample, generation)
        else:
            # Default format
            script = self._create_humaneval_test_script(sample, generation)

        logger.debug(f"📝 Test script for {task_name}:\n{script}\n")
        return script

    def _create_humaneval_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for HumanEval format."""
        entry_point = sample.get("entry_point", "solution")
        test_code = sample.get("test", "")
        prompt = sample.get("prompt", "")

        # The prompt contains the function signature, and generation should be the function body
        # We need to combine them properly
        script = f"""
{prompt}{generation}

{test_code}

if __name__ == "__main__":
    check({entry_point})
    print("All tests passed!")
"""
        return script

    def _create_mbpp_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for MBPP format."""
        test_imports = sample.get("test_imports", [])
        test_list = sample.get("test_list", [])

        # Fix function name mismatch before creating test script
        fixed_generation = self._fix_function_name_mismatch(generation, test_list)

        imports = "\n".join(test_imports)
        tests = "\n    ".join(test_list)

        script = f"""
{imports}

{fixed_generation}

if __name__ == "__main__":
    {tests}
    print("All tests passed!")
"""
        return script

    def _create_apps_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for APPS format."""
        # APPS has input/output pairs
        io_data = json.loads(sample.get("input_output", "{}"))
        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])

        tests = []
        for inp, out in zip(inputs, outputs):
            tests.append(f"assert str(solution({inp})) == '{out}'")

        test_code = "\n    ".join(tests)

        script = f"""
{generation}

if __name__ == "__main__":
    {test_code}
    print("All tests passed!")
"""
        return script

    def _fix_function_name_mismatch(self, code: str, test_list: List[str]) -> str:
        """
        Fix function name mismatches between generated code and test cases.

        Uses wrapper function approach for robustness across different code structures.

        Args:
            code: Generated code that may have wrong function name
            test_list: List of test assertions that specify expected function name

        Returns:
            Fixed code with wrapper function if needed
        """
        import re

        if not test_list or not code.strip():
            return code

        # Extract expected function name from test assertions
        expected_name = None
        # Built-in functions to skip when looking for the target function
        builtin_functions = {
            "set",
            "len",
            "str",
            "int",
            "float",
            "list",
            "tuple",
            "dict",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "reversed",
        }

        for test in test_list:
            # Find all function calls in assert statements
            function_calls = re.findall(r"(\w+)\s*\(", test)

            for func_name in function_calls:
                # Skip built-in functions and common test functions
                if func_name not in builtin_functions and func_name not in {
                    "assert",
                    "assertEqual",
                    "assertTrue",
                    "assertFalse",
                }:
                    expected_name = func_name
                    break

            if expected_name:
                break

        if not expected_name:
            return code  # No function name found in tests

        # Extract actual function name from generated code
        actual_name = None
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        if func_match:
            actual_name = func_match.group(1)

        if not actual_name:
            return code  # No function definition found

        if actual_name == expected_name:
            return code  # Names already match

        logger.debug(f"🔧 Function name mismatch detected: {actual_name} → {expected_name}")
        logger.debug("   Adding wrapper function for compatibility")

        # Add wrapper function to bridge the name gap
        wrapper = f"""
# Wrapper function for test compatibility
def {expected_name}(*args, **kwargs):
    return {actual_name}(*args, **kwargs)
"""

        return code + wrapper

    def _calculate_pass_at_k(self, execution_results: List[Dict], k: int) -> float:
        """Calculate pass@k metric."""
        total_passed = 0
        total_samples = len(execution_results)

        for result in execution_results:
            sample_results = result["results"][:k]
            if any(r["passed"] for r in sample_results):
                total_passed += 1

        return total_passed / total_samples if total_samples > 0 else 0.0

    def _evaluate_text_generation(self, task: BigCodeTask, generations: List[str]) -> List[float]:
        """Evaluate text generation tasks (e.g., code-to-text)."""
        # TODO: Implement BLEU scoring
        logger.warning("Text generation evaluation not yet implemented")
        return [0.0] * len(generations)


# Main interface for BigCode integration
_loader = None
_evaluator = None


def get_bigcode_loader() -> BigCodeTaskLoader:
    """Get the global BigCode task loader."""
    global _loader
    if _loader is None:
        _loader = BigCodeTaskLoader()
    return _loader


def get_bigcode_evaluator(docker_executor=None) -> BigCodeEvaluator:
    """Get the global BigCode evaluator."""
    global _evaluator
    if _evaluator is None:
        _evaluator = BigCodeEvaluator(docker_executor)
    return _evaluator


def is_bigcode_task(task_name: str) -> bool:
    """Check if a task is from BigCode."""
    return get_bigcode_loader().is_bigcode_task(task_name)


def load_bigcode_task(task_name: str, limit: Optional[int] = None) -> BigCodeTask:
    """Load a BigCode task."""
    return get_bigcode_loader().load_task(task_name, limit)


def evaluate_bigcode_task(task: BigCodeTask, generations: List[str], docker_executor=None) -> Dict[str, Any]:
    """Evaluate generations on a BigCode task."""
    evaluator = get_bigcode_evaluator(docker_executor)
    return evaluator.evaluate(task, generations)
