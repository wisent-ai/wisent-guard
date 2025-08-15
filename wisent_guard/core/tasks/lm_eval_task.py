"""
LM-Evaluation-Harness task wrapper for task-agnostic architecture.
"""

from typing import Any, Dict, List, Optional

from ..benchmark_extractors import BenchmarkExtractor, get_extractor
from ..task_interface import TaskInterface


class LMEvalTask(TaskInterface):
    """Wrapper for lm-evaluation-harness tasks."""

    def __init__(self, task_name: str, description: str, categories: List[str]):
        self.task_name = task_name
        self._description = description
        self._categories = categories
        self._extractor = get_extractor(task_name)

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data directly from lm-eval without Model dependency."""
        try:
            # Load data directly from lm-eval without creating a Model instance
            from lm_eval.tasks import get_task_dict

            # Get task directly from lm-eval
            task_dict = get_task_dict([self.task_name])
            if self.task_name not in task_dict:
                print(f"Warning: Task '{self.task_name}' not found in lm-eval")
                return []

            task = task_dict[self.task_name]

            # Get the task's test documents
            docs = []
            if hasattr(task, "test_docs"):
                # For lm-eval versions with test_docs method
                docs = list(task.test_docs())
            elif hasattr(task, "dataset"):
                # For newer lm-eval versions
                dataset = task.dataset
                if hasattr(dataset, "test"):
                    docs = list(dataset.test)
                elif hasattr(dataset, "validation"):
                    docs = list(dataset.validation)
                else:
                    # Fallback to the main dataset
                    docs = list(dataset)

            # Ensure docs are in dictionary format
            processed_docs = []
            for doc in docs:
                if isinstance(doc, dict):
                    processed_docs.append(doc)
                elif isinstance(doc, str):
                    # Handle string documents by wrapping them
                    processed_docs.append({"text": doc})
                else:
                    # Try to convert to dict if possible
                    try:
                        processed_docs.append(dict(doc))
                    except:
                        processed_docs.append({"data": str(doc)})

            docs = processed_docs

            # Apply limit if specified
            if limit and len(docs) > limit:
                docs = docs[:limit]

            return docs

        except Exception as e:
            print(f"Warning: Could not load lm-eval task '{self.task_name}': {e}")
            return []

    def get_extractor(self) -> BenchmarkExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor

    def get_name(self) -> str:
        """Get the task name."""
        return self.task_name

    def get_description(self) -> str:
        """Get the task description."""
        return self._description

    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return self._categories


class MBPPTask(LMEvalTask):
    """MBPP task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mbpp",
            description="MBPP: Mostly Basic Python Problems coding benchmark",
            categories=["coding", "reasoning", "python"],
        )


class HumanEvalTask(LMEvalTask):
    """HumanEval task implementation."""

    def __init__(self):
        super().__init__(
            task_name="humaneval",
            description="HumanEval: Human Evaluation of Python coding problems",
            categories=["coding", "reasoning", "python"],
        )


class MBPPPlusTask(LMEvalTask):
    """MBPP Plus task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mbpp_plus",
            description="MBPP Plus: Extended version of MBPP with additional test cases",
            categories=["coding", "reasoning", "python"],
        )


class GSM8KTask(LMEvalTask):
    """GSM8K task implementation."""

    def __init__(self):
        super().__init__(
            task_name="gsm8k",
            description="GSM8K: Grade School Math 8K problems",
            categories=["mathematics", "reasoning", "arithmetic"],
        )


class TruthfulQATask(LMEvalTask):
    """TruthfulQA task implementation."""

    def __init__(self):
        super().__init__(
            task_name="truthfulqa_mc1",
            description="TruthfulQA: Truthfulness evaluation benchmark",
            categories=["hallucination", "general-knowledge", "reasoning"],
        )

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load TruthfulQA data, which only has validation split."""
        try:
            from lm_eval.tasks import get_task_dict

            # Get task directly from lm-eval
            task_dict = get_task_dict([self.task_name])
            if self.task_name not in task_dict:
                print(f"Warning: Task '{self.task_name}' not found in lm-eval")
                return []

            task = task_dict[self.task_name]

            # TruthfulQA only has validation split, access it directly
            docs = []
            if hasattr(task, "dataset") and "validation" in task.dataset:
                validation_data = task.dataset["validation"]
                docs = list(validation_data)

            # Apply limit if specified
            if limit and len(docs) > limit:
                docs = docs[:limit]

            return docs

        except Exception as e:
            print(f"Warning: Could not load TruthfulQA task '{self.task_name}': {e}")
            import traceback

            traceback.print_exc()
            return []


class MMLUTask(LMEvalTask):
    """MMLU task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mmlu",
            description="MMLU: Massive Multitask Language Understanding",
            categories=["general-knowledge", "science", "reasoning"],
        )


# === CODING TASKS ===


class InstructHumanEvalTask(LMEvalTask):
    """InstructHumanEval task implementation."""

    def __init__(self):
        super().__init__(
            task_name="instructhumaneval",
            description="InstructHumanEval: Instruction-following HumanEval benchmark",
            categories=["coding", "reasoning", "python", "instruction-following"],
        )


class HumanEvalPlusTask(LMEvalTask):
    """HumanEval Plus task implementation."""

    def __init__(self):
        super().__init__(
            task_name="humaneval_plus",
            description="HumanEval Plus: Extended HumanEval with more tests",
            categories=["coding", "reasoning", "python"],
        )


class ConalaTask(LMEvalTask):
    """Conala task implementation."""

    def __init__(self):
        super().__init__(
            task_name="conala",
            description="Conala: Code generation from natural language",
            categories=["coding", "reasoning", "python", "nl2code"],
        )


class ConcodeTask(LMEvalTask):
    """Concode task implementation."""

    def __init__(self):
        super().__init__(
            task_name="concode",
            description="Concode: Code completion benchmark",
            categories=["coding", "reasoning", "completion"],
        )


class MercuryTask(LMEvalTask):
    """Mercury task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mercury",
            description="Mercury: Code generation benchmark",
            categories=["coding", "reasoning"],
        )


class AppsTask(LMEvalTask):
    """APPS task implementation."""

    def __init__(self):
        super().__init__(
            task_name="apps",
            description="APPS: Automated Programming Problems Synthesis",
            categories=["coding", "reasoning", "python", "competitive"],
        )


class DS1000Task(LMEvalTask):
    """DS1000 task implementation."""

    def __init__(self):
        super().__init__(
            task_name="ds1000",
            description="DS1000: Data Science coding tasks",
            categories=["coding", "reasoning", "python", "data-science"],
        )


class MultiplePyTask(LMEvalTask):
    """Multiple-Py task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_py",
            description="Multiple-Py: Multi-language Python tasks",
            categories=["coding", "reasoning", "python", "multi-language"],
        )


class MultipleJsTask(LMEvalTask):
    """Multiple-JS task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_js",
            description="Multiple-JS: Multi-language JavaScript tasks",
            categories=["coding", "reasoning", "javascript", "multi-language"],
        )


class MultipleJavaTask(LMEvalTask):
    """Multiple-Java task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_java",
            description="Multiple-Java: Multi-language Java tasks",
            categories=["coding", "reasoning", "java", "multi-language"],
        )


class MultipleCppTask(LMEvalTask):
    """Multiple-Cpp task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_cpp",
            description="Multiple-Cpp: Multi-language C++ tasks",
            categories=["coding", "reasoning", "cpp", "multi-language"],
        )


class MultipleRsTask(LMEvalTask):
    """Multiple-Rs task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_rs",
            description="Multiple-Rs: Multi-language Rust tasks",
            categories=["coding", "reasoning", "rust", "multi-language"],
        )


class MultipleGoTask(LMEvalTask):
    """Multiple-Go task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_go",
            description="Multiple-Go: Multi-language Go tasks",
            categories=["coding", "reasoning", "go", "multi-language"],
        )


class CodexglueCodeToTextPythonTask(LMEvalTask):
    """CodexGlue Code-to-Text Python task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_python",
            description="CodexGlue Code-to-Text Python: Python code summarization",
            categories=["coding", "reasoning", "python", "code-to-text"],
        )


class CodexglueCodeToTextGoTask(LMEvalTask):
    """CodexGlue Code-to-Text Go task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_go",
            description="CodexGlue Code-to-Text Go: Go code summarization",
            categories=["coding", "reasoning", "go", "code-to-text"],
        )


class CodexglueCodeToTextRubyTask(LMEvalTask):
    """CodexGlue Code-to-Text Ruby task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_ruby",
            description="CodexGlue Code-to-Text Ruby: Ruby code summarization",
            categories=["coding", "reasoning", "ruby", "code-to-text"],
        )


class CodexglueCodeToTextJavaTask(LMEvalTask):
    """CodexGlue Code-to-Text Java task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_java",
            description="CodexGlue Code-to-Text Java: Java code summarization",
            categories=["coding", "reasoning", "java", "code-to-text"],
        )


class CodexglueCodeToTextJavascriptTask(LMEvalTask):
    """CodexGlue Code-to-Text JavaScript task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_javascript",
            description="CodexGlue Code-to-Text JavaScript: JavaScript code summarization",
            categories=["coding", "reasoning", "javascript", "code-to-text"],
        )


class CodexglueCodeToTextPhpTask(LMEvalTask):
    """CodexGlue Code-to-Text PHP task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_php",
            description="CodexGlue Code-to-Text PHP: PHP code summarization",
            categories=["coding", "reasoning", "php", "code-to-text"],
        )


class RecodeTask(LMEvalTask):
    """Recode task implementation."""

    def __init__(self):
        super().__init__(
            task_name="recode",
            description="Recode: Perturbed HumanEval natural generation",
            categories=["coding", "reasoning", "python", "perturbation"],
        )


class Squad2Task(LMEvalTask):
    """SQuAD2 task implementation."""

    def __init__(self):
        super().__init__(
            task_name="squadv2",
            description="SQuAD2: Stanford Question Answering Dataset 2.0",
            categories=["reading-comprehension", "qa", "natural-language"],
        )

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load SQuAD2 data, which only has validation split."""
        try:
            from lm_eval.tasks import get_task_dict

            # Get task directly from lm-eval
            task_dict = get_task_dict([self.task_name])
            if self.task_name not in task_dict:
                print(f"Warning: Task '{self.task_name}' not found in lm-eval")
                return []

            task = task_dict[self.task_name]

            # SQuAD2 only has validation split, access it directly
            docs = []
            if hasattr(task, "dataset") and "validation" in task.dataset:
                validation_data = task.dataset["validation"]
                docs = list(validation_data)

            # Apply limit if specified
            if limit and len(docs) > limit:
                docs = docs[:limit]

            return docs

        except Exception as e:
            print(f"Warning: Could not load SQuAD2 task '{self.task_name}': {e}")
            import traceback

            traceback.print_exc()
            return []
