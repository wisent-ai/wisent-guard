from typing import TypedDict, Mapping

from lm_eval.api.task import ConfigurableTask
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

class LoadDataResult(TypedDict):
    """
    Structured output from a data loader used for training and evaluation.

    attributes:
        train_qa_pairs:
            The training set of question-answer pairs.
        test_qa_pairs:
            The test set of question-answer pairs.
        task_type:
            The high-level task category (e.g., "classification").
        lm_task_data:
            Tasks in the 'lm_eval' repository format, if applicable.

            When training/evaluating steering vectors with 'lm_eval', that
            library is responsible for downloading and preprocessing the data,
            and it provides the evaluation function that compares the steered
            model to the baseline, see: https://github.com/EleutherAI/lm-evaluation-harness.
            For custom data loaders, this is 'None'.
    """
    train_qa_pairs: ContrastivePairSet
    test_qa_pairs: ContrastivePairSet
    task_type: str
    lm_task_data: Mapping[str, ConfigurableTask] | ConfigurableTask | None
