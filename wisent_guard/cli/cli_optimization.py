# cli_optimization.py
from typing import Any, Dict, List, Optional

from wisent_guard.cli_workflows.optimize import (
    run_interactive_optimization,
    run_smart_optimization,
)

def _extract_test_pairs_for_opt(
    test_docs,
    task_data,
    from_files_or_group: bool,
    verbose: bool,
):
    test_qa_pairs = []
    for doc in test_docs:
        try:
            if from_files_or_group:
                test_qa_pairs.append(
                    {
                        "question": doc["question"],
                        "formatted_question": doc["question"],
                        "correct_answer": doc["correct_answer"],
                    }
                )
            else:
                raw_q = doc.get("question", str(doc))
                if hasattr(task_data, "doc_to_text"):
                    formatted = task_data.doc_to_text(doc)
                else:
                    formatted = raw_q
                test_qa_pairs.append(
                    {"question": raw_q, "formatted_question": formatted, "correct_answer": ""}
                )
        except Exception:
            continue
    return test_qa_pairs

def maybe_optimize_hparams(
    optimize: bool,
    mode: str,
    model,
    collector,
    task_name: str,
    model_name: str,
    contrastive_pairs,
    test_qa_pairs_source,
    task_data,
    layers,
    token_aggregation: str,
    classifier_type: str,
    detection_threshold: float,
    optimize_layers: str,
    optimize_metric: str,
    optimize_max_combinations: int,
    max_new_tokens: int,
    device: Optional[str],
    group_task_qa_format: bool,
    verbose: bool,
) -> Optional[Dict[str, Any]]:
    if not optimize:
        return None

    if mode == "interactive":
        questions = []
        for doc in test_qa_pairs_source[:2]:
            if isinstance(doc, dict) and "question" in doc:
                questions.append(doc["question"])
        if not questions:
            if verbose:
                print("⚠️ No test questions available for interactive optimization")
            return None
        return run_interactive_optimization(
            model=model,
            questions=questions,
            training_pairs=contrastive_pairs,
            max_new_tokens=max_new_tokens,
            max_combinations=optimize_max_combinations,
            verbose=verbose,
        )

    # smart optimization
    test_qa_pairs = _extract_test_pairs_for_opt(
        test_qa_pairs_source, task_data, from_files_or_group=group_task_qa_format, verbose=verbose
    )
    return run_smart_optimization(
        model=model,
        collector=collector,
        contrastive_pairs=contrastive_pairs,
        test_qa_pairs=test_qa_pairs,
        task_name=task_name,
        model_name=model_name,
        limit=None,
        ground_truth_method=mode,
        max_new_tokens=max_new_tokens,
        device=device,
        verbose=verbose,
        optimize_layers=optimize_layers,
    )
