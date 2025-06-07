"""
Evaluation functionality for guard effectiveness assessment.
"""

import json
import os
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def evaluate_guard(
    guard_pipeline,
    test_hidden_states: List[Any],
    test_labels: List[int],
    task_name: str,
    model_name: str,
    layer: int
) -> Dict[str, Any]:
    """
    Evaluate guard effectiveness on test data.
    
    Args:
        guard_pipeline: Trained GuardPipeline instance
        test_hidden_states: List of hidden states from test set
        test_labels: List of true labels (0=good, 1=bad)
        task_name: Name of the benchmark task
        model_name: Name of the language model
        layer: Layer used for activation extraction
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating guard on {len(test_hidden_states)} test samples")
    
    # Get predictions from guard
    predictions = guard_pipeline.batch_predict(test_hidden_states)
    
    # Calculate confusion matrix components
    tp = sum(1 for true, pred in zip(test_labels, predictions) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(test_labels, predictions) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(test_labels, predictions) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(test_labels, predictions) if true == 1 and pred == 0)
    
    total_samples = len(test_labels)
    total_bad = sum(test_labels)
    total_good = total_samples - total_bad
    
    # Calculate metrics
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Guard-specific metrics
    detection_rate = tp / total_bad if total_bad > 0 else 0.0
    false_positive_rate = fp / total_good if total_good > 0 else 0.0
    
    # Create comprehensive results
    results = {
        'task_name': task_name,
        'model_name': model_name,
        'layer': layer,
        'timestamp': datetime.now().isoformat(),
        
        # Basic metrics
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        
        # Confusion matrix
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        
        # Sample counts
        'total_samples': total_samples,
        'total_bad_samples': total_bad,
        'total_good_samples': total_good,
        
        # Guard effectiveness
        'bad_samples_detected': tp,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        
        # Performance summary
        'guard_effectiveness': f"{tp}/{total_bad} bad samples caught ({detection_rate:.1%})",
        'false_alarm_rate': f"{fp}/{total_good} false alarms ({false_positive_rate:.1%})"
    }
    
    logger.info(f"Guard evaluation complete:")
    logger.info(f"  Detection Rate: {detection_rate:.1%} ({tp}/{total_bad})")
    logger.info(f"  False Positive Rate: {false_positive_rate:.1%} ({fp}/{total_good})")
    logger.info(f"  F1 Score: {f1_score:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    return results


def evaluate_lm_harness_task(
    task,
    test_docs: List[Dict[str, Any]], 
    test_responses: List[str],
    test_references: List[str]
) -> Dict[str, Any]:
    """
    Evaluate using lm-evaluation-harness task evaluation.
    
    Args:
        task: Task object from lm_eval
        test_docs: Test documents
        test_responses: Generated responses
        test_references: Reference answers
        
    Returns:
        Dictionary with lm-harness evaluation results
    """
    # Extract task name
    from .labeler import get_task_name
    task_name = get_task_name(task)
    
    logger.info(f"Evaluating {len(test_responses)} responses using lm-harness for {task_name}")
    
    try:
        # Try to use task's evaluation method
        if hasattr(task, 'process_results'):
            # Prepare results in lm-eval format
            eval_results = []
            
            for doc, response in zip(test_docs, test_responses):
                # Different tasks expect different result formats
                result_item = {"response": response}
                
                try:
                    processed = task.process_results(doc, [result_item])
                    eval_results.append(processed)
                except Exception as e:
                    logger.warning(f"Failed to process result: {e}")
                    eval_results.append({"accuracy": 0.0})
            
            # Aggregate results
            if eval_results:
                # Try to extract common metrics
                metrics = {}
                
                for metric_name in ['acc', 'accuracy', 'exact_match', 'f1', 'rouge']:
                    values = []
                    for result in eval_results:
                        if isinstance(result, dict) and metric_name in result:
                            values.append(float(result[metric_name]))
                    
                    if values:
                        metrics[metric_name] = sum(values) / len(values)
                
                # If no metrics found, default to accuracy
                if not metrics:
                    metrics['accuracy'] = sum(1 for r in eval_results if r.get('accuracy', 0) > 0) / len(eval_results)
                
            else:
                metrics = {'accuracy': 0.0}
        
        else:
            # Fallback: simple string matching
            logger.info("Using fallback string matching evaluation")
            correct = 0
            
            for response, reference in zip(test_responses, test_references):
                if response.strip().lower() == reference.strip().lower():
                    correct += 1
            
            metrics = {'accuracy': correct / len(test_responses) if test_responses else 0.0}
        
        results = {
            'task_name': task_name,
            'total_samples': len(test_responses),
            'lm_harness_metrics': metrics,
            'primary_metric': metrics.get('accuracy', 0.0)
        }
        
        logger.info(f"LM-harness evaluation complete. Primary metric: {results['primary_metric']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to evaluate with lm-harness: {e}")
        return {
            'task_name': task_name,
            'total_samples': len(test_responses),
            'lm_harness_metrics': {'accuracy': 0.0},
            'primary_metric': 0.0,
            'error': str(e)
        }


def save_results_json(results: Dict[str, Any], output_dir: str, filename: Optional[str] = None) -> str:
    """
    Save evaluation results as JSON.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        filename: Optional filename (auto-generated if None)
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        task_name = results.get('task_name', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"guard_evaluation_{task_name}_{timestamp}.json"
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved evaluation results to {output_path}")
    return output_path


def save_results_csv(all_results: List[Dict[str, Any]], output_dir: str, filename: str = "guard_evaluation_summary.csv") -> str:
    """
    Save aggregated results as CSV.
    
    Args:
        all_results: List of result dictionaries from multiple tasks
        output_dir: Output directory
        filename: CSV filename
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten results for CSV
    flattened_results = []
    
    for result in all_results:
        flat_result = {
            'task_name': result.get('task_name', ''),
            'model_name': result.get('model_name', ''),
            'layer': result.get('layer', ''),
            'timestamp': result.get('timestamp', ''),
            'accuracy': result.get('accuracy', 0.0),
            'precision': result.get('precision', 0.0),
            'recall': result.get('recall', 0.0),
            'f1_score': result.get('f1_score', 0.0),
            'detection_rate': result.get('detection_rate', 0.0),
            'false_positive_rate': result.get('false_positive_rate', 0.0),
            'total_samples': result.get('total_samples', 0),
            'bad_samples_detected': result.get('bad_samples_detected', 0),
            'total_bad_samples': result.get('total_bad_samples', 0),
            'lm_harness_accuracy': result.get('lm_harness_metrics', {}).get('accuracy', 0.0) if 'lm_harness_metrics' in result else 0.0
        }
        flattened_results.append(flat_result)
    
    # Create DataFrame and save
    df = pd.DataFrame(flattened_results)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved aggregated results to {output_path}")
    return output_path


def create_evaluation_report(all_results: List[Dict[str, Any]], output_dir: str) -> str:
    """
    Create a comprehensive evaluation report.
    
    Args:
        all_results: List of evaluation results
        output_dir: Output directory
        
    Returns:
        Path to report file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"guard_evaluation_report_{timestamp}.md")
    
    with open(report_path, 'w') as f:
        f.write("# Wisent-Guard Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## Summary\n\n")
        f.write("| Task | Model | Layer | Detection Rate | F1 Score | Accuracy |\n")
        f.write("|------|-------|-------|----------------|----------|----------|\n")
        
        for result in all_results:
            task = result.get('task_name', 'N/A')
            model = result.get('model_name', 'N/A').split('/')[-1]  # Short model name
            layer = result.get('layer', 'N/A')
            detection_rate = result.get('detection_rate', 0.0)
            f1 = result.get('f1_score', 0.0)
            accuracy = result.get('accuracy', 0.0)
            
            f.write(f"| {task} | {model} | {layer} | {detection_rate:.1%} | {f1:.3f} | {accuracy:.3f} |\n")
        
        # Detailed results
        f.write("\n## Detailed Results\n\n")
        
        for result in all_results:
            task = result.get('task_name', 'Unknown')
            f.write(f"### {task}\n\n")
            
            f.write(f"- **Model**: {result.get('model_name', 'N/A')}\n")
            f.write(f"- **Layer**: {result.get('layer', 'N/A')}\n")
            f.write(f"- **Total Samples**: {result.get('total_samples', 0)}\n")
            f.write(f"- **Bad Samples**: {result.get('total_bad_samples', 0)}\n")
            f.write(f"- **Detection Rate**: {result.get('detection_rate', 0.0):.1%}\n")
            f.write(f"- **False Positive Rate**: {result.get('false_positive_rate', 0.0):.1%}\n")
            f.write(f"- **F1 Score**: {result.get('f1_score', 0.0):.4f}\n")
            f.write(f"- **Accuracy**: {result.get('accuracy', 0.0):.4f}\n")
            
            if 'lm_harness_metrics' in result:
                f.write(f"- **LM-Harness Accuracy**: {result['lm_harness_metrics'].get('accuracy', 0.0):.4f}\n")
            
            f.write("\n")
        
        # Footer
        f.write("---\n")
        f.write("*Generated by wisent-guard wg_harness evaluation pipeline*\n")
    
    logger.info(f"Created evaluation report: {report_path}")
    return report_path


def calculate_aggregate_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate aggregate metrics across all tasks.
    
    Args:
        all_results: List of evaluation results
        
    Returns:
        Dictionary with aggregate metrics
    """
    if not all_results:
        return {}
    
    # Extract numeric metrics
    detection_rates = [r.get('detection_rate', 0.0) for r in all_results]
    f1_scores = [r.get('f1_score', 0.0) for r in all_results]
    accuracies = [r.get('accuracy', 0.0) for r in all_results]
    
    total_bad_detected = sum(r.get('bad_samples_detected', 0) for r in all_results)
    total_bad_samples = sum(r.get('total_bad_samples', 0) for r in all_results)
    
    aggregate = {
        'mean_detection_rate': sum(detection_rates) / len(detection_rates),
        'mean_f1_score': sum(f1_scores) / len(f1_scores),
        'mean_accuracy': sum(accuracies) / len(accuracies),
        'overall_detection_rate': total_bad_detected / total_bad_samples if total_bad_samples > 0 else 0.0,
        'total_tasks': len(all_results),
        'total_bad_samples': total_bad_samples,
        'total_bad_detected': total_bad_detected
    }
    
    logger.info(f"Aggregate metrics: Detection rate {aggregate['overall_detection_rate']:.1%}, F1 {aggregate['mean_f1_score']:.3f}")
    
    return aggregate 