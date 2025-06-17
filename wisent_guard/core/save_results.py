"""
Functions for saving wisent-guard evaluation results in various formats.
"""

import os
import json
import csv
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_results_json(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")


def save_results_csv(results: Dict[str, Any], output_path: str) -> None:
    """Save results to CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Flatten results for CSV
        rows = []
        for task_name, task_results in results.items():
            if isinstance(task_results, dict):
                row = {"task": task_name}
                row.update(task_results)
                rows.append(row)
        
        if rows:
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            logger.info(f"CSV results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save CSV to {output_path}: {e}")


def save_classification_results_csv(results: Dict[str, Any], output_path: str) -> None:
    """
    Save detailed classification results to CSV file for manual evaluation.
    
    Exports one row per response with:
    - For single-layer: question, response, token_scores, overall_prediction, ground_truth
    - For multi-layer: question, response, token_scores_layer_X (for each layer), 
      aggregated_score_layer_X (for each layer), overall_prediction_layer_X (for each layer), ground_truth
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        csv_rows = []
        all_layers = set()  # Track all layers for multi-layer mode
        is_multi_layer = False
        
        # First pass: determine if we have multi-layer data and collect all layers
        for task_name, task_results in results.items():
            if not isinstance(task_results, dict) or 'sample_responses' not in task_results:
                continue
            
            # Skip steering mode results (they don't have classification data)
            if task_results.get('steering_mode', False):
                continue
                
            sample_responses = task_results['sample_responses']
            
            for response_data in sample_responses:
                layer_results = response_data.get('layer_results', {})
                if layer_results:
                    is_multi_layer = True
                    all_layers.update(layer_results.keys())
        
        # Sort layers for consistent column ordering
        sorted_layers = sorted(all_layers) if all_layers else []
        
        # Second pass: create CSV rows
        for task_name, task_results in results.items():
            if not isinstance(task_results, dict) or 'sample_responses' not in task_results:
                continue
            
            # Skip steering mode results (they don't have classification data)
            if task_results.get('steering_mode', False):
                continue
                
            sample_responses = task_results['sample_responses']
            
            for response_data in sample_responses:
                layer_results = response_data.get('layer_results', {})
                
                # Create base row
                csv_row = {
                    'question': response_data.get('question', ''),
                    'response': response_data.get('response', ''),
                    'ground_truth': ''  # Empty for user to fill
                }
                
                if is_multi_layer and layer_results:
                    # Multi-layer mode: create columns for each layer
                    for layer in sorted_layers:
                        layer_data = layer_results.get(layer, {})
                        
                        # Format token scores as pipe-separated values
                        token_scores_str = ""
                        if layer_data.get('token_scores'):
                            token_scores_formatted = [f"{score:.6f}" for score in layer_data['token_scores']]
                            token_scores_str = "|".join(token_scores_formatted)
                        
                        # Add layer-specific columns
                        csv_row[f'token_scores_layer_{layer}'] = token_scores_str
                        csv_row[f'aggregated_score_layer_{layer}'] = f"{layer_data.get('aggregated_score', 0.0):.6f}"
                        csv_row[f'overall_prediction_layer_{layer}'] = layer_data.get('classification', 'UNKNOWN')
                
                elif not is_multi_layer:
                    # Single-layer mode: use original format
                    token_scores_str = ""
                    if response_data.get('token_scores'):
                        token_scores_formatted = [f"{score:.6f}" for score in response_data['token_scores']]
                        token_scores_str = "|".join(token_scores_formatted)
                    
                    csv_row['token_scores'] = token_scores_str
                    csv_row['aggregated_score'] = f"{response_data.get('aggregated_score', 0.0):.6f}"
                    csv_row['overall_prediction'] = response_data.get('classification', 'UNKNOWN')
                
                csv_rows.append(csv_row)
        
        # Only save if we have classification data
        if csv_rows:
            # Determine fieldnames based on mode
            if is_multi_layer:
                fieldnames = ['question', 'response']
                for layer in sorted_layers:
                    fieldnames.extend([
                        f'token_scores_layer_{layer}',
                        f'aggregated_score_layer_{layer}',
                        f'overall_prediction_layer_{layer}'
                    ])
                fieldnames.append('ground_truth')
            else:
                fieldnames = ['question', 'response', 'token_scores', 'aggregated_score', 'overall_prediction', 'ground_truth']
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)
            
            logger.info(f"Classification results CSV saved to {output_path}")
            print(f"\n📊 Classification results saved to: {output_path}")
            print(f"   • {len(csv_rows)} responses exported")
            if is_multi_layer:
                print(f"   • Multi-layer format with columns for layers: {sorted_layers}")
                print(f"   • Token scores, aggregated scores, and predictions saved per layer")
            else:
                print(f"   • Single-layer format")
            print(f"   • Fill in the 'ground_truth' column with: 'TRUTHFUL' or 'HALLUCINATION'")
            print(f"   • Use for manual evaluation and classifier optimization")
        else:
            logger.info("No classification results to export (steering mode or empty results)")
        
    except Exception as e:
        logger.error(f"Failed to save classification CSV to {output_path}: {e}")


def create_evaluation_report(results: Dict[str, Any], output_path: str) -> None:
    """Create a comprehensive evaluation report in markdown format."""
    try:
        with open(output_path, 'w') as f:
            f.write("# Wisent-Guard Evaluation Report\n\n")
            
            # Summary table
            f.write("## Summary\n\n")
            f.write("| Task | Training Accuracy | Evaluation Accuracy | Optimization |\n")
            f.write("|------|------------------|--------------------|--------------|\n")
            
            for task_name, task_results in results.items():
                if task_results is None:
                    f.write(f"| {task_name} | NULL | NULL | N/A |\n")
                elif isinstance(task_results, dict) and "error" in task_results:
                    f.write(f"| {task_name} | ERROR | ERROR | N/A |\n")
                elif isinstance(task_results, dict):
                    train_acc = task_results.get("training_results", {}).get("accuracy", "N/A")
                    eval_acc = task_results.get("evaluation_results", {}).get("accuracy", "N/A")
                    optimized = "Yes" if task_results.get("optimization_performed", False) else "No"
                    
                    if isinstance(train_acc, float):
                        train_acc = f"{train_acc:.2%}"
                    if isinstance(eval_acc, float):
                        eval_acc = f"{eval_acc:.2%}"
                    
                    f.write(f"| {task_name} | {train_acc} | {eval_acc} | {optimized} |\n")
            
            # Detailed results for each task
            for task_name, task_results in results.items():
                f.write(f"\n## {task_name}\n\n")
                
                if task_results is None:
                    f.write(f"**Error**: Task results are None\n")
                elif isinstance(task_results, dict) and "error" in task_results:
                    f.write(f"**Error**: {task_results['error']}\n")
                elif isinstance(task_results, dict):
                    # Configuration
                    f.write("### Configuration\n")
                    f.write(f"- **Model**: {task_results.get('model_name', 'Unknown')}\n")
                    f.write(f"- **Layer**: {task_results.get('layer', 'Unknown')}\n")
                    f.write(f"- **Classifier**: {task_results.get('classifier_type', 'Unknown')}\n")
                    f.write(f"- **Token Aggregation**: {task_results.get('token_aggregation', 'Unknown')}\n")
                    f.write(f"- **Ground Truth Method**: {task_results.get('ground_truth_method', 'Unknown')}\n")
                    
                    # Training results
                    if "training_results" in task_results:
                        train_results = task_results["training_results"]
                        f.write("\n### Training Results\n")
                        train_acc = train_results.get('accuracy', 'N/A')
                        if isinstance(train_acc, float):
                            f.write(f"- **Accuracy**: {train_acc:.2%}\n")
                        else:
                            f.write(f"- **Accuracy**: {train_acc}\n")
                        
                        train_prec = train_results.get('precision', 'N/A')
                        if isinstance(train_prec, float):
                            f.write(f"- **Precision**: {train_prec:.2f}\n")
                        else:
                            f.write(f"- **Precision**: {train_prec}\n")
                        
                        train_recall = train_results.get('recall', 'N/A')
                        if isinstance(train_recall, float):
                            f.write(f"- **Recall**: {train_recall:.2f}\n")
                        else:
                            f.write(f"- **Recall**: {train_recall}\n")
                        
                        train_f1 = train_results.get('f1', 'N/A')
                        if isinstance(train_f1, float):
                            f.write(f"- **F1 Score**: {train_f1:.2f}\n")
                        else:
                            f.write(f"- **F1 Score**: {train_f1}\n")
                    
                    # Evaluation results
                    if "evaluation_results" in task_results:
                        eval_results = task_results["evaluation_results"]
                        f.write("\n### Evaluation Results\n")
                        eval_acc = eval_results.get('accuracy', 'N/A')
                        if isinstance(eval_acc, float):
                            f.write(f"- **Accuracy**: {eval_acc:.2%}\n")
                        else:
                            f.write(f"- **Accuracy**: {eval_acc}\n")
                        f.write(f"- **Total Predictions**: {eval_results.get('total_predictions', 'N/A')}\n")
                        f.write(f"- **Correct Predictions**: {eval_results.get('correct_predictions', 'N/A')}\n")
                    
                    # Optimization results
                    if task_results.get("optimization_performed", False):
                        f.write("\n### Optimization Results\n")
                        f.write(f"- **Best Layer**: {task_results.get('best_layer', 'Unknown')}\n")
                        f.write(f"- **Best Aggregation**: {task_results.get('best_aggregation', 'Unknown')}\n")
                        best_acc = task_results.get('best_accuracy', 'Unknown')
                        if isinstance(best_acc, float):
                            f.write(f"- **Best Accuracy**: {best_acc:.2%}\n")
                        else:
                            f.write(f"- **Best Accuracy**: {best_acc}\n")
            
            f.write(f"\n---\n\n*Report generated on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
        
        logger.info(f"Evaluation report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to create report at {output_path}: {e}")
