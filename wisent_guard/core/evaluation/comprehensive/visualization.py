"""
Visualization utilities for comprehensive evaluation results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px



def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10


def plot_evaluation_results(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive evaluation results visualization.
    
    Args:
        results: Complete evaluation results
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plotting_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Evaluation Results', fontsize=16, fontweight='bold')
    
    # 1. Benchmark Performance Comparison
    if "test_results" in results:
        test_results = results["test_results"]
        
        base_acc = test_results.get("base_model_benchmark_results", {}).get("accuracy", 0.0)
        steered_acc = test_results.get("steered_model_benchmark_results", {}).get("accuracy", 0.0)
        
        axes[0, 0].bar(['Base Model', 'Steered Model'], [base_acc, steered_acc], 
                      color=['lightcoral', 'lightblue'])
        axes[0, 0].set_title('Benchmark Performance Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate([base_acc, steered_acc]):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. Probe Performance Comparison
    if "test_results" in results:
        base_auc = test_results.get("base_model_probe_results", {}).get("auc", 0.5)
        steered_auc = test_results.get("steered_model_probe_results", {}).get("auc", 0.5)
        
        axes[0, 1].bar(['Base Model', 'Steered Model'], [base_auc, steered_auc],
                      color=['lightcoral', 'lightblue'])
        axes[0, 1].set_title('Probe Performance Comparison')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate([base_auc, steered_auc]):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 3. Training Performance by Layer
    if "probe_training_results" in results:
        training_results = results["probe_training_results"]
        
        layers = []
        aucs = []
        
        for layer_key, layer_results in training_results.items():
            layer_num = int(layer_key.split('_')[1])
            # Get best AUC for this layer
            best_auc = 0
            for c_key, metrics in layer_results.items():
                if isinstance(metrics, dict) and "auc" in metrics:
                    best_auc = max(best_auc, metrics["auc"])
            
            layers.append(layer_num)
            aucs.append(best_auc)
        
        axes[1, 0].plot(layers, aucs, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Probe Performance by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Best AUC')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
    
    # 4. Optimization Landscape
    if "steering_optimization_results" in results:
        optimization_results = results["steering_optimization_results"]
        all_configs = optimization_results.get("all_configs", [])
        
        if all_configs:
            scores = [config.get("combined_score", 0.0) for config in all_configs]
            config_names = [f"Config {i+1}" for i in range(len(scores))]
            
            axes[1, 1].bar(range(len(scores)), scores, color='lightgreen')
            axes[1, 1].set_title('Optimization Configuration Scores')
            axes[1, 1].set_xlabel('Configuration')
            axes[1, 1].set_ylabel('Combined Score')
            axes[1, 1].set_xticks(range(len(scores)))
            axes[1, 1].set_xticklabels([f"{i+1}" for i in range(len(scores))])
            
            # Highlight best configuration
            best_idx = scores.index(max(scores))
            axes[1, 1].bar(best_idx, scores[best_idx], color='gold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_dashboard(results: Dict[str, Any]):
    """
    Create interactive Plotly dashboard for evaluation results.
    
    Args:
        results: Complete evaluation results
        
    Returns:
        Plotly figure with interactive dashboard or None if plotly not available
    """
        
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Benchmark Performance', 'Probe Performance', 
                       'Layer Analysis', 'Optimization Landscape'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Benchmark Performance
    if "test_results" in results:
        test_results = results["test_results"]
        
        base_acc = test_results.get("base_model_benchmark_results", {}).get("accuracy", 0.0)
        steered_acc = test_results.get("steered_model_benchmark_results", {}).get("accuracy", 0.0)
        
        fig.add_trace(
            go.Bar(x=['Base Model', 'Steered Model'], y=[base_acc, steered_acc],
                   name='Benchmark Accuracy', marker_color=['lightcoral', 'lightblue']),
            row=1, col=1
        )
        
        # 2. Probe Performance
        base_auc = test_results.get("base_model_probe_results", {}).get("auc", 0.5)
        steered_auc = test_results.get("steered_model_probe_results", {}).get("auc", 0.5)
        
        fig.add_trace(
            go.Bar(x=['Base Model', 'Steered Model'], y=[base_auc, steered_auc],
                   name='Probe AUC', marker_color=['lightcoral', 'lightblue']),
            row=1, col=2
        )
    
    # 3. Layer Analysis
    if "probe_training_results" in results:
        training_results = results["probe_training_results"]
        
        layers = []
        aucs = []
        
        for layer_key, layer_results in training_results.items():
            layer_num = int(layer_key.split('_')[1])
            best_auc = 0
            for c_key, metrics in layer_results.items():
                if isinstance(metrics, dict) and "auc" in metrics:
                    best_auc = max(best_auc, metrics["auc"])
            
            layers.append(layer_num)
            aucs.append(best_auc)
        
        fig.add_trace(
            go.Scatter(x=layers, y=aucs, mode='lines+markers',
                      name='Layer Performance', line=dict(width=3), marker=dict(size=10)),
            row=2, col=1
        )
    
    # 4. Optimization Landscape
    if "steering_optimization_results" in results:
        optimization_results = results["steering_optimization_results"]
        all_configs = optimization_results.get("all_configs", [])
        
        if all_configs:
            scores = [config.get("combined_score", 0.0) for config in all_configs]
            config_labels = [f"Config {i+1}" for i in range(len(scores))]
            
            colors = ['gold' if score == max(scores) else 'lightgreen' for score in scores]
            
            fig.add_trace(
                go.Bar(x=config_labels, y=scores, name='Config Scores',
                      marker_color=colors),
                row=2, col=2
            )
    
    fig.update_layout(
        title_text="Comprehensive Evaluation Dashboard",
        title_x=0.5,
        height=800,
        showlegend=False
    )
    
    return fig


def plot_hyperparameter_heatmap(results: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
    """
    Create heatmap of hyperparameter performance.
    
    Args:
        results: Complete evaluation results
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    if "steering_optimization_results" not in results:
        return None
    
    optimization_results = results["steering_optimization_results"]
    all_configs = optimization_results.get("all_configs", [])
    
    if not all_configs:
        return None
    
    # Extract data for heatmap
    data = []
    for config in all_configs:
        steering_config = config.get("steering_config", {})
        probe_config = config.get("best_probe_config", {})
        
        data.append({
            'steering_layer': steering_config.get('layer', 0),
            'probe_layer': probe_config.get('layer', 0),
            'combined_score': config.get('combined_score', 0.0),
            'benchmark_score': config.get('benchmark_metrics', {}).get('accuracy', 0.0),
            'probe_score': config.get('probe_metrics', {}).get('auc', 0.5)
        })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Combined Score Heatmap
    pivot_combined = df.pivot_table(values='combined_score', 
                                   index='steering_layer', 
                                   columns='probe_layer', 
                                   aggfunc='mean')
    
    sns.heatmap(pivot_combined, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
    axes[0].set_title('Combined Score by Layer Configuration')
    axes[0].set_xlabel('Probe Layer')
    axes[0].set_ylabel('Steering Layer')
    
    # Benchmark Score Heatmap
    pivot_benchmark = df.pivot_table(values='benchmark_score', 
                                    index='steering_layer', 
                                    columns='probe_layer', 
                                    aggfunc='mean')
    
    sns.heatmap(pivot_benchmark, annot=True, fmt='.3f', cmap='Reds', ax=axes[1])
    axes[1].set_title('Benchmark Score by Layer Configuration')
    axes[1].set_xlabel('Probe Layer')
    axes[1].set_ylabel('Steering Layer')
    
    # Probe Score Heatmap
    pivot_probe = df.pivot_table(values='probe_score', 
                                index='steering_layer', 
                                columns='probe_layer', 
                                aggfunc='mean')
    
    sns.heatmap(pivot_probe, annot=True, fmt='.3f', cmap='Blues', ax=axes[2])
    axes[2].set_title('Probe Score by Layer Configuration')
    axes[2].set_xlabel('Probe Layer')
    axes[2].set_ylabel('Steering Layer')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_summary_report(results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Generate comprehensive HTML summary report.
    
    Args:
        results: Complete evaluation results
        config: Evaluation configuration
        
    Returns:
        HTML string of the report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comprehensive Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 30px 0; }}
            .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
            .config-table {{ border-collapse: collapse; width: 100%; }}
            .config-table th, .config-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .config-table th {{ background-color: #f2f2f2; }}
            .improvement {{ color: green; font-weight: bold; }}
            .decline {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Comprehensive Evaluation Report</h1>
            <p><strong>Model:</strong> {config.get('model_name', 'Unknown')}</p>
            <p><strong>Generated:</strong> {results.get('timestamp', 'Unknown')}</p>
        </div>
    """
    
    # Configuration Summary
    html += """
        <div class="section">
            <h2>Configuration Summary</h2>
            <table class="config-table">
                <tr><th>Parameter</th><th>Value</th></tr>
    """
    
    for key, value in config.items():
        if key not in ['wandb_tags', 'wandb_entity']:  # Skip some internal config
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
    
    html += "</table></div>"
    
    # Results Summary
    if "test_results" in results:
        test_results = results["test_results"]
        
        base_benchmark = test_results.get("base_model_benchmark_results", {}).get("accuracy", 0.0)
        steered_benchmark = test_results.get("steered_model_benchmark_results", {}).get("accuracy", 0.0)
        base_probe = test_results.get("base_model_probe_results", {}).get("auc", 0.5)
        steered_probe = test_results.get("steered_model_probe_results", {}).get("auc", 0.5)
        
        benchmark_change = steered_benchmark - base_benchmark
        probe_change = steered_probe - base_probe
        
        benchmark_class = "improvement" if benchmark_change > 0 else "decline"
        probe_class = "improvement" if probe_change > 0 else "decline"
        
        html += f"""
        <div class="section">
            <h2>Performance Results</h2>
            
            <div class="metric">
                <h3>Benchmark Performance</h3>
                <p><strong>Base Model:</strong> {base_benchmark:.3f} ({base_benchmark*100:.1f}%)</p>
                <p><strong>Steered Model:</strong> {steered_benchmark:.3f} ({steered_benchmark*100:.1f}%)</p>
                <p><strong>Change:</strong> <span class="{benchmark_class}">{benchmark_change:+.3f} ({benchmark_change*100:+.1f}%)</span></p>
            </div>
            
            <div class="metric">
                <h3>Probe Performance</h3>
                <p><strong>Base Model AUC:</strong> {base_probe:.3f}</p>
                <p><strong>Steered Model AUC:</strong> {steered_probe:.3f}</p>
                <p><strong>Change:</strong> <span class="{probe_class}">{probe_change:+.3f}</span></p>
            </div>
            
            <div class="metric">
                <h3>Optimized Configuration</h3>
                <p><strong>Steering:</strong> {test_results.get('optimized_steering_config', 'N/A')}</p>
                <p><strong>Probe:</strong> {test_results.get('optimized_probe_config', 'N/A')}</p>
                <p><strong>Validation Score:</strong> {test_results.get('validation_combined_score', 0.0):.3f}</p>
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    return html


def create_results_dashboard(results: Dict[str, Any]):
    """
    Alias for create_interactive_dashboard for backward compatibility.
    """
    return create_interactive_dashboard(results)