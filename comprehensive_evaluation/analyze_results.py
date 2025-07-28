"""
Results analysis script for GSM8K optimization pipeline.

This script helps analyze the results from the Optuna optimization and generate reports.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import numpy as np
from typing import Dict, Any, List
import optuna


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load results from the output directory."""
    results_path = Path(results_dir)
    
    # Find the most recent results file
    result_files = list(results_path.glob("final_results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results from: {latest_file}")
    return results


def load_study_trials(results_dir: str) -> pd.DataFrame:
    """Load Optuna study trials."""
    results_path = Path(results_dir)
    
    # Find the most recent trials file
    trial_files = list(results_path.glob("study_trials_*.csv"))
    if not trial_files:
        print("Warning: No study trials file found")
        return pd.DataFrame()
    
    latest_file = max(trial_files, key=lambda p: p.stat().st_mtime)
    
    trials_df = pd.read_csv(latest_file)
    print(f"Loaded {len(trials_df)} trials from: {latest_file}")
    return trials_df


def analyze_optimization_performance(trials_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the optimization performance."""
    if trials_df.empty:
        return {}
    
    analysis = {}
    
    # Basic statistics
    analysis['total_trials'] = len(trials_df)
    analysis['successful_trials'] = len(trials_df[trials_df['state'] == 'COMPLETE'])
    analysis['failed_trials'] = len(trials_df[trials_df['state'] == 'FAIL'])
    analysis['pruned_trials'] = len(trials_df[trials_df['state'] == 'PRUNED'])
    
    # Performance statistics
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE']
    if not completed_trials.empty:
        analysis['best_value'] = completed_trials['value'].max()
        analysis['worst_value'] = completed_trials['value'].min()
        analysis['mean_value'] = completed_trials['value'].mean()
        analysis['std_value'] = completed_trials['value'].std()
        analysis['median_value'] = completed_trials['value'].median()
    
    # Hyperparameter analysis
    param_columns = [col for col in trials_df.columns if col.startswith('params_')]
    if param_columns:
        analysis['hyperparameters'] = {}
        for param in param_columns:
            param_name = param.replace('params_', '')
            if completed_trials[param].dtype in ['float64', 'int64']:
                analysis['hyperparameters'][param_name] = {
                    'mean': completed_trials[param].mean(),
                    'std': completed_trials[param].std(),
                    'min': completed_trials[param].min(),
                    'max': completed_trials[param].max()
                }
            else:
                analysis['hyperparameters'][param_name] = {
                    'unique_values': completed_trials[param].unique().tolist()
                }
    
    return analysis


def create_optimization_plots(trials_df: pd.DataFrame, output_dir: str):
    """Create visualization plots for optimization results."""
    if trials_df.empty:
        print("No trials data available for plotting")
        return
    
    output_path = Path(output_dir)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Optimization history
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Trial values over time
    completed_trials = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    if not completed_trials.empty:
        completed_trials = completed_trials.sort_values('number')
        
        axes[0, 0].plot(completed_trials['number'], completed_trials['value'], 'o-', alpha=0.7)
        axes[0, 0].axhline(y=completed_trials['value'].max(), color='r', linestyle='--', alpha=0.7, label='Best')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Validation Accuracy')
        axes[0, 0].set_title('Optimization History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative best
        cumulative_best = completed_trials['value'].cummax()
        axes[0, 1].plot(completed_trials['number'], cumulative_best, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Trial Number')
        axes[0, 1].set_ylabel('Best Validation Accuracy')
        axes[0, 1].set_title('Cumulative Best Performance')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Trial state distribution
    state_counts = trials_df['state'].value_counts()
    axes[1, 0].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Trial State Distribution')
    
    # Value distribution
    if not completed_trials.empty:
        axes[1, 1].hist(completed_trials['value'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=completed_trials['value'].mean(), color='r', linestyle='--', label='Mean')
        axes[1, 1].axvline(x=completed_trials['value'].median(), color='g', linestyle='--', label='Median')
        axes[1, 1].set_xlabel('Validation Accuracy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Validation Accuracy Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Hyperparameter analysis
    param_columns = [col for col in trials_df.columns if col.startswith('params_')]
    if param_columns and not completed_trials.empty:
        n_params = len(param_columns)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, param_col in enumerate(param_columns):
            param_name = param_col.replace('params_', '')
            
            if completed_trials[param_col].dtype in ['float64', 'int64']:
                # Scatter plot for numerical parameters
                axes[i].scatter(completed_trials[param_col], completed_trials['value'], 
                              alpha=0.6, s=50)
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('Validation Accuracy')
                axes[i].set_title(f'{param_name} vs Performance')
                axes[i].grid(True, alpha=0.3)
            else:
                # Box plot for categorical parameters
                param_values = completed_trials[param_col].unique()
                box_data = [completed_trials[completed_trials[param_col] == val]['value'] 
                           for val in param_values]
                axes[i].boxplot(box_data, labels=param_values)
                axes[i].set_xlabel(param_name)
                axes[i].set_ylabel('Validation Accuracy')
                axes[i].set_title(f'{param_name} vs Performance')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to: {output_path}")


def generate_report(results: Dict[str, Any], analysis: Dict[str, Any], 
                   output_dir: str) -> str:
    """Generate a comprehensive report."""
    
    report = f"""
# GSM8K Optimization Results Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration Summary

- **Model:** {results.get('config', {}).get('model_name', 'N/A')}
- **Dataset:** {results.get('config', {}).get('train_dataset', 'N/A')}
- **Train/Val/Test samples:** {results.get('config', {}).get('train_limit', 'N/A')}/{results.get('config', {}).get('val_limit', 'N/A')}/{results.get('config', {}).get('test_limit', 'N/A')}
- **Layer search range:** {results.get('config', {}).get('layer_search_range', 'N/A')}
- **Steering methods:** {results.get('config', {}).get('steering_methods', 'N/A')}

## Optimization Results

### Trial Statistics
"""
    
    if analysis:
        report += f"""
- **Total trials:** {analysis.get('total_trials', 'N/A')}
- **Successful trials:** {analysis.get('successful_trials', 'N/A')}
- **Failed trials:** {analysis.get('failed_trials', 'N/A')}
- **Pruned trials:** {analysis.get('pruned_trials', 'N/A')}

### Performance Statistics
- **Best validation score:** {analysis.get('best_value', 'N/A'):.4f}
- **Mean validation score:** {analysis.get('mean_value', 'N/A'):.4f} ± {analysis.get('std_value', 'N/A'):.4f}
- **Median validation score:** {analysis.get('median_value', 'N/A'):.4f}

### Best Configuration
"""
    
    best_params = results.get('best_trial_params', {})
    for param, value in best_params.items():
        report += f"- **{param}:** {value}\n"
    
    report += f"""

## Final Test Results

### Benchmark Performance
- **Baseline accuracy:** {results.get('baseline_benchmark_metrics', {}).get('accuracy', 'N/A'):.4f}
- **Steered accuracy:** {results.get('steered_benchmark_metrics', {}).get('accuracy', 'N/A'):.4f}
- **Improvement:** {results.get('accuracy_improvement', 'N/A'):+.4f}

### Probe Performance
- **AUC:** {results.get('test_probe_metrics', {}).get('auc', 'N/A'):.4f}
- **Accuracy:** {results.get('test_probe_metrics', {}).get('accuracy', 'N/A'):.4f}
- **Precision:** {results.get('test_probe_metrics', {}).get('precision', 'N/A'):.4f}
- **Recall:** {results.get('test_probe_metrics', {}).get('recall', 'N/A'):.4f}
- **F1-Score:** {results.get('test_probe_metrics', {}).get('f1', 'N/A'):.4f}

## Summary

The optimization {'succeeded' if results.get('accuracy_improvement', 0) > 0 else 'did not achieve'} in improving model performance.
"""
    
    if results.get('accuracy_improvement', 0) > 0.05:
        report += "**Significant improvement achieved!** The steering method shows strong effectiveness.\n"
    elif results.get('accuracy_improvement', 0) > 0.01:
        report += "**Moderate improvement achieved.** Consider further hyperparameter tuning.\n"
    elif results.get('accuracy_improvement', 0) > -0.01:
        report += "**Minimal change observed.** The steering may not be effective for this configuration.\n"
    else:
        report += "**Performance decreased.** Check steering implementation and hyperparameters.\n"
    
    # Save report
    report_path = Path(output_dir) / "optimization_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze GSM8K optimization results")
    parser.add_argument("results_dir", help="Directory containing optimization results")
    parser.add_argument("--create-plots", action="store_true", 
                       help="Create visualization plots")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for analysis (defaults to results_dir)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    
    try:
        # Load results
        results = load_results(args.results_dir)
        trials_df = load_study_trials(args.results_dir)
        
        # Analyze optimization performance
        analysis = analyze_optimization_performance(trials_df)
        
        # Create plots if requested
        if args.create_plots:
            create_optimization_plots(trials_df, output_dir)
        
        # Generate report
        report = generate_report(results, analysis, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        if analysis:
            print(f"Total trials: {analysis.get('total_trials', 'N/A')}")
            print(f"Best validation score: {analysis.get('best_value', 'N/A'):.4f}")
        
        print(f"Test improvement: {results.get('accuracy_improvement', 'N/A'):+.4f}")
        print(f"Probe AUC: {results.get('test_probe_metrics', {}).get('auc', 'N/A'):.4f}")
        print(f"Analysis saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()