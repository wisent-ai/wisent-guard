#!/usr/bin/env python3
"""
Comprehensive comparison of all steering methods on TruthfulQA MC1.
Uses the existing CLI infrastructure to avoid reinventing the wheel.
"""

import subprocess
import json
import os
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SteeringMethodComparison:
    """Compare all steering methods using the existing CLI."""
    
    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct", 
                 layer: int = 15, limit: int = 50, output_dir: str = "steering_comparison"):
        self.model = model
        self.layer = layer
        self.limit = limit
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define all steering methods to test
        self.steering_methods = [
            {"name": "Baseline", "method": None, "params": {}},
            {"name": "CAA", "method": "CAA", "params": {}},
            {"name": "CAA_L2", "method": "CAA", "params": {"normalization_method": "l2_unit"}},
            {"name": "HPR", "method": "HPR", "params": {"hpr_beta": 1.0}},
            {"name": "HPR_Beta0.5", "method": "HPR", "params": {"hpr_beta": 0.5}},
            {"name": "BiPO", "method": "BiPO", "params": {"bipo_beta": 0.1, "bipo_epochs": 50}},
            {"name": "BiPO_Beta0.05", "method": "BiPO", "params": {"bipo_beta": 0.05, "bipo_epochs": 50}},
            {"name": "KSteering", "method": "KSteering", "params": {
                "ksteering_alpha": 5.0, 
                "ksteering_target_labels": "0",
                "ksteering_avoid_labels": ""
            }},
            {"name": "KSteering_Alpha3", "method": "KSteering", "params": {
                "ksteering_alpha": 3.0,
                "ksteering_target_labels": "0", 
                "ksteering_avoid_labels": ""
            }},
            {"name": "DAC", "method": "DAC", "params": {
                "dac_dynamic_control": True,
                "dac_entropy_threshold": 1.0
            }},
        ]
        
        # Different steering strengths to test
        self.steering_strengths = [0.5, 1.0, 1.5, 2.0]
        
    def run_baseline(self) -> Dict[str, Any]:
        """Run baseline (unsteered) evaluation."""
        print("🔄 Running baseline (unsteered) evaluation...")
        
        output_path = os.path.join(self.output_dir, f"baseline_{self.timestamp}.json")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        cmd = [
            "python", "-m", "wisent_guard.cli", "tasks", "truthfulqa_mc1",
            "--model", self.model,
            "--layer", str(self.layer),
            "--limit", str(self.limit),
            "--output", output_path,
            "--allow-small-dataset",
            "--output-mode", "likelihoods",
            "--verbose"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                print(f"❌ Baseline failed: {result.stderr}")
                return {"error": result.stderr}
            
            # Load baseline results directly from the output file
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    return json.load(f)
            else:
                return {"error": "Results file not found"}
                
        except subprocess.TimeoutExpired:
            return {"error": "Baseline evaluation timed out"}
        except Exception as e:
            return {"error": str(e)}
    
    def run_steering_method(self, method_config: Dict[str, Any], strength: float = 1.0) -> Dict[str, Any]:
        """Run a specific steering method."""
        method_name = method_config["name"]
        steering_method = method_config["method"]
        params = method_config["params"]
        
        print(f"🎯 Running {method_name} (strength={strength})...")
        
        output_path = os.path.join(self.output_dir, f"{method_name}_s{strength}_{self.timestamp}.json")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        cmd = [
            "python", "-m", "wisent_guard.cli", "tasks", "truthfulqa_mc1",
            "--model", self.model,
            "--layer", str(self.layer),
            "--limit", str(self.limit),
            "--output", output_path,
            "--allow-small-dataset",
            "--steering-mode",
            "--steering-strength", str(strength),
            "--output-mode", "likelihoods",
            "--verbose"
        ]
        
        # Add steering method
        if steering_method:
            cmd.extend(["--steering-method", steering_method])
        
        # Add method-specific parameters
        for param, value in params.items():
            param_name = f"--{param.replace('_', '-')}"
            if isinstance(value, bool):
                if value:  # Only add the flag if True
                    cmd.append(param_name)
            else:
                cmd.extend([param_name, str(value)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                print(f"❌ {method_name} failed: {result.stderr}")
                return {"error": result.stderr, "method": method_name, "strength": strength}
            
            # Load results directly from the output file
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    data = json.load(f)
                    data["method"] = method_name
                    data["strength"] = strength
                    return data
            else:
                return {"error": "Results file not found", "method": method_name, "strength": strength}
                
        except subprocess.TimeoutExpired:
            return {"error": "Evaluation timed out", "method": method_name, "strength": strength}
        except Exception as e:
            return {"error": str(e), "method": method_name, "strength": strength}
    
    def _find_latest_results_file(self, output_path: str) -> str:
        """Find the latest results JSON file in the output directory."""
        if not os.path.exists(output_path):
            return None
        
        json_files = [f for f in os.listdir(output_path) if f.startswith("results_") and f.endswith(".json")]
        if not json_files:
            return None
        
        # Get the latest file
        latest_file = max(json_files, key=lambda f: os.path.getctime(os.path.join(output_path, f)))
        return os.path.join(output_path, latest_file)
    
    def evaluate_responses_manually(self, responses: List[Dict], expected_answers: List[Dict]) -> Dict[str, float]:
        """Manually evaluate responses for truthfulness based on expected answers."""
        if not responses or not expected_answers:
            return {"accuracy": 0.0, "truthful_rate": 0.0, "total_responses": 0}
        
        correct_count = 0
        truthful_count = 0
        total_count = len(responses)
        
        for i, response in enumerate(responses):
            if i >= len(expected_answers):
                break
                
            expected = expected_answers[i]
            generated_text = response.get("response", "").lower()
            correct_answer = expected.get("correct_answer", "").lower()
            incorrect_answer = expected.get("incorrect_answer", "").lower()
            
            # Simple heuristic: if response contains key elements of correct answer
            # and doesn't strongly assert the incorrect answer, consider it truthful
            contains_correct_elements = any(word in generated_text for word in correct_answer.split()[:3])
            strongly_asserts_incorrect = incorrect_answer[:20].lower() in generated_text
            
            is_truthful = contains_correct_elements and not strongly_asserts_incorrect
            
            if is_truthful:
                truthful_count += 1
            
            # For classifier accuracy, check if classification matches our assessment
            classification = response.get("classification", "UNKNOWN")
            expected_classification = "TRUTHFUL" if is_truthful else "HALLUCINATION"
            
            if classification == expected_classification:
                correct_count += 1
        
        return {
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "truthful_rate": truthful_count / total_count if total_count > 0 else 0.0,
            "total_responses": total_count,
            "correct_classifications": correct_count,
            "truthful_responses": truthful_count
        }
    
    def run_comprehensive_comparison(self, quick_test: bool = False) -> pd.DataFrame:
        """Run comprehensive comparison of all steering methods."""
        print("🧪 COMPREHENSIVE STEERING METHODS COMPARISON")
        print("=" * 60)
        print(f"🎯 Model: {self.model}")
        print(f"🎯 Layer: {self.layer}")
        print(f"🎯 Sample limit: {self.limit}")
        print(f"🎯 Output directory: {self.output_dir}")
        print()
        
        all_results = []
        
        # Run baseline first
        print("📊 Step 1: Running baseline evaluation...")
        baseline_result = self.run_baseline()
        
        if "error" not in baseline_result:
            baseline_eval = self._extract_evaluation_metrics(baseline_result, "Baseline", 0.0)
            all_results.append(baseline_eval)
            print(f"✅ Baseline completed: {baseline_eval['truthful_rate']:.1%} truthful")
        else:
            print(f"❌ Baseline failed: {baseline_result['error']}")
            return pd.DataFrame()
        
        # Test steering methods
        print("\n🎯 Step 2: Testing steering methods...")
        
        methods_to_test = self.steering_methods[1:]  # Skip baseline
        strengths_to_test = [1.0] if quick_test else self.steering_strengths
        
        if quick_test:
            methods_to_test = methods_to_test[:3]  # Test only first 3 methods
            print(f"🔄 Quick test mode: testing {len(methods_to_test)} methods with strength 1.0")
        
        for method_config in methods_to_test:
            for strength in strengths_to_test:
                result = self.run_steering_method(method_config, strength)
                
                if "error" not in result:
                    eval_metrics = self._extract_evaluation_metrics(result, method_config["name"], strength)
                    all_results.append(eval_metrics)
                    print(f"✅ {method_config['name']} (s={strength}): {eval_metrics['truthful_rate']:.1%} truthful")
                else:
                    print(f"❌ {method_config['name']} (s={strength}) failed: {result['error']}")
                    # Add failed result
                    all_results.append({
                        "method": method_config["name"],
                        "strength": strength,
                        "accuracy": 0.0,
                        "truthful_rate": 0.0,
                        "total_responses": 0,
                        "error": result["error"]
                    })
        
        # Create results DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        results_file = os.path.join(self.output_dir, f"steering_comparison_{self.timestamp}.csv")
        df.to_csv(results_file, index=False)
        print(f"\n💾 Results saved to: {results_file}")
        
        return df
    
    def _extract_evaluation_metrics(self, result: Dict[str, Any], method_name: str, strength: float) -> Dict[str, Any]:
        """Extract evaluation metrics from CLI result using lm-harness scores."""
        task_result = result.get("truthfulqa_mc1", {})
        
        # Extract lm-harness evaluation results (now working properly!)
        evaluation_results = task_result.get("evaluation_results", {})
        accuracy = evaluation_results.get("accuracy", 0.0)
        
        # Handle the case where accuracy might be "N/A" from failed evaluations
        if isinstance(accuracy, str) or accuracy is None:
            accuracy = 0.0
        
        # Get baseline and steered log-likelihoods if available
        baseline_likelihoods = evaluation_results.get("baseline_likelihoods", [])
        steered_likelihoods = evaluation_results.get("steered_likelihoods", [])
        
        # Calculate steering effectiveness (how much the likelihoods changed)
        steering_effect = 0.0
        if baseline_likelihoods and steered_likelihoods:
            total_change = sum(abs(s - b) for s, b in zip(steered_likelihoods, baseline_likelihoods))
            steering_effect = total_change / len(baseline_likelihoods) if baseline_likelihoods else 0.0
        
        # Get other metadata
        steering_applied = evaluation_results.get("steering_applied", method_name != "Baseline")
        total_samples = len(baseline_likelihoods) if baseline_likelihoods else 0
        
        return {
            "method": method_name,
            "strength": strength,
            "accuracy": accuracy,
            "truthful_rate": accuracy,  # For TruthfulQA, accuracy == truthfulness rate
            "total_responses": total_samples,
            "steering_effect": steering_effect,
            "steering_applied": steering_applied,
            "baseline_likelihoods": baseline_likelihoods[:5] if baseline_likelihoods else [],  # First 5 for debugging
            "steered_likelihoods": steered_likelihoods[:5] if steered_likelihoods else []
        }
    
    def create_comparison_plots(self, df: pd.DataFrame):
        """Create visualization plots comparing steering methods."""
        if df.empty:
            print("❌ No data to plot")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Steering Methods Comparison on TruthfulQA MC1\n{self.model} - Layer {self.layer}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Truthful Rate by Method
        ax1 = axes[0, 0]
        baseline_rate = df[df['method'] == 'Baseline']['truthful_rate'].iloc[0] if 'Baseline' in df['method'].values else 0
        
        method_means = df.groupby('method')['truthful_rate'].mean().sort_values(ascending=False)
        colors = sns.color_palette("husl", len(method_means))
        
        bars = ax1.bar(range(len(method_means)), method_means.values, color=colors)
        ax1.axhline(y=baseline_rate, color='red', linestyle='--', alpha=0.7, label=f'Baseline ({baseline_rate:.1%})')
        ax1.set_xlabel('Steering Method')
        ax1.set_ylabel('Truthful Rate')
        ax1.set_title('Truthful Rate by Steering Method')
        ax1.set_xticks(range(len(method_means)))
        ax1.set_xticklabels(method_means.index, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, method_means.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Improvement over Baseline
        ax2 = axes[0, 1]
        improvements = method_means - baseline_rate
        colors_improvement = ['green' if x > 0 else 'red' for x in improvements.values]
        
        bars2 = ax2.bar(range(len(improvements)), improvements.values, color=colors_improvement, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Steering Method')
        ax2.set_ylabel('Improvement over Baseline')
        ax2.set_title('Improvement over Baseline')
        ax2.set_xticks(range(len(improvements)))
        ax2.set_xticklabels(improvements.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, improvements.values):
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.01 if value > 0 else -0.03), 
                    f'{value:+.1%}', ha='center', 
                    va='bottom' if value > 0 else 'top', fontweight='bold')
        
        # 3. Steering Strength Analysis (if multiple strengths tested)
        ax3 = axes[1, 0]
        steering_df = df[df['method'] != 'Baseline']
        if len(steering_df['strength'].unique()) > 1:
            for method in steering_df['method'].unique():
                method_data = steering_df[steering_df['method'] == method]
                ax3.plot(method_data['strength'], method_data['truthful_rate'], 
                        marker='o', label=method, linewidth=2)
            
            ax3.axhline(y=baseline_rate, color='red', linestyle='--', alpha=0.7, label=f'Baseline')
            ax3.set_xlabel('Steering Strength')
            ax3.set_ylabel('Truthful Rate')
            ax3.set_title('Truthful Rate vs Steering Strength')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Single Strength Tested\n(Strength = 1.0)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Steering Strength Analysis')
        
        # 4. Summary Statistics Table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create summary table
        summary_data = []
        for method in method_means.index:
            method_data = df[df['method'] == method]
            improvement = method_means[method] - baseline_rate
            summary_data.append([
                method,
                f"{method_means[method]:.1%}",
                f"{improvement:+.1%}",
                f"{method_data['total_responses'].mean():.0f}"
            ])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=['Method', 'Truthful Rate', 'Improvement', 'Responses'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.output_dir, f"steering_comparison_plot_{self.timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to: {plot_file}")
        
        plt.show()
    
    def print_summary(self, df: pd.DataFrame):
        """Print a comprehensive summary of results."""
        if df.empty:
            print("❌ No results to summarize")
            return
        
        print("\n" + "="*80)
        print("🏆 STEERING METHODS COMPARISON SUMMARY")
        print("="*80)
        
        baseline_rate = df[df['method'] == 'Baseline']['truthful_rate'].iloc[0] if 'Baseline' in df['method'].values else 0
        print(f"📊 Baseline Performance: {baseline_rate:.1%} truthful responses")
        print()
        
        # Best performing methods
        steering_df = df[df['method'] != 'Baseline']
        if not steering_df.empty:
            best_methods = steering_df.nlargest(5, 'truthful_rate')
            
            print("🥇 TOP 5 STEERING METHODS:")
            for i, (_, row) in enumerate(best_methods.iterrows(), 1):
                improvement = row['truthful_rate'] - baseline_rate
                print(f"{i:2}. {row['method']:<15} (s={row['strength']:<3}) | "
                      f"Truthful: {row['truthful_rate']:.1%} | "
                      f"Improvement: {improvement:+.1%}")
            
            print()
            
            # Method category analysis
            print("📈 IMPROVEMENT ANALYSIS:")
            successful_methods = steering_df[steering_df['truthful_rate'] > baseline_rate]
            failed_methods = steering_df[steering_df['truthful_rate'] <= baseline_rate]
            
            print(f"✅ Methods that improved over baseline: {len(successful_methods)}/{len(steering_df)}")
            print(f"❌ Methods that performed worse: {len(failed_methods)}/{len(steering_df)}")
            
            if len(successful_methods) > 0:
                avg_improvement = (successful_methods['truthful_rate'] - baseline_rate).mean()
                best_improvement = (successful_methods['truthful_rate'] - baseline_rate).max()
                print(f"📊 Average improvement: +{avg_improvement:.1%}")
                print(f"🎯 Best improvement: +{best_improvement:.1%}")
        
        print()
        print("💾 All results saved to:", self.output_dir)
        print("="*80)


def main():
    """Main function to run the steering methods comparison."""
    parser = argparse.ArgumentParser(description="Compare all steering methods on TruthfulQA MC1")
    
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Model to test (default: Llama-3.1-8B-Instruct)")
    parser.add_argument("--layer", type=int, default=15,
                       help="Layer to apply steering (default: 15)")
    parser.add_argument("--limit", type=int, default=50,
                       help="Number of samples to test (default: 50)")
    parser.add_argument("--output-dir", default="steering_comparison_results",
                       help="Output directory for results")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test with fewer methods and single strength")
    parser.add_argument("--plot-only", type=str, default=None,
                       help="Path to existing CSV results file to plot only")
    
    args = parser.parse_args()
    
    if args.plot_only:
        # Load existing results and create plots
        if os.path.exists(args.plot_only):
            df = pd.read_csv(args.plot_only)
            comparison = SteeringMethodComparison(args.model, args.layer, args.limit, args.output_dir)
            comparison.create_comparison_plots(df)
            comparison.print_summary(df)
        else:
            print(f"❌ Results file not found: {args.plot_only}")
        return
    
    # Run full comparison
    comparison = SteeringMethodComparison(
        model=args.model,
        layer=args.layer,
        limit=args.limit,
        output_dir=args.output_dir
    )
    
    print("🚀 Starting comprehensive steering methods comparison...")
    print(f"⏱️  Estimated time: {len(comparison.steering_methods) * 5} minutes")
    print()
    
    # Run comparison
    results_df = comparison.run_comprehensive_comparison(quick_test=args.quick_test)
    
    if not results_df.empty:
        # Create plots
        comparison.create_comparison_plots(results_df)
        
        # Print summary
        comparison.print_summary(results_df)
        
        print("\n✅ Steering methods comparison completed successfully!")
    else:
        print("\n❌ Comparison failed - no results generated")


if __name__ == "__main__":
    main()
