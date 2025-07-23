#!/usr/bin/env python3
"""
Model Benchmark Evaluation Pipeline

Modular and configurable evaluation system for training wisent_guard classifiers 
on one benchmark and evaluating on another (e.g., train on aime2024, eval on aime2025).

Features:
- ClearML experiment tracking
- Configurable training and evaluation benchmarks
- Support for different layers and model configurations
- Comprehensive logging and results tracking
"""

import os
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import subprocess
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv("/workspace/.env")

try:
    from clearml import Task, Logger
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("Warning: ClearML not available. Install with: pip install clearml")

# Add wisent_guard to path for importing
sys.path.insert(0, str(Path(__file__).parent))

from wisent_guard.core.tasks import register_all_tasks
from wisent_guard.core.task_interface import get_task


@dataclass
class EvaluationConfig:
    """Configuration for model benchmark evaluation."""
    
    # Model configuration
    model_name: str = "/workspace/models/llama31‑8b‑instruct‑hf"
    layer: int = 15
    
    # Training configuration
    train_benchmark: str = "aime2024"
    train_limit: Optional[int] = None
    
    # Evaluation configuration
    eval_benchmark: str = "aime2025"
    eval_limit: Optional[int] = None
    
    # Output configuration
    output_dir: str = "evaluation_results"
    experiment_name: str = "wisent_guard_benchmark_evaluation"
    
    # ClearML configuration
    clearml_project: str = "wisent-guard-evaluation"
    clearml_tags: List[str] = None
    
    # Advanced configuration
    enable_clearml: bool = True
    save_intermediate_results: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        if self.clearml_tags is None:
            self.clearml_tags = ["benchmark_evaluation", self.train_benchmark, self.eval_benchmark]


class BenchmarkEvaluator:
    """Main class for benchmark evaluation pipeline."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.clearml_task = None
        self.logger = self._setup_logging()
        self.results = {}
        
        # Register all available tasks
        register_all_tasks()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ClearML if available and enabled
        if CLEARML_AVAILABLE and config.enable_clearml:
            self._setup_clearml()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_clearml(self):
        """Initialize ClearML experiment tracking."""
        try:
            self.clearml_task = Task.init(
                project_name=self.config.clearml_project,
                task_name=f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=self.config.clearml_tags
            )
            
            # Log configuration
            self.clearml_task.connect(asdict(self.config))
            self.logger.info("ClearML experiment initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize ClearML: {e}")
            self.clearml_task = None
    
    def _run_wisent_command(self, benchmark: str, phase: str, additional_args: List[str] = None) -> Dict[str, Any]:
        """Run wisent_guard command and capture results."""
        cmd = [
            "python", "-m", "wisent_guard", "tasks", benchmark,
            "--model", self.config.model_name,
            "--layer", str(self.config.layer)
        ]
        
        # Add limit if specified
        limit = getattr(self.config, f"{phase}_limit")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        
        # Add any additional arguments
        if additional_args:
            cmd.extend(additional_args)
        
        self.logger.info(f"Running {phase} command: {' '.join(cmd)}")
        
        try:
            # Run the command and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent),
                timeout=3600  # 1 hour timeout
            )
            
            # Parse results
            output_data = {
                "command": ' '.join(cmd),
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
            if result.returncode != 0:
                self.logger.error(f"{phase.capitalize()} command failed: {result.stderr}")
            else:
                self.logger.info(f"{phase.capitalize()} command completed successfully")
            
            return output_data
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"{phase.capitalize()} command timed out")
            return {
                "command": ' '.join(cmd),
                "return_code": -1,
                "stdout": "",
                "stderr": "Command timed out",
                "success": False
            }
        except Exception as e:
            self.logger.error(f"Error running {phase} command: {e}")
            return {
                "command": ' '.join(cmd),
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
    
    def train_classifier(self) -> Dict[str, Any]:
        """Train classifier on the training benchmark."""
        self.logger.info(f"Training classifier on {self.config.train_benchmark}")
        
        # Train the classifier
        train_results = self._run_wisent_command(
            self.config.train_benchmark, 
            "train",
            ["--train"]  # Add training flag if available
        )
        
        self.results["training"] = {
            "benchmark": self.config.train_benchmark,
            "model": self.config.model_name,
            "layer": self.config.layer,
            "limit": self.config.train_limit,
            "results": train_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to ClearML if available
        if self.clearml_task:
            self.clearml_task.get_logger().report_text(
                "Training Results",
                "Training Output",
                iteration=0,
                value=train_results["stdout"]
            )
        
        return train_results
    
    def evaluate_classifier(self) -> Dict[str, Any]:
        """Evaluate classifier on the evaluation benchmark."""
        self.logger.info(f"Evaluating classifier on {self.config.eval_benchmark}")
        
        # Evaluate the classifier
        eval_results = self._run_wisent_command(
            self.config.eval_benchmark,
            "eval"
        )
        
        self.results["evaluation"] = {
            "benchmark": self.config.eval_benchmark,
            "model": self.config.model_name,
            "layer": self.config.layer,
            "limit": self.config.eval_limit,
            "results": eval_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log to ClearML if available
        if self.clearml_task:
            self.clearml_task.get_logger().report_text(
                "Evaluation Results",
                "Evaluation Output", 
                iteration=0,
                value=eval_results["stdout"]
            )
        
        return eval_results
    
    def save_results(self) -> str:
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        
        # Add metadata
        self.results["metadata"] = {
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat(),
            "wisent_guard_version": "0.4.3"  # From __init__.py
        }
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
        
        # Upload to ClearML if available
        if self.clearml_task:
            self.clearml_task.upload_artifact(
                name="evaluation_results",
                artifact_object=results_file
            )
        
        return str(results_file)
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline: train on one benchmark, evaluate on another."""
        self.logger.info("Starting full benchmark evaluation pipeline")
        self.logger.info(f"Training on: {self.config.train_benchmark}")
        self.logger.info(f"Evaluating on: {self.config.eval_benchmark}")
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Layer: {self.config.layer}")
        
        # Step 1: Train classifier
        train_results = self.train_classifier()
        if not train_results["success"]:
            self.logger.error("Training failed, aborting evaluation")
            return self.results
        
        # Step 2: Evaluate classifier
        eval_results = self.evaluate_classifier() 
        
        # Step 3: Save results
        results_file = self.save_results()
        
        # Summary
        summary = {
            "training_success": train_results["success"],
            "evaluation_success": eval_results["success"],
            "results_file": results_file
        }
        
        self.logger.info("Evaluation pipeline completed")
        self.logger.info(f"Training successful: {summary['training_success']}")
        self.logger.info(f"Evaluation successful: {summary['evaluation_success']}")
        self.logger.info(f"Results saved to: {summary['results_file']}")
        
        return self.results


def create_config_from_args(args) -> EvaluationConfig:
    """Create configuration from command line arguments."""
    return EvaluationConfig(
        model_name=args.model,
        layer=args.layer,
        train_benchmark=args.train_benchmark,
        train_limit=args.train_limit,
        eval_benchmark=args.eval_benchmark, 
        eval_limit=args.eval_limit,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        clearml_project=args.clearml_project,
        clearml_tags=args.tags if args.tags else None,
        enable_clearml=args.enable_clearml,
        verbose=args.verbose
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Model Benchmark Evaluation Pipeline")
    
    # Model configuration
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "/workspace/models/llama31‑8b‑instruct‑hf"),
                       help="Model name or path")
    parser.add_argument("--layer", type=int, default=15, help="Layer to extract activations from")
    
    # Benchmark configuration
    parser.add_argument("--train-benchmark", default="aime2024", help="Benchmark to train on")
    parser.add_argument("--eval-benchmark", default="aime2025", help="Benchmark to evaluate on")
    parser.add_argument("--train-limit", type=int, help="Limit training samples")
    parser.add_argument("--eval-limit", type=int, help="Limit evaluation samples")
    
    # Output configuration
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--experiment-name", default="wisent_guard_benchmark_evaluation", 
                       help="Experiment name")
    
    # ClearML configuration
    parser.add_argument("--clearml-project", default="wisent-guard-evaluation", help="ClearML project name")
    parser.add_argument("--tags", nargs="+", help="ClearML tags")
    parser.add_argument("--disable-clearml", dest="enable_clearml", action="store_false",
                       help="Disable ClearML logging")
    
    # Other options
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run evaluation
    evaluator = BenchmarkEvaluator(config)
    results = evaluator.run_full_evaluation()
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Training benchmark: {config.train_benchmark}")
    print(f"Evaluation benchmark: {config.eval_benchmark}")
    print(f"Model: {config.model_name}")
    print(f"Layer: {config.layer}")
    
    if "training" in results:
        print(f"Training success: {results['training']['results']['success']}")
    if "evaluation" in results:
        print(f"Evaluation success: {results['evaluation']['results']['success']}")


if __name__ == "__main__":
    main()