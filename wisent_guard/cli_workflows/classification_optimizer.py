"""
Classification Parameter Optimizer for Wisent-Guard.

Runs comprehensive optimization across all 37 available tasks to find:
1. Optimal classification layer per task
2. Optimal token aggregation strategy per task
3. Best overall default layer and strategy for new tasks

Uses existing hyperparameter optimization logic from the system.
"""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from wisent_guard.core.model_config_manager import ModelConfigManager

logger = logging.getLogger(__name__)


@dataclass
class OptimizationLogEntry:
    """Single log entry for detailed optimization tracking."""

    timestamp: str
    task_name: str
    message: str
    level: str  # "info", "debug", "warning", "error"
    data: Optional[Dict[str, Any]] = None  # Additional structured data


@dataclass
class DetailedOptimizationLogs:
    """Container for all detailed optimization logs."""

    optimization_start_time: str
    optimization_end_time: Optional[str]
    model_name: str
    configuration: Dict[str, Any]  # Optimization configuration
    task_logs: Dict[str, List[OptimizationLogEntry]]  # task_name -> log entries
    global_logs: List[OptimizationLogEntry]  # Global optimization logs
    task_progress: Dict[str, Dict[str, Any]]  # task_name -> progress data
    timing_data: Dict[str, float]  # Various timing measurements

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "optimization_start_time": self.optimization_start_time,
            "optimization_end_time": self.optimization_end_time,
            "model_name": self.model_name,
            "configuration": self.configuration,
            "task_logs": {
                task_name: [asdict(entry) for entry in entries] for task_name, entries in self.task_logs.items()
            },
            "global_logs": [asdict(entry) for entry in self.global_logs],
            "task_progress": self.task_progress,
            "timing_data": self.timing_data,
        }


class DetailedLogger:
    """Detailed logger for capturing optimization progress."""

    def __init__(self, model_name: str, configuration: Dict[str, Any]):
        self.logs = DetailedOptimizationLogs(
            optimization_start_time=datetime.now().isoformat(),
            optimization_end_time=None,
            model_name=model_name,
            configuration=configuration,
            task_logs={},
            global_logs=[],
            task_progress={},
            timing_data={},
        )
        self.current_task = None

    def log_global(self, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):
        """Log a global optimization message."""
        entry = OptimizationLogEntry(
            timestamp=datetime.now().isoformat(), task_name="", message=message, level=level, data=data
        )
        self.logs.global_logs.append(entry)

        # Also print to console if debug level
        if level == "debug":
            logger.debug(f"🔍 {message}")
        elif level == "info":
            logger.info(f"📊 {message}")
        elif level == "warning":
            logger.warning(f"⚠️ {message}")
        elif level == "error":
            logger.error(f"❌ {message}")

    def log_task(self, task_name: str, message: str, level: str = "info", data: Optional[Dict[str, Any]] = None):
        """Log a task-specific message."""
        if task_name not in self.logs.task_logs:
            self.logs.task_logs[task_name] = []

        entry = OptimizationLogEntry(
            timestamp=datetime.now().isoformat(), task_name=task_name, message=message, level=level, data=data
        )
        self.logs.task_logs[task_name].append(entry)

        # Also print to console
        if level == "debug":
            logger.debug(f"🔍 [{task_name}] {message}")
        elif level == "info":
            logger.info(f"📊 [{task_name}] {message}")
        elif level == "warning":
            logger.warning(f"⚠️ [{task_name}] {message}")
        elif level == "error":
            logger.error(f"❌ [{task_name}] {message}")

    def set_task_progress(self, task_name: str, progress_data: Dict[str, Any]):
        """Set progress data for a task."""
        self.logs.task_progress[task_name] = progress_data

    def set_timing(self, key: str, duration: float):
        """Set timing data."""
        self.logs.timing_data[key] = duration

    def finish_optimization(self):
        """Mark optimization as finished."""
        self.logs.optimization_end_time = datetime.now().isoformat()

    def save_to_json(self, filepath: str):
        """Save detailed logs to JSON file."""
        try:
            with open(filepath, "w") as f:
                json.dump(self.logs.to_dict(), f, indent=2)
            logger.info(f"💾 Detailed optimization logs saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save detailed logs: {e}")
            raise


@dataclass
class TaskOptimizationResult:
    """Results from optimizing a single task."""

    task_name: str
    best_layer: int
    best_aggregation: str
    best_threshold: float
    best_f1: float
    best_accuracy: float
    optimization_time_seconds: float
    total_configurations_tested: int
    convergence_epoch: Optional[int] = None
    error_message: Optional[str] = None
    classifier_save_path: Optional[str] = None  # Path where best classifier was saved
    classifier_metadata: Optional[Dict[str, Any]] = None  # Metadata about saved classifier


@dataclass
class ClassificationOptimizationSummary:
    """Summary of optimization across all tasks."""

    model_name: str
    total_tasks_tested: int
    successful_optimizations: int
    failed_optimizations: int
    total_time_minutes: float
    overall_best_layer: int
    overall_best_aggregation: str
    overall_best_threshold: float
    layer_frequency_analysis: Dict[int, int]  # layer -> count of tasks where it was optimal
    aggregation_frequency_analysis: Dict[str, int]  # aggregation -> count of tasks
    task_results: List[TaskOptimizationResult]
    optimization_date: str


class ClassificationOptimizer:
    """
    Comprehensive classification parameter optimizer.

    Runs optimization across all 37 available tasks to find optimal parameters
    for classification at both task-specific and model-wide levels.
    """

    def __init__(self, model_name: str, device: str = None, verbose: bool = False, tasks: Optional[List[str]] = None):
        """
        Initialize classification optimizer.

        Args:
            model_name: Name/path of the model to optimize
            device: Device to run optimization on
            verbose: Enable verbose logging
            tasks: List of specific tasks to optimize (if None, uses all available tasks)
        """
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.config_manager = ModelConfigManager()

        # Import available tasks
        try:
            from ..cli import AVAILABLE_BENCHMARKS

            all_available_tasks = list(AVAILABLE_BENCHMARKS.keys())

            # If specific tasks are provided, use only those (if they're valid)
            if tasks:
                self.available_tasks = [task for task in tasks if task in all_available_tasks]
                if not self.available_tasks:
                    raise ValueError(
                        f"None of the specified tasks {tasks} are available. Available tasks: {all_available_tasks}"
                    )
                if len(self.available_tasks) != len(tasks):
                    invalid_tasks = [task for task in tasks if task not in all_available_tasks]
                    logger.warning(f"Some specified tasks are not available: {invalid_tasks}")
                logger.info(f"✅ Using {len(self.available_tasks)} specified tasks for optimization")
            else:
                self.available_tasks = all_available_tasks
                logger.info(f"✅ Found {len(self.available_tasks)} available tasks for optimization")
        except ImportError:
            logger.error("❌ Could not import available tasks from CLI")
            self.available_tasks = []

    def _get_classifier_save_directory(self, save_classifiers_dir: Optional[str] = None) -> str:
        """
        Get the directory where classifiers should be saved.

        Args:
            save_classifiers_dir: Custom directory, if None uses default

        Returns:
            Path to classifier save directory
        """
        if save_classifiers_dir:
            base_dir = save_classifiers_dir
        else:
            # Use default: ./optimized_classifiers/model_name/
            safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
            base_dir = f"./optimized_classifiers/{safe_model_name}"

        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def run_comprehensive_optimization(
        self,
        limit: int = 1000,
        optimization_metric: str = "f1",
        max_time_per_task_minutes: Optional[float] = None,
        layer_range: Optional[str] = None,
        aggregation_methods: Optional[List[str]] = None,
        threshold_range: Optional[List[float]] = None,
        save_results: bool = True,
        results_file: Optional[str] = None,
        save_logs_json: Optional[str] = None,
        save_classifiers: bool = True,
        classifiers_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
    ) -> ClassificationOptimizationSummary:
        """
        Run comprehensive classification optimization across all available tasks.

        Args:
            limit: Maximum samples per task (default: 1000)
            optimization_metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
            max_time_per_task_minutes: Maximum time to spend per task
            layer_range: Layer range to test (e.g., "10-20", if None uses all layers)
            aggregation_methods: Token aggregation methods to test
            threshold_range: Detection thresholds to test
            save_results: Whether to save results to file
            results_file: Custom results file path
            save_logs_json: Path to save detailed logs as JSON
            save_classifiers: Whether to save the best classifier for each task (default: True)
            classifiers_dir: Directory to save classifiers (default: ./optimized_classifiers/model_name/)
            progress_callback: Optional callback function(task_idx, task_name, status) for progress updates

        Returns:
            ClassificationOptimizationSummary with comprehensive results
        """
        if not self.available_tasks:
            raise ValueError("No available tasks found for optimization")

        # Initialize detailed logger if JSON logging is requested
        detailed_logger = None
        if save_logs_json:
            config = {
                "limit": limit,
                "optimization_metric": optimization_metric,
                "max_time_per_task_minutes": max_time_per_task_minutes,
                "layer_range": layer_range,
                "aggregation_methods": aggregation_methods,
                "threshold_range": threshold_range,
                "total_tasks": len(self.available_tasks),
                "task_list": self.available_tasks,
                "save_classifiers": save_classifiers,
                "classifiers_dir": classifiers_dir,
            }
            detailed_logger = DetailedLogger(self.model_name, config)

        # Set up classifier saving directory if enabled
        classifier_save_dir = None
        if save_classifiers:
            classifier_save_dir = self._get_classifier_save_directory(classifiers_dir)
            logger.info(f"💾 Classifiers will be saved to: {classifier_save_dir}")
            if detailed_logger:
                detailed_logger.log_global(
                    f"Classifier save directory: {classifier_save_dir}",
                    "info",
                    {"classifier_save_dir": classifier_save_dir},
                )

        logger.info("🚀 Starting comprehensive classification optimization")
        logger.info(f"   📊 Model: {self.model_name}")
        logger.info(f"   📋 Tasks: {len(self.available_tasks)} available tasks")
        logger.info(f"   🔢 Limit per task: {limit}")
        logger.info(f"   📈 Metric: {optimization_metric}")
        if max_time_per_task_minutes is not None:
            logger.info(f"   ⏱️  Max time per task: {max_time_per_task_minutes:.1f} minutes")
        else:
            logger.info("   ⏱️  Max time per task: No limit")

        if detailed_logger:
            detailed_logger.log_global(
                "Starting comprehensive classification optimization",
                "info",
                {
                    "model": self.model_name,
                    "total_tasks": len(self.available_tasks),
                    "configuration": detailed_logger.logs.configuration,
                },
            )

        start_time = time.time()
        task_results = []
        successful_count = 0
        failed_count = 0

        # Default parameter ranges if not specified
        if aggregation_methods is None:
            aggregation_methods = ["average", "final", "first", "max", "min"]

        if threshold_range is None:
            threshold_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for i, task_name in enumerate(self.available_tasks, 1):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📊 OPTIMIZING TASK {i}/{len(self.available_tasks)}: {task_name.upper()}")
            logger.info(f"{'=' * 60}")

            task_start_time = time.time()

            # Call progress callback with "started" status
            if progress_callback:
                progress_callback(i - 1, task_name, "started")

            if detailed_logger:
                detailed_logger.log_task(
                    task_name,
                    f"Starting optimization (task {i}/{len(self.available_tasks)})",
                    "info",
                    {
                        "task_index": i,
                        "total_tasks": len(self.available_tasks),
                        "progress_percentage": (i - 1) / len(self.available_tasks) * 100,
                    },
                )

            try:
                # Run optimization for this specific task
                result = self._optimize_single_task(
                    task_name=task_name,
                    limit=limit,
                    optimization_metric=optimization_metric,
                    max_time_minutes=max_time_per_task_minutes,
                    layer_range=layer_range,
                    aggregation_methods=aggregation_methods,
                    threshold_range=threshold_range,
                    detailed_logger=detailed_logger,
                    save_classifier=save_classifiers,
                    classifier_save_dir=classifier_save_dir,
                )

                task_results.append(result)
                successful_count += 1

                logger.info(f"✅ Task {task_name} completed successfully!")
                logger.info(f"   🎯 Best layer: {result.best_layer}")
                logger.info(f"   🔧 Best aggregation: {result.best_aggregation}")
                logger.info(f"   📊 Best F1: {result.best_f1:.3f}")
                logger.info(f"   ⏱️  Time: {result.optimization_time_seconds:.1f}s")

                # Call progress callback with "completed" status
                if progress_callback:
                    progress_callback(i - 1, task_name, "completed")

                if detailed_logger:
                    detailed_logger.log_task(
                        task_name,
                        "Task completed successfully",
                        "info",
                        {
                            "success": True,
                            "best_layer": result.best_layer,
                            "best_aggregation": result.best_aggregation,
                            "best_threshold": result.best_threshold,
                            "best_f1": result.best_f1,
                            "best_accuracy": result.best_accuracy,
                            "optimization_time_seconds": result.optimization_time_seconds,
                            "configurations_tested": result.total_configurations_tested,
                        },
                    )
                    detailed_logger.set_task_progress(task_name, {"status": "completed", "result": asdict(result)})

            except Exception as e:
                # Skip failing benchmark and log error
                failed_count += 1
                logger.error(f"⚠️  Task {task_name} failed (skipping): {e}")

                # Import traceback for full stack trace
                import traceback

                error_details = {
                    "task_name": task_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_index": i,
                    "total_tasks": len(self.available_tasks),
                    "traceback": traceback.format_exc(),
                }

                # Log to error tracking file
                self._log_failed_benchmark(error_details)

                print(f"\n⚠️  SKIPPING FAILED TASK: {task_name}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {e!s}")
                print("   See failed_benchmarks.json for details\n")

                if detailed_logger:
                    detailed_logger.log_task(
                        task_name,
                        f"Task failed (skipped): {e!s}",
                        "error",
                        {
                            "success": False,
                            "error_message": str(e),
                            "error_type": type(e).__name__,
                            "optimization_time_seconds": time.time() - task_start_time,
                            "skipped": True,
                        },
                    )
                    detailed_logger.set_task_progress(task_name, {"status": "failed_skipped", "error": str(e)})

                # Call progress callback with "failed" status
                if progress_callback:
                    progress_callback(i - 1, task_name, "failed")

                # Continue to next task instead of raising
                continue

        # Analyze results across all tasks
        total_time = time.time() - start_time
        summary = self._analyze_optimization_results(
            task_results=task_results,
            total_time_seconds=total_time,
            successful_count=successful_count,
            failed_count=failed_count,
        )

        # Save results if requested
        if save_results:
            self._save_optimization_results(summary, results_file)

        # Save optimal parameters to model config
        self._save_optimal_parameters_to_config(summary)

        # Add timing data and final logs to detailed logger
        if detailed_logger:
            detailed_logger.set_timing("total_optimization_seconds", total_time)
            detailed_logger.set_timing("average_time_per_task", total_time / len(self.available_tasks))
            detailed_logger.set_timing(
                "average_time_per_successful_task", total_time / successful_count if successful_count > 0 else 0
            )

            detailed_logger.log_global(
                "Optimization completed",
                "info",
                {
                    "successful_tasks": successful_count,
                    "failed_tasks": failed_count,
                    "total_time_minutes": total_time / 60.0,
                    "overall_best_layer": summary.overall_best_layer,
                    "overall_best_aggregation": summary.overall_best_aggregation,
                    "overall_best_threshold": summary.overall_best_threshold,
                },
            )

            detailed_logger.finish_optimization()

            # Save detailed logs to JSON
            detailed_logger.save_to_json(save_logs_json)

        # Print final summary
        self._print_optimization_summary(summary)

        return summary

    def _optimize_single_task(
        self,
        task_name: str,
        limit: int,
        optimization_metric: str,
        max_time_minutes: float,
        layer_range: Optional[str],
        aggregation_methods: List[str],
        threshold_range: List[float],
        detailed_logger: Optional[DetailedLogger] = None,
        save_classifier: bool = False,
        classifier_save_dir: Optional[str] = None,
    ) -> TaskOptimizationResult:
        """
        Optimize parameters for a single task using existing hyperparameter optimizer.

        Args:
            task_name: Name of the task to optimize
            limit: Maximum samples for this task
            optimization_metric: Metric to optimize
            max_time_minutes: Maximum time for this task
            layer_range: Layer range to test
            aggregation_methods: Aggregation methods to test
            threshold_range: Thresholds to test

        Returns:
            TaskOptimizationResult with optimization results
        """
        logger.info(f"🔧 Starting optimization for task: {task_name}")

        if detailed_logger:
            detailed_logger.log_task(
                task_name,
                "Initializing hyperparameter optimization",
                "debug",
                {
                    "layer_range": layer_range or "all",
                    "aggregation_methods": aggregation_methods,
                    "threshold_range": threshold_range,
                    "max_time_minutes": max_time_minutes,
                    "max_combinations": 200,
                },
            )

        task_start_time = time.time()

        # Use the existing run_task_pipeline with optimize=True
        # This leverages the tested optimization logic from the CLI
        try:
            from ..cli import run_task_pipeline

            # Convert layer_range string to optimize_layers format
            optimize_layers_str = layer_range if layer_range else "all"  # Default: test all layers

            # Run optimization using the existing pipeline
            optimization_results = run_task_pipeline(
                task_name=task_name,
                model_name=self.model_name,
                layer="15",  # Will be optimized anyway
                limit=limit,
                optimize=True,
                optimize_layers=optimize_layers_str,
                optimize_metric=optimization_metric,
                optimize_max_combinations=200,
                split_ratio=0.8,
                classifier_type="logistic",
                seed=42,
                device=self.device,
                verbose=self.verbose,
                allow_small_dataset=True,
            )
        except Exception as e:
            # Re-raise the exception to be handled by the outer try-catch in optimize_all_tasks
            # This allows the skip-and-continue logic to work properly
            logger.debug(f"Single task optimization failed for {task_name}, re-raising to outer handler")
            raise e

        # Check if the optimization failed (e.g., evaluation returned error)
        if optimization_results.get("error", False):
            opt_result = optimization_results.get("optimization_result", {})
            error_msg = opt_result.get("error", "Unknown error during optimization")
            raise RuntimeError(f"Task optimization failed: {error_msg}")

        # Extract best results from run_task_pipeline optimization
        optimization_result = optimization_results.get("optimization_result", {})
        training_results = optimization_results.get("training_results", {})

        best_layer = optimization_result.get("best_layer", 15)
        best_aggregation = optimization_result.get("best_aggregation", "average")
        best_threshold = optimization_result.get("best_threshold", 0.6)
        # Ensure numeric values (handle string inputs like 'N/A')
        f1_value = training_results.get("f1", 0.0)
        accuracy_value = training_results.get("accuracy", 0.0)

        # Convert to float, handling 'N/A' and other non-numeric values
        try:
            best_f1 = float(f1_value) if f1_value != "N/A" else 0.0
        except (ValueError, TypeError):
            best_f1 = 0.0

        try:
            best_accuracy = float(accuracy_value) if accuracy_value != "N/A" else 0.0
        except (ValueError, TypeError):
            best_accuracy = 0.0

        optimization_time = time.time() - task_start_time

        # Save the best classifier if requested and available
        classifier_save_path = None
        classifier_metadata = None

        if save_classifier and classifier_save_dir:
            try:
                # Create classifier save path
                classifier_filename = f"{task_name}_classifier_layer_{best_layer}.pkl"
                classifier_save_path = os.path.join(classifier_save_dir, classifier_filename)

                # Create comprehensive metadata
                classifier_metadata = {
                    "model_name": self.model_name,
                    "task_name": task_name,
                    "layer": best_layer,
                    "token_aggregation": best_aggregation,
                    "detection_threshold": best_threshold,
                    "classifier_type": "logistic",
                    "optimization_metric": optimization_metric,
                    "training_samples": limit,
                    "split_ratio": 0.8,
                    "optimization_time_seconds": optimization_time,
                    "performance_metrics": {"f1": best_f1, "accuracy": best_accuracy},
                    "training_date": datetime.now().isoformat(),
                    "configurations_tested": optimization_result.get("total_combinations_tested", 0),
                    "optimizer_config": {
                        "layer_range": layer_range,
                        "aggregation_methods": aggregation_methods,
                        "threshold_range": threshold_range,
                        "max_time_minutes": max_time_minutes,
                    },
                }

                # Note: The classifier is saved by the run_task_pipeline function
                # We just track the path and metadata here
                if detailed_logger:
                    detailed_logger.log_task(
                        task_name,
                        "Optimization completed - classifier handled by pipeline",
                        "info",
                        {
                            "best_layer": best_layer,
                            "best_aggregation": best_aggregation,
                            "f1_score": best_f1,
                            "accuracy": best_accuracy,
                            "classifier_metadata": classifier_metadata,
                        },
                    )

                logger.info(f"💾 Task {task_name} optimization completed with layer {best_layer}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to process classifier metadata for {task_name}: {e}")
                if detailed_logger:
                    detailed_logger.log_task(
                        task_name, f"Failed to process classifier metadata: {e!s}", "warning", {"error": str(e)}
                    )

        # Create result object
        result = TaskOptimizationResult(
            task_name=task_name,
            best_layer=best_layer,
            best_aggregation=best_aggregation,
            best_threshold=best_threshold,
            best_f1=best_f1,
            best_accuracy=best_accuracy,
            optimization_time_seconds=optimization_time,
            total_configurations_tested=optimization_result.get("total_combinations_tested", 0),
            convergence_epoch=optimization_result.get("convergence_epoch"),
            classifier_save_path=classifier_save_path,
            classifier_metadata=classifier_metadata,
        )

        return result

    def _analyze_optimization_results(
        self,
        task_results: List[TaskOptimizationResult],
        total_time_seconds: float,
        successful_count: int,
        failed_count: int,
    ) -> ClassificationOptimizationSummary:
        """
        Analyze optimization results across all tasks to find best overall parameters.

        Args:
            task_results: Results from all task optimizations
            total_time_seconds: Total optimization time
            successful_count: Number of successful optimizations
            failed_count: Number of failed optimizations

        Returns:
            ClassificationOptimizationSummary with analysis
        """
        logger.info(f"📊 Analyzing optimization results across {len(task_results)} tasks")

        # Filter successful results
        successful_results = [r for r in task_results if r.error_message is None]

        if not successful_results:
            raise ValueError("No successful task optimizations to analyze")

        # Frequency analysis - count how often each parameter was optimal
        layer_frequency = {}
        aggregation_frequency = {}

        for result in successful_results:
            # Count layer frequency
            layer = result.best_layer
            layer_frequency[layer] = layer_frequency.get(layer, 0) + 1

            # Count aggregation frequency
            aggregation = result.best_aggregation
            aggregation_frequency[aggregation] = aggregation_frequency.get(aggregation, 0) + 1

        # Find most frequent (likely best overall) parameters
        most_common_layer = max(layer_frequency.items(), key=lambda x: x[1])[0]
        most_common_aggregation = max(aggregation_frequency.items(), key=lambda x: x[1])[0]

        # Calculate weighted average threshold based on performance
        total_weighted_f1 = sum(r.best_f1 for r in successful_results)
        if total_weighted_f1 > 0:
            weighted_avg_threshold = sum(r.best_threshold * r.best_f1 for r in successful_results) / total_weighted_f1
        else:
            weighted_avg_threshold = 0.6  # Default fallback

        # Create summary
        summary = ClassificationOptimizationSummary(
            model_name=self.model_name,
            total_tasks_tested=len(task_results),
            successful_optimizations=successful_count,
            failed_optimizations=failed_count,
            total_time_minutes=total_time_seconds / 60.0,
            overall_best_layer=most_common_layer,
            overall_best_aggregation=most_common_aggregation,
            overall_best_threshold=round(weighted_avg_threshold, 2),
            layer_frequency_analysis=layer_frequency,
            aggregation_frequency_analysis=aggregation_frequency,
            task_results=task_results,
            optimization_date=datetime.now().isoformat(),
        )

        return summary

    def _save_optimization_results(
        self, summary: ClassificationOptimizationSummary, results_file: Optional[str] = None
    ) -> str:
        """
        Save comprehensive optimization results to JSON file.

        Args:
            summary: Optimization summary to save
            results_file: Custom file path, if None uses default naming

        Returns:
            Path to saved results file
        """
        if results_file is None:
            # Create default filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_model_name = self.model_name.replace("/", "_").replace(":", "_")
            results_file = f"classification_optimization_{safe_model_name}_{timestamp}.json"

        # Convert summary to dict for JSON serialization
        results_data = {
            "model_name": summary.model_name,
            "optimization_metadata": {
                "total_tasks_tested": summary.total_tasks_tested,
                "successful_optimizations": summary.successful_optimizations,
                "failed_optimizations": summary.failed_optimizations,
                "total_time_minutes": summary.total_time_minutes,
                "optimization_date": summary.optimization_date,
            },
            "overall_optimal_parameters": {
                "best_layer": summary.overall_best_layer,
                "best_aggregation": summary.overall_best_aggregation,
                "best_threshold": summary.overall_best_threshold,
            },
            "frequency_analysis": {
                "layer_frequency": summary.layer_frequency_analysis,
                "aggregation_frequency": summary.aggregation_frequency_analysis,
            },
            "task_specific_results": [
                {
                    "task_name": r.task_name,
                    "best_layer": r.best_layer,
                    "best_aggregation": r.best_aggregation,
                    "best_threshold": r.best_threshold,
                    "best_f1": r.best_f1,
                    "best_accuracy": r.best_accuracy,
                    "optimization_time_seconds": r.optimization_time_seconds,
                    "total_configurations_tested": r.total_configurations_tested,
                    "convergence_epoch": r.convergence_epoch,
                    "error_message": r.error_message,
                    "classifier_save_path": r.classifier_save_path,
                    "classifier_metadata": r.classifier_metadata,
                }
                for r in summary.task_results
            ],
        }

        try:
            with open(results_file, "w") as f:
                json.dump(results_data, f, indent=2)

            logger.info(f"💾 Optimization results saved to: {results_file}")
            return results_file

        except Exception as e:
            logger.error(f"❌ Failed to save optimization results: {e}")
            raise

    def _save_optimal_parameters_to_config(self, summary: ClassificationOptimizationSummary):
        """
        Save optimal parameters to model configuration for future use.

        Args:
            summary: Optimization summary with optimal parameters
        """
        logger.info("💾 Saving optimal parameters to model configuration...")

        # Create task-specific overrides for tasks that have different optimal parameters
        task_specific_overrides = {}

        for result in summary.task_results:
            if result.error_message is not None:
                continue  # Skip failed optimizations

            # Only save override if task-specific params differ from overall best
            if (
                result.best_layer != summary.overall_best_layer
                or result.best_aggregation != summary.overall_best_aggregation
                or abs(result.best_threshold - summary.overall_best_threshold) > 0.05
            ):
                task_specific_overrides[result.task_name] = {
                    "classification_layer": result.best_layer,
                    "token_aggregation": result.best_aggregation,
                    "detection_threshold": result.best_threshold,
                }

        # Prepare optimization metrics summary
        successful_results = [r for r in summary.task_results if r.error_message is None]
        optimization_metrics = {
            "average_f1_across_tasks": sum(r.best_f1 for r in successful_results) / len(successful_results)
            if successful_results
            else 0.0,
            "average_accuracy_across_tasks": sum(r.best_accuracy for r in successful_results) / len(successful_results)
            if successful_results
            else 0.0,
            "total_tasks_optimized": summary.successful_optimizations,
            "optimization_time_minutes": summary.total_time_minutes,
            "most_common_layer": summary.overall_best_layer,
            "layer_consistency_percentage": (
                summary.layer_frequency_analysis.get(summary.overall_best_layer, 0)
                / summary.successful_optimizations
                * 100
            )
            if summary.successful_optimizations > 0
            else 0.0,
        }

        # Save to model configuration
        self.config_manager.save_model_config(
            model_name=self.model_name,
            classification_layer=summary.overall_best_layer,
            steering_layer=summary.overall_best_layer,  # Use same layer for steering by default
            token_aggregation=summary.overall_best_aggregation,
            detection_threshold=summary.overall_best_threshold,
            optimization_method="comprehensive_classification_optimization",
            optimization_metrics=optimization_metrics,
            task_specific_overrides=task_specific_overrides,
        )

        logger.info(f"✅ Model configuration saved with {len(task_specific_overrides)} task-specific overrides")

    def _log_failed_benchmark(self, error_details: Dict[str, Any]):
        """
        Log failed benchmark to a JSON file for tracking.

        Args:
            error_details: Dictionary containing error information
        """
        # Get model directory from config manager
        model_dir = os.path.join(os.path.expanduser("~/.wisent-guard/model_configs"), self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        error_file = os.path.join(model_dir, "failed_benchmarks.json")

        # Load existing errors if file exists
        if os.path.exists(error_file):
            try:
                with open(error_file) as f:
                    errors_data = json.load(f)
            except:
                errors_data = {"failed_benchmarks": []}
        else:
            errors_data = {"failed_benchmarks": []}

        # Add new error
        errors_data["failed_benchmarks"].append(error_details)

        # Update metadata
        errors_data["metadata"] = {
            "total_failures": len(errors_data["failed_benchmarks"]),
            "last_updated": datetime.now().isoformat(),
            "model_name": self.model_name,
        }

        # Save updated error log
        with open(error_file, "w") as f:
            json.dump(errors_data, f, indent=2)

        logger.info(f"📝 Error logged to {error_file}")

    def _print_optimization_summary(self, summary: ClassificationOptimizationSummary):
        """
        Print comprehensive optimization summary to console.

        Args:
            summary: Optimization summary to print
        """
        print(f"\n{'=' * 80}")
        print("🎯 CLASSIFICATION OPTIMIZATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"🤖 Model: {summary.model_name}")
        print(f"📅 Date: {summary.optimization_date}")
        print(f"⏱️  Total Time: {summary.total_time_minutes:.1f} minutes")
        print(f"📊 Tasks: {summary.successful_optimizations}/{summary.total_tasks_tested} successful")

        if summary.failed_optimizations > 0:
            print(f"❌ Failed: {summary.failed_optimizations} tasks")

        print("\n🎯 OPTIMAL PARAMETERS (Overall Best):")
        print(f"   📊 Classification Layer: {summary.overall_best_layer}")
        print(f"   🔧 Token Aggregation: {summary.overall_best_aggregation}")
        print(f"   📈 Detection Threshold: {summary.overall_best_threshold}")

        print("\n📈 PARAMETER FREQUENCY ANALYSIS:")
        print("   🏆 Layer Frequency (top 5):")
        sorted_layers = sorted(summary.layer_frequency_analysis.items(), key=lambda x: x[1], reverse=True)
        for layer, count in sorted_layers[:5]:
            percentage = (count / summary.successful_optimizations) * 100
            print(f"      Layer {layer}: {count} tasks ({percentage:.1f}%)")

        print("   🔧 Aggregation Frequency:")
        sorted_agg = sorted(summary.aggregation_frequency_analysis.items(), key=lambda x: x[1], reverse=True)
        for agg, count in sorted_agg:
            percentage = (count / summary.successful_optimizations) * 100
            print(f"      {agg}: {count} tasks ({percentage:.1f}%)")

        # Show top performing tasks
        successful_results = [r for r in summary.task_results if r.error_message is None]
        if successful_results:
            print("\n🏆 TOP PERFORMING TASKS (by F1 score):")
            top_tasks = sorted(successful_results, key=lambda x: x.best_f1, reverse=True)[:5]
            for result in top_tasks:
                print(
                    f"   {result.task_name}: F1={result.best_f1:.3f}, Layer={result.best_layer}, Agg={result.best_aggregation}"
                )

        # Show failed tasks if any
        failed_results = [r for r in summary.task_results if r.error_message is not None]
        if failed_results:
            print("\n❌ FAILED TASKS:")
            for result in failed_results:
                print(f"   {result.task_name}: {result.error_message}")

        # Show classifier saving information
        saved_classifiers = [r for r in summary.task_results if r.classifier_save_path is not None]
        if saved_classifiers:
            print("\n💾 SAVED CLASSIFIERS:")
            print(f"   📁 Total classifiers saved: {len(saved_classifiers)}")
            if saved_classifiers:
                # Show the directory where classifiers are saved
                first_saved_path = saved_classifiers[0].classifier_save_path
                classifier_dir = os.path.dirname(first_saved_path)
                print(f"   📂 Directory: {classifier_dir}")
                print("   🔍 Classifiers can be auto-discovered by the agent system")

        print("\n✅ Configuration saved to model config with task-specific overrides")
        print(f"{'=' * 80}")


# Convenience functions for CLI integration
def run_classification_optimization(
    model_name: str,
    limit: int = 1000,
    device: str = None,
    verbose: bool = False,
    tasks: Optional[List[str]] = None,
    skip_confirmation: bool = False,
    **kwargs,
) -> ClassificationOptimizationSummary:
    """
    Convenience function to run classification optimization.

    Args:
        model_name: Model to optimize
        limit: Sample limit per task
        device: Device to use
        verbose: Enable verbose logging
        tasks: List of specific tasks to optimize (if None, uses all available tasks)
        skip_confirmation: Skip confirmation prompt for long optimizations
        **kwargs: Additional arguments for optimization (including save_logs_json)

    Returns:
        ClassificationOptimizationSummary with results
    """
    optimizer = ClassificationOptimizer(model_name=model_name, device=device, verbose=verbose, tasks=tasks)

    # Estimate time if not skipping confirmation
    if not skip_confirmation:
        # Simple time estimation based on task count and limit
        num_tasks = len(tasks) if tasks else len(optimizer.available_tasks)
        estimated_seconds_per_task = 15 + (limit * 0.1)  # Base time + per-sample time
        total_estimated_seconds = num_tasks * estimated_seconds_per_task

        # Check if over 1 hour
        if total_estimated_seconds > 3600:
            hours = total_estimated_seconds / 3600
            print(f"\n⚠️  WARNING: Classification optimization estimated to take {hours:.1f} hours")
            print(f"   • Tasks to optimize: {num_tasks}")
            print(f"   • Samples per task: {limit}")

            import sys

            while True:
                response = input("\n   Do you want to continue? (y/n): ").strip().lower()
                if response == "y" or response == "yes":
                    print("\n✅ Continuing with optimization...")
                    break
                if response == "n" or response == "no":
                    print("\n❌ Optimization cancelled by user.")
                    sys.exit(0)
                else:
                    print("   Please enter 'y' for yes or 'n' for no.")

    return optimizer.run_comprehensive_optimization(limit=limit, **kwargs)
