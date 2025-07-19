"""
Benchmark configuration and result structures for evaluation orchestration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json
import os


class BenchmarkType(Enum):
    """Supported benchmark types."""
    LIVECODEBENCH = "livecodebench"
    BIGCODE = "bigcode"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


@dataclass
class BenchmarkConfig:
    """Configuration for a specific benchmark."""
    
    name: str
    benchmark_type: BenchmarkType
    
    # Model configuration
    model_path: str
    model_format: str = "huggingface"  # huggingface, vllm, etc.
    
    # Evaluation parameters
    temperature: float = 0.2
    max_length: int = 512
    n_samples: int = 1  # for pass@k evaluation
    batch_size: int = 1
    timeout: int = 30
    
    # Benchmark-specific parameters
    dataset_version: Optional[str] = None  # e.g., "v1", "v2" for LiveCodeBench
    task_subset: Optional[str] = None  # e.g., "python" for language-specific tasks
    problem_limit: Optional[int] = None  # limit number of problems for testing
    
    # Execution environment
    allow_code_execution: bool = True
    use_docker: bool = True
    gpu_device: Optional[str] = "auto"
    
    # Output configuration
    output_dir: str = "./evaluation_results"
    cache_generations: bool = True
    continue_existing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "name": self.name,
            "benchmark_type": self.benchmark_type.value,
            "model_path": self.model_path,
            "model_format": self.model_format,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "dataset_version": self.dataset_version,
            "task_subset": self.task_subset,
            "problem_limit": self.problem_limit,
            "allow_code_execution": self.allow_code_execution,
            "use_docker": self.use_docker,
            "gpu_device": self.gpu_device,
            "output_dir": self.output_dir,
            "cache_generations": self.cache_generations,
            "continue_existing": self.continue_existing
        }


@dataclass
class BenchmarkResult:
    """Results from a benchmark evaluation."""
    
    benchmark_name: str
    benchmark_type: BenchmarkType
    model_path: str
    
    # Evaluation metrics
    pass_at_1: Optional[float] = None
    pass_at_5: Optional[float] = None
    pass_at_10: Optional[float] = None
    
    # Additional metrics
    total_problems: int = 0
    solved_problems: int = 0
    compilation_rate: Optional[float] = None
    average_time: Optional[float] = None
    
    # Metadata
    timestamp: str = None
    duration_seconds: float = 0.0
    config: Optional[BenchmarkConfig] = None
    
    # Raw results
    detailed_results: Dict[str, Any] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.error_messages is None:
            self.error_messages = []
        if self.detailed_results is None:
            self.detailed_results = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "benchmark_type": self.benchmark_type.value,
            "model_path": self.model_path,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "pass_at_10": self.pass_at_10,
            "total_problems": self.total_problems,
            "solved_problems": self.solved_problems,
            "compilation_rate": self.compilation_rate,
            "average_time": self.average_time,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "config": self.config.to_dict() if self.config else None,
            "detailed_results": self.detailed_results,
            "error_messages": self.error_messages
        }
    
    def save_to_file(self, filepath: str):
        """Save result to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "BenchmarkResult":
        """Load result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert config back to BenchmarkConfig if present
        config = None
        if data.get("config"):
            config_data = data["config"]
            config_data["benchmark_type"] = BenchmarkType(config_data["benchmark_type"])
            config = BenchmarkConfig(**config_data)
        
        return cls(
            benchmark_name=data["benchmark_name"],
            benchmark_type=BenchmarkType(data["benchmark_type"]),
            model_path=data["model_path"],
            pass_at_1=data.get("pass_at_1"),
            pass_at_5=data.get("pass_at_5"),
            pass_at_10=data.get("pass_at_10"),
            total_problems=data.get("total_problems", 0),
            solved_problems=data.get("solved_problems", 0),
            compilation_rate=data.get("compilation_rate"),
            average_time=data.get("average_time"),
            timestamp=data.get("timestamp"),
            duration_seconds=data.get("duration_seconds", 0.0),
            config=config,
            detailed_results=data.get("detailed_results", {}),
            error_messages=data.get("error_messages", [])
        )


class BenchmarkRegistry:
    """Registry for benchmark configurations."""
    
    _benchmarks = {}
    
    @classmethod
    def register(cls, config: BenchmarkConfig):
        """Register a benchmark configuration."""
        cls._benchmarks[config.name] = config
    
    @classmethod
    def get(cls, name: str) -> Optional[BenchmarkConfig]:
        """Get benchmark configuration by name."""
        return cls._benchmarks.get(name)
    
    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmark names."""
        return list(cls._benchmarks.keys())
    
    @classmethod
    def get_default_configs(cls) -> Dict[str, BenchmarkConfig]:
        """Get default benchmark configurations."""
        return {
            "livecodebench": BenchmarkConfig(
                name="livecodebench",
                benchmark_type=BenchmarkType.LIVECODEBENCH,
                model_path="",  # Will be set during evaluation
                dataset_version="v2",
                n_samples=1,
                temperature=0.2,
                max_length=512,
                problem_limit=None,
                output_dir="./evaluation_results/livecodebench"
            ),
            "humaneval": BenchmarkConfig(
                name="humaneval",
                benchmark_type=BenchmarkType.HUMANEVAL,
                model_path="",  # Will be set during evaluation
                n_samples=1,
                temperature=0.2,
                max_length=512,
                problem_limit=None,
                output_dir="./evaluation_results/humaneval"
            ),
            "mbpp": BenchmarkConfig(
                name="mbpp",
                benchmark_type=BenchmarkType.MBPP,
                model_path="",  # Will be set during evaluation
                n_samples=1,
                temperature=0.2,
                max_length=512,
                problem_limit=None,
                output_dir="./evaluation_results/mbpp"
            )
        }


# Register default configurations
for config in BenchmarkRegistry.get_default_configs().values():
    BenchmarkRegistry.register(config)