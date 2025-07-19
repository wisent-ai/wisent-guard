"""
Experiment runner that orchestrates the complete pipeline from data loading to model training.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .steering_trainer import SteeringVectorTrainer, TrainingConfig, TrainingResults
from ..data_loaders import LiveCodeBenchLoader, SteeringDataExtractor
from ..data_loaders.version_manager import LiveCodeBenchVersionManager
from ..models import SteeringCompatibleModel

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    
    # Data configuration
    train_version: str = "release_v1"
    eval_version: str = "release_v2"
    version_split_type: str = "new_only"
    data_limit: Optional[int] = None
    
    # Contrastive pair configuration
    pair_strategy: str = "correct_vs_incorrect"
    pairs_per_problem: int = 1
    
    # Training configuration
    training_config: TrainingConfig = None
    
    # Output configuration
    output_directory: str = "./steering_experiments"
    experiment_name: str = "livecodebench_steering"
    
    def __post_init__(self):
        if self.training_config is None:
            self.training_config = TrainingConfig()


@dataclass
class ExperimentResults:
    """Results from a complete experiment."""
    
    config: ExperimentConfig
    training_results: TrainingResults
    data_info: Dict[str, Any]
    experiment_time: float
    timestamp: str
    steering_model: Optional[SteeringCompatibleModel] = None


class ExperimentRunner:
    """Orchestrates complete steering vector experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize components
        self.data_loader = LiveCodeBenchLoader()
        self.version_manager = LiveCodeBenchVersionManager()
        self.data_extractor = SteeringDataExtractor()
        
        logger.info(f"Initialized ExperimentRunner for {config.experiment_name}")
    
    def run_experiment(self) -> ExperimentResults:
        """
        Run a complete steering vector experiment.
        
        Returns:
            ExperimentResults with all experiment data
        """
        start_time = datetime.now()
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        try:
            # Step 1: Load and split data
            train_data, eval_data, data_info = self._load_and_split_data()
            
            # Step 2: Extract contrastive pairs
            contrastive_pairs = self._extract_contrastive_pairs(train_data)
            
            # Step 3: Train steering vectors
            training_results = self._train_steering_vectors(contrastive_pairs)
            
            # Step 4: Create steering-compatible model
            steering_model = self._create_steering_model(training_results)
            
            # Step 5: Save results
            self._save_experiment_results(training_results, data_info)
            
            # Create experiment results
            end_time = datetime.now()
            experiment_time = (end_time - start_time).total_seconds()
            
            results = ExperimentResults(
                config=self.config,
                training_results=training_results,
                data_info=data_info,
                experiment_time=experiment_time,
                timestamp=start_time.isoformat(),
                steering_model=steering_model
            )
            
            logger.info(f"Experiment completed in {experiment_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def _load_and_split_data(self) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        """Load and split data according to configuration."""
        logger.info(f"Loading data: {self.config.train_version} -> {self.config.eval_version}")
        
        # Get version split
        version_split = self.version_manager.get_version_split(
            train_version=self.config.train_version,
            eval_version=self.config.eval_version,
            split_type=self.config.version_split_type
        )
        
        train_data = version_split["train"]
        eval_data = version_split["eval"]
        
        # Apply data limit if specified
        if self.config.data_limit:
            train_data = train_data[:self.config.data_limit]
            eval_data = eval_data[:self.config.data_limit]
        
        # Create data info
        data_info = {
            "train_version": self.config.train_version,
            "eval_version": self.config.eval_version,
            "split_type": self.config.version_split_type,
            "train_problems": len(train_data),
            "eval_problems": len(eval_data),
            "contamination_free": self.config.version_split_type == "new_only",
            "data_limit": self.config.data_limit
        }
        
        logger.info(f"Data loaded: {len(train_data)} train, {len(eval_data)} eval problems")
        return train_data, eval_data, data_info
    
    def _extract_contrastive_pairs(self, train_data: List[Any]) -> List[Any]:
        """Extract contrastive pairs from training data."""
        logger.info(f"Extracting contrastive pairs using {self.config.pair_strategy}")
        
        contrastive_pairs = self.data_extractor.extract_contrastive_pairs(
            problems=train_data,
            strategy=self.config.pair_strategy,
            pairs_per_problem=self.config.pairs_per_problem
        )
        
        logger.info(f"Extracted {len(contrastive_pairs)} contrastive pairs")
        return contrastive_pairs
    
    def _train_steering_vectors(self, contrastive_pairs: List[Any]) -> TrainingResults:
        """Train steering vectors from contrastive pairs."""
        logger.info("Training steering vectors")
        
        with SteeringVectorTrainer(self.config.training_config) as trainer:
            training_results = trainer.train(contrastive_pairs)
        
        logger.info(f"Trained {len(training_results.steering_vectors)} steering vectors")
        return training_results
    
    def _create_steering_model(self, training_results: TrainingResults) -> SteeringCompatibleModel:
        """Create a steering-compatible model with trained vectors."""
        logger.info("Creating steering-compatible model")
        
        # Create model
        model = SteeringCompatibleModel.from_pretrained_with_steering(
            self.config.training_config.model_name
        )
        
        # Add steering vectors
        for layer_idx, steering_vector in training_results.steering_vectors.items():
            metadata = {
                "experiment_name": self.config.experiment_name,
                "training_method": self.config.training_config.steering_method,
                "timestamp": training_results.timestamp,
                "train_version": self.config.train_version,
                "pair_strategy": self.config.pair_strategy
            }
            
            model.add_steering_vector(layer_idx, steering_vector, metadata)
        
        logger.info(f"Created model with {len(training_results.steering_vectors)} steering vectors")
        return model
    
    def _save_experiment_results(self, training_results: TrainingResults, data_info: Dict[str, Any]):
        """Save experiment results to disk."""
        import os
        
        # Create experiment directory
        experiment_dir = os.path.join(
            self.config.output_directory,
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Save training results
        with SteeringVectorTrainer(self.config.training_config) as trainer:
            trainer.save_results(training_results, experiment_dir)
        
        # Save experiment configuration
        experiment_config = {
            "experiment_config": self.config.__dict__,
            "data_info": data_info,
            "timestamp": datetime.now().isoformat()
        }
        
        import json
        config_filepath = os.path.join(experiment_dir, "experiment_config.json")
        with open(config_filepath, "w") as f:
            json.dump(experiment_config, f, indent=2, default=str)
        
        logger.info(f"Saved experiment results to {experiment_dir}")
    
    def run_quick_experiment(
        self,
        model_name: str = "distilgpt2",
        target_layer: int = 6,
        data_limit: int = 10,
        pairs_per_problem: int = 1
    ) -> ExperimentResults:
        """
        Run a quick experiment with minimal configuration.
        
        Args:
            model_name: Model to use
            target_layer: Layer to train steering vector for
            data_limit: Limit on training data
            pairs_per_problem: Number of pairs per problem
            
        Returns:
            ExperimentResults
        """
        # Create minimal configuration
        training_config = TrainingConfig(
            model_name=model_name,
            target_layers=[target_layer],
            batch_size=2  # Small batch for quick testing
        )
        
        experiment_config = ExperimentConfig(
            train_version="release_v1",
            eval_version="release_v2",
            version_split_type="new_only",
            data_limit=data_limit,
            pairs_per_problem=pairs_per_problem,
            training_config=training_config,
            experiment_name="quick_test"
        )
        
        # Update configuration
        self.config = experiment_config
        
        # Run experiment
        return self.run_experiment()
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get information about the current experiment configuration."""
        return {
            "experiment_name": self.config.experiment_name,
            "data_versions": f"{self.config.train_version} -> {self.config.eval_version}",
            "split_type": self.config.version_split_type,
            "model_name": self.config.training_config.model_name,
            "target_layers": self.config.training_config.target_layers,
            "steering_method": self.config.training_config.steering_method,
            "pair_strategy": self.config.pair_strategy,
            "contamination_free": self.config.version_split_type == "new_only"
        }
    
    @classmethod
    def create_default_experiment(cls) -> "ExperimentRunner":
        """Create an experiment runner with default configuration."""
        config = ExperimentConfig()
        return cls(config)
    
    @classmethod
    def create_quick_experiment(cls, data_limit: int = 10) -> "ExperimentRunner":
        """Create a quick experiment for testing."""
        training_config = TrainingConfig(
            model_name="distilgpt2",
            target_layers=[5],  # DistilGPT2 has 6 layers (0-5)
            batch_size=2
        )
        
        config = ExperimentConfig(
            data_limit=data_limit,
            pairs_per_problem=1,
            training_config=training_config,
            experiment_name="quick_test"
        )
        
        return cls(config)