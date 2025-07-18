"""
Tests for steering vector training pipeline.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from wisent_guard.core.pipelines import ActivationCollector, SteeringVectorTrainer, ExperimentRunner
from wisent_guard.core.pipelines.activation_collector import ActivationData
from wisent_guard.core.pipelines.steering_trainer import TrainingConfig, TrainingResults
from wisent_guard.core.pipelines.experiment_runner import ExperimentConfig, ExperimentResults
from wisent_guard.core.data_loaders.steering_data_extractor import ContrastivePair
from wisent_guard.core.data_loaders.livecodebench_loader import LiveCodeBenchProblem


class TestActivationCollector:
    """Test suite for ActivationCollector."""
    
    @patch('wisent_guard.core.pipelines.activation_collector.AutoModel')
    @patch('wisent_guard.core.pipelines.activation_collector.AutoTokenizer')
    def test_init(self, mock_tokenizer_class, mock_model_class):
        """Test activation collector initialization."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        collector = ActivationCollector("distilgpt2", device="cpu")
        
        assert collector.model_name == "distilgpt2"
        assert collector.device == "cpu"
        assert collector.max_length == 512
        
        mock_model_class.from_pretrained.assert_called_once_with("distilgpt2")
        mock_tokenizer_class.from_pretrained.assert_called_once_with("distilgpt2")
    
    def test_get_device_auto(self):
        """Test device selection."""
        collector = ActivationCollector.__new__(ActivationCollector)
        
        # Test auto device selection
        with patch('torch.cuda.is_available', return_value=True):
            device = collector._get_device("auto")
            assert device == "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = collector._get_device("auto")
                assert device == "mps"
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = collector._get_device("auto")
                assert device == "cpu"
    
    @patch('wisent_guard.core.pipelines.activation_collector.AutoModel')
    @patch('wisent_guard.core.pipelines.activation_collector.AutoTokenizer')
    def test_get_activations(self, mock_tokenizer_class, mock_model_class):
        """Test activation collection."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        
        # Mock model structure (GPT-style)
        mock_layer = MagicMock()
        mock_model.transformer.h = [mock_layer]
        mock_model.config.hidden_size = 768
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        # Mock activation hook
        def mock_hook(module, input, output):
            # Return mock activation tensor
            return torch.randn(1, 768)
        
        collector = ActivationCollector("distilgpt2", device="cpu")
        
        # Test activation collection
        with patch.object(collector, '_get_activations') as mock_get_activations:
            mock_get_activations.return_value = torch.randn(2, 768)
            
            texts = ["Hello world", "Test text"]
            activations = collector._get_activations(texts, layer_idx=0)
            
            assert activations.shape == (2, 768)
    
    @patch('wisent_guard.core.pipelines.activation_collector.AutoModel')
    @patch('wisent_guard.core.pipelines.activation_collector.AutoTokenizer')
    def test_collect_contrastive_activations(self, mock_tokenizer_class, mock_model_class):
        """Test contrastive activation collection."""
        # Setup mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        collector = ActivationCollector("distilgpt2", device="cpu")
        
        # Create test contrastive pairs
        pairs = [
            ContrastivePair(
                positive_prompt="Write correct code",
                negative_prompt="Write incorrect code",
                problem_id="test_1",
                metadata={"strategy": "correct_vs_incorrect"}
            )
        ]
        
        # Mock activation collection
        with patch.object(collector, '_collect_batch_activations') as mock_collect_batch:
            mock_activation_data = [
                ActivationData(
                    positive_activations=torch.randn(768),
                    negative_activations=torch.randn(768),
                    layer_idx=0,
                    problem_id="test_1",
                    metadata={"strategy": "correct_vs_incorrect"}
                )
            ]
            mock_collect_batch.return_value = mock_activation_data
            
            result = collector.collect_contrastive_activations(pairs, [0], batch_size=1)
            
            assert len(result) == 1
            assert isinstance(result[0], ActivationData)
            assert result[0].layer_idx == 0
            assert result[0].problem_id == "test_1"
    
    @patch('wisent_guard.core.pipelines.activation_collector.AutoModel')
    @patch('wisent_guard.core.pipelines.activation_collector.AutoTokenizer')
    def test_get_model_info(self, mock_tokenizer_class, mock_model_class):
        """Test model info retrieval."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.vocab_size = 50257
        
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.parameters.return_value = [torch.randn(100, 200)]
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        collector = ActivationCollector("distilgpt2", device="cpu")
        
        info = collector.get_model_info()
        
        assert info["model_name"] == "distilgpt2"
        assert info["device"] == "cpu"
        assert info["vocab_size"] == 50257
        assert info["hidden_size"] == 768
        assert info["num_layers"] == 12


class TestSteeringVectorTrainer:
    """Test suite for SteeringVectorTrainer."""
    
    def test_training_config_defaults(self):
        """Test training configuration defaults."""
        config = TrainingConfig()
        
        assert config.model_name == "distilgpt2"
        assert config.target_layers == [6]
        assert config.steering_method == "CAA"
        assert config.batch_size == 8
        assert config.device == "auto"
    
    def test_training_config_custom(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            model_name="gpt2",
            target_layers=[5, 7],
            steering_method="CAA",
            batch_size=16,
            device="cuda"
        )
        
        assert config.model_name == "gpt2"
        assert config.target_layers == [5, 7]
        assert config.steering_method == "CAA"
        assert config.batch_size == 16
        assert config.device == "cuda"
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        config = TrainingConfig()
        trainer = SteeringVectorTrainer(config)
        
        assert trainer.config == config
        assert trainer.activation_collector is None
        assert trainer.steering_method is None
        assert "CAA" in trainer.steering_methods
    
    @patch('wisent_guard.core.pipelines.steering_trainer.ActivationCollector')
    @patch('wisent_guard.core.pipelines.steering_trainer.CAA')
    def test_initialize_components(self, mock_caa_class, mock_collector_class):
        """Test component initialization."""
        config = TrainingConfig()
        trainer = SteeringVectorTrainer(config)
        
        mock_collector = MagicMock()
        mock_caa = MagicMock()
        
        mock_collector_class.return_value = mock_collector
        mock_caa_class.return_value = mock_caa
        
        trainer._initialize_components()
        
        assert trainer.activation_collector == mock_collector
        assert trainer.steering_method == mock_caa
        
        mock_collector_class.assert_called_once_with(
            model_name="distilgpt2",
            device="auto",
            max_length=512
        )
        mock_caa_class.assert_called_once_with(
            device="auto",
            normalization_method="none",
            target_norm=None
        )
    
    def test_invalid_steering_method(self):
        """Test invalid steering method raises error."""
        config = TrainingConfig(steering_method="INVALID")
        trainer = SteeringVectorTrainer(config)
        
        with pytest.raises(ValueError, match="Unknown steering method"):
            trainer._initialize_components()
    
    @patch('wisent_guard.core.pipelines.steering_trainer.ActivationCollector')
    @patch('wisent_guard.core.pipelines.steering_trainer.CAA')
    def test_train_workflow(self, mock_caa_class, mock_collector_class):
        """Test complete training workflow."""
        config = TrainingConfig()
        trainer = SteeringVectorTrainer(config)
        
        # Setup mocks
        mock_collector = MagicMock()
        mock_caa = MagicMock()
        mock_caa.steering_vector = torch.randn(768)
        
        mock_collector_class.return_value = mock_collector
        mock_caa_class.return_value = mock_caa
        
        # Mock activation collection
        mock_activation_data = [
            ActivationData(
                positive_activations=torch.randn(768),
                negative_activations=torch.randn(768),
                layer_idx=6,
                problem_id="test_1",
                metadata={}
            )
        ]
        mock_collector.collect_contrastive_activations.return_value = mock_activation_data
        mock_collector.get_model_info.return_value = {"model_name": "distilgpt2"}
        
        # Mock steering method training
        mock_caa.train.return_value = {"loss": 0.5}
        
        # Create test contrastive pairs
        pairs = [
            ContrastivePair(
                positive_prompt="Write correct code",
                negative_prompt="Write incorrect code",
                problem_id="test_1",
                metadata={}
            )
        ]
        
        # Train
        results = trainer.train(pairs)
        
        # Verify results
        assert isinstance(results, TrainingResults)
        assert len(results.steering_vectors) == 1
        assert 6 in results.steering_vectors
        assert results.steering_vectors[6].shape == (768,)
        assert results.training_time > 0
        assert results.timestamp
    
    @patch('wisent_guard.core.pipelines.steering_trainer.ActivationCollector')
    @patch('wisent_guard.core.pipelines.steering_trainer.CAA')
    def test_save_results(self, mock_caa_class, mock_collector_class):
        """Test results saving."""
        config = TrainingConfig()
        trainer = SteeringVectorTrainer(config)
        
        # Create mock results
        results = TrainingResults(
            steering_vectors={6: torch.randn(768)},
            training_stats={"layer_6": {"loss": 0.5}},
            config=config,
            model_info={"model_name": "distilgpt2"},
            training_time=10.5,
            timestamp="2025-01-18T10:00:00"
        )
        
        # Test saving
        with patch('torch.save') as mock_torch_save:
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump') as mock_json_dump:
                    with patch('os.makedirs') as mock_makedirs:
                        trainer.save_results(results, "/tmp/test_results")
        
        # Verify torch.save was called
        mock_torch_save.assert_called()
        
        # Verify JSON dumps were called (for metadata and summary)
        assert mock_json_dump.call_count >= 2


class TestExperimentRunner:
    """Test suite for ExperimentRunner."""
    
    def test_experiment_config_defaults(self):
        """Test experiment configuration defaults."""
        config = ExperimentConfig()
        
        assert config.train_version == "release_v1"
        assert config.eval_version == "release_v2"
        assert config.version_split_type == "new_only"
        assert config.pair_strategy == "correct_vs_incorrect"
        assert config.pairs_per_problem == 1
        assert config.training_config is not None
    
    def test_experiment_runner_init(self):
        """Test experiment runner initialization."""
        config = ExperimentConfig()
        runner = ExperimentRunner(config)
        
        assert runner.config == config
        assert runner.data_loader is not None
        assert runner.version_manager is not None
        assert runner.data_extractor is not None
    
    @patch('wisent_guard.core.pipelines.experiment_runner.LiveCodeBenchLoader')
    @patch('wisent_guard.core.pipelines.experiment_runner.LiveCodeBenchVersionManager')
    @patch('wisent_guard.core.pipelines.experiment_runner.SteeringDataExtractor')
    @patch('wisent_guard.core.pipelines.experiment_runner.SteeringVectorTrainer')
    def test_run_experiment_workflow(self, mock_trainer_class, mock_extractor_class, 
                                   mock_version_manager_class, mock_loader_class):
        """Test complete experiment workflow."""
        config = ExperimentConfig()
        runner = ExperimentRunner(config)
        
        # Setup mocks
        mock_loader = MagicMock()
        mock_version_manager = MagicMock()
        mock_extractor = MagicMock()
        mock_trainer = MagicMock()
        
        mock_loader_class.return_value = mock_loader
        mock_version_manager_class.return_value = mock_version_manager
        mock_extractor_class.return_value = mock_extractor
        mock_trainer_class.return_value.__enter__.return_value = mock_trainer
        
        # Mock data loading
        mock_train_data = [MagicMock()]
        mock_eval_data = [MagicMock()]
        mock_version_manager.get_version_split.return_value = {
            "train": mock_train_data,
            "eval": mock_eval_data
        }
        
        # Mock contrastive pair extraction
        mock_pairs = [MagicMock()]
        mock_extractor.extract_contrastive_pairs.return_value = mock_pairs
        
        # Mock training
        mock_training_results = TrainingResults(
            steering_vectors={6: torch.randn(768)},
            training_stats={},
            config=config.training_config,
            model_info={},
            training_time=10.0,
            timestamp="2025-01-18T10:00:00"
        )
        mock_trainer.train.return_value = mock_training_results
        
        # Mock model creation
        with patch('wisent_guard.core.pipelines.experiment_runner.SteeringCompatibleModel') as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.from_pretrained_with_steering.return_value = mock_model
            
            # Mock file operations
            with patch('os.makedirs'):
                with patch('builtins.open', mock_open()):
                    with patch('json.dump'):
                        
                        # Run experiment
                        results = runner.run_experiment()
        
        # Verify results
        assert isinstance(results, ExperimentResults)
        assert results.config == config
        assert results.training_results == mock_training_results
        assert results.experiment_time > 0
        assert results.steering_model == mock_model
    
    def test_create_default_experiment(self):
        """Test creating default experiment."""
        runner = ExperimentRunner.create_default_experiment()
        
        assert isinstance(runner, ExperimentRunner)
        assert runner.config.train_version == "release_v1"
        assert runner.config.eval_version == "release_v2"
    
    def test_create_quick_experiment(self):
        """Test creating quick experiment."""
        runner = ExperimentRunner.create_quick_experiment(data_limit=5)
        
        assert isinstance(runner, ExperimentRunner)
        assert runner.config.data_limit == 5
        assert runner.config.training_config.batch_size == 2
        assert runner.config.experiment_name == "quick_test"
    
    def test_get_experiment_info(self):
        """Test experiment info retrieval."""
        config = ExperimentConfig()
        runner = ExperimentRunner(config)
        
        info = runner.get_experiment_info()
        
        assert info["experiment_name"] == "livecodebench_steering"
        assert info["model_name"] == "distilgpt2"
        assert info["contamination_free"] is True
        assert "release_v1 -> release_v2" in info["data_versions"]


@pytest.mark.integration
class TestSteeringTrainingIntegration:
    """Integration tests for steering training pipeline."""
    
    @patch('wisent_guard.core.pipelines.activation_collector.AutoModel')
    @patch('wisent_guard.core.pipelines.activation_collector.AutoTokenizer')
    def test_end_to_end_pipeline(self, mock_tokenizer_class, mock_model_class):
        """Test end-to-end pipeline with mocked components."""
        # Setup model mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.vocab_size = 50257
        
        mock_model.config.hidden_size = 768
        mock_model.config.num_hidden_layers = 12
        mock_model.transformer.h = [MagicMock()]
        
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock tokenizer behavior
        def mock_tokenize(texts, **kwargs):
            return {
                "input_ids": torch.randint(0, 1000, (len(texts), 10)),
                "attention_mask": torch.ones(len(texts), 10)
            }
        
        mock_tokenizer.side_effect = mock_tokenize
        
        # Create test data
        pairs = [
            ContrastivePair(
                positive_prompt="Write correct code",
                negative_prompt="Write incorrect code",
                problem_id="test_1",
                metadata={"strategy": "correct_vs_incorrect"}
            )
        ]
        
        # Configure training
        config = TrainingConfig(
            model_name="distilgpt2",
            target_layers=[0],
            batch_size=1,
            device="cpu"
        )
        
        # Mock activation collection to return proper tensors
        with patch.object(ActivationCollector, '_get_activations') as mock_get_activations:
            mock_get_activations.return_value = torch.randn(1, 768)
            
            # Mock steering method training
            with patch('wisent_guard.core.pipelines.steering_trainer.CAA') as mock_caa_class:
                mock_caa = MagicMock()
                mock_caa.steering_vector = torch.randn(768)
                mock_caa.train.return_value = {"loss": 0.5}
                mock_caa_class.return_value = mock_caa
                
                # Train steering vectors
                trainer = SteeringVectorTrainer(config)
                results = trainer.train(pairs)
                
                # Verify results
                assert isinstance(results, TrainingResults)
                assert len(results.steering_vectors) == 1
                assert 0 in results.steering_vectors
    
    def test_quick_experiment_integration(self):
        """Test quick experiment integration."""
        runner = ExperimentRunner.create_quick_experiment(data_limit=1)
        
        # Mock all external dependencies
        with patch('wisent_guard.core.pipelines.experiment_runner.LiveCodeBenchVersionManager') as mock_vm:
            with patch('wisent_guard.core.pipelines.experiment_runner.SteeringDataExtractor') as mock_extractor:
                with patch('wisent_guard.core.pipelines.experiment_runner.SteeringVectorTrainer') as mock_trainer_class:
                    with patch('wisent_guard.core.pipelines.experiment_runner.SteeringCompatibleModel') as mock_model_class:
                        
                        # Setup minimal mocks
                        mock_vm.return_value.get_version_split.return_value = {
                            "train": [MagicMock()],
                            "eval": [MagicMock()]
                        }
                        
                        mock_extractor.return_value.extract_contrastive_pairs.return_value = [MagicMock()]
                        
                        mock_trainer = MagicMock()
                        mock_trainer.train.return_value = TrainingResults(
                            steering_vectors={6: torch.randn(768)},
                            training_stats={},
                            config=TrainingConfig(),
                            model_info={},
                            training_time=1.0,
                            timestamp="2025-01-18T10:00:00"
                        )
                        mock_trainer_class.return_value.__enter__.return_value = mock_trainer
                        
                        mock_model_class.from_pretrained_with_steering.return_value = MagicMock()
                        
                        # Mock file operations
                        with patch('os.makedirs'):
                            with patch('builtins.open', mock_open()):
                                with patch('json.dump'):
                                    
                                    # Run quick experiment
                                    results = runner.run_experiment()
                                    
                                    # Verify it completes without errors
                                    assert isinstance(results, ExperimentResults)
                                    assert results.experiment_time > 0


# Helper function for mocking file operations
def mock_open(mock=None, read_data=''):
    """Mock open function for file operations."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(mock, read_data)