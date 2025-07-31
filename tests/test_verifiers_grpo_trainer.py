"""Tests for the VerifiersGRPOTrainer class."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from verifiers import Rubric, SingleTurnEnv
from verifiers.trainers import (
    EnvironmentRewardAdapter,
    VerifiersGRPOConfig,
    VerifiersGRPOTrainer,
)


class TestEnvironmentRewardAdapter:
    """Test cases for the EnvironmentRewardAdapter class."""

    def test_adapter_initialization(self, mock_singleturn_env):
        """Test EnvironmentRewardAdapter initialization."""
        adapter = EnvironmentRewardAdapter(mock_singleturn_env)

        assert adapter.env == mock_singleturn_env
        assert hasattr(adapter, "__name__")
        assert "EnvironmentRewardAdapter" in adapter.__name__
        assert hasattr(adapter, "logger")

    def test_adapter_callable_with_rubric(self, mock_singleturn_env):
        """Test adapter reward computation with rubric."""

        def mock_reward(prompt, completion, **kwargs):
            return len(completion.split())

        mock_singleturn_env.rubric = Rubric(funcs=[mock_reward])
        adapter = EnvironmentRewardAdapter(mock_singleturn_env)

        prompts = ["What is 2+2?", "Hello world"]
        completions = ["The answer is four", "Hi there"]

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = [MagicMock(total=4.0), MagicMock(total=2.0)]
            rewards = adapter(prompts, completions)

            assert len(rewards) == 2
            assert all(isinstance(r, float) for r in rewards)

    def test_adapter_callable_without_rubric(self, mock_singleturn_env):
        """Test adapter behavior when environment has no rubric."""
        mock_singleturn_env.rubric = None
        adapter = EnvironmentRewardAdapter(mock_singleturn_env)

        prompts = ["Test prompt"]
        completions = ["Test completion"]

        rewards = adapter(prompts, completions)

        assert len(rewards) == 1
        assert rewards[0] == 1.0

    def test_adapter_fallback_behavior(self, mock_singleturn_env):
        """Test adapter fallback when no rubric is available."""
        mock_singleturn_env.rubric = None
        adapter = EnvironmentRewardAdapter(mock_singleturn_env)

        prompts = ["Test prompt"]
        completions = ["Test completion"]

        rewards = adapter(prompts, completions)

        assert len(rewards) == 1
        assert rewards[0] == 1.0


class TestVerifiersGRPOConfig:
    """Test cases for the VerifiersGRPOConfig class."""

    def test_config_initialization_basic(self):
        """Test VerifiersGRPOConfig basic initialization."""
        config = VerifiersGRPOConfig(
            output_dir="./test_output",
            bf16=False,
            fp16=False,
        )

        assert config.output_dir == "./test_output"
        assert hasattr(config, "num_batches_ahead")
        assert hasattr(config, "enable_async_generation")
        assert hasattr(config, "async_timeout")

    def test_config_initialization_with_verifiers_params(self):
        """Test VerifiersGRPOConfig with verifiers-specific parameters."""
        config = VerifiersGRPOConfig(
            output_dir="./test_output",
            num_batches_ahead=3,
            enable_async_generation=False,
            async_timeout=120,
            bf16=False,
            fp16=False,
        )

        assert config.num_batches_ahead == 3
        assert config.enable_async_generation is False
        assert config.async_timeout == 120

    def test_config_trl_inheritance(self):
        """Test that VerifiersGRPOConfig inherits from TRL's GRPOConfig."""
        from trl.trainer.grpo_config import GRPOConfig as TRLGRPOConfig

        config = VerifiersGRPOConfig(
            output_dir="./test",
            bf16=False,
            fp16=False,
        )

        assert isinstance(config, TRLGRPOConfig)


class TestVerifiersGRPOTrainer:
    """Test cases for the VerifiersGRPOTrainer class."""

    @pytest.fixture
    def mock_env(self, sample_dataset):
        """Create a mock environment for testing."""

        def simple_reward(prompt, completion, **kwargs):
            return len(completion.split())

        rubric = Rubric(funcs=[simple_reward])
        return SingleTurnEnv(dataset=sample_dataset, rubric=rubric)

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        return VerifiersGRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=8,
            num_generations=8,
            learning_rate=1e-5,
            max_steps=10,
            enable_async_generation=False,
            num_batches_ahead=0,
            bf16=False,
            fp16=False,
            save_strategy="no",
            logging_steps=1,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        return tokenizer

    def test_trainer_initialization_basic(self, mock_env, mock_config, mock_tokenizer):
        """Test VerifiersGRPOTrainer basic initialization."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            assert trainer.env == mock_env
            assert trainer.enable_async_generation is False
            assert trainer.num_batches_ahead == 0
            assert hasattr(trainer, "logger")

    def test_trainer_initialization_with_async(self, mock_env, mock_tokenizer):
        """Test VerifiersGRPOTrainer initialization with async enabled."""
        config = VerifiersGRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=8,
            num_generations=8,
            enable_async_generation=True,
            num_batches_ahead=2,
            bf16=False,
            fp16=False,
            save_strategy="no",
        )

        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=config,
                processing_class=mock_tokenizer,
            )

            assert trainer.enable_async_generation is True
            assert trainer.num_batches_ahead == 2

    def test_trainer_trl_inheritance(self, mock_env, mock_config, mock_tokenizer):
        """Test that VerifiersGRPOTrainer inherits from TRL's GRPOTrainer."""
        from trl import GRPOTrainer as TRLGRPOTrainer

        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            assert isinstance(trainer, TRLGRPOTrainer)

    def test_trainer_reward_adapter_creation(
        self, mock_env, mock_config, mock_tokenizer
    ):
        """Test that trainer creates reward adapter from environment."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args

            assert "reward_funcs" in kwargs
            assert isinstance(kwargs["reward_funcs"], EnvironmentRewardAdapter)

    def test_trainer_dataset_extraction(self, mock_env, mock_config, mock_tokenizer):
        """Test that trainer extracts dataset from environment."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            mock_init.assert_called_once()
            args, kwargs = mock_init.call_args

            assert "train_dataset" in kwargs
            assert kwargs["train_dataset"] is not None

    def test_trainer_async_generator_lazy_init(
        self, mock_env, mock_config, mock_tokenizer
    ):
        """Test lazy initialization of async generator."""
        config = VerifiersGRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=8,
            num_generations=8,
            enable_async_generation=True,
            num_batches_ahead=2,
            bf16=False,
            fp16=False,
            save_strategy="no",
        )

        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=config,
                processing_class=mock_tokenizer,
            )

            assert trainer._async_generator is None

            with patch(
                "verifiers.trainers.verifiers_grpo_trainer.AsyncBatchGenerator"
            ) as mock_gen:
                mock_instance = MagicMock()
                mock_gen.return_value = mock_instance

                async_gen = trainer._get_async_generator()

                assert async_gen == mock_instance
                mock_instance.start.assert_called_once()

    def test_trainer_generate_completions_sync(
        self, mock_env, mock_config, mock_tokenizer
    ):
        """Test synchronous completion generation."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            with patch.object(
                trainer, "generate_completions", return_value=["test"]
            ) as mock_gen:
                prompts = ["What is 2+2?"]
                completions = trainer.generate_completions_async(prompts)

                assert completions == ["test"]
                mock_gen.assert_called_once_with(prompts)

    def test_trainer_compute_rewards(self, mock_env, mock_config, mock_tokenizer):
        """Test reward computation using environment."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = [MagicMock(total=3.0), MagicMock(total=2.0)]

                prompts = ["What is 2+2?", "Hello"]
                completions = ["The answer is four", "Hi there"]

                rewards = trainer.compute_rewards(prompts, completions)

                assert len(rewards) == 2
                assert all(isinstance(r, float) for r in rewards)

    def test_trainer_dataloader_wrapper(self, mock_env, mock_config, mock_tokenizer):
        """Test training dataloader with async wrapper."""
        config = VerifiersGRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=8,
            num_generations=8,
            enable_async_generation=True,
            num_batches_ahead=2,
            bf16=False,
            fp16=False,
            save_strategy="no",
        )

        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=config,
                processing_class=mock_tokenizer,
            )

            trainer.accelerator = MagicMock()  # Add accelerator attribute
            mock_dataloader = MagicMock()

            with patch(
                "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.get_train_dataloader"
            ) as mock_get_dl:
                mock_get_dl.return_value = mock_dataloader

                with patch(
                    "verifiers.trainers.verifiers_grpo_trainer.AsyncDataLoaderWrapper"
                ) as mock_wrapper:
                    mock_wrapper_instance = MagicMock()
                    mock_wrapper.return_value = mock_wrapper_instance
                    trainer.accelerator.prepare.return_value = mock_wrapper_instance

                    result = trainer.get_train_dataloader()

                    mock_wrapper.assert_called_once()
                    assert result == mock_wrapper_instance

    def test_trainer_cleanup_async_components(
        self, mock_env, mock_config, mock_tokenizer
    ):
        """Test cleanup of async components."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            mock_async_gen = MagicMock()
            trainer._async_generator = mock_async_gen

            trainer.cleanup_async_components()

            mock_async_gen.stop.assert_called_once()

    def test_trainer_has_log_stats_method(self, mock_env, mock_config, mock_tokenizer):
        """Test that trainer has log_stats method for enhanced logging."""
        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=mock_env,
                args=mock_config,
                processing_class=mock_tokenizer,
            )

            assert hasattr(trainer, "log_stats")
            assert callable(trainer.log_stats)


@pytest.mark.integration
class TestVerifiersGRPOTrainerIntegration:
    """Integration tests for VerifiersGRPOTrainer."""

    def test_trainer_full_initialization_with_real_env(self):
        """Test full trainer initialization with real environment components."""
        dataset = Dataset.from_dict({"question": ["What is 2+2?"], "answer": ["4"]})

        def simple_reward(prompt, completion, **kwargs):
            return 1.0

        rubric = Rubric(funcs=[simple_reward])
        env = SingleTurnEnv(dataset=dataset, rubric=rubric)

        config = VerifiersGRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=8,
            num_generations=8,
            max_steps=1,
            enable_async_generation=False,
            bf16=False,
            fp16=False,
            save_strategy="no",
            logging_steps=1,
        )

        tokenizer = MagicMock()
        tokenizer.pad_token = "[PAD]"
        tokenizer.eos_token = "[EOS]"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        with patch(
            "verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__"
        ) as mock_init:
            mock_init.return_value = None

            trainer = VerifiersGRPOTrainer(
                model="test-model",
                env=env,
                args=config,
                processing_class=tokenizer,
            )

            assert trainer.env == env
            assert isinstance(trainer, VerifiersGRPOTrainer)
