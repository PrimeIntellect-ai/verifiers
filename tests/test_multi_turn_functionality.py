"""
Tests for multi-turn functionality in VerifiersGRPOTrainer.

This test suite validates that the multi-turn extension works correctly with
different types of environments and maintains backwards compatibility.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datasets import Dataset

from verifiers.trainers import VerifiersGRPOTrainer, VerifiersGRPOConfig, MultiTurnMixin
from verifiers.trainers.multi_turn_mixin import MultiTurnMixin
from verifiers.envs.environment import Environment
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.envs.tool_env import ToolEnv


def create_test_config(run_name="test-run"):
    """Create a test configuration that works without GPU/bf16."""
    return VerifiersGRPOConfig(
        output_dir=f"test_outputs/{run_name}",
        run_name=run_name,
        bf16=False,  # Disable bf16 for testing
        fp16=False,  # Disable fp16 for testing
        num_train_epochs=1,
        per_device_train_batch_size=8,  # Must be divisible by num_generations
        num_generations=8,  # TRL default
        max_steps=1,
    )


class MockSingleTurnEnv(Environment):
    """Mock single-turn environment for testing."""
    
    def __init__(self):
        super().__init__(dataset=Dataset.from_dict({
            "prompt": [["user", "test prompt"], ["user", "test prompt2"], ["user", "test prompt3"]],
            "answer": ["test answer", "test answer2", "test answer3"]
        }))
    
    async def rollout(self, client, model, prompt, **kwargs):
        return ["assistant response"], {"completed": True}


class MockMultiTurnEnv(MultiTurnEnv):
    """Mock multi-turn environment for testing."""
    
    def __init__(self):
        super().__init__(dataset=Dataset.from_dict({
            "prompt": [["user", "test prompt"], ["user", "test prompt2"], ["user", "test prompt3"]],
            "answer": ["test answer", "test answer2", "test answer3"]
        }))
    
    def is_completed(self, messages, state, **kwargs):
        return len(messages) >= 3  # Simple completion condition
    
    def env_response(self, messages, state, **kwargs):
        return [{"role": "user", "content": "env response"}], state
    
    async def rollout(self, client, model, prompt, **kwargs):
        # Simulate multi-turn conversation
        conversation = [
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Environment feedback"},
            {"role": "assistant", "content": "Final response"}
        ]
        return conversation, {"turn": 2, "completed": True}


class MockToolEnv(ToolEnv):
    """Mock tool environment for testing."""
    
    def __init__(self):
        def mock_tool(arg: str):
            return f"Tool result for {arg}"
        
        super().__init__(
            tools=[mock_tool],
            dataset=Dataset.from_dict({
                "prompt": [["user", "Use the tool"], ["user", "Use the tool again"], ["user", "Use the tool third"]],
                "answer": ["correct answer", "correct answer2", "correct answer3"]
            })
        )


class TestMultiTurnMixin:
    """Test the MultiTurnMixin functionality."""
    
    def test_single_turn_detection(self):
        """Test that single-turn environments are correctly detected."""
        mixin = MultiTurnMixin()
        env = MockSingleTurnEnv()
        
        assert not mixin.is_multi_turn_environment(env)
    
    def test_multi_turn_detection_inheritance(self):
        """Test that MultiTurnEnv subclasses are correctly detected."""
        mixin = MultiTurnMixin()
        env = MockMultiTurnEnv()
        
        assert mixin.is_multi_turn_environment(env)
    
    def test_multi_turn_detection_methods(self):
        """Test that environments with multi-turn methods are detected."""
        mixin = MultiTurnMixin()
        env = MockSingleTurnEnv()
        
        # Add multi-turn methods
        env.is_completed = lambda messages, state, **kwargs: True
        env.env_response = lambda messages, state, **kwargs: ([], state)
        
        assert mixin.is_multi_turn_environment(env)
    
    def test_tool_env_detection(self):
        """Test that tool environments are detected as multi-turn."""
        mixin = MultiTurnMixin()
        env = MockToolEnv()
        
        assert mixin.is_multi_turn_environment(env)
    
    def test_oai_tools_detection(self):
        """Test that environments with OpenAI tools are multi-turn."""
        mixin = MultiTurnMixin()
        env = MockSingleTurnEnv()
        env.oai_tools = [{"type": "function", "function": {"name": "test"}}]
        
        assert mixin.is_multi_turn_environment(env)
    
    @patch('verifiers.trainers.multi_turn_mixin.asyncio')
    def test_compute_multi_turn_rewards_fallback(self, mock_asyncio):
        """Test reward computation fallback when async fails."""
        mixin = MultiTurnMixin()
        env = MockMultiTurnEnv()
        
        # Mock asyncio.run to raise an exception
        mock_asyncio.run.side_effect = Exception("Async failed")
        
        rewards = mixin._compute_rewards_sync(env, ["prompt"], ["completion"])
        assert rewards == [1.0]  # Fallback reward
    
    def test_create_multi_turn_dataset_structure(self):
        """Test that multi-turn dataset creation returns proper structure."""
        mixin = MultiTurnMixin()
        env = MockMultiTurnEnv()
        
        # Mock the environment methods
        env.generate = MagicMock(return_value=MagicMock(
            prompt=[["user", "test"]],
            completion=[["assistant", "response"]],
            reward=[0.8],
            state=[{"turn": 1}]
        ))
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = lambda x, **kwargs: "formatted_string"
        
        dataset = mixin.create_multi_turn_dataset(
            env=env,
            model=mock_model,
            tokenizer=mock_tokenizer,
            num_samples=1
        )
        
        assert len(dataset) == 1
        assert "prompt" in dataset.column_names
        assert "completion" in dataset.column_names
        assert "reward" in dataset.column_names
        assert "original_prompt" in dataset.column_names
        assert "original_completion" in dataset.column_names


class TestVerifiersGRPOTrainerMultiTurn:
    """Test VerifiersGRPOTrainer with multi-turn capabilities."""
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_single_turn_trainer_initialization(self, mock_trl_init):
        """Test trainer initialization with single-turn environment."""
        mock_trl_init.return_value = None
        env = MockSingleTurnEnv()
        
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        assert not trainer.is_multi_turn
        mock_trl_init.assert_called_once()
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_multi_turn_trainer_initialization(self, mock_trl_init):
        """Test trainer initialization with multi-turn environment."""
        mock_trl_init.return_value = None
        env = MockMultiTurnEnv()
        
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        assert trainer.is_multi_turn
        mock_trl_init.assert_called_once()
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_tool_env_trainer_initialization(self, mock_trl_init):
        """Test trainer initialization with tool environment."""
        mock_trl_init.return_value = None
        env = MockToolEnv()
        
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        assert trainer.is_multi_turn
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_single_turn_generation_routing(self, mock_trl_init):
        """Test that single-turn environments use environment generation."""
        mock_trl_init.return_value = None
        
        env = MockSingleTurnEnv()
        # Mock the environment's generate method
        env.generate = MagicMock(return_value=MagicMock(completion=["completion1", "completion2"]))
        
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        prompts = ["prompt1", "prompt2"]
        completions = trainer.generate_completions(prompts)
        
        assert completions == ["completion1", "completion2"]
        env.generate.assert_called_once()
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_multi_turn_generation_routing(self, mock_trl_init):
        """Test that multi-turn environments use environment generation."""
        mock_trl_init.return_value = None
        
        env = MockMultiTurnEnv()
        # Mock the environment's generate method for multi-turn
        env.generate = MagicMock(return_value=MagicMock(completion=["multi_turn_completion1", "multi_turn_completion2"]))
        
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        prompts = ["prompt1", "prompt2"]
        completions = trainer.generate_completions(prompts)
        
        assert completions == ["multi_turn_completion1", "multi_turn_completion2"]
        env.generate.assert_called_once()
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_single_turn_reward_computation(self, mock_trl_init):
        """Test reward computation for single-turn environments."""
        mock_trl_init.return_value = None
        
        env = MockSingleTurnEnv()
        env.rubric = MagicMock()
        
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        # Mock the reward adapter
        trainer._reward_adapter = MagicMock(return_value=[0.8, 0.9])
        
        rewards = trainer.compute_rewards(["prompt1", "prompt2"], ["comp1", "comp2"])
        assert rewards == [0.8, 0.9]
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_multi_turn_reward_computation(self, mock_trl_init):
        """Test reward computation for multi-turn environments uses environment reward adapter."""
        mock_trl_init.return_value = None
        
        env = MockMultiTurnEnv()
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        # Mock the reward adapter to return specific values
        mock_adapter = MagicMock(return_value=[0.7, 0.6])
        trainer._reward_adapter = mock_adapter
        
        rewards = trainer.compute_rewards(["prompt1", "prompt2"], ["comp1", "comp2"])
        assert rewards == [0.7, 0.6]
        mock_adapter.assert_called_once_with(["prompt1", "prompt2"], ["comp1", "comp2"])
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.get_train_dataloader')
    def test_single_turn_dataloader(self, mock_get_dataloader, mock_trl_init):
        """Test dataloader creation for single-turn environments."""
        mock_trl_init.return_value = None
        mock_dataloader = MagicMock()
        mock_get_dataloader.return_value = mock_dataloader
        
        env = MockSingleTurnEnv()
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        dataloader = trainer.get_train_dataloader()
        assert dataloader == mock_dataloader
        mock_get_dataloader.assert_called_once()
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.get_train_dataloader')
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_multi_turn_dataloader(self, mock_trl_init, mock_dataloader):
        """Test dataloader creation for multi-turn environments uses TRL's standard approach."""
        mock_trl_init.return_value = None
        mock_dataloader.return_value = MagicMock()
        
        env = MockMultiTurnEnv()
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        # Mock train_dataset to avoid the AttributeError
        trainer.train_dataset = MagicMock()
        
        dataloader = trainer.get_train_dataloader()
        
        # Check that TRL's dataloader was called (unified approach)
        mock_dataloader.assert_called_once()
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_fallback_on_multi_turn_generation_failure(self, mock_trl_init):
        """Test fallback to single-turn when multi-turn generation fails."""
        mock_trl_init.return_value = None
        
        env = MockMultiTurnEnv()
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=create_test_config("test-run")
        )
        
        # Make environment generate method fail
        env.generate = MagicMock(side_effect=Exception("Generation failed"))
        
        prompts = ["prompt1", "prompt2"]
        completions = trainer.generate_completions(prompts)
        
        # Should fall back to simple completions
        assert len(completions) == 2
        assert all("Generated completion for:" in comp for comp in completions)
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_config_preservation(self, mock_trl_init):
        """Test that verifiers-specific config is preserved."""
        mock_trl_init.return_value = None
        
        config = create_test_config("test-run")
        config.num_batches_ahead = 3
        config.enable_async_generation = False
        
        env = MockMultiTurnEnv()
        trainer = VerifiersGRPOTrainer(
            model="test-model",
            env=env,
            args=config
        )
        
        assert trainer.num_batches_ahead == 3
        assert trainer.enable_async_generation == False
        # async_timeout is not stored in our minimal implementation


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing verifiers code."""
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_existing_single_turn_code_works(self, mock_trl_init):
        """Test that existing single-turn code continues to work."""
        mock_trl_init.return_value = None
        
        # This simulates existing verifiers code that expects single-turn behavior
        env = MockSingleTurnEnv()
        trainer = VerifiersGRPOTrainer(model="test-model", env=env)
        
        # Should initialize without error and detect as single-turn
        assert not trainer.is_multi_turn
        assert hasattr(trainer, '_reward_adapter')
    
    @patch('verifiers.trainers.verifiers_grpo_trainer.TRLGRPOTrainer.__init__')
    def test_existing_environment_interfaces_preserved(self, mock_trl_init):
        """Test that existing environment interfaces are preserved."""
        mock_trl_init.return_value = None
        
        env = MockSingleTurnEnv()
        trainer = VerifiersGRPOTrainer(model="test-model", env=env)
        
        # All the existing interfaces should still work
        assert hasattr(trainer, 'env')
        assert hasattr(trainer, 'compute_rewards')
        assert hasattr(trainer, 'generate_completions')
        assert hasattr(trainer, 'get_train_dataloader')
    
    def test_multi_turn_environments_import_correctly(self):
        """Test that multi-turn environments can be imported and used."""
        from verifiers.trainers import MultiTurnMixin, VerifiersGRPOTrainer
        
        # Should be able to import without issues
        assert MultiTurnMixin is not None
        assert VerifiersGRPOTrainer is not None
    
    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        from verifiers.trainers import (
            VerifiersGRPOTrainer,
            VerifiersGRPOConfig,
            EnvironmentRewardAdapter,
            MultiTurnMixin
        )
        
        # All classes should be importable
        assert VerifiersGRPOTrainer is not None
        assert VerifiersGRPOConfig is not None
        assert EnvironmentRewardAdapter is not None
        assert MultiTurnMixin is not None