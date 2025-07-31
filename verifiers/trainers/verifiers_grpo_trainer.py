"""
VerifiersGRPOTrainer: Extension of TRL's GRPOTrainer with verifiers-specific capabilities.

This trainer provides a bridge between verifiers' Environment abstraction and TRL's
GRPOTrainer, enabling the use of TRL as a base while preserving verifiers' unique features
like async batch generation and Environment-based reward computation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from trl import GRPOTrainer as TRLGRPOTrainer
from trl.trainer.grpo_config import GRPOConfig as TRLGRPOConfig
from peft import PeftConfig

from verifiers import Environment
from verifiers.trainers.async_batch_generator import AsyncBatchGenerator, BatchRequest
from verifiers.trainers.async_dataloader_wrapper import AsyncDataLoaderWrapper
from verifiers.trainers.multi_turn_mixin import MultiTurnMixin


class VerifiersGRPOConfig(TRLGRPOConfig):
    """
    Extended GRPO configuration that includes verifiers-specific parameters.
    """

    def __init__(self, *args, **kwargs):
        # Extract verifiers-specific parameters
        self.num_batches_ahead = kwargs.pop("num_batches_ahead", 1)
        self.enable_async_generation = kwargs.pop("enable_async_generation", True)
        self.async_timeout = kwargs.pop("async_timeout", 300)

        # Initialize base config
        super().__init__(*args, **kwargs)


class EnvironmentRewardAdapter:
    """
    Adapter that converts verifiers Environment objects to TRL-compatible reward functions.
    """

    def __init__(self, env: Environment):
        self.env = env
        self.logger = logging.getLogger(__name__)
        self.__name__ = f"EnvironmentRewardAdapter_{type(env).__name__}"

    def __call__(
        self, prompts: List[str], completions: List[str], **kwargs
    ) -> List[float]:
        """
        Convert Environment reward computation to TRL format.

        Args:
            prompts: List of prompt strings
            completions: List of completion strings

        Returns:
            List of reward scores
        """
        try:
            # Use the environment's rubric to compute rewards
            if hasattr(self.env, "rubric") and self.env.rubric:
                import asyncio

                # Convert strings to Messages format expected by rubric
                prompt_messages = []
                completion_messages = []
                answers = []
                states = []
                tasks = []

                for prompt, completion in zip(prompts, completions):
                    # Convert to message format
                    prompt_msg = [{"role": "user", "content": prompt}]
                    completion_msg = [{"role": "assistant", "content": completion}]

                    prompt_messages.append(prompt_msg)
                    completion_messages.append(completion_msg)
                    answers.append("")  # No specific answer for reward computation
                    states.append({})  # Empty state
                    tasks.append("default")

                # Run async scoring function
                try:
                    if asyncio.get_event_loop().is_running():
                        # If we're already in an async context, create a new event loop
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.env.rubric.score_rollouts(
                                    prompt_messages,
                                    completion_messages,
                                    answers,
                                    states,
                                    tasks,
                                ),
                            )
                            rollout_scores = future.result()
                    else:
                        rollout_scores = asyncio.run(
                            self.env.rubric.score_rollouts(
                                prompt_messages,
                                completion_messages,
                                answers,
                                states,
                                tasks,
                            )
                        )

                    # Extract total scores - handle different return formats
                    if isinstance(rollout_scores, (list, tuple)):
                        if len(rollout_scores) > 0 and hasattr(
                            rollout_scores[0], "total"
                        ):
                            rewards = [score.total for score in rollout_scores]
                        else:
                            # Handle tuple format (rewards, metadata)
                            rewards = [
                                float(score) if isinstance(score, (int, float)) else 1.0
                                for score in rollout_scores
                            ]
                    else:
                        rewards = [1.0] * len(prompts)
                    return rewards

                except Exception as e:
                    self.logger.warning(f"Async scoring failed, using fallback: {e}")
                    # Fallback to simple scoring
                    return [1.0] * len(prompts)

            else:
                # Fallback: use a simple reward (this should be customized)
                self.logger.warning(
                    "No rubric found in environment, using fallback reward"
                )
                return [1.0] * len(prompts)

        except Exception as e:
            self.logger.error(f"Error in reward computation: {e}")
            # Return zero rewards on error to avoid training crashes
            return [0.0] * len(prompts)


class VerifiersGRPOTrainer(MultiTurnMixin, TRLGRPOTrainer):
    """
    Extended GRPO trainer that integrates verifiers Environment abstraction
    with TRL's GRPOTrainer base functionality.

    This trainer provides:
    - Environment-based reward computation
    - Async batch generation capabilities
    - Integration with verifiers' existing infrastructure
    - Full compatibility with TRL's training pipeline
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        env: Environment,
        args: Optional[VerifiersGRPOConfig] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        """
        Initialize VerifiersGRPOTrainer.

        Args:
            model: Model to train (string path or PreTrainedModel)
            env: Verifiers Environment object
            args: Training arguments (VerifiersGRPOConfig)
            processing_class: Tokenizer or processor
            callbacks: Training callbacks
            optimizers: Optimizer and scheduler tuple
            peft_config: PEFT configuration
            **kwargs: Additional arguments
        """
        self.logger = logging.getLogger(__name__)
        self.env = env

        # Set default config if none provided
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = VerifiersGRPOConfig(f"{model_name}-Verifiers-GRPO")

        # Store verifiers-specific config
        self.num_batches_ahead = getattr(args, "num_batches_ahead", 1)
        self.enable_async_generation = getattr(args, "enable_async_generation", True)
        self.async_timeout = getattr(args, "async_timeout", 300)

        # Detect multi-turn capability
        self.is_multi_turn = self.is_multi_turn_environment(env)
        self.logger.info(f"Environment detected as {'multi-turn' if self.is_multi_turn else 'single-turn'}")

        # Create reward function adapter from Environment
        reward_adapter = EnvironmentRewardAdapter(env)
        self._reward_adapter = reward_adapter

        # Get dataset from environment
        train_dataset = None
        eval_dataset = None

        if self.is_multi_turn:
            # For multi-turn environments, we'll generate datasets dynamically
            # during training rather than using static datasets
            self.logger.info("Multi-turn environment detected - datasets will be generated dynamically")
            train_dataset = None
            eval_dataset = None
        else:
            # For single-turn environments, use standard dataset loading
            try:
                train_dataset = env.get_dataset()
                if hasattr(env, "get_eval_dataset"):
                    eval_dataset = env.get_eval_dataset()
            except Exception as e:
                self.logger.warning(f"Could not get dataset from environment: {e}")

        # Initialize TRL trainer with adapted components
        super().__init__(
            model=model,
            reward_funcs=reward_adapter,  # Use our adapter as reward function
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Initialize async components if enabled
        if self.enable_async_generation and self.num_batches_ahead > 0:
            self._init_async_components()

        self.logger.info(
            f"Initialized VerifiersGRPOTrainer with async_generation={self.enable_async_generation}, num_batches_ahead={self.num_batches_ahead}"
        )

    def _init_async_components(self):
        """Initialize async batch generation components."""
        try:
            # Initialize async batch generator
            # Note: This needs client_config and other parameters from the original implementation
            # These would need to be passed in or configured based on the environment
            self._async_generator = (
                None  # Will be initialized when needed with proper config
            )
            self._async_dataloader = None  # Will be set in get_train_dataloader
            self._batch_id_counter = 0
            self.logger.info("Async components prepared for initialization")
        except Exception as e:
            self.logger.warning(f"Failed to prepare async components: {e}")
            self.enable_async_generation = False

    def _get_async_generator(self):
        """Lazy initialization of async generator with proper configuration."""
        if self._async_generator is None and self.enable_async_generation:
            try:
                # This would need to be configured based on your environment's client config
                # For now, we'll prepare the structure but actual initialization needs env-specific config
                client_config = getattr(self.env, "client_config", {})
                model_name = getattr(self, "model_name", "default-model")
                sampling_args = getattr(self.env, "sampling_args", {})

                self._async_generator = AsyncBatchGenerator(
                    env=self.env,
                    client_config=client_config,
                    model_name=model_name,
                    sampling_args=sampling_args,
                    num_batches_ahead=self.num_batches_ahead,
                    generation_timeout=self.async_timeout,
                )
                self._async_generator.start()
                self.logger.info("Async batch generator initialized and started")
            except Exception as e:
                self.logger.error(f"Failed to initialize async generator: {e}")
                self.enable_async_generation = False

        return self._async_generator

    def generate_completions(
        self, prompts: List[str], **generation_kwargs
    ) -> List[str]:
        """
        Generate completions using the environment's generation logic.
        
        For multi-turn environments, this uses the environment's rollout
        system to generate full conversations. For single-turn environments,
        it falls back to TRL's standard generation.
        """
        if self.is_multi_turn:
            return self._generate_multi_turn_completions(prompts, **generation_kwargs)
        else:
            return self._generate_single_turn_completions(prompts, **generation_kwargs)
    
    def _generate_single_turn_completions(
        self, prompts: List[str], **generation_kwargs
    ) -> List[str]:
        """Generate completions for single-turn environments."""
        if hasattr(self.env, "generate") and callable(self.env.generate):
            try:
                # Use environment's generation method
                return self.env.generate(prompts, **generation_kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Environment generation failed, falling back to TRL: {e}"
                )

        # Fallback to TRL's generation
        return super().generate_completions(prompts, **generation_kwargs)
    
    def _generate_multi_turn_completions(
        self, prompts: List[str], **generation_kwargs
    ) -> List[str]:
        """Generate completions for multi-turn environments using environment rollouts."""
        try:
            # For multi-turn, we need to generate fresh data each time
            batch_size = len(prompts)
            
            # Generate multi-turn dataset
            multi_turn_dataset = self.create_multi_turn_dataset(
                env=self.env,
                model=self.model,
                tokenizer=self.tokenizer,
                num_samples=batch_size,
                **generation_kwargs
            )
            
            # Store the current batch data for reward computation
            self._current_batch_data = multi_turn_dataset.to_list()
            
            # Extract completions from the generated dataset
            completions = [item['completion'] for item in self._current_batch_data]
            
            return completions
            
        except Exception as e:
            self.logger.error(f"Multi-turn generation failed: {e}")
            # Fallback to single-turn behavior
            return self._generate_single_turn_completions(prompts, **generation_kwargs)

    def compute_rewards(
        self, prompts: List[str], completions: List[str]
    ) -> List[float]:
        """
        Compute rewards using the Environment's rubric system.
        
        For multi-turn environments, this uses the rich conversation context
        preserved during generation. For single-turn environments, it uses
        the standard reward adapter.
        """
        try:
            if self.is_multi_turn:
                return self.compute_multi_turn_rewards(
                    env=self.env,
                    prompts=prompts,
                    completions=completions
                )
            else:
                # Use the reward adapter for single-turn environments
                if hasattr(self, "_reward_adapter"):
                    return self._reward_adapter(prompts, completions)
                else:
                    # Create temporary adapter
                    adapter = EnvironmentRewardAdapter(self.env)
                    return adapter(prompts, completions)

        except Exception as e:
            self.logger.error(f"Environment reward computation failed: {e}")
            # Return fallback rewards
            return [1.0] * len(prompts)

    def get_train_dataloader(self):
        """
        Get training dataloader with support for multi-turn and async generation.
        """
        if self.is_multi_turn:
            # For multi-turn environments, create a dummy dataset that will be
            # dynamically populated during generation
            return self._create_multi_turn_dataloader()
        
        dataloader = super().get_train_dataloader()

        if self.enable_async_generation and self.num_batches_ahead > 0:
            try:
                # Wrap with async dataloader for peek-ahead capabilities
                buffer_size = max(5, self.num_batches_ahead * 2)
                self._async_dataloader = AsyncDataLoaderWrapper(
                    dataloader, buffer_size=buffer_size
                )
                return self._async_dataloader
            except Exception as e:
                self.logger.warning(f"Failed to create async dataloader: {e}")

        return dataloader
    
    def _create_multi_turn_dataloader(self):
        """Create a dataloader for multi-turn environments."""
        from torch.utils.data import DataLoader, Dataset as TorchDataset
        
        class MultiTurnDataset(TorchDataset):
            """Dynamic dataset that generates multi-turn data on-the-fly."""
            
            def __init__(self, env, num_samples=1000):
                self.env = env
                self.num_samples = num_samples
            
            def __len__(self):
                return self.num_samples
            
            def __getitem__(self, idx):
                # Return dummy data - actual generation happens in generate_completions
                return {
                    "prompt": f"multi_turn_prompt_{idx}",
                    "completion": "",
                    "reward": 0.0
                }
        
        dataset = MultiTurnDataset(self.env, num_samples=getattr(self, 'args', MagicMock(num_train_epochs=1)).num_train_epochs * 1000)
        return DataLoader(
            dataset,
            batch_size=getattr(self, 'args', MagicMock(per_device_train_batch_size=8)).per_device_train_batch_size,
            shuffle=True
        )

    def log_stats(self, stats: Dict[str, Any]):
        """
        Enhanced logging that includes verifiers-specific metrics.
        """
        # Add environment-specific stats if available
        if hasattr(self.env, "get_stats"):
            try:
                env_stats = self.env.get_stats()
                stats.update({f"env_{k}": v for k, v in env_stats.items()})
            except Exception as e:
                self.logger.debug(f"Could not get environment stats: {e}")

        # Add async generation stats
        if self.enable_async_generation:
            stats.update(
                {
                    "async_generation_enabled": True,
                    "num_batches_ahead": self.num_batches_ahead,
                }
            )

        super().log_stats(stats)

    def generate_completions_async(
        self, prompts: List[str], **generation_kwargs
    ) -> List[str]:
        """
        Generate completions using async batch generation if enabled.

        This method submits generation requests asynchronously and can be used
        to overlap generation with training when num_batches_ahead > 0.
        """
        if not self.enable_async_generation or self.num_batches_ahead == 0:
            return self.generate_completions(prompts, **generation_kwargs)

        async_gen = self._get_async_generator()
        if async_gen is None:
            # Fallback to synchronous generation
            return self.generate_completions(prompts, **generation_kwargs)

        try:
            # Create batch request
            batch_id = self._batch_id_counter
            self._batch_id_counter += 1

            # Convert prompts to the format expected by AsyncBatchGenerator
            # This is a simplified version - may need adaptation based on actual Environment API
            env_inputs = {"prompts": prompts, **generation_kwargs}

            request = BatchRequest(
                batch_id=batch_id,
                env_inputs=env_inputs,
                processing_class=self.tokenizer,  # or self.processing_class
                mask_env_responses=True,
                max_seq_len=getattr(self.args, "max_seq_len", 2048),
                mask_truncated_completions=True,
                zero_truncated_completions=False,
                max_concurrent=getattr(self.args, "max_concurrent", 10),
            )

            # Submit and wait for result
            if async_gen.submit_batch(request):
                result = async_gen.get_batch(batch_id, timeout=self.async_timeout)

                # Extract completions from result
                # This would need to be adapted based on the actual result format
                if hasattr(result, "completions") and result.completions:
                    return result.completions
                else:
                    self.logger.warning(
                        "No completions in async result, falling back to sync"
                    )
                    return self.generate_completions(prompts, **generation_kwargs)
            else:
                self.logger.warning(
                    "Failed to submit async batch, falling back to sync"
                )
                return self.generate_completions(prompts, **generation_kwargs)

        except Exception as e:
            self.logger.error(f"Async generation failed: {e}, falling back to sync")
            return self.generate_completions(prompts, **generation_kwargs)

    def cleanup_async_components(self):
        """Clean up async components when training is finished."""
        if hasattr(self, "_async_generator") and self._async_generator:
            try:
                self._async_generator.stop()
                self.logger.info("Async batch generator stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping async generator: {e}")

        if hasattr(self, "_async_dataloader") and self._async_dataloader:
            try:
                # AsyncDataLoaderWrapper might not have a cleanup method
                # but we should release any resources if possible
                pass
            except Exception as e:
                self.logger.warning(f"Error cleaning up async dataloader: {e}")

    def __del__(self):
        """Cleanup when trainer is destroyed."""
        self.cleanup_async_components()
