"""
VerifiersGRPOTrainer: Minimal bridge between TRL's GRPOTrainer and verifiers Environment.

This provides only the essential functionality needed to use verifiers environments
with TRL's robust GRPOTrainer implementation.
"""

import asyncio
import concurrent.futures
import logging
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from trl import GRPOTrainer as TRLGRPOTrainer
from trl.trainer.grpo_config import GRPOConfig as TRLGRPOConfig
from peft import PeftConfig

from verifiers import Environment
from verifiers.trainers.multi_turn_mixin import MultiTurnMixin


class VerifiersGRPOConfig(TRLGRPOConfig):
    """
    Minimal extension of TRL's GRPOConfig for verifiers compatibility.
    """

    def __init__(self, *args, **kwargs):
        # Extract verifiers-specific parameters that TRL doesn't know about
        self.use_environment_reward = kwargs.pop("use_environment_reward", True)
        self.environment_reward_weight = kwargs.pop("environment_reward_weight", 1.0)

        # Remove any other verifiers-specific parameters that would break TRL
        self.enable_async_generation = kwargs.pop("enable_async_generation", False)
        self.num_batches_ahead = kwargs.pop("num_batches_ahead", 0)

        # Multi-turn conversation parameters
        self.enable_multi_turn = kwargs.pop(
            "enable_multi_turn", None
        )  # Auto-detect if None
        self.max_conversation_turns = kwargs.pop("max_conversation_turns", 10)

        # Initialize TRL config with remaining parameters
        super().__init__(*args, **kwargs)


class EnvironmentRewardAdapter:
    """
    Bridge between TRL's reward function interface and verifiers Environment.

    TRL expects: reward_func(prompts: list[str], completions: list[str]) -> list[float]
    Verifiers provides: rubric.score_rollouts(...) -> RolloutScores
    """

    def __init__(self, env: Environment):
        self.env = env
        self.logger = logging.getLogger(__name__)
        self.__name__ = f"EnvironmentRewardAdapter_{type(env).__name__}"

    def __call__(
        self, prompts: list[str], completions: list[str], **kwargs
    ) -> list[float]:
        """TRL-compatible reward function interface."""
        try:
            if not (hasattr(self.env, "rubric") and self.env.rubric):
                self.logger.warning("No rubric found, using default rewards")
                return [1.0] * len(prompts)

            # Convert to verifiers format
            prompt_messages = [
                [{"role": "user", "content": str(prompt)}] for prompt in prompts
            ]
            completion_messages = [
                [{"role": "assistant", "content": str(completion)}]
                for completion in completions
            ]

            # Default parameters for verifiers
            answers = [""] * len(prompts)
            states = [{}] * len(prompts)
            tasks = ["default"] * len(prompts)
            infos = [{}] * len(prompts)

            # Run scoring in isolated thread to avoid async conflicts
            def scoring_thread():
                return asyncio.run(
                    self.env.rubric.score_rollouts(
                        prompt_messages,
                        completion_messages,
                        answers,
                        states,
                        tasks,
                        infos,
                    )
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                rollout_scores = executor.submit(scoring_thread).result(timeout=60)

            return rollout_scores.reward

        except Exception as e:
            self.logger.error(f"Environment reward computation failed: {e}")
            return [0.0] * len(prompts)


class VerifiersGRPOTrainer(MultiTurnMixin, TRLGRPOTrainer):
    """
    Minimal extension of TRL's GRPOTrainer for verifiers Environment integration.

    This trainer only adds the essential Environment reward bridge - everything else
    leverages TRL's robust implementation.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        env: Environment,
        args: Optional[VerifiersGRPOConfig] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple = (None, None),
        peft_config: Optional[PeftConfig] = None,
        **kwargs,
    ):
        """
        Initialize VerifiersGRPOTrainer with Environment reward integration.
        """
        self.logger = logging.getLogger(__name__)
        self.env = env

        # Set default config
        if args is None:
            args = VerifiersGRPOConfig(output_dir="./outputs")

        # Create environment reward adapter
        reward_adapter = None
        if getattr(args, "use_environment_reward", True):
            reward_adapter = EnvironmentRewardAdapter(env)

        # Store for backwards compatibility
        self._reward_adapter = reward_adapter

        # Get dataset from environment
        train_dataset = env.get_dataset() if hasattr(env, "get_dataset") else None
        eval_dataset = (
            env.get_eval_dataset() if hasattr(env, "get_eval_dataset") else None
        )

        # Initialize TRL trainer
        super().__init__(
            model=model,
            reward_funcs=reward_adapter,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Store verifiers-specific config for multi-turn support
        self.num_batches_ahead = getattr(args, "num_batches_ahead", 0)
        self.enable_async_generation = getattr(args, "enable_async_generation", False)

        # Multi-turn configuration
        enable_multi_turn = getattr(args, "enable_multi_turn", None)
        self.max_conversation_turns = getattr(args, "max_conversation_turns", 10)

        # Auto-detect multi-turn capability if not explicitly set
        if enable_multi_turn is None:
            self.is_multi_turn = self.is_multi_turn_environment(env)
        else:
            self.is_multi_turn = enable_multi_turn

        self.logger.info(
            f"VerifiersGRPOTrainer initialized with Environment reward integration. "
            f"Multi-turn: {'enabled' if self.is_multi_turn else 'disabled'}"
        )
