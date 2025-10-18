"""
VerifiersOnlineDPOTrainer - Adapter to use TRL's OnlineDPOTrainer with verifiers Environments

This module provides a lightweight adapter (similar to VerifiersGRPOTrainer and VerifiersRLOOTrainer)
that allows using TRL's OnlineDPOTrainer with verifiers Environment reward system instead of
maintaining a separate copy of the trainer code.

Key features:
- 90% code reduction by wrapping TRL's OnlineDPOTrainer instead of copying it
- Automatic TRL updates and improvements
- Seamless integration with verifiers Environments
- Multi-turn support via MultiTurnMixin
- Uses shared EnvironmentRewardAdapter (no code duplication)
"""

from typing import Optional, Union

from datasets import Dataset, IterableDataset
from trl import OnlineDPOTrainer as TRLOnlineDPOTrainer
from trl import OnlineDPOConfig as TRLOnlineDPOConfig

from verifiers.envs import Environment

from .environment_reward_adapter import EnvironmentRewardAdapter
from .multi_turn_mixin import MultiTurnMixin


class VerifiersOnlineDPOConfig(TRLOnlineDPOConfig):
    """Configuration for VerifiersOnlineDPOTrainer, extending TRL's OnlineDPOConfig"""

    def __init__(self, *args, **kwargs):
        # Extract verifiers-specific parameters before passing to TRL
        self.use_environment_reward = kwargs.pop("use_environment_reward", True)
        self.environment_reward_weight = kwargs.pop("environment_reward_weight", 1.0)

        # Remove any legacy parameters that might cause issues with TRL
        kwargs.pop("enable_async_generation", None)
        kwargs.pop("num_batches_ahead", None)

        # Multi-turn conversation parameters
        self.enable_multi_turn = kwargs.pop("enable_multi_turn", None)

        # OnlineDPO can use either judge or reward_funcs
        # We'll use reward_funcs (EnvironmentRewardAdapter)
        # If user passes judge, we'll let TRL handle it

        # Call parent constructor with cleaned kwargs
        super().__init__(*args, **kwargs)


class VerifiersOnlineDPOTrainer(MultiTurnMixin, TRLOnlineDPOTrainer):
    """
    Online DPO Trainer with verifiers Environment integration.

    This is a lightweight adapter that:
    1. Wraps TRL's OnlineDPOTrainer (instead of copying 1500+ lines)
    2. Integrates verifiers Environment reward system
    3. Auto-detects single vs multi-turn environments
    4. Automatically gets TRL updates and improvements

    Online DPO generates preference pairs on-the-fly during training, making it more
    sample-efficient than offline DPO. It can use either:
    - A judge model to score completions
    - Reward functions (we use EnvironmentRewardAdapter)

    Usage:
        ```python
        import verifiers as vf
        from verifiers.trainers import VerifiersOnlineDPOTrainer, VerifiersOnlineDPOConfig

        # Load environment and model
        env = vf.load_environment("reverse-text")
        model, tokenizer = vf.get_model_and_tokenizer("willcb/Qwen3-0.6B")

        # Configure trainer
        args = VerifiersOnlineDPOConfig(
            output_dir="./output",
            use_environment_reward=True,
            use_vllm=True,
            vllm_mode="colocate",
        )

        # Train
        trainer = VerifiersOnlineDPOTrainer(
            model=model,
            env=env,
            args=args,
            processing_class=tokenizer,
        )
        trainer.train()
        ```
    """

    def __init__(
        self,
        model=None,
        ref_model=None,
        env: Optional[Environment] = None,
        args: Optional[VerifiersOnlineDPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class=None,
        reward_processing_classes=None,
        judge=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        **kwargs,
    ):
        # Create environment reward adapter if environment rewards are enabled
        # OnlineDPO can use either judge or reward_funcs
        # If judge is provided, we'll use that instead
        if (
            judge is None
            and env is not None
            and getattr(args, "use_environment_reward", True)
        ):
            reward_adapter = EnvironmentRewardAdapter(env)
        else:
            reward_adapter = None

        # Get dataset from environment if not provided
        if train_dataset is None and env is not None and hasattr(env, "get_dataset"):
            train_dataset = env.get_dataset()
        if (
            eval_dataset is None
            and env is not None
            and hasattr(env, "get_eval_dataset")
        ):
            eval_dataset = env.get_eval_dataset()

        # Store environment for later use
        self.env = env

        # Initialize TRL's OnlineDPOTrainer with our reward adapter
        super().__init__(
            model=model,
            ref_model=ref_model,
            reward_funcs=reward_adapter,
            judge=judge,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

        # Auto-detect multi-turn capability from environment
        enable_multi_turn = getattr(args, "enable_multi_turn", None)
        if enable_multi_turn is None and env is not None:
            # Let MultiTurnMixin auto-detect
            self.is_multi_turn = self.is_multi_turn_environment(env)
        elif enable_multi_turn is not None:
            self.is_multi_turn = enable_multi_turn
        else:
            self.is_multi_turn = False
