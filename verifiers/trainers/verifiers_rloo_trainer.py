"""
VerifiersRLOOTrainer - Adapter to use TRL's RLOOTrainer with verifiers Environments

This module provides a lightweight adapter (similar to VerifiersGRPOTrainer) that allows using
TRL's RLOOTrainer with verifiers Environment reward system instead of maintaining a separate
copy of the trainer code.

Key features:
- 90% code reduction by wrapping TRL's RLOOTrainer instead of copying it
- Automatic TRL updates and improvements
- Seamless integration with verifiers Environments
- Multi-turn support via MultiTurnMixin
"""

from typing import Optional, Union

from datasets import Dataset, IterableDataset
from trl import RLOOTrainer as TRLRLOOTrainer
from trl import RLOOConfig as TRLRLOOConfig

from verifiers.envs import Environment

from .environment_reward_adapter import EnvironmentRewardAdapter
from .multi_turn_mixin import MultiTurnMixin


class VerifiersRLOOConfig(TRLRLOOConfig):
    """Configuration for VerifiersRLOOTrainer, extending TRL's RLOOConfig"""

    def __init__(self, *args, **kwargs):
        # Extract verifiers-specific parameters before passing to TRL
        self.use_environment_reward = kwargs.pop("use_environment_reward", True)
        self.environment_reward_weight = kwargs.pop("environment_reward_weight", 1.0)

        # Remove any legacy parameters that might cause issues with TRL
        kwargs.pop("enable_async_generation", None)
        kwargs.pop("num_batches_ahead", None)

        # Multi-turn conversation parameters
        self.enable_multi_turn = kwargs.pop("enable_multi_turn", None)

        # Call parent constructor with cleaned kwargs
        super().__init__(*args, **kwargs)


class VerifiersRLOOTrainer(MultiTurnMixin, TRLRLOOTrainer):
    """
    RLOO Trainer with verifiers Environment integration.

    This is a lightweight adapter that:
    1. Wraps TRL's RLOOTrainer (instead of copying 1000+ lines)
    2. Integrates verifiers Environment reward system
    3. Auto-detects single vs multi-turn environments
    4. Automatically gets TRL updates and improvements

    Usage:
        ```python
        import verifiers as vf
        from verifiers.trainers import VerifiersRLOOTrainer, VerifiersRLOOConfig

        # Load environment and model
        env = vf.load_environment("reverse-text")
        model, tokenizer = vf.get_model_and_tokenizer("willcb/Qwen3-0.6B")

        # Configure trainer
        args = VerifiersRLOOConfig(
            output_dir="./output",
            use_environment_reward=True,
            use_vllm=True,
            vllm_mode="colocate",
        )

        # Train
        trainer = VerifiersRLOOTrainer(
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
        env: Optional[Environment] = None,
        args: Optional[VerifiersRLOOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class=None,
        reward_processing_classes=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
        **kwargs,
    ):
        # Create environment reward adapter if environment rewards are enabled
        if env is not None and getattr(args, "use_environment_reward", True):
            reward_adapter = EnvironmentRewardAdapter(env)
        else:
            reward_adapter = None

        # Get dataset from environment if not provided
        if train_dataset is None and env is not None and hasattr(env, "get_dataset"):
            train_dataset = env.get_dataset()
        if eval_dataset is None and env is not None and hasattr(env, "get_eval_dataset"):
            eval_dataset = env.get_eval_dataset()

        # Store environment for later use
        self.env = env

        # Initialize TRL's RLOOTrainer with our reward adapter
        super().__init__(
            model=model,
            reward_funcs=reward_adapter,
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
