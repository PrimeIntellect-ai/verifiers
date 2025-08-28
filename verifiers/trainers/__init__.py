try:
    import torch._dynamo  # type: ignore

    torch._dynamo.config.suppress_errors = True  # type: ignore
except ImportError:
    pass

from peft import LoraConfig  # type: ignore

from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer
from .verifiers_grpo_trainer import VerifiersGRPOTrainer, VerifiersGRPOConfig, EnvironmentRewardAdapter
from .multi_turn_mixin import MultiTurnMixin


def grpo_defaults(run_name: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=500,
        bf16=True,
        max_grad_norm=0.01,
        num_iterations=1,
        max_seq_len=4096,
        per_device_train_batch_size=8,
        num_generations=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=500,
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
    )


def lora_defaults(r=8, alpha=16) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


__all__ = [
    "GRPOConfig", 
    "GRPOTrainer", 
    "VerifiersGRPOTrainer", 
    "VerifiersGRPOConfig", 
    "EnvironmentRewardAdapter",
    "MultiTurnMixin",
    "grpo_defaults", 
    "lora_defaults"
]
