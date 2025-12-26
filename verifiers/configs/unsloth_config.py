from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft import LoraConfig
from transformers.trainer_utils import SchedulerType


@dataclass
class UnslothConfig:
    """
    Configuration class for Unsloth Trainer.
    """

    # Model Load parameters
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit precision for model weights."},
    )

    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit precision for model weights."},
    )

    load_in_16bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit precision for model weights."},
    )

    full_finetuning: bool = field(  
        default=False,
        metadata={"help": "Whether to fine-tune the entire model."},
    )

    use_exact_model_name: bool = field(
        default=False,
        metadata={"help": "Whether to use the exact model name without mapping."},
    )

    gpu_memory_utilization: float = field(
        default=0.8,
        metadata={"help": "Target GPU memory utilization for model loading."},
    )

    random_state: int = field(
        default=3407,
        metadata={"help": "Random state for reproducibility."},
    )

    max_lora_rank: int = field(
        default=64,
        metadata={"help": "Maximum allowable rank for LoRA adapters."},
    )

    token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface token for private model access."},
    )

    # Model Lora parameters
    r: int = field(
        default=16,
        metadata={"help": "LoRA rank."},
    )

    target_modules: List[str] = field(
        default= [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        metadata={"help": "Target modules for LoRA."},
    )

    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter."},
    )

    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout rate."},
    )

    use_gradient_checkpointing: str = field(
        default="unsloth",
        metadata={"help": "Gradient checkpointing strategy."},
    )

    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether to use RS-LoRA."},
    )

    loftq_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Configuration for LoFT-Q."},
    )
