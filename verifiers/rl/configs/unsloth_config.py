from dataclasses import dataclass, field
from typing import List, Optional

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

    # Additional Model Lora parameters

    use_gradient_checkpointing: str = field(
        default="unsloth",
        metadata={"help": "Gradient checkpointing strategy."},
    )

    loftq_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Configuration for LoFT-Q."},
    )

