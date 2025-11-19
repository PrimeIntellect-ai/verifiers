from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft import LoraConfig
from transformers import TrainingArguments
from transformers.trainer_utils import SchedulerType


@dataclass
class RLConfig(TrainingArguments):
    """
    Configuration class for RLTrainer.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS

    # LoRA parameters
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: List[str] | str | None = field(
        default=None,
        metadata={"help": "LoRA target modules (all linear layers by default)."},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Full model modules to train (instead of LoRA modules)."},
    )
    lora_use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether to use RSLoRA."},
    )
    lora_config: Optional[LoraConfig] = field(
        default=None,
        metadata={"help": "LoRA configuration."},
    )

    # batch arguments
    rollouts_per_example: int = field(
        default=16,
        metadata={"help": "Number of completions to generate for each example."},
    )
    batch_size: int = field(
        default=512,
        metadata={"help": "Number of total rollouts to use per batch."},
    )
    micro_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per device per step."},
    )
    max_seq_len: int = field(
        default=2048,
        metadata={"help": "Maximum length for training sequences."},
    )
    max_prompt_len: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    max_steps: int = field(
        default=500,
        metadata={"help": "Total number of training steps to perform."},
    )
    max_concurrent: int = field(
        default=1024,
        metadata={"help": "Maximum number of concurrent requests to the environment."},
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for `AdamW` optimizer."},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for `AdamW` optimizer."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for `AdamW` optimizer."},
    )
    mask_ratio_low: float = field(
        default=0.125,
        metadata={"help": "Mask ratio for low clipping."},
    )
    mask_ratio_high: float = field(
        default=8.0,
        metadata={"help": "Mask ratio for high clipping."},
    )
    mask_env_responses: bool = field(
        default=True,
        metadata={
            "help": "Whether to mask the environment responses. If `True`, the environment responses are masked, "
            "preventing them from being incorrectly penalized and introducing noise during training."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    zero_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Whether to give zero reward to truncated completions."},
    )
    # sampling_args for generation
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens to generate (per turn)."},
    )
    temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."
        },
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty (default 0.0)"},
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty (default 0.0)"},
    )

    # generation parameters
    generation_timeout: float = field(
        default=600.0,
        metadata={
            "help": "Timeout in seconds for generation. If a batch doesn't complete within this time, "
            "a TimeoutError is raised."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: float = field(
        default=300.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # other TrainingArguments parameters
    output_dir: str | None = field(
        default=None,
        metadata={"help": "Where to store artifacts and checkpoints."},
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "An optional experiment name for logging."},
    )
    lr_scheduler_type: str | SchedulerType = field(
        default="constant",
        metadata={"help": "Learning rate scheduler type."},
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "When to save checkpoints (no, steps, epoch)."},
    )
    save_steps: float = field(
        default=50,
        metadata={
            "help": "Save checkpoint every X updates steps when save_strategy=steps."
        },
    )
    eval_strategy: str = field(
        default="no",
        metadata={"help": "When to evaluate (no, steps, epoch)."},
    )
    eval_steps: float | None = field(
        default=50,
        metadata={"help": "Evaluate every X updates steps when eval_strategy=steps."},
    )
    save_only_model: bool = field(
        default=True,
        metadata={
            "help": "If True, save only model weights (not optimizer/scheduler)."
        },
    )
    logging_steps: float = field(
        default=1,
        metadata={"help": "Log every X updates steps."},
    )
    log_on_each_node: bool = field(
        default=False,
        metadata={"help": "Whether to log on each node in multi-node setup."},
    )
    report_to: Optional[Union[str, List[str]]] = field(
        default="wandb",
        metadata={"help": "Integration to report results and logs to (e.g., 'wandb')."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    def __post_init__(self):
        # configure output dir
        if self.output_dir is None:
            self.output_dir = f"outputs/{self.run_name}"

        # configure lora
        if not self.use_lora:
            self.lora_config = None
        else:
            if self.lora_target_modules is None:
                self.lora_target_modules = [
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                ]
            if self.lora_config is None:
                self.lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    target_modules=self.lora_target_modules,
                    task_type="CAUSAL_LM",
                )

        self.per_device_train_batch_size = self.micro_batch_size
        if self.eval_strategy != "no":
            self.per_device_eval_batch_size = self.micro_batch_size

        self.sampling_args = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens or self.max_seq_len,
            "n": 1,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logprobs": True,
            "extra_body": {
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                "include_stop_str_in_output": False,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
                "prompt_logprobs": True,
            },
        }
        self.gradient_accumulation_steps = 1
        super().__post_init__()

        num_processes = self.world_size
        assert self.batch_size % (self.micro_batch_size * num_processes) == 0, (
            "batch_size must be divisible by (micro_batch_size * num_processes)."
        )

        assert self.rollouts_per_example > 1, (
            "2 or more rollouts per example are required."
        )
