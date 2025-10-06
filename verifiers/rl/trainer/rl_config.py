from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft import LoraConfig
from transformers import TrainingArguments
from transformers.trainer_utils import SchedulerType


@dataclass
class RLConfig(TrainingArguments):
    r"""
    Configuration class for the [`RLTrainer`].

    Only the parameters specific to GRPO training are listed here. For details on other parameters, refer to the
    [`~transformers.TrainingArguments`] documentation.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS

    # LoRA parameters
    use_lora: bool = field(
        default=True,
        metadata={
            "help": "Whether to use LoRA. Must remain `True` - the trainer only supports LoRA fine-tuning."
        },
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA rank."},
    )
    lora_alpha: int = field(
        default=8,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout."},
    )
    lora_target_modules: List[str] | None = field(
        default=None,
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Full model modules to train (instead of LoRA modules)."},
    )
    lora_use_rslora: bool = field(
        default=True,
        metadata={"help": "Whether to use RSLoRA."},
    )
    lora_config: Optional[LoraConfig] = field(
        default=None,
        metadata={"help": "LoRA configuration."},
    )
    max_saved_adapters: int = field(
        default=3,
        metadata={
            "help": "Number of most recent LoRA adapters to keep on disk during training. Older adapters are deleted."
        },
    )

    # Batch size parameters
    micro_batch_size: int = field(
        default=16,
        metadata={"help": "Rollouts per device per optimizer micro step."},
    )
    batch_size: int = field(
        default=512,
        metadata={
            "help": "Global batch size measured in rollouts across all devices and accumulation steps."
        },
    )
    rollouts_per_example: int = field(
        default=16,
        metadata={
            "help": "Number of rollouts to sample per prompt when generating training data."
        },
    )

    # Parameters that control the training
    learning_rate: float = field(
        default=1e-5,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    scale_rewards: str = field(
        default="none",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_seq_len`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`micro_batch_size==1`, the loss is equivalent to the GRPO loss."
        },
    )
    mask_env_responses: bool = field(
        default=True,
        metadata={
            "help": "Whether to mask the environment responses. If `True`, the environment responses are masked, "
            "preventing them from being incorrectly penalized and introducing noise during training."
        },
    )
    mask_truncated_completions: bool = field(
        default=True,
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
    vllm_importance_sampling_cap: float = field(
        default=2.0,
        metadata={
            "help": "Truncation parameter C for Truncated Importance Sampling (TIS). This sets an upper bound on the "
            "importance sampling ratio, improving training stability."
        },
    )

    # Parameters that control the model and reference model
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )

    # Common TrainingArguments surfaced for better typing
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
    warmup_steps: int = field(
        default=0,
        metadata={"help": "Linear warmup over warmup_steps."},
    )
    max_steps: int = field(
        default=500,
        metadata={
            "help": "Total number of training steps to perform. -1 for full epochs."
        },
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bfloat16 precision."},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Max gradient norm for clipping."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        init=False,
        repr=False,
    )
    prompts_per_batch: int = field(
        default=0,
        init=False,
        repr=False,
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Enable gradient checkpointing to save memory."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "When to save checkpoints (no, steps, epoch)."},
    )
    save_steps: float = field(
        default=100,
        metadata={
            "help": "Save checkpoint every X updates steps when save_strategy=steps."
        },
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
        default=True,
        metadata={"help": "Whether to log on each node in multi-node setup."},
    )
    report_to: Optional[Union[str, List[str]]] = field(
        default="wandb",
        metadata={"help": "Integration to report results and logs to (e.g., 'wandb')."},
    )

    # Parameters that control the model and reference model
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be removed from the dataset."
        },
    )
    shuffle_dataset: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tokens to generate per turn."},
    )
    max_seq_len: Optional[int] = field(
        default=2048,
        metadata={"help": "Maximum number of tokens per training sequence."},
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
        default=None,
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
    max_data_workers: int = field(
        default=8,
        metadata={
            "help": "Maximum number of processes to use for filtering the dataset."
        },
    )
    max_concurrent: int = field(
        default=10000,
        metadata={"help": "Maximum number of concurrent requests to the environment."},
    )
    # Async generation parameters
    num_batches_ahead: int = field(
        default=1,
        metadata={
            "help": "Number of batches to generate ahead. Higher values can improve GPU utilization but use more memory. "
            "Set to 0 for synchronous generation (submit and wait immediately, no look-ahead)."
        },
    )
    async_generation_timeout: float = field(
        default=600.0,
        metadata={
            "help": "Timeout in seconds for async generation. If a batch doesn't complete within this time, "
            "a TimeoutError is raised."
        },
    )
    async_max_queue_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of batches that can be queued for async generation. If None, defaults to "
            "2 * num_batches_ahead."
        },
    )

    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."
        },
    )

    # Parameters that control the vLLM server
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to."},
    )
    vllm_server_timeout: float = field(
        default=600.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=True,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of completions to print with `rich`. If `None`, all completions are logged."
        },
    )
    wandb_log_unique_prompts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f"outputs/{self.run_name}"

        self.per_device_train_batch_size = self.micro_batch_size
        super().__post_init__()

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
        if not self.use_lora:
            raise ValueError("RLTrainer is LoRA-only; set `use_lora=True`.")

        if self.max_saved_adapters <= 0:
            raise ValueError("max_saved_adapters must be a positive integer.")

        if self.lora_config is None:
            self.lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                task_type="CAUSAL_LM",
            )

        if self.rollouts_per_example < 2:
            raise ValueError(
                "GRPO requires at least 2 rollouts per example to compute advantages."
            )
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        num_processes = self.world_size
        global_rollouts_per_step = self.micro_batch_size * num_processes
        if global_rollouts_per_step == 0:
            raise ValueError(
                "At least one process and a positive micro batch are required."
            )

        if self.batch_size % global_rollouts_per_step != 0:
            raise ValueError(
                "batch_size must be divisible by micro_batch_size * world_size."
            )
        self.gradient_accumulation_steps = self.batch_size // global_rollouts_per_step
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Derived gradient_accumulation_steps must be positive.")

        if self.batch_size % self.rollouts_per_example != 0:
            raise ValueError(
                "batch_size must be divisible by rollouts_per_example so each prompt receives an equal number of rollouts."
            )
        self.prompts_per_batch = self.batch_size // self.rollouts_per_example

        if self.eval_strategy != "no":
            global_eval_batch_size = self.per_device_eval_batch_size * num_processes
            if global_eval_batch_size % self.rollouts_per_example != 0:
                raise ValueError(
                    "The global eval batch size must be divisible by rollouts_per_example."
                )
        # print all device/batch size params with keys
        print("micro_batch_size", self.micro_batch_size)
        print("gradient_accumulation_steps", self.gradient_accumulation_steps)
        print("prompts_per_batch", self.prompts_per_batch)
        print("batch_size", self.batch_size)
        print("rollouts_per_example", self.rollouts_per_example)
        print("world_size", self.world_size)
