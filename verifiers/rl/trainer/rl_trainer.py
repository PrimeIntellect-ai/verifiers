import logging
import shutil
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sized, Tuple, Union, cast

import datasets
import numpy as np
import torch
from accelerate.utils import (
    broadcast_object_list,
    gather_object,
    is_peft_model,
)
from peft import get_peft_model
from torch.utils.data import DataLoader, Sampler
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
)
from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
)
from transformers.trainer import Trainer
from transformers.trainer_callback import (
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from trl.models.modeling_base import create_reference_model
from trl.models.utils import prepare_deepspeed
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
    pad,
    selective_log_softmax,
)

import verifiers as vf
import wandb
from verifiers.rl.inference.vllm_client import VLLMClient
from verifiers.rl.trainer.async_batch_generator import AsyncBatchGenerator, BatchRequest
from verifiers.rl.trainer.async_dataloader_wrapper import AsyncDataLoaderWrapper
from verifiers.rl.trainer.rl_config import RLConfig
from verifiers.utils.logging_utils import print_prompt_completions_sample


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        prompts_per_batch (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, prompts_per_batch=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         prompts_per_batch = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        prompts_per_batch: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.prompts_per_batch = prompts_per_batch
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (prompts_per_batch = 3)
        indexes = [
            indexes[i : i + self.prompts_per_batch]
            for i in range(0, len(indexes), self.prompts_per_batch)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.prompts_per_batch]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (
            (self.num_samples // self.prompts_per_batch)
            * self.prompts_per_batch
            * self.mini_repeat_count
            * self.repeat_count
        )


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean(
        (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
    )  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
        >>> x = torch.arange(12).reshape(6, 2)
        >>> y = torch.arange(6).reshape(6, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> split_tensor_dict(tensor_dict, 3)
        [
            {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
            {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
            {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
        ]
    """
    first_value = next(value for value in tensor_dict.values() if value is not None)
    if isinstance(first_value, list):
        chunk_size = len(first_value) // num_chunks
    else:
        chunk_size = first_value.shape[0] // num_chunks

    chunks: list[dict[str, Optional[torch.Tensor]]] = []
    for i in range(num_chunks):
        chunk: dict[str, Optional[torch.Tensor]] = {}
        for key, tensor in tensor_dict.items():
            if tensor is None:
                chunk[key] = None
            elif isinstance(tensor, list):
                chunk[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
            elif tensor.ndim == 0:
                chunk[key] = tensor
            else:
                chunk[key] = tensor[i * chunk_size : (i + 1) * chunk_size]
        chunks.append(chunk)
    return chunks


def shuffle_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]],
) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.

    Example:
        >>> x = torch.arange(6).reshape(3, 2)
        >>> y = torch.arange(3).reshape(3, 1)
        >>> tensor_dict = {"x": x, "y": y}
        >>> shuffle_tensor_dict(tensor_dict)
        {'x': tensor([[2, 3],
                      [0, 1],
                      [4, 5]]),
         'y': tensor([[1],
                      [0],
                      [2]])}
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {
        key: tensor[permutation] if tensor is not None and tensor.ndim > 0 else tensor
        for key, tensor in tensor_dict.items()
    }


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class RLTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel,
        env: vf.Environment,
        args: RLConfig,
        processing_class: PreTrainedTokenizerBase,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        **kwargs,
    ):
        self.logger = logging.getLogger(__name__)

        # configure LoRA
        if args.lora_config is None:
            raise ValueError("RLTrainer requires a LoRA configuration.")

        model = cast(PreTrainedModel, get_peft_model(model, args.lora_config))

        # Override sync_ref_model if PEFT is used since ref_model will be None
        if args.sync_ref_model:
            self.logger.warning(
                "sync_ref_model=True is not compatible with PEFT. Setting sync_ref_model=False."
            )
            args.sync_ref_model = False

        self.max_saved_adapters = args.max_saved_adapters
        self.adapter_save_dir: Path | None = None
        self._saved_adapters: deque[tuple[str, Path]] = deque()

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)  # type: ignore

        # Suppress irrelevant warning
        model.warnings_issued["estimate_tokens"] = True

        # Tokenizer pad token
        if processing_class.pad_token is None:  # type: ignore
            processing_class.pad_token = processing_class.eos_token  # type: ignore

        # Training arguments
        self.micro_batch_size = args.micro_batch_size
        self.max_prompt_length = args.max_prompt_length
        self.max_seq_len = args.max_seq_len  # max sequence length
        if self.max_seq_len is None:
            raise ValueError("max_seq_len must be configured for training")
        if self.max_prompt_length is None:
            self.max_prompt_length = self.max_seq_len
            self.logger.info(
                f"max_prompt_length is set to {self.max_prompt_length} (max_seq_len={self.max_seq_len})"
            )
        self.max_tokens = args.max_tokens  # max tokens per generation
        if self.max_tokens is None:
            self.max_tokens = self.max_seq_len
            self.logger.info(
                f"max_tokens is set to {self.max_tokens} (max_seq_len={self.max_seq_len})"
            )
        self.rollouts_per_example = args.rollouts_per_example
        self.batch_size = args.batch_size
        self.max_concurrent = args.max_concurrent
        self.max_data_workers = args.max_data_workers
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.top_k = args.top_k
        self.loss_type = args.loss_type
        raw_scale_rewards = args.scale_rewards
        if isinstance(raw_scale_rewards, bool):
            self.scale_rewards_mode = "group" if raw_scale_rewards else "none"
        elif isinstance(raw_scale_rewards, str):
            normalized = raw_scale_rewards.lower()
            if normalized in {"true", "group"}:
                self.scale_rewards_mode = "group"
            elif normalized == "batch":
                self.scale_rewards_mode = "batch"
            elif normalized in {"false", "none"}:
                self.scale_rewards_mode = "none"
            else:
                raise ValueError(
                    "scale_rewards must be a boolean or one of {'group', 'batch', 'none'}"
                )
        else:
            raise ValueError(
                "scale_rewards must be a boolean or one of {'group', 'batch', 'none'}"
            )
        self.scale_rewards = self.scale_rewards_mode != "none"
        self.mask_truncated_completions = args.mask_truncated_completions
        self.zero_truncated_completions = args.zero_truncated_completions
        self.delta = args.delta
        self.importance_sampling_level = args.importance_sampling_level
        self.vllm_importance_sampling_correction = True
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap

        # Reference model parameters
        self.beta = args.beta
        self.sync_ref_model = args.sync_ref_model

        # Multi-step
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self._step = 0
        self._buffered_inputs: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None

        # Data
        self.shuffle_dataset = args.shuffle_dataset
        train_dataset = env.get_dataset()
        assert train_dataset is not None

        eval_dataset = env.get_eval_dataset()

        if "prompt" not in train_dataset.column_names:
            raise ValueError("Train dataset must contain a 'prompt' column")
        if "answer" not in train_dataset.column_names:
            train_dataset = train_dataset.map(
                lambda x: {"answer": ""},
                num_proc=self.max_data_workers,
            )
        if eval_dataset is not None and "answer" not in eval_dataset.column_names:
            eval_dataset = eval_dataset.map(
                lambda x: {"answer": ""},
                num_proc=self.max_data_workers,
            )
        if "info" not in train_dataset.column_names:
            train_dataset = train_dataset.map(
                lambda x: {"info": {}},
                num_proc=self.max_data_workers,
            )
        if eval_dataset is not None and "info" not in eval_dataset.column_names:
            eval_dataset = eval_dataset.map(
                lambda x: {"info": {}},
                num_proc=self.max_data_workers,
            )

        if "task" not in train_dataset.column_names:
            train_dataset = train_dataset.map(
                lambda x: {"task": "default"},
                num_proc=self.max_data_workers,
            )
        if eval_dataset is not None and "task" not in eval_dataset.column_names:
            eval_dataset = eval_dataset.map(
                lambda x: {"task": "default"},
                num_proc=self.max_data_workers,
            )

        # Filter out prompts that are too long if max_prompt_length is set
        if self.max_prompt_length is not None:
            self.logger.info(
                f"Filtering dataset for prompts with length <= {self.max_prompt_length}"
            )
            max_length = self.max_prompt_length  # Capture for closure

            def filter_by_prompt_length(example, processing_class):
                prompt = example["prompt"]
                # Tokenize prompt to check length
                if isinstance(prompt, list):
                    # Chat format
                    prompt_text = processing_class.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # Completion format
                    prompt_text = prompt
                prompt_ids = processing_class.encode(prompt_text)  # type: ignore
                return len(prompt_ids) <= max_length

            original_size = len(train_dataset)
            train_dataset = train_dataset.filter(
                filter_by_prompt_length,
                num_proc=self.max_data_workers,
                fn_kwargs={"processing_class": processing_class},
            )
            filtered_size = len(train_dataset)
            if filtered_size < original_size:
                self.logger.info(
                    f"Filtered dataset from {original_size} to {filtered_size} examples ({original_size - filtered_size} prompts were too long)"
                )

        # dummy data collator
        def data_collator(features):
            return features

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        assert self.args.output_dir is not None
        print(self.args.output_dir)
        self.adapter_save_dir = self._init_adapter_save_dir(Path(self.args.output_dir))
        if self.accelerator.is_main_process:
            self.adapter_save_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.prompts_per_batch = args.prompts_per_batch

        if self.train_dataset is not None:
            dataset_size = len(self.train_dataset)  # type: ignore
            global_prompts_per_batch = self.prompts_per_batch
            global_rollouts = self.batch_size
            self.logger.info(
                f"Dataset size: {dataset_size}, prompts per global batch: {global_prompts_per_batch}"
            )
            self.logger.info(
                f"Gradient accumulation steps: {self.gradient_accumulation_steps}"
            )
            truncated_dataset_size = (
                dataset_size // global_prompts_per_batch
            ) * global_prompts_per_batch
            if truncated_dataset_size < dataset_size:
                self.logger.info(
                    f"Truncating dataset from {dataset_size} to {truncated_dataset_size} examples ({dataset_size - truncated_dataset_size} prompts dropped to fit whole batches)"
                )
                self.train_dataset = self.train_dataset.select(
                    range(truncated_dataset_size)
                )
            self.logger.info(
                f"Batches per epoch: {truncated_dataset_size / global_prompts_per_batch}"
            )
            self.logger.info(
                f"Rollouts per global batch: {global_rollouts}, gradient accumulation steps: {self.gradient_accumulation_steps}"
            )
        # Reference model
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            model_id = model.config._name_or_path
            model_init_kwargs = {"dtype": "auto"}
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_init_kwargs
            )
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)  # type: ignore

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print

        # Environment integration parameters
        self.mask_env_responses = args.mask_env_responses
        self.max_concurrent = args.max_concurrent

        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = (
            self.accelerator.num_processes
            * self.micro_batch_size
            * self.gradient_accumulation_steps
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }

        # OpenAI client for Environment generation (using vLLM server)
        host = args.vllm_server_host
        port = args.vllm_server_port
        vllm_base_url = f"http://{host}:{port}/v1"
        self.client_config = {
            "base_url": vllm_base_url,
            "api_key": "EMPTY",
            "http_client_args": {
                "limits": {"max_connections": args.max_concurrent},
                "timeout": args.async_generation_timeout,
            },
        }

        # vLLM client for weight syncing only; only import if used
        self.vllm_client = VLLMClient(
            host=host, port=port, connection_timeout=args.vllm_server_timeout
        )

        self._last_loaded_step = (
            0  # Initialize to 0 since vLLM already has initial weights
        )
        # Weight updates to vLLM happen only when generating new completions
        # Frequency: every `gradient_accumulation_steps` training steps
        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

        # Reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
        if self.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model,  # type: ignore
                    accelerator=self.accelerator,
                )
            )

        # Environment
        self.env = env

        # Async generation setup
        self._next_batch_id: int = 0
        self._async_started = False
        self.num_batches_ahead = args.num_batches_ahead

        # num_batches_ahead=0 will behave synchronously (submit and wait immediately)
        self.async_generator = AsyncBatchGenerator(
            env=self.env,
            client_config=self.client_config,
            model_name=self._get_model_name(),
            sampling_args=self._get_sampling_args(),
            num_batches_ahead=self.num_batches_ahead,
            max_queue_size=args.async_max_queue_size,
            generation_timeout=args.async_generation_timeout,
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        batch_size_per_process = (
            self.micro_batch_size * self.gradient_accumulation_steps
        )
        print("batch_size_per_process", batch_size_per_process)
        dataloader_params = {
            "batch_size": batch_size_per_process,  # type: ignore (None case handled by config __post_init__)
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)  # type: ignore

        # Always wrap with AsyncDataLoaderWrapper for consistent behavior
        # Store the wrapped dataloader for async access
        self._async_dataloader = AsyncDataLoaderWrapper(
            dataloader, buffer_size=max(5, self.num_batches_ahead * 2)
        )
        return self.accelerator.prepare(self._async_dataloader)

    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |    Accum step 0     |
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  rollouts_per_example=2
        #                                       <-───────> micro_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first gradient_accumulation_steps (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  grad_accum=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second gradient_accumulation_steps (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...

        return RepeatSampler(
            data_source=self.train_dataset,  # type: ignore
            mini_repeat_count=self.rollouts_per_example,
            prompts_per_batch=self.batch_size // self.rollouts_per_example,
            repeat_count=self.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: RLConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()  # type: ignore
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        assert isinstance(gradient_checkpointing_kwargs, dict)
        use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            # Clean up async generator on all processes
            if (
                self.async_generator
                and self._async_started
                and self.accelerator.is_main_process
            ):
                self.async_generator.stop()
            self._async_started = False

    def _get_last_hidden_state(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep=None
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        last_hidden_state = unwrapped_model.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        if logits_to_keep is not None:
            last_hidden_state = last_hidden_state[
                :, -logits_to_keep:, :
            ]  # (B, logits_to_keep, H)
        return last_hidden_state

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed  # type: ignore[unresolved-import]

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if not is_peft_model(self.model):
            raise RuntimeError("RLTrainer requires a PEFT LoRA model for vLLM syncing.")

        # Check if background batch generation is complete on main process
        # and broadcast the state to all processes
        is_generating = False
        if self.accelerator.is_main_process:
            is_generating = self.async_generator.is_generating

        # Broadcast generation state from main process to all processes
        is_generating_list = [is_generating]
        broadcast_object_list(is_generating_list, from_process=0)
        is_generating = is_generating_list[0]

        # All processes wait if generation is happening
        waits = 0
        while is_generating:
            time.sleep(0.5)
            waits += 1
            if waits % 10 == 0:
                self.logger.info(
                    "Waiting for background batch generation to complete before weight syncing."
                )

            # Check again and broadcast
            if self.accelerator.is_main_process:
                is_generating = self.async_generator.is_generating
            is_generating_list = [is_generating]
            broadcast_object_list(is_generating_list, from_process=0)
            is_generating = is_generating_list[0]

        # Ensure all processes are synchronized before weight update
        self.accelerator.wait_for_everyone()
        self.logger.info(
            f"Process {self.accelerator.process_index}: Starting weight sync to vLLM"
        )

        step = self.state.global_step
        base_model_name = self._get_model_name()
        adapter_name = self._format_adapter_name(base_model_name, step)
        adapter_dirname = self._format_adapter_path_name(step)

        # Gather only trainable (adapter) parameters under ZeRO-3 before saving
        peft_model = self.accelerator.unwrap_model(self.model)
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        with gather_if_zero3(trainable_params):  # type: ignore[arg-type]
            if self.accelerator.is_main_process:
                if self.adapter_save_dir is None:
                    raise RuntimeError(
                        "Adapter save directory has not been initialized."
                    )
                adapter_path = self.adapter_save_dir / adapter_dirname
                if adapter_path.exists():
                    shutil.rmtree(adapter_path)
                if not is_peft_model(peft_model):
                    raise RuntimeError("Expected a PEFT LoRA model to save adapter.")
                peft_model.save_pretrained(adapter_path, safe_serialization=True)  # type: ignore[attr-defined]
                assert isinstance(self.processing_class, PreTrainedTokenizerBase)
                if hasattr(self.processing_class, "save_pretrained"):
                    self.processing_class.save_pretrained(adapter_path)
                self.logger.info(
                    "Saved LoRA adapter '%s' to %s", adapter_name, adapter_path
                )
                self.vllm_client.load_lora_adapter(
                    adapter_name, str(adapter_path.resolve())
                )
                self._register_saved_adapter(adapter_name, adapter_path)
                self.vllm_client.reset_prefix_cache()
                # Point async generation to the freshly loaded adapter by name
                self.async_generator.model_name = adapter_name

        # Ensure all processes wait for the main process to finish updating weights
        self.accelerator.wait_for_everyone()

    def _format_adapter_name(self, model_name: str, step: int) -> str:
        return f"{model_name}-step-{int(step)}"

    def _format_adapter_path_name(self, step: int) -> str:
        return f"adapter-step-{int(step)}"

    def _init_adapter_save_dir(self, output_dir: Path) -> Path:
        """Compute where LoRA adapters are stored for vLLM hot-swapping.

        The directory always lives alongside the Trainer's checkpoint folders so
        that transient adapters do not get mixed with checkpoint artifacts like
        optimizer states.
        """

        run_dir = output_dir.resolve()
        if run_dir.name.startswith("checkpoint-"):
            run_dir = run_dir.parent
        return run_dir / "adapters"

    # helper removed; inlined into _move_model_to_vllm for clarity

    def _register_saved_adapter(self, adapter_name: str, adapter_path: Path) -> None:
        self._saved_adapters.append((adapter_name, adapter_path))
        while len(self._saved_adapters) > self.max_saved_adapters:
            old_name, old_path = self._saved_adapters.popleft()
            try:
                self.vllm_client.unload_lora_adapter(old_name)
            except Exception as exc:  # noqa: BLE001 - log and continue cleanup
                self.logger.warning(
                    "Failed to unload LoRA adapter '%s' from vLLM: %s", old_name, exc
                )
            if old_path.exists():
                shutil.rmtree(old_path, ignore_errors=True)

    def _get_sampling_args(self) -> Dict[str, Any]:
        """Get sampling arguments for Environment generation."""
        args = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
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
            },
        }
        return args

    def _get_model_name(self) -> str:
        """Get model name for Environment generation."""
        return self.model.config._name_or_path  # type: ignore

    def _ids_to_tensors(
        self,
        prompt_ids: List[List[int]],
        prompt_mask: List[List[int]],
        completion_ids: List[List[int]],
        completion_mask: List[List[int]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        ids = [prompt_ids[i] + completion_ids[i] for i in range(len(prompt_ids))]
        mask = [prompt_mask[i] + completion_mask[i] for i in range(len(prompt_mask))]
        max_len = max(len(ids[i]) for i in range(len(ids)))
        ids = [
            torch.cat(
                [
                    torch.tensor(ids[i], dtype=torch.long, device=device),
                    torch.zeros(max_len - len(ids[i]), dtype=torch.long, device=device),
                ]
            )
            for i in range(len(ids))
        ]
        mask = [
            torch.cat(
                [
                    torch.tensor(mask[i], dtype=torch.long, device=device),
                    torch.zeros(
                        max_len - len(mask[i]), dtype=torch.long, device=device
                    ),
                ]
            )
            for i in range(len(mask))
        ]
        ids = torch.stack(ids, dim=0)
        mask = torch.stack(mask, dim=0)
        return {"ids": ids, "mask": mask}

    def _gather_batch_data(
        self, batch_offset: int = 0
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
        """
        Gather batch data from all processes.

        Args:
            batch_offset: 0 for current batch, >0 for future batches (peek ahead)

        Returns:
            Tuple of (all_prompts, all_answers, all_tasks)
        """
        batches = self._async_dataloader.peek_ahead(batch_offset)

        if batch_offset == 0:
            batch = batches[0] if batches else None
        else:
            batch = batches[batch_offset - 1] if batches else None

        if batch is None:
            return [], [], [], []

        if isinstance(batch, dict):
            batch = [batch]

        # Gather batch data from all processes
        prompts = [x["prompt"] for x in batch]
        answers = [x["answer"] for x in batch]
        tasks = [x.get("task", "default") for x in batch]
        infos = [x.get("info", {}) for x in batch]
        all_prompts = gather_object(prompts)
        all_answers = gather_object(answers)
        all_tasks = gather_object(tasks)
        all_infos = gather_object(infos)
        return all_prompts, all_answers, all_tasks, all_infos

    def _prepare_inputs(  # type: ignore
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        inputs: raw inputs from the dataloader (per process)

        This method implements async batch generation with priming:
        1. Always maintain num_batches_ahead batches in the pipeline
        2. On first calls, prime by submitting num_batches_ahead batches before retrieving any
        3. On subsequent calls, submit new batches to maintain the pipeline
        """
        # Ensure all processes are synchronized at the start
        self.accelerator.wait_for_everyone()
        # inputs = list of dicts for all gradient accumulation steps
        generate_every = self.gradient_accumulation_steps

        # Check if we need to generate new completions
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            # Update weights to vLLM if needed
            if self.state.global_step > self._last_loaded_step:
                self.logger.info(
                    f"Syncing weights to vLLM at step {self.state.global_step}"
                )
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Start async generator if not started
            if not self._async_started and self.accelerator.is_main_process:
                self.async_generator.start()
            self._async_started = True
            self.accelerator.wait_for_everyone()

            # Calculate which batch we need for this step
            batch_id_to_retrieve = self._step // generate_every

            # Calculate the target: we want to always be num_batches_ahead batches ahead
            # This means we should have submitted up to batch_id_to_retrieve + num_batches_ahead
            target_batch_id = (
                batch_id_to_retrieve + self.async_generator.num_batches_ahead
            )

            # Submit any missing batches to maintain the pipeline
            # On first call, this submits batches 0 through num_batches_ahead
            # On subsequent calls, this submits new batches to stay ahead
            batches_submitted = 0
            steps_per_batch = generate_every
            max_batch_id = (
                self.state.max_steps * self.gradient_accumulation_steps - 1
            ) // steps_per_batch
            target_batch_id = min(target_batch_id, max_batch_id)

            for batch_id in range(self._next_batch_id, target_batch_id + 1):
                first_grad_step_for_batch = batch_id * steps_per_batch
                if (
                    first_grad_step_for_batch
                    >= self.state.max_steps * self.gradient_accumulation_steps
                ):
                    self.logger.info(
                        f"Reached max global steps ({self.state.max_steps}), stopping batch generation"
                    )
                    break
                batch_offset = batch_id - batch_id_to_retrieve
                all_prompts, all_answers, all_tasks, all_infos = (
                    self._gather_batch_data(batch_offset)
                )
                if not all_prompts:
                    self.logger.info(
                        f"No prompts for batch {batch_id}, stopping batch generation"
                    )
                    break

                # Submit batch (main process only)
                if self.accelerator.is_main_process:
                    request = BatchRequest(
                        batch_id=batch_id,
                        env_inputs={
                            "prompt": all_prompts,
                            "answer": all_answers,
                            "task": all_tasks,
                            "info": all_infos,
                        },
                        processing_class=self.processing_class,
                        mask_env_responses=self.mask_env_responses,
                        max_seq_len=self.max_seq_len or -1,
                        mask_truncated_completions=self.mask_truncated_completions,
                        zero_truncated_completions=self.zero_truncated_completions,
                        max_concurrent=self.max_concurrent,
                    )
                    self.async_generator.submit_batch(request)
                    batches_submitted += 1
                self.accelerator.wait_for_everyone()

            # Update next batch id
            if self.accelerator.is_main_process:
                self._next_batch_id = self._next_batch_id + batches_submitted
            self.accelerator.wait_for_everyone()
            # Synchronize next_batch_id across all processes
            next_batch_id_list = [
                self._next_batch_id if self.accelerator.is_main_process else 0
            ]
            broadcast_object_list(next_batch_id_list, from_process=0)
            self._next_batch_id = next_batch_id_list[0]
            self.accelerator.wait_for_everyone()

            # Now retrieve the batch we need for this step
            if self.accelerator.is_main_process:
                # Get batch result
                batch_result = self.async_generator.get_batch(batch_id_to_retrieve)
                processed_results = batch_result.processed_results

                # Package raw data for broadcast (not tensors yet)
                broadcast_data = {
                    "prompt_ids": processed_results.prompt_ids,
                    "prompt_mask": processed_results.prompt_mask,
                    "completion_ids": processed_results.completion_ids,
                    "completion_mask": processed_results.completion_mask,
                    "completion_logprobs": processed_results.completion_logprobs,
                    "rewards": processed_results.rewards,
                    "all_reward_dict": batch_result.all_reward_dict,
                    "completions": batch_result.completions,
                    "prompts": batch_result.prompts,
                }
            else:
                broadcast_data = None
            self.accelerator.wait_for_everyone()

            # Broadcast processed data
            broadcast_list = [broadcast_data]
            broadcast_object_list(broadcast_list, from_process=0)
            broadcast_data = broadcast_list[0]
            self.accelerator.wait_for_everyone()

            # Determine how many rollouts belong to each process in the global batch
            assert broadcast_data is not None, (
                "broadcast_data should be populated on all processes"
            )
            global_batch_size = len(broadcast_data["rewards"])
            if global_batch_size % self.accelerator.num_processes != 0:
                raise ValueError(
                    "Generation batch size must be divisible by the number of processes. "
                    "Check RepeatSampler configuration."
                )
            per_process_rollouts = global_batch_size // self.accelerator.num_processes
            if per_process_rollouts % self.gradient_accumulation_steps != 0:
                raise ValueError(
                    "Per-process rollouts must be divisible by gradient_accumulation_steps."
                )
            if inputs and len(inputs) != per_process_rollouts:
                raise ValueError(
                    "DataLoader provided a batch size that does not match the expected "
                    "per-process rollouts. Ensure RepeatSampler is configured correctly."
                )

            process_slice = slice(
                self.accelerator.process_index * per_process_rollouts,
                (self.accelerator.process_index + 1) * per_process_rollouts,
            )

            # Create rewards tensor and compute advantages using full batch
            all_rewards = torch.tensor(
                broadcast_data["rewards"], device=self.accelerator.device
            )
            all_advantages = self._compute_advantages(all_rewards)

            completion_token_counts = [
                sum(mask) for mask in broadcast_data["completion_mask"]
            ]
            num_items_in_batch = torch.tensor(
                float(sum(completion_token_counts)),
                device=self.accelerator.device,
            )

            # Now create tensors only for this process's slice
            input_ids_list = []
            attention_mask_list = []
            sampling_logprobs_list: list[torch.Tensor] = []

            for i in range(process_slice.start, process_slice.stop):
                input_ids_list.append(
                    torch.tensor(
                        broadcast_data["prompt_ids"][i]
                        + broadcast_data["completion_ids"][i],
                        device=self.accelerator.device,
                    )
                )
                attention_mask_list.append(
                    torch.tensor(
                        broadcast_data["prompt_mask"][i]
                        + broadcast_data["completion_mask"][i],
                        device=self.accelerator.device,
                    )
                )
                sampling_logprobs_list.append(
                    torch.tensor(
                        broadcast_data["completion_logprobs"][i],
                        device=self.accelerator.device,
                        dtype=torch.float32,
                    )
                )

            input_ids = pad(
                input_ids_list,
                padding_value=self.processing_class.pad_token_id,  # type: ignore
                padding_side="right",
            )  # type: ignore
            attention_mask = pad(attention_mask_list, padding_side="right")  # type: ignore
            if sampling_logprobs_list:
                sampling_per_token_logps = pad(
                    sampling_logprobs_list,
                    padding_value=0,
                    padding_side="right",
                )
            else:
                sampling_per_token_logps = None

            # Truncate if needed
            if self.max_seq_len is not None and input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len :]
                attention_mask = attention_mask[:, -self.max_seq_len :]

            # Take this process's slice of advantages
            advantages = all_advantages[process_slice]

            # Log metrics on main process only
            if self.accelerator.is_main_process:
                self._log_reward_metrics_primary(
                    mode="train",
                    all_reward_dict=broadcast_data["all_reward_dict"],
                    all_rewards=all_rewards,
                )

                self._log_textual_data_primary(
                    all_prompts=broadcast_data["prompts"],
                    all_completions=broadcast_data["completions"],
                    all_reward_dict=broadcast_data["all_reward_dict"],
                )

                # Log completion metrics using full batch data on CPU to save memory
                self._log_completion_metrics_primary(
                    mode="train",
                    all_completion_mask=broadcast_data["completion_mask"],
                    all_completion_ids=broadcast_data["completion_ids"],
                    all_prompt_mask=broadcast_data["prompt_mask"],
                )
            importance_sampling_ratio: Optional[torch.Tensor] = None

            with torch.no_grad():
                completion_mask = attention_mask[:, 1:]
                logits_to_keep = completion_mask.size(1)
                if sampling_per_token_logps is not None:
                    # Align width with sampling logprobs (completion-only) if present
                    logits_to_keep = min(
                        logits_to_keep, sampling_per_token_logps.size(1)
                    )
                    sampling_per_token_logps = sampling_per_token_logps[
                        :, :logits_to_keep
                    ]

                # Compute model per-token logps for the aligned width
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=self.micro_batch_size,
                )

                # Align completion mask width for downstream masking/where
                completion_mask = completion_mask[:, -logits_to_keep:]

                if (
                    sampling_per_token_logps is not None
                    and self.vllm_importance_sampling_correction
                ):
                    importance_sampling_ratio = torch.exp(
                        old_per_token_logps - sampling_per_token_logps
                    )
                    importance_sampling_ratio = torch.clamp(
                        importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                    importance_sampling_ratio = torch.where(
                        completion_mask.bool(),
                        importance_sampling_ratio,
                        torch.ones_like(importance_sampling_ratio),
                    )

            # Concatenate all data for shuffling
            full_batch: dict[str, torch.Tensor | None] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_per_token_logps": old_per_token_logps,
                "advantages": advantages,
                "num_items_in_batch": num_items_in_batch,
                "importance_sampling_ratio": importance_sampling_ratio,
            }

            # Shuffle and split for gradient accumulation
            full_batch = shuffle_tensor_dict(full_batch)
            self._buffered_inputs = split_tensor_dict(
                full_batch, self.gradient_accumulation_steps
            )
            self.accelerator.wait_for_everyone()
        # Return appropriate slice from buffer
        result = self._buffered_inputs[self._step % self.gradient_accumulation_steps]
        self._step += 1
        self.accelerator.wait_for_everyone()
        return result

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages from rewards with normalization using full batch statistics."""
        grouped_rewards = rewards.view(-1, self.rollouts_per_example)
        mean_grouped = grouped_rewards.mean(dim=1)
        mean_grouped = mean_grouped.repeat_interleave(self.rollouts_per_example, dim=0)
        advantages = rewards - mean_grouped

        if self.scale_rewards_mode == "batch":
            std = rewards.std()
            std = std.expand_as(rewards)
        else:
            std = grouped_rewards.std(dim=1)
            std = std.repeat_interleave(self.rollouts_per_example, dim=0)

        if self.scale_rewards_mode != "none":
            advantages = advantages / (std + 1e-4)

        return advantages

    def compute_loss(  # type: ignore
        self,  # type: ignore
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:  # type: ignore
        mode = "train"
        # Compute the per-token log probabilities for the model
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

        # prompt is at least 1 token
        completion_mask = attention_mask[:, 1:]
        logits_to_keep = completion_mask.size(1)
        # If importance_sampling_ratio provided (from sampling logprobs), align widths
        if (
            self.vllm_importance_sampling_correction
            and inputs.get("importance_sampling_ratio") is not None
        ):
            ratio = inputs["importance_sampling_ratio"]
            logits_to_keep = min(logits_to_keep, ratio.size(1))
            # Align completion mask width to ratio width
            completion_mask = completion_mask[:, -logits_to_keep:]
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        completion_mask_bool = completion_mask.bool()
        completion_mask = completion_mask_bool.float()
        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"]
        if old_per_token_logps is None:
            old_per_token_logps = per_token_logps.detach()

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (
                (log_ratio * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}"
            )

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)

        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():  # type: ignore
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()  # type: ignore
            )
        if (
            self.vllm_importance_sampling_correction
            and inputs.get("importance_sampling_ratio") is not None
        ):
            importance_sampling_ratio = inputs["importance_sampling_ratio"]
            # Ensure the ratio width matches current logits_to_keep
            if importance_sampling_ratio.size(1) != per_token_loss.size(1):
                importance_sampling_ratio = importance_sampling_ratio[
                    :, : per_token_loss.size(1)
                ]
            per_token_loss = per_token_loss * importance_sampling_ratio

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_seq_len
            )  # type: ignore
        elif self.loss_type == "dapo":
            normalizer = inputs.get("num_items_in_batch")
            if normalizer is None:
                normalizer = completion_mask.sum()
            loss = (per_token_loss * completion_mask).sum() / (
                normalizer / self.accelerator.num_processes
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()  # type: ignore
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()  # type: ignore
        )
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()  # type: ignore
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()  # type: ignore
        )
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()  # type: ignore
        )

        if (
            self.vllm_importance_sampling_correction
            and inputs.get("importance_sampling_ratio") is not None
        ):
            importance_sampling_ratio = inputs["importance_sampling_ratio"]
            flat_ratio: torch.Tensor = importance_sampling_ratio[completion_mask_bool]
            if flat_ratio.numel() == 0:
                flat_ratio = torch.tensor(
                    float("nan"), device=importance_sampling_ratio.device
                )
                mean_ratio = flat_ratio
            else:
                mean_ratio = flat_ratio.mean()
            min_ratio = nanmin(cast(torch.Tensor, self.accelerator.gather(flat_ratio)))
            max_ratio = nanmax(cast(torch.Tensor, self.accelerator.gather(flat_ratio)))
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                min_ratio.item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                cast(torch.Tensor, self.accelerator.gather(mean_ratio)).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                max_ratio.item()
            )
        return loss

    def _sanitize_tool_calls(
        self,
        completion: list[dict[str, Any]] | str,
    ) -> list[dict[str, Any]] | str:
        if isinstance(completion, str):
            return completion
        for msg in completion:
            if "tool_calls" in msg:
                tool_calls = []
                msg["tool_calls"] = []
                for tc in msg["tool_calls"]:
                    tool_calls.append(
                        {
                            "name": tc.get("function", {}).get("name", ""),
                            "args": tc.get("function", {}).get("arguments", {}),
                        }
                    )
                msg["content"] += str({"tool_calls": tool_calls})
                msg.pop("tool_calls")
            if "tool_call_id" in msg:
                msg.pop("tool_call_id")
        return completion

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs
    ):
        """
        Override the evaluate method to use env.evaluate() directly.
        This bypasses the standard batch-by-batch evaluation and uses the environment's
        built-in evaluation logic instead.
        """
        # Skip evaluation if no eval dataset is available
        if self.env.eval_dataset is None:
            self.logger.info("Skipping evaluation - no eval dataset available")
            return {}

        self.logger.info("Running evaluation using environment's evaluate method")

        # Only the main process computes evaluation to avoid duplicate work
        if self.accelerator.is_main_process:
            eval_results = self.async_generator.evaluate(num_samples=-1)
        else:
            eval_results = None

        # Broadcast the results from rank 0 to all other ranks
        broadcast_list = [eval_results]
        broadcast_object_list(broadcast_list, from_process=0)
        eval_results = broadcast_list[0]
        assert eval_results is not None
        # Process results to compute metrics
        metrics = {}

        # Compute reward statistics
        rewards = torch.tensor(eval_results.reward)
        metrics["eval_reward"] = rewards.mean().item()
        metrics["eval_reward_std"] = rewards.std().item()

        # Log individual reward function scores
        non_reward_metric_keys = [
            "reward",
            "prompt",
            "completion",
            "info",
            "answer",
            "state",
            "task",
        ]
        for key in eval_results.metrics:
            if key not in non_reward_metric_keys:
                reward_values = eval_results.metrics[key]
                if isinstance(reward_values, list):
                    metrics[f"eval_rewards/{key}"] = float(np.mean(reward_values))
                else:
                    try:
                        metrics[f"eval_rewards/{key}"] = reward_values.mean().item()
                    except Exception:
                        continue

        # Compute completion length statistics
        assert isinstance(self.processing_class, PreTrainedTokenizerBase)
        completions = eval_results.completion
        if isinstance(completions[0], str):
            # Completion format - directly tokenize strings
            completion_lengths = [
                len(self.processing_class.encode(c))  # type: ignore
                for c in completions
            ]
        else:
            # Chat format - use apply_chat_template
            completion_lengths = []
            for comp in completions:
                # Apply chat template to get the full text
                tokens = self.processing_class.apply_chat_template(
                    comp,  # type: ignore
                    tokenize=True,
                    add_generation_prompt=False,
                )
                # Tokenize and count
                completion_lengths.append(len(tokens))

        metrics["eval_completions/mean_length"] = float(np.mean(completion_lengths))
        metrics["eval_completions/min_length"] = int(np.min(completion_lengths))
        metrics["eval_completions/max_length"] = int(np.max(completion_lengths))

        # Log sample completions if requested
        if self.accelerator.is_main_process and self.log_completions:
            # Prepare textual logs
            prompts = eval_results.prompt[: self.num_completions_to_print]
            completions = eval_results.completion[: self.num_completions_to_print]

            # Extract rewards for logging
            reward_dict = {}
            reward_dict["reward"] = eval_results.reward[: self.num_completions_to_print]
            for key in eval_results.metrics:
                reward_dict[key] = eval_results.metrics[key][
                    : self.num_completions_to_print
                ]

            # Print sample
            print_prompt_completions_sample(
                prompts,
                completions,
                reward_dict["reward"],
                self.state.global_step,
            )

            # Log to wandb if available
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                table_data = {
                    "step": [str(self.state.global_step)] * len(prompts),
                    "prompt": prompts,
                    "completion": [
                        self._sanitize_tool_calls(c)  # type: ignore
                        for c in completions
                    ],
                }
                for k, v in reward_dict.items():
                    table_data[k] = v

                df = pd.DataFrame(table_data)
                wandb.log({"eval_completions": wandb.Table(dataframe=df)})

        # Log all metrics
        self.log(metrics)

        # Return metrics dict to match base class signature
        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval"  # type: ignore
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if len(self._textual_logs["prompt"]) > 0:
                print_prompt_completions_sample(
                    self._textual_logs["prompt"],
                    self._textual_logs["completion"],
                    self._textual_logs["rewards"]["reward"],
                    self.state.global_step,
                )

            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)]
                    * len(self._textual_logs["prompt"]),
                    "prompt": list(self._textual_logs["prompt"]),
                    "completion": [
                        self._sanitize_tool_calls(c)
                        for c in self._textual_logs["completion"]
                    ],
                    **{k: list(v) for k, v in self._textual_logs["rewards"].items()},
                }
                if len(table["prompt"]) > 0:
                    df = pd.DataFrame(table)
                    if self.wandb_log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            # Clear the textual logs after logging
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

    def _log_reward_metrics_primary(
        self,
        mode: str,
        all_reward_dict: Dict[str, Any],
        all_rewards: torch.Tensor,
    ) -> None:
        """
        Log generation metrics (PRIMARY PROCESS ONLY).
        This handles reward statistics and per-reward-function metrics using the full batch data.
        """
        # Log reward statistics using full batch
        mean_rewards = all_rewards.view(-1, self.rollouts_per_example).mean(dim=1)
        std_rewards = all_rewards.view(-1, self.rollouts_per_example).std(dim=1)
        self._metrics[mode]["reward"].append(mean_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())

        # Log individual reward function scores as metrics
        for reward_key in all_reward_dict:
            if reward_key != "reward":  # Skip the consolidated reward
                reward_values = all_reward_dict[reward_key]
                if isinstance(reward_values, list):
                    reward_tensor = torch.tensor(
                        reward_values, device=all_rewards.device
                    )
                else:
                    reward_tensor = reward_values
                mean_reward = reward_tensor.mean().item()
                self._metrics[mode][f"rewards/{reward_key}"].append(mean_reward)

    def _log_textual_data_primary(
        self,
        all_prompts: List[Union[str, List[Dict[str, Any]]]],
        all_completions: List[Union[str, List[Dict[str, Any]]]],
        all_reward_dict: Dict[str, Any],
    ) -> None:
        """
        Log textual data for wandb (PRIMARY PROCESS ONLY).
        This logs the full batch of prompts, completions, and rewards.
        """
        self._textual_logs["prompt"].extend(all_prompts)
        self._textual_logs["completion"].extend(all_completions)

        # Log all reward scores - both individual functions and consolidated
        for reward_key in all_reward_dict:
            reward_values = all_reward_dict[reward_key]
            self._textual_logs["rewards"][reward_key].extend(
                reward_values.tolist()
                if isinstance(reward_values, torch.Tensor)
                else reward_values
            )

    def _log_completion_metrics_primary(
        self,
        mode: str,
        all_completion_mask: List[List[int]],
        all_completion_ids: List[List[int]],
        all_prompt_mask: List[List[int]],
    ) -> None:
        """
        Log completion-related metrics (PRIMARY PROCESS ONLY).
        This handles completion length statistics using the full batch data.
        """
        # Log token count
        if mode == "train":
            total_tokens = sum(
                len(pm) + len(cm)
                for pm, cm in zip(all_prompt_mask, all_completion_mask)
            )
            self.state.num_input_tokens_seen += total_tokens
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        completion_lengths = [sum(mask) for mask in all_completion_mask]
        self._metrics[mode]["completions/mean_length"].append(
            float(sum(completion_lengths)) / len(completion_lengths)
        )
        self._metrics[mode]["completions/min_length"].append(
            float(min(completion_lengths))
        )
        self._metrics[mode]["completions/max_length"].append(
            float(max(completion_lengths))
        )

        # Check for EOS tokens
        term_lengths = []
        for comp_ids, comp_mask in zip(all_completion_ids, all_completion_mask):
            has_eos = any(
                token == self.processing_class.eos_token_id  # type: ignore
                for token, mask in zip(comp_ids, comp_mask)
                if mask
            )
            if has_eos:
                term_lengths.append(sum(comp_mask))

        clipped_completions_ratio = 1 - len(term_lengths) / len(completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )

        if len(term_lengths) == 0:
            term_lengths = [0]
        self._metrics[mode]["completions/mean_terminated_length"].append(
            float(sum(term_lengths)) / len(term_lengths)
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            float(min(term_lengths))
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            float(max(term_lengths))
        )
