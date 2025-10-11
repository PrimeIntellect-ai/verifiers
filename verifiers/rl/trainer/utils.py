"""
Adapted from:
- https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
- https://github.com/huggingface/trl/blob/main/trl/models/utils.py
"""

from typing import Any, Optional, cast

import numpy as np
import torch
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft import PeftConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
)


def get_model(
    model_name: str,
    use_liger: bool = True,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if use_liger:
        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_model_and_tokenizer(
    model_name: str, use_liger: bool = True, model_kwargs: dict[str, Any] | None = None
) -> tuple[Any, Any]:
    model = get_model(model_name, use_liger, model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    pad_to_multiple_of: Optional[int] = None,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`list[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        pad_to_multiple_of (`int`, *optional*, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.

    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
    ```python
    >>> import torch

    >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
    tensor([[1, 2, 3],
            [4, 5, 0]])

    >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
    tensor([[[1, 2],
            [3, 4]],
            [[5, 6],
            [0, 0]]])
    ```
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Apply pad_to_multiple_of to the first (sequence) dimension
    if pad_to_multiple_of is not None:
        remainder = output_shape[0] % pad_to_multiple_of
        if remainder != 0:
            output_shape[0] += pad_to_multiple_of - remainder

    # Create an output tensor filled with the padding value
    output = torch.full(
        (len(tensors), *output_shape),
        padding_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def enable_gradient_checkpointing(
    model: PreTrainedModel, gradient_checkpointing_kwargs: Optional[dict[str, Any]]
) -> PreTrainedModel:
    """Enables gradient checkpointing for the model."""
    # Enable gradient checkpointing on the base model for PEFT
    if isinstance(model, PeftModel):
        assert hasattr(model, "base_model")
        base_model = cast(PreTrainedModel, model.base_model)
        base_model.gradient_checkpointing_enable()

    # Enable gradient checkpointing for non-PEFT models
    else:
        model.gradient_checkpointing_enable()

    gradient_checkpointing_kwargs = gradient_checkpointing_kwargs or {}
    use_reentrant = (
        "use_reentrant" not in gradient_checkpointing_kwargs
        or gradient_checkpointing_kwargs["use_reentrant"]
    )

    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, inputs, output):
                if isinstance(output, tuple):
                    return tuple(
                        o.requires_grad_(True) if isinstance(o, torch.Tensor) else o
                        for o in output
                    )
                if isinstance(output, torch.Tensor):
                    return output.requires_grad_(True)
                return output

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def prepare_peft_model(
    model: PreTrainedModel, peft_config: PeftConfig, args: TrainingArguments
) -> PreTrainedModel:
    """Prepares a model for PEFT training."""
    # If the model is already a PeftModel, we need to merge and unload it.
    # Further information here: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
    if args.gradient_checkpointing:
        assert not isinstance(args.gradient_checkpointing_kwargs, str)
        model = enable_gradient_checkpointing(model, args.gradient_checkpointing_kwargs)
    model = cast(PreTrainedModel, get_peft_model(model, peft_config))

    return model


def init_stat_tracker(device: torch.device) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), device=device, dtype=torch.float32)
    return {
        "sum": zero.clone(),
        "count": zero.clone(),
        "min": torch.full((), float("inf"), device=device, dtype=torch.float32),
        "max": torch.full((), float("-inf"), device=device, dtype=torch.float32),
    }


def update_stat_tracker(
    tracker: dict[str, torch.Tensor], summary: dict[str, torch.Tensor]
) -> None:
    tracker["sum"] = tracker["sum"] + summary["sum"]
    tracker["count"] = tracker["count"] + summary["count"]
    tracker["min"] = torch.minimum(tracker["min"], summary["min"])
    tracker["max"] = torch.maximum(tracker["max"], summary["max"])


def finalize_stat_tracker(
    tracker: dict[str, torch.Tensor], accelerator
) -> dict[str, float] | None:
    total_count = accelerator.gather(tracker["count"]).sum()
    if total_count.item() == 0:
        return None

    total_sum = accelerator.gather(tracker["sum"]).sum()
    global_min = accelerator.gather(tracker["min"]).min()
    global_max = accelerator.gather(tracker["max"]).max()

    mean = (total_sum / total_count).float().item()
    min_value = global_min.float().item()
    max_value = global_max.float().item()

    return {"mean": mean, "min": min_value, "max": max_value}


def summarize_values(values: torch.Tensor) -> dict[str, torch.Tensor]:
    if values.numel() == 0:
        return init_stat_tracker(values.device)
    values = values.to(torch.float32)
    return {
        "sum": values.sum(),
        "count": torch.tensor(
            values.numel(), device=values.device, dtype=torch.float32
        ),
        "min": values.min(),
        "max": values.max(),
    }


def gather_logprobs_and_entropy(
    logits: torch.Tensor, target_token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Streamed variant of TRL's ``selective_log_softmax`` helper.

    Materializing ``torch.log_softmax`` for the full ``(batch, seq_len, vocab)``
    tensor increases peak memory versus the original TRL utility.  By
    processing one batch row at a time we match that footprint while also
    collecting entropy statistics for logging.
    """

    per_row_logprobs: list[torch.Tensor] = []
    per_row_entropies: list[torch.Tensor] = []

    for row_logits, row_token_ids in zip(logits, target_token_ids):
        row_logprobs = torch.log_softmax(row_logits, dim=-1)
        per_row_logprobs.append(
            row_logprobs.gather(dim=-1, index=row_token_ids.unsqueeze(-1)).squeeze(-1)
        )
        row_logprobs_detached = row_logprobs.detach()
        per_row_entropies.append(
            -(row_logprobs_detached.exp() * row_logprobs_detached).sum(dim=-1)
        )

    return torch.stack(per_row_logprobs, dim=0), torch.stack(per_row_entropies, dim=0)
