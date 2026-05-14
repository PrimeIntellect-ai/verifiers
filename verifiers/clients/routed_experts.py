from __future__ import annotations

import base64
from typing import Any, Mapping, cast

from verifiers.types import RoutedExperts

INT16_BYTES = 2


def _shape_numel(shape: list[int]) -> int:
    seq_len, num_layers, topk = shape
    return seq_len * num_layers * topk


def _token_stride(shape: list[int]) -> int:
    return shape[1] * shape[2] * INT16_BYTES


def _validate_routed_experts(payload: RoutedExperts) -> RoutedExperts:
    assert payload.dtype == "int16"
    assert len(payload.shape) == 3
    assert len(payload.data) == _shape_numel(payload.shape) * INT16_BYTES
    return payload


def _decode_routed_experts(raw: Any) -> RoutedExperts:
    if isinstance(raw, RoutedExperts):
        return _validate_routed_experts(raw)

    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="python")

    raw = cast(Mapping[str, Any], raw)
    assert raw["encoding"] == "base64"
    assert raw["dtype"] == "int16"
    shape = [int(dim) for dim in raw["shape"]]
    data = base64.b64decode(raw["data"])
    return _validate_routed_experts(RoutedExperts(shape=shape, data=data))


def slice_routed_experts(payload: RoutedExperts, end: int) -> RoutedExperts:
    payload = _validate_routed_experts(payload)
    assert 0 <= end <= payload.shape[0]
    stride = _token_stride(payload.shape)
    return RoutedExperts(
        shape=[end, payload.shape[1], payload.shape[2]],
        data=payload.data[: end * stride],
    )


def compose_split_routed_experts(
    *,
    prompt_routed_experts: Any,
    completion_routed_experts: Any,
    prompt_len: int,
    completion_len: int,
) -> RoutedExperts | None:
    """Compose split prompt/completion routing into compact int16 bytes."""

    if prompt_routed_experts is None and completion_routed_experts is None:
        return None

    prompt = _decode_routed_experts(prompt_routed_experts)
    assert prompt.shape[0] == prompt_len

    expected_completion_routed_len = max(completion_len - 1, 0)
    if expected_completion_routed_len == 0:
        completion_data = b""
    else:
        completion = _decode_routed_experts(completion_routed_experts)
        assert completion.shape[1:] == prompt.shape[1:]
        assert completion.shape[0] == expected_completion_routed_len
        completion_data = completion.data

    if completion_len == 0:
        return prompt

    stride = _token_stride(prompt.shape)
    return _validate_routed_experts(
        RoutedExperts(
            shape=[prompt_len + completion_len, prompt.shape[1], prompt.shape[2]],
            data=prompt.data + completion_data + (b"\0" * stride),
        )
    )
