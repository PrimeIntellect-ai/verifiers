from __future__ import annotations

import base64
import io
from typing import Any, cast

import numpy as np

RoutedExpertsList = list[list[list[int]]]


def _load_routed_experts(payload: str) -> np.ndarray:
    routed_experts = np.load(
        io.BytesIO(base64.b64decode(payload)),
        allow_pickle=False,
    )
    assert routed_experts.ndim == 3
    return routed_experts.astype(np.int32, copy=False)


def decode_routed_experts(payload: str | None) -> RoutedExpertsList | None:
    if payload is None:
        return None

    return cast(RoutedExpertsList, _load_routed_experts(payload).tolist())


def compose_routed_experts(
    *,
    prompt_routed_experts: Any,
    completion_routed_experts: Any,
    prompt_len: int,
    completion_len: int,
) -> RoutedExpertsList | None:
    if prompt_routed_experts is None:
        return decode_routed_experts(cast(str | None, completion_routed_experts))

    prompt = _load_routed_experts(cast(str, prompt_routed_experts))
    assert prompt.shape[0] == prompt_len

    expected_completion_len = max(completion_len - 1, 0)
    if expected_completion_len == 0:
        completion = np.empty((0, prompt.shape[1], prompt.shape[2]), dtype=np.int32)
    else:
        completion = _load_routed_experts(cast(str, completion_routed_experts))
        assert completion.shape[0] == expected_completion_len
        assert completion.shape[1:] == prompt.shape[1:]

    return cast(RoutedExpertsList, np.concatenate((prompt, completion), axis=0).tolist())
