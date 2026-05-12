from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np

RoutedExperts: TypeAlias = list[list[list[int]]]


@dataclass(frozen=True)
class _DecodedRoutedExperts:
    values: RoutedExperts
    shape: tuple[int, int, int]


def _decode_routed_experts(raw: Any) -> _DecodedRoutedExperts:
    if isinstance(raw, dict):
        assert raw["encoding"] == "base64"
        assert raw["dtype"] == "int16"
        shape = tuple(raw["shape"])
        assert len(shape) == 3
        values = np.frombuffer(base64.b64decode(raw["data"]), dtype=np.int16).reshape(shape).tolist()
        return _DecodedRoutedExperts(
            values=cast(RoutedExperts, values),
            shape=cast(tuple[int, int, int], shape),
        )

    routed_experts = cast(RoutedExperts, raw)
    seq_len = len(routed_experts)
    num_layers = len(routed_experts[0])
    topk = len(routed_experts[0][0])

    return _DecodedRoutedExperts(
        values=routed_experts,
        shape=(seq_len, num_layers, topk),
    )


def compose_split_routed_experts(
    *,
    prompt_routed_experts: Any,
    completion_routed_experts: Any,
    prompt_len: int,
    completion_len: int,
) -> RoutedExperts | None:
    """Compose split routed experts and align them to prompt + completion tokens.

    vLLM returns prompt routing at the response level and generated-token routing
    on the choice. The final generated token has no routing decision because it
    was not fed into another forward pass, so this appends one zero entry when a
    completion exists.
    """

    has_prompt = prompt_routed_experts is not None
    has_completion = completion_routed_experts is not None
    if not has_prompt and not has_completion:
        return None

    prompt = _decode_routed_experts(prompt_routed_experts)
    assert prompt.shape[0] == prompt_len

    expected_completion_routed_len = max(completion_len - 1, 0)
    if expected_completion_routed_len == 0:
        completion_values = []
    else:
        completion = _decode_routed_experts(completion_routed_experts)
        assert completion.shape[1:] == prompt.shape[1:]
        assert completion.shape[0] == expected_completion_routed_len
        completion_values = completion.values

    if completion_len == 0:
        return prompt.values

    assert prompt.shape[1] > 0 and prompt.shape[2] > 0
    zero_entry = [[0] * prompt.shape[2] for _ in range(prompt.shape[1])]
    return prompt.values + completion_values + [zero_entry]
