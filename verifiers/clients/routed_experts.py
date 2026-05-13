from __future__ import annotations

import base64
import io
from typing import cast

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
