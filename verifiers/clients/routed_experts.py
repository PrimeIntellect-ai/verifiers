import base64
from io import BytesIO
from typing import Any, cast

import numpy as np


def parse_routed_experts(raw: Any) -> str | None:
    if raw is None:
        return None
    return cast(str, raw)


def truncate_routed_experts(routed_experts: str | None, seq_len: int) -> str | None:
    if routed_experts is None:
        return None

    array = np.load(BytesIO(base64.b64decode(routed_experts)), allow_pickle=False)
    assert array.ndim == 3
    assert 0 <= seq_len <= array.shape[0]

    buffer = BytesIO()
    np.save(buffer, np.ascontiguousarray(array[:seq_len]), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
