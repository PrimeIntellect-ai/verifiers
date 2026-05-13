from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from verifiers.utils.usage_utils import extract_usage_tokens

from ..state import State


def record_response_usage(state: State, response: object) -> None:
    if getattr(response, "usage", None) is None and not (
        isinstance(response, Mapping)
        and cast(Mapping[str, object], response).get("usage") is not None
    ):
        return
    input_tokens, output_tokens = extract_usage_tokens(response)
    usage = state.setdefault("token_usage", {"input_tokens": 0.0, "output_tokens": 0.0})
    if not isinstance(usage, dict):
        raise TypeError("state.token_usage must be a mapping.")
    usage["input_tokens"] = float(usage.get("input_tokens", 0.0)) + float(input_tokens)
    usage["output_tokens"] = float(usage.get("output_tokens", 0.0)) + float(
        output_tokens
    )
    state["usage"] = usage
