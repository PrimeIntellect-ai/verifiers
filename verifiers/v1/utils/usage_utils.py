from typing import cast

from verifiers.types import Response
from verifiers.utils.usage_utils import usage_tokens

from ..state import State


def record_response_usage(state: State, response: Response) -> None:
    if response.usage is None:
        return
    input_tokens, output_tokens = usage_tokens(response.usage)
    usage = state.setdefault("token_usage", {"input_tokens": 0.0, "output_tokens": 0.0})
    if not isinstance(usage, dict):
        raise TypeError("state.token_usage must be a mapping.")
    usage = cast(dict[str, float], usage)
    usage["input_tokens"] = float(usage.get("input_tokens", 0.0)) + float(input_tokens)
    usage["output_tokens"] = float(usage.get("output_tokens", 0.0)) + float(
        output_tokens
    )
    state["usage"] = usage
