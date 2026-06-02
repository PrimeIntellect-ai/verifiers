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
    # Context ("final") token metrics, accumulated at write time from the live
    # Response. v1 serializes trajectory responses to plain dicts, so they can't
    # be recomputed from the trajectory afterward (the isinstance(Response) gate
    # in compute_context_token_metrics fails). Mirror that helper's formula for a
    # linear rollout: final_output is the running sum of completions; final_input
    # is the latest step's full context minus that sum.
    usage["final_output_tokens"] = float(usage.get("final_output_tokens", 0.0)) + float(
        output_tokens
    )
    last_step_total = float(input_tokens) + float(output_tokens)
    usage["final_input_tokens"] = max(
        0.0, last_step_total - usage["final_output_tokens"]
    )
    state["usage"] = usage
