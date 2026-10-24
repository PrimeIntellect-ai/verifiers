"""An ACP turn the agent cancelled because its model call failed is that provider failure."""

from types import SimpleNamespace

import pytest

from verifiers.v1.acp import _raise_failed_model_call
from verifiers.v1.errors import ProviderError
from verifiers.v1.trace import Error, ModelCall


def trace_with(*calls: ModelCall) -> SimpleNamespace:
    return SimpleNamespace(calls=list(calls))


def test_a_turn_cancelled_after_a_failed_model_call_raises_that_provider_error():
    """OpenCode ends the turn `cancelled` once the interception gives the 429 up: the turn
    raises it as a transient ProviderError (the rollout retries), not a graded ending."""
    failed = ModelCall(
        error=Error(
            type="ProviderError", message="upstream 429: too many", status_code=429
        )
    )
    with pytest.raises(ProviderError) as raised:
        _raise_failed_model_call(trace_with(ModelCall(node=0), failed), calls_before=0)
    assert raised.value.status_code == 429 and "upstream 429" in str(raised.value)


def test_a_cancel_after_a_good_call_or_from_an_earlier_turn_is_left_as_it_is():
    """A cancel whose last call of this turn succeeded (the rollout stopped the agent), or
    whose failed call belongs to an earlier turn, keeps its `cancelled` ending."""
    failed = ModelCall(
        error=Error(type="ProviderError", message="upstream 429", status_code=429)
    )
    _raise_failed_model_call(trace_with(failed, ModelCall(node=1)), calls_before=0)
    _raise_failed_model_call(trace_with(failed), calls_before=1)
    _raise_failed_model_call(
        trace_with(ModelCall(error=Error(type="HarnessError", message="x"))),
        calls_before=0,
    )
