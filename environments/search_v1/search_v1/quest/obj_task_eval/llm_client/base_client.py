"""Minimal LLM client protocol used by the vendored QUEST evaluator.

The original QUEST package ships several concrete provider clients. In
verifiers, the rubric owns provider construction and passes an object exposing
``async_response`` into the generated QUEST eval scripts, so only the protocol
is needed here.
"""

from typing import Any, Protocol


class LLMClient(Protocol):
    provider: str

    async def async_response(self, **kwargs: Any) -> Any: ...
