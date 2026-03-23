"""NullClient for environments that don't require LLM inference."""

from __future__ import annotations

import logging
from typing import Any

from verifiers.clients.client import Client
from verifiers.types import ClientConfig, Messages, Response, SamplingArgs, Tool


class NullClient(Client):
    """A no-op client for environments that don't use LLM inference.

    Satisfies the Client ABC so it can pass through resolve_client() unchanged.
    All inference methods raise NotImplementedError — they should never be called.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._config = None
        self._client = None

    def setup_client(self, config: ClientConfig) -> Any:
        raise NotImplementedError("NullClient does not support inference")

    async def to_native_tool(self, tool: Tool) -> Any:
        raise NotImplementedError("NullClient does not support inference")

    async def to_native_prompt(self, messages: Messages) -> tuple[Any, dict]:
        raise NotImplementedError("NullClient does not support inference")

    async def get_native_response(
        self,
        prompt: Any,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("NullClient does not support inference")

    async def raise_from_native_response(self, response: Any) -> None:
        raise NotImplementedError("NullClient does not support inference")

    async def from_native_response(self, response: Any) -> Response:
        raise NotImplementedError("NullClient does not support inference")

    async def close(self) -> None:
        pass
