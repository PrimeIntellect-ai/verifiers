import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from anthropic import (
    AuthenticationError as AnthropicAuthenticationError,
)
from anthropic import (
    PermissionDeniedError as AnthropicPermissionDeniedError,
)
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import PermissionDeniedError as OpenAIPermissionDeniedError

from verifiers.errors import Error, ModelError
from verifiers.types import (
    ClientConfig,
    Messages,
    Response,
    SamplingArgs,
    Tool,
)

if TYPE_CHECKING:
    pass

ClientT = TypeVar("ClientT")
MessagesT = TypeVar("MessagesT")
ResponseT = TypeVar("ResponseT")
ToolT = TypeVar("ToolT")

AUTH_ERRORS: tuple[type[Exception], ...] = (
    OpenAIAuthenticationError,
    OpenAIPermissionDeniedError,
    AnthropicAuthenticationError,
    AnthropicPermissionDeniedError,
)


class Client(ABC, Generic[ClientT, MessagesT, ResponseT, ToolT]):
    def __init__(self, client_or_config: ClientT | ClientConfig) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if isinstance(client_or_config, ClientConfig):
            self._client = self.setup_client(client_or_config)
        else:
            self._client = client_or_config

    @property
    def client(self) -> ClientT:
        return self._client

    @abstractmethod
    def setup_client(self, config: ClientConfig) -> ClientT: ...

    @abstractmethod
    async def to_native_tool(self, tool: Tool) -> ToolT:
        """Converts vf.Tool to the native tool format for this client."""
        ...

    @abstractmethod
    async def to_native_prompt(self, messages: Messages) -> tuple[MessagesT, dict]:
        """Converts vf.Messages to the native prompt format for this client + optional kwargs that are passed to the get_native_response"""
        ...

    @abstractmethod
    async def get_native_response(
        self,
        prompt: MessagesT,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[ToolT] | None = None,
        **kwargs,
    ) -> ResponseT:
        """Get the native response from the client."""
        ...

    @abstractmethod
    async def raise_from_native_response(self, response: ResponseT) -> None:
        """Raise vf.ModelError exceptions if the native response is not valid."""
        ...

    @abstractmethod
    async def from_native_response(self, response: ResponseT) -> Response:
        """Convert the native response to a vf.Response."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the underlying client, if applicable."""
        ...

    async def to_native_tools(self, tools: list[Tool] | None) -> list[ToolT] | None:
        """Converts a list of vf.Tools to the native tool format for this client."""
        if tools is None:
            return None
        native_tools: list[ToolT] = []
        for tool in tools:
            native_tools.append(await self.to_native_tool(tool))
        return native_tools

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Response:
        """Get the response from the client."""
        try:
            native_prompt, extra_kwargs = await self.to_native_prompt(prompt)
            native_tools = await self.to_native_tools(tools)
            native_response = await self.get_native_response(
                native_prompt,
                model,
                sampling_args,
                native_tools,
                **extra_kwargs,
                **kwargs,
            )
            await self.raise_from_native_response(native_response)
            response = await self.from_native_response(native_response)
            return response
        except Error:
            raise
        except AUTH_ERRORS:
            raise
        except Exception as e:
            raise ModelError from e
