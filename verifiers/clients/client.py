import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from verifiers.types import (
    ChatMessages,
    ChatResponse,
    ClientConfig,
    SamplingArgs,
    TextMessages,
    TextResponse,
    Tool,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    from verifiers.clients.ant_client import AntClient
    from verifiers.clients.oai_client import OAIClient

ClientT = TypeVar("ClientT")
TextMessagesT = TypeVar("TextMessagesT")
ChatMessagesT = TypeVar("ChatMessagesT")
TextResponseT = TypeVar("TextResponseT")
ChatResponseT = TypeVar("ChatResponseT")
ToolT = TypeVar("ToolT")


class Client(
    ABC,
    Generic[ClientT, TextResponseT, ChatResponseT, TextMessagesT, ChatMessagesT, ToolT],
):
    def __init__(self, client_or_config: ClientT | ClientConfig) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        if isinstance(client_or_config, ClientConfig):
            self._client = self.setup_client(client_or_config)
        else:
            self._client = client_or_config
        self.interleaved_thinking = True

    @overload
    @classmethod
    def from_client(cls, client: "AsyncOpenAI") -> "OAIClient": ...

    @overload
    @classmethod
    def from_client(cls, client: "AsyncAnthropic") -> "AntClient": ...

    @classmethod
    def from_client(
        cls, client: "AsyncOpenAI | AsyncAnthropic"
    ) -> "OAIClient | AntClient":
        from anthropic import AsyncAnthropic
        from openai import AsyncOpenAI

        from verifiers.clients.ant_client import AntClient
        from verifiers.clients.oai_client import OAIClient

        if isinstance(client, AsyncOpenAI):
            return OAIClient(client)
        elif isinstance(client, AsyncAnthropic):
            return AntClient(client)
        else:
            raise TypeError(f"Unsupported client type: {type(client)}")

    @property
    def client(self) -> ClientT:
        return self._client

    def set_interleaved_thinking(self, interleaved_thinking: bool) -> None:
        self.interleaved_thinking = interleaved_thinking

    @staticmethod
    @abstractmethod
    def setup_client(config: ClientConfig) -> ClientT: ...

    @abstractmethod
    def to_native_text_prompt(
        self, messages: TextMessages
    ) -> tuple[TextMessagesT, dict]: ...

    @abstractmethod
    async def get_native_text_response(
        self, prompt: TextMessagesT, model: str, sampling_args: SamplingArgs
    ) -> TextResponseT: ...

    @abstractmethod
    async def raise_from_native_text_response(
        self, response: TextResponseT
    ) -> None: ...

    @abstractmethod
    def from_native_text_response(self, response: TextResponseT) -> TextResponse: ...

    @abstractmethod
    def to_native_chat_prompt(
        self, messages: ChatMessages
    ) -> tuple[ChatMessagesT, dict]: ...

    @abstractmethod
    async def get_native_chat_response(
        self,
        prompt: ChatMessagesT,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[ToolT] | None,
    ) -> ChatResponseT: ...

    @abstractmethod
    async def raise_from_native_chat_response(
        self, response: ChatResponseT
    ) -> None: ...

    @abstractmethod
    def from_native_chat_response(self, response: ChatResponseT) -> ChatResponse: ...

    @abstractmethod
    def to_native_tool(self, tool: Tool) -> ToolT:
        """Convert a unified Tool to the native tool format for this client."""
        ...

    def to_native_tools(self, tools: list[Tool] | None) -> list[Any] | None:
        """Convert a list of unified Tools to native tool format for this client."""
        if tools is None:
            return None
        return [self.to_native_tool(tool) for tool in tools]

    async def get_text_response(
        self,
        prompt: TextMessages,
        model: str,
        sampling_args: SamplingArgs,
    ) -> TextResponse:
        native_prompt, kwargs = self.to_native_text_prompt(prompt)
        native_response = await self.get_native_text_response(
            native_prompt, model, sampling_args, **kwargs
        )
        await self.raise_from_native_text_response(native_response)
        response = self.from_native_text_response(native_response)
        return response

    async def get_chat_response(
        self,
        prompt: ChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponse:
        native_prompt, kwargs = self.to_native_chat_prompt(prompt)
        native_tools = self.to_native_tools(tools)
        native_response = await self.get_native_chat_response(
            native_prompt, model, sampling_args, native_tools, **kwargs
        )
        await self.raise_from_native_chat_response(native_response)
        response = self.from_native_chat_response(native_response)
        return response

    @abstractmethod
    async def get_chat_response_with_tokens(
        self,
        prompt: ChatMessages,
        prompt_ids: list[int],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponse: ...
