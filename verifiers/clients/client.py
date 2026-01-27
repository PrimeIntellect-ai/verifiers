import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from verifiers.types import (
    ChatMessages,
    ChatResponse,
    ClientConfig,
    SamplingArgs,
    TextMessages,
    TextResponse,
    Tool,
)

ClientT = TypeVar("ClientT")
TextMessagesT = TypeVar("TextMessagesT")
ChatMessagesT = TypeVar("ChatMessagesT")
TextResponseT = TypeVar("TextResponseT")
ChatResponseT = TypeVar("ChatResponseT")


class Client(
    ABC, Generic[ClientT, TextResponseT, ChatResponseT, TextMessagesT, ChatMessagesT]
):
    def __init__(self, config: ClientConfig):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config
        self._client = self.setup_client(config)

    @property
    def client(self) -> ClientT:
        return self._client

    @abstractmethod
    def setup_client(self, config: ClientConfig) -> ClientT: ...

    @abstractmethod
    def to_native_text_messages(self, messages: TextMessages) -> TextMessagesT: ...

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
    def to_native_chat_messages(self, messages: ChatMessages) -> ChatMessagesT: ...

    @abstractmethod
    async def get_native_chat_response(
        self,
        prompt: ChatMessagesT,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponseT: ...

    @abstractmethod
    async def raise_from_native_chat_response(
        self, response: ChatResponseT
    ) -> None: ...

    @abstractmethod
    def from_native_chat_response(self, response: ChatResponseT) -> ChatResponse: ...

    async def get_text_response(
        self,
        prompt: TextMessages,
        model: str,
        sampling_args: SamplingArgs,
    ) -> TextResponse:
        native_prompt = self.to_native_text_messages(prompt)
        native_response = await self.get_native_text_response(
            native_prompt, model, sampling_args
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
        native_prompt = self.to_native_chat_messages(prompt)
        native_response = await self.get_native_chat_response(
            native_prompt, model, sampling_args, tools
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
