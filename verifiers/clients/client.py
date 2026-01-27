import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from verifiers.types import (
    ChatMessage,
    ChatMessages,
    ChatResponse,
    ClientConfig,
    SamplingArgs,
    TextMessage,
    TextMessages,
    TextResponse,
    Tool,
)

ClientT = TypeVar("ClientT")
TextMessageT = TypeVar("TextMessageT")
ChatMessageT = TypeVar("ChatMessageT")
TextResponseT = TypeVar("TextResponseT")
ChatResponseT = TypeVar("ChatResponseT")


class Client(
    ABC, Generic[ClientT, TextResponseT, ChatResponseT, TextMessageT, ChatMessageT]
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
    def from_text_message(self, message: TextMessage) -> TextMessageT: ...

    @abstractmethod
    def to_text_message(self, response: TextResponseT) -> TextResponse: ...

    @abstractmethod
    async def get_text_response(
        self, prompt: TextMessageT, model: str, sampling_args: SamplingArgs
    ) -> TextResponseT: ...

    @abstractmethod
    async def raise_from_text_response(self, response: TextResponseT) -> None: ...

    @abstractmethod
    def from_chat_message(self, message: ChatMessage) -> ChatMessageT: ...

    @abstractmethod
    def to_chat_message(self, response: ChatResponseT) -> ChatResponse: ...

    @abstractmethod
    async def get_chat_response(
        self,
        prompt: list[ChatMessageT],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponseT: ...

    @abstractmethod
    async def raise_from_chat_response(self, response: ChatResponseT) -> None: ...

    async def get_text(
        self,
        prompt: TextMessages,
        model: str,
        sampling_args: SamplingArgs,
    ) -> TextResponse:
        normalized_prompt = self.from_text_message(prompt)
        response = await self.get_text_response(normalized_prompt, model, sampling_args)
        await self.raise_from_text_response(response)
        normalized_response = self.to_text_message(response)
        return normalized_response

    async def get_message(
        self,
        prompt: ChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponse:
        normalized_prompt = [self.from_chat_message(p) for p in prompt]
        response = await self.get_chat_response(
            normalized_prompt, model, sampling_args, tools
        )
        await self.raise_from_chat_response(response)
        normalized_response = self.to_chat_message(response)
        return normalized_response

    @abstractmethod
    async def get_chat_with_tokens(
        self,
        prompt: ChatMessages,
        prompt_ids: list[int],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> ChatResponse: ...
