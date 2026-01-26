import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from openai.types import Completion
from openai.types.chat import ChatCompletion

from verifiers.types import ChatMessages, ClientConfig, SamplingArgs, TextMessages, Tool

ClientT = TypeVar("ClientT")
TextResponseT = TypeVar("TextResponseT")
MessageResponseT = TypeVar("MessageResponseT")

NormalizedTextResponse = Completion
NormalizedMessageResponse = ChatCompletion


class Client(ABC, Generic[ClientT, TextResponseT, MessageResponseT]):
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
    async def get_text_response(
        self, prompt: TextMessages, model: str, sampling_args: SamplingArgs
    ) -> TextResponseT: ...

    @abstractmethod
    async def raise_from_text_response(self, response: TextResponseT) -> None: ...

    @abstractmethod
    async def normalize_text_response(
        self, response: TextResponseT
    ) -> NormalizedTextResponse: ...

    @abstractmethod
    async def get_message_response(
        self,
        prompt: ChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> MessageResponseT: ...

    @abstractmethod
    async def raise_from_message_response(self, response: MessageResponseT) -> None: ...

    @abstractmethod
    async def normalize_message_response(
        self, response: MessageResponseT
    ) -> NormalizedMessageResponse: ...

    async def get_text(
        self,
        prompt: TextMessages,
        model: str,
        sampling_args: SamplingArgs,
    ) -> NormalizedTextResponse:
        response = await self.get_text_response(prompt, model, sampling_args)
        await self.raise_from_text_response(response)
        return await self.normalize_text_response(response)

    async def get_message(
        self,
        prompt: ChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> NormalizedMessageResponse:
        response = await self.get_message_response(prompt, model, sampling_args, tools)
        await self.raise_from_message_response(response)
        return await self.normalize_message_response(response)

    @abstractmethod
    async def get_message_with_tokens(
        self,
        prompt: ChatMessages,
        prompt_ids: list[int],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None,
    ) -> NormalizedMessageResponse: ...
