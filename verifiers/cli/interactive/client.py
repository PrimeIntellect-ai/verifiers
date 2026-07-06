import asyncio
import logging
import time
import uuid

from verifiers.clients import Client
from verifiers.cli.interactive.app import InteractiveRolloutApp
from verifiers.types import (
    ClientConfig,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    Tool,
)


class HumanClient(Client[None, Messages, Response, Tool]):
    """A local client that routes model requests to an interactive TUI."""

    def __init__(
        self,
        *,
        show_prompt: bool = True,
        show_tools: bool = True,
        max_content_chars: int = 20000,
        headless: bool = False,
        answer: str | None = None,
        allow_remote_images: bool = False,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._client = None
        self._config = None
        self._app = InteractiveRolloutApp(
            show_prompt=show_prompt,
            show_tools=show_tools,
            max_content_chars=max_content_chars,
            answer=answer,
            allow_remote_images=allow_remote_images,
        )
        self._task: asyncio.Task[None] | None = None
        self._headless = headless

    def setup_client(self, config: ClientConfig) -> None:
        return None

    async def to_native_tool(self, tool: Tool) -> Tool:
        return tool

    async def to_native_prompt(self, messages: Messages) -> tuple[Messages, dict]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs: object,
    ) -> Response:
        _ = prompt, model, sampling_args, tools, kwargs
        raise NotImplementedError("HumanClient.get_response handles interaction.")

    async def raise_from_native_response(self, response: Response) -> None:
        _ = response

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        if self._task is None:
            return
        self._app.call_later(self._app.exit)
        try:
            await self._task
        finally:
            self._task = None

    async def start(self) -> None:
        """Launch the TUI. Called eagerly by vf-play so the window appears
        while the environment is still setting up; get_response falls back
        to launching lazily for direct client use."""
        if self._task is None:
            self._task = asyncio.create_task(
                self._app.run_async(headless=self._headless)
            )
        await self._app.ready.wait()

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs: object,
    ) -> Response:
        _ = sampling_args, kwargs
        await self.start()
        response = await self._app.ask(prompt, list(tools or []))
        return Response(
            id=f"human-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            usage=None,
            message=ResponseMessage(
                role="assistant",
                content=response.content,
                reasoning_content=response.reasoning,
                finish_reason="tool_calls" if response.tool_calls else "stop",
                is_truncated=False,
                tool_calls=response.tool_calls,
            ),
        )
