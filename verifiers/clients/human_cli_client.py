import time
import uuid

from rich.console import Console

from verifiers.clients.client import Client
from verifiers.errors import EmptyModelResponseError, ModelError
from verifiers.types import (
    ClientConfig,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    Tool,
)
from verifiers.utils.logging_utils import format_messages


class HumanCLIClient(Client[None, Messages, Response, Tool]):
    """Client that captures assistant responses from a human in the terminal."""

    def __init__(self, sentinel: str = ":wq") -> None:
        self.sentinel = sentinel
        self._console = Console()
        super().__init__(None)

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
        **kwargs,
    ) -> Response:
        raise NotImplementedError(
            "HumanCLIClient.get_native_response is not used. Call get_response()."
        )

    async def raise_from_native_response(self, response: Response) -> None:
        return None

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        return None

    def _read_human_response(self, prompt: Messages) -> str:
        self._console.rule("Human Debug")
        self._console.print(format_messages(prompt))
        self._console.print(
            f"\nEnter assistant response. End input with `{self.sentinel}` on its own line."
        )

        while True:
            lines: list[str] = []
            while True:
                try:
                    line = input()
                except EOFError as e:
                    raise EmptyModelResponseError(
                        "Reached EOF while waiting for human input"
                    ) from e

                if line.strip() == self.sentinel:
                    break
                lines.append(line)

            response_text = "\n".join(lines).strip()
            if response_text:
                return "\n".join(lines)

            self._console.print("Empty response. Please enter a non-empty response.")

    async def get_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Response:
        if tools:
            raise ModelError(
                "Human debug mode is text-only and does not support tool calls."
            )

        content = self._read_human_response(prompt)
        return Response(
            id=f"human-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            usage=None,
            message=ResponseMessage(
                content=content,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )
