"""Request-hook types."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from verifiers.v1.terminate import Terminate
from verifiers.v1.types import ToolMessage, UserMessage

RequestResult = str | UserMessage | ToolMessage | Terminate | None
RequestHandler = Callable[..., RequestResult | Awaitable[RequestResult]]


class RequestRewrite(BaseModel):
    """One request rewrite stored on a trace."""

    model_config = ConfigDict(frozen=True)

    handler: str
    target: str = ""
