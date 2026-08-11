"""Response-hook types."""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from verifiers.v1.terminate import Terminate
from verifiers.v1.types import AssistantMessage, ToolMessage

ResponseResult = str | AssistantMessage | ToolMessage | Terminate | None
ResponseHandler = Callable[..., ResponseResult | Awaitable[ResponseResult]]


class ResponseRewrite(BaseModel):
    """One response rewrite stored on a trace."""

    model_config = ConfigDict(frozen=True)

    handler: str
    target: str = ""
