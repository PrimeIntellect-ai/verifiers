"""Provider-agnostic wire types: messages, tools, model responses.

These are the only types that cross the boundary between verifiers and a model
provider. They are pydantic models — never dicts with normalizers. The single
place raw provider dicts enter is the client implementation, which validates
them into these models explicitly.

Also home to `EnvId` — the plain-string id of a taskset / harness / environment — and
the helpers that resolve it (`env_name`, `env_module`, `ensure_installed`).
"""

import logging
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, AliasChoices, BaseModel, ConfigDict, Field

from verifiers.utils.install_utils import (
    check_hub_env_installed,
    install_from_hub,
    is_hub_env,
    normalize_package_name,
    parse_env_id,
)

logger = logging.getLogger(__name__)


class StrictBaseModel(BaseModel):
    """A pydantic base that rejects unknown fields. Use for all closed data types."""

    model_config = ConfigDict(extra="forbid")


# --- messages -----------------------------------------------------------------


class SystemMessage(StrictBaseModel):
    """A system instruction message."""

    role: Literal["system"] = "system"
    content: str


class UserMessage(StrictBaseModel):
    """A user message."""

    role: Literal["user"] = "user"
    content: str


class ToolCall(StrictBaseModel):
    """A tool/function call requested by the model."""

    id: str
    name: str
    arguments: str
    """Raw JSON string of arguments, exactly as the model emitted it."""


class AssistantMessage(StrictBaseModel):
    """An assistant message, optionally carrying reasoning and tool calls."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolMessage(StrictBaseModel):
    """The result of a tool call, keyed to the originating call id."""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
Messages = list[Message]


# --- tools --------------------------------------------------------------------


class Tool(StrictBaseModel):
    """A function tool advertised to the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None


# --- responses ----------------------------------------------------------------


FinishReason = Literal["stop", "length", "tool_calls"] | None


class Usage(StrictBaseModel):
    """Token usage for one model response (total_tokens is derived)."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class TurnTokens(StrictBaseModel):
    """Token ids + sampling logprobs for one response, for training. Populated by the
    renderer client (client-side tokenization) or the chat client (parsed from vLLM's
    token ids); None when the provider returns neither."""

    prompt_ids: list[int] = Field(default_factory=list)
    completion_ids: list[int] = Field(default_factory=list)
    completion_logprobs: list[float] = Field(default_factory=list)


class Response(StrictBaseModel):
    """One model completion, provider-agnostic."""

    id: str
    created: int
    model: str
    message: AssistantMessage
    finish_reason: FinishReason
    usage: Usage | None = None
    tokens: TurnTokens | None = None
    """Token ids + logprobs for training (renderer client, or chat client via vLLM)."""


# --- sampling -----------------------------------------------------------------


class SamplingConfig(BaseModel):
    """Typed sampling knobs; provider-specific keys pass through (extra='allow')."""

    model_config = ConfigDict(extra="allow")
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_completion_tokens")
    )


# --- env id -------------------------------------------------------------------
#
# A taskset, harness, or v0 environment is selected by an id in one of three forms:
#   - `name`              a local package (already importable / pip-installed)
#   - `org/name`          the env hub, latest version
#   - `org/name@version`  the env hub, a pinned version


def _validate_env_id(env_id: str) -> str:
    """Validate the id's shape — a hub id must be a well-formed `org/name[@version]`; a
    local id is any module name. Returns it unchanged (the value stays a plain `str`)."""
    if is_hub_env(env_id):
        parse_env_id(env_id)  # raises ValueError on a malformed org/name[@version]
    return env_id


EnvId = Annotated[str, AfterValidator(_validate_env_id)]
"""A taskset / harness / environment id — `name`, `org/name`, or `org/name@version`.
A plain validated `str`; parse it with `env_name` / `env_module`."""


def env_name(env_id: str) -> str:
    """The bare name — org and version stripped (`org/gsm8k@1.0` -> `gsm8k`). Used for
    logging, display, and output paths."""
    return parse_env_id(env_id)[1] if is_hub_env(env_id) else env_id


def env_module(env_id: str) -> str:
    """The importable module name — `env_name` normalized (hyphens -> underscores)."""
    return normalize_package_name(env_name(env_id))


def ensure_installed(env_id: str) -> str:
    """Make `env_id` importable and return its module name.

    For a hub id (`org/name[@version]`) that isn't installed, install it from the
    Environments Hub — latest, or the pinned version — the same path `prime env install`
    uses. A local id is assumed already importable."""
    if is_hub_env(env_id) and not check_hub_env_installed(env_id):
        logger.info("installing %s from the environments hub", env_id)
        if not install_from_hub(env_id):
            raise ModuleNotFoundError(
                f"could not install {env_id!r} from the environments hub"
            )
    return env_module(env_id)
