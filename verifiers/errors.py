from typing import Callable

from verifiers.utils.error_utils import get_error_chain


class ErrorChain:
    """Helper class for error chains."""

    def __init__(
        self,
        error: BaseException,
        build_error_chain: Callable[
            [BaseException], list[BaseException]
        ] = get_error_chain,
    ):
        self.root_error = error
        self.chain = build_error_chain(error)

    def __hash__(self) -> int:
        return hash(tuple(type(e).__name__ for e in self.chain))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErrorChain):
            return NotImplemented
        return tuple(type(e).__name__ for e in self.chain) == tuple(
            type(e).__name__ for e in other.chain
        )

    def __contains__(self, error: BaseException) -> bool:
        return any(isinstance(error, e) for e in self.chain)

    def __repr__(self) -> str:
        return " -> ".join([type(e).__name__ for e in self.chain])

    def __str__(self) -> str:
        return ", caused by ".join([repr(e) for e in self.chain])


class Error(Exception):
    """Base class for all errors."""


class ModelError(Error):
    """Used to catch errors while interacting with the model."""

    pass


class EmptyModelResponseError(ModelError):
    """Used to catch empty or invalid model responses (e.g. response.choices is None)."""

    pass


class OverlongPromptError(Error):
    """Used to catch overlong prompt errors (e.g. prompt + requested number of tokens exceeds model context length)"""

    pass


class ToolError(Error):
    """Parent class for all tool errors."""

    pass


class ToolParseError(ToolError):
    """Used to catch errors while parsing tool calls."""

    pass


class ToolCallError(ToolError):
    """Used to catch errors while calling tools."""

    pass


class InfraError(Error):
    """Used to catch errors while interacting with infrastructure."""

    pass


class SandboxError(InfraError):
    """Used to catch errors while interacting with sandboxes."""

    pass
