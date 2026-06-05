from typing import Callable, TypeGuard, cast

import verifiers as vf
from verifiers.types import ErrorData


def get_error_chain(
    error: BaseException | None, parent_type: type[BaseException] | None = None
) -> list[BaseException]:
    """Get a causal error chain. If parent_type is specified, the chain will be truncated at the first error that is not a child of parent_type."""
    error_chain = []
    while error is not None:
        if parent_type is not None and not isinstance(error, parent_type):
            break
        error_chain.append(error)
        error = error.__cause__
    return error_chain


def get_vf_error_chain(error: BaseException) -> list[vf.Error]:
    """Get an error chain containing only vf errors."""
    return cast(list[vf.Error], get_error_chain(error, parent_type=vf.Error))


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

    def __contains__(self, error_cls: type[BaseException]) -> bool:
        return any(issubclass(type(e), error_cls) for e in self.chain)

    def __str__(self) -> str:
        return " -> ".join([type(e).__name__ for e in self.chain])

    def __repr__(self) -> str:
        return " -> ".join([repr(e) for e in self.chain])


def error_data(error: BaseException) -> ErrorData:
    error_chain = ErrorChain(error)
    return ErrorData(
        error=type(error).__name__,
        message=str(error) or type(error).__name__,
        error_chain_repr=repr(error_chain),
        error_chain_str=str(error_chain),
    )


def is_error_data(value: object) -> TypeGuard[ErrorData]:
    expected = {"error", "message", "error_chain_repr", "error_chain_str"}
    if not isinstance(value, dict):
        return False
    match value:
        case {
            "error": str(),
            "message": str(),
            "error_chain_repr": str(),
            "error_chain_str": str(),
        }:
            return set(value) == expected
    return False


def validate_error_data(value: object) -> ErrorData:
    if not is_error_data(value):
        raise TypeError(
            "ErrorData must contain string error, message, error_chain_repr, "
            "and error_chain_str fields."
        )
    return value


def error_data_to_exception(
    error: ErrorData,
    error_types: tuple[type[Exception], ...],
) -> Exception | None:
    chain = error["error_chain_str"] or error["error"]
    detail = error["message"] or error["error_chain_repr"] or error["error"]
    for error_type in error_types:
        if error_type.__name__ == error["error"]:
            return error_type(detail)
    for error_type in error_types:
        if error_type.__name__ in chain:
            return error_type(detail)
    return None


def error_from_data(error: ErrorData) -> vf.Error:
    exception = error_data_to_exception(error, vf_error_types())
    if isinstance(exception, vf.Error):
        return exception
    detail = error["message"] or error["error_chain_repr"] or error["error"]
    return vf.Error(detail)


def vf_error_types() -> tuple[type[vf.Error], ...]:
    return (
        vf.BrowserSandboxError,
        vf.SandboxError,
        vf.TunnelError,
        vf.InfraError,
        vf.EmptyModelResponseError,
        vf.InvalidModelResponseError,
        vf.ModelError,
        vf.ToolParseError,
        vf.ToolCallError,
        vf.ToolError,
        vf.OverlongPromptError,
        vf.Error,
    )
