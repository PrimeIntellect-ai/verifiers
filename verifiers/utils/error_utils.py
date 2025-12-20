from typing import Callable, cast

import verifiers as vf


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

    def __repr__(self) -> str:
        return " -> ".join([type(e).__name__ for e in self.chain])

    def __str__(self) -> str:
        return ", caused by ".join([repr(e) for e in self.chain])
