"""Helper for the base ``TaskSet.filter_fn`` kwarg.

``filter_fn`` accepts a Python expression string (typically a lambda) that
evaluates to a ``Callable[[dict], bool]`` and is used to filter the
taskset's dataset at construction time, after each concrete taskset's
``_process_example`` mapping. Predicates therefore see the
post-processed row shape (e.g. ``{"question", "info", "answer", ...}``).

Security note: the string is passed to ``eval()`` with a restricted
``__builtins__`` dict, but it is still ``eval()`` of user input. It is
intended for local ``vf-eval`` invocations, not for running untrusted
inputs.
"""

import re
from typing import Callable

# Small, safe-ish set of helpers commonly useful in filter expressions.
_ALLOWED_GLOBALS: dict = {
    "__builtins__": {},
    "re": re,
    "len": len,
    "all": all,
    "any": any,
    "sum": sum,
    "min": min,
    "max": max,
    "sorted": sorted,
    "set": set,
    "frozenset": frozenset,
}


def _resolve_filter_fn(expr: str) -> Callable[[dict], bool]:
    """Compile a filter expression string into a callable predicate.

    The expression is ``eval()``'d with a restricted globals dict (see
    ``_ALLOWED_GLOBALS``) and must evaluate to a callable taking a single
    dict argument and returning truthy/falsy. Typical usage::

        filter_fn="lambda x: x['info']['language'] == 'python' "
                 "and len(x['info'].get('FAIL_TO_PASS') or []) > 3"
    """
    if not isinstance(expr, str):
        raise TypeError(
            f"filter_fn must be a string Python expression, got {type(expr).__name__}"
        )
    try:
        fn = eval(expr, _ALLOWED_GLOBALS, {})  # noqa: S307 - documented, restricted
    except Exception as e:
        raise ValueError(f"Failed to evaluate filter_fn expression: {e!r}") from e
    if not callable(fn):
        raise TypeError(
            f"filter_fn expression must evaluate to a callable, got {type(fn).__name__}"
        )
    return fn
