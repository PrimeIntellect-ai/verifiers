"""Browser mode implementations."""

import warnings

from .base import BrowserMode
from .dom_mode import DOMMode
from .cua_mode import CUAMode, SANDBOX_AVAILABLE


def CUASandboxMode(**kwargs):
    """
    Deprecated: Use CUAMode(execution_mode='sandbox') instead.

    This function exists for backwards compatibility only and will be removed
    in a future version.
    """
    warnings.warn(
        "CUASandboxMode is deprecated, use CUAMode(execution_mode='sandbox') instead",
        DeprecationWarning,
        stacklevel=2,
    )
    kwargs.pop("execution_mode", None)
    return CUAMode(execution_mode="sandbox", **kwargs)


__all__ = ["BrowserMode", "DOMMode", "CUAMode", "CUASandboxMode", "SANDBOX_AVAILABLE"]
