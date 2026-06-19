"""Optional renderers types used by v1 import-time models.

``renderers`` is supplied by prime-rl in training runs, but lightweight environment
imports should not force a resolver to install a different published renderers
wheel.
"""

from __future__ import annotations

from typing import Any


def _missing_renderers(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise ImportError(
        "install renderers or run inside prime-rl to use renderer features"
    )


try:
    from renderers import RendererConfig, RenderedTokens
    from renderers import OverlongPromptError as RendererOverlongPromptError
    from renderers.base import MultiModalData, PlaceholderRange, is_multimodal
except ImportError:
    RendererConfig = Any
    RenderedTokens = Any
    RendererOverlongPromptError = Exception

    class MultiModalData:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_renderers(*args, **kwargs)

    class PlaceholderRange:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _missing_renderers(*args, **kwargs)

    def is_multimodal(*args: Any, **kwargs: Any) -> bool:
        _missing_renderers(*args, **kwargs)
