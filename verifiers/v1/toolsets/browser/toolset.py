from typing import Literal

from verifiers.v1 import ObjectsConfig
from verifiers.v1.toolset import Toolset

from .backends import BrowserBackend
from .tools import DECOMPOSED_TOOLS, computer

Mode = Literal["computer", "decomposed", "both"]

# Import ref for the rollout browser object. v1 requires object entries to be
# import-ref strings; per-rollout config (the backend, viewport, start url) is
# supplied through bindings instead of a closure.
_BROWSER_OBJECT_REF = "verifiers.v1.toolsets.browser.session:open_browser_session"


def _const(value: object):
    """A binding source that always resolves to ``value`` (a literal config)."""

    def source() -> object:
        return value

    return source


def browser_toolset(
    *,
    backend: BrowserBackend,
    mode: Mode = "both",
    width: int = 1280,
    height: int = 800,
    start_url: str | None = None,
) -> Toolset:
    """Build a Toolset for browser/computer control over raw CDP via ``backend``."""
    if backend is None:
        raise ValueError(
            "browser_toolset requires a backend (e.g. BrowserbaseBackend())."
        )
    if mode == "computer":
        tools = [computer]
    elif mode == "decomposed":
        tools = list(DECOMPOSED_TOOLS)
    elif mode == "both":
        tools = [computer, *DECOMPOSED_TOOLS]
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"Unknown mode: {mode!r}")

    bindings = {
        "browser.backend": _const(backend),
        "browser.width": _const(width),
        "browser.height": _const(height),
        "browser.start_url": _const(start_url),
        **{f"{tool.__name__}.session": "objects.browser" for tool in tools},
    }

    return Toolset(
        tools=tools,
        objects=ObjectsConfig.model_validate({"browser": _BROWSER_OBJECT_REF}),
        bindings=bindings,
        write=True,
        scope="rollout",
    )
