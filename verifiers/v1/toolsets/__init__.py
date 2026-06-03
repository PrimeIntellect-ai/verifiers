LAZY_EXPORTS = {
    "browser_toolset": (".browser", "browser_toolset"),
    "BrowserBackend": (".browser", "BrowserBackend"),
    "BrowserSession": (".browser", "BrowserSession"),
    "BrowserSessionHandle": (".browser", "BrowserSessionHandle"),
    "BrowserbaseBackend": (".browser", "BrowserbaseBackend"),
    "CDPBackend": (".browser", "CDPBackend"),
}

__all__ = [*LAZY_EXPORTS]


def __getattr__(name: str):
    if name in LAZY_EXPORTS:
        module_name, symbol_name = LAZY_EXPORTS[name]
        from importlib import import_module

        try:
            return getattr(import_module(module_name, __name__), symbol_name)
        except ModuleNotFoundError as exc:
            if exc.name == "websockets":
                raise ImportError(
                    f"To use {name}, install the browser extra: `verifiers[browser]`."
                ) from exc
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
