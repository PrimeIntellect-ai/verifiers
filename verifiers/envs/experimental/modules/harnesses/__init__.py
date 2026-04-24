__all__ = [
    "CliHarness",
    "EndpointHarness",
    "OpenCode",
]

_LAZY_IMPORTS = {
    "CliHarness": "verifiers.envs.experimental.modules.harnesses.cli_harness:CliHarness",
    "EndpointHarness": (
        "verifiers.envs.experimental.modules.harnesses.endpoint_harness:EndpointHarness"
    ),
    "OpenCode": "verifiers.envs.experimental.modules.harnesses.opencode_harness:OpenCode",
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = _LAZY_IMPORTS[name].split(":")
    return getattr(importlib.import_module(module_path), attr)
