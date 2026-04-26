__all__ = [
    "CliConfig",
    "CliHarness",
    "CliMetrics",
    "CliPaths",
    "EndpointConfig",
    "EndpointHarness",
    "OpenCode",
    "RLMHarness",
    "RunConfig",
    "SandboxConfig",
    "SandboxRuntime",
    "SandboxScoring",
    "SandboxSetup",
]

_LAZY_IMPORTS = {
    "CliConfig": "verifiers.envs.experimental.configs:CliConfig",
    "CliHarness": "verifiers.envs.experimental.modules.harnesses.cli_harness:CliHarness",
    "CliMetrics": "verifiers.envs.experimental.configs:CliMetrics",
    "CliPaths": "verifiers.envs.experimental.configs:CliPaths",
    "EndpointConfig": "verifiers.envs.experimental.configs:EndpointConfig",
    "EndpointHarness": (
        "verifiers.envs.experimental.modules.harnesses.endpoint_harness:EndpointHarness"
    ),
    "OpenCode": "verifiers.envs.experimental.modules.harnesses.opencode_harness:OpenCode",
    "RLMHarness": (
        "verifiers.envs.experimental.modules.harnesses.rlm_harness:RLMHarness"
    ),
    "RunConfig": "verifiers.envs.experimental.configs:RunConfig",
    "SandboxConfig": "verifiers.envs.experimental.configs:SandboxConfig",
    "SandboxRuntime": "verifiers.envs.experimental.configs:SandboxRuntime",
    "SandboxScoring": "verifiers.envs.experimental.configs:SandboxScoring",
    "SandboxSetup": "verifiers.envs.experimental.configs:SandboxSetup",
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_path, attr = _LAZY_IMPORTS[name].split(":")
    return getattr(importlib.import_module(module_path), attr)
