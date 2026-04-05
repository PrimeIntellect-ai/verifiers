"""Experimental environments and resource managers."""

from . import resource_managers

__all__ = [
    "CliAgentEnv",
    "HarborEnv",
    "NewCliAgentEnv",
    "NewHarborEnv",
    "NewSandboxEnv",
    "resource_managers",
]


def __getattr__(name: str):
    if name == "CliAgentEnv":
        from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
        return CliAgentEnv
    elif name == "HarborEnv":
        from verifiers.envs.experimental.harbor_env import HarborEnv
        return HarborEnv
    elif name == "NewCliAgentEnv":
        from verifiers.envs.experimental.new_cli_agent_env import NewCliAgentEnv
        return NewCliAgentEnv
    elif name == "NewHarborEnv":
        from verifiers.envs.experimental.new_harbor_env import NewHarborEnv
        return NewHarborEnv
    elif name == "NewSandboxEnv":
        from verifiers.envs.experimental.new_sandbox_env import NewSandboxEnv
        return NewSandboxEnv
    raise AttributeError(f"module 'verifiers.envs.experimental' has no attribute '{name}'")
