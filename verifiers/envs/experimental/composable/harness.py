"""Harness — agent-side configuration for ComposableEnv.

A Harness declares how to install and run an agent binary, and where it
expects to find task-provided content (instruction, system prompt).

The Task produces content, the Harness declares paths, the Environment
connects them.

::

    from opencode_agent import opencode_harness

    harness = opencode_harness(system_prompt="You are a coding agent...")
    env = ComposableEnv(taskset=taskset, harness=harness)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.envs.experimental.composable.task import MCPServerSpec, SandboxSpec


@dataclass
class Harness:
    """Agent-side configuration.

    Attributes
    ----------
    install_script:
        Shell command to install the agent binary in the sandbox.
    run_command:
        Shell command to start the agent.
    system_prompt:
        System prompt content. Written to ``system_prompt_path`` in the
        sandbox before the agent starts. None = no system prompt.
    system_prompt_path:
        Where the system prompt is written in the sandbox.
        Only used if ``system_prompt`` is not None.
    instruction_path:
        Where the task instruction is written in the sandbox.
    log_path:
        Optional path to the agent log file inside the sandbox.
    sandbox_spec:
        Default sandbox resources when the task doesn't provide a
        SandboxSpec (e.g. math + OpenCode — the agent needs a sandbox
        but the task doesn't specify one).
    mcp_servers:
        MCP servers merged from the TaskSet.  Populated by ComposableEnv
        at setup time — do not set manually.
    """

    install_script: str | None = None
    run_command: str = ""
    system_prompt: str | None = None
    system_prompt_path: str = "/task/system_prompt.txt"
    instruction_path: str = "/task/instruction.md"
    log_path: str | None = None
    sandbox_spec: SandboxSpec | None = None
    mcp_servers: dict[str, MCPServerSpec] = field(default_factory=dict)

    def format_mcp_config(self, servers: dict[str, MCPServerSpec]) -> str | None:
        """Format MCP server specs into agent-native config.

        Override in harness factories to produce agent-specific config.
        For example, an OpenCode harness would write the ``mcp`` section
        of ``opencode.json``.

        Parameters
        ----------
        servers:
            Mapping of server name → MCPServerSpec from the TaskSet.

        Returns
        -------
        str or None:
            Config content to write into the sandbox, or None if the
            harness handles MCP servers through other means (e.g. by
            regenerating its run_command).
        """
        return None
