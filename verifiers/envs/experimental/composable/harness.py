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

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.envs.experimental.composable.task import SandboxSpec


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
    """

    install_script: str | None = None
    run_command: str = ""
    system_prompt: str | None = None
    system_prompt_path: str = "/task/system_prompt.txt"
    instruction_path: str = "/task/instruction.md"
    log_path: str | None = None
    sandbox_spec: SandboxSpec | None = None
