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
    skills_path:
        Sandbox path where taskset skills are uploaded.  Setting this
        is the recommended way to enable skills upload.  Equivalent to
        ``upload_dir_mapping={"skills": skills_path}``.
        Example: ``"/task/rlm-skills"``.
    upload_dir_mapping:
        Maps logical directory names (declared by
        ``TaskSet.get_upload_dirs()``) to absolute sandbox paths.
        ``skills_path`` is merged into this mapping automatically.
        Use for non-skills directories; for skills prefer
        ``skills_path``.
    metrics_path:
        Glob pattern for a JSON metrics file inside the sandbox,
        collected after the rollout.  May contain ``{workdir}`` which is
        resolved to ``taskset.get_workdir(info)`` at runtime.
        Example: ``"{workdir}/.rlm/sessions/*/meta.json"``.
    metrics_prefix:
        String prepended to each metric key when surfaced in rollout
        state.  Example: ``"rlm_"`` turns ``"turns"`` into
        ``"rlm_turns"``.
    metrics_key:
        Optional key to drill into within the JSON file.  If set, only
        the value at this key is used as the metrics dict.  Example:
        ``"metrics"`` reads ``json["metrics"]`` instead of the top-level
        object.
    metrics_keys:
        Optional whitelist of metric keys to surface.  ``None`` means
        surface all keys found.
    """

    install_script: str | None = None
    install_timeout: int = 300
    run_command: str = ""
    system_prompt: str | None = None
    system_prompt_path: str = "/task/system_prompt.txt"
    instruction_path: str = "/task/instruction.md"
    log_path: str | None = None
    sandbox_spec: SandboxSpec | None = None
    skills_path: str | None = None
    upload_dir_mapping: dict[str, str] | None = None
    metrics_path: str | None = None
    metrics_prefix: str = ""
    metrics_key: str | None = None
    metrics_keys: list[str] | None = None

    def get_effective_upload_dir_mapping(self) -> dict[str, str] | None:
        """Return the merged upload mapping (skills_path + upload_dir_mapping)."""
        mapping = dict(self.upload_dir_mapping) if self.upload_dir_mapping else {}
        if self.skills_path:
            mapping.setdefault("skills", self.skills_path)
        return mapping or None
