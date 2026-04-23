"""Harness — agent-side configuration for ComposableEnv.

A Harness declares how to install and run an agent binary, and where it
expects to find task-provided content (instruction, system prompt).

The Task produces content, the Harness declares paths, the Environment
connects them.

::

    from harnesses.opencode import opencode_harness

    harness = opencode_harness(system_prompt="You are a coding agent...")
    env = ComposableEnv(taskset=taskset, harness=harness)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from verifiers.envs.composable_skills import TaskSkills
    from verifiers.envs.composable_tools import TaskTools


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
    skills_path:
        Sandbox path where taskset skills are uploaded. Equivalent to
        ``upload_dir_mapping={"skills": skills_path}``.
    upload_dir_mapping:
        Maps logical directory names from tasksets or harnesses to
        absolute sandbox paths.
    get_upload_dirs:
        Optional callable returning harness-owned local directories to
        upload before install.
    metrics_path:
        Glob pattern for a JSON metrics file inside the sandbox. May
        include ``{workdir}``.
    metrics_prefix:
        Prefix for metrics surfaced in rollout state.
    metrics_key:
        Optional key to drill into within the JSON metrics file.
    metrics_keys:
        Optional whitelist of metric keys to surface.
    tool_names:
        Names of the tools the agent uses internally. When non-empty,
        ``ComposableEnv`` auto-registers a ``ToolMonitorRubric`` that
        counts calls to each named tool (plus a total) from the
        assistant messages the harness emits into the trajectory.
        Example: ``["ipython", "summarize"]`` for the RLM harness.
    environment_vars:
        Harness-owned environment variables for the sandbox. Merged by
        ``ComposableEnv`` between the caller-supplied ``environment_vars=``
        and the taskset's ``get_env_vars()``: harness wins over caller,
        taskset wins over harness. This is the right place to put env
        vars that track other harness config (e.g. ``RLM_TOOLS`` paired
        with ``tool_names``) so they can't silently desync.
    post_install_uploads:
        Optional mapping from sandbox path → file content. Uploaded via
        the single-file upload path (same as instruction / system
        prompt) AFTER ``install_script`` finishes. Use for small
        harness-computed assets — e.g. RLM's ``/usr/local/bin/git``
        refusal shim. For large directories use ``upload_dir_mapping``
        instead.
    post_install_script:
        Optional shell snippet run AFTER ``post_install_uploads`` land in
        the sandbox. Typical use: ``chmod +x`` on the uploaded files, or
        any other wiring that needs them in place first. Failure is
        fatal, same as ``install_script``.
    configure_tools:
        Optional callback that returns a rollout-specific Harness with
        task-provided tools registered in the agent's native format.
    configure_skills:
        Optional callback that returns a rollout-specific Harness with
        task-provided skills registered in the agent's native format.
    """

    install_script: str | None = None
    install_timeout: int = 300
    run_command: str = ""
    system_prompt: str | None = None
    system_prompt_path: str = "/task/system_prompt.txt"
    instruction_path: str = "/task/instruction.md"
    log_path: str | None = None
    skills_path: str | None = None
    upload_dir_mapping: dict[str, str] | None = None
    get_upload_dirs: Callable[[], dict[str, Traversable | Path] | None] | None = None
    metrics_path: str | None = None
    metrics_prefix: str = ""
    metrics_key: str | None = None
    metrics_keys: list[str] | None = None
    tool_names: list[str] | None = None
    environment_vars: dict[str, str] | None = None
    post_install_uploads: dict[str, str] | None = None
    post_install_script: str | None = None
    configure_tools: Callable[["TaskTools"], "Harness"] | None = None
    configure_skills: Callable[["TaskSkills"], "Harness"] | None = None

    def get_effective_upload_dir_mapping(self) -> dict[str, str] | None:
        mapping = dict(self.upload_dir_mapping) if self.upload_dir_mapping else {}
        if self.skills_path:
            mapping.setdefault("skills", self.skills_path)
        return mapping or None

    def with_tools(self, tools: "TaskTools") -> "Harness":
        if not tools.has_harness_tools:
            return self
        if self.configure_tools is None:
            raise ValueError("This harness cannot register task-provided tools.")
        return self.configure_tools(tools)

    def with_skills(self, skills: "TaskSkills") -> "Harness":
        if not skills.has_skills:
            return self
        if skills.source_dir and not skills.skills_dir and self.skills_path:
            skills = replace(skills, skills_dir=self.skills_path)
        if self.configure_skills is not None:
            return self.configure_skills(skills)
        if self.skills_path and (
            not skills.skills_dir or skills.skills_dir == self.skills_path
        ):
            return self
        raise ValueError("This harness cannot register task-provided skills.")


MCPServerConfig = dict[str, Any] | str
ConfigurableHarnessBuilder = Callable[[list[MCPServerConfig], str | None], Harness]


def make_configurable_harness(
    build_harness: ConfigurableHarnessBuilder,
    *,
    mcp_servers: list[MCPServerConfig] | None = None,
    skills_dir: str | None = None,
    supports_tools: bool = True,
    supports_skills: bool = True,
) -> Harness:
    """Build a Harness that can be rebuilt with task-provided tools/skills."""

    def build(
        effective_mcp_servers: list[MCPServerConfig],
        effective_skills_dir: str | None,
    ) -> Harness:
        harness = build_harness(effective_mcp_servers, effective_skills_dir)

        def configure_tools(tools: "TaskTools") -> Harness:
            return build(
                [*effective_mcp_servers, *tools.mcp_servers],
                effective_skills_dir,
            )

        def configure_skills(skills: "TaskSkills") -> Harness:
            return build(
                effective_mcp_servers,
                skills.skills_dir or effective_skills_dir,
            )

        return replace(
            harness,
            configure_tools=configure_tools if supports_tools else None,
            configure_skills=configure_skills if supports_skills else None,
        )

    return build(list(mcp_servers or []), skills_dir)


def make_native_harness(
    *,
    build_run_command: Callable[..., str],
    run_kwargs: dict[str, Any],
    install_script: str | None = None,
    system_prompt: str | None = None,
    instruction_path: str = "/task/instruction.md",
    system_prompt_path: str = "/task/system_prompt.txt",
    log_path: str | None = None,
    default_skills_path: str | None = None,
    mcp_servers: list[MCPServerConfig] | None = None,
    skills_dir: str | None = None,
    supports_tools: bool = True,
    supports_skills: bool = True,
    pass_mcp_servers: bool = True,
    pass_skills_dir: bool = True,
) -> Harness:
    """Create a configurable Harness around a run-command builder."""

    def build_harness(
        effective_mcp_servers: list[MCPServerConfig],
        effective_skills_dir: str | None,
    ) -> Harness:
        effective_run_kwargs = dict(run_kwargs)
        if pass_mcp_servers:
            effective_run_kwargs["mcp_servers"] = effective_mcp_servers
        if pass_skills_dir:
            effective_run_kwargs["skills_dir"] = effective_skills_dir

        return Harness(
            install_script=install_script,
            run_command=build_run_command(**effective_run_kwargs),
            system_prompt=system_prompt,
            instruction_path=instruction_path,
            system_prompt_path=system_prompt_path,
            log_path=log_path,
            skills_path=effective_skills_dir or default_skills_path,
        )

    return make_configurable_harness(
        build_harness,
        mcp_servers=mcp_servers,
        skills_dir=skills_dir,
        supports_tools=supports_tools,
        supports_skills=supports_skills,
    )
