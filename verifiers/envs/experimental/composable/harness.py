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
from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol

if TYPE_CHECKING:
    from verifiers.envs.experimental.composable.composable_env import ComposableEnv
    from verifiers.envs.experimental.composable.task import SandboxSpec
    from verifiers.types import State, TrajectoryStep


class StateCollector(Protocol):
    """Harness-owned lifecycle hook for writing rollout artifacts into state."""

    async def post_sandbox_setup(self, env: ComposableEnv, state: State) -> None: ...

    async def post_rollout(self, env: ComposableEnv, state: State) -> None: ...


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
    get_upload_dirs:
        Optional callable returning harness-owned local directories to
        upload into the sandbox before install. These are merged with
        task-declared upload dirs by ``ComposableEnv`` and resolved via
        the same ``upload_dir_mapping`` logical-name contract.
        Example: ``lambda: {"agent_src": Path("/path/to/checkout")}``.
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
    tool_names:
        Names of the tools the agent uses internally.  When non-empty,
        ``ComposableEnv`` auto-registers a ``ToolMonitorRubric`` that
        counts calls to each named tool (plus a total) from the
        assistant messages the harness emits into the trajectory.
        Example: ``["ipython"]`` for the RLM harness.
    environment_vars:
        Callable taking the per-rollout ``State`` and returning the
        env-var dict for that rollout. Merged by ``ComposableEnv``
        between the caller-supplied ``environment_vars=`` and the
        taskset's ``get_env_vars()``: harness wins over caller, taskset
        wins over harness. Always a function (even for static dicts:
        return the same dict regardless of ``state``) so the merge
        path is uniform; the per-rollout ``state`` is what enables
        seeded draws / prompt-conditional config without touching env
        infrastructure.
    post_install_uploads:
        Optional mapping from sandbox path → file content. Uploaded via
        the single-file upload path (same as instruction / system
        prompt) AFTER ``install_script`` finishes. General post-install
        hook for layering small harness-computed assets onto a fully
        installed agent; for large directories use
        ``upload_dir_mapping`` instead. Not tied to any specific
        feature — left as a generic extension point.
    post_install_script:
        Optional shell snippet run AFTER ``post_install_uploads`` land in
        the sandbox. General post-install wiring hook (e.g.
        ``chmod +x`` on uploaded files, symlinks, anything that needs
        the uploads in place first). Failure is fatal, same as
        ``install_script``. Not tied to any specific feature — left as
        a generic extension point.
    keep_trajectory_step:
        Optional per-step filter. Called once per intercepted API
        response with ``(step, state, request_headers)``; return
        ``True`` to keep, ``False`` to drop. ``None`` (default) keeps
        every step. Use this to elide nested-agent traffic from the
        trajectory the trainer sees — e.g. rlm_harness uses it to drop
        sub-agent calls (``X-RLM-Depth`` header > 0) so only the
        parent agent's turns contribute to the policy gradient.
    state_collectors:
        Optional harness-owned artifact collectors. ``ComposableEnv``
        runs each collector after sandbox setup and after rollout, before
        scoring mutates the task sandbox.
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
    get_upload_dirs: Callable[[], dict[str, Traversable | Path] | None] | None = None
    metrics_path: str | None = None
    metrics_prefix: str = ""
    metrics_key: str | None = None
    metrics_keys: list[str] | None = None
    tool_names: list[str] | None = None
    environment_vars: Callable[[State], dict[str, str]] | None = None
    post_install_uploads: dict[str, str] | None = None
    post_install_script: str | None = None
    keep_trajectory_step: (
        Callable[[TrajectoryStep, State, dict[str, str]], bool] | None
    ) = None
    state_collectors: list[StateCollector] | None = None

    def get_effective_upload_dir_mapping(self) -> dict[str, str] | None:
        """Return the merged upload mapping (skills_path + upload_dir_mapping)."""
        mapping = dict(self.upload_dir_mapping) if self.upload_dir_mapping else {}
        if self.skills_path:
            mapping.setdefault("skills", self.skills_path)
        return mapping or None
