"""ComposableEnv — a CliAgentEnv that delegates to a TaskSet + Harness.

Subclasses ``CliAgentEnv`` and overrides its hooks to delegate to the
``TaskSet`` (what to solve) and ``Harness`` (how the agent runs).

Task / Harness contract
-----------------------

The task and harness each own different concerns.  ComposableEnv connects them:

**Task owns** (via TaskSet / SandboxTaskSet):
- Instruction text: ``get_instruction(info) -> str``
- Docker image + resources: ``get_sandbox_spec(info) -> SandboxSpec``
- Working directory: ``get_workdir(info) -> str`` (exported as ``AGENT_WORKDIR``)
- Sandbox setup: ``setup(sandbox_client, sandbox_id, state)``
- Evaluation: ``evaluate(...)``
- Environment variables: ``get_env_vars() -> dict``

**Harness owns** (via Harness dataclass):
- How to install the agent: ``install_script``
- How to run the agent: ``run_command``
- Where the agent reads instruction: ``instruction_path`` (default ``/opencode/prompt.txt``)
- Where the agent reads system prompt: ``system_prompt_path``
- System prompt content: ``system_prompt``
- Fallback sandbox resources: ``sandbox_spec`` (when task doesn't need sandbox)

**ComposableEnv connects them**:
1. Writes task's instruction text → harness's instruction_path
2. Writes harness's system prompt → harness's system_prompt_path
3. Harness's ``run_command`` reads from these paths

ComposableEnv exports the task's working directory as ``AGENT_WORKDIR`` for
harnesses that need a per-instance workdir while still using a static
``run_command``.
"""

from __future__ import annotations

import asyncio
import importlib.resources as resources
import json
import logging
import shlex
import tarfile
import tempfile
from importlib.abc import Traversable
from pathlib import Path
from typing import Any

import verifiers as vf
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.envs.experimental.composable.harness import Harness
from verifiers.envs.experimental.composable.task import TaskSet
from verifiers.envs.experimental.utils.file_locks import shared_path_lock
from verifiers.envs.tool_env import ToolMonitorRubric
from verifiers.types import State, TrajectoryStep

logger = logging.getLogger(__name__)


class HarnessMetricsRubricGroup(vf.RubricGroup):
    async def cleanup(self, state: State) -> None:
        for rubric in self.rubrics:
            await rubric.cleanup(state)
        harness_metrics = state.get("_harness_metrics")
        if not isinstance(harness_metrics, dict):
            return
        state_metrics = state.get("metrics")
        if not isinstance(state_metrics, dict):
            state_metrics = {}
            state["metrics"] = state_metrics
        for key, value in harness_metrics.items():
            if isinstance(key, str) and isinstance(value, (int, float)):
                state_metrics[key] = float(value)


class ComposableEnv(CliAgentEnv):
    """CliAgentEnv that delegates to a TaskSet and a Harness.

    For SandboxTaskSet: uses task's SandboxSpec for image, task's setup/evaluate.
    For plain TaskSet: uses harness default image, task evaluate takes no sandbox args.

    Parameters
    ----------
    taskset:
        A ``TaskSet`` or ``SandboxTaskSet``.
    harness:
        A ``Harness`` — the agent configuration.
    """

    def __init__(
        self,
        taskset: TaskSet,
        harness: Harness,
        *,
        install_env: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        kwargs["dataset"] = taskset.get_dataset()
        if "rubric" not in kwargs:
            kwargs["rubric"] = taskset.get_rubric()
        super().__init__(run_command=harness.run_command, **kwargs)

        self.taskset = taskset
        self.harness = harness
        self.install_env = dict(install_env) if install_env else None

        if harness.tool_names:
            self.add_rubric(ToolMonitorRubric(tool_names=list(harness.tool_names)))
        if harness.metrics_path:
            rubrics = (
                list(self.rubric.rubrics)
                if isinstance(self.rubric, vf.RubricGroup)
                else [self.rubric]
            )
            self.rubric = HarnessMetricsRubricGroup(rubrics=rubrics)

    # -- CliAgentEnv hooks --------------------------------------------------

    def _get_spec(self, state: State) -> Any:
        """Get SandboxSpec, cached on state to avoid redundant calls."""
        cached = state.get("_sandbox_spec")
        if cached is not None:
            return cached
        info = state.get("info") or {}
        spec = self.taskset.get_sandbox_spec(info)
        state["_sandbox_spec"] = spec
        return spec

    async def get_docker_image(self, state: State) -> str:
        spec = self._get_spec(state)
        if spec:
            return spec.image
        if self.harness.sandbox_spec:
            return self.harness.sandbox_spec.image
        return self.docker_image

    def get_sandbox_resources(self, state: State) -> dict[str, Any]:
        """Per-instance resources from SandboxSpec, or harness defaults."""
        spec = self._get_spec(state) or self.harness.sandbox_spec
        if spec:
            return {
                "cpu_cores": spec.cpu_cores,
                "memory_gb": spec.memory_gb,
                "disk_size_gb": spec.disk_size_gb,
                "gpu_count": spec.gpu_count,
                "gpu_type": spec.gpu_type,
                "vm": spec.gpu_count > 0,
                "timeout_minutes": spec.timeout_minutes,
            }
        return super().get_sandbox_resources(state)

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
        info = state.get("info") or {}
        harness_env_vars = self.harness.environment_vars
        if harness_env_vars:
            conflicts = (
                self.PROTECTED_ENV_VARS | {"AGENT_WORKDIR"}
            ) & harness_env_vars.keys()
            if conflicts:
                raise ValueError(
                    f"Harness.environment_vars must not override protected keys: {conflicts}."
                )
            env_vars.update(harness_env_vars)
        task_env_vars = self.taskset.get_env_vars()
        if task_env_vars:
            conflicts = (
                self.PROTECTED_ENV_VARS | {"AGENT_WORKDIR"}
            ) & task_env_vars.keys()
            if conflicts:
                raise ValueError(
                    f"TaskSet.get_env_vars() must not override protected keys: {conflicts}."
                )
            env_vars.update(task_env_vars)
        env_vars["AGENT_WORKDIR"] = self.taskset.get_workdir(info)
        return env_vars

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ) -> None:
        """Append the step unless the harness's filter says to drop it.

        Reads the originating request's headers from
        ``state["_last_request_headers"]`` — ``CliAgentEnv.get_model_response``
        stashes them there before clearing ``current_request_id``, since
        ``add_trajectory_step`` runs *after* that clear. Headers, step,
        and state are passed to ``harness.keep_trajectory_step``;
        ``True`` keeps, ``False`` drops. ``None`` filter (default) keeps
        every step.
        """
        if self.harness.keep_trajectory_step is not None:
            headers = state.get("_last_request_headers") or {}
            if not self.harness.keep_trajectory_step(trajectory_step, state, headers):
                return
        await super().add_trajectory_step(state, trajectory_step)

    async def post_sandbox_setup(self, state: State) -> None:
        """Task setup → upload instruction/system prompt → upload dirs →
        install agent → post-install (uploads + script).

        The post-install step runs ``Harness.post_install_uploads`` and
        ``Harness.post_install_script`` after the agent is fully
        installed — harnesses use it to layer small assets onto the
        installed agent (e.g. RLM's ``/usr/local/bin/git`` refusal
        shim)."""
        sandbox_id = state["sandbox_id"]

        await self._populate_sandbox_context(state)
        await self.taskset.setup(state)
        await self._create_harness_input_dirs(sandbox_id)
        await self._upload_harness_inputs(sandbox_id, state)
        await self._after_harness_inputs_uploaded(state)
        await self._install_agent(sandbox_id)
        await self._run_post_install(sandbox_id)

    async def post_rollout(self, state: State) -> None:
        """Collect agent logs and harness metrics after the agent finishes.

        Scoring is handled entirely by the rubric (via ``score_rollout``),
        not here.  Use ``keep_sandbox_for_scoring=True`` so the sandbox
        stays alive for the rubric to run tests / read files.
        """
        sandbox_id = state.get("sandbox_id")
        if sandbox_id and self.harness.log_path and "agent_logs" not in state:
            try:
                log_path = shlex.quote(self.harness.log_path)
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat {log_path} 2>/dev/null || echo '<no logs>'",
                    working_dir=None,
                )
                state["agent_logs"] = (result.stdout or "").strip()
            except Exception as e:
                self.logger.warning(f"Failed to collect agent logs: {e}")

        if sandbox_id and self.harness.metrics_path:
            await self._collect_harness_metrics(sandbox_id, state)

        await super().post_rollout(state)

    async def _populate_sandbox_context(self, state: State) -> None:
        """Populate sandbox-specific context used by setup/evaluate hooks."""
        state["sandbox_client"] = self.sandbox_client
        spec = self._get_spec(state)
        if spec:
            state["test_timeout"] = spec.timeout_minutes * 60
        elif self.harness.sandbox_spec:
            state["test_timeout"] = self.harness.sandbox_spec.timeout_minutes * 60
        else:
            state["test_timeout"] = 900

    async def _create_harness_input_dirs(self, sandbox_id: str) -> None:
        """Create parent directories for harness-managed task assets."""
        dirs = {self.harness.instruction_path.rsplit("/", 1)[0]}
        if self.harness.system_prompt:
            dirs.add(self.harness.system_prompt_path.rsplit("/", 1)[0])
        mkdir_args = " ".join(shlex.quote(path) for path in sorted(dirs))
        await self.sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {mkdir_args}", timeout=self.timeouts.mkdir
        )

    async def _upload_harness_inputs(self, sandbox_id: str, state: State) -> None:
        """Upload instruction and optional system prompt to harness-declared paths."""
        info = state.get("info") or {}
        instruction = self.taskset.get_instruction(info)
        if instruction.strip():
            await self.upload_content(
                sandbox_id, instruction, self.harness.instruction_path
            )

        if self.harness.system_prompt:
            await self.upload_content(
                sandbox_id, self.harness.system_prompt, self.harness.system_prompt_path
            )

    async def _after_harness_inputs_uploaded(self, state: State) -> None:
        """Upload task-declared directories to harness-declared sandbox paths.

        Joins task-declared and harness-declared upload directories with
        ``Harness.upload_dir_mapping`` (logical name → sandbox path).
        Only directories whose logical name appears in both are uploaded.
        """
        upload_dirs = self._get_upload_dirs()
        mapping = self.harness.get_effective_upload_dir_mapping()
        if not upload_dirs or not mapping:
            return
        sandbox_id = state["sandbox_id"]
        for name, local_source in upload_dirs.items():
            remote_dest = mapping.get(name)
            if remote_dest is not None:
                await self._upload_dir(sandbox_id, local_source, remote_dest)

    def _get_upload_dirs(self) -> dict[str, Traversable | Path]:
        """Merge task-owned and harness-owned upload directories."""
        task_upload_dirs = dict(self.taskset.get_upload_dirs() or {})
        harness_upload_dirs_value = (
            self.harness.get_upload_dirs() if self.harness.get_upload_dirs else None
        )
        harness_upload_dirs = dict(harness_upload_dirs_value or {})
        duplicate_names = sorted(set(task_upload_dirs) & set(harness_upload_dirs))
        if duplicate_names:
            names = ", ".join(repr(name) for name in duplicate_names)
            raise ValueError(
                "Upload directory names must be unique across task and harness; "
                f"duplicates: {names}."
            )
        task_upload_dirs.update(harness_upload_dirs)
        return task_upload_dirs

    def _get_install_execute_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to sandbox install command execution."""
        kwargs: dict[str, Any] = {"timeout": self.harness.install_timeout}
        if self.install_env:
            kwargs["env"] = self.install_env
        return kwargs

    async def _install_agent(self, sandbox_id: str) -> None:
        """Install the agent inside the sandbox when an install script is present."""
        if self.harness.install_script:
            self.logger.debug(f"Installing agent in sandbox {sandbox_id}")
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                self.harness.install_script,
                **self._get_install_execute_kwargs(),
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Agent install failed (exit={result.exit_code}): {output[:500]}"
                )

    async def _run_post_install(self, sandbox_id: str) -> None:
        """Upload harness ``post_install_uploads`` and run ``post_install_script``.

        Runs after ``_install_agent`` so harnesses can layer small assets
        on top of a fully-installed agent (e.g. RLM stages its git refusal
        shim into ``$HOME/.local/bin/git`` and chmods it executable).
        Uses the single-file upload path — not ``_upload_dir`` — because
        these are small, harness-computed blobs of content rather than
        local directories on disk.
        """
        uploads = self.harness.post_install_uploads
        if uploads:
            for remote_path, content in uploads.items():
                await self.upload_content(sandbox_id, content, remote_path)

        if self.harness.post_install_script:
            self.logger.debug(f"Running post-install script in sandbox {sandbox_id}")
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                self.harness.post_install_script,
                **self._get_install_execute_kwargs(),
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Post-install failed (exit={result.exit_code}): {output[:500]}"
                )

    # -- Directory upload ------------------------------------------------------

    async def _upload_dir(
        self,
        sandbox_id: str,
        local_source: Traversable | Path,
        remote_dest: str,
    ) -> None:
        """Tar, upload, and extract a directory into the sandbox.

        Building the gzipped tar is sync, CPU-bound, and for large sources can
        take hundreds of milliseconds; offload it to a worker thread so the
        event loop stays responsive when many rollouts upload in parallel.
        """
        remote_tar = f"/tmp/_upload_{remote_dest.strip('/').replace('/', '_')}.tar.gz"
        tmp_path = await asyncio.to_thread(
            self._build_dir_archive, local_source, remote_dest
        )
        try:
            await self.upload_file(sandbox_id, remote_tar, str(tmp_path))
            dest_parent = shlex.quote(str(Path(remote_dest).parent))
            quoted_remote_tar = shlex.quote(remote_tar)
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f"mkdir -p {dest_parent} && "
                f"tar -xzf {quoted_remote_tar} -C / && "
                f"rm -f {quoted_remote_tar}",
                timeout=self.timeouts.extract,
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Upload dir extract failed (exit={result.exit_code}): {output[:500]}"
                )
        finally:
            tmp_path.unlink(missing_ok=True)

    def _build_dir_archive(
        self, local_source: Traversable | Path, remote_dest: str
    ) -> Path:
        """Build a tar.gz archive of a directory, rooted at *remote_dest*."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = Path(tmp_file.name)
        arcname = remote_dest.lstrip("/")
        with tarfile.open(tar_path, "w:gz") as tar:
            if isinstance(local_source, Path):
                with shared_path_lock(local_source, suffix=".upload.lock"):
                    tar.add(local_source, arcname=arcname)
            else:
                with resources.as_file(local_source) as local_path:
                    tar.add(local_path, arcname=arcname)
        return tar_path

    # -- Harness metrics collection --------------------------------------------

    async def _collect_harness_metrics(self, sandbox_id: str, state: State) -> None:
        """Read a JSON metrics file from the sandbox and surface keys in state."""
        if not self.harness.metrics_path:
            return
        info = state.get("info") or {}
        workdir = self.taskset.get_workdir(info)
        metrics_glob = self.harness.metrics_path.format(workdir=workdir)
        try:
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f"f=$(ls {metrics_glob} 2>/dev/null | head -1) "
                '&& cat "$f" || echo "{}"',
                working_dir=None,
            )
            data = json.loads((result.stdout or "{}").strip())
            if self.harness.metrics_key:
                data = data.get(self.harness.metrics_key, {})
            prefix = self.harness.metrics_prefix
            allowed = self.harness.metrics_keys
            harness_metrics = state.get("_harness_metrics")
            if not isinstance(harness_metrics, dict):
                harness_metrics = {}
                state["_harness_metrics"] = harness_metrics
            for key, value in data.items():
                if allowed is None or key in allowed:
                    prefixed_key = f"{prefix}{key}"
                    state[prefixed_key] = value
                    if isinstance(value, (int, float)):
                        harness_metrics[prefixed_key] = float(value)
        except Exception as e:
            self.logger.warning(f"Failed to collect harness metrics: {e}")
