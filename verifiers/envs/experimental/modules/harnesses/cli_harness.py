from __future__ import annotations

import asyncio
import io
import json
import shlex
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterable
from importlib import resources as importlib_resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

from prime_sandboxes import (
    SandboxOOMError,
    SandboxTimeoutError,
)

from verifiers.decorators import cleanup, stop
from verifiers.envs.experimental.channels import (
    ChannelMap,
    Endpoint,
    SandboxResources,
    SandboxSeed,
    SandboxSpec,
    SandboxTimeouts,
    compose_rubrics,
)
from verifiers.envs.experimental.modules.harnesses.endpoint_harness import (
    EndpointHarness,
)
from verifiers.envs.experimental.task import Task
from verifiers.envs.tool_env import ToolMonitorRubric
from verifiers.errors import InfraError, SandboxError
from verifiers.rubrics.rubric import Rubric
from verifiers.types import State
from verifiers.utils.error_utils import error_info

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from verifiers.envs.experimental.resources import Resources


class SandboxMonitorRubric(Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.sandbox_oom)
        self.add_metric(self.sandbox_timeout)

    async def sandbox_oom(self, state: State) -> float:
        return float(bool(state.get("sandbox_oom")))

    async def sandbox_timeout(self, state: State) -> float:
        return float(bool(state.get("sandbox_timeout")))


class CliMonitorRubric(Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_metric(self.agent_timeout)
        self.add_metric(self.agent_error)

    async def agent_timeout(self, state: State) -> float:
        return float(bool(state.get("agent_timed_out")))

    async def agent_error(self, state: State) -> float:
        agent_exit_code = state.get("agent_exit_code")
        if agent_exit_code is None:
            return 0.0
        return float(agent_exit_code != 0)


class HarnessMetricsRubric(Rubric):
    async def score_rollout(self, state: State):
        state["reward"] = 0.0
        state["metrics"] = {}

    async def score_group(self, states: list[State]):
        for state in states:
            state["reward"] = 0.0
            state["metrics"] = {}

    async def cleanup(self, state: State):
        await super().cleanup(state)
        harness_metrics = state.get("harness_metrics")
        if not isinstance(harness_metrics, dict):
            return
        metrics = state.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            state["metrics"] = metrics
        for key, value in harness_metrics.items():
            if isinstance(key, str) and isinstance(value, int | float):
                metrics[key] = float(value)


class CliHarness(EndpointHarness):
    """Sandboxed CLI harness with OpenAI-compatible endpoint forwarding."""

    use_tunnel_for_endpoint = True

    PROTECTED_ENV_VARS = frozenset(
        {
            "OPENAI_BASE_URL",
            "OPENAI_TIMEOUT",
            "OPENAI_REQUEST_TIMEOUT",
            "HTTPX_TIMEOUT",
            "OPENAI_MODEL",
            "OPENAI_API_KEY",
            "AGENT_WORKDIR",
            "TASK_INSTRUCTION_PATH",
            "SYSTEM_PROMPT_PATH",
        }
    )

    def __init__(
        self,
        command: str,
        instruction_path: str = "/task/instruction.md",
        system_prompt_path: str = "/task/system_prompt.md",
        agent_workdir: str = "/workspace",
        log_path: str | None = None,
        system_prompt: str | None = None,
        sandbox: SandboxSpec | None = None,
        install_command: str | None = None,
        install_timeout: int = 300,
        install_env: dict[str, str] | None = None,
        post_install_uploads: dict[str, str] | None = None,
        post_install_command: str | None = None,
        skills_path: str | None = None,
        uploads: dict[str, object] | None = None,
        upload_mapping: dict[str, str] | None = None,
        metrics_path: str | None = None,
        metrics_prefix: str = "",
        metrics_key: str | None = None,
        metrics_keys: list[str] | None = None,
        tool_names: list[str] | None = None,
        timeout_seconds: float = 3600.0,
        poll_interval: float = 1.0,
        environment_vars: dict[str, str] | None = None,
        keep_sandbox_for_scoring: bool = False,
        sandbox_wait_for_creation_max_attempts: int = 120,
        sandbox_creations_per_minute: float | None = 128,
        sandbox_client_max_workers: int = 50,
        sandbox_client_max_connections: int = 1000,
        sandbox_client_max_keepalive_connections: int = 200,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        timeouts: SandboxTimeouts = SandboxTimeouts(),
        endpoint_port: int | None = None,
        endpoint_url: str | None = None,
        endpoint_secret: str | None = None,
        api_client_type: str = "openai_chat_completions",
        rubric: Rubric | None = None,
        tools: Iterable[object] | None = None,
        max_turns: int = -1,
        parallel_model_requests: bool = True,
        error_formatter: Callable[[Exception], str] = str,
        stop_errors: list[type[Exception]] | None = None,
    ):
        if tool_names:
            rubric = compose_rubrics(rubric, ToolMonitorRubric(tool_names=tool_names))
        if metrics_path:
            rubric = compose_rubrics(rubric, HarnessMetricsRubric())
        rubric = compose_rubrics(
            rubric,
            SandboxMonitorRubric(),
            CliMonitorRubric(),
        )
        super().__init__(
            endpoint_port=endpoint_port,
            endpoint_url=endpoint_url,
            endpoint_secret=endpoint_secret,
            api_client_type=api_client_type,
            max_turns=max_turns,
            poll_interval=poll_interval,
            rubric=rubric,
            system_prompt=system_prompt,
            tools=tools,
            parallel_model_requests=parallel_model_requests,
            error_formatter=error_formatter,
            stop_errors=stop_errors,
        )
        self.command = command
        self.instruction_path = instruction_path
        self.system_prompt_path = system_prompt_path
        self.agent_workdir = agent_workdir
        self.log_path = log_path
        self.sandbox_spec = sandbox or SandboxSpec()
        self.install_command = install_command
        self.install_timeout = install_timeout
        self.install_env = dict(install_env) if install_env else None
        self.post_install_uploads = dict(post_install_uploads or {})
        self.post_install_command = post_install_command
        self.skills_path = skills_path
        self.uploads = dict(uploads or {})
        self.upload_mapping = dict(upload_mapping or {})
        self.metrics_path = metrics_path
        self.metrics_prefix = metrics_prefix
        self.metrics_key = metrics_key
        self.metrics_keys = metrics_keys
        self.timeout_seconds = timeout_seconds
        self.environment_vars = dict(environment_vars or {})
        self.keep_sandbox_for_scoring = keep_sandbox_for_scoring
        self.sandbox_wait_for_creation_max_attempts = (
            sandbox_wait_for_creation_max_attempts
        )
        self.sandbox_creations_per_minute = sandbox_creations_per_minute
        self.sandbox_client_max_workers = sandbox_client_max_workers
        self.sandbox_client_max_connections = sandbox_client_max_connections
        self.sandbox_client_max_keepalive_connections = (
            sandbox_client_max_keepalive_connections
        )
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self.max_backoff_seconds = max_backoff_seconds
        self.jitter = jitter
        self.timeouts = timeouts
        self.sandbox_client = _CliSandboxClientProxy(self)
        self.with_retry = _CliSandboxRetryProxy(self)

    def channels(self, task: Task | None = None) -> ChannelMap:
        channels = dict(super().channels(task))
        channels["sandbox"] = {
            "spec": self.sandbox_spec,
            "uploads": self.uploads,
            "runtime": {
                "max_retries": self.max_retries,
                "base_delay": self.base_delay,
                "backoff_factor": self.backoff_factor,
                "max_backoff_seconds": self.max_backoff_seconds,
                "jitter": self.jitter,
                "client_max_workers": self.sandbox_client_max_workers,
                "client_max_connections": self.sandbox_client_max_connections,
                "client_max_keepalive_connections": (
                    self.sandbox_client_max_keepalive_connections
                ),
                "creations_per_minute": self.sandbox_creations_per_minute,
                "timeouts": self.timeouts,
            },
        }
        return channels

    def require_sandbox_runtime(self) -> SandboxResources:
        if self.resources is not None:
            return cast(SandboxResources, self.resources.require("sandbox_runtime"))
        raise RuntimeError("CliHarness requires sandbox resources.")

    async def build_env_vars(
        self, task: Task, state: State, resources: Resources
    ) -> dict[str, str]:
        env_vars = dict(self.environment_vars)
        overlap = self.PROTECTED_ENV_VARS & env_vars.keys()
        if overlap:
            raise ValueError(
                f"environment_vars must not override protected keys: {overlap}"
            )
        env_vars["OPENAI_BASE_URL"] = state["endpoint_base_url"]
        env_vars.setdefault("OPENAI_TIMEOUT", str(int(self.timeout_seconds)))
        env_vars.setdefault("OPENAI_REQUEST_TIMEOUT", str(int(self.timeout_seconds)))
        env_vars.setdefault("HTTPX_TIMEOUT", str(int(self.timeout_seconds)))
        endpoint = cast(Endpoint, resources.require("endpoint"))
        if endpoint.secret:
            env_vars["OPENAI_API_KEY"] = endpoint.secret
        env_vars["OPENAI_MODEL"] = resources.model
        workdir = self.task_workdir(task)
        state["agent_workdir"] = workdir
        env_vars["AGENT_WORKDIR"] = workdir
        env_vars["TASK_INSTRUCTION_PATH"] = self.instruction_path
        if getattr(resources, "system_prompt", ""):
            env_vars["SYSTEM_PROMPT_PATH"] = self.system_prompt_path
        return env_vars

    def resolve_sandbox_spec(self, task: Task, resources: Resources) -> SandboxSpec:
        seed = resources.get("sandbox_request")
        spec = self.sandbox_spec
        if isinstance(seed, SandboxSpec):
            return seed
        if not isinstance(seed, SandboxSeed):
            return spec
        return SandboxSpec(
            image=seed.image or spec.image,
            cpu_cores=seed.cpu_cores or spec.cpu_cores,
            memory_gb=seed.memory_gb or spec.memory_gb,
            disk_size_gb=seed.disk_size_gb or spec.disk_size_gb,
            gpu_count=seed.gpu_count if seed.gpu_count is not None else spec.gpu_count,
            gpu_type=seed.gpu_type or spec.gpu_type,
            vm=seed.vm if seed.vm is not None else spec.vm,
            network_access=(
                seed.network_access
                if seed.network_access is not None
                else spec.network_access
            ),
            timeout_minutes=seed.timeout_minutes or spec.timeout_minutes,
            start_command=seed.start_command or spec.start_command,
            environment_vars={**spec.environment_vars, **seed.environment_vars},
            secrets=seed.secrets or spec.secrets,
            team_id=seed.team_id or spec.team_id,
            advanced_configs=seed.advanced_configs or spec.advanced_configs,
            registry_credentials_id=(
                seed.registry_credentials_id or spec.registry_credentials_id
            ),
            labels=[*spec.labels, *seed.labels],
        )

    async def setup_sandbox(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        from prime_sandboxes import CreateSandboxRequest

        spec = self.resolve_sandbox_spec(task, resources)
        env_vars = await self.build_env_vars(task, state, resources)
        overlap = self.PROTECTED_ENV_VARS & spec.environment_vars.keys()
        if overlap:
            raise ValueError(
                f"task sandbox environment vars must not override protected keys: {overlap}"
            )
        env_vars.update(spec.environment_vars)
        task_name = str(task.info.get("task_name") or resources.taskset.name or "task")
        request = CreateSandboxRequest(
            name=f"{task_name}-{task.example_id}-{uuid.uuid4().hex[:8]}",
            docker_image=spec.image,
            start_command=spec.start_command,
            cpu_cores=spec.cpu_cores,
            memory_gb=spec.memory_gb,
            disk_size_gb=spec.disk_size_gb,
            gpu_count=spec.gpu_count,
            gpu_type=spec.gpu_type,
            vm=spec.vm if spec.vm is not None else spec.gpu_count > 0,
            network_access=spec.network_access,
            timeout_minutes=spec.timeout_minutes,
            environment_vars=env_vars,
            secrets=spec.secrets,
            team_id=spec.team_id,
            advanced_configs=spec.advanced_configs,
            registry_credentials_id=spec.registry_credentials_id,
            labels=spec.labels,
        )
        sandbox_runtime = self.require_sandbox_runtime()
        sandbox_id = await sandbox_runtime.create(
            request,
            max_attempts=self.sandbox_wait_for_creation_max_attempts,
        )
        state["sandbox_id"] = sandbox_id
        await self.setup_sandbox_contents(task, state, resources)

    def task_workdir(self, task: Task) -> str:
        return str(task.info.get("workdir") or self.agent_workdir)

    async def setup_task_sandbox_contents(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        sandbox_seed = resources.get("sandbox_request")
        if isinstance(sandbox_seed, SandboxSeed):
            for host_path, sandbox_path in sandbox_seed.files.items():
                await self.upload_path(
                    state["sandbox_id"], Path(host_path), sandbox_path
                )
            for host_path, sandbox_path in sandbox_seed.mounts.items():
                await self.upload_path(
                    state["sandbox_id"], Path(host_path), sandbox_path
                )
            for command in sandbox_seed.setup_commands:
                result = await self.with_retry(self.sandbox_client.execute_command)(
                    state["sandbox_id"], command, timeout=self.timeouts.extract
                )
                if result.exit_code != 0:
                    output = (result.stdout or "") + (result.stderr or "")
                    raise SandboxError(
                        f"Sandbox setup command failed (exit={result.exit_code}): {output[:500]}"
                    )

    async def setup_sandbox_contents(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        await self.setup_task_sandbox_contents(task, state, resources)
        sandbox_id = state["sandbox_id"]
        workdir = self.task_workdir(task)
        state["agent_workdir"] = workdir
        dirs = {str(Path(self.instruction_path).parent), workdir}
        system_prompt = getattr(resources, "system_prompt", "")
        if system_prompt:
            dirs.add(str(Path(self.system_prompt_path).parent))
        for path in dirs:
            await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id, f"mkdir -p {shlex.quote(path)}"
            )
        instruction = task.info.get("instruction")
        if instruction is None:
            if task.prompt:
                last = task.prompt[-1]
                instruction = (
                    last.get("content") if isinstance(last, dict) else last.content
                )
            else:
                instruction = ""
        await self.upload_content(sandbox_id, str(instruction), self.instruction_path)
        if system_prompt:
            await self.upload_content(
                sandbox_id, system_prompt, self.system_prompt_path
            )
        await self.upload_mapped_directories(task, state, resources)
        await self.install_agent(sandbox_id)
        await self.run_post_install(sandbox_id)

    def effective_upload_mapping(self) -> dict[str, str]:
        mapping = dict(self.upload_mapping)
        if self.skills_path:
            mapping.setdefault("skills", self.skills_path)
        return mapping

    def task_uploads(self, task: Task, resources: Resources) -> dict[str, object]:
        return dict(cast(dict[str, object], resources.require("sandbox_uploads")))

    async def upload_mapped_directories(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        mapping = self.effective_upload_mapping()
        if not mapping:
            return
        sandbox_id = state["sandbox_id"]
        for name, local_source in self.task_uploads(task, resources).items():
            sandbox_path = mapping.get(name)
            if sandbox_path is None:
                continue
            if callable(local_source):
                local_source = local_source()
            if isinstance(local_source, (str, Path)):
                await self.upload_path(sandbox_id, Path(local_source), sandbox_path)
            else:
                with importlib_resources.as_file(
                    cast(Traversable, local_source)
                ) as path:
                    await self.upload_path(sandbox_id, path, sandbox_path)

    async def install_agent(self, sandbox_id: str) -> None:
        if not self.install_command:
            return
        kwargs: dict[str, object] = {"timeout": self.install_timeout}
        if self.install_env:
            kwargs["env"] = self.install_env
        result = await self.with_retry(self.sandbox_client.execute_command)(
            sandbox_id,
            self.install_command,
            **kwargs,
        )
        if result.exit_code != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise SandboxError(
                f"Agent install failed (exit={result.exit_code}): {output[:500]}"
            )

    async def run_post_install(self, sandbox_id: str) -> None:
        for sandbox_path, content in self.post_install_uploads.items():
            await self.upload_content(sandbox_id, content, sandbox_path)
        if not self.post_install_command:
            return
        result = await self.with_retry(self.sandbox_client.execute_command)(
            sandbox_id,
            self.post_install_command,
            timeout=self.install_timeout,
        )
        if result.exit_code != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise SandboxError(
                f"Post-install failed (exit={result.exit_code}): {output[:500]}"
            )

    async def upload_path(
        self, sandbox_id: str, host_path: Path, sandbox_path: str
    ) -> None:
        if host_path.is_file():
            await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id,
                f"mkdir -p {shlex.quote(str(Path(sandbox_path).parent))}",
            )
            await self.with_retry(self.sandbox_client.upload_file)(
                sandbox_id, sandbox_path, str(host_path)
            )
            return
        if host_path.is_dir():
            await self.upload_directory(sandbox_id, host_path, sandbox_path)
            return
        raise FileNotFoundError(f"Cannot upload missing path: {host_path}")

    async def upload_content(
        self, sandbox_id: str, content: str, sandbox_path: str
    ) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(content)
            local_path = f.name
        try:
            await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id,
                f"mkdir -p {shlex.quote(str(Path(sandbox_path).parent))}",
            )
            await self.with_retry(self.sandbox_client.upload_file)(
                sandbox_id, sandbox_path, local_path
            )
        finally:
            await asyncio.to_thread(Path(local_path).unlink, missing_ok=True)

    async def upload_directory(
        self, sandbox_id: str, host_path: Path, sandbox_path: str
    ) -> None:
        def build_archive() -> str:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(host_path, arcname=".")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as f:
                f.write(buf.getvalue())
                return f.name

        archive = await asyncio.to_thread(build_archive)
        archive_remote = f"{sandbox_path.rstrip('/')}/_upload.tar.gz"
        try:
            quoted_dir = shlex.quote(sandbox_path)
            await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id,
                f"mkdir -p {quoted_dir}",
            )
            await self.with_retry(self.sandbox_client.upload_file)(
                sandbox_id, archive_remote, archive
            )
            quoted_archive = shlex.quote(archive_remote)
            result = await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id,
                f"tar -xzf {quoted_archive} -C {quoted_dir} && rm -f {quoted_archive}",
                timeout=self.timeouts.extract,
            )
            if result.exit_code != 0:
                raise SandboxError(
                    f"Failed to extract upload into {sandbox_path}: {result.stderr}"
                )
        finally:
            await asyncio.to_thread(Path(archive).unlink, missing_ok=True)

    async def execute(
        self,
        task: Task,
        state: State,
        resources: Resources,
        client: AsyncOpenAI,
    ) -> object:
        await self.setup_sandbox(task, state, resources)
        try:
            job = await self.with_retry(self.sandbox_client.start_background_job)(
                state["sandbox_id"], self.command
            )
        except Exception as e:
            raise SandboxError(f"Failed to start agent: {e}") from e
        state["background_job"] = str(job)
        state["agent_start_time"] = time.time()
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            try:
                status = await self.with_retry(self.sandbox_client.get_background_job)(
                    state["sandbox_id"], job, timeout=self.timeouts.poll
                )
            except SandboxOOMError as e:
                state["sandbox_oom"] = True
                raise SandboxError("Sandbox OOM while polling agent job.") from e
            except SandboxTimeoutError as e:
                state["sandbox_timeout"] = True
                raise SandboxError("Sandbox timed out while polling agent job.") from e
            if status.completed:
                state["agent_exit_code"] = status.exit_code
                state["agent_stdout"] = status.stdout
                state["agent_stderr"] = status.stderr
                if status.exit_code:
                    raise InfraError(
                        f"Sandbox agent exited with code {status.exit_code}: {status.stderr}"
                    )
                state["agent_completed"] = True
                return status
            await asyncio.sleep(self.poll_interval)
        state["agent_timed_out"] = True
        state["stop_condition"] = "timeout_reached"
        raise InfraError(f"Sandbox agent timed out after {self.timeout_seconds}s")

    async def finalize_state(
        self, task: Task, state: State, resources: Resources
    ) -> State:
        sandbox_id = state.get("sandbox_id")
        if sandbox_id and self.log_path and "agent_logs" not in state:
            try:
                log_path = shlex.quote(self.log_path)
                result = await self.with_retry(self.sandbox_client.execute_command)(
                    sandbox_id,
                    f"cat {log_path} 2>/dev/null || echo '<no logs>'",
                    working_dir=None,
                )
                state["agent_logs"] = (result.stdout or "").strip()
            except Exception as e:
                self.logger.warning(f"Failed to collect agent logs: {e}")
        if sandbox_id and self.metrics_path:
            await self.collect_harness_metrics(sandbox_id, state)
        return await super().finalize_state(task, state, resources)

    async def collect_harness_metrics(self, sandbox_id: str, state: State) -> None:
        if not self.metrics_path:
            return
        metrics_glob = self.metrics_path.format(
            workdir=state.get("agent_workdir") or self.agent_workdir
        )
        try:
            result = await self.with_retry(self.sandbox_client.execute_command)(
                sandbox_id,
                f"f=$(ls {metrics_glob} 2>/dev/null | head -1) "
                '&& cat "$f" || echo "{}"',
                working_dir=None,
            )
            data = json.loads((result.stdout or "{}").strip())
            if self.metrics_key:
                data = data.get(self.metrics_key, {})
            harness_metrics = state.get("harness_metrics")
            if not isinstance(harness_metrics, dict):
                harness_metrics = {}
                state["harness_metrics"] = harness_metrics
            for key, value in data.items():
                if self.metrics_keys is None or key in self.metrics_keys:
                    prefixed_key = f"{self.metrics_prefix}{key}"
                    state[prefixed_key] = value
                    if isinstance(value, int | float):
                        harness_metrics[prefixed_key] = float(value)
        except Exception as e:
            self.logger.warning(f"Failed to collect harness metrics: {e}")

    @cleanup
    async def cleanup_sandbox(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        sandbox_runtime = self.require_sandbox_runtime()
        sandbox_scoring = bool(resources.require("sandbox_scoring"))
        if (self.keep_sandbox_for_scoring or sandbox_scoring) and state.get(
            "is_completed"
        ):
            state["sandbox_retained_for_scoring"] = True
            sandbox_runtime.retain_for_scoring(sandbox_id)
            return
        try:
            await sandbox_runtime.delete(sandbox_id)
        except Exception as e:
            self.logger.warning(f"Failed to delete sandbox {sandbox_id}: {e}")

    @stop
    async def timeout_reached(
        self, task: Task, state: State, resources: Resources
    ) -> bool:
        elapsed = time.time() - state["timing"]["start_time"]
        if elapsed <= self.timeout_seconds:
            return False
        state["agent_timed_out"] = True
        if state.get("error") is None:
            state["error"] = error_info(
                InfraError(f"Sandbox agent timed out after {self.timeout_seconds}s")
            )
        return True


class _CliSandboxClientProxy:
    def __init__(self, harness: CliHarness):
        self.harness = harness

    def __getattr__(self, name: str):
        return getattr(self.harness.require_sandbox_runtime().client, name)


class _CliSandboxRetryProxy:
    def __init__(self, harness: CliHarness):
        self.harness = harness

    def __call__(self, func):
        return self.harness.require_sandbox_runtime().with_retry(func)
