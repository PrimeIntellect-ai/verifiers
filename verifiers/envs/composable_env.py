"""ComposableEnv delegates task behavior to a TaskSet and agent wiring to a Harness."""

from __future__ import annotations

import asyncio
import importlib.resources as resources
import inspect
import json
import logging
import shlex
import tarfile
import tempfile
from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast, get_type_hints

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, create_model
import verifiers as vf
from verifiers.envs.composable_skills import TaskSkills
from verifiers.envs.composable_tools import TaskTools
from verifiers.envs.experimental.cli_agent_env import CliAgentEnv
from verifiers.envs.tool_env import ToolMonitorRubric
from verifiers.types import State
from verifiers.utils.file_locks import shared_path_lock

if TYPE_CHECKING:
    from harnesses import Harness
    from tasksets import TaskSet

__all__ = [
    "ComposableEnv",
    "NamedComposableEnv",
    "build_composable_harness",
    "normalize_harness_config",
    "resolve_harness_workdir",
]

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


class ComposableEnvArgs(BaseModel):
    """TOML-friendly args for a named composable environment."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    max_turns: int = -1
    timeout_seconds: float = 3600.0
    start_command: str = "tail -f /dev/null"
    environment_vars: dict[str, str] | None = None
    team_id: str | None = None
    advanced_configs: Any | None = None
    labels: list[str] | None = None
    env_id: str | None = None
    keep_sandbox_for_scoring: bool | None = None
    harness: Any | None = None
    harness_config: dict[str, Any] | None = None
    agent_workdir: str | None = None


def normalize_harness_config(
    harness: Any | None,
    harness_config: dict[str, Any] | None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge TOML-friendly harness config from env args."""
    resolved = dict(harness_config or {})
    if isinstance(harness, str):
        resolved.setdefault("harness", harness)
    elif isinstance(harness, dict):
        resolved.update(harness)
    for key, value in (extra_config or {}).items():
        resolved.setdefault(key, value)
    return resolved


def resolve_harness_workdir(
    harness_config: dict[str, Any],
    default_workdir: str,
) -> str:
    """Extract the agent workdir from flat or nested harness config."""
    workdir_config = harness_config
    agent_config = workdir_config.get("agent")
    if isinstance(agent_config, dict) and isinstance(agent_config.get("harness"), dict):
        workdir_config = agent_config["harness"]
    for key in ("agent_workdir", "cwd", "workdir"):
        if workdir_config.get(key):
            return str(workdir_config[key])
    return default_workdir


def build_composable_harness(
    harness: Any | None,
    harness_config: dict[str, Any],
    *,
    agent_workdir: str,
    default_config: dict[str, Any] | None = None,
) -> Any:
    """Build a harness while preserving object pass-through semantics."""
    from harnesses import build_harness_from_config

    source = (
        harness
        if harness is not None and not isinstance(harness, str | dict)
        else harness_config or default_config
    )
    return build_harness_from_config(source, agent_workdir=agent_workdir)


def composable_taskset_arg_fields(
    taskset_cls: type[Any],
) -> list[tuple[str, Any, Any]]:
    type_hints = get_type_hints(taskset_cls.__init__)
    fields = []
    for key, param in inspect.signature(taskset_cls.__init__).parameters.items():
        if (
            key == "self"
            or key in ComposableEnvArgs.model_fields
            or param.kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ):
            continue

        annotation = type_hints.get(key, Any)
        default = ... if param.default == inspect.Parameter.empty else param.default
        if key == "max_examples":
            default = Field(
                default=default,
                validation_alias=AliasChoices("max_examples", "limit"),
            )
        fields.append((key, annotation, default))
    return fields


def build_composable_env_args_model(
    taskset_cls: type[Any],
) -> tuple[type[ComposableEnvArgs], set[str]]:
    fields = {
        key: (annotation, default)
        for key, annotation, default in composable_taskset_arg_fields(taskset_cls)
    }
    args_model = create_model(  # ty: ignore[no-matching-overload]
        f"{taskset_cls.__name__}EnvironmentArgs",
        __base__=ComposableEnvArgs,
        __config__=ConfigDict(
            arbitrary_types_allowed=True,
            extra="allow",
            populate_by_name=True,
        ),
        **fields,
    )
    return cast(type[ComposableEnvArgs], args_model), set(fields)


class ComposableEnv(CliAgentEnv):
    """CliAgentEnv that delegates to a TaskSet and a Harness."""

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

    def _get_harness(self, state: State) -> Harness:
        return state.get("_harness") or self.harness

    def _get_runtime_spec(self, state: State) -> Any:
        cached = state.get("_runtime_spec")
        if cached is not None:
            return cached
        info = state.get("info") or {}
        runtime = self.taskset.get_runtime_spec(info)
        state["_runtime_spec"] = runtime
        state["_sandbox_spec"] = runtime.sandbox
        return runtime

    def _get_spec(self, state: State) -> Any:
        return self._get_runtime_spec(state).sandbox

    async def get_docker_image(self, state: State) -> str:
        spec = self._get_spec(state)
        if spec:
            return spec.image
        return self.docker_image

    def get_sandbox_start_command(self, state: State) -> str:
        spec = self._get_spec(state)
        if spec:
            return spec.start_command
        return super().get_sandbox_start_command(state)

    def get_sandbox_resources(self, state: State) -> dict[str, Any]:
        spec = self._get_spec(state)
        if spec:
            return {
                "cpu_cores": spec.cpu_cores,
                "memory_gb": spec.memory_gb,
                "disk_size_gb": spec.disk_size_gb,
                "gpu_count": spec.gpu_count,
                "gpu_type": spec.gpu_type,
                "vm": spec.vm if spec.vm is not None else spec.gpu_count > 0,
                "timeout_minutes": spec.timeout_minutes,
            }
        return super().get_sandbox_resources(state)

    def get_agent_timeout_seconds(self, state: State) -> float:
        timeout = self._get_runtime_spec(state).agent_timeout_seconds
        return (
            float(timeout)
            if timeout is not None
            else super().get_agent_timeout_seconds(state)
        )

    async def build_env_vars(self, state: State) -> dict[str, str]:
        env_vars = await super().build_env_vars(state)
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
        runtime = self._get_runtime_spec(state)
        task_env_vars = {}
        if runtime.sandbox:
            task_env_vars.update(runtime.sandbox.environment_vars)
        task_env_vars.update(runtime.env_vars)
        if task_env_vars:
            conflicts = (
                self.PROTECTED_ENV_VARS | {"AGENT_WORKDIR"}
            ) & task_env_vars.keys()
            if conflicts:
                raise ValueError(
                    f"Task runtime env vars must not override protected keys: {conflicts}."
                )
            env_vars.update(task_env_vars)
        env_vars["AGENT_WORKDIR"] = runtime.workdir
        return env_vars

    async def build_agent_env_vars(self, state: State) -> dict[str, str]:
        env_vars = await super().build_agent_env_vars(state)
        tools = state.get("_task_tools")
        if not isinstance(tools, TaskTools) or not tools.env_vars:
            return env_vars

        conflicts = (
            self.PROTECTED_ENV_VARS | {"AGENT_WORKDIR"}
        ) & tools.env_vars.keys()
        if conflicts:
            raise ValueError(
                f"TaskTools.env_vars must not override protected keys: {conflicts}."
            )
        env_vars.update(tools.env_vars)
        return env_vars

    async def get_run_command(self, state: State) -> str:
        return self._get_harness(state).run_command

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
        state["_harness"] = self.harness
        await self._prepare_task_tools(state)
        await self._prepare_task_skills(state)
        await self._create_harness_input_dirs(state)
        await self._upload_harness_inputs(sandbox_id, state)
        await self._after_harness_inputs_uploaded(state)
        await self._install_agent(state)
        await self._run_post_install(state)

    async def post_rollout(self, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        harness = self._get_harness(state)
        if sandbox_id and harness.log_path and "agent_logs" not in state:
            try:
                log_path = shlex.quote(harness.log_path)
                result = await self.sandbox_client.execute_command(
                    sandbox_id,
                    f"cat {log_path} 2>/dev/null || echo '<no logs>'",
                    working_dir=None,
                )
                state["agent_logs"] = (result.stdout or "").strip()
            except Exception as e:
                self.logger.warning(f"Failed to collect agent logs: {e}")

        if sandbox_id and harness.metrics_path:
            await self._collect_harness_metrics(sandbox_id, state)

        await super().post_rollout(state)

    async def _populate_sandbox_context(self, state: State) -> None:
        state["sandbox_client"] = self.sandbox_client
        runtime = self._get_runtime_spec(state)
        spec = runtime.sandbox
        test_timeout = runtime.test_timeout_seconds
        state["test_timeout"] = (
            float(test_timeout)
            if test_timeout is not None
            else spec.timeout_minutes * 60
            if spec
            else 900
        )

    async def _prepare_task_tools(self, state: State) -> None:
        runtime = self._get_runtime_spec(state)
        tools = runtime.tools
        tools = await self.taskset.prepare_tools(state, tools)
        state["_task_tools"] = tools
        state["_harness"] = self._get_harness(state).with_tools(tools)

    async def _prepare_task_skills(self, state: State) -> None:
        runtime = self._get_runtime_spec(state)
        skills = runtime.skills
        skills = await self.taskset.prepare_skills(state, skills)
        if skills.source_dir and not skills.skills_dir:
            mapping = self._get_harness(state).get_effective_upload_dir_mapping() or {}
            if mapping.get("skills"):
                skills.skills_dir = mapping["skills"]
            else:
                raise ValueError(
                    "Task skills require a harness skills_path or "
                    "upload_dir_mapping['skills']."
                )
        state["_task_skills"] = skills
        state["_harness"] = self._get_harness(state).with_skills(skills)

    async def _create_harness_input_dirs(self, state: State) -> None:
        sandbox_id = state["sandbox_id"]
        harness = self._get_harness(state)
        dirs = {harness.instruction_path.rsplit("/", 1)[0]}
        if harness.system_prompt:
            dirs.add(harness.system_prompt_path.rsplit("/", 1)[0])
        mkdir_args = " ".join(shlex.quote(path) for path in sorted(dirs))
        await self.sandbox_client.execute_command(
            sandbox_id, f"mkdir -p {mkdir_args}", timeout=self.timeouts.mkdir
        )

    async def _upload_harness_inputs(self, sandbox_id: str, state: State) -> None:
        info = state.get("info") or {}
        harness = self._get_harness(state)
        instruction = self.taskset.get_instruction(info)
        if instruction.strip():
            await self.upload_content(sandbox_id, instruction, harness.instruction_path)

        if harness.system_prompt:
            await self.upload_content(
                sandbox_id, harness.system_prompt, harness.system_prompt_path
            )

    async def _after_harness_inputs_uploaded(self, state: State) -> None:
        upload_dirs = self._get_upload_dirs(state)
        mapping = self._get_harness(state).get_effective_upload_dir_mapping()
        if not upload_dirs or not mapping:
            return

        sandbox_id = state["sandbox_id"]
        for name, local_source in upload_dirs.items():
            remote_dest = mapping.get(name)
            if remote_dest is not None:
                await self._upload_dir(sandbox_id, local_source, remote_dest)

    def _get_upload_dirs(
        self, state: State | None = None
    ) -> dict[str, Traversable | Path]:
        lookup_state = state if state is not None else cast(State, {})
        runtime = self._get_runtime_spec(lookup_state)
        upload_dirs = dict(runtime.upload_dirs or {})
        skills = runtime.skills
        if isinstance(skills, TaskSkills) and skills.source_dir is not None:
            if "skills" in upload_dirs:
                raise ValueError(
                    "Upload directory name 'skills' is reserved for TaskSkills."
                )
            upload_dirs["skills"] = skills.source_dir
        harness = self._get_harness(lookup_state)
        harness_upload_dirs = (
            harness.get_upload_dirs() if harness.get_upload_dirs else None
        )
        harness_dirs = dict(harness_upload_dirs or {})
        duplicate_names = sorted(set(upload_dirs) & set(harness_dirs))
        if duplicate_names:
            names = ", ".join(repr(name) for name in duplicate_names)
            raise ValueError(
                "Upload directory names must be unique across task and harness; "
                f"duplicates: {names}."
            )
        upload_dirs.update(harness_dirs)
        return upload_dirs

    def _get_install_execute_kwargs(self, state: State) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self._get_harness(state).install_timeout}
        if self.install_env:
            kwargs["env"] = self.install_env
        return kwargs

    async def _install_agent(self, state: State) -> None:
        sandbox_id = state["sandbox_id"]
        harness = self._get_harness(state)
        if not harness.install_script:
            return

        self.logger.debug(f"Installing agent in sandbox {sandbox_id}")
        result = await self.sandbox_client.execute_command(
            sandbox_id,
            harness.install_script,
            **self._get_install_execute_kwargs(state),
        )
        if result.exit_code != 0:
            output = (result.stdout or "") + (result.stderr or "")
            raise vf.SandboxError(
                f"Agent install failed (exit={result.exit_code}): {output[:500]}"
            )

    async def _run_post_install(self, state: State) -> None:
        """Upload harness ``post_install_uploads`` and run ``post_install_script``.

        Runs after ``_install_agent`` so harnesses can layer small assets
        on top of a fully-installed agent (e.g. RLM uploads its
        ``/usr/local/bin/git`` refusal shim and chmods it executable).
        Uses the single-file upload path — not ``_upload_dir`` — because
        these are small, harness-computed blobs of content rather than
        local directories on disk.
        """
        sandbox_id = state["sandbox_id"]
        harness = self._get_harness(state)
        uploads = harness.post_install_uploads
        if uploads:
            for remote_path, content in uploads.items():
                await self.upload_content(sandbox_id, content, remote_path)

        if harness.post_install_script:
            self.logger.debug(f"Running post-install script in sandbox {sandbox_id}")
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                harness.post_install_script,
                **self._get_install_execute_kwargs(state),
            )
            if result.exit_code != 0:
                output = (result.stdout or "") + (result.stderr or "")
                raise vf.SandboxError(
                    f"Post-install failed (exit={result.exit_code}): {output[:500]}"
                )

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

    async def _collect_harness_metrics(self, sandbox_id: str, state: State) -> None:
        harness = self._get_harness(state)
        if not harness.metrics_path:
            return

        workdir = self._get_runtime_spec(state).workdir
        metrics_glob = harness.metrics_path.format(workdir=workdir)
        try:
            result = await self.sandbox_client.execute_command(
                sandbox_id,
                f"f=$(ls {metrics_glob} 2>/dev/null | head -1) "
                '&& cat "$f" || echo "{}"',
                working_dir=None,
            )
            data = json.loads((result.stdout or "{}").strip())
            if harness.metrics_key:
                data = data.get(harness.metrics_key, {})
            prefix = harness.metrics_prefix
            allowed = harness.metrics_keys
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


class NamedComposableEnv(ComposableEnv):
    """ComposableEnv that maps flat TOML/CLI args onto a declared TaskSet."""

    taskset_cls: ClassVar[type[Any]]
    env_id: ClassVar[str]
    default_agent_workdir: ClassVar[str]
    default_harness_config: ClassVar[dict[str, Any]]
    keep_sandbox_for_scoring: ClassVar[bool]
    args_model: ClassVar[type[ComposableEnvArgs]]
    taskset_arg_names: ClassVar[set[str]]

    def __init_subclass__(
        cls,
        *,
        taskset: type[Any],
        env_id: str | None = None,
        default_agent_workdir: str | None = None,
        default_harness_config: dict[str, Any] | None = None,
        keep_sandbox_for_scoring: bool = True,
        **kwargs: Any,
    ):
        super().__init_subclass__(**kwargs)
        cls.taskset_cls = taskset
        cls.env_id = env_id or getattr(taskset, "env_id")
        cls.default_agent_workdir = default_agent_workdir or getattr(
            taskset, "default_workdir", "/app"
        )
        cls.default_harness_config = dict(default_harness_config or {})
        cls.keep_sandbox_for_scoring = keep_sandbox_for_scoring
        cls.args_model, cls.taskset_arg_names = build_composable_env_args_model(taskset)

    def __init__(self, **env_args: Any):
        args = self.args_model.model_validate(env_args)
        env_kwargs: dict[str, Any] = {
            "env_id": args.env_id or self.env_id,
            "max_turns": args.max_turns,
            "timeout_seconds": args.timeout_seconds,
            "start_command": args.start_command,
            "environment_vars": args.environment_vars,
            "team_id": args.team_id,
            "advanced_configs": args.advanced_configs,
            "labels": args.labels,
            "keep_sandbox_for_scoring": (
                self.keep_sandbox_for_scoring
                if args.keep_sandbox_for_scoring is None
                else args.keep_sandbox_for_scoring
            ),
        }
        harness_config = normalize_harness_config(
            args.harness,
            args.harness_config,
            dict(args.model_extra or {}),
        )
        agent_workdir = str(args.agent_workdir or self.default_agent_workdir)
        agent_workdir = resolve_harness_workdir(harness_config, agent_workdir)

        taskset_args = args.model_dump(
            include=self.taskset_arg_names,
            exclude_unset=True,
        )
        taskset_params = inspect.signature(self.taskset_cls.__init__).parameters
        if "agent_workdir" in taskset_params:
            taskset_args["agent_workdir"] = agent_workdir
        if "harness_config" in taskset_params:
            taskset_args["harness_config"] = harness_config

        taskset = self.taskset_cls(**taskset_args)
        harness = build_composable_harness(
            args.harness,
            harness_config,
            agent_workdir=agent_workdir,
            default_config=getattr(taskset, "harness_config", None)
            or self.default_harness_config,
        )
        super().__init__(taskset=taskset, harness=harness, **env_kwargs)
