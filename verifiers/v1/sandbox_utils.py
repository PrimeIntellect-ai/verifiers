from __future__ import annotations

import json
import shlex
import uuid
from collections.abc import Mapping
from typing import Any, cast

from verifiers.errors import SandboxError
from verifiers.utils.async_utils import maybe_call_with_named_args

from .program_utils import command_argv, command_env, float_config, int_config
from .runtime import Runtime
from .state import State
from .task import Task


class SandboxLease:
    def __init__(
        self,
        client: object,
        sandbox_id: str,
        scope: str,
        key: str,
    ):
        self.client = client
        self.id = sandbox_id
        self.scope = scope
        self.key = key
        self.deleted = False

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> object:
        result = await maybe_call_with_named_args(
            getattr(self.client, "execute_command"),
            sandbox_id=self.id,
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
        )
        return result

    async def upload_bytes(
        self, path: str, content: bytes, filename: str | None = None
    ) -> object:
        return await maybe_call_with_named_args(
            getattr(self.client, "upload_bytes"),
            sandbox_id=self.id,
            file_path=path,
            file_bytes=content,
            filename=filename or path.rsplit("/", 1)[-1] or "file",
        )

    async def read_file(self, path: str) -> object:
        return await maybe_call_with_named_args(
            getattr(self.client, "read_file"),
            sandbox_id=self.id,
            path=path,
        )

    async def delete(self) -> None:
        if self.deleted:
            return
        self.deleted = True
        try:
            await cast(Any, self.client).delete(self.id)
        finally:
            aclose = getattr(self.client, "aclose", None)
            if callable(aclose):
                await aclose()


class SandboxHandle:
    def __init__(self, lease: SandboxLease, state: State):
        self.lease = lease
        self.state = state
        self.id = lease.id
        self.scope = lease.scope
        self.key = lease.key
        attach_tool_sandbox_ref(state, lease)

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> object:
        result = await self.lease.execute(
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
        )
        record_tool_sandbox_command(self.state, self.lease, command, result)
        return result

    async def upload_bytes(
        self, path: str, content: bytes, filename: str | None = None
    ) -> object:
        return await self.lease.upload_bytes(path, content, filename)

    async def read_file(self, path: str) -> object:
        return await self.lease.read_file(path)

    async def delete(self) -> None:
        await self.lease.delete()


async def ensure_tool_sandbox(
    toolset: object, task: Task, state: State
) -> SandboxHandle:
    _ = task
    lease = await create_tool_sandbox_lease(toolset)
    return SandboxHandle(lease, state)


async def create_tool_sandbox_lease(toolset: object) -> SandboxLease:
    sandbox_config = getattr(toolset, "sandbox", None)
    if not isinstance(sandbox_config, Mapping):
        raise TypeError("Toolset sandbox must be a mapping.")
    key = tool_sandbox_key(toolset)
    from prime_sandboxes import AsyncSandboxClient

    scope = sandbox_scope(sandbox_config)
    client = AsyncSandboxClient()
    sandbox_id = await create_sandbox(client, sandbox_config)
    lease = SandboxLease(client, sandbox_id, scope, key)
    await setup_tool_sandbox(lease, sandbox_config)
    return lease


async def run_sandbox_command(
    program: Mapping[str, object],
    sandbox_config: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
) -> State:
    from prime_sandboxes import AsyncSandboxClient

    scope = sandbox_scope(sandbox_config)
    client = AsyncSandboxClient()
    sandbox_id: str | None = None
    succeeded = False
    try:
        sandbox_id = await create_sandbox(client, sandbox_config)
        state["sandbox_id"] = sandbox_id
        state.setdefault("runtime", {})
        state["runtime"]["sandbox"] = {"id": sandbox_id, "scope": scope}
        await upload_program_files(client, sandbox_id, program, task, state, runtime)
        workdir = cast(str | None, sandbox_config.get("workdir"))
        if workdir:
            await client.execute_command(sandbox_id, f"mkdir -p {shlex.quote(workdir)}")
        argv = await command_argv(program, task, state, runtime)
        env = await command_env(program, task, state, runtime, include_base=False)
        result = await client.execute_command(
            sandbox_id,
            shlex.join(argv),
            working_dir=workdir,
            env=env,
            timeout=cast(int | None, sandbox_config.get("command_timeout")),
        )
        state["command"] = {
            "argv": argv,
            "returncode": result.exit_code,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
        state["completion"] = [
            {"role": "assistant", "content": state["command"]["stdout"].strip()}
        ]
        if result.exit_code:
            raise SandboxError(
                f"Sandbox command exited with {result.exit_code}: {result.stderr}"
            )
        await collect_sandbox_artifacts(client, sandbox_id, program, state)
        succeeded = True
        return state
    finally:
        if sandbox_id is not None:
            if scope == "rollout" or not succeeded:
                await client.delete(sandbox_id)
            await client.aclose()


async def release_group_sandboxes(states: list[State]) -> None:
    from prime_sandboxes import AsyncSandboxClient

    sandbox_ids: set[str] = set()
    for state in states:
        sandbox = state.get("runtime", {}).get("sandbox")
        if not isinstance(sandbox, Mapping):
            continue
        if sandbox.get("scope") != "group":
            continue
        sandbox_id = sandbox.get("id")
        if isinstance(sandbox_id, str):
            sandbox_ids.add(sandbox_id)
    if not sandbox_ids:
        return
    client = AsyncSandboxClient()
    try:
        for sandbox_id in sandbox_ids:
            await client.delete(sandbox_id)
    finally:
        await client.aclose()


async def create_sandbox(client: object, sandbox_config: Mapping[str, object]) -> str:
    from prime_sandboxes import CreateSandboxRequest

    request = CreateSandboxRequest(
        name=f"vf-v1-{uuid.uuid4().hex[:8]}",
        docker_image=str(sandbox_config.get("image") or "python:3.11-slim"),
        start_command=str(sandbox_config.get("start_command") or "tail -f /dev/null"),
        cpu_cores=float_config(sandbox_config, "cpu_cores", 1.0),
        memory_gb=float_config(sandbox_config, "memory_gb", 2.0),
        disk_size_gb=float_config(sandbox_config, "disk_size_gb", 5.0),
        gpu_count=int_config(sandbox_config, "gpu_count", 0),
        network_access=bool(sandbox_config.get("network_access", True)),
        timeout_minutes=int_config(sandbox_config, "timeout_minutes", 60),
    )
    sandbox = await cast(Any, client).create(request)
    sandbox_id = str(sandbox.id)
    await cast(Any, client).wait_for_creation(sandbox_id)
    return sandbox_id


async def setup_tool_sandbox(
    handle: SandboxLease, sandbox_config: Mapping[str, object]
) -> None:
    packages = sandbox_config.get("packages") or []
    if isinstance(packages, str):
        packages = shlex.split(packages)
    if packages:
        if not isinstance(packages, list):
            raise TypeError("sandbox.packages must be a list or string.")
        package_args = " ".join(shlex.quote(str(package)) for package in packages)
        result = cast(
            Any,
            await handle.execute(
                f"python -m pip install --disable-pip-version-check {package_args}",
                timeout=int_config(sandbox_config, "install_timeout", 300),
            ),
        )
        if result.exit_code:
            raise SandboxError(f"Sandbox package install failed: {result.stderr}")
    commands = sandbox_config.get("setup_commands") or []
    if isinstance(commands, str):
        commands = [commands]
    if not isinstance(commands, list):
        raise TypeError("sandbox.setup_commands must be a list or string.")
    for command in commands:
        result = cast(
            Any,
            await handle.execute(
                str(command), timeout=int_config(sandbox_config, "setup_timeout", 300)
            ),
        )
        if result.exit_code:
            raise SandboxError(f"Sandbox setup command failed: {result.stderr}")


def sandbox_scope(sandbox_config: Mapping[str, object]) -> str:
    scope = str(sandbox_config.get("scope") or "rollout")
    if scope not in {"rollout", "group", "global"}:
        raise ValueError("sandbox.scope must be 'rollout', 'group', or 'global'.")
    return scope


def attach_tool_sandbox_ref(state: State, lease: SandboxLease) -> None:
    state.setdefault("runtime", {})
    tool_sandboxes = state["runtime"].setdefault("tool_sandboxes", {})
    if not isinstance(tool_sandboxes, dict):
        raise TypeError("state.runtime.tool_sandboxes must be a mapping.")
    tool_sandboxes[lease.key] = {"id": lease.id, "scope": lease.scope}


def record_tool_sandbox_command(
    state: State, lease: SandboxLease, command: str, result: object
) -> None:
    result = cast(Any, result)
    command_record = {
        "command": command,
        "returncode": result.exit_code,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
    state.setdefault("sandbox_commands", []).append(command_record)
    state.setdefault("runtime", {})
    tool_sandboxes = state["runtime"].setdefault("tool_sandboxes", {})
    if isinstance(tool_sandboxes, dict):
        tool_state = tool_sandboxes.setdefault(
            lease.key, {"id": lease.id, "scope": lease.scope}
        )
        if isinstance(tool_state, dict):
            tool_state.setdefault("commands", []).append(command_record)


def tool_sandbox_key(toolset: object) -> str:
    from .toolset import flatten_toolsets, tool_name

    names = [tool_name(tool) for tool in flatten_toolsets((toolset,))]
    if names:
        return "tools:" + ",".join(sorted(names))
    return f"toolset:{id(toolset)}"


async def upload_program_files(
    client: object,
    sandbox_id: str,
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    files = program.get("files", {})
    if not isinstance(files, Mapping):
        raise TypeError("program.files must be a mapping.")
    for path, source in files.items():
        if not isinstance(path, str):
            raise TypeError("program.files keys must be strings.")
        from .program_utils import resolve_program_value

        content = await resolve_program_value(source, task, state, runtime)
        if not isinstance(content, str):
            content = str(content)
        await maybe_call_with_named_args(
            getattr(client, "upload_bytes"),
            sandbox_id=sandbox_id,
            file_path=path,
            file_bytes=content.encode(),
            filename=path.rsplit("/", 1)[-1] or "file",
        )


async def collect_sandbox_artifacts(
    client: object, sandbox_id: str, program: Mapping[str, object], state: State
) -> None:
    artifacts = program.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise TypeError("program.artifacts must be a mapping.")
    if not artifacts:
        return
    state.setdefault("artifacts", {})
    for name, spec in artifacts.items():
        if not isinstance(name, str):
            raise TypeError("program.artifacts keys must be strings.")
        if not isinstance(spec, Mapping):
            raise TypeError("program.artifacts values must be mappings.")
        spec = cast(Mapping[str, object], spec)
        path = spec.get("path")
        if not isinstance(path, str):
            raise TypeError("program artifact path must be a string.")
        result = await maybe_call_with_named_args(
            getattr(client, "execute_command"),
            sandbox_id=sandbox_id,
            command=f"cat {shlex.quote(path)}",
        )
        if result.exit_code:
            raise FileNotFoundError(f"Sandbox artifact not found: {path}")
        content = result.stdout or ""
        artifact_format = spec.get("format", "text")
        if artifact_format == "json":
            value: object = json.loads(content)
        elif artifact_format == "text":
            value = content
        else:
            raise ValueError(f"Unsupported artifact format: {artifact_format!r}")
        state["artifacts"][name] = value
