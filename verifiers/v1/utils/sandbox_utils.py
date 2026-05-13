from __future__ import annotations

import asyncio
import hashlib
import importlib.resources as resources
import json
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, cast

from verifiers.errors import SandboxError
from verifiers.utils.async_utils import maybe_call_with_named_args

from .artifact_utils import artifact_format, artifact_key, artifact_optional
from .artifact_utils import artifact_path
from .program_utils import command_argv, command_env, float_config, int_config
from .program_utils import program_tools_setup, resolve_program_value
from .program_utils import validate_program_bindings
from ..runtime import Runtime
from ..state import State
from ..task import Task

VF_STATE_INPUT_PATH_KEY = "_vf_state_input_path"


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
        self.lock = asyncio.Lock()

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

    async def upload_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await maybe_call_with_named_args(
            getattr(self.client, "upload_file"),
            sandbox_id=self.id,
            file_path=path,
            local_file_path=local_path,
            timeout=timeout,
        )

    async def download_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await maybe_call_with_named_args(
            getattr(self.client, "download_file"),
            sandbox_id=self.id,
            file_path=path,
            local_file_path=local_path,
            timeout=timeout,
        )

    async def read_file(self, path: str) -> object:
        return await maybe_call_with_named_args(
            getattr(self.client, "read_file"),
            sandbox_id=self.id,
            path=path,
        )

    async def run_background_job(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> object:
        return await maybe_call_with_named_args(
            getattr(self.client, "run_background_job"),
            sandbox_id=self.id,
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
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
        attach_sandbox_ref(state, lease)

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

    async def upload_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await self.lease.upload_file(path, local_path, timeout)

    async def download_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await self.lease.download_file(path, local_path, timeout)

    async def read_file(self, path: str) -> object:
        return await self.lease.read_file(path)

    async def run_background_job(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> object:
        result = await self.lease.run_background_job(
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
        )
        record_tool_sandbox_command(self.state, self.lease, command, result)
        return result

    async def delete(self) -> None:
        await self.lease.delete()


async def create_tool_sandbox_lease(toolset: object) -> SandboxLease:
    return await create_scoped_sandbox_lease(toolset, tool_sandbox_key(toolset))


async def create_sandbox_lease(
    sandbox_config: Mapping[str, object], key: str
) -> SandboxLease:
    from prime_sandboxes import AsyncSandboxClient

    scope = sandbox_scope(sandbox_config)
    client = AsyncSandboxClient()
    try:
        sandbox_id = await create_sandbox(client, sandbox_config)
    except BaseException:
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            await aclose()
        raise
    lease = SandboxLease(client, sandbox_id, scope, key)
    try:
        await setup_sandbox(lease, sandbox_config)
    except BaseException:
        await lease.delete()
        raise
    return lease


async def create_scoped_sandbox_lease(
    owner: object, key: str | None = None
) -> SandboxLease:
    sandbox_config = getattr(owner, "sandbox", None)
    if not isinstance(sandbox_config, Mapping):
        raise TypeError("Sandbox owner must define a sandbox mapping.")
    return await create_sandbox_lease(sandbox_config, key or sandbox_owner_key(owner))


async def run_sandbox_command(
    program: Mapping[str, object],
    sandbox_config: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
) -> State:
    validate_program_bindings(program)
    lease = await runtime.resolve_program_sandbox(sandbox_config, task, state)
    async with lease.lock:
        state["sandbox_id"] = lease.id
        state.setdefault("runtime", {})
        lease_scope_key = getattr(lease, "scope_key", None) or runtime.scope_key(
            lease.scope, state
        )
        state["runtime"]["sandbox"] = {
            "id": lease.id,
            "scope": lease.scope,
            "key": lease.key,
            "lease_key": [lease_scope_key, lease.key],
        }
        handle = SandboxHandle(lease, state)
        await runtime.setup_rollout(
            task,
            state,
            setup_handlers=program_setup_handlers(lease, program, runtime),
            sandbox=handle,
        )
        workdir = cast(str | None, sandbox_config.get("workdir"))
        if workdir:
            await lease.client.execute_command(
                lease.id, f"mkdir -p {shlex.quote(workdir)}"
            )
        argv = await command_argv(program, task, state, runtime)
        env = await command_env(program, task, state, runtime, include_base=False)
        command = shlex.join(argv)
        command_timeout = cast(int | None, sandbox_config.get("command_timeout"))
        result = await lease.run_background_job(
            command,
            timeout=command_timeout,
            working_dir=workdir,
            env=env,
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
        state._set_stop_condition("command_completed")
        await collect_sandbox_artifacts(lease.client, lease.id, program, state)
        return state


def program_setup_handlers(
    lease: SandboxLease, program: Mapping[str, object], runtime: Runtime
) -> list[Callable[..., object]]:
    handlers: list[Callable[..., object]] = [
        _program_setup_handler(
            lease,
            program,
            runtime,
            upload_program_files,
            "program_upload_files",
            200,
        ),
        _program_setup_handler(
            lease,
            program,
            runtime,
            upload_program_dirs,
            "program_upload_dirs",
            190,
        ),
        _program_setup_handler(
            lease,
            program,
            runtime,
            run_program_setup,
            "program_setup",
            100,
        ),
        _program_setup_handler(
            lease,
            program,
            runtime,
            upload_state_input,
            "program_state_input",
            -50,
        ),
    ]
    for tool_type, setup_item, priority in program_tools_setup(program):
        handlers.append(
            _program_tools_setup_handler(
                lease,
                program,
                runtime,
                str(tool_type),
                setup_item,
                priority,
            )
        )
    return handlers


def _program_setup_handler(
    lease: SandboxLease,
    program: Mapping[str, object],
    runtime: Runtime,
    fn: Callable[..., Awaitable[None]],
    name: str,
    priority: int,
) -> Callable[..., object]:
    async def handler(task: Task, state: State) -> None:
        await fn(lease.client, lease.id, program, task, state, runtime)

    handler.__name__ = name
    setattr(handler, "setup", True)
    setattr(handler, "setup_priority", priority)
    return handler


def _program_tools_setup_handler(
    lease: SandboxLease,
    program: Mapping[str, object],
    runtime: Runtime,
    tool_type: str,
    setup_item: object,
    priority: int,
) -> Callable[..., object]:
    async def handler(task: Task, state: State) -> None:
        await run_program_items(
            lease.client,
            lease.id,
            program,
            task,
            state,
            runtime,
            items=[setup_item],
            error_prefix=f"Program {tool_type} tools setup failed",
        )

    handler.__name__ = f"program_{tool_type}_tools_setup"
    setattr(handler, "setup", True)
    setattr(handler, "setup_priority", priority)
    return handler


async def create_sandbox(client: object, sandbox_config: Mapping[str, object]) -> str:
    import os

    from prime_sandboxes import CreateSandboxRequest

    team_id = (
        str(sandbox_config.get("team_id") or "") or os.getenv("PRIME_TEAM_ID") or None
    )
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
        team_id=team_id,
    )
    sandbox = await cast(Any, client).create(request)
    sandbox_id = str(sandbox.id)
    try:
        await cast(Any, client).wait_for_creation(sandbox_id)
    except BaseException:
        await cast(Any, client).delete(sandbox_id)
        raise
    return sandbox_id


async def setup_sandbox(
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
                python_package_install_command(package_args),
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


def python_package_install_command(package_args: str) -> str:
    return (
        "set -e\n"
        "if command -v python3 >/dev/null 2>&1; then PYTHON=python3; "
        "elif command -v python >/dev/null 2>&1; then PYTHON=python; "
        "elif command -v apt-get >/dev/null 2>&1; then "
        "apt-get -o Acquire::Retries=3 update && "
        "apt-get -o Acquire::Retries=3 install -y python3 python3-pip && "
        "PYTHON=python3; "
        "else echo 'python is required to install sandbox packages' >&2; exit 127; fi\n"
        "$PYTHON -m pip --version >/dev/null 2>&1 || "
        "$PYTHON -m ensurepip --upgrade || "
        "(command -v apt-get >/dev/null 2>&1 && "
        "apt-get -o Acquire::Retries=3 update && "
        "apt-get -o Acquire::Retries=3 install -y python3-pip)\n"
        "$PYTHON -m pip install --disable-pip-version-check --break-system-packages "
        f"{package_args} || "
        "$PYTHON -m pip install --disable-pip-version-check "
        f"{package_args}"
    )


def python_runtime_setup_command() -> str:
    return (
        "set -e\n"
        "if command -v python3 >/dev/null 2>&1; then exit 0; fi\n"
        "if command -v python >/dev/null 2>&1; then exit 0; fi\n"
        "if command -v apt-get >/dev/null 2>&1; then "
        "apt-get -o Acquire::Retries=3 update && "
        "apt-get -o Acquire::Retries=3 install -y python3; exit 0; fi\n"
        "echo 'python is required for sandbox Python programs' >&2\n"
        "exit 127"
    )


def python_runtime_command(script_path: str, *args: str) -> list[str]:
    command = (
        "PYTHON=$(command -v python3 || command -v python || true); "
        'if [ -z "$PYTHON" ]; then '
        "echo 'python is required for sandbox Python programs' >&2; exit 127; "
        "fi; "
        f'exec "$PYTHON" {shlex.quote(script_path)}'
    )
    for arg in args:
        command += f" {shlex.quote(arg)}"
    return ["/bin/sh", "-lc", command]


def attach_sandbox_ref(state: State, lease: SandboxLease) -> None:
    state.setdefault("runtime", {})
    sandboxes = state["runtime"].setdefault("sandboxes", {})
    if not isinstance(sandboxes, dict):
        raise TypeError("state.runtime.sandboxes must be a mapping.")
    sandboxes[lease.key] = {"id": lease.id, "scope": lease.scope}


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
    sandboxes = state["runtime"].setdefault("sandboxes", {})
    if isinstance(sandboxes, dict):
        tool_state = sandboxes.setdefault(
            lease.key, {"id": lease.id, "scope": lease.scope}
        )
        if isinstance(tool_state, dict):
            tool_state.setdefault("commands", []).append(command_record)


def tool_sandbox_key(toolset: object) -> str:
    from ..toolset import MCPTool, flatten_toolsets, tool_name

    names = [
        tool_name(tool)
        for tool in flatten_toolsets((toolset,))
        if not isinstance(tool, MCPTool)
    ]
    if names:
        return "tools:" + ",".join(sorted(names))
    return f"toolset:{id(toolset)}"


def program_sandbox_key(sandbox_config: Mapping[str, object]) -> str:
    try:
        fingerprint = json.dumps(sandbox_config, sort_keys=True)
    except TypeError as exc:
        raise TypeError("Program sandbox config must be JSON-serializable.") from exc
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    return f"program:{digest}"


def sandbox_owner_key(owner: object) -> str:
    fn = getattr(owner, "fn", None)
    name = getattr(fn, "__name__", None)
    if isinstance(name, str) and name:
        return f"user:{name}"
    return f"sandbox:{id(owner)}"


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
        content = await resolve_program_value(source, task, state, runtime, program)
        if not isinstance(content, str):
            content = str(content)
        await maybe_call_with_named_args(
            getattr(client, "upload_bytes"),
            sandbox_id=sandbox_id,
            file_path=path,
            file_bytes=content.encode(),
            filename=path.rsplit("/", 1)[-1] or "file",
        )


async def upload_program_dirs(
    client: object,
    sandbox_id: str,
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    dirs = program.get("dirs", {})
    if not isinstance(dirs, Mapping):
        raise TypeError("program.dirs must be a mapping.")
    for path, source in dirs.items():
        if not isinstance(path, str):
            raise TypeError("program.dirs keys must be strings.")
        local_source = await resolve_program_value(
            source, task, state, runtime, program
        )
        await upload_program_dir(client, sandbox_id, path, local_source)


async def upload_program_dir(
    client: object, sandbox_id: str, remote_path: str, local_source: object
) -> None:
    if isinstance(local_source, str):
        local_source = Path(local_source)
    if not isinstance(local_source, (Path, Traversable)):
        raise TypeError("program.dirs values must resolve to paths.")
    remote_tar = f"/tmp/_vf_upload_{remote_path.strip('/').replace('/', '_')}.tar.gz"
    tmp_path = await asyncio.to_thread(build_dir_archive, local_source, remote_path)
    try:
        await cast(Any, client).upload_file(sandbox_id, remote_tar, str(tmp_path))
        result = await maybe_call_with_named_args(
            getattr(client, "execute_command"),
            sandbox_id=sandbox_id,
            command=(
                f"mkdir -p {shlex.quote(str(Path(remote_path).parent))} && "
                f"tar -xzf {shlex.quote(remote_tar)} -C / && "
                f"rm -f {shlex.quote(remote_tar)}"
            ),
        )
        if result.exit_code:
            raise SandboxError(f"Program dir upload failed: {result.stderr}")
    finally:
        tmp_path.unlink(missing_ok=True)


def build_dir_archive(local_source: Path | Traversable, remote_path: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    arcname = remote_path.lstrip("/")
    with tarfile.open(tar_path, "w:gz") as tar:
        if isinstance(local_source, Path):
            tar.add(local_source, arcname=arcname, filter=upload_tar_filter)
        else:
            with resources.as_file(local_source) as local_path:
                tar.add(local_path, arcname=arcname, filter=upload_tar_filter)
    return tar_path


UPLOAD_IGNORE_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def upload_tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if any(part in UPLOAD_IGNORE_PARTS for part in Path(tarinfo.name).parts):
        return None
    return tarinfo


async def run_program_setup(
    client: object,
    sandbox_id: str,
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    await run_program_commands(
        client,
        sandbox_id,
        program,
        task,
        state,
        runtime,
        key="setup",
        error_prefix="Program setup failed",
    )


async def upload_state_input(
    client: object,
    sandbox_id: str,
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    _ = task, runtime
    path = program.get(VF_STATE_INPUT_PATH_KEY)
    if path is None:
        return
    if not isinstance(path, str):
        raise TypeError(f"{VF_STATE_INPUT_PATH_KEY} must be a string.")
    await maybe_call_with_named_args(
        getattr(client, "upload_bytes"),
        sandbox_id=sandbox_id,
        file_path=path,
        file_bytes=json.dumps(state).encode(),
        filename=path.rsplit("/", 1)[-1] or "file",
    )


async def run_program_commands(
    client: object,
    sandbox_id: str,
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    key: str,
    error_prefix: str,
) -> None:
    raw_setup = program.get(key) or []
    if isinstance(raw_setup, str):
        setup: list[object] = [raw_setup]
    elif isinstance(raw_setup, list):
        setup = list(raw_setup)
    else:
        setup = [raw_setup]
    await run_program_items(
        client,
        sandbox_id,
        program,
        task,
        state,
        runtime,
        items=setup,
        error_prefix=error_prefix,
    )


async def run_program_items(
    client: object,
    sandbox_id: str,
    program: Mapping[str, object],
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    items: list[object],
    error_prefix: str,
) -> None:
    env = await command_env(program, task, state, runtime, include_base=False)
    for command in items:
        command = await resolve_program_value(command, task, state, runtime, program)
        result = await maybe_call_with_named_args(
            getattr(client, "execute_command"),
            sandbox_id=sandbox_id,
            command=str(command),
            env=env,
        )
        if result.exit_code:
            raise SandboxError(f"{error_prefix}: {result.stderr}")


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
        path = artifact_path(spec)
        optional = artifact_optional(spec)
        try:
            content = await read_sandbox_artifact(
                client, sandbox_id, path.format(**state)
            )
        except FileNotFoundError:
            if optional:
                state["artifacts"][name] = None
                continue
            raise
        format_name = artifact_format(spec)
        if format_name == "json":
            value: object = json.loads(content)
        elif format_name == "text":
            value = content
        else:
            raise ValueError(f"Unsupported artifact format: {format_name!r}")
        key = artifact_key(spec)
        if key is not None:
            value = cast(Mapping[str, object], value)[key]
        state["artifacts"][name] = value


async def read_sandbox_artifact(client: object, sandbox_id: str, path: str) -> str:
    script = (
        "import glob, pathlib, sys\n"
        f"matches = sorted(glob.glob({path!r}))\n"
        "if not matches:\n"
        "    sys.exit(2)\n"
        "sys.stdout.write(pathlib.Path(matches[0]).read_text())\n"
    )
    command = (
        "PYTHON=$(command -v python3 || command -v python || true); "
        'if [ -z "$PYTHON" ]; then '
        "echo 'python is required to read sandbox artifacts' >&2; exit 127; "
        "fi; "
        f'exec "$PYTHON" -c {shlex.quote(script)}'
    )
    result = await maybe_call_with_named_args(
        getattr(client, "execute_command"),
        sandbox_id=sandbox_id,
        command=command,
    )
    if result.exit_code == 2:
        raise FileNotFoundError(f"Sandbox artifact not found: {path}")
    if result.exit_code:
        raise SandboxError(
            "Sandbox artifact reader failed: "
            f"{getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
        )
    return result.stdout or ""
