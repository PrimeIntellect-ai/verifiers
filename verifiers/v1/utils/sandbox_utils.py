import asyncio
import hashlib
import importlib.resources as resources
import json
import random
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Mapping
from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from prime_sandboxes import CommandTimeoutError

from verifiers.errors import Error, SandboxError
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.error_utils import error_info

from .artifact_utils import artifact_format, artifact_key, artifact_optional
from .artifact_utils import artifact_path
from .program_utils import command_argv, command_env, float_config, int_config
from .program_utils import program_option_mapping, program_channel_setup
from .program_utils import resolve_program_value
from .program_utils import validate_program_bindings
from ..runtime import Runtime
from ..state import State
from ..task import Task
from ..types import ConfigMap, Handler, JsonData, ProgramValue

if TYPE_CHECKING:
    from ..toolset import Toolset

VF_STATE_INPUT_PATH_KEY = "_vf_state_input_path"
SANDBOX_CREATE_RATE_LIMIT_ATTEMPTS = 6


class SandboxRecord(Protocol):
    id: object


class SandboxCommandResult(Protocol):
    exit_code: int
    stdout: str | None
    stderr: str | None


class SandboxBackgroundJob(Protocol):
    job_id: str


class SandboxBackgroundJobStatus(SandboxCommandResult, Protocol):
    completed: bool


class SandboxBackgroundJobRecord:
    def __init__(
        self,
        *,
        job_id: str,
        sandbox_id: str,
        pid_file: str,
        stdout_log_file: str,
        stderr_log_file: str,
        exit_file: str,
    ):
        self.job_id = job_id
        self.sandbox_id = sandbox_id
        self.pid_file = pid_file
        self.stdout_log_file = stdout_log_file
        self.stderr_log_file = stderr_log_file
        self.exit_file = exit_file


class SandboxClient(Protocol):
    async def create(self, request: object) -> SandboxRecord: ...

    async def wait_for_creation(self, sandbox_id: str) -> object: ...

    async def get(self, sandbox_id: str) -> object: ...

    async def delete(self, sandbox_id: str) -> object: ...

    async def aclose(self) -> object: ...

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult: ...

    async def upload_bytes(
        self,
        sandbox_id: str,
        file_path: str,
        file_bytes: bytes,
        *,
        filename: str | None = None,
        timeout: int | None = None,
    ) -> object: ...

    async def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        *,
        timeout: int | None = None,
    ) -> object: ...

    async def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        *,
        timeout: int | None = None,
    ) -> object: ...

    async def read_file(self, sandbox_id: str, path: str) -> object: ...

    async def start_background_job(
        self,
        sandbox_id: str,
        command: str,
        *,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxBackgroundJob: ...

    async def get_background_job(
        self,
        sandbox_id: str,
        job: SandboxBackgroundJob,
        *,
        timeout: int | None = None,
    ) -> SandboxBackgroundJobStatus: ...


class SandboxLease:
    def __init__(
        self,
        client: SandboxClient,
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
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        result = await maybe_call_with_named_args(
            getattr(self.client, "execute_command"),
            sandbox_id=self.id,
            command=command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
        )
        return cast(SandboxCommandResult, result)

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
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        return await run_sandbox_background_command(
            self.client,
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
            await self.client.delete(self.id)
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
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
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
        env: dict[str, str] | None = None,
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


async def create_tool_sandbox_lease(toolset: "Toolset") -> SandboxLease:
    return await create_scoped_sandbox_lease(toolset, tool_sandbox_key(toolset))


async def create_sandbox_lease(sandbox_config: ConfigMap, key: str) -> SandboxLease:
    from prime_sandboxes import AsyncSandboxClient

    scope = sandbox_scope(sandbox_config)
    client = cast(SandboxClient, AsyncSandboxClient())
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
    program: ConfigMap,
    sandbox_config: ConfigMap,
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
        try:
            result = await lease.run_background_job(
                command,
                timeout=command_timeout,
                working_dir=workdir,
                env=env,
            )
        except CommandTimeoutError as e:
            timeout_seconds = command_timeout or getattr(e, "timeout", None)
            stderr = f"Command timed out after {timeout_seconds}s"
            state["command"] = {
                "argv": argv,
                "returncode": 124,
                "stdout": "",
                "stderr": stderr,
            }
            state["completion"] = [{"role": "assistant", "content": ""}]
            state._set_error(error_info(SandboxError(stderr)))
            state._set_stop_condition("command_timeout")
            return state
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
    lease: SandboxLease, program: ConfigMap, runtime: Runtime
) -> list[Handler]:
    handlers: list[Handler] = [
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
    for channel, setup_item, priority in program_channel_setup(program):
        handlers.append(
            _program_channel_setup_handler(
                lease,
                program,
                runtime,
                str(channel),
                setup_item,
                priority,
            )
        )
    return handlers


def _program_setup_handler(
    lease: SandboxLease,
    program: ConfigMap,
    runtime: Runtime,
    fn: Callable[..., Awaitable[None]],
    name: str,
    priority: int,
) -> Handler:
    async def handler(task: Task, state: State) -> None:
        try:
            await fn(lease.client, lease.id, program, task, state, runtime)
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox setup handler {name} failed: {exc}") from exc

    handler.__name__ = name
    setattr(handler, "setup", True)
    setattr(handler, "setup_priority", priority)
    return handler


def _program_channel_setup_handler(
    lease: SandboxLease,
    program: ConfigMap,
    runtime: Runtime,
    channel: str,
    setup_item: ProgramValue,
    priority: int,
) -> Handler:
    async def handler(task: Task, state: State) -> None:
        name = f"program_{channel}_channel_setup"
        try:
            await run_program_items(
                lease.client,
                lease.id,
                program,
                task,
                state,
                runtime,
                items=[setup_item],
                error_prefix=f"Program {channel} channel setup failed",
            )
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox setup handler {name} failed: {exc}") from exc

    handler.__name__ = f"program_{channel}_channel_setup"
    setattr(handler, "setup", True)
    setattr(handler, "setup_priority", priority)
    return handler


async def create_sandbox(client: SandboxClient, sandbox_config: ConfigMap) -> str:
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
    sandbox = await create_sandbox_record(client, request)
    sandbox_id = str(sandbox.id)
    try:
        await wait_for_sandbox_running(client, sandbox_id, timeout=300)
        await wait_for_sandbox_ready(client, sandbox_id, timeout=300)
    except BaseException:
        try:
            await client.delete(sandbox_id)
        except BaseException:
            pass
        raise
    return sandbox_id


async def create_sandbox_record(
    client: SandboxClient, request: object
) -> SandboxRecord:
    for attempt in range(SANDBOX_CREATE_RATE_LIMIT_ATTEMPTS):
        try:
            return await client.create(request)
        except Exception as exc:
            final_attempt = attempt + 1 == SANDBOX_CREATE_RATE_LIMIT_ATTEMPTS
            if final_attempt or not sandbox_rate_limited(exc):
                raise
            delay = min(5 * 2**attempt, 60) + random.uniform(0, 2)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def sandbox_rate_limited(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        message = str(current)
        if "HTTP 429" in message or "Rate exceeded" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


async def wait_for_sandbox_running(
    client: SandboxClient,
    sandbox_id: str,
    timeout: int = 300,
    stability_checks: int = 3,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    stable_checks = 0
    last_error: BaseException | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            sandbox = await maybe_call_with_named_args(
                getattr(client, "get"),
                sandbox_id=sandbox_id,
            )
        except BaseException as exc:
            if not sandbox_rate_limited(exc):
                raise
            last_error = exc
            await asyncio.sleep(2)
            continue

        status = str(getattr(sandbox, "status", "") or "")
        if status == "RUNNING":
            stable_checks += 1
            if stable_checks >= stability_checks:
                return
        else:
            stable_checks = 0
            if status in {"ERROR", "TERMINATED", "TIMEOUT"}:
                raise SandboxError(f"Sandbox {sandbox_id} entered status {status}.")
        await asyncio.sleep(2)
    if last_error is not None:
        raise SandboxError(
            f"Sandbox {sandbox_id} did not reach RUNNING."
        ) from last_error
    raise SandboxError(f"Sandbox {sandbox_id} did not reach RUNNING.")


async def wait_for_sandbox_ready(
    client: SandboxClient, sandbox_id: str, timeout: int = 180
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_error: BaseException | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            sandbox = await maybe_call_with_named_args(
                getattr(client, "get"),
                sandbox_id=sandbox_id,
            )
        except BaseException as exc:
            if not sandbox_rate_limited(exc):
                raise
            last_error = exc
            await asyncio.sleep(2)
            continue
        status = str(getattr(sandbox, "status", "") or "")
        if status == "RUNNING":
            try:
                result = await maybe_call_with_named_args(
                    getattr(client, "execute_command"),
                    sandbox_id=sandbox_id,
                    command="true",
                    timeout=10,
                )
                result = cast(SandboxCommandResult, result)
                if result.exit_code == 0:
                    return
            except BaseException as exc:
                last_error = exc
        elif status in {"ERROR", "TERMINATED", "TIMEOUT"}:
            raise SandboxError(f"Sandbox {sandbox_id} entered status {status}.")
        await asyncio.sleep(2)
    if last_error is not None:
        raise SandboxError(
            f"Sandbox {sandbox_id} was not command-ready."
        ) from last_error
    raise SandboxError(f"Sandbox {sandbox_id} was not command-ready.")


async def setup_sandbox(handle: SandboxLease, sandbox_config: ConfigMap) -> None:
    packages = sandbox_config.get("packages") or []
    if isinstance(packages, str):
        packages = shlex.split(packages)
    if packages:
        if not isinstance(packages, list):
            raise TypeError("sandbox.packages must be a list or string.")
        package_args = " ".join(shlex.quote(str(package)) for package in packages)
        try:
            result = await handle.run_background_job(
                python_package_install_command(package_args),
                timeout=int_config(sandbox_config, "install_timeout", 300),
            )
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox package install failed: {exc}") from exc
        if result.exit_code:
            raise SandboxError(f"Sandbox package install failed: {result.stderr}")
    commands = sandbox_config.get("setup_commands") or []
    if isinstance(commands, str):
        commands = [commands]
    if not isinstance(commands, list):
        raise TypeError("sandbox.setup_commands must be a list or string.")
    for command in commands:
        try:
            result = await handle.run_background_job(
                str(command), timeout=int_config(sandbox_config, "setup_timeout", 300)
            )
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox setup command failed: {exc}") from exc
        if result.exit_code:
            raise SandboxError(f"Sandbox setup command failed: {result.stderr}")


def sandbox_scope(sandbox_config: ConfigMap) -> str:
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
    state: State, lease: SandboxLease, command: str, result: SandboxCommandResult
) -> None:
    command_record: JsonData = {
        "command": command,
        "returncode": result.exit_code,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
    commands = state.setdefault("sandbox_commands", [])
    if not isinstance(commands, list):
        raise TypeError("state.sandbox_commands must be a list.")
    commands = cast(list[JsonData], commands)
    commands.append(command_record)
    state.setdefault("runtime", {})
    sandboxes = state["runtime"].setdefault("sandboxes", {})
    if isinstance(sandboxes, dict):
        tool_state = sandboxes.setdefault(
            lease.key, {"id": lease.id, "scope": lease.scope}
        )
        if isinstance(tool_state, dict):
            tool_commands = tool_state.setdefault("commands", [])
            if isinstance(tool_commands, list):
                tool_commands = cast(list[JsonData], tool_commands)
                tool_commands.append(command_record)


def tool_sandbox_key(toolset: "Toolset") -> str:
    from ..toolset import MCPTool, flatten_toolsets, tool_name

    names = [
        tool_name(tool)
        for tool in flatten_toolsets((toolset,))
        if not isinstance(tool, MCPTool)
    ]
    if names:
        return "tools:" + ",".join(sorted(names))
    return f"toolset:{id(toolset)}"


def program_sandbox_key(sandbox_config: ConfigMap) -> str:
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
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigMap,
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    files = program_option_mapping(program.get("files"), "program.files")
    for path, source in files.items():
        content = await resolve_program_value(source, task, state, runtime, program)
        if not isinstance(content, str):
            content = str(content)
        await maybe_call_with_named_args(
            upload_sandbox_bytes,
            client=client,
            sandbox_id=sandbox_id,
            file_path=path,
            file_bytes=content.encode(),
            filename=path.rsplit("/", 1)[-1] or "file",
        )


async def upload_program_dirs(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigMap,
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    dirs = program_option_mapping(program.get("dirs"), "program.dirs")
    for path, source in dirs.items():
        local_source = await resolve_program_value(
            source, task, state, runtime, program
        )
        await upload_program_dir(client, sandbox_id, path, local_source)


async def upload_program_dir(
    client: SandboxClient, sandbox_id: str, remote_path: str, local_source: object
) -> None:
    if isinstance(local_source, str):
        local_source = Path(local_source)
    if not isinstance(local_source, (Path, Traversable)):
        raise TypeError("program.dirs values must resolve to paths.")
    remote_tar = f"/tmp/_vf_upload_{remote_path.strip('/').replace('/', '_')}.tar.gz"
    tmp_path = await asyncio.to_thread(build_dir_archive, local_source, remote_path)
    try:
        await maybe_call_with_named_args(
            getattr(client, "upload_file"),
            sandbox_id=sandbox_id,
            file_path=remote_tar,
            local_file_path=str(tmp_path),
        )
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
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigMap,
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
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigMap,
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
        upload_sandbox_bytes,
        client=client,
        sandbox_id=sandbox_id,
        file_path=path,
        file_bytes=json.dumps(state).encode(),
        filename=path.rsplit("/", 1)[-1] or "file",
    )


async def upload_sandbox_bytes(
    client: SandboxClient,
    *,
    sandbox_id: str,
    file_path: str,
    file_bytes: bytes,
    filename: str,
    attempts: int = 10,
    timeout: int = 30,
) -> object:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await maybe_call_with_named_args(
                getattr(client, "upload_bytes"),
                sandbox_id=sandbox_id,
                file_path=file_path,
                file_bytes=file_bytes,
                filename=filename,
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            await asyncio.sleep(min(2 + attempt, 10))
    raise SandboxError(f"Sandbox upload failed: {file_path}") from last_error


async def run_program_commands(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigMap,
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    key: str,
    error_prefix: str,
) -> None:
    raw_setup = program.get(key) or []
    if isinstance(raw_setup, str):
        setup: list[ProgramValue] = [raw_setup]
    elif isinstance(raw_setup, list):
        setup = [cast(ProgramValue, item) for item in raw_setup]
    else:
        setup = [cast(ProgramValue, raw_setup)]
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
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigMap,
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    items: list[ProgramValue],
    error_prefix: str,
) -> None:
    env = await command_env(program, task, state, runtime, include_base=False)
    timeout = int_config(program, "setup_timeout", 300)
    for command in items:
        command = await resolve_program_value(command, task, state, runtime, program)
        result = await run_sandbox_background_command(
            client,
            sandbox_id=sandbox_id,
            command=str(command),
            env=env,
            timeout=timeout,
        )
        if result.exit_code:
            raise SandboxError(f"{error_prefix}: {result.stderr}")


async def run_sandbox_background_command(
    client: SandboxClient,
    *,
    sandbox_id: str,
    command: str,
    timeout: int | None,
    working_dir: str | None = None,
    env: dict[str, str] | None = None,
) -> SandboxCommandResult:
    wait_seconds = timeout if timeout is not None else 900
    poll_interval = 1
    job_id = uuid.uuid4().hex[:8]
    pid_file = f"/tmp/job_{job_id}.pid"
    stdout_log_file = f"/tmp/job_{job_id}.stdout.log"
    stderr_log_file = f"/tmp/job_{job_id}.stderr.log"
    exit_file = f"/tmp/job_{job_id}.exit"

    env_prefix = ""
    if env:
        exports = []
        for key, value in env.items():
            if not key.replace("_", "").isalnum() or key[:1].isdigit():
                raise ValueError(f"Invalid environment variable name: {key!r}")
            exports.append(f"export {key}={shlex.quote(value)}")
        env_prefix = "; ".join(exports)
        if env_prefix:
            env_prefix += "; "

    dir_prefix = f"cd {shlex.quote(working_dir)} && " if working_dir else ""
    command_body = f"{env_prefix}{dir_prefix}{command}"
    sh_command = (
        f"echo $$ > {shlex.quote(pid_file)}; "
        f"({command_body}) > {shlex.quote(stdout_log_file)} "
        f"2> {shlex.quote(stderr_log_file)}; echo $? > {shlex.quote(exit_file)}"
    )
    bg_cmd = (
        "if command -v setsid >/dev/null 2>&1; then "
        f"nohup setsid sh -c {shlex.quote(sh_command)} "
        "< /dev/null > /dev/null 2>&1 & "
        "else "
        f"nohup sh -c {shlex.quote(sh_command)} "
        "< /dev/null > /dev/null 2>&1 & "
        "fi"
    )
    start_timeout = min(max(wait_seconds, 60), 120)
    start_result = await maybe_call_with_named_args(
        getattr(client, "execute_command"),
        sandbox_id=sandbox_id,
        command=bg_cmd,
        timeout=start_timeout,
    )
    start_result = cast(SandboxCommandResult, start_result)
    if start_result.exit_code:
        raise SandboxError(f"Sandbox background command failed: {start_result.stderr}")

    job = SandboxBackgroundJobRecord(
        job_id=job_id,
        sandbox_id=sandbox_id,
        pid_file=pid_file,
        stdout_log_file=stdout_log_file,
        stderr_log_file=stderr_log_file,
        exit_file=exit_file,
    )
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while asyncio.get_running_loop().time() < deadline:
        status = await maybe_call_with_named_args(
            getattr(client, "get_background_job"),
            sandbox_id=sandbox_id,
            job=job,
            timeout=30,
        )
        status = cast(SandboxBackgroundJobStatus, status)
        if status.completed:
            return status
        await asyncio.sleep(poll_interval)

    await terminate_sandbox_background_job(client, sandbox_id, job)
    raise CommandTimeoutError(sandbox_id, command, wait_seconds)


async def terminate_sandbox_background_job(
    client: SandboxClient,
    sandbox_id: str,
    job: SandboxBackgroundJobRecord,
) -> None:
    command = (
        f"pid_file={shlex.quote(job.pid_file)}; "
        'if [ -s "$pid_file" ]; then '
        'pid=$(cat "$pid_file"); '
        'kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; '
        "sleep 2; "
        'kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; '
        "fi"
    )
    await maybe_call_with_named_args(
        getattr(client, "execute_command"),
        sandbox_id=sandbox_id,
        command=command,
        timeout=30,
    )


async def collect_sandbox_artifacts(
    client: object, sandbox_id: str, program: ConfigMap, state: State
) -> None:
    artifacts = program_option_mapping(program.get("artifacts"), "program.artifacts")
    if not artifacts:
        return
    state.setdefault("artifacts", {})
    for name, spec in artifacts.items():
        if not isinstance(spec, Mapping):
            raise TypeError("program.artifacts values must be mappings.")
        spec = cast(ConfigMap, spec)
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
        try:
            if format_name == "json":
                value: object = json.loads(content)
            elif format_name == "text":
                value = content
            else:
                raise ValueError(f"Unsupported artifact format: {format_name!r}")
            key = artifact_key(spec)
            if key is not None:
                value = cast(ConfigMap, value)[key]
        except Exception as exc:
            raise SandboxError(f"Sandbox artifact parsing failed: {name}") from exc
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
    try:
        result = await maybe_call_with_named_args(
            getattr(client, "execute_command"),
            sandbox_id=sandbox_id,
            command=command,
            timeout=60,
        )
    except Error:
        raise
    except Exception as exc:
        raise SandboxError(f"Sandbox artifact reader failed: {path}") from exc
    if result.exit_code == 2:
        raise FileNotFoundError(f"Sandbox artifact not found: {path}")
    if result.exit_code:
        raise SandboxError(
            "Sandbox artifact reader failed: "
            f"{getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}"
        )
    return result.stdout or ""
