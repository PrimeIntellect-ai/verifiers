import asyncio
import hashlib
import importlib.resources as resources
import json
import logging
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

import tenacity as tc

from verifiers.decorators import setup as setup_handler
from verifiers.errors import Error, SandboxError
from verifiers.utils.async_utils import maybe_call_with_named_args

from .program_utils import command_argv, command_env, float_config, int_config
from .program_utils import program_channels
from .program_utils import program_option_mapping, program_channel_setup
from .program_utils import resolve_program_value
from .program_utils import validate_program_bindings
from .sandbox_python_utils import (
    python_package_install_command,
    python_package_list,
    sandbox_python_path_command,
)
from ..runtime import Runtime
from ..sandbox import SandboxConfig
from ..program import ProgramValue
from ..state import State
from ..task import Task
from ..types import ConfigData, Handler

if TYPE_CHECKING:
    from ..toolset import Toolset

VF_STATE_INPUT_PATH_KEY = "_vf_state_input_path"
SANDBOX_RETRY_ATTEMPTS = 6
SANDBOX_WAIT_FOR_CREATION_ATTEMPTS = 120
T = TypeVar("T")
logger = logging.getLogger(__name__)


class SandboxRecord(Protocol):
    id: object


class RetryLogger(Protocol):
    def log(
        self, level: int, msg: str, /, *args: object, **kwargs: object
    ) -> object: ...


class SandboxCommandResult(Protocol):
    exit_code: int
    stdout: str | None
    stderr: str | None


class SandboxOwner(Protocol):
    @property
    def sandbox(self) -> SandboxConfig | Literal["program"] | None: ...


class SandboxClient(Protocol):
    async def create(self, request: object) -> SandboxRecord: ...

    async def wait_for_creation(
        self,
        sandbox_id: str,
        *,
        max_attempts: int = SANDBOX_WAIT_FOR_CREATION_ATTEMPTS,
    ) -> object: ...

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

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        poll_interval: int = 3,
    ) -> SandboxCommandResult: ...


async def with_sandbox_retry(operation: Callable[[], Awaitable[T]]) -> T:
    retry_logger = cast(RetryLogger, logger)
    async for attempt in tc.AsyncRetrying(
        stop=tc.stop_after_attempt(SANDBOX_RETRY_ATTEMPTS),
        wait=tc.wait_exponential_jitter(initial=0.5, max=30, jitter=1e-3),
        before_sleep=tc.before_sleep_log(retry_logger, logging.WARNING),
        sleep=asyncio.sleep,
        reraise=True,
    ):
        with attempt:
            return await operation()
    raise AssertionError("sandbox retry loop exited without running")


async def close_sandbox_client(client: SandboxClient) -> None:
    teardown = getattr(client, "teardown", None)
    if callable(teardown):
        teardown()
        return
    aclose = getattr(client, "aclose", None)
    if callable(aclose):
        await aclose()


def sandbox_failure_kind(exc: BaseException) -> Literal["oom", "timeout"] | None:
    if isinstance(exc, TimeoutError):
        return "timeout"
    name = type(exc).__name__
    text = str(exc)
    if name == "SandboxOOMError" or "OOM" in text or "OOM_KILLED" in text:
        return "oom"
    if name in {"SandboxTimeoutError", "CommandTimeoutError"} or "timed out" in text:
        return "timeout"
    return None


def mark_sandbox_failure(
    state: State,
    lease: "SandboxLease | None",
    exc: BaseException,
    *,
    phase: Literal["background_job", "create", "setup", "execute", "command"]
    | None = None,
) -> Literal["oom", "timeout"] | None:
    kind = sandbox_failure_kind(exc)
    if kind == "oom":
        state["sandbox_oom"] = True
    elif kind == "timeout":
        state["sandbox_timeout"] = True
    if kind is not None:
        logger.warning(
            "Sandbox failure detected (kind=%s, phase=%s, sandbox_id=%s, scope=%s): %s",
            kind,
            phase,
            lease.id if lease is not None else None,
            lease.scope if lease is not None else None,
            exc,
            exc_info=True,
        )
    return kind


class SandboxLease:
    def __init__(
        self,
        client: SandboxClient,
        sandbox_id: str,
        scope: str,
        key: str,
        *,
        owns_client: bool = True,
    ):
        self.client = client
        self.id = sandbox_id
        self.scope = scope
        self.key = key
        self.owns_client = owns_client
        self.scope_key: str | None = None
        self.deleted = False
        self.lock = asyncio.Lock()
        self.delete_lock = asyncio.Lock()

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        result = await maybe_call_with_named_args(
            self.client.execute_command,
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
            self.client.upload_bytes,
            sandbox_id=self.id,
            file_path=path,
            file_bytes=content,
            filename=filename or path.rsplit("/", 1)[-1] or "file",
        )

    async def upload_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await maybe_call_with_named_args(
            self.client.upload_file,
            sandbox_id=self.id,
            file_path=path,
            local_file_path=local_path,
            timeout=timeout,
        )

    async def download_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await maybe_call_with_named_args(
            self.client.download_file,
            sandbox_id=self.id,
            file_path=path,
            local_file_path=local_path,
            timeout=timeout,
        )

    async def read_file(self, path: str) -> object:
        return await maybe_call_with_named_args(
            self.client.read_file,
            sandbox_id=self.id,
            path=path,
        )

    async def run_background_job(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        poll_interval: int = 3,
    ) -> SandboxCommandResult:
        call_args: ConfigData = {
            "sandbox_id": self.id,
            "command": command,
            "working_dir": working_dir,
            "env": env,
            "poll_interval": poll_interval,
        }
        if timeout is not None:
            call_args["timeout"] = timeout
        return await maybe_call_with_named_args(
            getattr(self.client, "run_background_job"),
            **call_args,
        )

    async def delete(self) -> None:
        async with self.delete_lock:
            if self.deleted:
                return
            self.deleted = True
            try:
                await with_sandbox_retry(lambda: self.client.delete(self.id))
            except BaseException:
                self.deleted = False
                raise
            if self.owns_client:
                await close_sandbox_client(self.client)


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
        try:
            result = await self.lease.execute(
                command=command,
                timeout=timeout,
                working_dir=working_dir,
                env=env,
            )
        except Error:
            raise
        except Exception as exc:
            kind = mark_sandbox_failure(self.state, self.lease, exc, phase="execute")
            if kind is not None:
                raise SandboxError(
                    f"Sandbox {self.lease.id} failed during execute ({kind}): {exc}"
                ) from exc
            raise
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
        poll_interval: int = 3,
    ) -> SandboxCommandResult:
        try:
            result = await self.lease.run_background_job(
                command=command,
                timeout=timeout,
                working_dir=working_dir,
                env=env,
                poll_interval=poll_interval,
            )
        except Error:
            raise
        except Exception as exc:
            kind = mark_sandbox_failure(
                self.state, self.lease, exc, phase="background_job"
            )
            if kind is not None:
                raise SandboxError(
                    f"Sandbox {self.lease.id} failed during background job ({kind}): {exc}"
                ) from exc
            raise
        record_tool_sandbox_command(self.state, self.lease, command, result)
        return result

    async def delete(self) -> None:
        await self.lease.delete()


async def create_sandbox_lease(
    sandbox_config: SandboxConfig,
    key: str,
    client: SandboxClient | None = None,
) -> SandboxLease:
    sandbox_data = sandbox_config.data()
    owns_client = client is None
    if client is None:
        from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient

        client = cast(SandboxClient, ThreadedAsyncSandboxClient())
    sandbox_id = await create_sandbox(client, sandbox_data, owns_client=owns_client)
    lease = SandboxLease(
        client, sandbox_id, sandbox_config.scope, key, owns_client=owns_client
    )
    try:
        await setup_sandbox(lease, sandbox_data)
    except BaseException:
        await asyncio.shield(lease.delete())
        raise
    return lease


async def create_scoped_sandbox_lease(
    owner: SandboxOwner,
    key: str | None = None,
    client: SandboxClient | None = None,
) -> SandboxLease:
    sandbox = owner.sandbox
    if not isinstance(sandbox, SandboxConfig):
        raise TypeError("Sandbox owner must define a sandbox config.")
    return await create_sandbox_lease(sandbox, key or sandbox_owner_key(owner), client)


async def run_sandbox_command(
    program: ConfigData,
    sandbox_config: SandboxConfig,
    task: Task,
    state: State,
    runtime: Runtime,
) -> State:
    validate_program_bindings(program)
    sandbox_data = sandbox_config.data()
    try:
        lease = await runtime.resolve_program_sandbox(sandbox_config, task, state)
    except Exception as exc:
        mark_sandbox_failure(state, None, exc, phase="create")
        raise
    async with lease.lock:
        state["sandbox_id"] = lease.id
        runtime_state = state.runtime_state()
        lease_scope_key = lease.scope_key or runtime.scope_key(lease.scope, state)
        lease.scope_key = lease_scope_key
        runtime_state["sandbox"] = {
            "id": lease.id,
            "scope": lease.scope,
            "key": lease.key,
            "lease_key": [lease_scope_key, lease.key],
        }
        handle = SandboxHandle(lease, state)
        use_sandbox_python_path = bool(
            python_package_list(sandbox_data.get("packages"))
        )
        try:
            await runtime.setup_rollout(
                task,
                state,
                setup_handlers=program_setup_handlers(
                    lease,
                    program,
                    runtime,
                    use_sandbox_python_path=use_sandbox_python_path,
                ),
                sandbox=handle,
            )
        except Exception as exc:
            mark_sandbox_failure(state, lease, exc, phase="setup")
            raise
        workdir = sandbox_config.workdir
        if workdir:
            await lease.client.execute_command(
                lease.id, f"mkdir -p {shlex.quote(workdir)}"
            )
        argv = await command_argv(program, task, state, runtime)
        env = await command_env(program, task, state, runtime, include_base=False)
        command = shlex.join(argv)
        if use_sandbox_python_path or "mcp" in program_channels(program):
            command = sandbox_python_path_command(command)
        command_timeout = sandbox_config.command_timeout
        try:
            result = await lease.run_background_job(
                command,
                timeout=command_timeout,
                working_dir=workdir,
                env=env,
                poll_interval=int_config(sandbox_data, "poll_interval", 3),
            )
        except Error:
            raise
        except Exception as exc:
            kind = mark_sandbox_failure(state, lease, exc, phase="command")
            if kind is not None:
                raise SandboxError(
                    f"Sandbox {lease.id} failed during command ({kind}): {exc}"
                ) from exc
            raise
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
        return state


def program_setup_handlers(
    lease: SandboxLease,
    program: ConfigData,
    runtime: Runtime,
    *,
    use_sandbox_python_path: bool = False,
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
            use_sandbox_python_path=use_sandbox_python_path,
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
                use_sandbox_python_path=use_sandbox_python_path,
            )
        )
    return handlers


def _program_setup_handler(
    lease: SandboxLease,
    program: ConfigData,
    runtime: Runtime,
    fn: Callable[..., Awaitable[None]],
    name: str,
    priority: int,
    use_sandbox_python_path: bool = False,
) -> Handler:
    async def handler(task: Task, state: State) -> None:
        try:
            await maybe_call_with_named_args(
                fn,
                client=lease.client,
                sandbox_id=lease.id,
                program=program,
                task=task,
                state=state,
                runtime=runtime,
                use_sandbox_python_path=use_sandbox_python_path,
            )
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox setup handler {name} failed: {exc}") from exc

    handler.__name__ = name
    return setup_handler(handler, priority=priority)


def _program_channel_setup_handler(
    lease: SandboxLease,
    program: ConfigData,
    runtime: Runtime,
    channel: str,
    setup_item: ProgramValue,
    priority: int,
    use_sandbox_python_path: bool = False,
) -> Handler:
    name = f"program_{channel}_channel_setup"

    async def handler(task: Task, state: State) -> None:
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
                use_sandbox_python_path=use_sandbox_python_path,
            )
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(
                f"Sandbox {channel} channel setup handler {name} failed: {exc}"
            ) from exc

    handler.__name__ = name
    return setup_handler(handler, priority=priority)


async def create_sandbox(
    client: SandboxClient,
    sandbox_config: ConfigData,
    *,
    owns_client: bool = False,
) -> str:
    from prime_sandboxes import CreateSandboxRequest

    labels = sandbox_config.get("labels")
    gpu_count = int_config(sandbox_config, "gpu_count", 0)
    vm = sandbox_config.get("vm")
    environment_vars = sandbox_config.get("environment_vars")
    secrets = sandbox_config.get("secrets")
    request = CreateSandboxRequest(
        name=f"vf-v1-{uuid.uuid4().hex[:8]}",
        docker_image=str(sandbox_config.get("image") or "python:3.11-slim"),
        start_command=str(sandbox_config.get("start_command") or "tail -f /dev/null"),
        cpu_cores=float_config(sandbox_config, "cpu_cores", 1.0),
        memory_gb=float_config(sandbox_config, "memory_gb", 2.0),
        disk_size_gb=float_config(sandbox_config, "disk_size_gb", 5.0),
        gpu_count=gpu_count,
        gpu_type=str(sandbox_config["gpu_type"])
        if sandbox_config.get("gpu_type") is not None
        else None,
        vm=bool(vm) if vm is not None else gpu_count > 0,
        network_access=bool(sandbox_config.get("network_access", True)),
        timeout_minutes=int_config(sandbox_config, "timeout_minutes", 60),
        labels=[str(label) for label in labels] if isinstance(labels, list) else [],
        environment_vars={
            str(key): str(value) for key, value in environment_vars.items()
        }
        if isinstance(environment_vars, dict) and environment_vars
        else None,
        secrets={str(key): str(value) for key, value in secrets.items()}
        if isinstance(secrets, dict) and secrets
        else None,
        team_id=str(sandbox_config["team_id"])
        if sandbox_config.get("team_id") is not None
        else None,
        region=str(sandbox_config["region"])
        if sandbox_config.get("region") is not None
        else None,
        registry_credentials_id=str(sandbox_config["registry_credentials_id"])
        if sandbox_config.get("registry_credentials_id") is not None
        else None,
        guaranteed=bool(sandbox_config.get("guaranteed", False)),
    )
    create_task = asyncio.create_task(
        with_sandbox_retry(lambda: client.create(request))
    )
    try:
        create_waiter = asyncio.shield(create_task)
        if sandbox_config.get("create_timeout") is not None:
            sandbox = await asyncio.wait_for(
                create_waiter, int_config(sandbox_config, "create_timeout", 0)
            )
        else:
            sandbox = await create_waiter
    except (asyncio.CancelledError, TimeoutError):
        try:
            sandbox = cast(SandboxRecord, await asyncio.shield(create_task))
        except BaseException:
            if owns_client:
                await close_sandbox_client(client)
            raise
        await asyncio.shield(
            delete_sandbox_id(
                client,
                str(sandbox.id),
                close_client=owns_client,
                reason="cancelled creation",
            )
        )
        raise
    except BaseException:
        if owns_client:
            await close_sandbox_client(client)
        raise
    sandbox_id = str(sandbox.id)
    try:
        wait = client.wait_for_creation(
            sandbox_id,
            max_attempts=SANDBOX_WAIT_FOR_CREATION_ATTEMPTS,
        )
        if sandbox_config.get("wait_timeout") is not None:
            await asyncio.wait_for(wait, int_config(sandbox_config, "wait_timeout", 0))
        else:
            await wait
    except BaseException:
        delete_task = asyncio.create_task(
            delete_sandbox_id(
                client,
                sandbox_id,
                close_client=owns_client,
                reason="creation failure",
            )
        )
        await asyncio.shield(delete_task)
        raise
    return sandbox_id


async def delete_sandbox_id(
    client: SandboxClient,
    sandbox_id: str,
    *,
    close_client: bool,
    reason: str,
) -> None:
    try:
        await with_sandbox_retry(lambda: client.delete(sandbox_id))
    except Exception as cleanup_exc:
        logger.warning(
            "Failed to delete sandbox %s after %s: %s",
            sandbox_id,
            reason,
            cleanup_exc,
            exc_info=True,
        )
    finally:
        if close_client:
            await close_sandbox_client(client)


async def setup_sandbox(handle: SandboxLease, sandbox_config: ConfigData) -> None:
    packages = python_package_list(sandbox_config.get("packages"))
    if packages:
        package_args = " ".join(shlex.quote(str(package)) for package in packages)
        try:
            result = await handle.execute(
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
    use_sandbox_python_path = bool(packages)
    for command in commands:
        command = str(command)
        if use_sandbox_python_path:
            command = sandbox_python_path_command(command)
        try:
            result = await handle.execute(
                command,
                timeout=int_config(sandbox_config, "setup_timeout", 300),
            )
        except Error:
            raise
        except Exception as exc:
            raise SandboxError(f"Sandbox setup command failed: {exc}") from exc
        if result.exit_code:
            raise SandboxError(f"Sandbox setup command failed: {result.stderr}")


def attach_sandbox_ref(state: State, lease: SandboxLease) -> None:
    sandboxes = cast(ConfigData, state.runtime_state().setdefault("sandboxes", {}))
    sandboxes[lease.key] = {"id": lease.id, "scope": lease.scope}


def record_tool_sandbox_command(
    state: State, lease: SandboxLease, command: str, result: SandboxCommandResult
) -> None:
    command_record: ConfigData = {
        "command": command,
        "returncode": result.exit_code,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
    commands = cast(list[ConfigData], state.setdefault("sandbox_commands", []))
    commands.append(command_record)
    sandboxes = cast(ConfigData, state.runtime_state().setdefault("sandboxes", {}))
    tool_state = cast(
        ConfigData,
        sandboxes.setdefault(lease.key, {"id": lease.id, "scope": lease.scope}),
    )
    tool_commands = cast(list[ConfigData], tool_state.setdefault("commands", []))
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


def program_sandbox_key(sandbox_config: SandboxConfig) -> str:
    try:
        fingerprint = json.dumps(sandbox_config.data(), sort_keys=True)
    except TypeError as exc:
        raise TypeError("Program sandbox config must be JSON-serializable.") from exc
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
    return f"program:{digest}"


def sandbox_owner_key(owner: object) -> str:
    return f"sandbox:{id(owner)}"


async def upload_program_files(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigData,
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    from prime_sandboxes import APIError, UploadTimeoutError

    files = program_option_mapping(program.get("files"), "program.files")
    for path, source in files.items():
        content = await resolve_program_value(source, task, state, runtime, program)
        if not isinstance(content, str):
            content = str(content)
        try:
            await maybe_call_with_named_args(
                getattr(client, "upload_bytes"),
                sandbox_id=sandbox_id,
                file_path=path,
                file_bytes=content.encode(),
                filename=path.rsplit("/", 1)[-1] or "file",
            )
        except (APIError, UploadTimeoutError) as exc:
            raise SandboxError(
                f"Program file upload failed for {path!r} in sandbox {sandbox_id}: {exc}"
            ) from exc


async def upload_program_dirs(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigData,
    task: Task,
    state: State,
    runtime: Runtime,
) -> None:
    dirs = program_option_mapping(program.get("dirs"), "program.dirs")
    for path, source in dirs.items():
        local_source = await resolve_program_value(
            source, task, state, runtime, program
        )
        if local_source is None:
            continue
        if isinstance(local_source, str):
            local_source = Path(local_source)
        if not isinstance(local_source, (Path, Traversable)):
            raise TypeError("program.dirs values must resolve to paths.")
        remote_tar = f"/tmp/_vf_upload_{path.strip('/').replace('/', '_')}.tar.gz"
        archive_path = await runtime.cached_upload_archive(local_source, path)
        await maybe_call_with_named_args(
            client.upload_file,
            sandbox_id=sandbox_id,
            file_path=remote_tar,
            local_file_path=str(archive_path),
        )
        result = await maybe_call_with_named_args(
            client.execute_command,
            sandbox_id=sandbox_id,
            command=(
                f"mkdir -p {shlex.quote(str(Path(path).parent))} && "
                f"tar -xzf {shlex.quote(remote_tar)} -C / && "
                f"rm -f {shlex.quote(remote_tar)}"
            ),
        )
        if result.exit_code:
            raise SandboxError(f"Program dir upload failed: {result.stderr}")


def build_dir_archive(local_source: Path | Traversable, remote_path: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)
    arcname = remote_path.lstrip("/")
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            if isinstance(local_source, Path):
                tar.add(local_source, arcname=arcname, filter=upload_tar_filter)
            else:
                with resources.as_file(local_source) as local_path:
                    tar.add(local_path, arcname=arcname, filter=upload_tar_filter)
    except BaseException:
        tar_path.unlink(missing_ok=True)
        raise
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
    program: ConfigData,
    task: Task,
    state: State,
    runtime: Runtime,
    use_sandbox_python_path: bool = False,
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
        use_sandbox_python_path=use_sandbox_python_path,
    )


async def upload_state_input(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigData,
    state: State,
) -> None:
    path = program.get(VF_STATE_INPUT_PATH_KEY)
    if path is None:
        return
    if not isinstance(path, str):
        raise TypeError(f"{VF_STATE_INPUT_PATH_KEY} must be a string.")
    await maybe_call_with_named_args(
        client.upload_bytes,
        sandbox_id=sandbox_id,
        file_path=path,
        file_bytes=json.dumps(state).encode(),
        filename=path.rsplit("/", 1)[-1] or "file",
    )


async def run_program_commands(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigData,
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    key: str,
    error_prefix: str,
    use_sandbox_python_path: bool = False,
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
        use_sandbox_python_path=use_sandbox_python_path,
    )


async def run_program_items(
    client: SandboxClient,
    sandbox_id: str,
    program: ConfigData,
    task: Task,
    state: State,
    runtime: Runtime,
    *,
    items: list[ProgramValue],
    error_prefix: str,
    use_sandbox_python_path: bool = False,
) -> None:
    env = await command_env(program, task, state, runtime, include_base=False)
    timeout = int_config(program, "setup_timeout", 300)
    for command in items:
        command = await resolve_program_value(command, task, state, runtime, program)
        command = str(command)
        if use_sandbox_python_path:
            command = sandbox_python_path_command(command)
        result = await maybe_call_with_named_args(
            client.execute_command,
            sandbox_id=sandbox_id,
            command=command,
            env=env,
            timeout=timeout,
        )
        if result.exit_code:
            raise SandboxError(f"{error_prefix}: {result.stderr}")


async def read_sandbox_artifact(
    client: SandboxClient, sandbox_id: str, path: str
) -> str:
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
    command = sandbox_python_path_command(command)
    result = await maybe_call_with_named_args(
        client.execute_command,
        sandbox_id=sandbox_id,
        command=command,
    )
    if result.exit_code == 2:
        raise FileNotFoundError(f"Sandbox artifact not found: {path}")
    if result.exit_code:
        raise SandboxError(
            f"Sandbox artifact reader failed: {result.stderr or result.stdout or ''}"
        )
    return result.stdout or ""
