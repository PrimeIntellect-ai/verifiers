import asyncio
import base64
import hashlib
import inspect
import importlib.resources as resources
import json
import logging
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

import httpx
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
    from .endpoint_utils import Endpoint
    from ..toolset import Toolset

VF_STATE_INPUT_PATH_KEY = "_vf_state_input_path"
BRIDGE_ROOT = "/tmp/vf_interception_bridge"
BRIDGE_PROXY_PATH = "/tmp/vf_interception_bridge.py"
BRIDGE_PORT = 13131
SANDBOX_RETRY_ATTEMPTS = 6
SANDBOX_WAIT_FOR_CREATION_ATTEMPTS = 120
T = TypeVar("T")
logger = logging.getLogger(__name__)
ProgramPrepare = Callable[[ConfigData, State], ConfigData]


class RetryLogger(Protocol):
    def log(
        self, level: int, msg: str, /, *args: object, **kwargs: object
    ) -> object: ...


class SandboxRecord(Protocol):
    id: object


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
    ) -> SandboxCommandResult: ...


async def with_sandbox_retry(operation: Callable[[], Awaitable[T]]) -> T:
    retry_logger = cast(RetryLogger, logger)
    async for attempt in tc.AsyncRetrying(
        stop=tc.stop_after_attempt(SANDBOX_RETRY_ATTEMPTS),
        wait=tc.wait_exponential_jitter(initial=0.5, max=30, jitter=1e-3),
        before_sleep=tc.before_sleep_log(retry_logger, logging.WARNING),
        sleep=sandbox_retry_sleep,
        reraise=True,
    ):
        with attempt:
            return await operation()
    raise AssertionError("sandbox retry loop exited without running")


async def sandbox_retry_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def close_sandbox_client(client: SandboxClient) -> None:
    teardown = getattr(client, "teardown", None)
    if callable(teardown):
        teardown()
        return
    aclose = getattr(client, "aclose", None)
    if callable(aclose):
        await aclose()


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

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        result = await call_sandbox_client(
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
        return await call_sandbox_client(
            self.client.upload_bytes,
            sandbox_id=self.id,
            file_path=path,
            file_bytes=content,
            filename=filename or path.rsplit("/", 1)[-1] or "file",
        )

    async def upload_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await call_sandbox_client(
            self.client.upload_file,
            sandbox_id=self.id,
            file_path=path,
            local_file_path=local_path,
            timeout=timeout,
        )

    async def download_file(
        self, path: str, local_path: str, timeout: int | None = None
    ) -> object:
        return await call_sandbox_client(
            self.client.download_file,
            sandbox_id=self.id,
            file_path=path,
            local_file_path=local_path,
            timeout=timeout,
        )

    async def read_file(self, path: str) -> object:
        return await call_sandbox_client(
            self.client.read_file,
            sandbox_id=self.id,
            file_path=path,
            path=path,
        )

    async def run_background_job(
        self,
        command: str,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxCommandResult:
        call_args: ConfigData = {
            "sandbox_id": self.id,
            "command": command,
            "working_dir": working_dir,
            "env": env,
        }
        if timeout is not None:
            call_args["timeout"] = timeout
        result = await call_sandbox_client(
            getattr(self.client, "run_background_job"),
            **call_args,
        )
        return cast(SandboxCommandResult, result)

    async def delete(self) -> None:
        if self.deleted:
            return
        self.deleted = True
        try:
            await with_sandbox_retry(lambda: self.client.delete(self.id))
        finally:
            if self.owns_client:
                await close_sandbox_client(self.client)


async def call_sandbox_client(func: Callable, **objects: object) -> object:
    sig = inspect.signature(func)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        call_args = objects
    else:
        call_args = {
            key: value for key, value in objects.items() if key in sig.parameters
        }
    if inspect.iscoroutinefunction(func):
        return await func(**call_args)
    result = await asyncio.to_thread(func, **call_args)
    if inspect.isawaitable(result):
        return await result
    return result


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


async def create_tool_sandbox_lease(
    toolset: "Toolset", client: SandboxClient | None = None
) -> SandboxLease:
    return await create_scoped_sandbox_lease(toolset, tool_sandbox_key(toolset), client)


async def create_sandbox_lease(
    sandbox_config: SandboxConfig, key: str, client: SandboxClient | None = None
) -> SandboxLease:
    sandbox_data = sandbox_config.data()
    owns_client = client is None
    if client is None:
        from verifiers.utils.threaded_sandbox_client import ThreadedAsyncSandboxClient

        client = cast(SandboxClient, ThreadedAsyncSandboxClient())
    try:
        sandbox_id = await create_sandbox(client, sandbox_data)
    except BaseException:
        if owns_client:
            await close_sandbox_client(client)
        raise
    lease = SandboxLease(
        client, sandbox_id, sandbox_config.scope, key, owns_client=owns_client
    )
    try:
        await setup_sandbox(lease, sandbox_data)
    except BaseException:
        await lease.delete()
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
    endpoint: "Endpoint | None" = None,
    prepare_program: ProgramPrepare | None = None,
) -> State:
    sandbox_data = sandbox_config.data()
    lease = await runtime.resolve_program_sandbox(sandbox_config, task, state)
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
        async with sandbox_interception_bridge(lease, endpoint, state):
            if prepare_program is not None:
                program = prepare_program(program, state)
            validate_program_bindings(program)
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
            workdir = sandbox_config.workdir
            if workdir:
                await lease.execute(f"mkdir -p {shlex.quote(workdir)}")
            argv = await command_argv(program, task, state, runtime)
            env = await command_env(program, task, state, runtime, include_base=False)
            command = shlex.join(argv)
            if use_sandbox_python_path or "mcp" in program_channels(program):
                command = sandbox_python_path_command(command)
            command_timeout = sandbox_config.command_timeout
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
        return state


@asynccontextmanager
async def sandbox_interception_bridge(
    lease: SandboxLease,
    endpoint: "Endpoint | None",
    state: State,
) -> AsyncIterator[None]:
    rollout_key = state.get("endpoint_rollout_key")
    if endpoint is None or not isinstance(rollout_key, str):
        yield
        return

    original_root = state.get("endpoint_root_url")
    original_base = state.get("endpoint_base_url")
    await start_sandbox_bridge_proxy(lease)
    stop = asyncio.Event()
    forwarder = asyncio.create_task(run_sandbox_bridge_forwarder(lease, endpoint, stop))
    state["endpoint_root_url"] = f"http://127.0.0.1:{BRIDGE_PORT}/rollout/{rollout_key}"
    state["endpoint_base_url"] = f"{state['endpoint_root_url']}/v1"
    try:
        yield
    finally:
        stop.set()
        await asyncio.gather(forwarder, return_exceptions=True)
        await stop_sandbox_bridge_proxy(lease)
        if isinstance(original_root, str):
            state["endpoint_root_url"] = original_root
        if isinstance(original_base, str):
            state["endpoint_base_url"] = original_base


async def start_sandbox_bridge_proxy(lease: SandboxLease) -> None:
    await lease.upload_bytes(BRIDGE_PROXY_PATH, sandbox_bridge_proxy_source().encode())
    root = shlex.quote(BRIDGE_ROOT)
    proxy_path = shlex.quote(BRIDGE_PROXY_PATH)
    command = (
        f"mkdir -p {root}/requests {root}/responses && "
        f"rm -f {root}/requests/*.json {root}/responses/*.json && "
        "PYTHON=$(command -v python3 || command -v python) && "
        f'(nohup "$PYTHON" {proxy_path} --root {root} --port {BRIDGE_PORT} '
        f"> {root}/proxy.log 2>&1 & echo $! > {root}/proxy.pid)"
    )
    result = await lease.execute(command, timeout=10)
    if result.exit_code:
        raise SandboxError(f"Sandbox interception bridge failed: {result.stderr}")
    await wait_for_sandbox_bridge_proxy(lease)


async def wait_for_sandbox_bridge_proxy(lease: SandboxLease) -> None:
    command = (
        "PYTHON=$(command -v python3 || command -v python) && "
        '"$PYTHON" -c '
        + shlex.quote(
            "import socket, sys, time\n"
            "deadline = time.time() + 10\n"
            "while time.time() < deadline:\n"
            "    try:\n"
            f"        socket.create_connection(('127.0.0.1', {BRIDGE_PORT}), 1).close()\n"
            "        sys.exit(0)\n"
            "    except OSError:\n"
            "        time.sleep(0.1)\n"
            "sys.exit(1)\n"
        )
    )
    result = await lease.execute(command, timeout=15)
    if result.exit_code:
        raise SandboxError("Sandbox interception bridge did not become ready.")


async def stop_sandbox_bridge_proxy(lease: SandboxLease) -> None:
    root = shlex.quote(BRIDGE_ROOT)
    await lease.execute(
        f"if [ -f {root}/proxy.pid ]; then kill $(cat {root}/proxy.pid) 2>/dev/null || true; fi",
        timeout=10,
    )


async def run_sandbox_bridge_forwarder(
    lease: SandboxLease,
    endpoint: "Endpoint",
    stop: asyncio.Event,
) -> None:
    pending: set[asyncio.Task[None]] = set()
    seen: set[str] = set()
    while not stop.is_set():
        for path in await list_sandbox_bridge_requests(lease):
            if path in seen:
                continue
            seen.add(path)
            pending.add(
                asyncio.create_task(
                    forward_sandbox_bridge_request(
                        lease,
                        endpoint,
                        path,
                        bridge_response_path(path),
                    )
                )
            )
        done = {task for task in pending if task.done()}
        for task in done:
            pending.remove(task)
            await task
        await asyncio.sleep(0.05)
    if pending:
        await asyncio.gather(*pending)


async def list_sandbox_bridge_requests(lease: SandboxLease) -> list[str]:
    root = shlex.quote(f"{BRIDGE_ROOT}/requests")
    result = await lease.execute(
        f"find {root} -maxdepth 1 -type f -name '*.json' -print 2>/dev/null || true",
        timeout=5,
    )
    return [
        line
        for line in (result.stdout or "").splitlines()
        if line.startswith(f"{BRIDGE_ROOT}/requests/") and line.endswith(".json")
    ]


def bridge_response_path(request_path: str) -> str:
    return f"{BRIDGE_ROOT}/responses/{Path(request_path).name}"


async def forward_sandbox_bridge_request(
    lease: SandboxLease,
    endpoint: "Endpoint",
    request_path: str,
    response_path: str,
) -> None:
    raw_request = await lease.read_file(request_path)
    if hasattr(raw_request, "content"):
        raw_request = raw_request.content
    request = json.loads(str(raw_request))
    body = base64.b64decode(str(request.get("body") or ""))
    headers = bridge_request_headers(request.get("headers"))
    method = str(request.get("method") or "POST")
    path = str(request.get("path") or "/")
    timeout = httpx.Timeout(None)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                f"http://127.0.0.1:{endpoint.port}{path}",
                headers=headers,
                content=body,
            )
        payload = {
            "status": response.status_code,
            "headers": dict(response.headers),
            "body": base64.b64encode(response.content).decode(),
        }
    except Exception as exc:
        payload = {
            "status": 500,
            "headers": {"content-type": "application/json"},
            "body": base64.b64encode(json.dumps({"error": str(exc)}).encode()).decode(),
        }
    await lease.upload_bytes(response_path, json.dumps(payload).encode())
    await lease.execute(f"rm -f {shlex.quote(request_path)}", timeout=5)


def bridge_request_headers(value: object) -> dict[str, str]:
    headers = dict(cast(ConfigData, value or {}))
    for key in ("host", "content-length", "connection", "transfer-encoding"):
        headers.pop(key, None)
        headers.pop(key.title(), None)
    return {str(key): str(item) for key, item in headers.items()}


def sandbox_bridge_proxy_source() -> str:
    return r"""
import argparse
import base64
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "vf-interception-bridge"

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def do_OPTIONS(self):
        self.forward()

    def forward(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length) if length else b""
        request_id = uuid.uuid4().hex
        root = Path(self.server.bridge_root)
        request_path = root / "requests" / f"{request_id}.json"
        response_path = root / "responses" / f"{request_id}.json"
        payload = {
            "id": request_id,
            "method": self.command,
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "body": base64.b64encode(body).decode(),
        }
        tmp_path = request_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, request_path)
        deadline = time.time() + self.server.bridge_timeout
        while time.time() < deadline:
            if response_path.exists():
                response = json.loads(response_path.read_text())
                response_path.unlink(missing_ok=True)
                self.send_response(int(response.get("status") or 500))
                for key, value in dict(response.get("headers") or {}).items():
                    if key.lower() in {"content-length", "connection", "transfer-encoding"}:
                        continue
                    self.send_header(str(key), str(value))
                data = base64.b64decode(str(response.get("body") or ""))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            time.sleep(0.05)
        self.send_response(504)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"Verifiers bridge timed out"}')

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    root = Path(args.root)
    (root / "requests").mkdir(parents=True, exist_ok=True)
    (root / "responses").mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), BridgeHandler)
    server.bridge_root = str(root)
    server.bridge_timeout = args.timeout
    server.serve_forever()


if __name__ == "__main__":
    main()
"""


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
            run_program_post_setup,
            "program_post_setup",
            50,
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


async def create_sandbox(client: SandboxClient, sandbox_config: ConfigData) -> str:
    from prime_sandboxes import CreateSandboxRequest

    labels = sandbox_config.get("labels")
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
        labels=[str(label) for label in labels] if isinstance(labels, list) else [],
    )
    sandbox = await with_sandbox_retry(lambda: client.create(request))
    sandbox_id = str(sandbox.id)
    try:
        await client.wait_for_creation(
            sandbox_id,
            max_attempts=SANDBOX_WAIT_FOR_CREATION_ATTEMPTS,
        )
    except BaseException:
        try:
            await with_sandbox_retry(lambda: client.delete(sandbox_id))
        except Exception as cleanup_exc:
            logger.warning(
                "Failed to delete sandbox %s after creation failure: %s",
                sandbox_id,
                cleanup_exc,
                exc_info=True,
            )
        raise
    return sandbox_id


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
    sandboxes = state.runtime_state().setdefault("sandboxes", {})
    if not isinstance(sandboxes, dict):
        raise TypeError("state.runtime.sandboxes must be a mapping.")
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
    commands = state.setdefault("sandbox_commands", [])
    if not isinstance(commands, list):
        raise TypeError("state.sandbox_commands must be a list.")
    commands = cast(list[ConfigData], commands)
    commands.append(command_record)
    sandboxes = state.runtime_state().setdefault("sandboxes", {})
    if isinstance(sandboxes, dict):
        tool_state = sandboxes.setdefault(
            lease.key, {"id": lease.id, "scope": lease.scope}
        )
        if isinstance(tool_state, dict):
            tool_commands = tool_state.setdefault("commands", [])
            if isinstance(tool_commands, list):
                tool_commands = cast(list[ConfigData], tool_commands)
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
            client.upload_file,
            sandbox_id=sandbox_id,
            file_path=remote_tar,
            local_file_path=str(tmp_path),
        )
        result = await maybe_call_with_named_args(
            client.execute_command,
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


async def run_program_post_setup(
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
        key="post_setup",
        error_prefix="Program post_setup failed",
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
        result = cast(
            SandboxCommandResult,
            await call_sandbox_client(
                client.execute_command,
                sandbox_id=sandbox_id,
                command=command,
                env=env,
                timeout=timeout,
            ),
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
