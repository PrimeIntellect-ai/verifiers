"""Run tool adapters in A, with workspace execution over the native worker bridge.

The evaluator sends only policy-approved calls and pins their results. Adapter
state and all Runtime operations live in A. Only explicit JSON configuration is
sent across the boundary; host Runtime objects and provider clients stay on host.
"""

import asyncio
import contextlib
import importlib
import inspect
import json
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.bridge import ExecutionBridge
from verifiers.v1.runtimes.stdio_rpc import ProcessRPC
from verifiers.v1.types import ToolCall, ToolMessage


@dataclass(frozen=True)
class RemoteToolsSpec:
    factory: Callable
    kwargs: dict = field(default_factory=dict)
    defer_until_gate: bool = False


class SandboxTools:
    def __init__(
        self, runtime: Runtime, connection: ExecutionBridge, spec: RemoteToolsSpec
    ):
        self.runtime, self.connection, self.spec = runtime, connection, spec
        self.defer_until_gate = spec.defer_until_gate
        self.uses_local_gate = not spec.defer_until_gate
        self.local_gate_url: str | None = None
        self._stack = AsyncExitStack()
        self._python = ""
        self._endpoint: dict | None = None
        self._rpc: ProcessRPC | None = None
        self._daemon = None
        self._slot_id = uuid.uuid4().hex
        self._configured = False
        self._abandon_on_close = False
        self._fast_kill_on_close = False

    async def _spawn_rpc(self) -> tuple[ProcessRPC, dict]:
        process = await self._start_adapter_process()
        rpc = ProcessRPC(process)
        ready: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        rpc._pending["__ready__"] = ready
        try:
            info = await ready
        finally:
            rpc._pending.pop("__ready__", None)
        return rpc, info

    async def _start_adapter_process(self) -> RuntimeProcess:
        from verifiers.v1.runtimes.controller import current_job

        argv = [
            self._python,
            "-I",
            "-u",
            "-m",
            __name__,
            self.spec.factory.__module__,
            self.spec.factory.__qualname__,
        ]
        if current_job.get() and self.runtime.type == "prime":
            from verifiers.v1.runtimes.lan_execution import LanExecution
            from verifiers.v1.runtimes.private_network import lan_host

            transport = LanExecution(
                self.runtime, lan_host(self.runtime.backend), direct=True
            )
            self._stack.push_async_callback(
                lambda: (
                    transport.abandon() if self._abandon_on_close else transport.close()
                )
            )
            await transport.start(self.runtime)
            return await transport.exec(self.runtime, argv, {})
        return await self.runtime.exec(argv, {})

    async def prepare(self) -> None:
        from verifiers.v1.harnesses.utils.launch import bundle_program
        from verifiers.v1.mcp.launch import install_in_runtime
        from verifiers.v1.runtimes import rpc_wire, workspace_worker

        source = bundle_program(inspect.getsource(workspace_worker), rpc_wire)
        has_lan = getattr(self.connection, "transport", None) is not None
        lease = getattr(self.runtime, "_lease", None)

        async def install_and_spawn() -> tuple[ProcessRPC | None, dict, object]:
            if lease is not None:
                daemon = await lease.ensure_daemon()
                if daemon is not None:
                    return None, {}, daemon
            self._python = await install_in_runtime(
                self.spec.factory, self.runtime, label="harness tool adapter"
            )
            rpc, info = await self._spawn_rpc()
            return rpc, info, None

        async def register_and_prewarm() -> dict | None:
            worker_python = await self.connection.runtime.resolve_python()
            self.connection.register(
                "tool-adapter", [worker_python, "-I", "-u", "-c", source], {}
            )
            if has_lan:
                return await self.connection.prewarm("tool-adapter")
            return None

        try:
            async with asyncio.TaskGroup() as group:
                adapter_rpc = group.create_task(install_and_spawn())
                prewarmed = group.create_task(register_and_prewarm())
        except ExceptionGroup as failed:
            raise failed.exceptions[0]
        self._rpc, ready_info, self._daemon = adapter_rpc.result()
        self._endpoint = prewarmed.result()
        if self._daemon is not None:
            if self.uses_local_gate:
                self.local_gate_url = self._daemon.local_gate_url
        elif self.uses_local_gate and ready_info.get("gate_port"):
            self.local_gate_url = f"http://127.0.0.1:{ready_info['gate_port']}/tool"

    async def connect(
        self,
        url: str,
        secret: str,
        *,
        interception_url: str | None = None,
    ) -> None:
        if self._configured:
            return
        if (
            self._daemon is None
            and (lease := getattr(self.runtime, "_lease", None)) is not None
        ):
            self._daemon = await lease.ensure_daemon()
            if self._daemon is not None and self.uses_local_gate:
                self.local_gate_url = self._daemon.local_gate_url
        if self._daemon is None and self._rpc is None:
            self._rpc, ready_info = await self._spawn_rpc()
            if self.uses_local_gate and ready_info.get("gate_port"):
                self.local_gate_url = f"http://127.0.0.1:{ready_info['gate_port']}/tool"
        for rt in (self.runtime, self.connection.runtime):
            await rt.wait_lan_ready()
        gate_target = interception_url if self.uses_local_gate else None
        command: list[str] | None = None
        if self._endpoint is None:
            from verifiers.v1.harnesses.utils.remote_runtime import (
                prepare_remote_bridge,
            )

            command = await prepare_remote_bridge(
                self.runtime, url, secret, "tool-adapter"
            )
        if self._daemon is not None:
            await self._daemon.register_tool_slot(
                slot_id=self._slot_id,
                module=self.spec.factory.__module__,
                factory=self.spec.factory.__qualname__,
                kwargs=self.spec.kwargs,
                endpoint=self._endpoint,
                command=command,
                interception_url=gate_target,
                secret=secret,
            )
        elif self._endpoint is not None:
            assert self._rpc is not None
            await self._rpc.call(
                "configure",
                module=self.spec.factory.__module__,
                factory=self.spec.factory.__qualname__,
                kwargs=self.spec.kwargs,
                endpoint=self._endpoint,
                interception_url=gate_target,
                secret=secret,
            )
        else:
            assert self._rpc is not None and command is not None
            await self._rpc.call(
                "configure",
                module=self.spec.factory.__module__,
                factory=self.spec.factory.__qualname__,
                kwargs=self.spec.kwargs,
                command=command,
                interception_url=gate_target,
                secret=secret,
            )
        self._configured = True

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if not self._configured:
            raise HarnessError("sandbox tool adapter is not connected")
        if self._daemon is not None:
            result = await self._daemon.execute_tool_slot(
                self._slot_id, call.model_dump()
            )
            return ToolMessage.model_validate(result)
        if self._rpc is None:
            raise HarnessError("sandbox tool adapter is not connected")
        result = await self._rpc.call("execute", call=call.model_dump())
        return ToolMessage.model_validate(result)

    async def aclose(self) -> None:
        try:
            if self._daemon is not None:
                await self._daemon.unregister_tool_slot(self._slot_id)
                self._daemon = None
            elif self._rpc is not None:
                if self._abandon_on_close:
                    await self._rpc.process.abandon()
                    await super(ProcessRPC, self._rpc).close()
                    self._rpc = None
                elif self._fast_kill_on_close:
                    await self._rpc.close()
                    self._rpc = None
                else:
                    try:
                        async with asyncio.timeout(10):
                            await self._rpc.call("close")
                    finally:
                        await self._rpc.close()
                        self._rpc = None
        finally:
            await self._stack.aclose()


def _resolve_factory(module_name: str, qualname: str) -> Callable:
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    assert callable(target)
    return target


def _same_arguments(raw_call: dict, arguments: object) -> bool:
    raw_args = raw_call.get("arguments", "")
    try:
        return json.dumps(json.loads(raw_args), sort_keys=True) == json.dumps(
            arguments, sort_keys=True
        )
    except (json.JSONDecodeError, TypeError):
        return raw_args == arguments


def _match_queued_item(
    items: list[dict],
    completed: dict[tuple[str, str], dict],
    tool_call_id: str,
    name: str | None,
    arguments: object,
) -> list[dict]:
    keys = {tool_call_id}
    if isinstance(arguments, dict):
        keys.update(
            arguments[field]
            for field in ("tool_call_id", "call_id", "toolCallId")
            if isinstance(arguments.get(field), str)
        )

    def id_matches(pool: list[dict]) -> list[dict]:
        return [
            item
            for item in pool
            if any(
                key == (cid := item["call"]["id"])
                or key.startswith(f"{cid}|")
                or key.endswith(f":{cid}")
                for key in keys
            )
        ]

    uncompleted = [item for item in items if tuple(item["key"]) not in completed]
    matches = (
        id_matches(uncompleted)
        or id_matches(items)
        or [
            item
            for item in uncompleted
            if (name is None or item["call"]["name"] == name)
            and _same_arguments(item["call"], arguments)
        ]
    )
    if len(matches) > 1:
        matches = [
            item
            for item in matches
            if (name is None or item["call"]["name"] == name)
            and _same_arguments(item["call"], arguments)
        ] or matches
    return matches


async def _read_http_request(
    reader: asyncio.StreamReader,
) -> tuple[str, dict[str, str], bytes] | None:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head[:-4].decode("latin-1").split("\r\n")
    if not lines or not lines[0]:
        return None
    parts = lines[0].split(" ", 2)
    path = parts[1] if len(parts) > 1 else "/"
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    if headers.get("transfer-encoding", "").lower() == "chunked":
        chunks = bytearray()
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line[:-2].split(b";", 1)[0], 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                break
            chunks.extend(await reader.readexactly(size))
            await reader.readexactly(2)
        return path, headers, bytes(chunks)
    length = int(headers.get("content-length", "0"))
    body = await reader.readexactly(length) if length > 0 else b""
    return path, headers, body


async def main() -> None:
    import httpx

    from verifiers.v1.runtimes.remote_backend import RemoteBackend
    from verifiers.v1.runtimes.rpc_wire import encode, serve

    preloaded: Callable | None = None
    if len(sys.argv) >= 3:
        preloaded = _resolve_factory(sys.argv[1], sys.argv[2])

    backend = None
    adapter = None
    http: httpx.AsyncClient | None = None
    interception_url: str | None = None
    gate_secret = ""
    configured_ready = asyncio.Event()
    batch_event = asyncio.Event()
    session_closed = False
    closing = False
    queued_items: list[dict] = []
    completed_verdicts: dict[tuple[str, str], dict] = {}
    key_locks: dict[tuple[str, str], asyncio.Lock] = {}
    poller_task: asyncio.Task | None = None
    bg_tasks: set[asyncio.Task] = set()

    def post_bg(payload: dict) -> None:
        async def send() -> None:
            if http is None or interception_url is None:
                return
            for attempt in range(2):
                try:
                    resp = await http.post(interception_url, json=payload)
                    if resp.status_code < 500:
                        return
                except httpx.HTTPError:
                    if attempt == 0:
                        await asyncio.sleep(0.05)

        task = asyncio.create_task(send())
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)

    async def poll_loop() -> None:
        nonlocal session_closed
        assert http is not None and interception_url is not None
        while not closing and not session_closed:
            try:
                resp = await http.post(interception_url, json={"op": "poll"})
            except httpx.HTTPError:
                if closing:
                    return
                await asyncio.sleep(0.05)
                continue
            if resp.status_code != 200:
                session_closed = True
                batch_event.set()
                return
            data = resp.json()
            if data.get("closed"):
                session_closed = True
                batch_event.set()
                return
            new_items = data.get("items") or []
            if new_items:
                queued_items.extend(new_items)
                batch_event.set()

    async def forward_upstream(
        tool_call_id: str, name: str | None, arguments: object
    ) -> tuple[int, dict]:
        assert http is not None and interception_url is not None
        resp = await http.post(
            interception_url,
            json={
                "tool_call_id": tool_call_id,
                "name": name,
                "arguments": arguments,
            },
        )
        return resp.status_code, resp.json()

    async def decide_local(body: dict) -> tuple[int, dict]:
        await configured_ready.wait()
        if adapter is None or http is None or interception_url is None:
            return 503, {"error": "sandbox tool gate not configured"}
        tool_call_id = str(body.get("tool_call_id", ""))
        name = body.get("name")
        arguments = body.get("arguments")
        deadline = time.monotonic() + 1.5
        while True:
            matches = _match_queued_item(
                queued_items, completed_verdicts, tool_call_id, name, arguments
            )
            if matches or session_closed:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            batch_event.clear()
            matches = _match_queued_item(
                queued_items, completed_verdicts, tool_call_id, name, arguments
            )
            if matches or session_closed:
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(batch_event.wait(), timeout=remaining)

        if (
            session_closed
            or len(matches) != 1
            or (name is not None and name != matches[0]["call"]["name"])
        ):
            return await forward_upstream(tool_call_id, name, arguments)

        item = matches[0]
        item_key = (str(item["key"][0]), str(item["key"][1]))
        call_dict = item["call"]
        alias_prefix = str(item["alias_prefix"])
        lock = key_locks.setdefault(item_key, asyncio.Lock())
        async with lock:
            if cached := completed_verdicts.get(item_key):
                return 200, {"action": "deny", "message": cached}
            if item["denied"] is not None:
                verdict_dict = item["denied"]
                completed_verdicts[item_key] = verdict_dict
                post_bg(
                    {
                        "op": "ack_denied",
                        "key": item["key"],
                        "alias_prefix": alias_prefix,
                        "call": call_dict,
                        "arguments": arguments,
                    }
                )
                return 200, {"action": "deny", "message": verdict_dict}
            verdict = await adapter(ToolCall.model_validate(call_dict))
            verdict_dict = bound_tool_message(verdict.model_dump(exclude_none=True))
            completed_verdicts[item_key] = verdict_dict
            post_bg(
                {
                    "op": "pin",
                    "key": item["key"],
                    "alias_prefix": alias_prefix,
                    "call": call_dict,
                    "message": verdict_dict,
                    "arguments": arguments,
                }
            )
            return 200, {"action": "deny", "message": verdict_dict}

    async def handle_gate_conn(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while not closing:
                try:
                    parsed = await _read_http_request(reader)
                except (asyncio.IncompleteReadError, ConnectionError, ValueError):
                    break
                if parsed is None:
                    break
                _path, headers, raw_body = parsed
                await configured_ready.wait()
                auth = headers.get("authorization", "")
                if not gate_secret or auth != f"Bearer {gate_secret}":
                    status, payload = 401, {"error": "unauthorized"}
                else:
                    try:
                        status, payload = await decide_local(
                            json.loads(raw_body or b"{}")
                        )
                    except Exception as error:  # noqa: BLE001
                        status, payload = 400, {"error": str(error)}
                encoded = json.dumps(payload).encode()
                reason = "OK" if status == 200 else "Error"
                close_conn = headers.get("connection", "").lower() == "close"
                conn_hdr = "close" if close_conn else "keep-alive"
                writer.write(
                    f"HTTP/1.1 {status} {reason}\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(encoded)}\r\n"
                    f"Connection: {conn_hdr}\r\n\r\n".encode("latin-1")
                    + encoded
                )
                await writer.drain()
                if close_conn:
                    break
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    gate_server = await asyncio.start_server(handle_gate_conn, "127.0.0.1", 0)
    gate_port = gate_server.sockets[0].getsockname()[1]

    async def close():
        nonlocal backend, adapter, http, poller_task, closing
        closing = True
        if bg_tasks:
            await asyncio.gather(*list(bg_tasks), return_exceptions=True)
        if poller_task is not None:
            poller_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await poller_task
            poller_task = None
        gate_server.close()
        with contextlib.suppress(Exception):
            await gate_server.wait_closed()
        if http is not None:
            await http.aclose()
            http = None
        try:
            if close_adapter := getattr(adapter, "aclose", None):
                await close_adapter()
        finally:
            adapter = None
            if backend is not None:
                await backend.stop()
                backend = None

    async def handle(message):
        nonlocal backend, adapter, http, interception_url, gate_secret, poller_task
        operation = message["operation"]
        if operation == "configure":
            if backend is not None:
                raise HarnessError("sandbox tool adapter already configured")
            factory = preloaded or _resolve_factory(
                message["module"], message["factory"]
            )
            if message.get("endpoint") is not None:
                backend = RemoteBackend(endpoint=message["endpoint"])
            else:
                backend = RemoteBackend(message["command"])
            await backend.start()
            adapter = factory(backend.bind(), **message["kwargs"])
            if message.get("interception_url"):
                interception_url = str(message["interception_url"])
                gate_secret = str(message.get("secret") or "")
                http = httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {gate_secret}"},
                    timeout=httpx.Timeout(None, connect=30.0),
                    trust_env=False,
                )
                poller_task = asyncio.create_task(poll_loop())
            configured_ready.set()
            return None
        if operation == "execute":
            if adapter is None:
                raise HarnessError("sandbox tool adapter is not configured")
            return (
                await adapter(ToolCall.model_validate(message["call"]))
            ).model_dump()
        if operation == "close":
            await close()
            return None
        raise HarnessError("unknown sandbox tool adapter operation")

    sys.stdout.buffer.write(
        encode({"id": "__ready__", "result": {"gate_port": gate_port}})
    )
    sys.stdout.buffer.flush()
    try:
        await serve(handle)
    finally:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
