"""The ordinary RuntimeProcess contract carried over a deployment's private LAN.

Provider stdio owns startup and shutdown. A process-scoped capability permits only
stdio and lifecycle operations on one registered worker. It grants no model,
policy, filesystem-management, or command-registration capability.
"""

import asyncio
import base64
import contextlib
import json
import logging
import secrets
import uuid

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.lan_stream import CLIENT_SOURCE, PYTHON
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

_PYTHON = PYTHON

# Run inside the same user, environment, cwd and filesystem view as the command.
# The capability arrives over provider stdin, never in the child environment.
_SERVER = r"""
import base64, concurrent.futures, hmac, json, os, signal, subprocess, sys, threading
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve

config = json.loads(sys.stdin.readline())
secret = config.pop("secret")
process_id = config.pop("process_id")
process = subprocess.Popen(config["argv"], stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           start_new_session=True)
locks = {name: threading.Lock() for name in ("stdin", "stdout", "stderr", "receipts")}
receipts = {}
retained = 0

def retain(size):
    global retained
    if retained + size > 128 * 1024 * 1024:
        raise ValueError("split execution exceeded its replay buffer budget")
    retained += size

def stop(sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass

def execute(body):
    if body.get("process_id") != process_id:
        raise ValueError("unknown execution process")
    operation = body.get("operation")
    if operation == "open":
        return {"process_id": process_id}
    if operation == "write":
        data = base64.b64decode(body["data"], validate=True)
        with locks["stdin"]:
            process.stdin.write(data)
            process.stdin.flush()
        return {}
    if operation == "read":
        stream = body.get("stream")
        if stream not in ("stdout", "stderr"):
            raise ValueError("invalid execution stream")
        with locks[stream]:
            data = os.read(getattr(process, stream).fileno(), 65536)
        with locks["receipts"]:
            retain(((len(data) + 2) // 3) * 4)
        return {"data": base64.b64encode(data).decode() if data else None, "eof": not data}
    if operation == "poll":
        return {"exit_code": process.poll()}
    if operation == "wait":
        return {"exit_code": process.wait()}
    if operation in ("kill", "terminate"):
        stop(signal.SIGKILL if operation == "kill" else signal.SIGTERM)
        return {"exit_code": process.wait()} if operation == "kill" else {}
    raise ValueError("unknown execution operation")

def handle(body):
    request_id = body.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("execution requires a request_id")
    # Process control is idempotent and cannot be blocked by replay limits.
    if body.get("operation") in ("kill", "terminate", "poll", "wait"):
        return execute(body)
    fingerprint = json.dumps(body, sort_keys=True)
    with locks["receipts"]:
        receipt = receipts.get(request_id)
        owner = receipt is None
        if owner:
            if len(receipts) >= 100000:
                raise ValueError("execution exceeded its receipt budget")
            retain(len(fingerprint.encode()))
            future = concurrent.futures.Future()
            receipts[request_id] = (fingerprint, future)
        else:
            previous, future = receipt
            if previous != fingerprint:
                raise ValueError("execution request_id was reused")
    if owner:
        try:
            future.set_result(execute(body))
        except BaseException as error:
            future.set_exception(error)
    return future.result()

connections = set()
connections_lock = threading.Lock()

def authenticate(connection, request):
    if request.path != "/execution":
        return connection.respond(404, "Unknown execution endpoint\n")
    if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer " + secret):
        return connection.respond(401, "Unauthorized\n")

def stream(connection):
    with connections_lock:
        connections.add(connection)
    slots = threading.BoundedSemaphore(8)

    def reply(body):
        try:
            result = {"result": handle(body)}
        except Exception as error:
            result = {"error": str(error)}
        try:
            connection.send(json.dumps({"id": body.get("request_id"), **result}))
        except ConnectionClosed:
            pass

    def work(body):
        try:
            reply(body)
        finally:
            slots.release()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            for raw in connection:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    connection.close(1008, "Invalid execution request")
                    break
                # Termination must remain available while reads/wait are blocked.
                if body.get("operation") in ("kill", "terminate", "poll"):
                    reply(body)
                elif slots.acquire(blocking=False):
                    pool.submit(work, body)
                else:
                    connection.close(1008, "Too many pending execution requests")
                    break
    except ConnectionClosed:
        pass
    finally:
        with connections_lock:
            connections.discard(connection)

server = serve(stream, "0.0.0.0", 0, process_request=authenticate,
               compression=None, max_size=32 * 1024 * 1024, close_timeout=1)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(json.dumps({"port": server.socket.getsockname()[1]}), flush=True)
try:
    # Only the owning evaluator's provider stream can end the listener. A native
    # client may finish while its output still needs draining or replaying.
    sys.stdin.readline()
finally:
    stop(signal.SIGKILL)
    server.shutdown()
    with connections_lock:
        active = list(connections)
    for connection in active:
        connection.close()
"""

# Trusted evaluator setup and lifecycle calls can reach LAN endpoints through A.
# Harness clients and adapters use /execution directly from A for workspace I/O.
_RELAY = (
    CLIENT_SOURCE
    + r"""
import sys
config = json.loads(sys.stdin.readline())
lock = threading.Lock()
client = LanStream(config)

def relay(message):
    result = {"id": message["id"]}
    try:
        result["result"] = client.request(message["body"])
    except Exception as error:
        result["error"] = str(error)
    with lock:
        print(json.dumps(result), flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    try:
        for line in sys.stdin:
            message = json.loads(line)
            if message["body"].get("operation") in ("kill", "terminate", "poll"):
                relay(message)
            else:
                pool.submit(relay, message)
    finally:
        client.close()
"""
)


class LanProcess(RuntimeProcess):
    def __init__(
        self, owner: RuntimeProcess, harness: Runtime, python: str, endpoint: dict
    ) -> None:
        self.owner = owner
        self.harness = harness
        self.python = python
        self.execution_endpoint = endpoint
        self._relay: RuntimeProcess | None = None
        self._lock = asyncio.Lock()
        self._stdin_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._finished: set[str] = set()
        self._exit_code: int | None = None
        self.stdout = self._stream("stdout")
        self.stderr = self._stream("stderr")

    async def _receive(self) -> None:
        assert self._relay is not None
        buffer = b""
        try:
            async for chunk in self._relay.stdout:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    response = json.loads(line)
                    future = self._pending.get(response["id"])
                    if future is not None and not future.done():
                        if "error" in response:
                            future.set_exception(SandboxError(response["error"]))
                        else:
                            future.set_result(response["result"])
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(SandboxError("LAN execution relay closed"))

    async def _rpc(self, operation: str, **values) -> dict:
        if self._closed:
            raise SandboxError("LAN execution process is closed")
        async with self._lock:
            if self._relay is None:
                relay = await self.harness.exec(
                    [self.python, "-I", "-u", "-c", _RELAY], {}
                )
                try:
                    await relay.write(
                        (json.dumps(self.execution_endpoint) + "\n").encode()
                    )
                    if self._closed:
                        raise SandboxError("LAN execution process is closed")
                except BaseException:
                    with contextlib.suppress(Exception):
                        await relay.kill()
                    raise
                self._relay = relay
                self._tasks.append(asyncio.create_task(self._receive()))

                async def drain() -> None:
                    assert self._relay is not None
                    async for _ in self._relay.stderr:
                        pass

                self._tasks.append(asyncio.create_task(drain()))
            request_id = uuid.uuid4().hex
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            body = {
                "operation": operation,
                "process_id": self.execution_endpoint["process_id"],
                "request_id": request_id,
                **values,
            }
            try:
                await self._relay.write(
                    (json.dumps({"id": request_id, "body": body}) + "\n").encode()
                )
            except BaseException:
                self._pending.pop(request_id, None)
                future.cancel()
                raise
        try:
            result = await future
            if operation == "wait":
                self._exit_code = result["exit_code"]
                self._finished.add("wait")
            elif operation == "read" and result.get("eof"):
                self._finished.add(values["stream"])
            if len(self._finished) == 3:
                await self.aclose()
            return result
        finally:
            self._pending.pop(request_id, None)

    async def _stream(self, name: str):
        while True:
            result = await self._rpc("read", stream=name)
            if result.get("data"):
                yield base64.b64decode(result["data"])
            if result["eof"]:
                return

    async def write(self, data: bytes) -> None:
        async with self._stdin_lock:
            for offset in range(0, len(data), 65536):
                await self._rpc(
                    "write",
                    data=base64.b64encode(data[offset : offset + 65536]).decode(),
                )

    async def wait(self) -> int:
        if self._exit_code is not None:
            return self._exit_code
        return (await self._rpc("wait"))["exit_code"]

    async def poll(self) -> int | None:
        if self._exit_code is not None:
            return self._exit_code
        return (await self._rpc("poll"))["exit_code"]

    async def terminate(self) -> None:
        if self._closed:
            return
        await self._rpc("terminate")

    async def kill(self) -> None:
        if self._closed:
            return
        await self._rpc("kill")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.owner.write(b"close\n")
            async with asyncio.timeout(5):
                await self.owner.wait()
        finally:
            with contextlib.suppress(Exception):
                await self.owner.kill()
            if self._relay is not None:
                with contextlib.suppress(Exception):
                    await self._relay.kill()
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)


class LanExecution:
    """Select LAN once for the deployment, underneath every harness adapter."""

    def __init__(self, harness: Runtime, host: str) -> None:
        self.harness = harness
        self.host = host
        self._python = ""
        self._harness_python = ""
        self._processes: list[LanProcess] = []
        self._closed = False

    async def start(self, workspace: Runtime) -> None:
        worker, relay = await asyncio.gather(
            workspace.prepare_uv_script(_PYTHON, activate=False),
            self.harness.prepare_uv_script(_PYTHON, activate=False),
        )
        self._python, self._harness_python = worker[0], relay[0]

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        if self._closed:
            raise SandboxError("LAN execution transport is closed")
        owner = await runtime.backend.exec(
            runtime,
            [self._python, "-I", "-u", "-c", _SERVER],
            env,
        )
        secret, process_id = secrets.token_urlsafe(32), uuid.uuid4().hex
        try:
            await owner.write(
                (
                    json.dumps(
                        {"secret": secret, "process_id": process_id, "argv": argv}
                    )
                    + "\n"
                ).encode()
            )
            async with asyncio.timeout(30):
                output = b""
                while b"\n" not in output:
                    output += await anext(owner.stdout)
                port = json.loads(output.split(b"\n", 1)[0])["port"]
            if self._closed:
                raise SandboxError("LAN execution transport closed during startup")
            process = LanProcess(
                owner,
                self.harness,
                self._harness_python,
                {
                    "url": f"ws://{self.host}:{port}/execution",
                    "secret": secret,
                    "process_id": process_id,
                },
            )
            self._processes.append(process)
            return process
        except BaseException:
            with contextlib.suppress(Exception):
                await owner.kill()
            raise

    async def close(self) -> None:
        self._closed = True
        processes, self._processes = self._processes, []
        results = await asyncio.gather(
            *(process.aclose() for process in processes), return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to close LAN execution process - %s: %s",
                    type(result).__name__,
                    clip(str(result)),
                    exc_info=(
                        (type(result), result, result.__traceback__)
                        if logger.isEnabledFor(logging.DEBUG)
                        else None
                    ),
                )
