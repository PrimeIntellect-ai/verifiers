"""The ordinary RuntimeProcess contract carried over a deployment's private LAN.

Provider stdio owns startup and shutdown. A process-scoped capability permits only
stdio and lifecycle operations on one registered worker. It grants no model,
policy, filesystem-management, or command-registration capability.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import uuid

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.lan_stream import WIRE_SOURCE, AsyncLanStream
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.logging import clip, get_logger

logger = get_logger(__name__)

# Run inside the same user, environment, cwd and filesystem view as the command.
# The capability is minted locally and returned over provider stdout, never in env.
_SERVER = (
    WIRE_SOURCE
    + r"""
import base64, concurrent.futures, hmac, os, secrets, signal, socket, subprocess, sys, threading

secret = secrets.token_urlsafe(32)
process_id = sys.argv[1]
process = subprocess.Popen(sys.argv[2:], stdin=subprocess.PIPE,
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

def handle_connection(conn):
    send_lock = threading.Lock()
    try:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(10)
        auth = _read_frame(conn)
        provided = auth.get("secret", "") if isinstance(auth, dict) else ""
        if not isinstance(provided, str) or not hmac.compare_digest(provided, secret):
            _send_frame(conn, {"ok": False, "error": "Unauthorized"})
            return
        _send_frame(conn, {"ok": True})
        conn.settimeout(None)
    except Exception:
        conn.close()
        return

    with connections_lock:
        connections.add(conn)
    slots = threading.BoundedSemaphore(8)

    def reply(body):
        try:
            result = {"result": handle(body)}
        except Exception as error:
            result = {"error": str(error)}
        try:
            with send_lock:
                _send_frame(conn, {"id": body.get("request_id"), **result})
        except OSError:
            pass

    def work(body):
        try:
            reply(body)
        finally:
            slots.release()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                body = _read_frame(conn)
                if not isinstance(body, dict):
                    break
                # Termination must remain available while reads/wait are blocked.
                if body.get("operation") in ("kill", "terminate", "poll"):
                    reply(body)
                elif slots.acquire(blocking=False):
                    pool.submit(work, body)
                else:
                    break
    except (EOFError, OSError, ValueError):
        pass
    finally:
        with connections_lock:
            connections.discard(conn)
        try:
            conn.close()
        except OSError:
            pass

server = socket.create_server(("0.0.0.0", 0), backlog=16)

def accept_loop():
    while True:
        try:
            conn, _ = server.accept()
        except OSError:
            return
        threading.Thread(target=handle_connection, args=(conn,), daemon=True).start()

threading.Thread(target=accept_loop, daemon=True).start()
print(json.dumps({"port": server.getsockname()[1], "secret": secret}), flush=True)
try:
    # Only the owning evaluator's provider stream can end the listener. A native
    # client may finish while its output still needs draining or replaying.
    sys.stdin.readline()
finally:
    stop(signal.SIGKILL)
    try:
        server.close()
    except OSError:
        pass
    with connections_lock:
        active = list(connections)
    for conn in active:
        try:
            conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            conn.close()
        except OSError:
            pass
"""
)


class LanProcess(RuntimeProcess):
    def __init__(self, owner: RuntimeProcess | None, endpoint: dict) -> None:
        self.owner = owner
        self.execution_endpoint = endpoint
        self._direct = AsyncLanStream(endpoint)
        self._stdin_lock = asyncio.Lock()
        self._closed = False
        self._finished: set[str] = set()
        self._exit_code: int | None = None
        self.stdout = self._stream("stdout")
        self.stderr = self._stream("stderr")

    async def _rpc(self, operation: str, **values) -> dict:
        if self._closed:
            raise SandboxError("LAN execution process is closed")
        try:
            result = await self._direct.request(operation, **values)
        except Exception as error:
            raise SandboxError(str(error)) from error
        if operation == "wait":
            self._exit_code = result["exit_code"]
            self._finished.add("wait")
        elif operation == "read" and result.get("eof"):
            self._finished.add(values["stream"])
        if len(self._finished) == 3:
            await self.aclose()
        return result

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
        if not self._closed:
            await self._rpc("terminate")

    async def kill(self) -> None:
        if not self._closed:
            await self._rpc("kill")

    async def abandon(self) -> None:
        """Drop local streams without remote RPCs when the owning VM is stopping."""
        if self._closed:
            return
        self._closed = True
        await self._direct.aclose()
        if self.owner is not None:
            await self.owner.abandon()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._direct.aclose()
            if self.owner is not None:
                await self.owner.write(b"close\n")
                async with asyncio.timeout(5):
                    await self.owner.wait()
        finally:
            if self.owner is not None:
                with contextlib.suppress(Exception):
                    await self.owner.kill()


class LanExecution:
    """Select LAN once for the deployment, underneath every harness adapter."""

    def __init__(self, harness: Runtime, host: str) -> None:
        self.harness = harness
        self.host = host
        self._python = ""
        self._processes: list[LanProcess] = []
        self._closed = False

    async def start(self, workspace: Runtime) -> None:
        self._python = await workspace.resolve_python()

    async def exec(
        self, runtime: Runtime, argv: list[str], env: dict[str, str]
    ) -> LanProcess:
        if self._closed:
            raise SandboxError("LAN execution transport is closed")
        python = self._python or await runtime.resolve_python()
        process_id = uuid.uuid4().hex
        owner = await runtime.backend.exec(
            runtime,
            [python, "-I", "-u", "-c", _SERVER, process_id, *argv],
            env,
        )
        try:
            async with asyncio.timeout(30):
                output = b""
                while b"\n" not in output:
                    output += await anext(owner.stdout)
                announced = json.loads(output.split(b"\n", 1)[0])
                port, secret = announced["port"], announced["secret"]
            if self._closed:
                raise SandboxError("LAN execution transport closed during startup")
            process = LanProcess(
                owner,
                {
                    "url": f"tcp://{self.host}:{port}",
                    "host": self.host,
                    "port": port,
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

    async def abandon(self) -> None:
        self._closed = True
        processes, self._processes = self._processes, []
        await asyncio.gather(
            *(process.abandon() for process in processes), return_exceptions=True
        )

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
