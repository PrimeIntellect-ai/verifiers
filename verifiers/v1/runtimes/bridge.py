"""Host-owned registration and stdio transport for credential-free workers."""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
from dataclasses import dataclass, field

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.aio import run_shielded
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_RECEIPTS = 100_000
_MAX_RECEIPT_BYTES = 128 * 1024 * 1024


@dataclass
class _Worker:
    name: str | None
    process: RuntimeProcess
    stdout_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stderr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    eof: set[str] = field(default_factory=set)


class ExecutionBridge:
    """Only host harness setup can register commands and their environment.

    Native harnesses can open only registered workers. The trusted tool adapter
    receives a separate key to launch approved commands as the task user.
    Neither capability enters the workspace. Receipts preserve calls across retries.
    """

    def __init__(self, runtime: Runtime):
        self.runtime = runtime
        self.commands: dict[str, tuple[list[str], dict[str, str]]] = {}
        self.workers: dict[str, _Worker] = {}
        self.completed: dict[str, int] = {}
        self.adapter_key = secrets.token_urlsafe(32)
        self.receipts: dict[str, tuple[bytes, asyncio.Task]] = {}
        self._open_lock = asyncio.Lock()
        self._receipt_bytes = 0
        self.closed = False

    def _retain(self, size: int) -> None:
        # Never evict unacknowledged reads: retrying one could consume new bytes.
        # End an oversized transport instead of letting worker output exhaust host RAM.
        if self._receipt_bytes + size > _MAX_RECEIPT_BYTES:
            raise HarnessError("split execution exceeded its replay buffer budget")
        self._receipt_bytes += size

    def register(self, name: str, argv: list[str], env: dict[str, str]) -> None:
        if self.closed or name in self.commands or not argv:
            raise HarnessError(f"invalid split worker registration {name!r}")
        self.commands[name] = (list(argv), dict(env))

    async def _exec(self, argv: list[str], env: dict[str, str]) -> RuntimeProcess:
        return await self.runtime.exec(argv, env)

    async def handle(self, body: dict) -> dict:
        if self.closed:
            raise HarnessError("split execution bridge is closed")
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise HarnessError("split execution requires a request_id")
        # Lifecycle calls are idempotent and must remain available after a data
        # receipt limit is reached. Destructive reads still retain every receipt.
        if body.get("operation") in {"kill", "terminate", "poll", "wait"}:
            return await self._handle(body)
        # The adapter acknowledges the previous chunk when it asks for the next.
        # Keep the receipt so a late duplicate cannot consume another chunk.
        acknowledged = body.get("acknowledge_read")
        if isinstance(acknowledged, str) and (prior := self.receipts.get(acknowledged)):
            _, task = prior
            if task.done() and not task.cancelled() and task.exception() is None:
                reply = task.result()
                if data := reply.get("data"):
                    self._receipt_bytes -= len(data)
                    reply.clear()
                    reply["acknowledged"] = True
        encoded = json.dumps(body, sort_keys=True).encode()
        fingerprint = hashlib.sha256(encoded).digest()
        receipt = self.receipts.get(request_id)
        if receipt is None:
            if len(self.receipts) >= _MAX_RECEIPTS:
                raise HarnessError(
                    "split execution exceeded its request receipt budget"
                )
            size = len(encoded)
            self._retain(size + len(fingerprint))

            async def execute():
                try:
                    return await self._handle(body)
                finally:
                    self._receipt_bytes -= size

            task = asyncio.create_task(execute())
            self.receipts[request_id] = (fingerprint, task)
        else:
            previous, task = receipt
            if previous != fingerprint:
                raise HarnessError("split execution request_id was reused")
        del encoded
        # A lost HTTP request must not lose bytes already read from the worker.
        return await asyncio.shield(task)

    async def _handle(self, body: dict) -> dict:
        operation = body.get("operation")
        process_id = body.get("process_id")
        if not isinstance(process_id, str) or not process_id:
            raise HarnessError("split execution requires a process_id")
        if operation in {"open", "exec"}:
            if operation == "exec":
                key = body.get("adapter_key")
                if not isinstance(key, str) or not secrets.compare_digest(
                    key.encode(), self.adapter_key.encode()
                ):
                    raise HarnessError(
                        "workspace commands require the tool adapter capability"
                    )
                argv, env = body.get("argv"), body.get("env")
                if (
                    not isinstance(argv, list)
                    or not argv
                    or not all(isinstance(arg, str) for arg in argv)
                ):
                    raise HarnessError("invalid workspace command arguments")
                if not isinstance(env, dict) or not all(
                    isinstance(v, str) for v in env.values()
                ):
                    raise HarnessError("invalid workspace command environment")
                name = None
            else:
                name = body.get("worker")
                if not isinstance(name, str) or name not in self.commands:
                    raise HarnessError(
                        "split execution requested an unregistered worker"
                    )
                argv, env = self.commands[name]
            async with self._open_lock:
                existing = self.workers.get(process_id)
                if (
                    operation == "open"
                    and existing is not None
                    and existing.name == name
                ):
                    return {"process_id": process_id}
                if process_id in self.workers or process_id in self.completed:
                    raise HarnessError("split process_id was reused")

                async def start() -> None:
                    process = await self._exec(argv, env)
                    self.workers[process_id] = _Worker(name, process)

                await run_shielded(start())
            return {"process_id": process_id}
        if process_id in self.completed:
            if operation == "read" and body.get("stream") in ("stdout", "stderr"):
                return {"eof": True, "data": None}
            if operation in {"wait", "poll", "kill", "terminate"}:
                return {"exit_code": self.completed[process_id]}
            raise HarnessError("workspace process has exited")
        worker = self.workers.get(process_id)
        if worker is None:
            raise HarnessError("split execution requested an unknown process")
        if operation == "write":
            try:
                data = base64.b64decode(body["data"], validate=True)
            except (KeyError, ValueError, TypeError) as error:
                raise HarnessError("invalid split stdin payload") from error
            async with worker.stdin_lock:
                await worker.process.write(data)
            return {}
        if operation == "read":
            stream = body.get("stream")
            if stream not in ("stdout", "stderr"):
                raise HarnessError("invalid split output stream")
            async with getattr(worker, f"{stream}_lock"):
                chunk = (
                    None
                    if stream in worker.eof
                    else await anext(getattr(worker.process, stream), None)
                )
                if chunk is None:
                    worker.eof.add(stream)
                    if len(worker.eof) == 2:
                        await self._retire(
                            process_id, worker, await worker.process.poll()
                        )
            self._retain(((len(chunk) + 2) // 3) * 4 if chunk is not None else 0)
            return {
                "data": base64.b64encode(chunk).decode() if chunk is not None else None,
                "eof": chunk is None,
            }
        if operation == "terminate":
            await worker.process.terminate()
            return {}
        if operation in {"wait", "poll", "kill"}:
            if operation == "kill":
                await worker.process.kill()
            status = (
                await worker.process.poll()
                if operation == "poll"
                else await worker.process.wait()
            )
            await self._retire(process_id, worker, status)
            return {"exit_code": status}
        raise HarnessError(f"unknown split execution operation {operation!r}")

    async def _retire(
        self, process_id: str, worker: _Worker, status: int | None
    ) -> None:
        if (
            status is not None
            and worker.name is None
            and len(worker.eof) == 2
            and self.workers.pop(process_id, None) is not None
        ):
            self.completed[process_id] = status
            await worker.process.abandon()

    async def abandon(self) -> None:
        """Cancel local bridge tasks without remote RPCs when VMs are being deleted."""
        if self.closed:
            return
        self.closed = True
        tasks = [task for _, task in self.receipts.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(worker.process.abandon() for worker in self.workers.values()),
            return_exceptions=True,
        )
        self.receipts.clear()
        self.workers.clear()
        self.completed.clear()
        self._receipt_bytes = 0

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        tasks = [task for _, task in self.receipts.values()]
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(*tasks, return_exceptions=True)
                processes = [worker.process for worker in self.workers.values()]
                await asyncio.gather(
                    *(proc.kill() for proc in processes),
                    return_exceptions=True,
                )
                await asyncio.gather(
                    *(proc.wait() for proc in processes),
                    return_exceptions=True,
                )
        except TimeoutError:
            logger.warning(
                "Timed out waiting for split workers to exit",
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
        self.receipts.clear()
        self.workers.clear()
        self.completed.clear()
        self._receipt_bytes = 0
