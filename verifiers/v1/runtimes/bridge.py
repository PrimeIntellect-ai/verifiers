"""Host-owned stdio for trusted supervisors talking to credential-free workers."""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.runtime import Runtime
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)

_MAX_RECEIPTS = 100_000
_MAX_RECEIPT_BYTES = 128 * 1024 * 1024


@dataclass
class _Worker:
    name: str
    process: RuntimeProcess
    stdout_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stderr_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stdin_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ExecutionBridge:
    """Only host harness setup can register commands and their environment.

    The A-side transport authenticates with its existing rollout capability. B
    gets an ordinary stdin/stdout pipe and never receives that capability or a
    network endpoint. Request receipts retain destructive reads across retries.
    """

    def __init__(self, runtime: Runtime, *, setup_runtime: Runtime | None = None):
        self.runtime = runtime
        self.setup_runtime = setup_runtime or runtime
        self.commands: dict[str, tuple[list[str], dict[str, str]]] = {}
        self.workers: dict[str, _Worker] = {}
        self.receipts: dict[str, tuple[str, asyncio.Task]] = {}
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

    async def handle(self, body: dict) -> dict:
        if self.closed:
            raise HarnessError("split execution bridge is closed")
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise HarnessError("split execution requires a request_id")
        fingerprint = json.dumps(body, sort_keys=True)
        receipt = self.receipts.get(request_id)
        if receipt is None:
            if len(self.receipts) >= _MAX_RECEIPTS:
                raise HarnessError(
                    "split execution exceeded its request receipt budget"
                )
            self._retain(len(fingerprint.encode()))
            task = asyncio.create_task(self._handle(body))
            self.receipts[request_id] = (fingerprint, task)
        else:
            previous, task = receipt
            if previous != fingerprint:
                raise HarnessError("split execution request_id was reused")
        # A lost HTTP request must not lose bytes already read from the worker.
        return await asyncio.shield(task)

    async def _handle(self, body: dict) -> dict:
        operation = body.get("operation")
        process_id = body.get("process_id")
        if not isinstance(process_id, str) or not process_id:
            raise HarnessError("split execution requires a process_id")
        if operation == "open":
            name = body.get("worker")
            if not isinstance(name, str) or name not in self.commands:
                raise HarnessError("split execution requested an unregistered worker")
            async with self._open_lock:
                if process_id not in self.workers:
                    argv, env = self.commands[name]
                    process = await self.runtime.exec(argv, env)
                    self.workers[process_id] = _Worker(name, process)
                elif self.workers[process_id].name != name:
                    raise HarnessError("split process_id was reused for another worker")
            return {"process_id": process_id}
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
                chunk = await anext(getattr(worker.process, stream), None)
            self._retain(((len(chunk) + 2) // 3) * 4 if chunk is not None else 0)
            return {
                "data": base64.b64encode(chunk).decode() if chunk is not None else None,
                "eof": chunk is None,
            }
        if operation == "wait":
            return {"exit_code": await worker.process.wait()}
        if operation == "kill":
            await worker.process.kill()
            return {"exit_code": await worker.process.wait()}
        raise HarnessError(f"unknown split execution operation {operation!r}")

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
                await asyncio.gather(
                    *(worker.process.kill() for worker in self.workers.values()),
                    return_exceptions=True,
                )
                await asyncio.gather(
                    *(worker.process.wait() for worker in self.workers.values()),
                    return_exceptions=True,
                )
        except TimeoutError:
            logger.warning(
                "Timed out waiting for split workers to exit",
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
        self.receipts.clear()
        self.workers.clear()
        self._receipt_bytes = 0
