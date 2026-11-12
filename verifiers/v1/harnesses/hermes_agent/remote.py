"""Hermes' native tool registry in a persistent, credential-free B worker."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field

from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.types import ToolCall, ToolMessage

# Provider-backed orchestration belongs to A. Only native execution tools are
# exposed here; unknown tools produce a result without attempting local fallback.
_TOOLS = frozenset(
    {
        "terminal",
        "process",
        "process_manage",
        "read_file",
        "write_file",
        "patch",
        "search_files",
        "todo",
        "todo_write",
        "todo_read",
        "read_image",
    }
)
_WORKER = r"""
import contextlib, json, sys
with contextlib.redirect_stdout(sys.stderr):
    from model_tools import handle_function_call
for line in sys.stdin:
    request = json.loads(line)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = handle_function_call(request["name"], request["arguments"], "verifiers-split", tool_call_id=request["id"])
        content = result if isinstance(result, str) else json.dumps(result)
    except Exception as error:
        content = f"error: {type(error).__name__}: {error}"
    print(request["marker"] + json.dumps({"content": content}), flush=True)
"""


@dataclass
class HermesRemoteTools:
    runtime: Runtime
    python: str
    home: str
    _worker: RuntimeProcess | None = None
    _stderr: asyncio.Task | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _buffer: bytes = b""

    async def aclose(self) -> None:
        if self._worker is not None:
            await self._worker.terminate()
            self._worker = None
        if self._stderr is not None:
            self._stderr.cancel()
            await asyncio.gather(self._stderr, return_exceptions=True)

    async def __call__(self, call: ToolCall) -> ToolMessage:
        content = f"error: split Hermes has no execution-side implementation for {call.name!r}"
        if call.name not in _TOOLS:
            return ToolMessage(tool_call_id=call.id, name=call.name, content=content)
        try:
            arguments = json.loads(call.arguments or "{}")
            if not isinstance(arguments, dict):
                raise TypeError("arguments must be an object")
        except (ValueError, TypeError) as error:
            return ToolMessage(
                tool_call_id=call.id, name=call.name, content=f"error: {error}"
            )
        async with self._lock:
            if self._worker is None:
                self._worker = await self.runtime.exec(
                    [self.python, "-P", "-u", "-c", _WORKER],
                    {"HERMES_HOME": self.home, "TERMINAL_ENV": "local"},
                )

                async def drain():
                    assert self._worker is not None
                    async for _ in self._worker.stderr:
                        pass

                self._stderr = asyncio.create_task(drain())
            marker = f"__vf_{uuid.uuid4().hex}:"
            await self._worker.write(
                (
                    json.dumps(
                        {
                            "name": call.name,
                            "arguments": arguments,
                            "id": call.id,
                            "marker": marker,
                        }
                    )
                    + "\n"
                ).encode()
            )
            try:
                async with asyncio.timeout(3600):
                    while True:
                        if b"\n" not in self._buffer:
                            self._buffer += await anext(self._worker.stdout)
                            continue
                        line, self._buffer = self._buffer.split(b"\n", 1)
                        if line.startswith(marker.encode()):
                            content = json.loads(line[len(marker) :])["content"]
                            break
            except (TimeoutError, StopAsyncIteration):
                await self.aclose()
                content = "error: remote Hermes tool worker stopped or timed out"
        return ToolMessage(tool_call_id=call.id, name=call.name, content=content)
