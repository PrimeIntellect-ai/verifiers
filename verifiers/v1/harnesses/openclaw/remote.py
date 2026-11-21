"""OpenClaw's shell/process tools executed through the task Runtime."""

import asyncio
import base64
import fnmatch
import json
import shlex
import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verifiers.v1.harnesses.utils.remote import NativeRemoteTools
from verifiers.v1.runtimes import Runtime, RuntimeProcess
from verifiers.v1.types import ToolCall, ToolMessage


class _Exec(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    command: str
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    yieldMs: float = Field(default=10000, ge=0)
    awaitResults: bool = False
    background: bool = False
    timeoutSeconds: float = Field(default=1800, ge=0)
    pty: bool = False


class _Process(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    action: Literal[
        "list",
        "poll",
        "log",
        "write",
        "send-keys",
        "submit",
        "paste",
        "kill",
        "clear",
        "remove",
    ]
    sessionId: str | None = None
    data: str = ""
    keys: list[str] = Field(default_factory=list)
    hex: list[str] = Field(default_factory=list)
    literal: str = ""
    text: str = ""
    bracketed: bool = False
    eof: bool = False
    offset: int | None = Field(default=None, ge=0)
    limit: int = Field(default=200, ge=0)
    timeout: float = Field(default=0, ge=0)


# Encode stdin chunks so EOF can close the child's pipe without closing the
# provider's control stream. A PTY receives the terminal's EOF character.
_WORKER = r"""
if [[ -n $1 ]]; then cd -- "$1" || exit; fi
exec 0< <(
    while IFS= read -r data; do
        if [[ $data == . ]]; then
            [[ $2 == pty ]] || break
            printf '\004'
        elif [[ -n $data ]]; then
            printf '%s' "$data" | base64 -d || break
        fi
    done
)
reader=$!
trap 'kill "$reader" 2>/dev/null || :' EXIT
if [[ $2 != pty ]]; then
    bash -c "$3" 2>&1
elif [[ $OSTYPE == darwin* ]] && command -v script >/dev/null; then
    script -qe /dev/null bash -c "$3"
elif script --version 2>/dev/null | grep -q util-linux; then
    script -qe -c "$3" /dev/null
else
    # Minimal images may have Python's standard library in place of script(1).
    python3 -c 'import os, pty, sys; sys.exit(os.waitstatus_to_exitcode(pty.spawn(sys.argv[1:])))' bash -c "$3"
fi
"""


@dataclass
class _Session:
    process: RuntimeProcess
    command: str
    output: bytearray = field(default_factory=bytearray)
    cursor: int = 0
    done: asyncio.Task | None = None
    timer: asyncio.Task | None = None

    async def collect(self) -> int:
        async def drain(stream):
            async for chunk in stream:
                self.output.extend(chunk)
                if len(self.output) > 1024 * 1024:
                    removed = len(self.output) - 1024 * 1024
                    del self.output[:removed]
                    self.cursor = max(0, self.cursor - removed)

        readers = [
            asyncio.create_task(drain(stream))
            for stream in (self.process.stdout, self.process.stderr)
        ]
        try:
            code = await self.process.wait()
            await asyncio.gather(*readers)
            return code
        finally:
            for reader in readers:
                if not reader.done():
                    reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)

    async def expire(self, timeout: float) -> None:
        await asyncio.sleep(timeout)
        if self.done is not None and not self.done.done():
            await self.process.kill()

    async def wait(self, timeout: float | None) -> None:
        assert self.done is not None
        if timeout is None:
            await asyncio.shield(self.done)
        elif timeout > 0:
            try:
                await asyncio.wait_for(asyncio.shield(self.done), timeout)
            except TimeoutError:
                pass

    def result(self, session_id: str, *, log: _Process | None = None) -> dict:
        finished = self.done is not None and self.done.done()
        if log is None:
            content = bytes(self.output[self.cursor :]).decode(errors="replace")
            self.cursor = len(self.output)
        else:
            lines = self.output.decode(errors="replace").splitlines()
            offset = (
                log.offset if log.offset is not None else max(0, len(lines) - log.limit)
            )
            content = "\n".join(lines[offset : offset + log.limit])
        return {
            "sessionId": session_id,
            "status": "completed" if finished else "running",
            "exitCode": self.done.result() if finished else None,
            "output": content,
        }


@dataclass
class OpenClawRemoteTools:
    runtime: Runtime
    disabled_tools: tuple[str, ...] = ()
    files: NativeRemoteTools = field(init=False)
    sessions: dict[str, _Session] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.files = NativeRemoteTools(
            self.runtime, "openclaw", disabled_tools=self.disabled_tools
        )

    async def aclose(self) -> None:
        await self.files.aclose()
        for session in self.sessions.values():
            if session.timer is not None:
                session.timer.cancel()
            await session.process.kill()
        await asyncio.gather(
            *(
                session.done
                for session in self.sessions.values()
                if session.done is not None
            ),
            *(
                session.timer
                for session in self.sessions.values()
                if session.timer is not None
            ),
            return_exceptions=True,
        )

    async def __call__(self, call: ToolCall) -> ToolMessage:
        if any(
            fnmatch.fnmatchcase(call.name, pattern) for pattern in self.disabled_tools
        ):
            return ToolMessage(
                tool_call_id=call.id,
                name=call.name,
                content=f"error: tool {call.name!r} is disabled",
            )
        if call.name not in {"exec", "process"}:
            return await self.files(call)
        try:
            if call.name == "exec":
                args = _Exec.model_validate_json(call.arguments)
                command = shlex.join(
                    [
                        "env",
                        "--",
                        *(f"{k}={v}" for k, v in args.env.items()),
                        "bash",
                        "-c",
                        args.command,
                    ]
                )
                process = await self.runtime.exec(
                    [
                        "bash",
                        "-c",
                        _WORKER,
                        "openclaw",
                        args.workdir or "",
                        "pty" if args.pty else "pipe",
                        command,
                    ],
                    {},
                )
                session = _Session(process, args.command)
                self.sessions[session_id := uuid.uuid4().hex] = session
                session.done = asyncio.create_task(session.collect())
                if args.timeoutSeconds:
                    session.timer = asyncio.create_task(
                        session.expire(args.timeoutSeconds)
                    )
                await session.wait(
                    0
                    if args.background
                    else None
                    if args.awaitResults
                    else args.yieldMs / 1000
                )
                result = session.result(session_id)
            else:
                result = await self.control(
                    _Process.model_validate_json(call.arguments)
                )
            content = json.dumps(result)
        except (ValidationError, ValueError) as error:
            content = f"error: invalid {call.name} arguments: {error}"
        return ToolMessage(tool_call_id=call.id, name=call.name, content=content)

    async def control(self, args: _Process) -> dict:
        if args.action == "list":
            return {
                "sessions": [
                    {
                        "sessionId": key,
                        "command": session.command,
                        "running": not session.done.done(),
                    }
                    for key, session in self.sessions.items()
                ]
            }
        if args.sessionId not in self.sessions:
            raise ValueError("unknown sessionId")
        session = self.sessions[args.sessionId]
        if args.action in {"kill", "remove"}:
            await session.process.kill()
            await session.wait(None)
        if args.action in {"clear", "remove"}:
            if not session.done.done():
                raise ValueError("cannot clear a running session")
            if session.timer is not None:
                session.timer.cancel()
            del self.sessions[args.sessionId]
            return {"sessionId": args.sessionId, "status": "removed"}
        if args.action in {"write", "send-keys", "submit", "paste"}:
            if session.done.done():
                raise ValueError("session has exited")
            data = args.data.encode()
            if args.action == "submit":
                data = b"\r"
            elif args.action == "paste":
                data = args.text.encode()
                if args.bracketed:
                    data = b"\x1b[200~" + data + b"\x1b[201~"
            elif args.action == "send-keys":
                named = {
                    "Enter": "\r",
                    "Return": "\r",
                    "Tab": "\t",
                    "Space": " ",
                    "Escape": "\x1b",
                    "Esc": "\x1b",
                    "BSpace": "\x7f",
                    "Up": "\x1b[A",
                    "Down": "\x1b[B",
                    "Right": "\x1b[C",
                    "Left": "\x1b[D",
                }
                pieces = []
                for key in args.keys:
                    if key in named:
                        pieces.append(named[key])
                    elif (
                        key.startswith("C-")
                        and len(key) == 3
                        and "@" <= key[-1].upper() <= "_"
                    ):
                        pieces.append(chr(ord(key[-1].upper()) & 31))
                    elif len(key) == 1:
                        pieces.append(key)
                    else:
                        raise ValueError(f"unsupported key {key!r}; use literal or hex")
                data = (
                    "".join(pieces).encode()
                    + bytes.fromhex("".join(args.hex))
                    + args.literal.encode()
                )
            await session.process.write(
                base64.b64encode(data) + b"\n" + (b".\n" if args.eof else b"")
            )
        if args.action == "poll":
            await session.wait(min(args.timeout, 30000) / 1000)
        return session.result(
            args.sessionId, log=args if args.action == "log" else None
        )
