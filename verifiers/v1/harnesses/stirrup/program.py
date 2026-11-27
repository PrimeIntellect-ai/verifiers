# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "stirrup[mcp] @ https://github.com/ArtificialAnalysis/Stirrup/archive/{version}.tar.gz",
#     "mcp==1.30.0",
#     "agent-client-protocol==1.0.0",
# ]
# ///
"""Expose Stirrup's agent loop as a rollout-scoped ACP agent over stdio."""

import argparse
import asyncio
import fcntl
import json
import os
import select
import signal
import struct
import sys
import termios
import threading
from contextlib import redirect_stdout, suppress
from pathlib import Path
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    RequestError,
    run_agent,
    stdio_streams,
)
from acp.schema import (
    AgentCapabilities,
    AgentMessageChunk,
    CloseSessionResponse,
    InitializeResponse,
    McpCapabilities,
    NewSessionResponse,
    PromptResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    TextContentBlock,
)
from openai import APIError
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.core.agent import SessionAgent
from stirrup.core.exceptions import OutputTokenLimitError
from stirrup.core.models import UserMessage
from stirrup.tools import LocalCodeExecToolProvider, WebToolProvider
from stirrup.tools.code_backends.base import CommandResult
from stirrup.tools.mcp import MCPConfig, MCPToolProvider

COMMAND_CLEANUP_TIMEOUT = 5.0
OUTPUT_GRACE = 0.5
"""Seconds to keep reading after bash exits, e.g. for a process substitution."""
PIPE_READ_SIZE = 65536
MAX_KEPT_BYTES = 1 << 20
"""Bytes of each stream a call keeps, as its first and last halves."""
BACKGROUND_NOTE = "[a process started by this command is still running and holds its output; later output is not shown]"


class OutputPipe:
    """A command's stdout or stderr, read on the event loop while the command runs."""

    def __init__(self):
        self.fd, self.write_fd = os.pipe()
        self.head = bytearray()
        self.tail = bytearray()
        self.size = 0  # Bytes read in all, kept or not.
        self.held_open = False
        self.loop = asyncio.get_running_loop()
        self.closed = self.loop.create_future()  # Done once every writer has closed it.
        try:
            self.loop.add_reader(self.fd, self.read)
        except BaseException:
            os.close(self.fd)
            os.close(self.write_fd)
            raise

    def read(self):
        chunk = os.read(self.fd, PIPE_READ_SIZE)
        self.keep(chunk)
        if not chunk:
            self.loop.remove_reader(self.fd)
            self.closed.set_result(None)

    def keep(self, chunk: bytes):
        """Keep the first and last halves of the output; drop the middle beyond them."""
        self.size += len(chunk)
        half = MAX_KEPT_BYTES // 2
        room = half - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        self.tail += chunk
        if len(self.tail) > 2 * half:  # Trimmed in batches, not on every read.
            del self.tail[:-half]

    def text(self) -> str:
        half = MAX_KEPT_BYTES // 2
        omitted = self.size - len(self.head) - min(len(self.tail), half)
        if omitted <= 0:
            return (self.head + self.tail).decode(errors="replace")
        # Cut at character boundaries.
        head = self.head[: _complete_utf8(self.head)]
        tail = self.tail[-half:]
        start = 0
        while start < 3 and start < len(tail) and tail[start] & 0xC0 == 0x80:
            start += 1
        tail = tail[start:]
        omitted += len(self.head) - len(head) + start
        head_text = head.decode(errors="replace")
        tail_text = tail.decode(errors="replace")
        # The marker gets a line of its own, without adding a blank line.
        before = "" if head_text.endswith("\n") else "\n"
        after = "" if tail_text.startswith(("\n", "\r\n")) else "\n"
        unit = "byte" if omitted == 1 else "bytes"
        return (
            f"{head_text}{before}[... {omitted} {unit} of output omitted ...]"
            f"{after}{tail_text}"
        )

    def close_write_end(self):
        if self.write_fd is not None:
            fd, self.write_fd = self.write_fd, None
            with suppress(OSError):
                os.close(fd)

    def finish(self, *, drain: bool = True) -> str:
        """Take the output waiting in the pipe now and stop collecting; never raises.

        With `drain`, a pipe still held open is read to EOF off the loop; else closed.
        """
        if self.fd is not None:
            fd, self.fd = self.fd, None
            open_ = False
            try:
                self.loop.remove_reader(fd)
                if not self.closed.done():
                    pending = _pending(fd)
                    while pending > 0 and (chunk := os.read(fd, pending)):
                        self.keep(chunk)
                        pending -= len(chunk)
                    open_ = not _at_end(fd)
            except Exception:  # noqa: BLE001 - unreadable: keep what was read
                open_ = True
            self.held_open = open_
            if open_ and drain:
                try:
                    threading.Thread(
                        target=_discard_until_eof, args=(fd,), daemon=True
                    ).start()
                except Exception:  # noqa: BLE001 - the process may get SIGPIPE
                    open_ = False
            if not (open_ and drain):
                with suppress(OSError):
                    os.close(fd)
        self.close_write_end()
        return self.text()


def _complete_utf8(data: bytes | bytearray) -> int:
    """Length of `data` without a UTF-8 character cut off at its end."""
    for i in range(len(data) - 1, max(len(data) - 4, -1), -1):
        byte = data[i]
        if byte < 0x80:
            break
        if byte >= 0xC0:  # The first byte of a character.
            length = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            return i if len(data) - i < length else len(data)
    return len(data)


def _pending(fd: int) -> int:
    """Bytes waiting in a pipe right now."""
    return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0" * 4))[0]


def _at_end(fd: int) -> bool:
    """Whether every writer has closed the pipe; late bytes read here are dropped."""
    poller = select.poll()  # select.select rejects fds >= 1024
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(0)) and not os.read(fd, PIPE_READ_SIZE)


def _discard_until_eof(fd: int) -> None:
    """Read a held pipe to EOF, so its writer neither blocks nor gets SIGPIPE."""
    try:
        # uvloop may leave the fd non-blocking, where an empty read is not EOF.
        os.set_blocking(fd, True)
        while os.read(fd, PIPE_READ_SIZE):
            pass
    except OSError:
        pass
    finally:
        with suppress(OSError):
            os.close(fd)


class RuntimeCodeExecToolProvider(LocalCodeExecToolProvider):
    """Borrow the container workspace; its lifetime belongs to Verifiers."""

    def __init__(self, cwd: str):
        super().__init__()
        self.workspace = Path(cwd)

    async def __aenter__(self):
        # The pinned local backend only exposes temporary directories. The
        # runtime already provides isolation and owns this workspace's lifetime.
        self._temp_dir = self.workspace
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._temp_dir = None

    def _resolve_and_validate_path(self, path: str) -> Path:
        # Commands and file helpers share the container filesystem. The cwd is
        # only the base for relative paths; Verifiers owns sandbox isolation.
        if self._temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        resolved = Path(path)
        return resolved if resolved.is_absolute() else self._temp_dir / resolved

    async def run_command(self, cmd, *, timeout=None):
        # The pinned local backend cleans up on timeouts, but not task
        # cancellation. ACP cancellation must also stop the command's children.
        timeout = self._shell_timeout if timeout is None else timeout
        process = None
        completed = False
        stdout = stderr = None
        try:
            try:
                stdout = OutputPipe()
                stderr = OutputPipe()
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    cmd,
                    cwd=self.workspace,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=stdout.write_fd,
                    stderr=stderr.write_fd,
                    start_new_session=True,
                )
            finally:
                for pipe in (stdout, stderr):
                    if pipe is not None:
                        pipe.close_write_end()
            try:
                # Wait for bash, not for EOF on its pipes.
                await asyncio.wait_for(process.wait(), timeout)
            except TimeoutError:
                pass  # The command is killed below, and the timeout result follows.
            else:
                # A process bash did not wait for may still be writing.
                await asyncio.wait([stdout.closed, stderr.closed], timeout=OUTPUT_GRACE)
                # Only now, so a cancellation during the grace kills the group.
                completed = True
                out, err = stdout.finish(), stderr.finish()
                if stdout.held_open or stderr.held_open:
                    # Status messages go to stderr, as for a timeout.
                    if err and not err.endswith("\n"):
                        err += "\n"
                    err += BACKGROUND_NOTE
                return CommandResult(
                    exit_code=process.returncode, stdout=out, stderr=err
                )
        except Exception as error:  # noqa: BLE001 - shell execution failures are recoverable tool results
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(error),
                error_kind="execution_error",
            )
        finally:
            try:
                if process is not None and not completed:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    with suppress(TimeoutError):
                        await asyncio.wait_for(process.wait(), COMMAND_CLEANUP_TIMEOUT)
            finally:
                # Pipes of a killed command are closed, not drained.
                for pipe in (stdout, stderr):
                    if pipe is not None:
                        pipe.finish(drain=completed)
        # Timed out: exit 124, as timeout(1). Not an error result, which hides stdout.
        err = stderr.finish()
        if err and not err.endswith("\n"):
            err += "\n"
        return CommandResult(
            exit_code=124,
            stdout=stdout.finish(),
            stderr=f"{err}Command timed out after {timeout} seconds",
        )


class RuntimeSessionAgent(SessionAgent):
    """Retain accepted history when a model or summarizer hits its output limit."""

    async def step(self, messages, *args, **kwargs):
        self.accepted_history = list(messages[1:])
        result = await super().step(messages, *args, **kwargs)
        assistant, tools, _ = result
        self.accepted_history.extend([assistant, *tools])
        return result

    async def summarize_messages(self, messages, *args, **kwargs):
        # Tools may already have changed the workspace or an MCP service. Save
        # their results before the summarizer makes another fallible model call.
        self.accepted_history = list(messages[1:])
        removed, active = await super().summarize_messages(messages, *args, **kwargs)
        self.accepted_history = list(active[1:])
        return removed, active


class Session:
    """One worker owns the tool context for its entire lifetime.

    MCP uses task-scoped context managers. Entering in session/new and exiting in
    session/close's separate RPC task would violate their lifetime rules.
    """

    def __init__(self, args, cwd, servers):
        self.args = args
        self.cwd = cwd
        self.servers = servers
        self.queue = asyncio.Queue()
        self.ready = asyncio.get_running_loop().create_future()
        self.worker = asyncio.create_task(self.run())

    async def run(self):
        current = None
        client = None
        try:
            args = self.args
            client = ChatCompletionsClient(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                context_window_tokens=args.context_window_tokens,
                max_retries=0,  # Verifiers owns request recovery.
            )
            tools = [RuntimeCodeExecToolProvider(self.cwd), WebToolProvider()]
            if self.servers:
                tools.append(
                    MCPToolProvider(
                        MCPConfig.model_validate(
                            {
                                "mcpServers": self.servers,
                            }
                        )
                    )
                )
            agent = Agent(
                name="stirrup",
                client=client,
                tools=tools,
                system_prompt=args.system_prompt,
                max_turns=sys.maxsize,  # Verifiers owns per-run limits.
            )
            async with agent.session(
                skills_dir=args.skills_dir if Path(args.skills_dir).is_dir() else None,
                cache_on_interrupt=False,
            ) as active_session:
                # The pinned API requires the parent and its live session state.
                session = RuntimeSessionAgent.from_agent(
                    active_session, active_session._output_state
                )
                for name in json.loads(args.disabled_tools):
                    session.tools.pop(name, None)
                self.ready.set_result(None)
                history = []
                while item := await self.queue.get():
                    prompt, current = item
                    session.accepted_history = [*history, UserMessage(content=prompt)]
                    try:
                        finish, groups, _ = await session.run(session.accepted_history)
                    except OutputTokenLimitError as error:
                        history = session.accepted_history
                        current.set_exception(error)
                        current = None
                        continue
                    except Exception as error:  # noqa: BLE001 - preserve the original error before MCP teardown can group it
                        current.set_exception(error)
                        current = None
                        return
                    # run() prepends its system prompt each time. Retain only
                    # the active tail (including any summary), not compacted groups.
                    history = groups[-1][1:]
                    if finish is None:
                        raise RuntimeError("Stirrup ended without a finish result")
                    if not current.done():
                        current.set_result(finish.reason)
                    current = None
        except asyncio.CancelledError:
            if current is not None and not current.done():
                current.set_result(None)
        except Exception as error:  # noqa: BLE001 - propagate worker failures to RPC callers
            target = current if self.ready.done() else self.ready
            if target is not None and not target.done():
                target.set_exception(error)
        finally:
            if not self.ready.done():
                self.ready.set_exception(
                    RuntimeError("Stirrup session closed during setup")
                )
            while not self.queue.empty():
                item = self.queue.get_nowait()
                if item is not None and not item[1].done():
                    item[1].set_exception(RuntimeError("Stirrup session is closed"))
            if client is not None:
                await client._client.close()

    async def prompt(self, text):
        if self.worker.done():
            raise RuntimeError("Stirrup session is closed")
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((text, future))
        return await asyncio.shield(future)

    async def close(self, *, cancel=False):
        if cancel:
            self.worker.cancel()
        else:
            await self.queue.put(None)
        await self.worker


class StirrupACPAgent:
    def __init__(self, args):
        self.args = args
        self.sessions = {}

    def on_connect(self, conn):
        self.connection = conn

    async def initialize(self, protocol_version, **kwargs):
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                mcp_capabilities=McpCapabilities(http=True, sse=False),
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities()
                ),
            ),
        )

    async def new_session(self, cwd, mcp_servers=None, **kwargs):
        if not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise RequestError.invalid_params(
                {"details": "cwd must be an existing absolute directory"}
            )
        servers = {}
        for server in mcp_servers or []:
            if getattr(server, "type", None) != "http":
                raise RequestError.invalid_params(
                    {"details": "Stirrup accepts HTTP MCP servers only"}
                )
            servers[server.name] = {
                "url": server.url,
                "headers": {header.name: header.value for header in server.headers},
                "timeout": self.args.tool_timeout,
                "sse_read_timeout": self.args.tool_timeout,
            }
        session_id = uuid4().hex
        session = Session(self.args, cwd, servers)
        self.sessions[session_id] = session
        try:
            await session.ready
        except BaseException:
            self.sessions.pop(session_id, None)
            await session.close(cancel=True)
            raise
        return NewSessionResponse(session_id=session_id)

    def session(self, session_id):
        if session_id not in self.sessions:
            raise RequestError.resource_not_found(session_id)
        return self.sessions[session_id]

    async def prompt(self, session_id, prompt, **kwargs):
        if not prompt or any(block.type != "text" for block in prompt):
            raise RequestError.invalid_params(
                {"details": "Stirrup accepts text prompts only"}
            )
        try:
            reply = await self.session(session_id).prompt(
                "".join(block.text for block in prompt)
            )
        except OutputTokenLimitError:
            return PromptResponse(stop_reason="max_tokens")
        except APIError as error:
            raise RequestError.internal_error(
                {
                    "kind": "model_transport",
                    "details": str(error),
                }
            ) from error
        if reply is None:
            return PromptResponse(stop_reason="cancelled")
        await self.connection.session_update(
            session_id=session_id,
            update=AgentMessageChunk(content=TextContentBlock(text=reply)),
        )
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id, **kwargs):
        await self.session(session_id).close(cancel=True)

    async def close_session(self, session_id, **kwargs):
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await session.close()
        return CloseSessionResponse()

    async def close(self):
        for session in self.sessions.values():
            await session.close(cancel=True)
        self.sessions.clear()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--system-prompt-file")
    parser.add_argument("--context-window-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--tool-timeout", type=float, required=True)
    parser.add_argument("--disabled-tools", default="[]")
    args = parser.parse_args()
    args.system_prompt = (
        Path(args.system_prompt_file).read_text(encoding="utf-8")
        if args.system_prompt_file
        else None
    )
    agent = StirrupACPAgent(args)
    reader, writer = await stdio_streams()
    # Reserve stdout for ACP packets; Stirrup's logger prints rich console output.
    with redirect_stdout(sys.stderr):
        try:
            await run_agent(
                agent,
                input_stream=writer,
                output_stream=reader,
                use_unstable_protocol=True,  # session/close in the pinned SDK.
            )
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
