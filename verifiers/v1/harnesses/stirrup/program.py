# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "stirrup[mcp] @ https://github.com/ArtificialAnalysis/Stirrup/archive/{version}.tar.gz",
#     "mcp==1.28.1",
#     "agent-client-protocol==0.12.1",
# ]
# ///
"""Expose Stirrup's agent loop as a rollout-scoped ACP agent over stdio."""

import argparse
import asyncio
import json
import os
import signal
import sys
from contextlib import redirect_stdout, suppress
from pathlib import Path
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    RequestError,
    run_agent,
    stdio_streams,
    update_agent_message_text,
)
from acp.schema import (
    AgentCapabilities,
    CloseSessionResponse,
    InitializeResponse,
    McpCapabilities,
    NewSessionResponse,
    PromptResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
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
        process = None
        completed = False
        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                cmd,
                cwd=self.workspace,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                self._shell_timeout if timeout is None else timeout,
            )
            completed = True
            return CommandResult(
                exit_code=process.returncode,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
            )
        except TimeoutError:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr="Command timed out",
                error_kind="timeout",
            )
        except Exception as error:  # noqa: BLE001 - shell execution failures are recoverable tool results
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(error),
                error_kind="execution_error",
            )
        finally:
            if process is not None and not completed:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                try:
                    await asyncio.wait_for(process.wait(), COMMAND_CLEANUP_TIMEOUT)
                except TimeoutError:
                    # A detached child can retain the pipes after Bash exits.
                    # asyncio has no public API to close subprocess read pipes.
                    for stream in (process.stdout, process.stderr):
                        stream._transport.close()
                    with suppress(TimeoutError):
                        await asyncio.wait_for(process.wait(), COMMAND_CLEANUP_TIMEOUT)


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
            update=update_agent_message_text(reply),
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
    parser.add_argument("--system-prompt")
    parser.add_argument("--context-window-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--skills-dir", required=True)
    parser.add_argument("--tool-timeout", type=float, required=True)
    parser.add_argument("--disabled-tools", default="[]")
    agent = StirrupACPAgent(parser.parse_args())
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
