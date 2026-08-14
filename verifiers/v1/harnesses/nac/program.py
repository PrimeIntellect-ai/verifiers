# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["agent-client-protocol==0.11.0", "httpx==0.28.1"]
# ///
"""Expose NAC's HTTP session API as an ACP agent."""

import asyncio
import os
import re
import signal
import sys
from collections import deque
from contextlib import suppress
from typing import Any

import httpx
from acp import (
    PROTOCOL_VERSION,
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.interfaces import Client
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    HttpMcpServer,
    Implementation,
    McpServerStdio,
    SseMcpServer,
    TextContentBlock,
)

LISTENING = re.compile(r"nac-web listening on (http://\S+)")


class NacAgent(Agent):
    _conn: Client

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr = deque(maxlen=20)
        self._http: httpx.AsyncClient | None = None
        self._sessions: set[str] = set()
        self._sequence_ids: dict[str, int] = {}

    def on_connect(self, conn: Client) -> None:
        self._conn = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(
                name="nac", title="NAC", version=os.environ["VF_NAC_VERSION"]
            ),
        )

    async def _start_server(self, cwd: str) -> None:
        self._process = await asyncio.create_subprocess_exec(
            os.environ["VF_NAC_BIN"],
            "--bind",
            "127.0.0.1:0",
            "--no-open",
            "-C",
            cwd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert self._process.stderr is not None
        try:
            while True:
                line = await asyncio.wait_for(
                    self._process.stderr.readline(), timeout=30
                )
                if not line:
                    raise RuntimeError(
                        "NAC server exited during startup: " + "".join(self._stderr)
                    )
                text = line.decode(errors="replace")
                self._stderr.append(text)
                if match := LISTENING.search(text):
                    self._http = httpx.AsyncClient(base_url=match.group(1), timeout=30)
                    self._stderr_task = asyncio.create_task(
                        self._drain_stderr(self._process.stderr)
                    )
                    return
        except BaseException:
            await self.close()
            raise

    async def _drain_stderr(self, stream: asyncio.StreamReader) -> None:
        while line := await stream.readline():
            self._stderr.append(line.decode(errors="replace"))

    async def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError("NAC server is not running")
        response = await self._http.request(method, path, **kwargs)
        if response.is_error:
            try:
                detail = response.json().get("error", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"NAC HTTP {response.status_code}: {detail}")
        return response.json() if response.content else {}

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        if self._process is None:
            await self._start_server(cwd)
        snapshot = await self._json(
            "POST",
            "/sessions",
            json={
                "cwd": cwd,
                "model": os.environ["VF_NAC_MODEL"],
                "base_url": os.environ["VF_NAC_BASE_URL"],
                "backend": os.environ["VF_NAC_BACKEND"],
                "api_key_env": os.environ["VF_NAC_API_KEY_ENV"],
            },
        )
        session_id = snapshot["metadata"]["session_id"]
        self._sessions.add(session_id)
        self._sequence_ids[session_id] = 0
        return NewSessionResponse(session_id=session_id, modes=None)

    async def prompt(
        self,
        session_id: str,
        prompt: list[TextContentBlock],
        **kwargs: Any,
    ) -> PromptResponse:
        if session_id not in self._sessions:
            raise RuntimeError(f"unknown NAC session {session_id!r}")
        if any(not isinstance(block, TextContentBlock) for block in prompt):
            raise ValueError("NAC ACP supports text prompts only")
        text = "".join(block.text for block in prompt)
        if not text.strip():
            raise ValueError("NAC ACP prompt must not be empty")
        submitted = await self._json(
            "POST", f"/sessions/{session_id}/runs", json={"prompt": text}
        )
        run_id = submitted["run_id"]
        failure_detail = ""

        while True:
            page = await self._json(
                "GET",
                f"/sessions/{session_id}/events",
                params={
                    "after_sequence_id": self._sequence_ids[session_id],
                    "limit": 1000,
                },
            )
            for envelope in page["events"]:
                self._sequence_ids[session_id] = max(
                    self._sequence_ids[session_id], envelope["sequence_id"]
                )
                if envelope.get("run_id") != run_id:
                    continue
                event = envelope["event"]
                if event["type"] == "agent" and event["event"]["type"] in {
                    "error",
                    "model_error",
                }:
                    failure_detail = event["event"]["message"]
                match event["type"]:
                    case "run_completed":
                        await self._conn.session_update(
                            session_id,
                            update_agent_message(text_block(event["response"])),
                        )
                        return PromptResponse(stop_reason="end_turn")
                    case "run_failed":
                        detail = failure_detail or event["message"]
                        print(f"NAC run failed: {detail}", file=sys.stderr)
                        raise RuntimeError(detail)
                    case "run_cancelled":
                        return PromptResponse(stop_reason="cancelled")
            await asyncio.sleep(0.1)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        if session_id in self._sessions:
            await self._json("POST", f"/sessions/{session_id}/cancel-active-run")

    async def close(self) -> None:
        if self._http is not None:
            for session_id in self._sessions:
                with suppress(Exception):
                    await self._json("DELETE", f"/sessions/{session_id}")
            await self._http.aclose()
            self._http = None
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        if self._stderr_task is not None:
            if not self._stderr_task.done():
                self._stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None


async def main() -> None:
    agent = NacAgent()
    task = asyncio.current_task()
    assert task is not None
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await run_agent(agent)
    except asyncio.CancelledError:
        pass
    finally:
        await asyncio.shield(agent.close())


if __name__ == "__main__":
    asyncio.run(main())
