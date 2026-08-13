# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Add exact VF session import to Kimi Code's native ACP server."""

import asyncio
import json
import os
import shutil
import sys
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

SESSION_IMPORT = "verifiers.dev/sessionImport"


class SessionImportError(ValueError):
    pass


def content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [
        {"type": "text", "text": part["text"]}
        if part["type"] == "text"
        else {
            "type": "image_url",
            "imageUrl": {"url": part["image_url"]["url"]},
        }
        for part in content
    ]


def native_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise SessionImportError("unsupported verifiers.dev/sessionImport version")
    messages = value.get("messages")
    if not isinstance(messages, list):
        raise SessionImportError("verifiers.dev/sessionImport.messages must be a list")

    native = []
    for message in messages:
        role = message["role"]
        if message.get("provider_state") not in (None, []):
            raise SessionImportError(
                "Kimi cannot exactly import opaque provider state with its native context"
            )

        content = []
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            content.append({"type": "think", "think": reasoning})
        if message.get("content") is not None:
            content.extend(content_parts(message["content"]))
        imported = {"role": role, "content": content}
        if role == "assistant":
            calls = message.get("tool_calls") or []
            imported["toolCalls"] = [
                {
                    "type": "function",
                    "id": call["id"],
                    "name": call["name"],
                    "arguments": call["arguments"],
                }
                for call in calls
            ]
        elif role == "user":
            imported["origin"] = {"kind": "user"}
        elif role == "tool":
            imported["toolCallId"] = message["tool_call_id"]
            if name := message.get("name"):
                imported["name"] = name
        native.append(imported)
    return native


class KimiACPProxy:
    def __init__(self) -> None:
        self.binary = os.environ["KIMI_ACP_BINARY"]
        self.child: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task[None] | None = None
        self.waiting_id: Any = None
        self.response: asyncio.Future[dict[str, Any]] | None = None
        self.initialize_params: dict[str, Any] | None = None

    def output(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode()
        sys.stdout.buffer.write(line + b"\n")
        sys.stdout.buffer.flush()

    async def start(self) -> None:
        env = os.environ.copy()
        env.pop("KIMI_ACP_BINARY", None)
        self.child = await asyncio.create_subprocess_exec(
            self.binary,
            "acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=env,
        )
        self.reader = asyncio.create_task(self.relay(self.child))

    async def relay(self, child: asyncio.subprocess.Process) -> None:
        assert child.stdout is not None
        while line := await child.stdout.readline():
            message = json.loads(line)
            if (
                "method" not in message
                and "id" in message
                and message["id"] == self.waiting_id
                and self.response is not None
            ):
                self.response.set_result(message)
            else:
                self.output(message)
        if self.response is not None and not self.response.done():
            self.response.set_exception(
                RuntimeError("Kimi ACP server exited before replying")
            )

    async def send(self, message: dict[str, Any]) -> None:
        if self.child is None or self.child.stdin is None:
            raise RuntimeError("Kimi ACP server is not running")
        self.child.stdin.write(
            json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
        )
        await self.child.stdin.drain()

    async def request(self, message: dict[str, Any]) -> dict[str, Any]:
        self.waiting_id = message["id"]
        self.response = asyncio.get_running_loop().create_future()
        try:
            await self.send(message)
            return await self.response
        finally:
            self.waiting_id = None
            self.response = None

    async def internal_request(self, method: str, params: dict[str, Any]) -> None:
        response = await self.request(
            {
                "jsonrpc": "2.0",
                "id": f"vf-session-import-{method}",
                "method": method,
                "params": params,
            }
        )
        if error := response.get("error"):
            raise RuntimeError(f"Kimi {method} failed after session import: {error}")

    async def stop(self) -> None:
        child, self.child = self.child, None
        reader, self.reader = self.reader, None
        if child is None:
            return
        with suppress(ProcessLookupError):
            child.terminate()
        try:
            await asyncio.wait_for(child.wait(), 10)
        except TimeoutError:
            child.kill()
            await child.wait()
        if reader is not None:
            await reader

    @staticmethod
    def import_wire(session: Path, messages: list[dict[str, Any]]) -> None:
        wire = session / "agents/main/wire.jsonl"
        records = b"".join(
            json.dumps(
                {"type": "context.append_message", "message": message},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            + b"\n"
            for message in messages
        )
        target = wire.with_name("wire.jsonl.vf-import")
        target.write_bytes(wire.read_bytes() + records)
        target.replace(wire)

    async def initialize(self, message: dict[str, Any]) -> None:
        response = await self.request(message)
        if result := response.get("result"):
            capabilities = result.setdefault("agentCapabilities", {})
            capabilities.setdefault("_meta", {})[SESSION_IMPORT] = {"version": 1}
            self.initialize_params = deepcopy(message.get("params") or {})
        self.output(response)

    async def new_session(self, message: dict[str, Any], value: Any) -> None:
        imported = native_messages(value)
        request = deepcopy(message)
        params = request["params"]
        metadata = params.get("_meta") or {}
        metadata.pop(SESSION_IMPORT)
        if not metadata:
            params.pop("_meta", None)
        response = await self.request(request)
        result = response.get("result")
        if not isinstance(result, dict):
            self.output(response)
            return
        if self.initialize_params is None:
            raise RuntimeError("Kimi session/new arrived before initialize")
        session_id = result["sessionId"]
        sessions = Path(os.environ["KIMI_CODE_HOME"]).resolve() / "sessions"
        matches = list(sessions.glob(f"*/{session_id}"))
        session = matches[0] if len(matches) == 1 else None
        try:
            if session is None:
                raise RuntimeError(
                    f"expected one Kimi session for {session_id}, found {len(matches)}"
                )
            await self.stop()
            self.import_wire(session, imported)
            await self.start()
            await self.internal_request("initialize", self.initialize_params)
            await self.internal_request(
                "session/resume",
                {
                    "cwd": params["cwd"],
                    "sessionId": session_id,
                    "mcpServers": params.get("mcpServers") or [],
                },
            )
        except Exception as error:
            if session is not None:
                shutil.rmtree(session, ignore_errors=True)
            if self.child is None:
                try:
                    await self.start()
                except (OSError, RuntimeError) as restart_error:
                    raise RuntimeError(
                        f"{error}; Kimi restart failed: {restart_error}"
                    ) from error
            raise RuntimeError(str(error)) from error
        self.output(response)

    async def run(self) -> None:
        await self.start()
        while line := await asyncio.to_thread(sys.stdin.buffer.readline):
            message = json.loads(line)
            try:
                if message.get("method") == "initialize":
                    await self.initialize(message)
                elif (
                    message.get("method") == "session/new"
                    and (
                        metadata := (message.get("params") or {}).get("_meta") or {}
                    ).get(SESSION_IMPORT)
                    is not None
                ):
                    await self.new_session(message, metadata[SESSION_IMPORT])
                else:
                    await self.send(message)
            except Exception as error:  # noqa: BLE001 - return child failures over JSON-RPC
                self.output(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {
                            "code": -32602
                            if isinstance(error, SessionImportError)
                            else -32603,
                            "message": str(error),
                        },
                    }
                )


async def main() -> None:
    proxy = KimiACPProxy()
    try:
        await proxy.run()
    finally:
        await proxy.stop()


if __name__ == "__main__":
    asyncio.run(main())
