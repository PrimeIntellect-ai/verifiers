# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["websockets==15.0.1"]
# ///
"""Mediate Codex's Code Mode host through LiveACPClient tool policy."""

import asyncio
import json
import signal
import sys
from contextlib import suppress
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

MAX_FRAME_BYTES = 64 * 1024 * 1024
DUAL_WEBSOCKET = "dual-websocket-v1"


def unpack_frame(frame: bytes) -> dict:
    if len(frame) < 4:
        raise ValueError("Code Mode frame is missing its length prefix")
    size = int.from_bytes(frame[:4], "little")
    if size > MAX_FRAME_BYTES or len(frame) != size + 4:
        raise ValueError("Code Mode frame has an invalid payload length")
    return json.loads(frame[4:])


def pack_frame(message: dict) -> bytes:
    payload = json.dumps(message, separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Code Mode frame exceeds the protocol limit")
    return len(payload).to_bytes(4, "little") + payload


def find_host() -> str:
    launcher = Path(sys.argv[1]).resolve()
    node_modules = next(
        (parent for parent in launcher.parents if parent.name == "node_modules"), None
    )
    if node_modules is None:
        raise RuntimeError(f"Cannot locate Codex package from {launcher}")
    matches = list(
        node_modules.glob("@openai/codex-*/vendor/*/bin/codex-code-mode-host")
    )
    if len(matches) != 1:
        raise RuntimeError(f"Found {len(matches)} Codex Code Mode hosts")
    return str(matches[0])


class PolicyClient:
    def __init__(self) -> None:
        self.request_id = 0
        self.lock = asyncio.Lock()

    async def request(self, body: dict) -> dict:
        async with self.lock:
            self.request_id += 1
            request_id = self.request_id
            print(json.dumps({"id": request_id, "body": body}), flush=True)
            line = await asyncio.to_thread(sys.stdin.buffer.readline)
            if not line:
                raise EOFError("LiveACPClient closed the interception channel")
            response = json.loads(line)
            if response.get("id") != request_id:
                raise ValueError("LiveACPClient returned an invalid response ID")
            if error := response.get("error"):
                raise RuntimeError(error)
            decision = response.get("decision")
            if not isinstance(decision, dict) or decision.get("action") not in {
                "allow",
                "rewrite",
                "stop",
            }:
                raise ValueError("LiveACPClient returned an invalid tool decision")
            return decision


async def intercept(
    policy: PolicyClient,
    phase: str,
    call_id: str,
    name: str,
    content,
) -> dict:
    decision = await policy.request(
        {
            "phase": phase,
            "content": "any",
            "detachedParent": "exec",
            "message": {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
                "name": name,
            },
        }
    )
    if decision["action"] == "rewrite":
        message = decision.get("message")
        if not isinstance(message, dict):
            raise ValueError("Tool interception omitted its replacement")
        if message.get("tool_call_id") != call_id or message.get("name") != name:
            raise ValueError("Tool interception changed the Code Mode tool identity")
    return decision


def response_content(response: dict):
    if len(response) != 1:
        raise ValueError("Code Mode returned an invalid runtime response")
    variant, value = next(iter(response.items()))
    if variant not in {"Yielded", "Terminated", "Result"}:
        raise ValueError(f"Code Mode returned an unknown response: {variant}")
    parts = []
    for item in value.get("content_items", []):
        if item.get("type") == "input_text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "input_image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item.get("image_url", "")},
                }
            )
        else:
            raise ValueError("Code Mode interception does not support audio results")
    if variant == "Result" and value.get("error_text") is not None:
        parts.append({"type": "text", "text": f"Script error:\n{value['error_text']}"})
    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def wire_content(content) -> list[dict]:
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    result = []
    for part in parts:
        if part.get("type") == "text":
            result.append({"type": "input_text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            result.append(
                {
                    "type": "input_image",
                    "image_url": (part.get("image_url") or {}).get("url", ""),
                }
            )
        else:
            raise ValueError("Tool interception returned unsupported content")
    return result


def replacement(decision: dict):
    if decision["action"] == "rewrite":
        return decision["message"].get("content", "")
    return decision.get("reason") or "Rollout terminated by interception."


async def apply_result_policy(
    policy: PolicyClient, response: dict, call_id: str, name: str
) -> None:
    decision = await intercept(
        policy, "after", call_id, name, response_content(response)
    )
    if decision["action"] == "allow":
        return
    content = replacement(decision)
    variant, value = next(iter(response.items()))
    value["content_items"] = wire_content(content)
    if variant == "Result":
        value["error_text"] = None


async def run_connection(client, host_url: str, policy: PolicyClient) -> None:
    executions: dict[int, tuple[str, str]] = {}
    cells: dict[str, tuple[str, str]] = {}
    continuations: dict[int, str] = {}
    send_lock = asyncio.Lock()
    async with connect(
        host_url,
        compression=None,
        max_size=MAX_FRAME_BYTES + 4,
        proxy=None,
    ) as host:

        async def send_client(message: dict) -> None:
            async with send_lock:
                await client.send(pack_frame(message))

        async def forward_client() -> None:
            async for frame in client:
                if not isinstance(frame, bytes):
                    raise TypeError("Code Mode websocket messages must be binary")
                message = unpack_frame(frame)
                if message.get("type") == "connection/hello":
                    if DUAL_WEBSOCKET in message.get("requiredCapabilities", []):
                        raise ValueError(
                            "Code Mode proxy cannot satisfy required dual websockets"
                        )
                    message["optionalCapabilities"] = [
                        item
                        for item in message.get("optionalCapabilities", [])
                        if item != DUAL_WEBSOCKET
                    ]
                operation = message.get("request", {})
                method = operation.get("method")
                if (
                    message.get("type") == "operation/request"
                    and method == "session/execute"
                ):
                    request_id = message["id"]
                    call_id = operation["request"]["tool_call_id"]
                    decision = await intercept(policy, "before", call_id, "exec", "")
                    if decision["action"] == "allow":
                        executions[request_id] = (call_id, "exec")
                    else:
                        cell_id = f"vf-blocked-{request_id}"
                        runtime_response = {
                            "Result": {
                                "cell_id": cell_id,
                                "content_items": wire_content(replacement(decision)),
                                "error_text": None,
                            }
                        }
                        await send_client(
                            {
                                "type": "operation/response",
                                "id": request_id,
                                "result": {
                                    "status": "ok",
                                    "value": {
                                        "type": "execution/started",
                                        "cellId": cell_id,
                                    },
                                },
                            }
                        )
                        await send_client(
                            {
                                "type": "execute/initialResponse",
                                "id": request_id,
                                "result": {
                                    "status": "ok",
                                    "value": runtime_response,
                                },
                            }
                        )
                        await send_client(
                            {
                                "type": "cell/closed",
                                "sessionId": operation["sessionId"],
                                "cellId": cell_id,
                            }
                        )
                        continue
                elif message.get("type") == "operation/request" and method in {
                    "session/wait",
                    "session/terminate",
                }:
                    continuations[message["id"]] = (
                        operation["request"]["cell_id"]
                        if method == "session/wait"
                        else operation["cellId"]
                    )
                await host.send(pack_frame(message))

        async def forward_host() -> None:
            async for frame in host:
                if not isinstance(frame, bytes):
                    raise TypeError("Code Mode websocket messages must be binary")
                message = unpack_frame(frame)
                request_id = message.get("id")
                result = message.get("result", {})
                response = None
                call = None
                if (
                    request_id in executions
                    and message.get("type") == "execute/initialResponse"
                    and result.get("status") == "ok"
                ):
                    call = executions.pop(request_id)
                    response = result["value"]
                elif (
                    request_id in continuations
                    and message.get("type") == "operation/response"
                    and result.get("status") == "ok"
                ):
                    cell_id = continuations.pop(request_id)
                    outcome = result.get("value", {}).get("outcome")
                    if isinstance(outcome, dict) and len(outcome) == 1:
                        response = next(iter(outcome.values()))
                        call = cells.get(cell_id)
                if response is not None and call is not None:
                    variant, value = next(iter(response.items()))
                    cell_id = value.get("cell_id")
                    if variant == "Yielded":
                        if cell_id:
                            cells[cell_id] = call
                    else:
                        if cell_id:
                            cells.pop(cell_id, None)
                        await apply_result_policy(policy, response, *call)
                await send_client(message)

        tasks = [
            asyncio.create_task(forward_client()),
            asyncio.create_task(forward_host()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            with suppress(ConnectionClosed):
                task.result()


async def run() -> None:
    host = await asyncio.create_subprocess_exec(
        find_host(),
        "--listen",
        "ws://127.0.0.1:0",
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
    )
    assert host.stdout is not None
    try:
        host_url = (
            (await asyncio.wait_for(host.stdout.readline(), timeout=15))
            .decode()
            .strip()
        )
        if not host_url.startswith("ws://127.0.0.1:"):
            raise RuntimeError(
                f"Codex Code Mode host returned an invalid endpoint: {host_url!r}"
            )
        claimed = False
        policy = PolicyClient()

        async def accept(client) -> None:
            nonlocal claimed
            if claimed:
                await client.close(1008, "Code Mode host is already connected")
                return
            claimed = True
            await run_connection(client, host_url, policy)

        async with serve(
            accept,
            "127.0.0.1",
            0,
            compression=None,
            max_size=MAX_FRAME_BYTES + 4,
        ) as server:
            port = server.sockets[0].getsockname()[1]
            print(f"ws://127.0.0.1:{port}", flush=True)
            await host.wait()
            raise RuntimeError(
                f"Codex Code Mode host exited with status {host.returncode}"
            )
    finally:
        if host.returncode is None:
            host.terminate()
            try:
                await asyncio.wait_for(host.wait(), timeout=5)
            except TimeoutError:
                host.kill()
                await host.wait()


async def main() -> None:
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    if task is not None:
        for event in (signal.SIGTERM, signal.SIGINT):
            with suppress(NotImplementedError):
                loop.add_signal_handler(event, task.cancel)
    with suppress(asyncio.CancelledError):
        await run()


if __name__ == "__main__":
    asyncio.run(main())
