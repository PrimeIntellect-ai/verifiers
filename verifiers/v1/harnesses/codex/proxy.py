# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = ["websockets==15.0.1"]
# ///
"""Mediate Codex's versioned Code Mode host protocol through tool policy."""

import asyncio
import json
import signal
import sys
from contextlib import suppress
from pathlib import Path
from urllib.request import Request as UrlRequest
from urllib.request import urlopen as openUrl

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

maxFrameBytes = 64 * 1024 * 1024
dualWebsocket = "dual-websocket-v1"


def unpackFrame(frame: bytes) -> dict:
    if len(frame) < 4:
        raise ValueError("Code Mode frame is missing its length prefix")
    size = int.from_bytes(frame[:4], "little")
    if size > maxFrameBytes or len(frame) != size + 4:
        raise ValueError("Code Mode frame has an invalid payload length")
    return json.loads(frame[4:])


def packFrame(message: dict) -> bytes:
    payload = json.dumps(message, separators=(",", ":")).encode()
    if len(payload) > maxFrameBytes:
        raise ValueError("Code Mode frame exceeds the protocol limit")
    return len(payload).to_bytes(4, "little") + payload


def findHost() -> str:
    launcher = Path(sys.argv[1]).resolve()
    nodeModules = next(
        (parent for parent in launcher.parents if parent.name == "node_modules"), None
    )
    if nodeModules is None:
        raise RuntimeError(f"Cannot locate Codex package from {launcher}")
    matches = list(
        nodeModules.glob("@openai/codex-*/vendor/*/bin/codex-code-mode-host")
    )
    if len(matches) != 1:
        raise RuntimeError(f"Found {len(matches)} Codex Code Mode hosts")
    return str(matches[0])


def callPolicy(configuration: dict[str, str], body: dict) -> dict:
    request = UrlRequest(
        configuration["url"],
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {configuration['secret']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with openUrl(request, timeout=35) as response:
        decision = json.loads(response.read())
    if not isinstance(decision, dict) or decision.get("action") not in {
        "allow",
        "rewrite",
        "stop",
    }:
        raise ValueError("Tool interception returned an invalid decision")
    return decision


async def intercept(
    configuration: dict[str, str],
    phase: str,
    callId: str,
    name: str,
    content,
    toolArguments: dict | None = None,
) -> dict:
    body = {
        "phase": phase,
        "content": "any",
        "resultFraming": "codex_code_mode",
        "message": {
            "role": "tool",
            "tool_call_id": callId,
            "content": content,
            "name": name,
        },
    }
    if toolArguments is not None:
        body["toolArguments"] = toolArguments
    decision = await asyncio.to_thread(
        callPolicy,
        configuration,
        body,
    )
    resolvedId = decision.get("toolCallId", callId)
    if not isinstance(resolvedId, str) or not resolvedId:
        raise ValueError("Tool interception did not resolve the tool call ID")
    decision["toolCallId"] = resolvedId
    return decision


def responseContent(response: dict):
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


def wireContent(content) -> list[dict]:
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


def replacement(decision: dict, callId: str, name: str):
    if decision["action"] == "rewrite":
        message = decision.get("message")
        if not isinstance(message, dict):
            raise ValueError("Tool interception omitted its replacement")
        if message.get("tool_call_id") != callId or message.get("name") != name:
            raise ValueError("Tool interception changed the Code Mode tool identity")
        return message.get("content", "")
    return decision.get("reason") or "Rollout terminated by interception."


async def runConnection(client, hostUrl: str, configuration: dict[str, str]) -> None:
    requests: dict[int, tuple[str, str]] = {}
    sendLock = asyncio.Lock()
    async with connect(
        hostUrl,
        compression=None,
        max_size=maxFrameBytes + 4,
        proxy=None,
    ) as host:

        async def sendClient(message: dict) -> None:
            async with sendLock:
                await client.send(packFrame(message))

        async def forwardClient() -> None:
            async for frame in client:
                if not isinstance(frame, bytes):
                    raise TypeError("Code Mode websocket messages must be binary")
                message = unpackFrame(frame)
                if message.get("type") == "connection/hello":
                    if dualWebsocket in message.get("requiredCapabilities", []):
                        raise ValueError(
                            "Code Mode proxy cannot satisfy required dual websockets"
                        )
                    message["optionalCapabilities"] = [
                        item
                        for item in message.get("optionalCapabilities", [])
                        if item != dualWebsocket
                    ]
                operation = message.get("request", {})
                method = operation.get("method")
                if message.get("type") == "operation/request" and method in {
                    "session/execute",
                    "session/wait",
                    "session/terminate",
                }:
                    requestId = message["id"]
                    if method == "session/execute":
                        name = "exec"
                        callId = operation["request"]["tool_call_id"]
                        toolArguments = None
                    else:
                        name = "wait"
                        callId = ""
                        cellId = (
                            operation["request"]["cell_id"]
                            if method == "session/wait"
                            else operation["cellId"]
                        )
                        toolArguments = {"cell_id": cellId}
                    decision = await intercept(
                        configuration,
                        "before",
                        callId,
                        name,
                        "",
                        toolArguments,
                    )
                    callId = decision["toolCallId"]
                    if decision["action"] == "allow":
                        requests[requestId] = (callId, name)
                    else:
                        content = replacement(decision, callId, name)
                        cellId = (
                            f"vf-blocked-{requestId}"
                            if method == "session/execute"
                            else cellId
                        )
                        runtimeResponse = {
                            "Result": {
                                "cell_id": cellId,
                                "content_items": wireContent(content),
                                "error_text": None,
                            }
                        }
                        if method != "session/execute":
                            await sendClient(
                                {
                                    "type": "operation/response",
                                    "id": requestId,
                                    "result": {
                                        "status": "ok",
                                        "value": {
                                            "type": "wait/completed",
                                            "outcome": {"LiveCell": runtimeResponse},
                                        },
                                    },
                                }
                            )
                            continue
                        await sendClient(
                            {
                                "type": "operation/response",
                                "id": requestId,
                                "result": {
                                    "status": "ok",
                                    "value": {
                                        "type": "execution/started",
                                        "cellId": cellId,
                                    },
                                },
                            }
                        )
                        await sendClient(
                            {
                                "type": "execute/initialResponse",
                                "id": requestId,
                                "result": {
                                    "status": "ok",
                                    "value": {**runtimeResponse},
                                },
                            }
                        )
                        await sendClient(
                            {
                                "type": "cell/closed",
                                "sessionId": operation["sessionId"],
                                "cellId": cellId,
                            }
                        )
                        continue
                await host.send(packFrame(message))

        async def forwardHost() -> None:
            async for frame in host:
                if not isinstance(frame, bytes):
                    raise TypeError("Code Mode websocket messages must be binary")
                message = unpackFrame(frame)
                pending = requests.get(message.get("id"))
                result = message.get("result", {})
                response = None
                if (
                    pending is not None
                    and message.get("type") == "execute/initialResponse"
                    and result.get("status") == "ok"
                ):
                    response = result["value"]
                elif (
                    pending is not None
                    and message.get("type") == "operation/response"
                    and result.get("status") == "ok"
                    and result.get("value", {}).get("type") == "wait/completed"
                ):
                    outcome = result["value"]["outcome"]
                    if len(outcome) != 1:
                        raise ValueError("Code Mode returned an invalid wait result")
                    response = next(iter(outcome.values()))
                if response is not None:
                    callId, name = requests.pop(message["id"])
                    decision = await intercept(
                        configuration,
                        "after",
                        callId,
                        name,
                        responseContent(response),
                    )
                    if decision["action"] != "allow":
                        content = replacement(decision, callId, name)
                        variant, value = next(iter(response.items()))
                        value["content_items"] = wireContent(content)
                        if variant == "Result":
                            value["error_text"] = None
                await sendClient(message)

        tasks = [
            asyncio.create_task(forwardClient()),
            asyncio.create_task(forwardHost()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            with suppress(ConnectionClosed):
                task.result()


async def run() -> None:
    configuration = json.loads(await asyncio.to_thread(sys.stdin.buffer.read))
    if not isinstance(configuration, dict) or not all(
        isinstance(configuration.get(field), str) for field in ("url", "secret")
    ):
        raise ValueError("Code Mode interception credentials are invalid")
    host = await asyncio.create_subprocess_exec(
        findHost(),
        "--listen",
        "ws://127.0.0.1:0",
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
    )
    assert host.stdout is not None
    try:
        hostUrl = (
            (await asyncio.wait_for(host.stdout.readline(), timeout=15))
            .decode()
            .strip()
        )
        if not hostUrl.startswith("ws://127.0.0.1:"):
            raise RuntimeError(
                f"Codex Code Mode host returned an invalid endpoint: {hostUrl!r}"
            )
        claimed = False

        async def accept(client) -> None:
            nonlocal claimed
            if claimed:
                await client.close(1008, "Code Mode host is already connected")
                return
            claimed = True
            await runConnection(client, hostUrl, configuration)

        async with serve(
            accept,
            "127.0.0.1",
            0,
            compression=None,
            max_size=maxFrameBytes + 4,
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
